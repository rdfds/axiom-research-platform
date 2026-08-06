#!/usr/bin/env python3
"""Build human-readable scorecard packets from the market-pricing scorecard artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True, help="Scorecard JSONL artifact")
    parser.add_argument("--out", required=True, help="Output JSON packet path")
    parser.add_argument("--companyfacts-root", help="Optional SEC companyfacts directory for issuer names")
    parser.add_argument("--limit", type=int, default=12, help="Rows per packet")
    return parser.parse_args()


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _node_value(features: Dict[str, Any], name: str) -> float | None:
    node = features.get(name) or {}
    value = node.get("value")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _load_company_name(companyfacts_root: Path | None, company_id: str) -> str | None:
    if companyfacts_root is None:
        return None
    path = companyfacts_root / f"CIK{company_id}.json"
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None
    name = obj.get("entityName")
    if name is None:
        return None
    return str(name)


def _thesis(row: Dict[str, Any]) -> str:
    quality = row.get("quality_score")
    balance = row.get("balance_sheet_score")
    risk = row.get("risk_score")
    value = row.get("value_score")
    gap = row.get("valuation_gap_score")
    parts: List[str] = []
    if quality is not None and quality >= 70:
        parts.append("strong quality")
    elif quality is not None and quality <= 35:
        parts.append("weak quality")
    if balance is not None and balance >= 70:
        parts.append("solid balance sheet")
    elif balance is not None and balance <= 35:
        parts.append("strained balance sheet")
    if risk is not None and risk >= 70:
        parts.append("stable tape/credit profile")
    elif risk is not None and risk <= 35:
        parts.append("fragile tape/credit profile")
    if value is not None and value >= 70:
        parts.append("cheap on value metrics")
    elif value is not None and value <= 30:
        parts.append("rich valuation")
    if gap is not None and gap >= 20:
        parts.append("fundamentals outrun price")
    elif gap is not None and gap <= -20:
        parts.append("price already discounts strength")
    return ", ".join(parts[:4])


def _packet_row(row: Dict[str, Any], *, name: str | None) -> Dict[str, Any]:
    features = row.get("features") or {}
    return {
        "company_id": str(row.get("company_id") or ""),
        "company_name": name,
        "overall_score": _node_value(features, "market.comp_overall_score"),
        "valuation_gap_score": _node_value(features, "market.valuation_gap_score"),
        "value_score": _node_value(features, "market.value_score"),
        "quality_score": _node_value(features, "market.quality_score"),
        "balance_sheet_score": _node_value(features, "market.balance_sheet_score"),
        "risk_score": _node_value(features, "market.risk_score"),
        "ev_ebitda": _node_value(features, "market.ev_ebitda"),
        "fcf_yield": _node_value(features, "market.fcf_yield"),
        "revenue_yoy_last_q": _node_value(features, "operating.revenue_yoy_last_q"),
        "ebitda_margin_ttm": _node_value(features, "operating.ebitda_margin_ttm"),
        "net_leverage_normalized": _node_value(features, "capital_structure.net_leverage_normalized"),
        "available_liquidity_normalized": _node_value(features, "liquidity.available_liquidity_normalized"),
        "credit_window_proxy": _node_value(features, "market.credit_window_proxy"),
        "maturity_wall_ratio_24m": _node_value(features, "capital_structure.maturity_wall_ratio_24m"),
    }


def _top_longs(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    candidates = []
    for row in rows:
        overall = row.get("overall_score")
        gap = row.get("valuation_gap_score")
        quality = row.get("quality_score") or 0.0
        if overall is None or gap is None:
            continue
        if gap <= 0:
            continue
        score = 0.45 * gap + 0.35 * overall + 0.20 * quality
        candidates.append((score, row))
    ranked = [row for _, row in sorted(candidates, key=lambda item: item[0], reverse=True)[:limit]]
    for row in ranked:
        row["packet_score"] = round(0.45 * row["valuation_gap_score"] + 0.35 * row["overall_score"] + 0.20 * (row["quality_score"] or 0.0), 4)
        row["packet_label"] = "top_longs"
        row["thesis"] = _thesis(row)
    return ranked


def _fragile_shorts(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    candidates = []
    for row in rows:
        quality = row.get("quality_score")
        balance = row.get("balance_sheet_score")
        risk = row.get("risk_score")
        if quality is None or risk is None:
            continue
        balance_term = balance if balance is not None else 35.0
        fragility = 0.35 * (100.0 - quality) + 0.35 * (100.0 - balance_term) + 0.30 * (100.0 - risk)
        candidates.append((fragility, row))
    ranked = [row for _, row in sorted(candidates, key=lambda item: item[0], reverse=True)[:limit]]
    for row in ranked:
        quality = row["quality_score"] or 0.0
        balance = row["balance_sheet_score"] if row["balance_sheet_score"] is not None else 35.0
        risk = row["risk_score"] or 0.0
        row["packet_score"] = round(0.35 * (100.0 - quality) + 0.35 * (100.0 - balance) + 0.30 * (100.0 - risk), 4)
        row["packet_label"] = "fragile_shorts"
        row["thesis"] = _thesis(row)
    return ranked


def _mispriced_quality(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    candidates = []
    for row in rows:
        quality = row.get("quality_score")
        gap = row.get("valuation_gap_score")
        if quality is None or gap is None:
            continue
        if quality < 55 or gap <= 0:
            continue
        balance = row.get("balance_sheet_score") or 50.0
        risk = row.get("risk_score") or 50.0
        score = 0.45 * quality + 0.30 * gap + 0.15 * balance + 0.10 * risk
        candidates.append((score, row))
    ranked = [row for _, row in sorted(candidates, key=lambda item: item[0], reverse=True)[:limit]]
    for row in ranked:
        row["packet_score"] = round(
            0.45 * row["quality_score"]
            + 0.30 * row["valuation_gap_score"]
            + 0.15 * (row["balance_sheet_score"] or 50.0)
            + 0.10 * (row["risk_score"] or 50.0),
            4,
        )
        row["packet_label"] = "mispriced_quality"
        row["thesis"] = _thesis(row)
    return ranked


def build_packets(rows: List[Dict[str, Any]], *, companyfacts_root: Path | None, limit: int) -> Dict[str, Any]:
    packet_rows = []
    for row in rows:
        company_id = str(row.get("company_id") or "")
        name = _load_company_name(companyfacts_root, company_id)
        packet_rows.append(_packet_row(row, name=name))
    return {
        "top_longs": _top_longs([dict(row) for row in packet_rows], limit),
        "fragile_shorts": _fragile_shorts([dict(row) for row in packet_rows], limit),
        "mispriced_quality": _mispriced_quality([dict(row) for row in packet_rows], limit),
    }


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    companyfacts_root = Path(args.companyfacts_root) if args.companyfacts_root else None
    packets = build_packets(list(iter_rows(artifact_path)), companyfacts_root=companyfacts_root, limit=args.limit)
    payload = {
        "generated_at": Path(args.artifact_path).stat().st_mtime,
        "artifact_path": str(artifact_path),
        "packets": packets,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Built market-pricing packets -> {out_path}")


if __name__ == "__main__":
    main()
