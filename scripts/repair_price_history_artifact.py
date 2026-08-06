#!/usr/bin/env python3
"""Repair price-history-derived market metrics in a materialized artifact."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import pyarrow.parquet as pq


REPAIR_METRICS = [
    "market.volatility_30d",
    "market.volatility_90d",
    "market.drawdown_90d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True, help="Input company-state JSONL artifact")
    parser.add_argument("--market-cache-path", help="Optional CRSP market cache parquet path")
    parser.add_argument("--out", required=True, help="Output repaired JSONL artifact")
    parser.add_argument("--summary-out", help="Optional summary JSON")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _node_support(node: Dict[str, Any] | None) -> str:
    if not node:
        return "unsupported"
    return str(node.get("support_mode") or "unsupported")


def _union_provenance(*nodes: Dict[str, Any] | None) -> list[Dict[str, Any]]:
    merged: list[Dict[str, Any]] = []
    seen = set()
    for node in nodes:
        for prov in (node or {}).get("provenance") or []:
            key = json.dumps(prov, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            merged.append(copy.deepcopy(prov))
    return merged


def _base_repaired_node(node: Dict[str, Any], *, computed_at: str) -> Dict[str, Any]:
    repaired = copy.deepcopy(node)
    repaired["computed_at"] = computed_at
    repaired["missing_reason"] = None
    repaired["quality_flags"] = repaired.get("quality_flags") or None
    return repaired


def _needs_exact_price_history_repair(node: Dict[str, Any] | None) -> bool:
    if not node:
        return False
    if node.get("value") is None:
        return True
    fallback_used = str(node.get("fallback_used") or "")
    quality_flags = {str(flag) for flag in (node.get("quality_flags") or [])}
    support_mode = str(node.get("support_mode") or "unsupported")
    component_breakdown = node.get("component_breakdown") or {}
    formula = str(component_breakdown.get("formula") or "")
    source_kind = str(component_breakdown.get("source_kind") or "")
    selected_series = component_breakdown.get("selected_price_series") or {}
    selected_source_kind = str(selected_series.get("source_kind") or "")

    if support_mode != "exact":
        return True
    if fallback_used == "monthly_price_history_proxy":
        return True
    if "monthly_price_history_proxy" in quality_flags:
        return True
    if "monthly_returns" in formula or "monthly_price_window" in formula:
        return True
    if source_kind and source_kind != "crsp_market_cache":
        return True
    if selected_source_kind and selected_source_kind != "crsp_market_cache":
        return True
    return False


def _infer_market_cache_path(artifact_path: Path) -> Path | None:
    for row in iter_rows(artifact_path):
        features = row.get("features") or {}
        for metric in ("market.total_return_3m_standardized", "market.total_return_12m_standardized"):
            node = features.get(metric) or {}
            for prov in node.get("provenance") or []:
                source = prov.get("source")
                if source and str(source).endswith(".parquet"):
                    return Path(str(source))
    return None


def _load_market_cache(path: Path) -> Dict[str, pd.DataFrame]:
    table = pq.read_table(path, columns=["permno", "trade_date", "price_proxy", "close_price"])
    df = table.to_pandas()
    df["trade_date"] = pd.to_datetime(df["trade_date"], utc=True, errors="coerce")
    df["price"] = pd.to_numeric(df["price_proxy"], errors="coerce")
    missing_proxy = df["price"].isna()
    if missing_proxy.any():
        df.loc[missing_proxy, "price"] = pd.to_numeric(df.loc[missing_proxy, "close_price"], errors="coerce")
    df = df.dropna(subset=["permno", "trade_date", "price"]).copy()
    df = df[df["price"] > 0].copy()
    df["permno"] = df["permno"].astype(int).astype(str)
    df = df.sort_values(["permno", "trade_date"]).drop_duplicates(subset=["permno", "trade_date"], keep="last")
    out: Dict[str, pd.DataFrame] = {}
    for permno, group in df.groupby("permno", sort=False):
        out[str(permno)] = group[["trade_date", "price"]].reset_index(drop=True)
    return out


def _price_metrics(frame: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if frame is None or frame.empty:
        return {}
    frame = frame.sort_values("trade_date").drop_duplicates(subset=["trade_date"], keep="last").copy()
    frame["ret"] = frame["price"].pct_change()
    window_end = frame["trade_date"].iloc[-1]

    results: Dict[str, Dict[str, Any]] = {}

    returns_30 = frame.loc[frame["trade_date"] > (window_end - pd.Timedelta(days=30)), ["trade_date", "ret"]].dropna()
    if len(returns_30) >= 10:
        results["market.volatility_30d"] = {
            "value": float(returns_30["ret"].std(ddof=0) * math.sqrt(252)),
            "component_breakdown": {
                "formula": "stddev(daily_returns_30d) * sqrt(252)",
                "return_observations": int(len(returns_30)),
                "annualization_factor": 252,
                "price_field": "price_proxy",
                "window_start": str(returns_30["trade_date"].iloc[0]),
                "window_end": str(returns_30["trade_date"].iloc[-1]),
                "source_kind": "crsp_market_cache",
            },
        }

    returns_90 = frame.loc[frame["trade_date"] > (window_end - pd.Timedelta(days=90)), ["trade_date", "ret"]].dropna()
    if len(returns_90) >= 20:
        results["market.volatility_90d"] = {
            "value": float(returns_90["ret"].std(ddof=0) * math.sqrt(252)),
            "component_breakdown": {
                "formula": "stddev(daily_returns_90d) * sqrt(252)",
                "return_observations": int(len(returns_90)),
                "annualization_factor": 252,
                "price_field": "price_proxy",
                "window_start": str(returns_90["trade_date"].iloc[0]),
                "window_end": str(returns_90["trade_date"].iloc[-1]),
                "source_kind": "crsp_market_cache",
            },
        }

    price_window_90 = frame.loc[frame["trade_date"] > (window_end - pd.Timedelta(days=90)), ["trade_date", "price"]].copy()
    if len(price_window_90) >= 20:
        peak_price = float(price_window_90["price"].max())
        trough_price = float(price_window_90["price"].min())
        if peak_price != 0:
            results["market.drawdown_90d"] = {
                "value": (trough_price / peak_price) - 1.0,
                "component_breakdown": {
                    "formula": "min(price_window_90d) / max(price_window_90d) - 1",
                    "price_observations": int(len(price_window_90)),
                    "price_field": "price_proxy",
                    "peak_price": peak_price,
                    "trough_price": trough_price,
                    "window_start": str(price_window_90["trade_date"].iloc[0]),
                    "window_end": str(price_window_90["trade_date"].iloc[-1]),
                    "source_kind": "crsp_market_cache",
                },
            }

    return results


def repair_price_history_metrics(
    *,
    features: Dict[str, Any],
    price_metrics: Dict[str, Dict[str, Any]],
    permno: str | None,
    computed_at: str,
) -> bool:
    changed = False
    ret3_node = features.get("market.total_return_3m_standardized")
    ret12_node = features.get("market.total_return_12m_standardized")
    provenance = _union_provenance(ret3_node, ret12_node)

    for metric_name in REPAIR_METRICS:
        target = features.get(metric_name)
        if not target or not _needs_exact_price_history_repair(target):
            continue
        repaired_payload = price_metrics.get(metric_name)
        if not repaired_payload:
            continue
        repaired = _base_repaired_node(target, computed_at=computed_at)
        repaired["value"] = repaired_payload["value"]
        repaired["fallback_used"] = "crsp_market_cache_price_history"
        repaired["support_mode"] = "exact"
        repaired["provenance"] = provenance
        component_breakdown = copy.deepcopy(repaired_payload["component_breakdown"])
        component_breakdown["selected_price_series"] = {
            "source_kind": "crsp_market_cache",
            "price_field": "price_proxy",
            "time_field": "trade_date",
            "group_field": "permno",
            "group_value": permno,
        }
        repaired["component_breakdown"] = component_breakdown
        repaired["quality_flags"] = None
        features[metric_name] = repaired
        changed = True
    return changed


def build_summary(path: Path) -> Dict[str, Dict[str, int]]:
    counters: Dict[str, Counter[str]] = {metric: Counter() for metric in REPAIR_METRICS}
    for row in iter_rows(path):
        features = row.get("features") or {}
        for metric in REPAIR_METRICS:
            node = features.get(metric) or {}
            mode = str(node.get("support_mode") or "unsupported")
            if node.get("value") is None:
                mode = "unsupported"
            counters[metric][mode] += 1
    summary: Dict[str, Dict[str, int]] = {}
    for metric, counter in counters.items():
        summary[metric] = {
            "exact": counter["exact"],
            "proxy_missing_component": counter["proxy_missing_component"],
            "unsupported": counter["unsupported"],
        }
    return summary


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)
    market_cache_path = Path(args.market_cache_path) if args.market_cache_path else _infer_market_cache_path(artifact_path)
    if market_cache_path is None:
        raise SystemExit("Could not infer market cache parquet path from artifact provenance.")

    price_cache = _load_market_cache(market_cache_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    computed_at = _now_iso()

    with out_path.open("w") as out_handle:
        for row in iter_rows(artifact_path):
            features = row.get("features") or {}
            ret_node = features.get("market.total_return_3m_standardized") or features.get("market.total_return_12m_standardized") or {}
            permno = str(((ret_node.get("component_breakdown") or {}).get("permno") or "")).strip() or None
            frame = price_cache.get(permno) if permno is not None else None
            metrics = _price_metrics(frame) if frame is not None else {}
            repair_price_history_metrics(
                features=features,
                price_metrics=metrics,
                permno=permno,
                computed_at=computed_at,
            )
            out_handle.write(json.dumps(row) + "\n")

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.write_text(json.dumps(build_summary(out_path), indent=2))

    print(f"Repaired price-history metrics -> {out_path}")


if __name__ == "__main__":
    main()
