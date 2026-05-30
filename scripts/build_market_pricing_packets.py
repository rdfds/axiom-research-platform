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


