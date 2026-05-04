#!/usr/bin/env python3
"""Repair operating growth/history metrics in a materialized company-state artifact.

This pass fills a narrow set of growth and trend metrics that are often empty in
the artifact even though SEC companyfacts already contains enough quarterly
history to recover them.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

REPAIR_METRICS = [
    "operating.revenue_yoy_last_q",
    "operating.revenue_cagr_3y",
    "operating.ebitda_margin_trend_8q",
    "operating.margin_volatility_8q",
]

REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueServicesNet",
    "Revenues",
]
OPERATING_INCOME_TTM_CONCEPTS = ["OperatingIncomeLoss"]
NET_INCOME_TTM_CONCEPTS = ["NetIncomeLoss"]
INTEREST_TTM_CONCEPTS = ["InterestExpense"]
TAX_TTM_CONCEPTS = ["IncomeTaxExpenseBenefit"]
MAX_SEC_FACT_AGE_DAYS = 550
DEPRECIATION_TTM_CONCEPT_GROUPS = [
    ["DepreciationAmortizationAndAccretionNet"],
    ["DepreciationDepletionAndAmortization"],
    ["DepreciationAndAmortization"],
    ["Depreciation"],
    ["Depreciation", "AmortizationOfIntangibleAssets"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True, help="Input company-state JSONL artifact")
    parser.add_argument("--companyfacts-root", required=True, help="SEC companyfacts folder")
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


def _parse_iso_date(value: Any) -> Optional[date]:
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except Exception:  # noqa: BLE001
        return None


def _node_support(node: Dict[str, Any] | None) -> str:
    if not node:
        return "unsupported"
    return str(node.get("support_mode") or "unsupported")


