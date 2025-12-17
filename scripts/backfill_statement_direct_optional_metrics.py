#!/usr/bin/env python3
"""Backfill optional statement-direct metrics into the v1 input-layer artifact.

These metrics come from the local fact registry rather than the provider sidecar.
They are useful additions, but they are not part of the tightest universal core
because coverage is materially lower than the provider-direct baseline.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import duckdb
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backfill_input_layer_v1_metrics as core  # noqa: E402

MAX_SEC_FACT_AGE_DAYS = 550
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_COMPANYFACTS_ROOT = REPO_ROOT / "data" / "sec" / "companyfacts"
DEBT_COMPONENT_ALIGNMENT_MAX_GAP_DAYS = 10
STATEMENT_DEBT_REPAIR_MAX_GAP_DAYS = 45
STATEMENT_DEBT_EXACT_MAX_AGE_DAYS = 130
STATEMENT_DEBT_MATCH_TOLERANCE = 1.0
DEPRECIATION_TTM_CONCEPT_GROUPS = [
    ["DepreciationAmortizationAndAccretionNet"],
    ["DepreciationDepletionAndAmortization"],
    ["DepreciationAndAmortization"],
    ["Depreciation"],
    ["Depreciation", "AmortizationOfIntangibleAssets"],
]
INTEREST_EXPENSE_TTM_EXACT_CONCEPTS = [
    "InterestExpense",
]

STATEMENT_FACT_SPECS = {
    "liquidity.cash_and_equivalents_statement_direct": {
        "fact_type": "financial.cash",
        "unit": "usd",
    },
    "capital_structure.current_debt_statement_direct": {
        "fact_type": "financial.debt_current",
        "unit": "usd",
    },
    "capital_structure.long_term_debt_statement_direct": {
        "fact_type": "financial.debt_long_term",
        "unit": "usd",
    },
    "operating.ebit_statement_direct": {
        "fact_type": "financial.ebit",
        "unit": "usd",
    },
    "capital_structure.interest_expense_statement_direct": {
        "fact_type": "financial.interest_expense",
        "unit": "usd",
    },
}


class _CompanyProcessingTimeout(RuntimeError):
    """Raised when a single-company optional-metric build exceeds the allowed timeout."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-path", required=True, help="Input snapshot JSONL path")
    parser.add_argument("--facts-path", required=True, help="Local fact registry parquet")
    parser.add_argument(
        "--entity-batch-size",
        type=int,
        default=128,
        help="Number of companies to enrich per fact-registry query batch.",
    )
    parser.add_argument(
        "--company-processing-timeout-seconds",
        type=float,
        default=15.0,
        help="Fail open on a single company if statement-optional enrichment exceeds this timeout. Use 0 to disable.",
    )
    parser.add_argument(
        "--companyfacts-root",
        help="Optional SEC companyfacts folder for EBITDA repair. Defaults to the local canonical companyfacts root when present.",
    )
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--summary-out", help="Optional summary JSON path")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_snapshot_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


