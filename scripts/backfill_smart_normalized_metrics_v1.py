#!/usr/bin/env python3
"""Materialize the first smart-normalized layer on top of the v1 input artifact.

This script is intentionally conservative:
- it reads the ontology/policy JSON artifacts created for the smart layer
- it emits the first normalized metric ids
- it uses `proxy_missing_component` whenever the metric is economically useful
  but still missing required exact components for full promotion

The goal is not to pretend the ontology is finished; the goal is to make the
current smartest defensible layer explicit and machine-usable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

try:
    import pandas as pd
except Exception:  # noqa: BLE001
    pd = None

try:
    import requests
except Exception:  # noqa: BLE001
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:  # noqa: BLE001
    BeautifulSoup = None

try:
    from backfill_statement_direct_optional_metrics import (
        DEPRECIATION_TTM_CONCEPT_GROUPS,
        _compute_ttm_from_concept,
    )
    from backfill_input_layer_v1_metrics import _build_sec_core_metric
except Exception:  # noqa: BLE001
    from scripts.backfill_statement_direct_optional_metrics import (  # type: ignore
        DEPRECIATION_TTM_CONCEPT_GROUPS,
        _compute_ttm_from_concept,
    )
    from scripts.backfill_input_layer_v1_metrics import _build_sec_core_metric  # type: ignore


SMART_METRIC_NAMES = [
    "capital_structure.debt_like_obligations_normalized",
    "capital_structure.net_pension_liability",
    "capital_structure.other_postretirement_benefit_liability",
    "capital_structure.combined_retirement_liability",
    "capital_structure.debt_like_obligations_including_pension",
    "capital_structure.debt_like_obligations_including_retirement",
    "liquidity.available_liquidity_normalized",
    "operating.operating_earnings_normalized",
    "capital_structure.net_debt_normalized",
    "capital_structure.net_debt_including_pension",
    "capital_structure.net_debt_including_retirement",
    "capital_structure.gross_leverage_normalized",
    "capital_structure.gross_leverage_including_pension",
    "capital_structure.gross_leverage_including_retirement",
    "capital_structure.net_leverage_normalized",
    "capital_structure.net_leverage_including_pension",
    "capital_structure.net_leverage_including_retirement",
]

RECONCILIATION_TOLERANCE = 1.0
LEASE_STALE_CARRY_FORWARD_MAX_AGE_DAYS = 420
LEASE_ROU_FRESH_MAX_AGE_DAYS = 220
LEASE_IMMATERIAL_STALE_COMPONENT_MAX_ABS_USD = 25_000_000.0
COMPANYFACTS_LOAD_TIMEOUT_SECONDS = 1.5
RETIREMENT_NOTE_PARSE_TIMEOUT_SECONDS = 2.0
LEASE_IMMATERIAL_STALE_COMPONENT_MAX_RELATIVE_TO_DEBT = 0.01
FRESHER_COMPANYFACTS_OVERRIDE_MIN_GAP_DAYS = 7
OPERATING_EARNINGS_TAX_CONCEPTS = [
    "IncomeTaxExpenseBenefit",
]
OPERATING_EARNINGS_INTEREST_CONCEPTS = [
    "InterestExpense",
]
CASH_EQ_COMPANYFACTS_CONCEPTS = [
    "CashAndCashEquivalentsAtCarryingValue",
    "Cash",
]
NET_PENSION_LIABILITY_EXACT_TOTAL_CONCEPTS = [
    "DefinedBenefitPensionPlanProjectedBenefitObligationExcessPlanAssets",
    "DefinedBenefitPensionPlanProjectedBenefitObligationExcessFairValueOfPlanAssets",
    "DefinedBenefitPensionPlanUnderfundedStatus",
    "DefinedBenefitPensionPlanNetLiabilityRecognized",
]
NET_PENSION_LIABILITY_EXACT_CURRENT_CONCEPTS = [
    "DefinedBenefitPensionPlanLiabilitiesCurrent",
]
NET_PENSION_LIABILITY_EXACT_NONCURRENT_CONCEPTS = [
    "DefinedBenefitPensionPlanLiabilitiesNoncurrent",
]
NET_PENSION_LIABILITY_PROXY_TOTAL_CONCEPTS = [
    "PensionAndOtherPostretirementDefinedBenefitPlansLiabilities",
    "PensionAndOtherPostretirementAndPostemploymentBenefitPlansLiabilities",
]
NET_PENSION_LIABILITY_PROXY_CURRENT_CONCEPTS = [
    "PensionAndOtherPostretirementDefinedBenefitPlansCurrentLiabilities",
    "PensionAndOtherPostretirementAndPostemploymentBenefitPlansCurrentLiabilities",
]
NET_PENSION_LIABILITY_PROXY_NONCURRENT_CONCEPTS = [
    "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent",
    "PensionAndOtherPostretirementAndPostemploymentBenefitPlansLiabilitiesNoncurrent",
]
DEFAULT_MARKET_AVAILABILITY_OVERRIDES_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "liquidity_market_availability_overrides.json"
)
DEFAULT_LOCAL_COMPANYFACTS_ROOT = Path(__file__).resolve().parents[1] / "data" / "sec" / "companyfacts"
DEFAULT_SEC_RETIREMENT_CACHE_ROOT = Path("/tmp/sec_retirement_note_cache")
RETIREMENT_NOTE_CARRYFORWARD_MAX_AGE_DAYS = 430
RETIREMENT_NOTE_MAX_FILINGS_TO_SCAN = 6
_COMPANYFACTS_UNSET = object()
SEC_USER_AGENT = "Codex/axiom_v1 retirement note support"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
RETIREMENT_NOTE_TABLE_CUES = (
    "pension benefits",
    "other benefits",
    "other postretirement",
    "postretirement benefits",
    "retirement-related benefits",
    "funded status",
    "benefit obligation",
)
RETIREMENT_NOTE_PENSION_COLUMN_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bpension\b",
        r"defined[\s\-]benefit",
    )
]
RETIREMENT_NOTE_OTHER_POSTRETIREMENT_COLUMN_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"other\s+benefits",
        r"other\s+postretirement",
        r"postretirement",
    )
]
RETIREMENT_FUNDED_STATUS_LABEL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"funded status",
        r"net amount recognized",
        r"net amount .* recognized",
    )
]
RETIREMENT_DIRECT_LIABILITY_LABEL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"liabilit(?:y|ies)",
        r"accrued .* benefit",
        r"amount recognized .* balance sheet",
    )
]
RETIREMENT_BENEFIT_OBLIGATION_LABEL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"projected benefit obligation",
        r"accumulated benefit obligation",
        r"benefit obligation .* end of year",
        r"benefit obligation",
    )
]
RETIREMENT_PLAN_ASSETS_LABEL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"fair value of plan assets",
        r"plan assets .* end of year",
        r"plan assets",
    )
]


SMART_METRIC_UNITS = {
    "capital_structure.debt_like_obligations_normalized": "usd",
    "capital_structure.net_pension_liability": "usd",
    "capital_structure.other_postretirement_benefit_liability": "usd",
    "capital_structure.combined_retirement_liability": "usd",
    "capital_structure.debt_like_obligations_including_pension": "usd",
    "capital_structure.debt_like_obligations_including_retirement": "usd",
    "liquidity.available_liquidity_normalized": "usd",
    "operating.operating_earnings_normalized": "usd",
    "capital_structure.net_debt_normalized": "usd",
    "capital_structure.net_debt_including_pension": "usd",
    "capital_structure.net_debt_including_retirement": "usd",
    "capital_structure.gross_leverage_normalized": "x",
    "capital_structure.gross_leverage_including_pension": "x",
    "capital_structure.gross_leverage_including_retirement": "x",
    "capital_structure.net_leverage_normalized": "x",
    "capital_structure.net_leverage_including_pension": "x",
    "capital_structure.net_leverage_including_retirement": "x",
}


class _CompanyProcessingTimeout(RuntimeError):
    """Raised when a single-company smart-normalized build exceeds the allowed timeout."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-path", required=True, help="Input snapshot JSONL path")
    parser.add_argument("--metric-registry-path", required=True, help="Smart metric registry JSON")
    parser.add_argument("--component-policy-path", required=True, help="Component inclusion policy JSON")
    parser.add_argument("--source-precedence-path", required=True, help="Source precedence policy JSON")
    parser.add_argument(
        "--companyfacts-root",
        help="Optional SEC companyfacts folder for operating-earnings repair. Defaults to the local canonical companyfacts root when present.",
    )
    parser.add_argument(
        "--sec-filing-cache-root",
        help=(
            "Optional cache root for SEC filing-note HTML used to separate pension from other postretirement "
            "liabilities when companyfacts only exposes combined concepts. Defaults to /tmp/sec_retirement_note_cache."
        ),
    )
    parser.add_argument(
        "--market-availability-overrides-path",
        help="Optional JSON file with explicit market-availability cash adjustments",
    )
    parser.add_argument(
        "--company-processing-timeout-seconds",
        type=float,
        default=15.0,
        help="Fail open on a single company if smart-normalized enrichment exceeds this timeout. Use 0 to disable.",
    )
    parser.add_argument(
        "--resume-if-exists",
        action="store_true",
        help="If the output file already exists, skip completed company ids and append remaining rows.",
    )
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--summary-out", help="Optional summary JSON path")
    return parser.parse_args()


