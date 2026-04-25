"""
CompanyStateSnapshot builder (World Model spec).

This module assembles a point-in-time company snapshot using only
information available as-of a given timestamp. It is intentionally
conservative and auditable: every feature carries provenance and
confidence metadata. Missing data is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re
import subprocess
import time
import uuid

import duckdb
import numpy as np
import pandas as pd

from scripts.backfill_sec_companyfacts_components import (
    _extract_lease_liabilities as _extract_sec_companyfacts_lease_liabilities,
    _extract_marketable_securities as _extract_sec_companyfacts_marketable_securities,
    _extract_restricted_cash as _extract_sec_companyfacts_restricted_cash,
    _extract_revolver_undrawn as _extract_sec_companyfacts_revolver_undrawn,
)
from scripts.backfill_input_layer_v1_metrics import (
    DEBT_CURRENT_CONCEPTS,
    DEBT_NONCURRENT_CONCEPTS,
    SHORT_TERM_BORROWINGS_CONCEPTS,
    _build_sec_core_metric,
    _instant_candidates,
)
from scripts.backfill_smart_normalized_metrics_v1 import materialize_smart_metrics_for_row

from .company_state_input_source_registry import CompanyStateInputSourceRegistry
from .company_state_components import RegimeClassifier, PeerSetResolver
from .data_paths import resolve_companyfacts_root, resolve_data_path
from .metric_policy import MetricPolicyEngine, TaxonomyContext


ROOT = Path(__file__).resolve().parents[1]
MAX_SEC_FACT_AGE_DAYS = 550
COMPANYFACTS_FRESHER_OVERRIDE_MIN_DAYS = 7
COMPANYFACTS_LOAD_TIMEOUT_SECONDS = 1.5
COMPANYFACTS_CASH_EQ_CONCEPTS = ["CashAndCashEquivalentsAtCarryingValue", "Cash"]
INTEREST_EXPENSE_TTM_EXACT_CONCEPTS = ["InterestExpense"]


def _smart_metric_registry_path() -> Path:
    env = str(os.environ.get("AXIOM_SMART_METRIC_REGISTRY_PATH", "") or "").strip()
    if env:
        return Path(env)
    return ROOT / "out" / "smart_metric_registry_v1.json"


def _market_availability_overrides_path() -> Path:
    env = str(os.environ.get("AXIOM_MARKET_AVAILABILITY_OVERRIDES_PATH", "") or "").strip()
    if env:
        return Path(env)
    return ROOT / "configs" / "liquidity_market_availability_overrides.json"

EXACT_SUPPORT_MODES = {"exact", "exact_not_applicable", "exact_structural_zero"}
PROXY_COMPONENT_SUPPORT_FLAGS = {
    "decision_uses_reported_view",
    "lease_adjustment_missing_assumed_zero",
    "lease_adjusted_denominator_fallback_to_ebitda",
    "lease_adjusted_denominator_missing_lease_expense",
    "lease_expense_estimated_from_liabilities",
    "lease_fixed_charge_proxy_from_liability",
    "marketable_securities_included_at_par_proxy",
    "minimum_cash_policy_proxy_not_applied_to_market_view",
    "mixed_quarter_value_basis",
    "pension_excluded_from_debt",
    "preferred_equity_excluded_pending_hybrid_review",
    "provider_fcf_fallback",
    "quarter_value_derived_from_ytd_delta",
    "reference_ebitda_fallback",
    "reference_total_debt_fallback",
    "reference_total_debt_used_for_completeness",
    "recent_total_debt_peak_used_for_completeness",
    "restricted_cash_missing_assumed_zero",
    "supplier_finance_included_without_payables_extension_test",
    "convertibles_excluded_pending_hybrid_review",
}
NON_PROXY_DIAGNOSTIC_FLAGS = {
    "companyfacts_cash_fresher",
    "companyfacts_total_debt_fresher",
    "fixed_charge_coverage_preferred",
    "latest_recurring_dividend_outside_active_window",
    "multiple_price_series_candidates",
    "no_recurring_dividend_events_in_history",
    "no_strategic_actions_in_window",
    "non_price_series_filtered",
    "pe_history_unavailable",
    "price_shares_fallback",
    "provider_market_cap_missing",
    "reference_market_cap_fallback",
    "reference_market_cap_preferred_over_stale_price_shares",
    "shares_basic_fallback",
}


# -------------------------------
# Data classes / schemas
# -------------------------------


