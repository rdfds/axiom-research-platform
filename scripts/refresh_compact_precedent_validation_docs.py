#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model_feature_bundle import _STATE_VECTOR_V1_FEATURES, build_model_feature_bundle

ARTIFACT_PATH = REPO_ROOT / "out/materialized_feedback_20260405/company_state_snapshots_asof=2024-12-31.input_layer_v1_smart_normalized_with_sec.feedback_pipeline.full_inputs_v3.jsonl.gz"
VALIDATION_DOC_PATH = REPO_ROOT / "out/metric_explainers/Compact_precedent_validation_examples_2026_04_05.md"
RAW_VALIDATION_DOC_PATH = REPO_ROOT / "out/metric_explainers/Compact_precedent_raw_and_compact_validation_2026_04_05.md"
FOLLOWUP_NOTE_PATH = REPO_ROOT / "out/metric_explainers/State_vector_input_completeness_followup_2026_04_05.md"

COMPANY_IDS = {
    "0000080424": "The Procter & Gamble Company",
    "0001018724": "Amazon.com, Inc.",
    "0001318605": "Tesla, Inc.",
    "0000104169": "Walmart Inc.",
}

RAW_METRIC_ORDER = [
    "operating.revenue_ttm_provider_direct",
    "operating.revenue_ttm_lag_1y",
    "operating.ebitda_ltm_provider_direct",
    "operating.ebitda_margin_ttm",
    "capital_structure.total_debt_provider_direct",
    "capital_structure.net_debt_normalized",
    "capital_structure.lease_liabilities_sec_exact",
    "capital_structure.combined_retirement_liability",
    "liquidity.cash_and_short_term_investments_provider_direct",
    "liquidity.marketable_securities_sec_exact",
    "liquidity.revolver_undrawn",
    "liquidity.available_liquidity_normalized",
    "capital_structure.current_debt_statement_direct",
    "capital_structure.current_debt_provider_direct",
    "capital_structure.interest_expense_statement_direct",
    "capital_structure.interest_coverage",
    "market.market_cap_provider_direct",
    "market.enterprise_value",
    "market.ev_ebitda",
    "market.fcf_yield",
    "market.volatility_90d",
    "market.drawdown_90d",
    "market.credit_window_proxy",
    "market.equity_window_proxy",
    "market.credit_spread_level",
    "macro.fed_funds_effective",
    "macro.hy_oas",
]


def _load_snapshots() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with gzip.open(ARTIFACT_PATH, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            company_id = str(row.get("company_id") or "")
            if company_id in COMPANY_IDS:
                rows[company_id] = row
    missing = set(COMPANY_IDS) - set(rows)
    if missing:
        raise RuntimeError(f"Missing rows in artifact for company ids: {sorted(missing)}")
    return rows


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and not math.isfinite(value))


def _fmt_value(value: Any) -> str:
    if _is_missing(value):
        return "`null`"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    if isinstance(value, int):
        return f"`{value}`"
    if isinstance(value, float):
        return f"`{value:.4f}`"
    return f"`{value}`"


