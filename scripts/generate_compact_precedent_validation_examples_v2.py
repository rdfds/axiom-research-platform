#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RECO_DISABLE_PRECEDENT_NARRATIVE", "1")

import sys

sys.path.insert(0, str(REPO_ROOT))

from src.model_feature_bundle import _STATE_VECTOR_V1_FEATURES, build_model_feature_bundle
from src.pipeline.precedent_brain import (
    augment_precedent_state_vector_columns,
    build_precedent_pack_v2,
    build_precedent_retrieval_index,
)
from src.pipeline.run import (
    _baseline_from_world_model_features,
    _default_precedent_outcomes_path,
    adapt_snapshot,
    attach_model_feature_bundle,
    feature_view_from_snapshot,
)

SNAPSHOT_PATH = (
    REPO_ROOT
    / "out/materialized_feedback_20260405/company_state_snapshots_asof=2024-12-31.input_layer_v1_smart_normalized_with_sec.feedback_pipeline.full_inputs_v3.jsonl.gz"
)
OUT_DIR = REPO_ROOT / "out/metric_explainers"
SUMMARY_DOC_PATH = OUT_DIR / "Compact_precedent_validation_examples_2026_04_06.md"
DETAIL_DOC_PATH = OUT_DIR / "Compact_precedent_raw_and_compact_validation_2026_04_06.md"
NOTE_DOC_PATH = OUT_DIR / "Compact_precedent_validation_richer_historical_refresh_2026_04_06.md"

TARGETS = [
    {
        "company_id": "0000080424",
        "name": "The Procter & Gamble Company",
        "action_id": "capital_return.dividend_increase",
    },
    {
        "company_id": "0001018724",
        "name": "Amazon.com, Inc.",
        "action_id": "capital_return.open_market_buyback",
    },
    {
        "company_id": "0001318605",
        "name": "Tesla, Inc.",
        "action_id": "capital_return.open_market_buyback",
    },
    {
        "company_id": "0000104169",
        "name": "Walmart Inc.",
        "action_id": "capital_return.dividend_increase",
    },
]

TARGET_RAW_METRIC_ORDER = [
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

HISTORICAL_RAW_FIELD_ORDER = [
    "base_revenue_ttm",
    "base_revenue_ttm_lag_1y",
    "base_revenue_growth_yoy",
    "base_ebitda_ttm",
    "base_total_debt",
    "base_current_debt",
    "base_cash",
    "base_available_liquidity",
    "base_interest_expense",
    "base_market_cap",
    "base_ev_ebitda",
    "base_fcf_yield",
    "base_volatility_30d",
    "base_volatility_90d",
    "base_drawdown_90d",
    "base_momentum_60d",
    "base_credit_spread_level",
    "base_equity_window_proxy",
    "base_credit_window_proxy",
    "base_net_debt",
    "base_leverage",
    "base_margin",
    "macro_fed_funds_effective",
    "macro_hy_oas",
    "macro_real_gdp_growth_yoy",
]


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


def _load_snapshot_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with gzip.open(SNAPSHOT_PATH, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            company_id = str(row.get("company_id") or "")
            if company_id:
                rows[company_id] = row
    return rows


def _target_context_lines(row: dict[str, Any], bundle: dict[str, Any]) -> list[str]:
    features = row.get("features") or {}
    support = bundle["state_vector_v1"]["support"]
    proxy = [key for key in _STATE_VECTOR_V1_FEATURES if (support.get(key) or {}).get("support_mode") == "proxy_missing_component"]
    missing = [
        key
        for key in _STATE_VECTOR_V1_FEATURES
        if (support.get(key) or {}).get("support_mode") in {None, "unsupported"}
        and _is_missing(bundle["state_vector_v1"]["values"].get(key))
    ]
    sector = (features.get("taxonomy.sector") or {}).get("value")
    subsector = (features.get("taxonomy.subsector") or {}).get("value")
    regime = (features.get("capital_structure.retirement_obligation_regime") or {}).get("value")
    return [
        f"- Sector: `{sector}`",
        f"- Subsector: `{subsector}`",
        f"- Retirement regime: `{regime}`",
        f"- Proxy compact features: {', '.join(f'`{key}`' for key in proxy) if proxy else '`None`'}",
        f"- Missing compact features: {', '.join(f'`{key}`' for key in missing) if missing else '`None`'}",
    ]


