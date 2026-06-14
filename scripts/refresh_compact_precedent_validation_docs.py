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


def _feature_table(bundle: dict[str, Any]) -> str:
    values = bundle["state_vector_v1"]["values"]
    lines = ["| Feature | Value |", "|---|---:|"]
    for key in _STATE_VECTOR_V1_FEATURES:
        lines.append(f"| `{key}` | {_fmt_value(values.get(key))} |")
    return "\n".join(lines)


def _support_lines(row: dict[str, Any], bundle: dict[str, Any], *, missing_label: str) -> str:
    features = row.get("features") or {}
    support = bundle["state_vector_v1"]["support"]
    proxy = [key for key in _STATE_VECTOR_V1_FEATURES if (support.get(key) or {}).get("support_mode") == "proxy_missing_component"]
    missing = [key for key in _STATE_VECTOR_V1_FEATURES if (support.get(key) or {}).get("support_mode") in {None, "unsupported"} and _is_missing(bundle["state_vector_v1"]["values"].get(key))]
    sector = (features['taxonomy.sector'] or {}).get("value")
    subsector = (features.get("taxonomy.subsector") or {}).get("value")
    regime = (features.get("capital_structure.retirement_obligation_regime") or {}).get("value")
    proxy_text = ", ".join(f"`{key}`" for key in proxy) if proxy else "`None`"
    missing_text = ", ".join(f"`{key}`" for key in missing) if missing else "`None`"
    return "\n".join(
        [
            f"- Sector: `{sector}`",
            f"- Subsector: `{subsector}`",
            f"- Retirement regime: `{regime}`",
            f"- Proxy features: {proxy_text}",
            f"- {missing_label}: {missing_text}",
        ]
    )


def _raw_metric_table(row: dict[str, Any]) -> str:
    features = row.get("features") or {}
    lines = ["| Raw metric | Value | Support |", "|---|---:|---|"]
    for metric in RAW_METRIC_ORDER:
        record = features.get(metric) or {}
        value = record.get("value")
        support_mode = record.get("support_mode") or "unsupported"
        lines.append(f"| `{metric}` | {_fmt_value(value)} | `{support_mode}` |")
    return "\n".join(lines)


