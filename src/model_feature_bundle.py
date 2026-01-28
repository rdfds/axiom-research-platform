from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterable, Optional, Tuple

from .runtime_feature_adapter import resolve_feature_record


_BUNDLE_KEY = "_model_feature_bundle"
_EXACTISH_SUPPORT_MODES = {
    "exact",
    "exact_not_applicable",
    "exact_structural_zero",
}
_VIEW_NAMES = (
    "candidate_generation",
    "mechanism",
    "causal",
    "precedent",
    "dossier",
)
_RETIREMENT_REGIME_FLAGS = {
    "pension_exact": "capital.retirement_regime_pension_exact",
    "pension_proxy_split_note": "capital.retirement_regime_pension_proxy_split_note",
    "combined_retirement_only": "capital.retirement_regime_combined_retirement_only",
    "defined_contribution_only": "capital.retirement_regime_defined_contribution_only",
    "retirement_not_surfaced": "capital.retirement_regime_not_surfaced",
}
_STATE_VECTOR_V1_FEATURES = (
    "state_vector_v1.size_log_revenue",
    "state_vector_v1.profitability",
    "state_vector_v1.growth",
    "state_vector_v1.gross_obligation_burden",
    "state_vector_v1.net_obligation_burden",
    "state_vector_v1.liquidity_flexibility",
    "state_vector_v1.interest_coverage",
    "state_vector_v1.valuation_multiple",
    "state_vector_v1.cash_generation",
    "state_vector_v1.market_stress",
    "state_vector_v1.market_access",
    "state_vector_v1.rates_level",
    "state_vector_v1.credit_spread",
)
_LEGACY_COMPAT_KEYS = (
    "capital_structure.net_debt",
    "capital_structure.net_leverage",
    "capital_structure.gross_leverage",
    "liquidity.available_for_actions",
    "operating.ebitda_ttm",
    "macro.rate_10y",
    "macro.rate_2y",
    "macro.sofr",
    "market.ig_oas",
    "market.hy_oas",
    "market.pe",
)
_CANONICAL_SPECS: Dict[str, Tuple[str, ...]] = {
    "scale.market_cap": ("market.market_cap_provider_direct", "market.market_cap"),
    "scale.enterprise_value": ("market.enterprise_value", "market.enterprise_value_provider_direct"),
    "scale.revenue_ttm": ("operating.revenue_ttm_provider_direct", "operating.revenue_ttm"),
    "scale.ebitda_ttm": (
        "operating.ebitda_ltm_provider_direct",
        "operating.ebitda_ttm",
        "operating.operating_earnings_normalized",
    ),
    "operating.revenue_ttm": ("operating.revenue_ttm_provider_direct", "operating.revenue_ttm"),
    "operating.revenue_ttm_lag_1y": (
        "operating.revenue_ttm_lag_1y",
        "operating.revenue_ttm_prior_year",
        "operating.revenue_ttm_prev_year",
    ),
    "operating.revenue_yoy_last_q": ("operating.revenue_yoy_last_q",),
    "operating.ebitda_margin_ttm": ("operating.ebitda_margin_ttm",),
    "operating.operating_earnings_normalized": ("operating.operating_earnings_normalized",),
    "operating.roic": ("operating.roic",),
    "operating.fcf_conversion": ("operating.fcf_conversion",),
    "cash_flow.free_cash_flow_ttm": (
        "cash_flow.free_cash_flow_ttm",
        "operating.free_cash_flow_ttm",
        "cash_flow.free_cash_flow",
        "free_cash_flow_ttm",
    ),
    "capital.total_debt": (
        "capital_structure.total_debt_provider_direct",
        "capital_structure.total_debt_reported",
        "capital_structure.total_debt",
    ),
    "capital.net_debt": (
        "capital_structure.net_debt_normalized",
        "capital_structure.net_debt_standardized",
        "capital_structure.net_debt",
    ),
    "capital.debt_like_obligations": ("capital_structure.debt_like_obligations_normalized",),
    "capital.gross_leverage": ("capital_structure.gross_leverage",),
    "capital.net_leverage": ("capital_structure.net_leverage",),
    "capital.interest_coverage": ("capital_structure.interest_coverage",),
    "capital.interest_expense": (
        "capital_structure.interest_expense_statement_direct",
        "capital_structure.interest_expense",
    ),
    "capital.debt_due_0_12m": ("capital_structure.debt_due_0_12m",),
    "capital.debt_due_12_24m": ("capital_structure.debt_due_12_24m",),
    "capital.debt_due_next_24m": ("capital_structure.debt_due_next_24m",),
    "capital.current_debt": (
        "capital_structure.current_debt_statement_direct",
        "capital_structure.current_debt_provider_direct",
        "capital_structure.current_debt",
    ),
    "capital.maturity_wall_ratio_24m": ("capital_structure.maturity_wall_ratio_24m",),
    "capital.rating_state": ("capital_structure.rating_state",),
    "capital.lease_liabilities": ("capital_structure.lease_liabilities_sec_exact",),
    "capital.net_pension_liability": ("capital_structure.net_pension_liability",),
    "capital.combined_retirement_liability": ("capital_structure.combined_retirement_liability",),
    "capital.debt_like_obligations_including_retirement": (
        "capital_structure.debt_like_obligations_including_retirement",
    ),
    "capital.net_debt_including_retirement": ("capital_structure.net_debt_including_retirement",),
    "capital.gross_leverage_including_retirement": ("capital_structure.gross_leverage_including_retirement",),
    "capital.net_leverage_including_retirement": ("capital_structure.net_leverage_including_retirement",),
    "capital.retirement_obligation_regime": ("capital_structure.retirement_obligation_regime",),
    "liquidity.cash": ("liquidity.cash_and_short_term_investments_provider_direct", "liquidity.cash"),
    "liquidity.minimum_cash_policy_proxy": ("liquidity.minimum_cash_policy_proxy",),
    "liquidity.available_for_actions": ("liquidity.available_for_actions",),
    "liquidity.available_liquidity_normalized": ("liquidity.available_liquidity_normalized",),
    "liquidity.runway_months": ("liquidity.runway_months",),
    "liquidity.revolver_undrawn": ("liquidity.revolver_undrawn",),
    "liquidity.marketable_securities": (
        "liquidity.marketable_securities_sec_exact",
        "liquidity.marketable_securities",
    ),
    "market.market_cap": ("market.market_cap",),
    "market.pe": ("market.pe", "market.pe_ratio"),
    "market.ev_ebitda": ("market.ev_ebitda",),
    "market.drawdown_90d": ("market.drawdown_90d",),
    "market.volatility_30d": ("market.volatility_30d",),
    "market.volatility_90d": ("market.volatility_90d",),
    "market.credit_spread_level": ("market.credit_spread_level",),
    "market.equity_window_proxy": ("market.equity_window_proxy",),
    "market.credit_window_proxy": ("market.credit_window_proxy",),
    "market.fcf_yield": ("market.fcf_yield",),
    "market.fcf_yield_percentile_peers": ("market.fcf_yield_percentile_peers",),
    "market.ev_ebitda_vs_peer_z": ("market.ev_ebitda_vs_peer_z",),
    "market.vix": ("market.vix",),
    "macro.ust_10y_yield": ("macro.ust_10y_yield", "macro.rate_10y", "macro.us10y_treasury_yield"),
    "macro.ust_2y_yield": ("macro.ust_2y_yield", "macro.rate_2y"),
    "macro.sofr": ("macro.sofr",),
    "macro.fed_funds_effective": ("macro.fed_funds_effective",),
    "macro.real_gdp_growth_yoy": ("macro.real_gdp_growth_yoy",),
    "macro.ig_oas": ("macro.ig_oas", "market.ig_oas", "macro.us_ig_oas"),
    "macro.hy_oas": ("macro.hy_oas", "market.hy_oas"),
    "macro.curve_2s10s": ("macro.curve_2s10s",),
    "taxonomy.sector": ("taxonomy.sector",),
    "taxonomy.subsector": ("taxonomy.subsector",),
}


def _feature_value(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw.get("value")
    return raw


def _support_mode(raw: Any) -> Optional[str]:
    if isinstance(raw, dict):
        value = raw.get("support_mode")
        return str(value).strip().lower() if value is not None else None
    return None


def _applicability_status(raw: Any) -> Optional[str]:
    if isinstance(raw, dict):
        value = raw.get("applicability_status")
        return str(value).strip().lower() if value is not None else None
    return None


def _quality_flags(raw: Any) -> list[str]:
    if not isinstance(raw, dict):
        return []
    return [str(flag) for flag in (raw.get("quality_flags") or []) if flag is not None]


def _reliability_score(record: Any, *, source_metric: str) -> float:
    if record is None:
        return 0.0
    support_mode = _support_mode(record)
    if support_mode in _EXACTISH_SUPPORT_MODES:
        return 1.0
    if support_mode == "proxy_missing_component":
        return 0.65
    if support_mode and support_mode.startswith("proxy"):
        return 0.45
    if ".normalized" not in source_metric and support_mode not in {"unsupported", None}:
        return 0.70
    if support_mode is None and _feature_value(record) is not None:
        return 0.70
    return 0.0


def _copy_record_for_target(record: Any, *, target_key: str, source_key: str) -> Optional[Dict[str, Any]]:
    if record is None:
        return None
    if isinstance(record, dict):
        out = copy.deepcopy(record)
    else:
        out = {"value": record}
    out["name"] = target_key
    source_map = dict(out.get("component_breakdown") or {})
    source_map["model_feature_bundle"] = {
        "target_metric": target_key,
        "source_metric": source_key,
    }
    out["component_breakdown"] = source_map
    return out


