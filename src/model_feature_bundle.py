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


def _resolved_action_family(*, action_type: Optional[str] = None, action_id: Optional[str] = None) -> Optional[str]:
    family = str(action_type or "").strip().lower()
    if family:
        return family
    aid = str(action_id or "").strip().lower()
    if "." in aid:
        return aid.split(".", 1)[0]
    return aid or None


def _resolve_first_record(
    features: Dict[str, Any],
    source_keys: Iterable[str],
    *,
    action_type: Optional[str] = None,
    action_id: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    for source_key in source_keys:
        record = resolve_feature_record(
            features,
            source_key,
            action_family=action_type,
            action_id=action_id,
        )
        if record is None:
            continue
        if _feature_value(record) is None:
            continue
        actual_source = source_key
        if isinstance(record, dict):
            runtime_adapter_meta = dict((record['component_breakdown'] or {}).get("runtime_feature_adapter", {}) or {})
            actual_source = str(runtime_adapter_meta.get("source_metric") or source_key)
        return record, actual_source
    return None, None


def _canonical_meta(record: Any, *, source_metric: Optional[str]) -> Dict[str, Any]:
    return {
        "source_metric": source_metric,
        "support_mode": _support_mode(record),
        "applicability_status": _applicability_status(record),
        "quality_flags": _quality_flags(record),
        "is_proxy": bool(_support_mode(record) and _support_mode(record) not in _EXACTISH_SUPPORT_MODES),
        "is_legacy": bool(source_metric and ".normalized" not in source_metric and source_metric.startswith(("capital_structure.", "liquidity.", "operating.", "macro.", "market."))),
    }


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(_feature_value(value))
    except Exception:
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _is_exactish_support_mode(mode: Optional[str]) -> bool:
    normalized = str(mode).strip().lower() if mode is not None else None
    return normalized in _EXACTISH_SUPPORT_MODES


def _all_exactish_support(
    support: Dict[str, Dict[str, Any]],
    *keys: Optional[str],
) -> bool:
    usable = [str(key) for key in keys if key]
    if not usable:
        return False
    return all(_is_exactish_support_mode((support.get(key, {}) or {}).get("support_mode")) for key in usable)


def _derived_support_meta(
    *,
    source_metric: str,
    support_mode: Optional[str],
    quality_flags: Iterable[str] = (),
) -> Dict[str, Any]:
    normalized_mode = str(support_mode).strip().lower() if support_mode is not None else None
    return {
        "source_metric": source_metric,
        "support_mode": normalized_mode,
        "applicability_status": None,
        "quality_flags": [str(flag) for flag in quality_flags if flag],
        "is_proxy": bool(normalized_mode and normalized_mode not in _EXACTISH_SUPPORT_MODES),
        "is_legacy": False,
    }


