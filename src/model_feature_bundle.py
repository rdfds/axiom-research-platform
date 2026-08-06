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
            runtime_adapter_meta = dict((record.get("component_breakdown", {}) or {}).get("runtime_feature_adapter", {}) or {})
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


def _state_vector_record(
    *,
    key: str,
    value: Any,
    support_mode: Optional[str],
    formula: Optional[str] = None,
    component_values: Optional[Dict[str, Any]] = None,
    quality_flags: Optional[Iterable[str]] = None,
    fallback_used: Optional[str] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "name": key,
        "value": value,
        "support_mode": support_mode,
    }
    breakdown = dict(component_values or {})
    if formula:
        breakdown["formula"] = formula
    if breakdown:
        out["component_breakdown"] = breakdown
    flags = [str(flag) for flag in (quality_flags or []) if flag]
    if flags:
        out["quality_flags"] = flags
    if fallback_used:
        out["fallback_used"] = fallback_used
    return out


def _copy_state_metric(
    state_values: Dict[str, Any],
    state_records: Dict[str, Dict[str, Any]],
    state_support: Dict[str, Dict[str, Any]],
    state_reliability: Dict[str, float],
    state_sources: Dict[str, str],
    *,
    key: str,
    source_key: str,
    canonical: Dict[str, Any],
    records: Dict[str, Dict[str, Any]],
    support: Dict[str, Dict[str, Any]],
    reliability: Dict[str, float],
    sources: Dict[str, str],
    transform: Optional[str] = None,
    value: Any = None,
    extra_quality_flags: Iterable[str] = (),
) -> None:
    source_record = records.get(source_key)
    source_metric = str(sources.get(source_key) or source_key)
    state_value = canonical.get(source_key) if value is None else value
    support_mode = (support.get(source_key, {}) or {}).get("support_mode")
    if state_value is None and support_mode is None:
        support_mode = "unsupported"
    quality_flags = list(dict.fromkeys([*_quality_flags(source_record), *list(extra_quality_flags)]))
    record = (
        _copy_record_for_target(
            source_record,
            target_key=key,
            source_key=source_metric,
        )
        if source_record is not None
        else _state_vector_record(
            key=key,
            value=state_value,
            support_mode=support_mode,
            quality_flags=quality_flags,
        )
    )
    if transform:
        breakdown = dict((record or {}).get("component_breakdown") or {})
        breakdown["state_vector_transform"] = transform
        record["component_breakdown"] = breakdown
    if state_value is not None:
        record["value"] = state_value
    if quality_flags:
        record["quality_flags"] = quality_flags
    state_values[key] = state_value
    state_records[key] = record
    state_support[key] = _derived_support_meta(
        source_metric=f"derived:{key}",
        support_mode=support_mode,
        quality_flags=quality_flags,
    )
    state_reliability[key] = float(reliability.get(source_key, 0.0))
    state_sources[key] = source_metric


def _set_state_metric(
    state_values: Dict[str, Any],
    state_records: Dict[str, Dict[str, Any]],
    state_support: Dict[str, Dict[str, Any]],
    state_reliability: Dict[str, float],
    state_sources: Dict[str, str],
    *,
    key: str,
    value: Any,
    support_mode: Optional[str],
    source_metric: str,
    formula: Optional[str] = None,
    component_values: Optional[Dict[str, Any]] = None,
    component_reliability: Iterable[float] = (),
    quality_flags: Iterable[str] = (),
    fallback_used: Optional[str] = None,
) -> None:
    flags = [str(flag) for flag in quality_flags if flag]
    state_values[key] = value
    state_records[key] = _state_vector_record(
        key=key,
        value=value,
        support_mode=support_mode,
        formula=formula,
        component_values=component_values,
        quality_flags=flags,
        fallback_used=fallback_used,
    )
    state_support[key] = _derived_support_meta(
        source_metric=source_metric,
        support_mode=support_mode,
        quality_flags=flags,
    )
    rel_values = [float(x) for x in component_reliability if x is not None]
    state_reliability[key] = float(sum(rel_values) / len(rel_values)) if rel_values else 0.0
    state_sources[key] = source_metric


def _build_state_vector_v1(
    snapshot: Dict[str, Any],
    *,
    canonical: Dict[str, Any],
    records: Dict[str, Dict[str, Any]],
    support: Dict[str, Dict[str, Any]],
    reliability: Dict[str, float],
    sources: Dict[str, str],
) -> Dict[str, Any]:
    features = snapshot.get("features") or {}
    state_values: Dict[str, Any] = {}
    state_records: Dict[str, Dict[str, Any]] = {}
    state_support: Dict[str, Dict[str, Any]] = {}
    state_reliability: Dict[str, float] = {}
    state_sources: Dict[str, str] = {}

    revenue = _safe_float(canonical.get("scale.revenue_ttm"))
    ebitda = _safe_float(canonical.get("scale.ebitda_ttm"))
    if revenue is not None and revenue > 0:
        _set_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.size_log_revenue",
            value=math.log10(max(revenue, 1.0)),
            support_mode=(support.get("scale.revenue_ttm", {}) or {}).get("support_mode"),
            source_metric="derived:state_vector_v1.size_log_revenue",
            formula="log10(revenue_ttm)",
            component_values={
                "revenue_ttm": revenue,
                "source_metric": sources.get("scale.revenue_ttm", "operating.revenue_ttm"),
                "log_base": 10,
            },
            component_reliability=[reliability.get("scale.revenue_ttm", 0.0)],
            quality_flags=_quality_flags(records.get("scale.revenue_ttm")),
        )
    else:
        _set_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.size_log_revenue",
            value=None,
            support_mode="unsupported",
            source_metric="derived:state_vector_v1.size_log_revenue",
            formula="log10(revenue_ttm)",
            quality_flags=["revenue_ttm_missing_or_non_positive"],
        )

    if revenue is not None and revenue > 0 and ebitda is not None:
        profitability_support_mode = (
            "exact"
            if _all_exactish_support(support, "scale.revenue_ttm", "scale.ebitda_ttm")
            else "proxy_missing_component"
        )
        _set_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.profitability",
            value=ebitda / revenue,
            support_mode=profitability_support_mode,
            source_metric="derived:state_vector_v1.profitability",
            formula="ebitda_ttm / revenue_ttm",
            component_values={
                "revenue_ttm": revenue,
                "ebitda_ttm": ebitda,
                "revenue_source_metric": sources.get("scale.revenue_ttm", "operating.revenue_ttm_provider_direct"),
                "ebitda_source_metric": sources.get("scale.ebitda_ttm", "operating.ebitda_ltm_provider_direct"),
            },
            component_reliability=[
                reliability.get("scale.revenue_ttm", 0.0),
                reliability.get("scale.ebitda_ttm", 0.0),
            ],
            quality_flags=list(
                dict.fromkeys(
                    [
                        *_quality_flags(records.get("scale.revenue_ttm")),
                        *_quality_flags(records.get("scale.ebitda_ttm")),
                    ]
                )
            ),
        )
    else:
        _copy_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.profitability",
            source_key="operating.ebitda_margin_ttm",
            canonical=canonical,
            records=records,
            support=support,
            reliability=reliability,
            sources=sources,
            extra_quality_flags=["profitability_uses_margin_fallback"],
        )

    revenue_lag = _safe_float(canonical.get("operating.revenue_ttm_lag_1y"))
    if revenue is not None and revenue > 0 and revenue_lag is not None and revenue_lag > 0:
        growth_support_mode = (
            "exact"
            if _all_exactish_support(support, "scale.revenue_ttm", "operating.revenue_ttm_lag_1y")
            else "proxy_missing_component"
        )
        _set_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.growth",
            value=(revenue / revenue_lag) - 1.0,
            support_mode=growth_support_mode,
            source_metric="derived:state_vector_v1.growth",
            formula="(revenue_ttm / revenue_ttm_lag_1y) - 1",
            component_values={
                "revenue_ttm": revenue,
                "revenue_ttm_lag_1y": revenue_lag,
                "revenue_source_metric": sources.get("scale.revenue_ttm", "operating.revenue_ttm_provider_direct"),
                "revenue_lag_source_metric": sources.get("operating.revenue_ttm_lag_1y", "operating.revenue_ttm_lag_1y"),
            },
            component_reliability=[
                reliability.get("scale.revenue_ttm", 0.0),
                reliability.get("operating.revenue_ttm_lag_1y", 0.0),
            ],
            quality_flags=list(
                dict.fromkeys(
                    [
                        *_quality_flags(records.get("scale.revenue_ttm")),
                        *_quality_flags(records.get("operating.revenue_ttm_lag_1y")),
                    ]
                )
            ),
        )
    else:
        growth_source_key = "operating.revenue_yoy_last_q"
        growth_flags = ["growth_uses_last_q_proxy"]
        _copy_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.growth",
            source_key=growth_source_key,
            canonical=canonical,
            records=records,
            support=support,
            reliability=reliability,
            sources=sources,
            extra_quality_flags=growth_flags if canonical.get(growth_source_key) is not None else (),
        )

    market_cap = _safe_float(canonical.get("scale.market_cap"))
    free_cash_flow = _safe_float(canonical.get("cash_flow.free_cash_flow_ttm"))
    if free_cash_flow is not None and market_cap is not None and market_cap > 0:
        cash_generation_support_mode = (
            "exact"
            if _all_exactish_support(support, "cash_flow.free_cash_flow_ttm", "scale.market_cap")
            else "proxy_missing_component"
        )
        _set_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.cash_generation",
            value=free_cash_flow / market_cap,
            support_mode=cash_generation_support_mode,
            source_metric="derived:state_vector_v1.cash_generation",
            formula="free_cash_flow_ttm / equity_market_cap",
            component_values={
                "free_cash_flow_ttm": free_cash_flow,
                "equity_market_cap": market_cap,
                "free_cash_flow_source_metric": sources.get("cash_flow.free_cash_flow_ttm", "cash_flow.free_cash_flow_ttm"),
                "equity_market_cap_source_metric": sources.get("scale.market_cap", "market.market_cap_provider_direct"),
            },
            component_reliability=[
                reliability.get("cash_flow.free_cash_flow_ttm", 0.0),
                reliability.get("scale.market_cap", 0.0),
            ],
            quality_flags=list(
                dict.fromkeys(
                    [
                        *_quality_flags(records.get("cash_flow.free_cash_flow_ttm")),
                        *_quality_flags(records.get("scale.market_cap")),
                    ]
                )
            ),
        )
    else:
        _copy_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.cash_generation",
            source_key="market.fcf_yield",
            canonical=canonical,
            records=records,
            support=support,
            reliability=reliability,
            sources=sources,
            extra_quality_flags=["cash_generation_uses_market_fcf_yield_fallback"] if canonical.get("market.fcf_yield") is not None else (),
        )

    for target_key, source_key in (
        ("state_vector_v1.rates_level", "macro.fed_funds_effective"),
        ("state_vector_v1.credit_spread", "macro.hy_oas"),
    ):
        _copy_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key=target_key,
            source_key=source_key,
            canonical=canonical,
            records=records,
            support=support,
            reliability=reliability,
            sources=sources,
        )

    gross_debt = _safe_float(canonical.get("capital.total_debt"))
    net_debt = _safe_float(canonical.get("capital.net_debt"))
    lease_liabilities = _safe_float(canonical.get("capital.lease_liabilities"))
    retirement_liabilities = _safe_float(canonical.get("capital.combined_retirement_liability"))

    if gross_debt is not None and ebitda not in (None, 0) and ebitda > 0:
        gross_flags = []
        lease_component = lease_liabilities if lease_liabilities is not None else 0.0
        retirement_component = retirement_liabilities if retirement_liabilities is not None else 0.0
        if lease_liabilities is None:
            gross_flags.append("lease_liabilities_missing_assumed_zero")
        if retirement_liabilities is None:
            gross_flags.append("retirement_liabilities_missing_assumed_zero")
        gross_support_mode = (
            "exact"
            if _all_exactish_support(support, "capital.total_debt", "scale.ebitda_ttm", "capital.lease_liabilities", "capital.combined_retirement_liability")
            and not gross_flags
            else "proxy_missing_component"
        )
        _set_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.gross_obligation_burden",
            value=(gross_debt + lease_component + retirement_component) / ebitda,
            support_mode=gross_support_mode,
            source_metric="derived:state_vector_v1.gross_obligation_burden",
            formula="(gross_debt + lease_liabilities + retirement_liabilities) / ebitda_ttm",
            component_values={
                "gross_debt": gross_debt,
                "lease_liabilities": lease_component,
                "retirement_liabilities": retirement_component,
                "ebitda_ttm": ebitda,
                "gross_debt_source_metric": sources.get("capital.total_debt", "capital_structure.total_debt_provider_direct"),
                "lease_liabilities_source_metric": sources.get("capital.lease_liabilities", "capital_structure.lease_liabilities_sec_exact"),
                "retirement_liabilities_source_metric": sources.get("capital.combined_retirement_liability", "capital_structure.combined_retirement_liability"),
                "ebitda_source_metric": sources.get("scale.ebitda_ttm", "operating.ebitda_ltm_provider_direct"),
            },
            component_reliability=[
                reliability.get("capital.total_debt", 0.0),
                reliability.get("capital.lease_liabilities", 0.0),
                reliability.get("capital.combined_retirement_liability", 0.0),
                reliability.get("scale.ebitda_ttm", 0.0),
            ],
            quality_flags=gross_flags,
        )
    else:
        _copy_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.gross_obligation_burden",
            source_key="capital.gross_leverage_including_retirement",
            canonical=canonical,
            records=records,
            support=support,
            reliability=reliability,
            sources=sources,
            extra_quality_flags=["gross_obligation_uses_leverage_fallback"],
        )

    if net_debt is not None and ebitda not in (None, 0) and ebitda > 0:
        net_flags = []
        retirement_component = retirement_liabilities if retirement_liabilities is not None else 0.0
        if retirement_liabilities is None:
            net_flags.append("retirement_liabilities_missing_assumed_zero")
        net_support_mode = (
            "exact"
            if _all_exactish_support(support, "capital.net_debt", "scale.ebitda_ttm", "capital.combined_retirement_liability")
            and not net_flags
            else "proxy_missing_component"
        )
        _set_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.net_obligation_burden",
            value=(net_debt + retirement_component) / ebitda,
            support_mode=net_support_mode,
            source_metric="derived:state_vector_v1.net_obligation_burden",
            formula="(net_debt + retirement_liabilities) / ebitda_ttm",
            component_values={
                "net_debt": net_debt,
                "retirement_liabilities": retirement_component,
                "ebitda_ttm": ebitda,
                "net_debt_source_metric": sources.get("capital.net_debt", "capital_structure.net_debt_normalized"),
                "retirement_liabilities_source_metric": sources.get("capital.combined_retirement_liability", "capital_structure.combined_retirement_liability"),
                "ebitda_source_metric": sources.get("scale.ebitda_ttm", "operating.ebitda_ltm_provider_direct"),
            },
            component_reliability=[
                reliability.get("capital.net_debt", 0.0),
                reliability.get("capital.combined_retirement_liability", 0.0),
                reliability.get("scale.ebitda_ttm", 0.0),
            ],
            quality_flags=net_flags,
        )
    else:
        _copy_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.net_obligation_burden",
            source_key="capital.net_leverage_including_retirement",
            canonical=canonical,
            records=records,
            support=support,
            reliability=reliability,
            sources=sources,
            extra_quality_flags=["net_obligation_uses_leverage_fallback"],
        )

    liquidity_candidates = (
        ("liquidity.available_liquidity_normalized", "preferred_available_liquidity"),
        ("liquidity.available_for_actions", "available_for_actions_fallback"),
        ("liquidity.cash", "cash_only_fallback"),
    )
    debt_candidates = (
        ("capital.debt_due_next_24m", "debt_due_next_24m"),
        ("capital.debt_due_0_12m", "debt_due_0_12m_fallback"),
        ("capital.current_debt", "current_debt_fallback"),
    )
    debt_key = next((key for key, _ in debt_candidates if (_safe_float(canonical.get(key)) or 0.0) > 0), None)
    liquidity_key = next((key for key, _ in liquidity_candidates if _safe_float(canonical.get(key)) is not None), None)
    debt_value = _safe_float(canonical.get(debt_key)) if debt_key else None
    debt_support_key = debt_key
    debt_source_metric = sources.get(debt_key or "", debt_key)
    debt_reliability_value = reliability.get(debt_key, 0.0) if debt_key else 0.0
    raw_debt_record = None
    liquidity_flags: list[str] = []
    canonical_debt_flags = _quality_flags(records.get(debt_key)) if debt_key else []
    if "current_debt_fallback" in canonical_debt_flags:
        liquidity_flags.append("current_debt_fallback")
    if "debt_due_0_12m_fallback" in canonical_debt_flags:
        liquidity_flags.append("debt_due_0_12m_fallback")
    if debt_value in (None, 0):
        raw_debt_record, raw_debt_source = _resolve_first_record(
            features,
            (
                "capital_structure.debt_due_next_24m",
                "capital_structure.debt_due_0_12m",
                "capital_structure.current_debt_statement_direct",
                "capital_structure.current_debt_provider_direct",
                "capital_structure.current_debt",
            ),
        )
        raw_debt_value = _safe_float(raw_debt_record)
        if raw_debt_value not in (None, 0) and raw_debt_value > 0:
            debt_value = raw_debt_value
            debt_source_metric = raw_debt_source or "capital_structure.current_debt"
            debt_reliability_value = _reliability_score(raw_debt_record, source_metric=debt_source_metric)
            if raw_debt_source == "capital_structure.debt_due_next_24m":
                debt_support_key = None
            elif raw_debt_source == "capital_structure.debt_due_0_12m":
                debt_support_key = None
                liquidity_flags.append("debt_due_0_12m_fallback")
            else:
                debt_support_key = None
                liquidity_flags.append("current_debt_fallback")
    else:
        raw_debt_source = None
    cash_value = _safe_float(canonical.get("liquidity.cash"))
    marketable_value = _safe_float(canonical.get("liquidity.marketable_securities"))
    revolver_value = _safe_float(canonical.get("liquidity.revolver_undrawn"))
    component_liquidity = None
    if cash_value is not None:
        component_liquidity = cash_value + (marketable_value or 0.0) + (revolver_value or 0.0)
        if marketable_value is None:
            liquidity_flags.append("marketable_securities_missing_assumed_zero")
        if revolver_value is None:
            liquidity_flags.append("revolver_undrawn_missing_assumed_zero")
    liquidity_value = component_liquidity
    liquidity_source_metric = "derived:available_liquidity_components"
    if liquidity_value is None and liquidity_key:
        liquidity_value = _safe_float(canonical.get(liquidity_key))
        liquidity_source_metric = sources.get(liquidity_key or "", liquidity_key or "liquidity.available_liquidity_normalized")
        if liquidity_key != "liquidity.available_liquidity_normalized":
            liquidity_flags.append(liquidity_candidates[[k for k, _ in liquidity_candidates].index(liquidity_key)][1])
    if debt_key and debt_key != "capital.debt_due_next_24m":
        liquidity_flags.append(debt_candidates[[k for k, _ in debt_candidates].index(debt_key)][1])
    if liquidity_value is not None and debt_value not in (None, 0) and debt_value > 0:
        liquidity_ratio_value = liquidity_value / debt_value
        if "current_debt_fallback" in liquidity_flags and math.isfinite(liquidity_ratio_value):
            # Current debt is only a weak maturity proxy; cap the ratio so tiny current maturities
            # do not masquerade as an infinitely safer liquidity profile.
            liquidity_ratio_value = min(liquidity_ratio_value, 25.0)
            if liquidity_ratio_value < (liquidity_value / debt_value):
                liquidity_flags.append("current_debt_proxy_ratio_capped")
        component_keys = [
            "liquidity.cash" if cash_value is not None else None,
            "liquidity.marketable_securities" if marketable_value is not None else None,
            "liquidity.revolver_undrawn" if revolver_value is not None else None,
            debt_support_key,
        ]
        liquidity_support_mode = (
            "exact"
            if not liquidity_flags and (
                _all_exactish_support(support, *component_keys)
                or (raw_debt_record is not None and _is_exactish_support_mode(_support_mode(raw_debt_record)))
            )
            else "proxy_missing_component"
        )
        _set_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.liquidity_flexibility",
            value=liquidity_ratio_value,
            support_mode=liquidity_support_mode,
            source_metric="derived:state_vector_v1.liquidity_flexibility",
            formula="liquidity / near_term_debt",
            component_values={
                "liquidity": liquidity_value,
                "near_term_debt": debt_value,
                "ratio_cap": 25.0 if "current_debt_fallback" in liquidity_flags else None,
                "cash_and_short_term_investments": cash_value,
                "marketable_securities": marketable_value,
                "undrawn_revolver": revolver_value,
                "liquidity_source_metric": liquidity_source_metric,
                "near_term_debt_source_metric": debt_source_metric,
            },
            component_reliability=[
                reliability.get("liquidity.cash", 0.0) if cash_value is not None else 0.0,
                reliability.get("liquidity.marketable_securities", 0.0) if marketable_value is not None else 0.0,
                reliability.get("liquidity.revolver_undrawn", 0.0) if revolver_value is not None else 0.0,
                debt_reliability_value,
            ],
            quality_flags=liquidity_flags,
            fallback_used="available_liquidity_over_near_term_debt",
        )
    else:
        _set_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.liquidity_flexibility",
            value=None,
            support_mode="unsupported",
            source_metric="derived:state_vector_v1.liquidity_flexibility",
            formula="liquidity / near_term_debt",
            quality_flags=["liquidity_or_near_term_debt_missing"],
        )

    interest_expense = _safe_float(canonical.get("capital.interest_expense"))
    if ebitda is not None and interest_expense not in (None, 0) and interest_expense > 0:
        interest_support_mode = (
            "exact"
            if _all_exactish_support(support, "scale.ebitda_ttm", "capital.interest_expense")
            else "proxy_missing_component"
        )
        _set_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.interest_coverage",
            value=ebitda / interest_expense,
            support_mode=interest_support_mode,
            source_metric="derived:state_vector_v1.interest_coverage",
            formula="ebitda_ttm / interest_expense",
            component_values={
                "ebitda_ttm": ebitda,
                "interest_expense": interest_expense,
                "ebitda_source_metric": sources.get("scale.ebitda_ttm", "operating.ebitda_ltm_provider_direct"),
                "interest_expense_source_metric": sources.get("capital.interest_expense", "capital_structure.interest_expense_statement_direct"),
            },
            component_reliability=[
                reliability.get("scale.ebitda_ttm", 0.0),
                reliability.get("capital.interest_expense", 0.0),
            ],
        )
    else:
        _copy_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.interest_coverage",
            source_key="capital.interest_coverage",
            canonical=canonical,
            records=records,
            support=support,
            reliability=reliability,
            sources=sources,
            extra_quality_flags=["interest_coverage_external_fallback"],
        )

    cash_and_sti = _safe_float(canonical.get("liquidity.cash"))
    valuation_flags = []
    if lease_liabilities is None:
        lease_liabilities = 0.0
        valuation_flags.append("lease_liabilities_missing_assumed_zero")
    if market_cap is not None and gross_debt is not None and cash_and_sti is not None and ebitda not in (None, 0) and ebitda > 0:
        valuation_support_mode = (
            "exact"
            if not valuation_flags and _all_exactish_support(
                support,
                "scale.market_cap",
                "capital.total_debt",
                "capital.lease_liabilities",
                "liquidity.cash",
                "scale.ebitda_ttm",
            )
            else "proxy_missing_component"
        )
        _set_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.valuation_multiple",
            value=(market_cap + gross_debt + lease_liabilities - cash_and_sti) / ebitda,
            support_mode=valuation_support_mode,
            source_metric="derived:state_vector_v1.valuation_multiple",
            formula="(equity_market_cap + gross_debt + lease_liabilities - cash_and_short_term_investments) / ebitda_ttm",
            component_values={
                "equity_market_cap": market_cap,
                "gross_debt": gross_debt,
                "lease_liabilities": lease_liabilities,
                "cash_and_short_term_investments": cash_and_sti,
                "ebitda_ttm": ebitda,
                "equity_market_cap_source_metric": sources.get("scale.market_cap", "market.market_cap_provider_direct"),
                "gross_debt_source_metric": sources.get("capital.total_debt", "capital_structure.total_debt_provider_direct"),
                "lease_liabilities_source_metric": sources.get("capital.lease_liabilities", "capital_structure.lease_liabilities_sec_exact"),
                "cash_source_metric": sources.get("liquidity.cash", "liquidity.cash_and_short_term_investments_provider_direct"),
                "ebitda_source_metric": sources.get("scale.ebitda_ttm", "operating.ebitda_ltm_provider_direct"),
            },
            component_reliability=[
                reliability.get("scale.market_cap", 0.0),
                reliability.get("capital.total_debt", 0.0),
                reliability.get("capital.lease_liabilities", 0.0),
                reliability.get("liquidity.cash", 0.0),
                reliability.get("scale.ebitda_ttm", 0.0),
            ],
            quality_flags=valuation_flags,
        )
    else:
        _copy_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.valuation_multiple",
            source_key="market.ev_ebitda",
            canonical=canonical,
            records=records,
            support=support,
            reliability=reliability,
            sources=sources,
            extra_quality_flags=["valuation_multiple_uses_market_ev_ebitda_fallback"],
        )

    vol_90d = _safe_float(canonical.get("market.volatility_90d"))
    drawdown_90d = _safe_float(canonical.get("market.drawdown_90d"))
    stress_components = []
    stress_weights = []
    stress_flags = []
    if vol_90d is not None:
        stress_components.append(vol_90d)
        stress_weights.append(0.6)
    else:
        stress_flags.append("volatility_90d_missing")
    if drawdown_90d is not None:
        stress_components.append(abs(min(drawdown_90d, 0.0)))
        stress_weights.append(0.4)
    else:
        stress_flags.append("drawdown_90d_missing")
    if stress_components:
        weight_total = sum(stress_weights) or 1.0
        stress_value = sum(component * weight for component, weight in zip(stress_components, stress_weights)) / weight_total
        stress_support_mode = "exact" if all(
            (support.get(key, {}) or {}).get("support_mode") in _EXACTISH_SUPPORT_MODES
            for key in ("market.volatility_90d", "market.drawdown_90d")
            if _safe_float(canonical.get(key)) is not None
        ) and not stress_flags else "proxy_missing_component"
        _set_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.market_stress",
            value=stress_value,
            support_mode=stress_support_mode,
            source_metric="derived:state_vector_v1.market_stress",
            formula="weighted_average(volatility_90d, abs(min(drawdown_90d, 0)))",
            component_values={
                "volatility_90d": vol_90d,
                "drawdown_90d_abs": None if drawdown_90d is None else abs(min(drawdown_90d, 0.0)),
                "weights": {"volatility_90d": 0.6, "drawdown_90d_abs": 0.4},
            },
            component_reliability=[
                reliability.get("market.volatility_90d", 0.0),
                reliability.get("market.drawdown_90d", 0.0),
            ],
            quality_flags=stress_flags,
        )
    elif _safe_float(canonical.get("market.vix")) is not None:
        vix = _safe_float(canonical.get("market.vix"))
        vix_support_mode = (support.get("market.vix", {}) or {}).get("support_mode")
        _set_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.market_stress",
            value=max(0.0, min(1.0, float(vix) / 80.0)),
            support_mode="exact" if _is_exactish_support_mode(vix_support_mode) else "proxy_missing_component",
            source_metric="derived:state_vector_v1.market_stress",
            formula="clip(market.vix / 80.0, 0, 1)",
            component_values={
                "market.vix": vix,
                "market.vix_denominator": 80.0,
            },
            component_reliability=[reliability.get("market.vix", 0.0)],
            quality_flags=["volatility_90d_missing", "drawdown_90d_missing", "market_stress_vix_fallback"],
            fallback_used="market.vix",
        )
    else:
        _set_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.market_stress",
            value=None,
            support_mode="unsupported",
            source_metric="derived:state_vector_v1.market_stress",
            formula="weighted_average(volatility_90d, abs(min(drawdown_90d, 0)))",
            quality_flags=["market_stress_inputs_missing"],
        )

    credit_window = _safe_float(canonical.get("market.credit_window_proxy"))
    equity_window = _safe_float(canonical.get("market.equity_window_proxy"))
    credit_spread_level = _safe_float(canonical.get("market.credit_spread_level"))
    access_components = []
    access_weights = []
    access_flags = []
    if credit_window is not None:
        access_components.append(credit_window)
        access_weights.append(0.4)
    else:
        access_flags.append("credit_window_proxy_missing")
    if equity_window is not None:
        access_components.append(equity_window)
        access_weights.append(0.4)
    else:
        access_flags.append("equity_window_proxy_missing")
    if credit_spread_level is not None:
        spread_access = max(0.0, min(1.0, 1.0 - (credit_spread_level / 0.08)))
        access_components.append(spread_access)
        access_weights.append(0.2)
    else:
        access_flags.append("credit_spread_level_missing")
        spread_access = None
    if access_components:
        weight_total = sum(access_weights) or 1.0
        access_value = sum(component * weight for component, weight in zip(access_components, access_weights)) / weight_total
        access_support_mode = "exact" if all(
            (support.get(key, {}) or {}).get("support_mode") in _EXACTISH_SUPPORT_MODES
            for key in ("market.credit_window_proxy", "market.equity_window_proxy", "market.credit_spread_level")
            if _safe_float(canonical.get(key)) is not None
        ) and not access_flags else "proxy_missing_component"
        _set_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.market_access",
            value=access_value,
            support_mode=access_support_mode,
            source_metric="derived:state_vector_v1.market_access",
            formula="weighted_average(credit_window_proxy, equity_window_proxy, normalized_credit_spread_level)",
            component_values={
                "credit_window_proxy": credit_window,
                "equity_window_proxy": equity_window,
                "credit_spread_level": credit_spread_level,
                "normalized_credit_spread_access": spread_access,
                "weights": {
                    "credit_window_proxy": 0.4,
                    "equity_window_proxy": 0.4,
                    "normalized_credit_spread_access": 0.2,
                },
                "credit_spread_cap": 0.08,
            },
            component_reliability=[
                reliability.get("market.credit_window_proxy", 0.0),
                reliability.get("market.equity_window_proxy", 0.0),
                reliability.get("market.credit_spread_level", 0.0),
            ],
            quality_flags=access_flags,
        )
    else:
        _set_state_metric(
            state_values,
            state_records,
            state_support,
            state_reliability,
            state_sources,
            key="state_vector_v1.market_access",
            value=None,
            support_mode="unsupported",
            source_metric="derived:state_vector_v1.market_access",
            formula="weighted_average(credit_window_proxy, equity_window_proxy, normalized_credit_spread_level)",
            quality_flags=["market_access_inputs_missing"],
        )

    sector = canonical.get("taxonomy.sector")
    if sector is None:
        sector = _feature_value((snapshot or {}).get("sector"))
    subsector = canonical.get("taxonomy.subsector")
    if subsector is None:
        subsector = _feature_value((snapshot or {}).get("subsector"))
    retirement_regime = canonical.get("capital.retirement_obligation_regime")
    proxy_features = sorted(
        key
        for key, meta in state_support.items()
        if (meta.get("support_mode") or "") not in _EXACTISH_SUPPORT_MODES and meta.get("support_mode") != "unsupported"
    )
    missing_features = sorted(
        key for key, meta in state_support.items() if meta.get("support_mode") == "unsupported"
    )
    data_quality_flags = sorted(
        {
            flag
            for meta in state_support.values()
            for flag in (meta.get("quality_flags") or [])
            if flag
        }
    )
    return {
        "values": state_values,
        "records": state_records,
        "support": state_support,
        "reliability": state_reliability,
        "sources": state_sources,
        "meta": {
            "version": "state_vector_v1",
            "sector": sector,
            "subsector": subsector,
            "retirement_regime": retirement_regime,
            "data_quality_flags": data_quality_flags,
            "proxy_features": proxy_features,
            "missing_features": missing_features,
            "feature_order": list(_STATE_VECTOR_V1_FEATURES),
        },
    }


def _build_canonical_block(
    features: Dict[str, Any],
    *,
    action_type: Optional[str] = None,
    action_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, float], Dict[str, str]]:
    canonical: Dict[str, Any] = {}
    records: Dict[str, Dict[str, Any]] = {}
    support: Dict[str, Any] = {}
    reliability: Dict[str, float] = {}
    sources: Dict[str, str] = {}
    for canonical_key, source_keys in _CANONICAL_SPECS.items():
        record, source_key = _resolve_first_record(
            features,
            source_keys,
            action_type=action_type,
            action_id=action_id,
        )
        if record is None or source_key is None:
            continue
        target_record = _copy_record_for_target(record, target_key=canonical_key, source_key=source_key)
        canonical[canonical_key] = _feature_value(target_record)
        records[canonical_key] = target_record or {"value": _feature_value(record)}
        support[canonical_key] = _canonical_meta(record, source_metric=source_key)
        reliability[canonical_key] = _reliability_score(record, source_metric=source_key)
        sources[canonical_key] = source_key

    market_cap = canonical.get("scale.market_cap")
    if market_cap is not None:
        try:
            canonical["scale.log_market_cap"] = math.log10(max(float(market_cap), 1.0))
        except Exception:
            pass
        bucket = "micro"
        try:
            cap = float(market_cap)
            if cap >= 200_000_000_000:
                bucket = "mega"
            elif cap >= 10_000_000_000:
                bucket = "large"
            elif cap >= 2_000_000_000:
                bucket = "mid"
            elif cap >= 300_000_000:
                bucket = "small"
        except Exception:
            bucket = "micro"
        canonical["scale.market_cap_bucket"] = bucket

    retirement_regime = canonical.get("capital.retirement_obligation_regime")
    if retirement_regime is not None:
        regime_value = str(retirement_regime).strip()
        regime_record = records.get("capital.retirement_obligation_regime")
        regime_source = sources.get("capital.retirement_obligation_regime", "capital_structure.retirement_obligation_regime")
        for observed_regime, canonical_key in _RETIREMENT_REGIME_FLAGS.items():
            flag_value = 1.0 if regime_value == observed_regime else 0.0
            canonical[canonical_key] = flag_value
            if regime_record is not None:
                target_record = _copy_record_for_target(
                    regime_record,
                    target_key=canonical_key,
                    source_key=regime_source,
                )
                if isinstance(target_record, dict):
                    target_record["value"] = flag_value
                    breakdown = dict(target_record.get("component_breakdown") or {})
                    breakdown["model_feature_bundle_regime_encoding"] = {
                        "source_metric": "capital.retirement_obligation_regime",
                        "observed_regime": regime_value,
                        "target_regime": observed_regime,
                        "encoding": "one_hot",
                    }
                    target_record["component_breakdown"] = breakdown
                records[canonical_key] = target_record or {"value": flag_value}
                support[canonical_key] = _canonical_meta(regime_record, source_metric=regime_source)
                reliability[canonical_key] = _reliability_score(regime_record, source_metric=regime_source)
            else:
                records[canonical_key] = {"value": flag_value}
                support[canonical_key] = {
                    "source_metric": "capital_structure.retirement_obligation_regime",
                    "support_mode": None,
                    "applicability_status": None,
                    "quality_flags": [],
                    "is_proxy": False,
                    "is_legacy": True,
                }
                reliability[canonical_key] = 0.0
            sources[canonical_key] = regime_source
    return canonical, records, support, reliability, sources


def _build_view(
    features: Dict[str, Any],
    *,
    action_type: Optional[str] = None,
    action_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    view = copy.deepcopy(features)
    overrides: Dict[str, str] = {}
    for key in _LEGACY_COMPAT_KEYS:
        record, source_key = _resolve_first_record(
            features,
            [key],
            action_type=action_type,
            action_id=action_id,
        )
        if record is None:
            continue
        view[key] = _copy_record_for_target(record, target_key=key, source_key=str(source_key or key))
        overrides[key] = str(source_key or key)
    return view, overrides


def build_model_feature_bundle(
    snapshot: Dict[str, Any],
    *,
    action_id: Optional[str] = None,
    action_type: Optional[str] = None,
    regime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw_features = dict((snapshot or {}).get("features", {}) or {})
    resolved_regime = dict(regime or ((snapshot or {}).get("regime", {}) or {}))
    resolved_action_type = _resolved_action_family(action_type=action_type, action_id=action_id)
    canonical, records, support, reliability, sources = _build_canonical_block(
        raw_features,
        action_type=resolved_action_type,
        action_id=action_id,
    )
    state_vector_v1 = _build_state_vector_v1(
        snapshot,
        canonical=canonical,
        records=records,
        support=support,
        reliability=reliability,
        sources=sources,
    )
    for key, value in dict(state_vector_v1.get("values", {}) or {}).items():
        if value is None:
            continue
        canonical[key] = value
    records.update(dict(state_vector_v1.get("records", {}) or {}))
    support.update(dict(state_vector_v1.get("support", {}) or {}))
    reliability.update(dict(state_vector_v1.get("reliability", {}) or {}))
    sources.update(dict(state_vector_v1.get("sources", {}) or {}))
    views: Dict[str, Dict[str, Any]] = {}
    overrides_by_view: Dict[str, Dict[str, str]] = {}
    for view_name in _VIEW_NAMES:
        views[view_name], overrides_by_view[view_name] = _build_view(
            raw_features,
            action_type=resolved_action_type,
            action_id=action_id,
        )
        for state_key in _STATE_VECTOR_V1_FEATURES:
            record = records.get(state_key)
            if not isinstance(record, dict):
                continue
            views[view_name][state_key] = _copy_record_for_target(
                record,
                target_key=state_key,
                source_key=str(sources.get(state_key) or state_key),
            )

    diagnostics = {
        "action_family": resolved_action_type,
        "action_id": str(action_id or "") or None,
        "canonical_count": len(canonical),
        "canonical_sources": sources,
        "view_override_counts": {name: len(rows) for name, rows in overrides_by_view.items()},
        "view_overrides": overrides_by_view,
    }
    return {
        "meta": {
            "company_id": str((snapshot or {}).get("company_id", "") or ""),
            "as_of_time": str((snapshot or {}).get("as_of_time", "") or ""),
            "snapshot_id": str((snapshot or {}).get("snapshot_id", "") or ""),
            "action_family": resolved_action_type,
            "action_id": str(action_id or "") or None,
            "regime": resolved_regime,
        },
        "raw_features": raw_features,
        "canonical": canonical,
        "records": records,
        "support": support,
        "reliability": reliability,
        "state_vector_v1": state_vector_v1,
        "views": views,
        "diagnostics": diagnostics,
    }


def attach_model_feature_bundle(
    snapshot: Dict[str, Any],
    *,
    action_id: Optional[str] = None,
    action_type: Optional[str] = None,
    regime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out = dict(snapshot or {})
    out[_BUNDLE_KEY] = build_model_feature_bundle(
        snapshot,
        action_id=action_id,
        action_type=action_type,
        regime=regime,
    )
    return out


def get_model_feature_bundle(
    snapshot: Dict[str, Any],
    *,
    action_id: Optional[str] = None,
    action_type: Optional[str] = None,
    regime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if (
        action_id is None
        and action_type is None
        and isinstance(snapshot, dict)
        and isinstance(snapshot.get(_BUNDLE_KEY), dict)
    ):
        return dict(snapshot.get(_BUNDLE_KEY) or {})
    return build_model_feature_bundle(
        snapshot,
        action_id=action_id,
        action_type=action_type,
        regime=regime,
    )


def feature_view_from_snapshot(
    snapshot: Dict[str, Any],
    *,
    view_name: str,
    action_id: Optional[str] = None,
    action_type: Optional[str] = None,
    regime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    bundle = get_model_feature_bundle(
        snapshot,
        action_id=action_id,
        action_type=action_type,
        regime=regime,
    )
    views = dict(bundle.get("views", {}) or {})
    view = views.get(view_name)
    if isinstance(view, dict):
        return view
    return dict((snapshot or {}).get("features", {}) or {})


def candidate_generation_view(
    bundle: Dict[str, Any],
) -> Dict[str, Any]:
    return dict((bundle.get("views", {}) or {}).get("candidate_generation", {}) or {})


def mechanism_view(
    bundle: Dict[str, Any],
) -> Dict[str, Any]:
    return dict((bundle.get("views", {}) or {}).get("mechanism", {}) or {})


def causal_view(
    bundle: Dict[str, Any],
) -> Dict[str, Any]:
    return dict((bundle.get("views", {}) or {}).get("causal", {}) or {})


def precedent_view(
    bundle: Dict[str, Any],
) -> Dict[str, Any]:
    return dict((bundle.get("views", {}) or {}).get("precedent", {}) or {})


def dossier_view(
    bundle: Dict[str, Any],
) -> Dict[str, Any]:
    return dict((bundle.get("views", {}) or {}).get("dossier", {}) or {})


def get_bundle_record(bundle: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    record = dict((bundle.get("records", {}) or {}).get(key, {}) or {})
    return record or None


def get_bundle_value(bundle: Dict[str, Any], key: str, default: Any = None) -> Any:
    if key in dict(bundle.get("canonical", {}) or {}):
        value = bundle["canonical"].get(key)
        return default if value is None else value
    record = get_bundle_record(bundle, key)
    if record is not None:
        value = _feature_value(record)
        return default if value is None else value
    return default
