"""Shared causal-model feature contract.

This module is the single source of truth for the causal model's feature
schema. Both runtime inference and offline training should resolve features
through this contract instead of hard-coding parallel legacy mappings.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple


CONTRACT_VERSION = "causal_feature_contract_v2"
RETIREMENT_REGIME_ONE_HOT_FEATURES = {
    "capital.retirement_regime_pension_exact": "pension_exact",
    "capital.retirement_regime_pension_proxy_split_note": "pension_proxy_split_note",
    "capital.retirement_regime_combined_retirement_only": "combined_retirement_only",
    "capital.retirement_regime_defined_contribution_only": "defined_contribution_only",
    "capital.retirement_regime_not_surfaced": "retirement_not_surfaced",
}
RETIREMENT_REGIME_SOURCE_KEYS = (
    "capital.retirement_obligation_regime",
    "capital_structure.retirement_obligation_regime",
    "base_retirement_obligation_regime",
)

FEATURE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "scale.market_cap": ("scale.market_cap", "base_market_cap"),
    "operating.ebitda_margin_ttm": ("operating.ebitda_margin_ttm", "base_margin"),
    "capital.total_debt": ("capital.total_debt", "base_total_debt"),
    "capital.net_debt": ("capital.net_debt", "base_net_debt"),
    "capital.net_leverage": ("capital.net_leverage", "base_leverage"),
    "capital.net_pension_liability": ("capital.net_pension_liability", "base_net_pension_liability"),
    "capital.combined_retirement_liability": (
        "capital.combined_retirement_liability",
        "base_combined_retirement_liability",
    ),
    "capital.debt_like_obligations_including_retirement": (
        "capital.debt_like_obligations_including_retirement",
        "base_debt_like_obligations_including_retirement",
    ),
    "capital.net_debt_including_retirement": (
        "capital.net_debt_including_retirement",
        "base_net_debt_including_retirement",
    ),
    "capital.gross_leverage_including_retirement": (
        "capital.gross_leverage_including_retirement",
        "base_gross_leverage_including_retirement",
    ),
    "capital.net_leverage_including_retirement": (
        "capital.net_leverage_including_retirement",
        "base_net_leverage_including_retirement",
    ),
    "capital.retirement_regime_pension_exact": (
        "capital.retirement_regime_pension_exact",
        "base_retirement_regime_pension_exact",
    ),
    "capital.retirement_regime_pension_proxy_split_note": (
        "capital.retirement_regime_pension_proxy_split_note",
        "base_retirement_regime_pension_proxy_split_note",
    ),
    "capital.retirement_regime_combined_retirement_only": (
        "capital.retirement_regime_combined_retirement_only",
        "base_retirement_regime_combined_retirement_only",
    ),
    "capital.retirement_regime_defined_contribution_only": (
        "capital.retirement_regime_defined_contribution_only",
        "base_retirement_regime_defined_contribution_only",
    ),
    "capital.retirement_regime_not_surfaced": (
        "capital.retirement_regime_not_surfaced",
        "base_retirement_regime_not_surfaced",
    ),
    "liquidity.available_liquidity": (
        "liquidity.available_liquidity",
        "base_available_liquidity",
        "base_cash",
    ),
    "scale.revenue_ttm": ("scale.revenue_ttm", "base_revenue_ttm"),
    "operating.roic": ("operating.roic", "base_roic"),
    "operating.fcf_conversion": ("operating.fcf_conversion", "base_fcf_margin"),
    "market.pe": ("market.pe", "base_pe"),
    "market.ev_ebitda": ("market.ev_ebitda", "base_ev_ebitda"),
    "macro.ust_10y_yield": ("macro.ust_10y_yield", "macro_rate_10y"),
    "macro.ust_2y_yield": ("macro.ust_2y_yield", "macro_rate_2y"),
    "macro.sofr": ("macro.sofr", "macro_sofr"),
    "macro.fed_funds_effective": ("macro.fed_funds_effective", "macro_fed_funds_effective"),
    "macro.real_gdp_growth_yoy": ("macro.real_gdp_growth_yoy", "macro_real_gdp_growth_yoy"),
    "macro.ig_oas": ("macro.ig_oas", "macro_ig_oas"),
    "macro.hy_oas": ("macro.hy_oas", "macro_hy_oas"),
    "market.vix": ("market.vix", "macro_vix"),
    "action.size_absolute_usd": ("action.size_absolute_usd", "action_size"),
    "action.funding_mix_cash": ("action.funding_mix_cash", "funding_mix_cash"),
    "action.funding_mix_debt": ("action.funding_mix_debt", "funding_mix_debt"),
    "action.funding_mix_equity": ("action.funding_mix_equity", "funding_mix_equity"),
    "regime.credit_tight": ("regime.credit_tight", "regime_credit_tight"),
    "regime.vol_high": ("regime.vol_high", "regime_vol_high"),
}

FEATURE_ORDER = list(FEATURE_ALIASES.keys())
INFERENCE_SOURCE_KEYS: Dict[str, str] = {
    "scale.market_cap": "market.market_cap",
    "operating.ebitda_margin_ttm": "operating.ebitda_margin_ttm",
    "capital.total_debt": "capital_structure.total_debt",
    "capital.net_debt": "capital_structure.net_debt",
    "capital.net_leverage": "capital_structure.net_leverage",
    "capital.net_pension_liability": "capital_structure.net_pension_liability",
    "capital.combined_retirement_liability": "capital_structure.combined_retirement_liability",
    "capital.debt_like_obligations_including_retirement": "capital_structure.debt_like_obligations_including_retirement",
    "capital.net_debt_including_retirement": "capital_structure.net_debt_including_retirement",
    "capital.gross_leverage_including_retirement": "capital_structure.gross_leverage_including_retirement",
    "capital.net_leverage_including_retirement": "capital_structure.net_leverage_including_retirement",
    "liquidity.available_liquidity": "liquidity.available_liquidity_normalized",
    "scale.revenue_ttm": "operating.revenue_ttm",
    "operating.roic": "operating.roic",
    "operating.fcf_conversion": "operating.fcf_conversion",
    "market.pe": "market.pe",
    "market.ev_ebitda": "market.ev_ebitda",
    "macro.ust_10y_yield": "macro.rate_10y",
    "macro.ust_2y_yield": "macro.rate_2y",
    "macro.sofr": "macro.sofr",
    "macro.fed_funds_effective": "macro.fed_funds_effective",
    "macro.real_gdp_growth_yoy": "macro.real_gdp_growth_yoy",
    "macro.ig_oas": "market.ig_oas",
    "macro.hy_oas": "market.hy_oas",
    "market.vix": "market.vix",
}

USD_MILLIONS_FEATURES = {
    "scale.market_cap",
    "capital.total_debt",
    "capital.net_debt",
    "capital.net_pension_liability",
    "capital.combined_retirement_liability",
    "capital.debt_like_obligations_including_retirement",
    "capital.net_debt_including_retirement",
    "liquidity.available_liquidity",
    "scale.revenue_ttm",
    "action.size_absolute_usd",
}
RATE_PERCENT_FEATURES = {
    "macro.ust_10y_yield",
    "macro.ust_2y_yield",
    "macro.sofr",
    "macro.fed_funds_effective",
}
OAS_PERCENT_FEATURES = {
    "macro.ig_oas",
    "macro.hy_oas",
}
SIGNED_LOG1P_FEATURES = {
    "scale.market_cap",
    "capital.total_debt",
    "capital.net_debt",
    "capital.net_pension_liability",
    "capital.combined_retirement_liability",
    "capital.debt_like_obligations_including_retirement",
    "capital.net_debt_including_retirement",
    "liquidity.available_liquidity",
    "scale.revenue_ttm",
    "action.size_absolute_usd",
    "market.pe",
    "market.ev_ebitda",
}

LEGACY_TO_CANONICAL = {
    alias: canonical
    for canonical, aliases in FEATURE_ALIASES.items()
    for alias in aliases[1:]
}


def _feature_value(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw.get("value")
    return raw


def canonicalize_feature_name(name: str) -> str:
    key = str(name or "").strip()
    return LEGACY_TO_CANONICAL.get(key, key)


def feature_aliases(name: str) -> Tuple[str, ...]:
    canonical = canonicalize_feature_name(name)
    return FEATURE_ALIASES.get(canonical, (canonical,))


def primary_legacy_alias(name: str) -> Optional[str]:
    aliases = feature_aliases(name)
    return aliases[1] if len(aliases) > 1 else None


def resolve_mapping_value(mapping: Mapping[str, Any], feature_name: str, default: Any = None) -> Any:
    if not isinstance(mapping, Mapping):
        return default
    for alias in feature_aliases(feature_name):
        if alias not in mapping:
            continue
        value = _feature_value(mapping.get(alias))
        if value is not None:
            return value
    return default


def normalize_feature_value(feature_name: str, value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None

    canonical = canonicalize_feature_name(feature_name)
    if canonical in USD_MILLIONS_FEATURES and abs(out) >= 1e7:
        out = out / 1e6
    if canonical in RATE_PERCENT_FEATURES and abs(out) <= 1.0:
        out = out * 100.0
    if canonical in OAS_PERCENT_FEATURES and abs(out) >= 50.0:
        out = out / 100.0
    if canonical in SIGNED_LOG1P_FEATURES:
        out = math.copysign(math.log1p(abs(out)), out)
    return float(out)


def build_contract_feature_map(
    feature_source: Mapping[str, Any],
    *,
    params: Optional[Mapping[str, Any]] = None,
    regime: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    source = feature_source if isinstance(feature_source, Mapping) else {}
    for feature_name in FEATURE_ORDER:
        out[feature_name] = resolve_mapping_value(source, feature_name, default=None)

    params_map = params if isinstance(params, Mapping) else {}
    regime_map = regime if isinstance(regime, Mapping) else {}

    raw_market_cap = resolve_mapping_value(source, "scale.market_cap", default=out.get("scale.market_cap"))
    size_abs = None
    raw_size_abs = params_map.get("size_absolute_usd")
    if raw_size_abs is not None:
        size_abs = raw_size_abs
    else:
        try:
            size_pct = float(params_map.get("size_pct_market_cap")) if params_map.get("size_pct_market_cap") is not None else None
        except Exception:
            size_pct = None
        try:
            market_cap = float(raw_market_cap) if raw_market_cap is not None else None
        except Exception:
            market_cap = None
        if size_pct is not None and market_cap is not None:
            size_abs = size_pct * market_cap
    if size_abs is not None:
        out["action.size_absolute_usd"] = size_abs

    funding_mix = params_map.get("funding_mix")
    if isinstance(funding_mix, Mapping):
        out["action.funding_mix_cash"] = funding_mix.get("cash", out.get("action.funding_mix_cash"))
        out["action.funding_mix_debt"] = funding_mix.get("debt", out.get("action.funding_mix_debt"))
        out["action.funding_mix_equity"] = funding_mix.get("equity", out.get("action.funding_mix_equity"))

    credit_regime = str(regime_map.get("credit_regime", "")).strip().lower()
    vol_regime = str(regime_map.get("vol_regime", "")).strip().lower()
    if credit_regime:
        out["regime.credit_tight"] = 1.0 if credit_regime == "tight" else 0.0
    if vol_regime:
        out["regime.vol_high"] = 1.0 if vol_regime == "high" else 0.0

    retirement_regime = None
    for key in RETIREMENT_REGIME_SOURCE_KEYS:
        if key not in source:
            continue
        candidate = _feature_value(source.get(key))
        if candidate is None:
            continue
        retirement_regime = str(candidate).strip()
        if retirement_regime:
            break
        retirement_regime = None
    if retirement_regime:
        for feature_name, target_regime in RETIREMENT_REGIME_ONE_HOT_FEATURES.items():
            out[feature_name] = 1.0 if retirement_regime == target_regime else 0.0
    return out
