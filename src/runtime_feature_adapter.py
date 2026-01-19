from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, Tuple


_ADAPTER_ENV_KEY = "AXIOM_ENABLE_RUNTIME_FEATURE_ADAPTER"
_RULES_ENV_KEY = "AXIOM_RUNTIME_FEATURE_ADAPTER_RULES"
_PROFILE_ENV_KEY = "AXIOM_RUNTIME_FEATURE_ADAPTER_PROFILE"
_MISSING = object()
_EXACTISH_SUPPORT_MODES = {
    "exact",
    "exact_not_applicable",
    "exact_structural_zero",
}
_DEFAULT_PROFILE = "leverage_only"
_PROFILE_RULES: Dict[str, Optional[set[str]]] = {
    "leverage_only": {
        "normalized_net_leverage",
        "normalized_gross_leverage",
    },
    "debt_liquidity_only": {
        "normalized_net_debt",
        "normalized_available_liquidity",
    },
    "earnings_pe_only": {
        "normalized_operating_earnings_fill",
        "pe_ratio_compatibility_alias",
    },
    "macro_credit_only": {
        "ust_10y_alias",
        "ust_2y_alias",
        "ust_10y_minus_curve_2s10s",
        "sofr_compatibility_fallback",
        "credit_ig_alias",
        "credit_hy_alias",
    },
    "all_non_macro": {
        "normalized_net_debt",
        "normalized_available_liquidity",
        "normalized_net_leverage",
        "normalized_gross_leverage",
        "normalized_operating_earnings_fill",
        "pe_ratio_compatibility_alias",
    },
    "all": None,
    "none": set(),
}
_ACTION_GATED_RULES: Dict[str, set[str]] = {
    "normalized_net_debt": {"capital_structure"},
    "normalized_available_liquidity": {"capital_structure"},
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


def _quality_flags(raw: Any) -> set[str]:
    if not isinstance(raw, dict):
        return set()
    return {
        str(flag).strip().lower()
        for flag in (raw.get("quality_flags") or [])
        if flag is not None
    }


def _alias_source_supported(raw: Any) -> bool:
    if raw is None:
        return False
    if _feature_value(raw) is None:
        return False
    if _support_mode(raw) == "unsupported":
        return False
    if _applicability_status(raw) in {"unsupported", "diagnostic"}:
        return False
    flags = _quality_flags(raw)
    if "unsupported_metric" in flags or "sector_native_metrics_required" in flags:
        return False
    return True


def _support_rank(raw: Any) -> int:
    if not _alias_source_supported(raw):
        return -1
    support_mode = _support_mode(raw)
    if support_mode in _EXACTISH_SUPPORT_MODES:
        return 2
    return 1


def _direct_feature_value(features: Dict[str, Any], key: str, default: Any = None) -> Any:
    if not isinstance(features, dict):
        return default
    if key not in features:
        return default
    return _feature_value(features.get(key))


