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


