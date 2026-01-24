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


def runtime_feature_adapter_enabled() -> bool:
    raw = str(os.environ.get(_ADAPTER_ENV_KEY, "0")).strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _profile_name() -> Optional[str]:
    raw = str(os.environ.get(_PROFILE_ENV_KEY, "")).strip().lower().replace("-", "_")
    if not raw or raw == "default":
        if not runtime_feature_adapter_enabled():
            return None
        return _DEFAULT_PROFILE
    if raw in _PROFILE_RULES:
        return raw
    if not runtime_feature_adapter_enabled():
        return None
    return _DEFAULT_PROFILE


def _allowed_rules() -> Optional[set[str]]:
    raw = str(os.environ.get(_RULES_ENV_KEY, "")).strip()
    if not raw:
        profile = _profile_name()
        if profile is None:
            return None
        profile_rules = _PROFILE_RULES.get(profile)
        if profile_rules is None:
            return None
        return set(profile_rules)
    return {
        token.strip()
        for token in raw.split(",")
        if token is not None and token.strip()
    }


def _resolved_action_family(*, action_family: Optional[str] = None, action_id: Optional[str] = None) -> Optional[str]:
    family = str(action_family or "").strip().lower()
    if family:
        return family
    aid = str(action_id or "").strip().lower()
    if "." in aid:
        return aid.split(".", 1)[0]
    return aid or None


def _rule_enabled(rule: str, *, action_family: Optional[str] = None, action_id: Optional[str] = None) -> bool:
    allowed = _allowed_rules()
    if allowed is not None and rule not in allowed:
        return False
    gated_families = _ACTION_GATED_RULES.get(rule)
    if gated_families is None:
        return True
    return _resolved_action_family(action_family=action_family, action_id=action_id) in gated_families


def _rules_profile_label() -> Optional[str]:
    raw = str(os.environ.get(_RULES_ENV_KEY, "")).strip()
    if raw:
        return "custom"
    return _profile_name()


def _copy_record(
    source: Any,
    *,
    target_key: str,
    source_key: str,
    rule: str,
    value: Any = _MISSING,
    unit: Optional[str] = None,
    formula: Optional[str] = None,
    ignored_legacy: bool = False,
) -> Dict[str, Any]:
    if isinstance(source, dict):
        record = copy.deepcopy(source)
    else:
        record = {"value": source}
    record["name"] = target_key
    if value is not _MISSING:
        record["value"] = value
    if unit is not None:
        record["unit"] = unit
    quality_flags = list(record['quality_flags'] or [])
    for flag in ["runtime_feature_adapter_applied", f"runtime_feature_adapter_rule:{rule}"]:
        if flag not in quality_flags:
            quality_flags.append(flag)
    record["quality_flags"] = quality_flags or None
    component_breakdown = dict(record.get("component_breakdown") or {})
    component_breakdown["runtime_feature_adapter"] = {
        "target_metric": target_key,
        "source_metric": source_key,
        "rule": rule,
        "ignored_legacy": bool(ignored_legacy),
        "formula": formula,
    }
    record["component_breakdown"] = component_breakdown
    return record


def _resolution(
    *,
    target_key: str,
    source_key: str,
    record: Dict[str, Any],
    ignored_legacy: bool,
    rule: str,
    synthetic: bool = False,
) -> Dict[str, Any]:
    return {
        "target_key": target_key,
        "source_key": source_key,
        "record": record,
        "ignored_legacy": bool(ignored_legacy),
        "support_mode": _support_mode(record),
        "synthetic": bool(synthetic),
        "rule": rule,
    }


def _prefer_alias_source(
    features: Dict[str, Any],
    *,
    target_key: str,
    source_keys: List[str],
    rule: str,
    action_family: Optional[str] = None,
    action_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    target_raw = features.get(target_key)
    target_present = target_key in features
    if not _rule_enabled(rule, action_family=action_family, action_id=action_id):
        if target_present:
            return _resolution(
                target_key=target_key,
                source_key=target_key,
                record=target_raw if isinstance(target_raw, dict) else {"value": _feature_value(target_raw)},
                ignored_legacy=False,
                rule="legacy_direct",
            )
        return None
    target_rank = _support_rank(target_raw) if target_present else -1
    for source_key in source_keys:
        source_raw = features.get(source_key)
        if not _alias_source_supported(source_raw):
            continue
        source_rank = _support_rank(source_raw)
        if target_present and source_rank < target_rank:
            continue
        record = _copy_record(
            source_raw,
            target_key=target_key,
            source_key=source_key,
            rule=rule,
            ignored_legacy=bool(target_present),
        )
        return _resolution(
            target_key=target_key,
            source_key=source_key,
            record=record,
            ignored_legacy=bool(target_present),
            rule=rule,
        )
    if target_present:
        return _resolution(
            target_key=target_key,
            source_key=target_key,
            record=target_raw if isinstance(target_raw, dict) else {"value": _feature_value(target_raw)},
            ignored_legacy=False,
            rule="legacy_direct",
        )
    return None


