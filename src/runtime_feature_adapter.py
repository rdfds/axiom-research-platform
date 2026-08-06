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
    quality_flags = list(record.get("quality_flags") or [])
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


def _fill_alias_source(
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
    if target_present and _alias_source_supported(target_raw):
        return _resolution(
            target_key=target_key,
            source_key=target_key,
            record=target_raw if isinstance(target_raw, dict) else {"value": _feature_value(target_raw)},
            ignored_legacy=False,
            rule="legacy_direct",
        )
    for source_key in source_keys:
        source_raw = features.get(source_key)
        if not _alias_source_supported(source_raw):
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


def _macro_rate_2y_resolution(
    features: Dict[str, Any],
    *,
    action_family: Optional[str] = None,
    action_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    direct = _prefer_alias_source(
        features,
        target_key="macro.rate_2y",
        source_keys=["macro.ust_2y_yield"],
        rule="ust_2y_alias",
        action_family=action_family,
        action_id=action_id,
    )
    if direct is not None and direct.get("source_key") != "macro.rate_2y":
        return direct

    target_raw = features.get("macro.rate_2y")
    target_present = "macro.rate_2y" in features
    ust10_raw = features.get("macro.ust_10y_yield")
    curve_raw = features.get("macro.curve_2s10s")
    if _rule_enabled("ust_10y_minus_curve_2s10s", action_family=action_family, action_id=action_id) and _alias_source_supported(ust10_raw) and _alias_source_supported(curve_raw):
        ust10_val = _feature_value(ust10_raw)
        curve_val = _feature_value(curve_raw)
        try:
            value = float(ust10_val) - float(curve_val)
        except Exception:
            value = None
        if value is not None:
            record = _copy_record(
                ust10_raw,
                target_key="macro.rate_2y",
                source_key="macro.ust_10y_yield|macro.curve_2s10s",
                rule="ust_10y_minus_curve_2s10s",
                value=value,
                formula="macro.ust_10y_yield - macro.curve_2s10s",
                ignored_legacy=bool(target_present),
            )
            return _resolution(
                target_key="macro.rate_2y",
                source_key="macro.ust_10y_yield|macro.curve_2s10s",
                record=record,
                ignored_legacy=bool(target_present),
                rule="ust_10y_minus_curve_2s10s",
                synthetic=True,
            )
    if target_present:
        return _resolution(
            target_key="macro.rate_2y",
            source_key="macro.rate_2y",
            record=target_raw if isinstance(target_raw, dict) else {"value": _feature_value(target_raw)},
            ignored_legacy=False,
            rule="legacy_direct",
        )
    return None


def _resolve_feature_record(
    features: Dict[str, Any],
    key: str,
    *,
    action_family: Optional[str] = None,
    action_id: Optional[str] = None,
) -> Any:
    if key == "capital_structure.net_debt":
        resolution = _prefer_alias_source(
            features,
            target_key=key,
            source_keys=["capital_structure.net_debt_normalized"],
            rule="normalized_net_debt",
            action_family=action_family,
            action_id=action_id,
        )
        return resolution.get("record") if resolution is not None else None
    if key == "capital_structure.net_leverage":
        resolution = _prefer_alias_source(
            features,
            target_key=key,
            source_keys=["capital_structure.net_leverage_normalized"],
            rule="normalized_net_leverage",
            action_family=action_family,
            action_id=action_id,
        )
        return resolution.get("record") if resolution is not None else None
    if key == "capital_structure.gross_leverage":
        resolution = _prefer_alias_source(
            features,
            target_key=key,
            source_keys=["capital_structure.gross_leverage_normalized"],
            rule="normalized_gross_leverage",
            action_family=action_family,
            action_id=action_id,
        )
        return resolution.get("record") if resolution is not None else None
    if key == "liquidity.available_for_actions":
        resolution = _prefer_alias_source(
            features,
            target_key=key,
            source_keys=["liquidity.available_liquidity_normalized"],
            rule="normalized_available_liquidity",
            action_family=action_family,
            action_id=action_id,
        )
        return resolution.get("record") if resolution is not None else None
    if key == "operating.ebitda_ttm":
        resolution = _fill_alias_source(
            features,
            target_key=key,
            source_keys=["operating.operating_earnings_normalized"],
            rule="normalized_operating_earnings_fill",
            action_family=action_family,
            action_id=action_id,
        )
        return resolution.get("record") if resolution is not None else None
    if key == "macro.rate_10y":  # gitleaks:allow — schema field name, not a credential
        resolution = _prefer_alias_source(
            features,
            target_key=key,
            source_keys=["macro.ust_10y_yield", "macro.us10y_treasury_yield"],
            rule="ust_10y_alias",
            action_family=action_family,
            action_id=action_id,
        )
        return resolution.get("record") if resolution is not None else None
    if key == "macro.rate_2y":
        resolution = _macro_rate_2y_resolution(
            features,
            action_family=action_family,
            action_id=action_id,
        )
        return resolution.get("record") if resolution is not None else None
    if key == "macro.sofr":
        resolution = _fill_alias_source(
            features,
            target_key=key,
            source_keys=["macro.sofr", "macro.sofr_or_fed_funds"],
            rule="sofr_compatibility_fallback",
            action_family=action_family,
            action_id=action_id,
        )
        return resolution.get("record") if resolution is not None else None
    if key == "market.ig_oas":
        resolution = _prefer_alias_source(
            features,
            target_key=key,
            source_keys=["macro.ig_oas", "macro.us_ig_oas"],
            rule="credit_ig_alias",
            action_family=action_family,
            action_id=action_id,
        )
        return resolution.get("record") if resolution is not None else None
    if key == "market.hy_oas":
        resolution = _prefer_alias_source(
            features,
            target_key=key,
            source_keys=["macro.hy_oas"],
            rule="credit_hy_alias",
            action_family=action_family,
            action_id=action_id,
        )
        return resolution.get("record") if resolution is not None else None
    if key == "market.pe":
        resolution = _fill_alias_source(
            features,
            target_key=key,
            source_keys=["market.pe_ratio"],
            rule="pe_ratio_compatibility_alias",
            action_family=action_family,
            action_id=action_id,
        )
        return resolution.get("record") if resolution is not None else None
    return features.get(key)


def resolve_feature_record(
    features: Dict[str, Any],
    key: str,
    *,
    action_family: Optional[str] = None,
    action_id: Optional[str] = None,
) -> Any:
    if not isinstance(features, dict):
        return None
    if not runtime_feature_adapter_enabled():
        return features.get(key)

    return _resolve_feature_record(
        features,
        key,
        action_family=action_family,
        action_id=action_id,
    )


def resolve_feature_value(
    features: Dict[str, Any],
    key: str,
    default: Any = None,
    *,
    action_family: Optional[str] = None,
    action_id: Optional[str] = None,
) -> Any:
    record = resolve_feature_record(
        features,
        key,
        action_family=action_family,
        action_id=action_id,
    )
    if record is None:
        return default
    value = _feature_value(record)
    return default if value is None else value


def adapt_snapshot(
    snapshot: Dict[str, Any],
    *,
    action_family: Optional[str] = None,
    action_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    features = dict((snapshot or {}).get("features", {}) or {})
    if not runtime_feature_adapter_enabled():
        diagnostics = {
            "enabled": False,
            "profile": _rules_profile_label(),
            "action_family": _resolved_action_family(action_family=action_family, action_id=action_id),
            "action_id": str(action_id or "") or None,
            "allowed_rules": sorted(_allowed_rules() or []),
            "replacement_count": 0,
            "ignored_legacy_count": 0,
            "counts_by_target": {},
            "counts_by_source": {},
            "counts_by_support_mode": {},
            "replacements": [],
        }
        adapted = dict(snapshot or {})
        adapted["features"] = features
        return adapted, diagnostics

    target_keys = [
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
    ]
    adapted_features = dict(features)
    replacements: List[Dict[str, Any]] = []
    counts_by_target: Dict[str, int] = {}
    counts_by_source: Dict[str, int] = {}
    counts_by_support_mode: Dict[str, int] = {}
    ignored_legacy_count = 0

    for key in target_keys:
        resolution_record = resolve_feature_record(
            features,
            key,
            action_family=action_family,
            action_id=action_id,
        )
        if resolution_record is None:
            continue
        source_key = key
        rule = "legacy_direct"
        ignored_legacy = False
        synthetic = False
        if runtime_feature_adapter_enabled():
            if key == "capital_structure.net_debt":
                info = _prefer_alias_source(
                    features,
                    target_key=key,
                    source_keys=["capital_structure.net_debt_normalized"],
                    rule="normalized_net_debt",
                    action_family=action_family,
                    action_id=action_id,
                )
            elif key == "capital_structure.net_leverage":
                info = _prefer_alias_source(
                    features,
                    target_key=key,
                    source_keys=["capital_structure.net_leverage_normalized"],
                    rule="normalized_net_leverage",
                    action_family=action_family,
                    action_id=action_id,
                )
            elif key == "capital_structure.gross_leverage":
                info = _prefer_alias_source(
                    features,
                    target_key=key,
                    source_keys=["capital_structure.gross_leverage_normalized"],
                    rule="normalized_gross_leverage",
                    action_family=action_family,
                    action_id=action_id,
                )
            elif key == "liquidity.available_for_actions":
                info = _prefer_alias_source(
                    features,
                    target_key=key,
                    source_keys=["liquidity.available_liquidity_normalized"],
                    rule="normalized_available_liquidity",
                    action_family=action_family,
                    action_id=action_id,
                )
            elif key == "operating.ebitda_ttm":
                info = _fill_alias_source(
                    features,
                    target_key=key,
                    source_keys=["operating.operating_earnings_normalized"],
                    rule="normalized_operating_earnings_fill",
                    action_family=action_family,
                    action_id=action_id,
                )
            elif key == "macro.rate_10y":  # gitleaks:allow — schema field name, not a credential
                info = _prefer_alias_source(
                    features,
                    target_key=key,
                    source_keys=["macro.ust_10y_yield", "macro.us10y_treasury_yield"],
                    rule="ust_10y_alias",
                    action_family=action_family,
                    action_id=action_id,
                )
            elif key == "macro.rate_2y":
                info = _macro_rate_2y_resolution(
                    features,
                    action_family=action_family,
                    action_id=action_id,
                )
            elif key == "macro.sofr":
                info = _fill_alias_source(
                    features,
                    target_key=key,
                    source_keys=["macro.sofr", "macro.sofr_or_fed_funds"],
                    rule="sofr_compatibility_fallback",
                    action_family=action_family,
                    action_id=action_id,
                )
            elif key == "market.ig_oas":
                info = _prefer_alias_source(
                    features,
                    target_key=key,
                    source_keys=["macro.ig_oas", "macro.us_ig_oas"],
                    rule="credit_ig_alias",
                    action_family=action_family,
                    action_id=action_id,
                )
            elif key == "market.hy_oas":
                info = _prefer_alias_source(
                    features,
                    target_key=key,
                    source_keys=["macro.hy_oas"],
                    rule="credit_hy_alias",
                    action_family=action_family,
                    action_id=action_id,
                )
            elif key == "market.pe":
                info = _fill_alias_source(
                    features,
                    target_key=key,
                    source_keys=["market.pe_ratio"],
                    rule="pe_ratio_compatibility_alias",
                    action_family=action_family,
                    action_id=action_id,
                )
            else:
                info = None
            if info is not None:
                source_key = str(info.get("source_key") or key)
                rule = str(info.get("rule") or "legacy_direct")
                ignored_legacy = bool(info.get("ignored_legacy", False))
                synthetic = bool(info.get("synthetic", False))

        if source_key == key and key in features:
            continue

        adapted_features[key] = resolution_record
        counts_by_target[key] = counts_by_target.get(key, 0) + 1
        counts_by_source[source_key] = counts_by_source.get(source_key, 0) + 1
        support_mode = _support_mode(resolution_record) or "unspecified"
        counts_by_support_mode[support_mode] = counts_by_support_mode.get(support_mode, 0) + 1
        if ignored_legacy:
            ignored_legacy_count += 1
        replacements.append(
            {
                "target_key": key,
                "source_key": source_key,
                "support_mode": support_mode,
                "ignored_legacy": ignored_legacy,
                "synthetic": synthetic,
                "rule": rule,
            }
        )

    adapted = dict(snapshot or {})
    adapted["features"] = adapted_features
    provenance = dict(adapted.get("provenance", {}) or {})
    provenance["runtime_feature_adapter"] = {
        "enabled": True,
        "replacement_count": len(replacements),
        "ignored_legacy_count": ignored_legacy_count,
    }
    adapted["provenance"] = provenance
    diagnostics = {
        "enabled": True,
        "profile": _rules_profile_label(),
        "action_family": _resolved_action_family(action_family=action_family, action_id=action_id),
        "action_id": str(action_id or "") or None,
        "allowed_rules": sorted(_allowed_rules() or []),
        "replacement_count": len(replacements),
        "ignored_legacy_count": ignored_legacy_count,
        "counts_by_target": counts_by_target,
        "counts_by_source": counts_by_source,
        "counts_by_support_mode": counts_by_support_mode,
        "replacements": replacements,
    }
    return adapted, diagnostics


__all__ = [
    "adapt_snapshot",
    "resolve_feature_record",
    "resolve_feature_value",
    "runtime_feature_adapter_enabled",
]
