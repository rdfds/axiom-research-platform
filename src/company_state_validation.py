"""Validation helpers for CompanyState snapshots."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


VALID_SUPPORT_MODES = {
    "exact",
    "exact_not_applicable",
    "exact_structural_zero",
    "proxy",
    "proxy_missing_component",
    "inferred",
    "unsupported",
}
VALID_APPLICABILITY_STATUSES = {"primary", "secondary", "diagnostic", "unsupported"}
VALID_VIEW_TYPES = {"reported", "market", "decision"}


def _feature(snapshot: dict, name: str) -> Dict[str, Any]:
    return snapshot.get("features", {}).get(name, {}) or {}


def _value(snapshot: dict, name: str) -> Any:
    return _feature(snapshot, name).get("value")


def _metric_context(snapshot: dict, name: str) -> Dict[str, Any]:
    provenance = snapshot.get("provenance", {}) or {}
    lineage = (provenance.get("feature_lineage") or {}).get(name) or {}
    return dict(lineage.get("metric_context") or {})


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _close_enough(left: Any, right: Any, *, rel_tol: float = 0.01, abs_tol: float = 1.0) -> bool:
    if left is None or right is None:
        return left is right
    left_f = _to_float(left)
    right_f = _to_float(right)
    if left_f is None or right_f is None:
        return left == right
    tolerance = max(abs_tol, abs(right_f) * rel_tol)
    return abs(left_f - right_f) <= tolerance


def _metric_feature_names(snapshot: dict) -> List[str]:
    features = snapshot.get("features", {}) or {}
    return [
        name
        for name, feat in features.items()
        if isinstance(feat, dict) and feat.get("metric_policy_id")
    ]


def _validate_metric_metadata(snapshot: dict) -> List[str]:
    errors: List[str] = []
    required_fields = [
        "metric_policy_id",
        "market_owner",
        "methodology_registry_id",
        "methodology_metric_id",
        "canonical_owner_id",
        "canonical_classification",
        "market_layer_status",
        "current_alignment_status",
        "support_mode",
        "applicability_status",
        "view_type",
    ]
    for name in _metric_feature_names(snapshot):
        feat = _feature(snapshot, name)
        for field in required_fields:
            if not feat.get(field):
                errors.append(f"missing_metric_metadata:{name}:{field}")
        support_mode = feat.get("support_mode")
        applicability_status = feat.get("applicability_status")
        view_type = feat.get("view_type")
        if support_mode is not None and support_mode not in VALID_SUPPORT_MODES:
            errors.append(f"invalid_metric_support_mode:{name}")
        if applicability_status is not None and applicability_status not in VALID_APPLICABILITY_STATUSES:
            errors.append(f"invalid_metric_applicability:{name}")
        if view_type is not None and view_type not in VALID_VIEW_TYPES:
            errors.append(f"invalid_metric_view_type:{name}")

        metric_context = _metric_context(snapshot, name)
        if not metric_context:
            errors.append(f"missing_metric_lineage_context:{name}")
            continue
        for field in required_fields:
            if metric_context.get(field) != feat.get(field):
                errors.append(f"metric_lineage_context_mismatch:{name}:{field}")
        if feat.get("component_breakdown") != metric_context.get("component_breakdown"):
            errors.append(f"metric_lineage_component_mismatch:{name}")
        if feat.get("quality_flags") != metric_context.get("quality_flags"):
            errors.append(f"metric_lineage_quality_flags_mismatch:{name}")
    return errors


def _validate_metric_views(snapshot: dict) -> List[str]:
    errors: List[str] = []
    features = snapshot.get("features", {}) or {}
    for name, feat in features.items():
        if not isinstance(feat, dict):
            continue
        if feat.get("view_type") != "decision":
            continue
        if name.endswith("_reported") or name.endswith("_market"):
            continue
        reported = features.get(f"{name}_reported")
        market = features.get(f"{name}_market")
        if not isinstance(reported, dict) or not isinstance(market, dict):
            continue

        applicability_status = str(feat.get("applicability_status") or "").lower()
        support_mode = str(feat.get("support_mode") or "").lower()
        if applicability_status == "unsupported":
            if feat.get("value") is not None:
                errors.append(f"unsupported_decision_metric_has_value:{name}")
            if support_mode != "unsupported":
                errors.append(f"unsupported_decision_metric_bad_support_mode:{name}")
            continue

        if feat.get("fallback_used") == "reported_view_fallback":
            if not _close_enough(feat.get("value"), reported.get("value")):
                errors.append(f"decision_reported_fallback_mismatch:{name}")
            quality_flags = feat.get("quality_flags") or []
            if "decision_uses_reported_view" not in quality_flags:
                errors.append(f"decision_reported_fallback_missing_flag:{name}")
            continue

        if market.get("value") is not None and not _close_enough(feat.get("value"), market.get("value")):
            errors.append(f"decision_view_not_equal_market:{name}")
    return errors


def _validate_metric_arithmetic(snapshot: dict) -> List[str]:
    errors: List[str] = []
    total_debt_market = _feature(snapshot, "capital_structure.total_debt_market")
    if total_debt_market:
        breakdown = total_debt_market.get("component_breakdown") or {}
        reported_debt = _to_float(breakdown.get("reported_debt"))
        if reported_debt is not None:
            expected = reported_debt
            included_lease = breakdown.get("included_lease_liabilities")
            if included_lease is not None:
                expected += _to_float(included_lease) or 0.0
            else:
                expected += (_to_float(breakdown.get("lease_liabilities")) or 0.0) * (
                    _to_float(breakdown.get("lease_weight")) or 0.0
                )

            included_supplier_finance = breakdown.get("included_supplier_finance")
            if included_supplier_finance is not None:
                expected += _to_float(included_supplier_finance) or 0.0
            else:
                expected += (_to_float(breakdown.get("supplier_finance")) or 0.0) * (
                    _to_float(breakdown.get("supplier_finance_weight")) or 0.0
                )

            expected += (_to_float(breakdown.get("preferred_equity")) or 0.0) * (
                _to_float(breakdown.get("preferred_weight")) or 0.0
            )
            expected += (_to_float(breakdown.get("convertibles")) or 0.0) * (
                _to_float(breakdown.get("convertible_weight")) or 0.0
            )
            expected += (_to_float(breakdown.get("unfunded_pension")) or 0.0) * (
                _to_float(breakdown.get("pension_weight")) or 0.0
            )
            if total_debt_market.get("value") is not None and not _close_enough(total_debt_market.get("value"), expected):
                errors.append("market_total_debt_component_mismatch")

    net_debt_market = _feature(snapshot, "capital_structure.net_debt_market")
    if net_debt_market:
        breakdown = net_debt_market.get("component_breakdown") or {}
        economic_debt = _to_float(breakdown.get("economic_debt"))
        usable_cash = _to_float(breakdown.get("usable_cash_market"))
        if economic_debt is not None and usable_cash is not None:
            expected = economic_debt - usable_cash
            if net_debt_market.get("value") is not None and not _close_enough(net_debt_market.get("value"), expected):
                errors.append("market_net_debt_component_mismatch")
    return errors


def check_invariants(snapshot: dict) -> List[str]:
    errors: List[str] = []

    snap_asof = snapshot.get("as_of_time")
    for k, feat in snapshot.get("features", {}).items():
        feat_asof = feat.get("as_of_time")
        if feat_asof and snap_asof and feat_asof > snap_asof:
            errors.append(f"feature_asof_gt_snapshot:{k}")

    cash = _value(snapshot, "liquidity.cash")
    liq = _value(snapshot, "liquidity.liquidity_total")
    if cash is not None and liq is not None and liq < cash:
        errors.append("liquidity_total_lt_cash")

    total_debt_reported = _value(snapshot, "capital_structure.total_debt_reported")
    if total_debt_reported is None:
        total_debt_reported = _value(snapshot, "capital_structure.total_debt")
    net_debt_reported = _value(snapshot, "capital_structure.net_debt_reported")
    if net_debt_reported is None:
        net_debt_reported = _value(snapshot, "capital_structure.net_debt")
    if total_debt_reported is not None and cash is not None and net_debt_reported is not None:
        if abs((total_debt_reported - cash) - net_debt_reported) > max(1.0, abs(net_debt_reported) * 0.01):
            errors.append("net_debt_mismatch")

    total_debt_market = _value(snapshot, "capital_structure.total_debt_market")
    net_debt_market = _value(snapshot, "capital_structure.net_debt_market")
    usable_cash_market = _value(snapshot, "liquidity.usable_cash_market")
    if total_debt_market is not None and usable_cash_market is not None and net_debt_market is not None:
        if abs((total_debt_market - usable_cash_market) - net_debt_market) > max(1.0, abs(net_debt_market) * 0.01):
            errors.append("market_net_debt_mismatch")

    mcap = _value(snapshot, "market.market_cap")
    ev = _value(snapshot, "market.enterprise_value")
    if mcap is not None and total_debt_reported is not None and cash is not None and ev is not None:
        target = mcap + total_debt_reported - cash
        if abs(ev - target) > max(1.0, abs(ev) * 0.01):
            errors.append("enterprise_value_mismatch")

    errors.extend(_validate_metric_metadata(snapshot))
    errors.extend(_validate_metric_views(snapshot))
    errors.extend(_validate_metric_arithmetic(snapshot))
    return errors


def validate_peer_percentiles(snapshot: dict) -> List[str]:
    errors: List[str] = []
    keys = [
        "peer_context.valuation_percentile",
        "peer_context.leverage_percentile",
        "peer_context.margin_percentile",
        "peer_context.action_rate_percentile",
    ]
    for k in keys:
        val = _value(snapshot, k)
        if val is None:
            continue
        try:
            v = float(val)
        except Exception:
            errors.append(f"peer_percentile_not_numeric:{k}")
            continue
        if v < 0 or v > 100:
            errors.append(f"peer_percentile_out_of_range:{k}")
    return errors


def validate_peer_zscores(snapshot: dict) -> List[str]:
    errors: List[str] = []
    keys = [
        "peer_context.valuation_z",
        "peer_context.leverage_z",
        "peer_context.margin_z",
        "peer_context.action_rate_z",
    ]
    for k in keys:
        val = _value(snapshot, k)
        if val is None:
            continue
        try:
            float(val)
        except Exception:
            errors.append(f"peer_zscore_not_numeric:{k}")
    return errors


def validate_peer_bands(snapshot: dict) -> List[str]:
    errors: List[str] = []
    keys = [
        "peer_context.valuation_band",
        "peer_context.leverage_band",
        "peer_context.margin_band",
        "peer_context.action_rate_band",
    ]
    for k in keys:
        val = _value(snapshot, k)
        if val is None:
            continue
        if val not in ("q1", "q2", "q3", "q4"):
            errors.append(f"peer_band_invalid:{k}")
    return errors
