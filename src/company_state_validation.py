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


