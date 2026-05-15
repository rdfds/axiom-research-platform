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


