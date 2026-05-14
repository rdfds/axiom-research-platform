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


