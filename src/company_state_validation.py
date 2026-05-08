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
    return snapshot['features'].get(name, {}) or {}


def _value(snapshot: dict, name: str) -> Any:
    return _feature(snapshot, name).get("value")


