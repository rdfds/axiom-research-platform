from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd

from .action_ontology import build_default_action_schema_registry


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_RELEVANT_ACTION_SOURCE_FILES: Sequence[Path] = (
    REPO_ROOT / "src/candidate_generation.py",
    REPO_ROOT / "src/mechanism_brain.py",
    REPO_ROOT / "src/planner_brain.py",
)


def coverage_status(count: int) -> str:
    if count >= 1000:
        return "strong"
    if count >= 100:
        return "usable"
    if count >= 1:
        return "thin"
    return "missing"


def extract_relevant_action_ids(
    *,
    source_files: Optional[Iterable[Path]] = None,
    valid_action_ids: Optional[Set[str]] = None,
) -> List[str]:
    registry = build_default_action_schema_registry()
    allowed_action_ids = valid_action_ids or {action["action_id"] for action in registry.actions}
    pattern = re.compile(r"([a-z_]+\.[a-z_]+)")
    found: Set[str] = set()
    for path in source_files or DEFAULT_RELEVANT_ACTION_SOURCE_FILES:
        text = Path(path).read_text()
        for match in pattern.finditer(text):
            candidate = match.group(1)
            if candidate in allowed_action_ids:
                found.add(candidate)
    return sorted(found)


def build_action_support_report(
    *,
    outcomes_path: str | Path,
    relevant_action_ids: Optional[Sequence[str]] = None,
    source_files: Optional[Iterable[Path]] = None,
) -> Dict[str, Any]:
    path = Path(outcomes_path)
    df = pd.read_parquet(path, columns=["normalized_action_family", "normalized_action_id"])
    relevant_ids = list(relevant_action_ids or extract_relevant_action_ids(source_files=source_files))
    action_counts = df["normalized_action_id"].value_counts(dropna=True).to_dict()
    family_counts = {
        str(key): int(value)
        for key, value in df["normalized_action_family"].value_counts(dropna=False).items()
        if pd.notna(key)
    }

    relevant_actions: List[Dict[str, Any]] = []
    exact_status_counts: Dict[str, int] = {}
    support_mode_counts: Dict[str, int] = {}
    missing_relevant_actions: List[str] = []
    for action_id in relevant_ids:
        family = action_id.split(".", 1)[0] if "." in action_id else ""
        exact_count = int(action_counts.get(action_id, 0))
        family_count = int(family_counts.get(family, 0))
        exact_status = coverage_status(exact_count)
        if exact_count > 0:
            support_mode = "exact_supported"
        elif family_count > 0:
            support_mode = "family_only"
        else:
            support_mode = "unsupported"
        relevant_actions.append(
            {
                "action_id": action_id,
                "family": family,
                "exact_count": exact_count,
                "family_count": family_count,
                "exact_support_status": exact_status,
                "support_mode": support_mode,
            }
        )
        exact_status_counts[exact_status] = exact_status_counts.get(exact_status, 0) + 1
        support_mode_counts[support_mode] = support_mode_counts.get(support_mode, 0) + 1
        if support_mode != "exact_supported":
            missing_relevant_actions.append(action_id)

    return {
        "outcomes_path": str(path),
        "family_counts": family_counts,
        "relevant_action_count": len(relevant_actions),
        "exact_status_counts": dict(sorted(exact_status_counts.items())),
        "support_mode_counts": dict(sorted(support_mode_counts.items())),
        "relevant_actions": relevant_actions,
        "non_exact_supported_actions": missing_relevant_actions,
    }


def resolve_action_support(
    *,
    action_id: str,
    action_family: Optional[str],
    support_report: Dict[str, Any],
) -> Dict[str, Any]:
    normalized_id = str(action_id or "")
    family = str(action_family or "")
    if not family and "." in normalized_id:
        family = normalized_id.split(".", 1)[0]
    for item in list(support_report.get("relevant_actions", []) or []):
        if str(item.get("action_id") or "") == normalized_id:
            return dict(item)
    family_count = int((support_report.get("family_counts", {}) or {}).get(family, 0) or 0)
    support_mode = "family_only" if family_count > 0 else "unsupported"
    return {
        "action_id": normalized_id,
        "family": family,
        "exact_count": 0,
        "family_count": family_count,
        "exact_support_status": "missing",
        "support_mode": support_mode,
    }


def load_action_support_report(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_action_support_report(report: Dict[str, Any], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return out_path


__all__ = [
    "DEFAULT_RELEVANT_ACTION_SOURCE_FILES",
    "build_action_support_report",
    "coverage_status",
    "extract_relevant_action_ids",
    "load_action_support_report",
    "resolve_action_support",
    "write_action_support_report",
]
