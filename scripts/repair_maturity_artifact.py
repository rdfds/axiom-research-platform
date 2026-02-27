#!/usr/bin/env python3
"""Repair debt-due and maturity-wall metrics in a materialized company-state artifact."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd


REPAIR_METRICS = [
    "capital_structure.debt_due_0_12m",
    "capital_structure.debt_due_12_24m",
    "capital_structure.maturity_wall_ratio_24m_reported",
    "capital_structure.maturity_wall_ratio_24m_market",
    "capital_structure.maturity_wall_ratio_24m",
    "capital_structure.refi_pressure_flag_reported",
    "capital_structure.refi_pressure_flag_market",
    "capital_structure.refi_pressure_flag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True, help="Input company-state JSONL artifact")
    parser.add_argument("--out", required=True, help="Output repaired JSONL artifact")
    parser.add_argument("--summary-out", help="Optional summary JSON")
    parser.add_argument(
        "--private-debt-schedule-path",
        help="Optional parquet with due_0_12/due_12_24 style schedule buckets keyed by company_id",
    )
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _node_value(node: Dict[str, Any] | None) -> float | None:
    if not node:
        return None
    value = node.get("value")
    if value is None:
        return None
    return float(value)


def _node_support(node: Dict[str, Any] | None) -> str:
    if not node:
        return "unsupported"
    return str(node.get("support_mode") or "unsupported")


def _union_provenance(*nodes: Dict[str, Any] | None) -> list[Dict[str, Any]]:
    merged: list[Dict[str, Any]] = []
    seen = set()
    for node in nodes:
        for prov in (node or {}).get("provenance") or []:
            key = json.dumps(prov, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            merged.append(copy.deepcopy(prov))
    return merged


def _base_repaired_node(node: Dict[str, Any], *, computed_at: str) -> Dict[str, Any]:
    repaired = copy.deepcopy(node)
    repaired["computed_at"] = computed_at
    repaired["missing_reason"] = None
    repaired["quality_flags"] = repaired.get("quality_flags") or None
    return repaired


def _normalize_company_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return text
    return digits.zfill(10)


def _pick_column(columns: list[str], *candidates: str) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match:
            return match
    return None


def _load_private_debt_schedule(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None or not path.exists():
        return {}
    frame = pd.read_parquet(path)
    if frame.empty:
        return {}
    columns = list(frame.columns)
    company_col = _pick_column(columns, "company_id", "entity_id", "cik")
    if company_col is None:
        return {}

    bucket_map = {
        "due_0_12": _pick_column(columns, "due_0_12", "debt_due_0_12m"),
        "due_12_24": _pick_column(columns, "due_12_24", "debt_due_12_24m"),
        "due_24_36": _pick_column(columns, "due_24_36", "debt_due_24_36m"),
        "due_36_60": _pick_column(columns, "due_36_60", "debt_due_36_60m"),
        "due_60_plus": _pick_column(columns, "due_60_plus", "debt_due_60m_plus"),
    }

    schedules: dict[str, dict[str, float]] = {}
    for _, row in frame.iterrows():
        company_id = _normalize_company_id(row.get(company_col))
        if company_id is None:
            continue
        entry: dict[str, float] = {}
        for bucket, column in bucket_map.items():
            if column is None:
                continue
            value = pd.to_numeric(row.get(column), errors="coerce")
            if pd.isna(value):
                continue
            entry[bucket] = float(value)
        if entry:
            schedules[company_id] = entry
    return schedules


def repair_debt_due_0_12m(
    *,
    features: Dict[str, Any],
    schedule_entry: dict[str, float] | None,
    computed_at: str,
) -> bool:
    target = features.get("capital_structure.debt_due_0_12m")
    if not target or target.get("value") is not None:
        return False

    current_debt_node = features.get("capital_structure.current_debt_statement_direct")
    current_debt = _node_value(current_debt_node)
    fallback_used = None
    support_mode = None
    component_breakdown = None

    if schedule_entry and schedule_entry.get("due_0_12") is not None:
        current_debt = float(schedule_entry["due_0_12"])
        fallback_used = "private_debt_schedule"
        support_mode = "exact"
        component_breakdown = {
            "due_0_12": current_debt,
            "formula": "private_debt_schedule.due_0_12",
            "schedule_source": "private_debt_schedule",
        }
    elif current_debt is not None:
        fallback_used = "current_debt_statement_direct_as_due_0_12m"
        support_mode = _node_support(current_debt_node)
        component_breakdown = {
            "current_debt_statement_direct": current_debt,
            "formula": "current_debt_statement_direct",
        }

    if current_debt is None:
        return False

    repaired = _base_repaired_node(target, computed_at=computed_at)
    repaired["value"] = current_debt
    repaired["fallback_used"] = fallback_used
    repaired["support_mode"] = support_mode or "proxy_missing_component"
    repaired["provenance"] = _union_provenance(current_debt_node)
    repaired["component_breakdown"] = component_breakdown
    repaired["quality_flags"] = None
    features["capital_structure.debt_due_0_12m"] = repaired
    return True


def repair_debt_due_12_24m(
    *,
    features: Dict[str, Any],
    schedule_entry: dict[str, float] | None,
    computed_at: str,
) -> bool:
    target = features.get("capital_structure.debt_due_12_24m")
    if not target or target.get("value") is not None:
        return False
    if not schedule_entry or schedule_entry.get("due_12_24") is None:
        return False

    repaired = _base_repaired_node(target, computed_at=computed_at)
    repaired["value"] = float(schedule_entry["due_12_24"])
    repaired["fallback_used"] = "private_debt_schedule"
    repaired["support_mode"] = "exact"
    repaired["component_breakdown"] = {
        "due_12_24": float(schedule_entry["due_12_24"]),
        "formula": "private_debt_schedule.due_12_24",
        "schedule_source": "private_debt_schedule",
    }
    repaired["quality_flags"] = None
    features["capital_structure.debt_due_12_24m"] = repaired
    return True


