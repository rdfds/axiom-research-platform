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


def _repair_ratio_node(
    *,
    target: Dict[str, Any] | None,
    due_0_12_node: Dict[str, Any] | None,
    due_12_24_node: Dict[str, Any] | None,
    denominator_node: Dict[str, Any] | None,
    denominator_metric: str,
    computed_at: str,
    exact_fallback: str,
    lower_bound_fallback: str,
) -> Dict[str, Any] | None:
    if not target or target.get("value") is not None:
        return None

    due_0_12 = _node_value(due_0_12_node)
    due_12_24 = _node_value(due_12_24_node)
    denominator = _node_value(denominator_node)
    if due_0_12 in (None,) or denominator in (None, 0):
        return None

    lower_bound_only = due_12_24 is None
    due_24m = float(due_0_12) + (float(due_12_24) if due_12_24 is not None else 0.0)
    if due_24m > float(denominator) * 1.25:
        return None

    repaired = _base_repaired_node(target, computed_at=computed_at)
    repaired["value"] = due_24m / float(denominator)
    repaired["fallback_used"] = lower_bound_fallback if lower_bound_only else exact_fallback
    repaired["support_mode"] = (
        "proxy_missing_component"
        if lower_bound_only or _node_support(denominator_node) != "exact"
        else "exact"
    )
    repaired["provenance"] = _union_provenance(due_0_12_node, due_12_24_node, denominator_node)
    repaired["component_breakdown"] = {
        "debt_due_0_12m": float(due_0_12),
        "debt_due_12_24m": float(due_12_24) if due_12_24 is not None else None,
        "debt_due_24m": due_24m,
        "lower_bound_only": lower_bound_only,
        denominator_metric: float(denominator),
        "formula": f"(debt_due_0_12m + debt_due_12_24m_or_0) / {denominator_metric}",
    }
    repaired["quality_flags"] = ["lower_bound_only"] if lower_bound_only else None
    return repaired


def repair_maturity_and_refi_metrics(
    *,
    features: Dict[str, Any],
    computed_at: str,
) -> int:
    repairs = 0
    due_0_12_node = features.get("capital_structure.debt_due_0_12m")
    due_12_24_node = features.get("capital_structure.debt_due_12_24m")
    reported_debt_node = features.get("capital_structure.total_debt_provider_direct")
    market_debt_node = features.get("capital_structure.debt_like_obligations_normalized") or reported_debt_node

    reported_ratio = _repair_ratio_node(
        target=features.get("capital_structure.maturity_wall_ratio_24m_reported"),
        due_0_12_node=due_0_12_node,
        due_12_24_node=due_12_24_node,
        denominator_node=reported_debt_node,
        denominator_metric="reported_debt",
        computed_at=computed_at,
        exact_fallback="private_debt_schedule_plus_reported_debt",
        lower_bound_fallback="current_debt_lower_bound_plus_reported_debt",
    )
    if reported_ratio is not None:
        features["capital_structure.maturity_wall_ratio_24m_reported"] = reported_ratio
        repairs += 1

    market_ratio = _repair_ratio_node(
        target=features.get("capital_structure.maturity_wall_ratio_24m_market"),
        due_0_12_node=due_0_12_node,
        due_12_24_node=due_12_24_node,
        denominator_node=market_debt_node,
        denominator_metric="economic_debt",
        computed_at=computed_at,
        exact_fallback="private_debt_schedule_plus_economic_debt",
        lower_bound_fallback="current_debt_lower_bound_plus_economic_debt",
    )
    if market_ratio is not None:
        features["capital_structure.maturity_wall_ratio_24m_market"] = market_ratio
        repairs += 1

    target = features.get("capital_structure.maturity_wall_ratio_24m")
    if target and target.get("value") is None:
        preferred = market_ratio or features.get("capital_structure.maturity_wall_ratio_24m_market")
        fallback = reported_ratio or features.get("capital_structure.maturity_wall_ratio_24m_reported")
        source = preferred if _node_value(preferred) is not None else fallback
        if source and _node_value(source) is not None:
            repaired = _base_repaired_node(target, computed_at=computed_at)
            repaired["value"] = float(source["value"])
            repaired["fallback_used"] = source.get("fallback_used")
            repaired["support_mode"] = source.get("support_mode") or "proxy_missing_component"
            repaired["provenance"] = _union_provenance(source)
            repaired["component_breakdown"] = copy.deepcopy(source.get("component_breakdown"))
            repaired["quality_flags"] = copy.deepcopy(source.get("quality_flags"))
            features["capital_structure.maturity_wall_ratio_24m"] = repaired
            repairs += 1

    for metric_name, ratio_metric in [
        ("capital_structure.refi_pressure_flag_reported", "capital_structure.maturity_wall_ratio_24m_reported"),
        ("capital_structure.refi_pressure_flag_market", "capital_structure.maturity_wall_ratio_24m_market"),
        ("capital_structure.refi_pressure_flag", "capital_structure.maturity_wall_ratio_24m"),
    ]:
        target_flag = features.get(metric_name)
        ratio_node = features.get(ratio_metric)
        ratio_value = _node_value(ratio_node)
        if not target_flag or target_flag.get("value") is not None or ratio_value is None:
            continue
        repaired = _base_repaired_node(target_flag, computed_at=computed_at)
        repaired["value"] = 1.0 if ratio_value > 0.25 else 0.0
        repaired["fallback_used"] = ratio_node.get("fallback_used")
        repaired["support_mode"] = ratio_node.get("support_mode") or "proxy_missing_component"
        repaired["provenance"] = _union_provenance(ratio_node)
        repaired["component_breakdown"] = {
            "maturity_wall_ratio_24m": ratio_value,
            "threshold": 0.25,
            "formula": "maturity_wall_ratio_24m > 0.25",
        }
        repaired["quality_flags"] = copy.deepcopy(ratio_node.get("quality_flags"))
        features[metric_name] = repaired
        repairs += 1

    return repairs


def build_summary(path: Path) -> Dict[str, Dict[str, int]]:
    counters: Dict[str, Counter[str]] = {metric: Counter() for metric in REPAIR_METRICS}
    for row in iter_rows(path):
        features = row.get("features") or {}
        for metric in REPAIR_METRICS:
            node = features.get(metric) or {}
            mode = str(node.get("support_mode") or "unsupported")
            if node.get("value") is None:
                mode = "unsupported"
            counters[metric][mode] += 1
    summary: Dict[str, Dict[str, int]] = {}
    for metric, counter in counters.items():
        summary[metric] = {
            "exact": counter["exact"],
            "proxy_missing_component": counter["proxy_missing_component"],
            "unsupported": counter["unsupported"],
        }
    return summary


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    computed_at = _now_iso()
    schedule_by_company = _load_private_debt_schedule(
        Path(args.private_debt_schedule_path) if args.private_debt_schedule_path else None
    )

    with out_path.open("w") as out_handle:
        for row in iter_rows(artifact_path):
            company_id = _normalize_company_id(row.get("company_id")) or ""
            features = row.get("features") or {}
            schedule_entry = schedule_by_company.get(company_id)
            repair_debt_due_0_12m(
                features=features,
                schedule_entry=schedule_entry,
                computed_at=computed_at,
            )
            repair_debt_due_12_24m(
                features=features,
                schedule_entry=schedule_entry,
                computed_at=computed_at,
            )
            repair_maturity_and_refi_metrics(
                features=features,
                computed_at=computed_at,
            )
            out_handle.write(json.dumps(row) + "\n")

    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(build_summary(out_path), indent=2))

    print(f"Repaired maturity metrics -> {out_path}")


if __name__ == "__main__":
    main()
