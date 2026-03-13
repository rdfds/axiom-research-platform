#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import duckdb
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model_feature_bundle import _STATE_VECTOR_V1_FEATURES
from src.pipeline.precedent_brain import augment_precedent_state_vector_columns
from scripts.build_precedent_quality_supervision_dataset import (
    _PAIRWISE_FEATURE_GAP_SUMMARY_FEATURES,
    _enrich_match_compact,
    _normalize_as_of_time,
    _snapshot_catalog_index,
    _target_compact_values,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich an existing pairwise supervision dataset with fuller target and precedent features.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--summary-path", default="")
    parser.add_argument("--snapshot-catalog-path", default="")
    parser.add_argument("--outcomes-path", default="")
    return parser.parse_args()


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _parse_precedent_decision_time(precedent_id: str) :
    parts = str(precedent_id or "").split("::")
    if len(parts) >= 2:
        return _normalize_as_of_time(parts[1])
    return ""


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _load_needed_precedent_outcomes_lookup(
    outcomes_path: Path,
    *,
    rows: List[Dict[str, Any]],
) -> Dict[tuple[str, str, str], Dict[str, Any]]:
    company_ids = sorted(
        {
            str(value).strip()
            for row in rows
            for value in (
                row.get("positive_precedent_company_id"),
                row.get("negative_precedent_company_id"),
            )
            if str(value or "").strip()
        }
    )
    action_ids = sorted(
        {
            str(value).strip()
            for row in rows
            for value in (
                row.get("anchor_action_id"),
                row.get("competitor_action_id"),
            )
            if str(value or "").strip()
        }
    )
    decision_times = [
        pd.to_datetime(_parse_precedent_decision_time(str(value or "")), utc=True, errors="coerce")
        for row in rows
        for value in (
            row.get("positive_precedent_id"),
            row.get("negative_precedent_id"),
        )
    ]
    decision_times = [stamp for stamp in decision_times if pd.notna(stamp)]
    if not company_ids or not action_ids:
        return {}
    company_sql = ", ".join(_sql_literal(value) for value in company_ids)
    action_sql = ", ".join(_sql_literal(value) for value in action_ids)
    where_clauses = [
        f"CAST(company_id AS VARCHAR) IN ({company_sql})",
        f"CAST(normalized_action_id AS VARCHAR) IN ({action_sql})",
    ]
    if decision_times:
        min_date = min(decision_times).date().isoformat()
        max_date = max(decision_times).date().isoformat()
        where_clauses.append(
            f"CAST(action_date AS DATE) BETWEEN DATE '{min_date}' AND DATE '{max_date}'"
        )
    query = f"""
        SELECT *
        FROM read_parquet(?)
        WHERE {' AND '.join(where_clauses)}
    """
    frame = duckdb.execute(query, [str(outcomes_path)]).df()
    if frame.empty:
        return {}
    frame = augment_precedent_state_vector_columns(frame)
    frame["company_id"] = frame["company_id"].astype(str)
    frame["normalized_action_id"] = frame["normalized_action_id"].astype(str)
    frame["action_date"] = pd.to_datetime(frame["action_date"], utc=True, errors="coerce")
    lookup: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        action_date = row.get("action_date")
        if pd.isna(action_date):
            continue
        key = (
            str(row.get("company_id") or "").strip(),
            str(row.get("normalized_action_id") or "").strip(),
            _normalize_as_of_time(str(action_date)),
        )
        if key[0] and key[1] and key[2] and key not in lookup:
            lookup[key] = row
    return lookup


def _gap_summary(target_compact: Dict[str, Any], positive_compact: Dict[str, Any], negative_compact: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for feature in _PAIRWISE_FEATURE_GAP_SUMMARY_FEATURES:
        target_value = target_compact.get(feature)
        positive_value = positive_compact.get(feature)
        negative_value = negative_compact.get(feature)
        positive_abs_diff = None if target_value is None or positive_value is None else abs(float(target_value) - float(positive_value))
        negative_abs_diff = None if target_value is None or negative_value is None else abs(float(target_value) - float(negative_value))
        out[feature] = {
            "positive_abs_diff": positive_abs_diff,
            "negative_abs_diff": negative_abs_diff,
        }
    return out


