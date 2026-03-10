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


