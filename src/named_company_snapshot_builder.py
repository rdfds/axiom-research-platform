from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .company_state_builder import CompanyStateBuilder, snapshot_to_json
from .data_paths import resolve_companyfacts_root, resolve_data_path
from .named_company_metric_benchmarks import (
    DEFAULT_TARGETS_PATH,
    _snapshot_metric_packet,
    load_named_company_targets,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRESH_SNAPSHOT_ROOT = Path("/tmp/named_company_snapshots_fresh/keyed")


def _default_facts_path() -> Path:
    return resolve_data_path(ROOT / "data" / "inputs_layer" / "extracted_fact_registry_validity")


def _default_entity_table_path() -> Path:
    return resolve_data_path(ROOT / "data" / "inputs_layer" / "entity.parquet")


def _default_taxonomy_reference_path() -> Path:
    return resolve_data_path(ROOT / "data" / "refinitiv" / "fundamentals_all.parquet")


def _default_issuer_ratings_path() -> Path:
    return resolve_data_path(ROOT / "data" / "inputs_layer" / "issuer_rating_history.parquet")


def _default_companyfacts_root() -> Optional[Path]:
    return resolve_companyfacts_root(ROOT / "data" / "sec" / "companyfacts")

def _clean_case_ids(case_ids: Optional[Iterable[str]]) -> Optional[set[str]]:
    if case_ids is None:
        return None
    values = {str(case_id).strip() for case_id in case_ids if str(case_id).strip()}
    return values or None


def _as_of_year(as_of_date: str) -> int:
    return int(str(as_of_date).split("-", 1)[0])


def required_fact_years(as_of_date: str, lookback_years: int = 5) -> List[int]:
    year = _as_of_year(as_of_date)
    start_year = max(2000, year - max(1, lookback_years) + 1)
    return list(range(start_year, year + 1))


