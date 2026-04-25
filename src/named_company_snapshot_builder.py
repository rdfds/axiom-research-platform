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


