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


def _is_readable_file(path: Path) -> bool:
    try:
        if not path.exists():
            return False
        stat_result = os.stat(path)
        if stat_result.st_size < 4:
            return False
        with path.open("rb") as handle:
            handle.read(4)
        return True
    except Exception:
        return False


def _snapshot_path(snapshot_root: Path, company_id: str, as_of_date: str) -> Path:
    return snapshot_root / f"as_of_date={as_of_date}" / f"company_id={company_id}.json"


def _default_null_path(name: str) -> Path:
    return Path("/tmp") / f"named_company_snapshot_builder_null_{name}"


def _required_input_paths(
    *,
    as_of_date: str,
    facts_path: Path,
    facts_lookback_years: int,
    entity_table_path: Path,
    taxonomy_reference_path: Path,
    issuer_ratings_path: Path,
) -> List[Path]:
    paths = [
        facts_path / f"year={year}" / "part.parquet"
        for year in required_fact_years(as_of_date, facts_lookback_years)
    ]
    paths.extend(
        [
            entity_table_path,
            taxonomy_reference_path,
            issuer_ratings_path,
        ]
    )
    return paths


def _builder_for_target(
    *,
    facts_path: Path,
    facts_years: List[int],
    entity_table_path: Path,
    taxonomy_reference_path: Path,
    issuer_ratings_path: Path,
    debug: bool,
) -> CompanyStateBuilder:
    companyfacts_root = _default_companyfacts_root()
    return CompanyStateBuilder(
        raw_timeseries_path=_default_null_path("raw_timeseries.parquet"),
        macro_timeseries_path=_default_null_path("macro_timeseries.parquet"),
        event_store_path=_default_null_path("event_store.parquet"),
        facts_path=facts_path,
        ownership_summary_path=_default_null_path("ownership_13f_summary.parquet"),
        issuer_ratings_path=issuer_ratings_path,
        entity_graph_path=_default_null_path("entity_graph.parquet"),
        entity_identifier_path=_default_null_path("entity_identifier.parquet"),
        entity_table_path=entity_table_path,
        taxonomy_reference_path=taxonomy_reference_path,
        companyfacts_root=companyfacts_root if companyfacts_root and companyfacts_root.exists() else None,
        enable_market_relevant_smart_normalized_inputs=True,
        skip_timeseries=True,
        skip_macro=True,
        skip_events=True,
        skip_peer_context=True,
        facts_years=facts_years,
        debug=debug,
    )


