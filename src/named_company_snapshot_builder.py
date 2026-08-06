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


def build_named_company_snapshots(
    targets_path: Path | str | None = None,
    *,
    snapshot_root: Path | str = DEFAULT_FRESH_SNAPSHOT_ROOT,
    facts_path: Path | str | None = None,
    entity_table_path: Path | str | None = None,
    taxonomy_reference_path: Path | str | None = None,
    issuer_ratings_path: Path | str | None = None,
    facts_lookback_years: int = 5,
    case_ids: Optional[Iterable[str]] = None,
    force: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    payload = load_named_company_targets(targets_path or DEFAULT_TARGETS_PATH)
    selected_case_ids = _clean_case_ids(case_ids)
    targets = [
        target for target in payload["targets"]
        if selected_case_ids is None or str(target.get("case_id")) in selected_case_ids
    ]

    snapshot_root_path = Path(snapshot_root)
    facts_path = Path(facts_path) if facts_path is not None else _default_facts_path()
    entity_table_path = Path(entity_table_path) if entity_table_path is not None else _default_entity_table_path()
    taxonomy_reference_path = (
        Path(taxonomy_reference_path)
        if taxonomy_reference_path is not None
        else _default_taxonomy_reference_path()
    )
    issuer_ratings_path = (
        Path(issuer_ratings_path)
        if issuer_ratings_path is not None
        else _default_issuer_ratings_path()
    )
    snapshot_root_path.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    summary = {
        "total_targets": len(targets),
        "built": 0,
        "skipped_existing": 0,
        "blocked_unmaterialized_inputs": 0,
        "build_failed": 0,
    }

    builders: Dict[tuple[str, tuple[int, ...]], CompanyStateBuilder] = {}

    for target in targets:
        case_id = str(target.get("case_id") or "")
        company_id = str(target.get("company_id") or "")
        ticker = str(target.get("ticker") or "").strip()
        as_of_date = str(target.get("as_of_date") or "").strip()
        fact_years = required_fact_years(as_of_date, facts_lookback_years)
        required_paths = _required_input_paths(
            as_of_date=as_of_date,
            facts_path=facts_path,
            facts_lookback_years=facts_lookback_years,
            entity_table_path=entity_table_path,
            taxonomy_reference_path=taxonomy_reference_path,
            issuer_ratings_path=issuer_ratings_path,
        )
        blocked_input_paths = [str(path) for path in required_paths if not _is_readable_file(path)]
        output_path = _snapshot_path(snapshot_root_path, company_id, as_of_date)
        result: Dict[str, Any] = {
            "case_id": case_id,
            "company_id": company_id,
            "ticker": ticker,
            "display_name": target.get("display_name"),
            "as_of_date": as_of_date,
            "expected_archetype": target.get("expected_archetype"),
            "required_fact_years": fact_years,
            "snapshot_path": str(output_path),
            "blocked_input_paths": blocked_input_paths,
        }
        if blocked_input_paths:
            result["build_status"] = "blocked_unmaterialized_inputs"
            summary["blocked_unmaterialized_inputs"] += 1
            results.append(result)
            continue

        if output_path.exists() and _is_readable_file(output_path) and not force:
            snapshot = json.loads(output_path.read_text())
            packet = _snapshot_metric_packet(snapshot)
            result.update(packet)
            result["build_status"] = "skipped_existing"
            result["actual_archetype"] = packet.get("market_metric_context", {}).get("archetype")
            result["archetype_match"] = (
                None
                if result.get("expected_archetype") is None
                else result["actual_archetype"] == result.get("expected_archetype")
            )
            summary["skipped_existing"] += 1
            results.append(result)
            continue

        builder_key = (as_of_date, tuple(fact_years))
        builder = builders.get(builder_key)
        if builder is None:
            builder = _builder_for_target(
                facts_path=facts_path,
                facts_years=fact_years,
                entity_table_path=entity_table_path,
                taxonomy_reference_path=taxonomy_reference_path,
                issuer_ratings_path=issuer_ratings_path,
                debug=debug,
            )
            builders[builder_key] = builder

        try:
            snapshot_obj = builder.build(company_id, as_of_date, extra_aliases=[ticker] if ticker else None)
            snapshot = snapshot_to_json(snapshot_obj)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(snapshot, indent=2))
            packet = _snapshot_metric_packet(snapshot)
            result.update(packet)
            result["build_status"] = "built"
            result["actual_archetype"] = packet.get("market_metric_context", {}).get("archetype")
            result["archetype_match"] = (
                None
                if result.get("expected_archetype") is None
                else result["actual_archetype"] == result.get("expected_archetype")
            )
            summary["built"] += 1
        except Exception as exc:
            result["build_status"] = "build_failed"
            result["error"] = f"{type(exc).__name__}: {exc}"
            summary["build_failed"] += 1
        results.append(result)

    return {
        "targets_path": payload["path"],
        "metadata": payload["metadata"],
        "snapshot_root": str(snapshot_root_path),
        "facts_path": str(facts_path),
        "facts_lookback_years": facts_lookback_years,
        "summary": summary,
        "results": results,
    }
