from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, List, Optional
import json

import pandas as pd

from .company_state_builder import CompanyStateBuilder
from .company_state_validation import check_invariants
from .data_paths import resolve_companyfacts_root


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDENS_PATH = ROOT / "configs" / "metric_goldens" / "consumer_industrials_wave1.json"


def _default_companyfacts_root() -> Optional[Path]:
    return resolve_companyfacts_root(ROOT / "data" / "sec" / "companyfacts")

def _clean_case_ids(case_ids: Optional[Iterable[str]]) -> Optional[set[str]]:
    if case_ids is None:
        return None
    values = {str(case_id).strip() for case_id in case_ids if str(case_id).strip()}
    return values or None


def load_metric_goldens(path: Path | str | None = None) -> Dict[str, Any]:
    goldens_path = Path(path) if path is not None else DEFAULT_GOLDENS_PATH
    payload = json.loads(goldens_path.read_text())
    if isinstance(payload, list):
        return {"metadata": {}, "cases": payload, "path": str(goldens_path)}
    return {
        "metadata": dict(payload.get("metadata") or {}),
        "cases": list(payload.get("cases") or []),
        "path": str(goldens_path),
    }


def _write_parquet_if_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _synthetic_builder_for_case(case: Dict[str, Any], workdir: Path) -> CompanyStateBuilder:
    inputs = dict(case.get("inputs") or {})
    facts_rows = list(inputs.get("facts") or [])
    if not facts_rows:
        raise ValueError(f"golden_case_missing_facts:{case.get('case_id')}")

    facts_path = workdir / "facts.parquet"
    entity_path = workdir / "entity.parquet"
    entity_identifier_path = workdir / "entity_identifier.parquet"
    taxonomy_reference_path = workdir / "taxonomy_reference.parquet"
    events_path = workdir / "events.parquet"
    ownership_path = workdir / "ownership.parquet"
    issuer_ratings_path = workdir / "issuer_ratings.parquet"
    timeseries_path = workdir / "timeseries.parquet"

    _write_parquet_if_rows(facts_path, facts_rows)
    _write_parquet_if_rows(entity_path, list(inputs.get("entity") or []))
    _write_parquet_if_rows(entity_identifier_path, list(inputs.get("entity_identifiers") or []))
    _write_parquet_if_rows(taxonomy_reference_path, list(inputs.get("taxonomy_reference") or []))
    _write_parquet_if_rows(events_path, list(inputs.get("events") or []))
    _write_parquet_if_rows(ownership_path, list(inputs.get("ownership") or []))
    _write_parquet_if_rows(issuer_ratings_path, list(inputs.get("issuer_ratings") or []))
    _write_parquet_if_rows(timeseries_path, list(inputs.get("timeseries") or []))

    return CompanyStateBuilder(
        raw_timeseries_path=timeseries_path,
        macro_timeseries_path=timeseries_path,
        event_store_path=events_path,
        facts_path=facts_path,
        ownership_summary_path=ownership_path,
        issuer_ratings_path=issuer_ratings_path,
        entity_table_path=entity_path,
        entity_identifier_path=entity_identifier_path,
        taxonomy_reference_path=taxonomy_reference_path,
        skip_timeseries=not bool(inputs.get("timeseries")),
        skip_macro=True,
        skip_events=not bool(inputs.get("events")),
        skip_peer_context=True,
        historical_backfill_mode=bool(case.get("historical_backfill_mode", False)),
    )


def _live_builder() :
    companyfacts_root = _default_companyfacts_root()
    return CompanyStateBuilder(
        skip_peer_context=True,
        companyfacts_root=companyfacts_root if companyfacts_root and companyfacts_root.exists() else None,
        enable_market_relevant_smart_normalized_inputs=True,
    )


def _build_snapshot(case: Dict[str, Any], workdir: Path):
    if case.get("inputs"):
        builder = _synthetic_builder_for_case(case, workdir)
    else:
        builder = _live_builder()
    company_id = str(case.get("company_id") or "").strip()
    as_of_date = str(case.get("as_of_date") or "").strip()
    if not company_id or not as_of_date:
        raise ValueError(f"golden_case_missing_identity:{case.get('case_id')}")
    return builder.build(company_id, as_of_date)


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _values_match(left: Any, right: Any, *, rel_tol: float = 0.01, abs_tol: float = 1.0) -> bool:
    if left is None or right is None:
        return left is right
    left_float = _to_float(left)
    right_float = _to_float(right)
    if left_float is None or right_float is None:
        return left == right
    tolerance = max(abs_tol, abs(right_float) * rel_tol)
    return abs(left_float - right_float) <= tolerance


