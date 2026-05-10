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


def _live_builder() -> CompanyStateBuilder:
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


def _feature(snapshot: Dict[str, Any], name: str) -> Dict[str, Any]:
    return dict((snapshot.get("features") or {}).get(name) or {})


def _validate_taxonomy(snapshot: Dict[str, Any], expected: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
    provenance = dict((snapshot.get("provenance") or {}).get("market_metric_context") or {})
    actual = {
        "archetype": _feature(snapshot, "taxonomy.archetype").get("value") or provenance.get("archetype"),
        "sector": _feature(snapshot, "taxonomy.sector").get("value") or provenance.get("sector"),
        "subsector": _feature(snapshot, "taxonomy.subsector").get("value") or provenance.get("subsector"),
        "override_level_applied": _feature(snapshot, "taxonomy.override_level_applied").get("value") or provenance.get("override_level_applied"),
        "support_mode": provenance.get("support_mode"),
    }
    for key, expected_value in (expected or {}).items():
        if expected_value is None:
            continue
        if actual.get(key) != expected_value:
            errors.append(f"taxonomy_mismatch:{key}:expected={expected_value}:actual={actual.get(key)}")
    return actual


def _validate_metric(snapshot: Dict[str, Any], metric_name: str, expectation: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
    feat = _feature(snapshot, metric_name)
    actual = {
        "exists": bool(feat),
        "value": feat.get("value"),
        "support_mode": feat.get("support_mode"),
        "applicability_status": feat.get("applicability_status"),
        "view_type": feat.get("view_type"),
        "missing_reason": feat.get("missing_reason"),
        "fallback_used": feat.get("fallback_used"),
        "component_breakdown": dict(feat.get("component_breakdown") or {}),
        "quality_flags": list(feat.get("quality_flags") or []),
    }
    if expectation.get("exists", True) and not feat:
        errors.append(f"missing_metric:{metric_name}")
        return actual
    if not expectation.get("exists", True):
        if feat:
            errors.append(f"unexpected_metric_present:{metric_name}")
        return actual

    if "expected_value" in expectation:
        rel_tol = float(expectation.get("rel_tol", 0.01))
        abs_tol = float(expectation.get("abs_tol", 1.0))
        if not _values_match(feat.get("value"), expectation.get("expected_value"), rel_tol=rel_tol, abs_tol=abs_tol):
            errors.append(
                f"value_mismatch:{metric_name}:expected={expectation.get('expected_value')}:actual={feat.get('value')}"
            )
    if "min_value" in expectation and _to_float(feat.get("value")) is not None:
        if float(feat.get("value")) < float(expectation["min_value"]):
            errors.append(f"value_below_min:{metric_name}:min={expectation['min_value']}:actual={feat.get('value')}")
    if "max_value" in expectation and _to_float(feat.get("value")) is not None:
        if float(feat.get("value")) > float(expectation["max_value"]):
            errors.append(f"value_above_max:{metric_name}:max={expectation['max_value']}:actual={feat.get('value')}")

    field_map = {
        "expected_support_mode": "support_mode",
        "expected_applicability_status": "applicability_status",
        "expected_view_type": "view_type",
        "expected_missing_reason": "missing_reason",
        "expected_fallback_used": "fallback_used",
    }
    for expected_field, actual_field in field_map.items():
        if expected_field in expectation and feat.get(actual_field) != expectation.get(expected_field):
            errors.append(
                f"field_mismatch:{metric_name}:{actual_field}:expected={expectation.get(expected_field)}:actual={feat.get(actual_field)}"
            )

    for key, expected_value in dict(expectation.get("component_breakdown_contains") or {}).items():
        actual_value = (feat.get("component_breakdown") or {}).get(key)
        rel_tol = float(expectation.get("component_rel_tol", expectation.get("rel_tol", 0.01)))
        abs_tol = float(expectation.get("component_abs_tol", expectation.get("abs_tol", 1.0)))
        if not _values_match(actual_value, expected_value, rel_tol=rel_tol, abs_tol=abs_tol):
            errors.append(
                f"component_mismatch:{metric_name}:{key}:expected={expected_value}:actual={actual_value}"
            )

    quality_flags = set(feat.get("quality_flags") or [])
    for expected_flag in list(expectation.get("quality_flags_contains") or []):
        if expected_flag not in quality_flags:
            errors.append(f"missing_quality_flag:{metric_name}:{expected_flag}")
    for forbidden_flag in list(expectation.get("quality_flags_excludes") or []):
        if forbidden_flag in quality_flags:
            errors.append(f"unexpected_quality_flag:{metric_name}:{forbidden_flag}")
    return actual


def validate_golden_case(case: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    with TemporaryDirectory(prefix="metric_golden_") as tmp_dir:
        snapshot_obj = _build_snapshot(case, Path(tmp_dir))
        snapshot = asdict(snapshot_obj)
    invariant_errors = check_invariants(snapshot)
    if invariant_errors:
        errors.extend(f"invariant:{err}" for err in invariant_errors)

    actual_taxonomy = _validate_taxonomy(snapshot, dict(case.get("expected_taxonomy") or {}), errors)
    actual_metrics: Dict[str, Any] = {}
    for metric_name, expectation in dict(case.get("metrics") or {}).items():
        actual_metrics[metric_name] = _validate_metric(snapshot, metric_name, dict(expectation or {}), errors)

    return {
        "case_id": case.get("case_id"),
        "description": case.get("description"),
        "company_id": case.get("company_id"),
        "as_of_date": case.get("as_of_date"),
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "actual_taxonomy": actual_taxonomy,
        "actual_metrics": actual_metrics,
    }


def validate_metric_goldens(
    path: Path | str | None = None,
    *,
    case_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    payload = load_metric_goldens(path)
    selected_case_ids = _clean_case_ids(case_ids)
    selected_cases = [
        case for case in payload["cases"]
        if selected_case_ids is None or str(case.get("case_id")) in selected_case_ids
    ]
    results = [validate_golden_case(case) for case in selected_cases]
    summary = {
        "total_cases": len(results),
        "passed_cases": sum(1 for result in results if result.get("passed")),
        "failed_cases": sum(1 for result in results if not result.get("passed")),
    }
    return {
        "goldens_path": payload["path"],
        "metadata": payload["metadata"],
        "summary": summary,
        "results": results,
    }
