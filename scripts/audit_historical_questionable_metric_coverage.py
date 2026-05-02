#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
import signal
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.company_state_builder import CompanyStateBuilder


DEFAULT_DATES = [
    "2014-12-31",
    "2018-12-31",
    "2020-12-31",
    "2022-12-31",
    "2024-12-31",
]

QUESTIONABLE_METRICS = [
    "liquidity.revolver_undrawn",
    "liquidity.marketable_securities",
    "liquidity.available_for_actions",
    "capital_structure.interest_coverage",
    "capital_structure.current_debt_statement_direct",
    "capital_structure.long_term_debt_statement_direct",
    "capital_structure.interest_expense_statement_direct",
    "capital_structure.net_pension_liability",
    "capital_structure.combined_retirement_liability",
    "capital_structure.retirement_obligation_regime",
]

EXACTISH = {
    "exact",
    "exact_not_applicable",
    "exact_structural_zero",
    "present",
}

DEFAULT_ARTIFACT_CANDIDATES = [
    REPO_ROOT
    / "out"
    / "materialized_feedback_20260405"
    / "company_state_snapshots_asof=2024-12-31.input_layer_v1_smart_normalized_with_sec.feedback_pipeline.jsonl.gz",
    Path(
        "/tmp/consumer_industrial_snapshots_2024_12_31_feedback_20260401/"
        "company_state_snapshots_asof=2024-12-31.input_layer_v1_smart_normalized_with_sec.fix2.debtrepair_v4."
        "feedback_additions_v7_retirement_carryforward_regime.jsonl.gz"
    ),
]


def _default_inputs_root() -> Path:
    tmp_root = Path("/tmp/axiom_v1_inputs")
    return tmp_root if tmp_root.exists() else (REPO_ROOT / "data")


def _default_artifact_path() -> Path:
    for candidate in DEFAULT_ARTIFACT_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_ARTIFACT_CANDIDATES[0]


def _load_company_ids(ids_file: Path | None, artifact: Path | None) -> list[str]:
    if ids_file is not None and ids_file.exists():
        ids = [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]
        if ids:
            return ids
    if artifact is None or not artifact.exists():
        raise FileNotFoundError("Need either --company-ids-file or --artifact")
    opener = gzip.open if artifact.suffix == ".gz" else open
    ids: list[str] = []
    with opener(artifact, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            company_id = str(row.get("company_id") or "").strip()
            if company_id:
                ids.append(company_id)
    return sorted(set(ids))


def _feature_dict(snapshot: Any) :
    features = getattr(snapshot, "features", {}) or {}
    return features


def _build_with_timeout(builder: CompanyStateBuilder, company_id: str, as_of_time: str, timeout_seconds: int) -> Any:
    if timeout_seconds <= 0:
        return builder.build(company_id, as_of_time)

    def _handler(signum: int, frame: Any) -> None:
        raise TimeoutError(f"builder timed out after {timeout_seconds}s")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_seconds)
    try:
        return builder.build(company_id, as_of_time)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _metric_status(feature: dict[str, Any] | None) -> str:
    if not feature:
        return "missing_feature"
    support_mode = feature.get("support_mode")
    value = feature.get("value")
    missing_reason = feature.get("missing_reason")
    quality_flags = feature.get("quality_flags") or []
    fallback_used = feature.get("fallback_used")
    if support_mode:
        return str(support_mode)
    if value is None:
        return str(missing_reason or "missing_value")
    if quality_flags or fallback_used:
        return "present_proxy"
    return "present"


def _is_available_status(status: str) -> bool:
    return status not in {
        "build_failed",
        "unsupported",
        "component_unavailable",
        "not_disclosed",
        "missing_feature",
        "missing_value",
        "unavailable",
        "unsupported_for_archetype",
    }


def _build_builder(repo_root: Path, inputs_root: Path, facts_years: list[int]) -> CompanyStateBuilder:
    policy_path, methodology_registry_path, input_source_registry_path = _ensure_local_registry_files()
    facts_path = repo_root / "data" / "inputs_layer" / "facts_asof_2026"
    raw_ts_path = inputs_root / "raw_timeseries.parquet"
    taxonomy_path = inputs_root / "fundamentals_all.parquet"
    if not facts_path.exists():
        facts_path = repo_root / "data" / "inputs_layer" / "facts_asof_2026.parquet"
    if not raw_ts_path.exists():
        raw_ts_path = repo_root / "data" / "inputs_layer" / "raw_timeseries.parquet"
    if not taxonomy_path.exists():
        taxonomy_path = repo_root / "data" / "refinitiv" / "fundamentals_all.parquet"
    return CompanyStateBuilder(
        raw_timeseries_path=raw_ts_path,
        macro_timeseries_path=repo_root / "data" / "inputs_layer" / "raw_timeseries.parquet",
        facts_path=facts_path,
        taxonomy_reference_path=taxonomy_path,
        entity_graph_path=repo_root / "data" / "inputs_layer" / "entity_graph.parquet",
        entity_identifier_path=repo_root / "data" / "inputs_layer" / "entity_identifier.parquet",
        entity_table_path=repo_root / "data" / "inputs_layer" / "entity.parquet",
        historical_backfill_mode=True,
        companyfacts_root=repo_root / "data" / "sec" / "companyfacts",
        enable_market_relevant_smart_normalized_inputs=True,
        metric_policy_path=policy_path,
        methodology_registry_path=methodology_registry_path,
        input_source_registry_path=input_source_registry_path,
        skip_timeseries=True,
        skip_macro=True,
        skip_events=True,
        skip_peer_context=True,
        cache_facts=True,
        cache_ownership=True,
        cache_ratings=True,
        facts_years=facts_years,
    )


