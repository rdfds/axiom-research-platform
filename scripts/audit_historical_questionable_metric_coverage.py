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


def _feature_dict(snapshot: Any) -> dict[str, dict[str, Any]]:
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


def _ensure_local_registry_files() -> tuple[Path, Path, Path]:
    policy_path = Path("/tmp/historical_metric_policy_minimal.json")
    methodology_registry_path = Path("/tmp/historical_metric_methodology_registry_minimal.json")
    input_source_registry_path = Path("/tmp/historical_company_state_input_source_registry_minimal.json")

    if not policy_path.exists():
        metrics = {
            metric_id: {
                "label": metric_id,
                "market_owner": "axiom",
                "primary_source_basis": "public_filing_or_market_data",
                "default_applicability": "primary",
                "archetype_applicability": {},
            }
            for metric_id in [
                "liquidity.usable_cash",
                "liquidity.available_for_actions",
                "capital_structure.total_debt",
                "capital_structure.net_debt",
                "capital_structure.gross_leverage",
                "capital_structure.net_leverage",
                "capital_structure.interest_coverage",
                "capital_structure.fixed_charge_coverage",
                "capital_structure.current_debt_statement_direct",
                "capital_structure.long_term_debt_statement_direct",
                "capital_structure.interest_expense_statement_direct",
                "capital_structure.net_pension_liability",
                "capital_structure.other_postretirement_benefit_liability",
                "capital_structure.combined_retirement_liability",
                "capital_structure.debt_like_obligations_normalized",
                "capital_structure.net_debt_including_retirement",
                "capital_structure.gross_leverage_including_retirement",
                "capital_structure.net_leverage_including_retirement",
            ]
        }
        policy = {
            "policy_id": "historical_metric_policy_minimal_v1",
            "version": 1,
            "primary_credit_anchor": "moodys_primary_v1",
            "taxonomy": {
                "sector_field_candidates": [],
                "subsector_field_candidates": [],
                "archetypes": {
                    "generic_corporate": {"rules": {}},
                    "lease_heavy": {
                        "fingerprints": {"lease_liability_to_reported_debt_min": 0.5},
                        "rules": {"lease_adjusted_metrics": True},
                    },
                    "financial_institution": {"rules": {}},
                },
                "issuer_overrides": {},
            },
            "metrics": metrics,
        }
        policy_path.write_text(json.dumps(policy, indent=2))

    if not methodology_registry_path.exists():
        methodology_registry = {
            "registry_id": "historical_metric_methodology_registry_minimal_v1",
            "version": 1,
            "canonical_owners": {"axiom": {"name": "Axiom"}},
            "metrics": {},
        }
        methodology_registry_path.write_text(json.dumps(methodology_registry, indent=2))

    if not input_source_registry_path.exists():
        input_source_registry = {
            "registry_id": "historical_company_state_input_source_registry_minimal_v1",
            "version": "1.0.0",
            "owners": {},
            "metrics": {},
        }
        input_source_registry_path.write_text(json.dumps(input_source_registry, indent=2))

    return policy_path, methodology_registry_path, input_source_registry_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--company-ids-file", type=Path, default=Path("/tmp/historical_metric_audit_company_ids.txt"))
    parser.add_argument(
        "--artifact",
        type=Path,
        default=_default_artifact_path(),
    )
    parser.add_argument("--dates", default=",".join(DEFAULT_DATES))
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=12)
    args = parser.parse_args()

    repo_root = REPO_ROOT
    inputs_root = _default_inputs_root()
    dates = [part.strip() for part in args.dates.split(",") if part.strip()]
    company_ids = _load_company_ids(args.company_ids_file, args.artifact)
    if args.limit is not None:
        company_ids = company_ids[: args.limit]

    summary: dict[str, Any] = {
        "dates": dates,
        "company_count": len(company_ids),
        "metrics": QUESTIONABLE_METRICS,
        "by_date": {},
    }

    for as_of_date in dates:
        year = int(as_of_date[:4])
        facts_years = [y for y in range(max(2005, year - 2), year + 1)]
        builder = _build_builder(repo_root, inputs_root, facts_years)
        metric_counters: dict[str, Counter[str]] = {metric: Counter() for metric in QUESTIONABLE_METRICS}
        regime_examples: dict[str, list[str]] = defaultdict(list)
        failures: list[dict[str, str]] = []

        for index, company_id in enumerate(company_ids, start=1):
            try:
                snapshot = _build_with_timeout(
                    builder,
                    company_id,
                    f"{as_of_date}T00:00:00Z",
                    args.timeout_seconds,
                )
                features = _feature_dict(snapshot)
                for metric in QUESTIONABLE_METRICS:
                    feature = features.get(metric)
                    status = _metric_status(feature)
                    metric_counters[metric][status] += 1
                    if metric == "capital_structure.retirement_obligation_regime" and feature and feature.get("value") is not None:
                        regime = str(feature["value"])
                        if len(regime_examples[regime]) < 5:
                            regime_examples[regime].append(company_id)
            except Exception as exc:
                failures.append({"company_id": company_id, "error": type(exc).__name__})
                for metric in QUESTIONABLE_METRICS:
                    metric_counters[metric]["build_failed"] += 1
            if index % 20 == 0:
                print(f"[{as_of_date}] processed {index}/{len(company_ids)}")

        by_metric: dict[str, Any] = {}
        for metric, counter in metric_counters.items():
            available = sum(count for status, count in counter.items() if _is_available_status(status))
            exactish = sum(count for status, count in counter.items() if status in EXACTISH)
            by_metric[metric] = {
                "status_counts": dict(counter),
                "available_count": available,
                "available_pct": round(100.0 * available / len(company_ids), 2) if company_ids else 0.0,
                "exactish_count": exactish,
                "exactish_pct": round(100.0 * exactish / len(company_ids), 2) if company_ids else 0.0,
            }

        summary["by_date"][as_of_date] = {
            "facts_years": facts_years,
            "metrics": by_metric,
            "retirement_regime_examples": dict(regime_examples),
            "failure_count": len(failures),
            "failure_examples": failures[:20],
        }

    args.out_json.write_text(json.dumps(summary, indent=2))

    lines = [
        "# Historical Questionable Metric Coverage Audit",
        "",
        f"- Company count: `{len(company_ids)}`",
        f"- Dates: `{', '.join(dates)}`",
        "",
    ]
    for as_of_date in dates:
        lines.append(f"## {as_of_date}")
        date_summary = summary["by_date"][as_of_date]
        lines.append(f"- Build failures: `{date_summary['failure_count']}`")
        for metric in QUESTIONABLE_METRICS:
            metric_summary = date_summary["metrics"][metric]
            lines.append(
                f"- `{metric}`: available `{metric_summary['available_count']}/{len(company_ids)}` "
                f"({metric_summary['available_pct']}%), exactish `{metric_summary['exactish_count']}/{len(company_ids)}` "
                f"({metric_summary['exactish_pct']}%), statuses `{metric_summary['status_counts']}`"
            )
        lines.append("")
    args.out_md.write_text("\n".join(lines))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
