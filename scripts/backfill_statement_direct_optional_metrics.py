#!/usr/bin/env python3
"""Backfill optional statement-direct metrics into the v1 input-layer artifact.

These metrics come from the local fact registry rather than the provider sidecar.
They are useful additions, but they are not part of the tightest universal core
because coverage is materially lower than the provider-direct baseline.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import duckdb
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backfill_input_layer_v1_metrics as core  # noqa: E402

MAX_SEC_FACT_AGE_DAYS = 550
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_COMPANYFACTS_ROOT = REPO_ROOT / "data" / "sec" / "companyfacts"
DEBT_COMPONENT_ALIGNMENT_MAX_GAP_DAYS = 10
STATEMENT_DEBT_REPAIR_MAX_GAP_DAYS = 45
STATEMENT_DEBT_EXACT_MAX_AGE_DAYS = 130
STATEMENT_DEBT_MATCH_TOLERANCE = 1.0
DEPRECIATION_TTM_CONCEPT_GROUPS = [
    ["DepreciationAmortizationAndAccretionNet"],
    ["DepreciationDepletionAndAmortization"],
    ["DepreciationAndAmortization"],
    ["Depreciation"],
    ["Depreciation", "AmortizationOfIntangibleAssets"],
]
INTEREST_EXPENSE_TTM_EXACT_CONCEPTS = [
    "InterestExpense",
]

STATEMENT_FACT_SPECS = {
    "liquidity.cash_and_equivalents_statement_direct": {
        "fact_type": "financial.cash",
        "unit": "usd",
    },
    "capital_structure.current_debt_statement_direct": {
        "fact_type": "financial.debt_current",
        "unit": "usd",
    },
    "capital_structure.long_term_debt_statement_direct": {
        "fact_type": "financial.debt_long_term",
        "unit": "usd",
    },
    "operating.ebit_statement_direct": {
        "fact_type": "financial.ebit",
        "unit": "usd",
    },
    "capital_structure.interest_expense_statement_direct": {
        "fact_type": "financial.interest_expense",
        "unit": "usd",
    },
}


class _CompanyProcessingTimeout(RuntimeError):
    """Raised when a single-company optional-metric build exceeds the allowed timeout."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-path", required=True, help="Input snapshot JSONL path")
    parser.add_argument("--facts-path", required=True, help="Local fact registry parquet")
    parser.add_argument(
        "--entity-batch-size",
        type=int,
        default=128,
        help="Number of companies to enrich per fact-registry query batch.",
    )
    parser.add_argument(
        "--company-processing-timeout-seconds",
        type=float,
        default=15.0,
        help="Fail open on a single company if statement-optional enrichment exceeds this timeout. Use 0 to disable.",
    )
    parser.add_argument(
        "--companyfacts-root",
        help="Optional SEC companyfacts folder for EBITDA repair. Defaults to the local canonical companyfacts root when present.",
    )
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--summary-out", help="Optional summary JSON path")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_snapshot_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_row_batches(rows: Iterable[Dict[str, Any]], batch_size: int) -> Iterable[list[Dict[str, Any]]]:
    batch: list[Dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _fact_parquet_source_arg(facts_path: Path, as_of_time: str) -> str:
    if facts_path.is_file():
        return f"'{facts_path.as_posix()}'"

    as_of_date = pd.Timestamp(as_of_time).tz_convert("UTC").normalize()
    lookback_start = (as_of_date - pd.Timedelta(days=MAX_SEC_FACT_AGE_DAYS + 365)).year
    candidate_paths: list[Path] = []
    for year in range(int(lookback_start), int(as_of_date.year) + 1):
        part = facts_path / f"year={year}" / "part.parquet"
        if part.exists():
            candidate_paths.append(part)
    if not candidate_paths:
        fallback = sorted(facts_path.glob("year=*/part.parquet"))
        candidate_paths = fallback if fallback else [facts_path]
    quoted = ",".join(f"'{path.as_posix()}'" for path in candidate_paths)
    return f"[{quoted}]"


def _parse_iso_date(value: Any):
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def _iter_component_ends(component_breakdown: Any) -> list[str]:
    ends: list[str] = []
    if isinstance(component_breakdown, dict):
        if component_breakdown.get("end"):
            ends.append(component_breakdown["end"])
        for child in component_breakdown.values():
            ends.extend(_iter_component_ends(child))
    elif isinstance(component_breakdown, list):
        for child in component_breakdown:
            ends.extend(_iter_component_ends(child))
    return ends


def _selected_debt_component_gap_days(component_breakdown: dict[str, Any] | None) -> int | None:
    if not isinstance(component_breakdown, dict):
        return None
    ends = []
    for key in (
        "combined_debt",
        "current",
        "noncurrent",
        "short_term_borrowings",
        "current_statement_debt",
        "long_term_statement_debt",
    ):
        for end_text in _iter_component_ends(component_breakdown.get(key)):
            parsed = _parse_iso_date(end_text)
            if parsed is not None:
                ends.append(parsed)
    if len(ends) < 2:
        return 0 if ends else None
    return (max(ends) - min(ends)).days


def _statement_fact_end_date(component_breakdown: dict[str, Any] | None):
    if not isinstance(component_breakdown, dict):
        return None
    return _parse_iso_date(component_breakdown.get("end")) or _parse_iso_date(component_breakdown.get("effective_at"))


def _normalize_statement_fact_candidate(row: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    end_dt = _parse_iso_date(row.get("effective_at"))
    fact_time = row.get("fact_time")
    if end_dt is None:
        return None
    fact_time_text = None if fact_time is None else str(fact_time)
    return {
        "value": float(row["fact_value"]),
        "end_dt": end_dt,
        "source_type": row.get("source_type"),
        "source_id": row.get("source_id"),
        "raw_pointer": row.get("raw_pointer"),
        "meta": {
            "fact_type": row.get("fact_type"),
            "fact_id": row.get("fact_id"),
            "source_id": row.get("source_id"),
            "source_type": row.get("source_type"),
            "raw_pointer": row.get("raw_pointer"),
            "registry_unit": row.get("unit"),
            "effective_at": end_dt.isoformat(),
            "end": end_dt.isoformat(),
            "fact_time": fact_time_text,
            "formula": "statement_direct_fact",
        },
    }


def _best_statement_debt_pair(
    current_candidates: list[dict[str, Any]] | None,
    long_term_candidates: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int | None]:
    best_pair = None
    best_key = None
    for current_row in current_candidates or []:
        current = _normalize_statement_fact_candidate(current_row)
        if current is None:
            continue
        for long_term_row in long_term_candidates or []:
            long_term = _normalize_statement_fact_candidate(long_term_row)
            if long_term is None:
                continue
            if current["source_type"] != long_term["source_type"]:
                continue
            gap_days = abs((current["end_dt"] - long_term["end_dt"]).days)
            if gap_days > STATEMENT_DEBT_REPAIR_MAX_GAP_DAYS:
                continue
            pair_key = (
                max(current["end_dt"], long_term["end_dt"]),
                -gap_days,
                1 if current["source_type"] == "sec_edgar_xbrl" else 0,
            )
            if best_key is None or pair_key > best_key:
                best_key = pair_key
                best_pair = (current, long_term, gap_days)
    if best_pair is None:
        return None, None, None
    return best_pair


def _load_companyfacts(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def _candidate_units_map(companyfacts: dict, concept_name: str) -> dict | None:
    for taxonomy in ("us-gaap", "dei", "ifrs-full"):
        facts = (companyfacts.get("facts") or {}).get(taxonomy) or {}
        if concept_name in facts:
            return facts[concept_name].get("units") or {}
    return None


def _collect_duration_entries(companyfacts: dict, concept_name: str, as_of_date: str) -> list[dict[str, Any]]:
    units_map = _candidate_units_map(companyfacts, concept_name)
    if not units_map:
        return []
    as_of_dt = _parse_iso_date(as_of_date)
    rows = []
    for unit, entries in units_map.items():
        if unit.upper() != "USD":
            continue
        for entry in entries:
            start_dt = _parse_iso_date(entry.get("start"))
            end_dt = _parse_iso_date(entry.get("end"))
            filed_dt = _parse_iso_date(entry.get("filed"))
            value = entry.get("val")
            if start_dt is None or end_dt is None or value is None:
                continue
            if end_dt > as_of_dt:
                continue
            if filed_dt is not None and filed_dt > as_of_dt:
                continue
            if (as_of_dt - end_dt).days > MAX_SEC_FACT_AGE_DAYS:
                continue
            duration_days = max(1, (end_dt - start_dt).days + 1)
            rows.append(
                {
                    "concept": concept_name,
                    "start": start_dt,
                    "end": end_dt,
                    "filed": filed_dt or end_dt,
                    "value": float(value),
                    "fy": entry.get("fy"),
                    "fp": entry.get("fp"),
                    "frame": entry.get("frame"),
                    "form": entry.get("form"),
                    "duration_days": duration_days,
                }
            )
    rows.sort(key=lambda item: (item["end"], item["filed"], item["duration_days"]))
    return rows


def _compute_ttm_from_concept(companyfacts: dict, concept_name: str, as_of_date: str) -> tuple[float | None, dict[str, Any] | None]:
    entries = _collect_duration_entries(companyfacts, concept_name, as_of_date)
    if not entries:
        return None, None

    latest = max(entries, key=lambda item: (item["end"], item["filed"], item["duration_days"]))
    latest_fp = str(latest.get("fp") or "").upper()
    if latest_fp == "FY" or latest["duration_days"] >= 300:
        return latest["value"], {
            "concept": concept_name,
            "mode": "latest_fy",
            "end": latest["end"].isoformat(),
            "filed": latest["filed"].isoformat(),
            "fy": latest.get("fy"),
            "fp": latest.get("fp"),
            "frame": latest.get("frame"),
            "form": latest.get("form"),
            "formula": "latest_fiscal_year_value",
        }

    if latest_fp not in {"Q1", "Q2", "Q3"}:
        return None, None

    current_fy = latest.get("fy")
    if current_fy is None:
        return None, None
    try:
        prior_fy = int(current_fy) - 1
    except Exception:  # noqa: BLE001
        return None, None

    annual = None
    prior_same = None
    for entry in entries:
        if entry.get("fy") == prior_fy and str(entry.get("fp") or "").upper() == "FY":
            if annual is None or (entry["end"], entry["filed"], entry["duration_days"]) > (annual["end"], annual["filed"], annual["duration_days"]):
                annual = entry
        if entry.get("fy") == prior_fy and str(entry.get("fp") or "").upper() == latest_fp:
            if prior_same is None or (entry["end"], entry["filed"], entry["duration_days"]) > (prior_same["end"], prior_same["filed"], prior_same["duration_days"]):
                prior_same = entry

    if annual is None or prior_same is None:
        return None, None

    return float(latest["value"] + annual["value"] - prior_same["value"]), {
        "concept": concept_name,
        "mode": "ytd_plus_prior_fy_minus_prior_ytd",
        "latest": {
            "end": latest["end"].isoformat(),
            "filed": latest["filed"].isoformat(),
            "fy": latest.get("fy"),
            "fp": latest.get("fp"),
            "frame": latest.get("frame"),
            "form": latest.get("form"),
            "value": latest["value"],
        },
        "prior_fy": {
            "end": annual["end"].isoformat(),
            "filed": annual["filed"].isoformat(),
            "fy": annual.get("fy"),
            "fp": annual.get("fp"),
            "frame": annual.get("frame"),
            "form": annual.get("form"),
            "value": annual["value"],
        },
        "prior_same_period": {
            "end": prior_same["end"].isoformat(),
            "filed": prior_same["filed"].isoformat(),
            "fy": prior_same.get("fy"),
            "fp": prior_same.get("fp"),
            "frame": prior_same.get("frame"),
            "form": prior_same.get("form"),
            "value": prior_same["value"],
        },
        "formula": "latest_ytd + prior_fy - prior_same_period_ytd",
    }


def _ebitda_repair_from_statement_ebit(
    *,
    current_node: dict[str, Any],
    statement_ebit_node: dict[str, Any] | None,
    companyfacts: dict | None,
    companyfacts_path: Path | None,
    as_of_time: str,
    computed_at: str,
) -> dict[str, Any] | None:
    if current_node.get("support_mode") != "unsupported":
        return None
    if current_node.get("missing_reason") != "sec_operating_income_ttm_unavailable":
        return None
    if not statement_ebit_node or statement_ebit_node.get("support_mode") != "exact" or statement_ebit_node.get("value") is None:
        return None
    if companyfacts is None or companyfacts_path is None:
        return None

    depreciation_value = None
    depreciation_meta = None
    depreciation_support_mode = "exact"
    quality_flags: list[str] = []
    for concept_group in DEPRECIATION_TTM_CONCEPT_GROUPS:
        if len(concept_group) == 1:
            depreciation_value, depreciation_meta = _compute_ttm_from_concept(companyfacts, concept_group[0], as_of_time[:10])
            if depreciation_value is not None:
                if concept_group[0] == "Depreciation":
                    depreciation_support_mode = "proxy_missing_component"
                    quality_flags.append("partial_depreciation_without_full_amortization")
                break
        else:
            parts = []
            parts_meta = []
            for concept_name in concept_group:
                part_value, part_meta = _compute_ttm_from_concept(companyfacts, concept_name, as_of_time[:10])
                if part_value is None:
                    parts = []
                    break
                parts.append(part_value)
                parts_meta.append(part_meta)
            if parts:
                depreciation_value = float(sum(parts))
                depreciation_meta = {
                    "mode": "sum_concepts",
                    "components": parts_meta,
                    "formula": "sum_component_ttm_values",
                }
                break

    if depreciation_value is None:
        return None

    repaired = dict(current_node)
    repaired["value"] = float(statement_ebit_node["value"] + depreciation_value)
    repaired["computed_at"] = computed_at
    repaired["confidence"] = 1.0
    repaired["missing_reason"] = None
    repaired["support_mode"] = depreciation_support_mode
    repaired["primary_source_basis"] = "statement_direct_plus_sec_companyfacts"
    repaired["input_source_classification"] = "statement_direct_plus_sec_companyfacts"
    repaired["input_layer_bucket_reason"] = "statement_ebit_plus_sec_dna"
    repaired["provenance"] = list(statement_ebit_node.get("provenance") or []) + [
        {
            "artifact_type": "SecCompanyFacts",
            "artifact_id": f"sec_companyfacts:{companyfacts_path.name}",
            "source": str(companyfacts_path),
            "published_at": as_of_time,
            "ingested_at": computed_at,
            "hash": None,
        }
    ]
    repaired["component_breakdown"] = {
        "mode": "statement_ebit_plus_depreciation_amortization",
        "statement_ebit": statement_ebit_node.get("component_breakdown"),
        "depreciation_amortization": depreciation_meta,
        "formula": "statement_ebit + depreciation_amortization_ttm",
    }
    repaired["quality_flags"] = quality_flags or None
    return repaired


def _interest_expense_repair_from_companyfacts(
    *,
    current_node: dict[str, Any],
    companyfacts: dict | None,
    companyfacts_path: Path | None,
    as_of_time: str,
    computed_at: str,
) -> dict[str, Any] | None:
    if current_node.get("support_mode") != "unsupported":
        return None
    if current_node.get("missing_reason") != "statement_fact_unavailable":
        return None
    if companyfacts is None or companyfacts_path is None:
        return None

    repaired_value = None
    repaired_meta = None
    repaired_concept = None
    for concept_name in INTEREST_EXPENSE_TTM_EXACT_CONCEPTS:
        repaired_value, repaired_meta = _compute_ttm_from_concept(companyfacts, concept_name, as_of_time[:10])
        if repaired_value is not None:
            repaired_concept = concept_name
            break

    if repaired_value is None or repaired_meta is None or repaired_concept is None:
        return None

    repaired = dict(current_node)
    repaired["value"] = float(repaired_value)
    repaired["computed_at"] = computed_at
    repaired["confidence"] = 1.0
    repaired["missing_reason"] = None
    repaired["support_mode"] = "exact"
    repaired["primary_source_basis"] = "sec_companyfacts"
    repaired["input_source_classification"] = "sec_companyfacts"
    repaired["input_layer_bucket_reason"] = "sec_companyfacts_ttm_interest_expense"
    repaired["provenance"] = [
        {
            "artifact_type": "SecCompanyFacts",
            "artifact_id": f"sec_companyfacts:{companyfacts_path.name}",
            "source": str(companyfacts_path),
            "published_at": as_of_time,
            "ingested_at": computed_at,
            "hash": None,
        }
    ]
    repaired["component_breakdown"] = {
        "mode": "companyfacts_ttm_interest_expense",
        "concept": repaired_concept,
        "ttm_context": repaired_meta,
        "formula": "companyfacts_interest_expense_ttm",
    }
    repaired["quality_flags"] = None
    return repaired


def _feature_template(
    *,
    metric_name: str,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
    support_mode: str,
    value: Any,
    unit: str,
    missing_reason: str | None,
    component_breakdown: Dict[str, Any] | None,
    quality_flags: list[str] | None,
) -> Dict[str, Any]:
    return {
        "name": metric_name,
        "value": value,
        "unit": unit,
        "computed_at": computed_at,
        "as_of_time": as_of_time,
        "window": None,
        "confidence": 1.0 if value is not None else None,
        "provenance": [
            {
                "artifact_type": "StatementFact",
                "artifact_id": f"statement_direct:{Path(provenance_source).name}",
                "source": provenance_source,
                "published_at": as_of_time,
                "ingested_at": computed_at,
                "hash": None,
            }
        ],
        "missing_reason": missing_reason,
        "fallback_used": None,
        "metric_policy_id": None,
        "market_owner": None,
        "primary_source_basis": "statement_direct",
        "methodology_registry_id": None,
        "methodology_metric_id": None,
        "canonical_owner_id": None,
        "canonical_owner_name": None,
        "canonical_classification": None,
        "market_layer_status": None,
        "current_alignment_status": None,
        "primary_source_document_id": None,
        "recommended_metric_name": None,
        "input_source_registry_id": None,
        "input_source_owner_id": None,
        "input_source_owner_name": None,
        "input_source_classification": "statement_direct",
        "input_source_formula_basis": None,
        "input_source_alignment_status": "aligned",
        "input_source_document_ids": None,
        "definition_requirement": None,
        "definition_requirement_reason": None,
        "methodology_execution_decision": None,
        "methodology_execution_reason": None,
        "input_layer_bucket": "reference",
        "input_layer_bucket_reason": "statement_fact_registry",
        "strict_market_defined": None,
        "archetype": None,
        "sector": None,
        "subsector": None,
        "override_level_applied": None,
        "support_mode": support_mode,
        "applicability_status": None,
        "component_breakdown": component_breakdown,
        "quality_flags": quality_flags,
        "view_type": None,
    }


def _is_mixed_period_exact_debt(node: dict[str, Any] | None) -> bool:
    if not node or node.get("support_mode") != "exact":
        return False
    breakdown = node.get("component_breakdown") or {}
    gap_days = _selected_debt_component_gap_days(breakdown)
    return gap_days is not None and gap_days > DEBT_COMPONENT_ALIGNMENT_MAX_GAP_DAYS


def _values_match(left: Any, right: Any, tolerance: float = STATEMENT_DEBT_MATCH_TOLERANCE) -> bool:
    if left is None or right is None:
        return False
    return abs(float(left) - float(right)) <= tolerance


def _repair_total_debt_from_single_statement_component(
    *,
    current_node: dict[str, Any] | None,
    current_debt_statement_node: dict[str, Any] | None,
    long_term_debt_statement_node: dict[str, Any] | None,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
) -> dict[str, Any] | None:
    if not current_node:
        return None

    prior_breakdown = current_node.get("component_breakdown") or {}
    prior_mode = prior_breakdown.get("mode")
    if prior_mode not in {
        "partial_debt_stack",
        "current_plus_noncurrent_debt",
        "current_plus_noncurrent_debt_plus_short_term_borrowings",
        "short_term_borrowings_only",
    }:
        return None

    if current_node.get("missing_reason") not in {
        "debt_component_missing",
        "debt_component_period_mismatch",
        "long_term_debt_components_missing",
    }:
        return None

    total_debt_value = current_node.get("value")
    if total_debt_value is None:
        return None

    current_exact = (
        current_debt_statement_node is not None
        and current_debt_statement_node.get("support_mode") == "exact"
        and current_debt_statement_node.get("value") is not None
    )
    long_term_exact = (
        long_term_debt_statement_node is not None
        and long_term_debt_statement_node.get("support_mode") == "exact"
        and long_term_debt_statement_node.get("value") is not None
    )

    matched_component = None
    inferred_zero_component = None
    matched_value = None

    if (
        current_exact
        and (long_term_debt_statement_node is None or long_term_debt_statement_node.get("support_mode") == "unsupported")
        and _values_match(total_debt_value, current_debt_statement_node.get("value"))
    ):
        matched_component = current_debt_statement_node
        matched_value = float(current_debt_statement_node["value"])
        inferred_zero_component = "capital_structure.long_term_debt_statement_direct"
    elif (
        long_term_exact
        and (current_debt_statement_node is None or current_debt_statement_node.get("support_mode") == "unsupported")
        and _values_match(total_debt_value, long_term_debt_statement_node.get("value"))
    ):
        matched_component = long_term_debt_statement_node
        matched_value = float(long_term_debt_statement_node["value"])
        inferred_zero_component = "capital_structure.current_debt_statement_direct"

    if matched_component is None or inferred_zero_component is None or matched_value is None:
        return None

    as_of_date = _parse_iso_date(as_of_time)
    matched_end = _statement_fact_end_date(matched_component.get("component_breakdown"))
    age_days = None if as_of_date is None or matched_end is None else (as_of_date - matched_end).days
    support_mode = "exact"
    missing_reason = None
    quality_flags = None
    if age_days is not None and age_days > STATEMENT_DEBT_EXACT_MAX_AGE_DAYS:
        support_mode = "proxy_missing_component"
        missing_reason = "statement_debt_pair_stale"
        quality_flags = ["statement_debt_pair_stale"]

    repaired = core._build_metric_from_value(
        metric_name="capital_structure.total_debt_provider_direct",
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit="usd",
        value=matched_value,
        support_mode=support_mode,
        missing_reason=missing_reason,
        component_breakdown={
            "mode": "statement_direct_single_component_total_debt",
            "matched_statement_debt": matched_component.get("component_breakdown"),
            "inferred_zero_component": inferred_zero_component,
            "repaired_prior_breakdown": prior_breakdown,
            "age_days": age_days,
            "formula": (
                "exact_current_debt_statement_direct + 0_inferred_long_term_debt"
                if inferred_zero_component == "capital_structure.long_term_debt_statement_direct"
                else "exact_long_term_debt_statement_direct + 0_inferred_current_debt"
            ),
        },
        quality_flags=quality_flags,
        primary_source_basis="statement_direct",
        provenance_artifact_type="StatementFact",
        input_layer_bucket_reason="statement_fact_registry",
    )
    repaired["input_source_classification"] = "statement_direct_repair"
    repaired["input_layer_bucket_reason"] = "statement_direct_debt_repair"
    return repaired


def _repair_total_debt_from_statement_split(
    *,
    current_node: dict[str, Any] | None,
    current_debt_statement_node: dict[str, Any] | None,
    long_term_debt_statement_node: dict[str, Any] | None,
    current_debt_statement_candidates: list[dict[str, Any]] | None,
    long_term_debt_statement_candidates: list[dict[str, Any]] | None,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
) -> dict[str, Any] | None:
    current_is_mixed_period = _is_mixed_period_exact_debt(current_node)
    prior_breakdown = (current_node or {}).get("component_breakdown") or {}
    prior_has_capital_lease_overlap = bool(prior_breakdown.get("capital_lease_overlap_detected"))
    as_of_date = _parse_iso_date(as_of_time)
    aligned_current_fact, aligned_long_term_fact, aligned_gap_days = _best_statement_debt_pair(
        current_debt_statement_candidates,
        long_term_debt_statement_candidates,
    )

    if aligned_current_fact is not None and aligned_long_term_fact is not None:
        total_value = float(aligned_current_fact["value"] + aligned_long_term_fact["value"])
        latest_end_dt = max(aligned_current_fact["end_dt"], aligned_long_term_fact["end_dt"])
        age_days = None if as_of_date is None else (as_of_date - latest_end_dt).days
        support_mode = "exact"
        missing_reason = None
        quality_flags = None
        if prior_has_capital_lease_overlap:
            support_mode = "proxy_missing_component"
            missing_reason = "finance_lease_adjustment_unavailable"
            quality_flags = ["finance_lease_adjustment_unavailable"]
        elif age_days is not None and age_days > STATEMENT_DEBT_EXACT_MAX_AGE_DAYS:
            support_mode = "proxy_missing_component"
            missing_reason = "statement_debt_pair_stale"
            quality_flags = ["statement_debt_pair_stale"]
        repaired = core._build_metric_from_value(
            metric_name="capital_structure.total_debt_provider_direct",
            as_of_time=as_of_time,
            computed_at=computed_at,
            provenance_source=provenance_source,
            unit="usd",
            value=total_value,
            support_mode=support_mode,
            missing_reason=missing_reason,
            component_breakdown={
                "mode": "statement_direct_current_plus_noncurrent_debt",
                "current_statement_debt": aligned_current_fact["meta"],
                "long_term_statement_debt": aligned_long_term_fact["meta"],
                "repaired_prior_breakdown": prior_breakdown,
                "capital_lease_overlap_detected": prior_has_capital_lease_overlap,
                "alignment_gap_days": aligned_gap_days,
                "age_days": age_days,
                "formula": "current_debt_statement_direct + long_term_debt_statement_direct",
            },
            quality_flags=quality_flags,
            primary_source_basis="statement_direct",
            provenance_artifact_type="StatementFact",
            input_layer_bucket_reason="statement_fact_registry",
        )
        repaired["input_source_classification"] = "statement_direct_repair"
        repaired["input_layer_bucket_reason"] = "statement_direct_debt_repair"
        return repaired

    repaired_single_component = _repair_total_debt_from_single_statement_component(
        current_node=current_node,
        current_debt_statement_node=current_debt_statement_node,
        long_term_debt_statement_node=long_term_debt_statement_node,
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
    )
    if repaired_single_component is not None:
        return repaired_single_component

    if not current_debt_statement_node or not long_term_debt_statement_node:
        if not current_is_mixed_period:
            return None
        downgraded = dict(current_node or {})
        downgraded["support_mode"] = "proxy_missing_component"
        downgraded["missing_reason"] = "debt_component_period_mismatch"
        downgraded["computed_at"] = computed_at
        flags = list(downgraded.get("quality_flags") or [])
        if "debt_component_period_mismatch" not in flags:
            flags.append("debt_component_period_mismatch")
        downgraded["quality_flags"] = flags
        return downgraded

    if current_debt_statement_node.get("support_mode") != "exact" or long_term_debt_statement_node.get("support_mode") != "exact":
        if not current_is_mixed_period:
            return None
        downgraded = dict(current_node or {})
        downgraded["support_mode"] = "proxy_missing_component"
        downgraded["missing_reason"] = "debt_component_period_mismatch"
        downgraded["computed_at"] = computed_at
        flags = list(downgraded.get("quality_flags") or [])
        if "debt_component_period_mismatch" not in flags:
            flags.append("debt_component_period_mismatch")
        downgraded["quality_flags"] = flags
        return downgraded

    current_debt_value = current_debt_statement_node.get("value")
    long_term_debt_value = long_term_debt_statement_node.get("value")
    if current_debt_value is None or long_term_debt_value is None:
        if not current_is_mixed_period:
            return None
        downgraded = dict(current_node or {})
        downgraded["support_mode"] = "proxy_missing_component"
        downgraded["missing_reason"] = "debt_component_period_mismatch"
        downgraded["computed_at"] = computed_at
        flags = list(downgraded.get("quality_flags") or [])
        if "debt_component_period_mismatch" not in flags:
            flags.append("debt_component_period_mismatch")
        downgraded["quality_flags"] = flags
        return downgraded

    if current_node and current_node.get("support_mode") == "exact" and not current_is_mixed_period:
        return None

    current_end = _statement_fact_end_date(current_debt_statement_node.get("component_breakdown"))
    long_term_end = _statement_fact_end_date(long_term_debt_statement_node.get("component_breakdown"))
    age_days = None
    if as_of_date is not None and current_end is not None and long_term_end is not None:
        age_days = (as_of_date - max(current_end, long_term_end)).days

    support_mode = "proxy_missing_component" if prior_has_capital_lease_overlap else "exact"
    missing_reason = "finance_lease_adjustment_unavailable" if prior_has_capital_lease_overlap else None
    quality_flags = ["finance_lease_adjustment_unavailable"] if prior_has_capital_lease_overlap else None
    if age_days is not None and age_days > STATEMENT_DEBT_EXACT_MAX_AGE_DAYS:
        support_mode = "proxy_missing_component"
        missing_reason = "statement_debt_pair_stale"
        quality_flags = ["statement_debt_pair_stale"]

    repaired = core._build_metric_from_value(
        metric_name="capital_structure.total_debt_provider_direct",
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit="usd",
        value=float(current_debt_value + long_term_debt_value),
        support_mode=support_mode,
        missing_reason=missing_reason,
        component_breakdown={
            "mode": "statement_direct_current_plus_noncurrent_debt",
            "current_statement_debt": current_debt_statement_node.get("component_breakdown"),
            "long_term_statement_debt": long_term_debt_statement_node.get("component_breakdown"),
            "repaired_prior_breakdown": prior_breakdown,
            "capital_lease_overlap_detected": prior_has_capital_lease_overlap,
            "age_days": age_days,
            "formula": "current_debt_statement_direct + long_term_debt_statement_direct",
        },
        quality_flags=quality_flags,
        primary_source_basis="statement_direct",
        provenance_artifact_type="StatementFact",
        input_layer_bucket_reason="statement_fact_registry",
    )
    repaired["input_source_classification"] = "statement_direct_repair"
    repaired["input_layer_bucket_reason"] = "statement_direct_debt_repair"
    return repaired


def _recompute_standardized_debt_metrics(
    *,
    features: dict[str, Any],
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
) -> None:
    total_debt = core._metric_value(features, "capital_structure.total_debt_provider_direct")
    total_debt_support = core._metric_support(features, "capital_structure.total_debt_provider_direct")
    cash_sti = core._metric_value(features, "liquidity.cash_and_short_term_investments_provider_direct")
    cash_sti_support = core._metric_support(features, "liquidity.cash_and_short_term_investments_provider_direct")
    ebitda = core._metric_value(features, "operating.ebitda_ltm_provider_direct")
    ebitda_support = core._metric_support(features, "operating.ebitda_ltm_provider_direct")

    net_debt = None if total_debt is None or cash_sti is None else total_debt - cash_sti

    features["capital_structure.net_debt_standardized"] = core._build_combo_metric(
        metric_name="capital_structure.net_debt_standardized",
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit="usd",
        numerator=net_debt,
        denominator=None,
        extra_components={
            "total_debt_provider_direct": total_debt,
            "cash_and_short_term_investments_provider_direct": cash_sti,
        },
        component_supports={
            "total_debt_provider_direct": total_debt_support,
            "cash_and_short_term_investments_provider_direct": cash_sti_support,
        },
        formula="total_debt_provider_direct - cash_and_short_term_investments_provider_direct",
        allow_numerator_only=True,
    )
    features["capital_structure.gross_leverage_standardized"] = core._build_combo_metric(
        metric_name="capital_structure.gross_leverage_standardized",
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit="x",
        numerator=total_debt,
        denominator=ebitda,
        extra_components={
            "total_debt_provider_direct": total_debt,
            "ebitda_ltm_provider_direct": ebitda,
        },
        component_supports={
            "total_debt_provider_direct": total_debt_support,
            "ebitda_ltm_provider_direct": ebitda_support,
        },
        formula="total_debt_provider_direct / ebitda_ltm_provider_direct",
    )
    features["capital_structure.net_leverage_standardized"] = core._build_combo_metric(
        metric_name="capital_structure.net_leverage_standardized",
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit="x",
        numerator=net_debt,
        denominator=ebitda,
        extra_components={
            "net_debt_standardized": net_debt,
            "ebitda_ltm_provider_direct": ebitda,
        },
        component_supports={
            "net_debt_standardized": features["capital_structure.net_debt_standardized"]["support_mode"],
            "ebitda_ltm_provider_direct": ebitda_support,
        },
        formula="net_debt_standardized / ebitda_ltm_provider_direct",
    )


def _load_statement_fact_rows(
    facts_path: Path,
    entity_ids: list[str],
    as_of_time: str,
) -> list[dict[str, Any]]:
    entity_sql = ",".join(f"'{entity_id}'" for entity_id in sorted(set(entity_ids)))
    fact_types = sorted({spec["fact_type"] for spec in STATEMENT_FACT_SPECS.values()})
    fact_sql = ",".join(f"'{fact_type}'" for fact_type in fact_types)
    as_of_sql = as_of_time.replace("'", "''")
    facts_source_arg = _fact_parquet_source_arg(facts_path, as_of_time)
    query = f"""
        SELECT
            entity_id,
            fact_type,
            fact_value,
            unit,
            fact_id,
            source_id,
            source_type,
            raw_pointer,
            effective_at,
            fact_time,
            ingested_at
        FROM read_parquet({facts_source_arg})
        WHERE entity_id IN ({entity_sql})
          AND fact_type IN ({fact_sql})
          AND valid_from <= TIMESTAMPTZ '{as_of_sql}'
          AND (valid_to IS NULL OR valid_to >= TIMESTAMPTZ '{as_of_sql}')
        ORDER BY entity_id, fact_type, COALESCE(effective_at, fact_time) DESC, ingested_at DESC, fact_id DESC
    """
    return duckdb.sql(query).fetchdf().to_dict(orient="records")


def _split_statement_fact_rows(
    rows: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["entity_id"], row["fact_type"])
        latest.setdefault(key, row)
        grouped.setdefault(key, []).append(row)
    return latest, grouped


@contextmanager
def _company_processing_guard(timeout_seconds: float | None):
    if (
        timeout_seconds is None
        or timeout_seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "setitimer")
    ):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _handle_timeout(signum, frame):  # noqa: ARG001
        raise _CompanyProcessingTimeout(f"company_processing_timeout_after_{timeout_seconds:g}s")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _build_fail_open_optional_metrics(
    *,
    as_of_time: str,
    computed_at: str,
    provenance_source: str,
    error_type: str,
    error_message: str,
) -> dict[str, dict[str, Any]]:
    missing_reason = "company_processing_timeout" if error_type == "company_processing_timeout" else "company_processing_failed"
    error_text = str(error_message).strip()[:240]
    breakdown = {
        "error_type": error_type,
        "error_message": error_text,
    }
    metric_units = {
        **{metric_name: spec["unit"] for metric_name, spec in STATEMENT_FACT_SPECS.items()},
        "capital_structure.interest_expense_statement_direct": "usd",
    }
    nodes: dict[str, dict[str, Any]] = {}
    for metric_name, unit in metric_units.items():
        nodes[metric_name] = _feature_template(
            metric_name=metric_name,
            as_of_time=as_of_time,
            computed_at=computed_at,
            provenance_source=provenance_source,
            support_mode="unsupported",
            value=None,
            unit=unit,
            missing_reason=missing_reason,
            component_breakdown=breakdown,
            quality_flags=["statement_optional_fail_open", error_type],
        )
    return nodes


def main() -> None:
    args = parse_args()
    snapshot_path = Path(args.snapshot_path)
    facts_path = Path(args.facts_path)
    companyfacts_root = Path(args.companyfacts_root) if args.companyfacts_root else (
        DEFAULT_LOCAL_COMPANYFACTS_ROOT if DEFAULT_LOCAL_COMPANYFACTS_ROOT.exists() else None
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    companyfacts_cache: dict[str, dict | None] = {}
    computed_at = _now_iso()
    counters: Counter[str] = Counter()

    rows_by_asof: dict[str, list[Dict[str, Any]]] = {}
    for row in iter_snapshot_rows(snapshot_path):
        rows_by_asof.setdefault(row["as_of_time"], []).append(row)

    saw_any_rows = bool(rows_by_asof)
    with out_path.open("w", buffering=1) as out_handle:
        for batch_as_of_time, asof_rows in rows_by_asof.items():
            batch_entity_ids = [row["company_id"] for row in asof_rows]
            statement_fact_rows = _load_statement_fact_rows(facts_path, batch_entity_ids, batch_as_of_time)
            latest_facts, debt_fact_candidates = _split_statement_fact_rows(statement_fact_rows)

            for row_batch in _iter_row_batches(asof_rows, max(1, int(args.entity_batch_size))):
                for row in row_batch:
                    entity_id = row["company_id"]
                    features = row.setdefault("features", {})
                    as_of_time = row["as_of_time"]
                    try:
                        with _company_processing_guard(args.company_processing_timeout_seconds):
                            for metric_name, spec in STATEMENT_FACT_SPECS.items():
                                fact = latest_facts.get((entity_id, spec["fact_type"]))
                                if fact is None or fact.get("fact_value") is None:
                                    node = _feature_template(
                                        metric_name=metric_name,
                                        as_of_time=as_of_time,
                                        computed_at=computed_at,
                                        provenance_source=str(facts_path),
                                        support_mode="unsupported",
                                        value=None,
                                        unit=spec["unit"],
                                        missing_reason="statement_fact_unavailable",
                                        component_breakdown={"fact_type": spec["fact_type"]},
                                        quality_flags=["statement_fact_unavailable"],
                                    )
                                else:
                                    node = _feature_template(
                                        metric_name=metric_name,
                                        as_of_time=as_of_time,
                                        computed_at=computed_at,
                                        provenance_source=str(facts_path),
                                        support_mode="exact",
                                        value=float(fact["fact_value"]),
                                        unit=spec["unit"],
                                        missing_reason=None,
                                        component_breakdown={
                                            "fact_type": spec["fact_type"],
                                            "fact_id": fact.get("fact_id"),
                                            "source_id": fact.get("source_id"),
                                            "source_type": fact.get("source_type"),
                                            "raw_pointer": fact.get("raw_pointer"),
                                            "registry_unit": fact.get("unit"),
                                            "effective_at": None if fact.get("effective_at") is None else str(fact.get("effective_at"))[:10],
                                            "end": None if fact.get("effective_at") is None else str(fact.get("effective_at"))[:10],
                                            "fact_time": None if fact.get("fact_time") is None else str(fact.get("fact_time")),
                                            "formula": "statement_direct_fact",
                                        },
                                        quality_flags=None,
                                    )
                                features[metric_name] = node
                                counters[f"{metric_name}:{node['support_mode']}"] += 1

                            if companyfacts_root is not None:
                                interest_node = features.get("capital_structure.interest_expense_statement_direct") or {}
                                ebitda_node = features.get("operating.ebitda_ltm_provider_direct") or {}
                                statement_ebit_node = features.get("operating.ebit_statement_direct") or {}
                                needs_interest_repair = (
                                    interest_node.get("support_mode") == "unsupported"
                                    and interest_node.get("missing_reason") == "statement_fact_unavailable"
                                )
                                needs_ebitda_repair = (
                                    ebitda_node.get("support_mode") == "unsupported"
                                    and ebitda_node.get("missing_reason") == "sec_operating_income_ttm_unavailable"
                                    and statement_ebit_node.get("support_mode") == "exact"
                                    and statement_ebit_node.get("value") is not None
                                )
                                if needs_interest_repair or needs_ebitda_repair:
                                    if entity_id not in companyfacts_cache:
                                        companyfacts_cache[entity_id] = _load_companyfacts(companyfacts_root / f"CIK{entity_id}.json")
                                    companyfacts_path = companyfacts_root / f"CIK{entity_id}.json"
                                    companyfacts = companyfacts_cache.get(entity_id)

                                    if needs_interest_repair:
                                        repaired_interest_expense = _interest_expense_repair_from_companyfacts(
                                            current_node=interest_node,
                                            companyfacts=companyfacts,
                                            companyfacts_path=companyfacts_path,
                                            as_of_time=as_of_time,
                                            computed_at=computed_at,
                                        )
                                        if repaired_interest_expense is not None:
                                            prior_mode = interest_node.get("support_mode")
                                            if prior_mode:
                                                counters[f"capital_structure.interest_expense_statement_direct:{prior_mode}"] -= 1
                                            features["capital_structure.interest_expense_statement_direct"] = repaired_interest_expense
                                            counters[
                                                f"capital_structure.interest_expense_statement_direct:{repaired_interest_expense['support_mode']}"
                                            ] += 1

                                    if needs_ebitda_repair:
                                        repaired_ebitda = _ebitda_repair_from_statement_ebit(
                                            current_node=ebitda_node,
                                            statement_ebit_node=statement_ebit_node,
                                            companyfacts=companyfacts,
                                            companyfacts_path=companyfacts_path,
                                            as_of_time=as_of_time,
                                            computed_at=computed_at,
                                        )
                                        if repaired_ebitda is not None:
                                            prior_mode = ebitda_node.get("support_mode")
                                            if prior_mode:
                                                counters[f"operating.ebitda_ltm_provider_direct:{prior_mode}"] -= 1
                                            features["operating.ebitda_ltm_provider_direct"] = repaired_ebitda
                                            counters[f"operating.ebitda_ltm_provider_direct:{repaired_ebitda['support_mode']}"] += 1

                            repaired_total_debt = _repair_total_debt_from_statement_split(
                                current_node=features.get("capital_structure.total_debt_provider_direct"),
                                current_debt_statement_node=features.get("capital_structure.current_debt_statement_direct"),
                                long_term_debt_statement_node=features.get("capital_structure.long_term_debt_statement_direct"),
                                current_debt_statement_candidates=debt_fact_candidates.get((entity_id, "financial.debt_current")),
                                long_term_debt_statement_candidates=debt_fact_candidates.get((entity_id, "financial.debt_long_term")),
                                as_of_time=as_of_time,
                                computed_at=computed_at,
                                provenance_source=str(facts_path),
                            )
                            if repaired_total_debt is not None:
                                features["capital_structure.total_debt_provider_direct"] = repaired_total_debt
                                _recompute_standardized_debt_metrics(
                                    features=features,
                                    as_of_time=as_of_time,
                                    computed_at=computed_at,
                                    provenance_source=str(facts_path),
                                )
                    except _CompanyProcessingTimeout as exc:
                        counters["row_fail_open:company_processing_timeout"] += 1
                        features.update(
                            _build_fail_open_optional_metrics(
                                as_of_time=as_of_time,
                                computed_at=computed_at,
                                provenance_source=str(facts_path),
                                error_type="company_processing_timeout",
                                error_message=str(exc),
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        counters["row_fail_open:company_processing_failed"] += 1
                        features.update(
                            _build_fail_open_optional_metrics(
                                as_of_time=as_of_time,
                                computed_at=computed_at,
                                provenance_source=str(facts_path),
                                error_type="company_processing_failed",
                                error_message=str(exc),
                            )
                        )

                    out_handle.write(json.dumps(row) + "\n")

    if not saw_any_rows:
        raise ValueError(f"No rows found in snapshot: {snapshot_path}")

    if args.summary_out:
        summary = {}
        for metric_name in STATEMENT_FACT_SPECS:
            summary[metric_name] = {
                "exact": counters[f"{metric_name}:exact"],
                "unsupported": counters[f"{metric_name}:unsupported"],
            }
        summary["row_fail_open"] = {
            "company_processing_timeout": counters["row_fail_open:company_processing_timeout"],
            "company_processing_failed": counters["row_fail_open:company_processing_failed"],
        }
        Path(args.summary_out).write_text(json.dumps(summary, indent=2))

    print(f"Wrote statement-direct optional metrics -> {out_path}")
    if counters["row_fail_open:company_processing_timeout"] or counters["row_fail_open:company_processing_failed"]:
        print(
            "row_fail_open:"
            f" company_processing_timeout={counters['row_fail_open:company_processing_timeout']}"
            f" company_processing_failed={counters['row_fail_open:company_processing_failed']}"
        )


if __name__ == "__main__":
    main()
