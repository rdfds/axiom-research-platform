#!/usr/bin/env python3
"""Materialize the first smart-normalized layer on top of the v1 input artifact.

This script is intentionally conservative:
- it reads the ontology/policy JSON artifacts created for the smart layer
- it emits the first normalized metric ids
- it uses `proxy_missing_component` whenever the metric is economically useful
  but still missing required exact components for full promotion

The goal is not to pretend the ontology is finished; the goal is to make the
current smartest defensible layer explicit and machine-usable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

try:
    import pandas as pd
except Exception:  # noqa: BLE001
    pd = None

try:
    import requests
except Exception:  # noqa: BLE001
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:  # noqa: BLE001
    BeautifulSoup = None

try:
    from backfill_statement_direct_optional_metrics import (
        DEPRECIATION_TTM_CONCEPT_GROUPS,
        _compute_ttm_from_concept,
    )
    from backfill_input_layer_v1_metrics import _build_sec_core_metric
except Exception:  # noqa: BLE001
    from scripts.backfill_statement_direct_optional_metrics import (  # type: ignore
        DEPRECIATION_TTM_CONCEPT_GROUPS,
        _compute_ttm_from_concept,
    )
    from scripts.backfill_input_layer_v1_metrics import _build_sec_core_metric  # type: ignore


SMART_METRIC_NAMES = [
    "capital_structure.debt_like_obligations_normalized",
    "capital_structure.net_pension_liability",
    "capital_structure.other_postretirement_benefit_liability",
    "capital_structure.combined_retirement_liability",
    "capital_structure.debt_like_obligations_including_pension",
    "capital_structure.debt_like_obligations_including_retirement",
    "liquidity.available_liquidity_normalized",
    "operating.operating_earnings_normalized",
    "capital_structure.net_debt_normalized",
    "capital_structure.net_debt_including_pension",
    "capital_structure.net_debt_including_retirement",
    "capital_structure.gross_leverage_normalized",
    "capital_structure.gross_leverage_including_pension",
    "capital_structure.gross_leverage_including_retirement",
    "capital_structure.net_leverage_normalized",
    "capital_structure.net_leverage_including_pension",
    "capital_structure.net_leverage_including_retirement",
]

RECONCILIATION_TOLERANCE = 1.0
LEASE_STALE_CARRY_FORWARD_MAX_AGE_DAYS = 420
LEASE_ROU_FRESH_MAX_AGE_DAYS = 220
LEASE_IMMATERIAL_STALE_COMPONENT_MAX_ABS_USD = 25_000_000.0
COMPANYFACTS_LOAD_TIMEOUT_SECONDS = 1.5
RETIREMENT_NOTE_PARSE_TIMEOUT_SECONDS = 2.0
LEASE_IMMATERIAL_STALE_COMPONENT_MAX_RELATIVE_TO_DEBT = 0.01
FRESHER_COMPANYFACTS_OVERRIDE_MIN_GAP_DAYS = 7
OPERATING_EARNINGS_TAX_CONCEPTS = [
    "IncomeTaxExpenseBenefit",
]
OPERATING_EARNINGS_INTEREST_CONCEPTS = [
    "InterestExpense",
]
CASH_EQ_COMPANYFACTS_CONCEPTS = [
    "CashAndCashEquivalentsAtCarryingValue",
    "Cash",
]
NET_PENSION_LIABILITY_EXACT_TOTAL_CONCEPTS = [
    "DefinedBenefitPensionPlanProjectedBenefitObligationExcessPlanAssets",
    "DefinedBenefitPensionPlanProjectedBenefitObligationExcessFairValueOfPlanAssets",
    "DefinedBenefitPensionPlanUnderfundedStatus",
    "DefinedBenefitPensionPlanNetLiabilityRecognized",
]
NET_PENSION_LIABILITY_EXACT_CURRENT_CONCEPTS = [
    "DefinedBenefitPensionPlanLiabilitiesCurrent",
]
NET_PENSION_LIABILITY_EXACT_NONCURRENT_CONCEPTS = [
    "DefinedBenefitPensionPlanLiabilitiesNoncurrent",
]
NET_PENSION_LIABILITY_PROXY_TOTAL_CONCEPTS = [
    "PensionAndOtherPostretirementDefinedBenefitPlansLiabilities",
    "PensionAndOtherPostretirementAndPostemploymentBenefitPlansLiabilities",
]
NET_PENSION_LIABILITY_PROXY_CURRENT_CONCEPTS = [
    "PensionAndOtherPostretirementDefinedBenefitPlansCurrentLiabilities",
    "PensionAndOtherPostretirementAndPostemploymentBenefitPlansCurrentLiabilities",
]
NET_PENSION_LIABILITY_PROXY_NONCURRENT_CONCEPTS = [
    "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent",
    "PensionAndOtherPostretirementAndPostemploymentBenefitPlansLiabilitiesNoncurrent",
]
DEFAULT_MARKET_AVAILABILITY_OVERRIDES_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "liquidity_market_availability_overrides.json"
)
DEFAULT_LOCAL_COMPANYFACTS_ROOT = Path(__file__).resolve().parents[1] / "data" / "sec" / "companyfacts"
DEFAULT_SEC_RETIREMENT_CACHE_ROOT = Path("/tmp/sec_retirement_note_cache")
RETIREMENT_NOTE_CARRYFORWARD_MAX_AGE_DAYS = 430
RETIREMENT_NOTE_MAX_FILINGS_TO_SCAN = 6
_COMPANYFACTS_UNSET = object()
SEC_USER_AGENT = "Codex/axiom_v1 retirement note support"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
RETIREMENT_NOTE_TABLE_CUES = (
    "pension benefits",
    "other benefits",
    "other postretirement",
    "postretirement benefits",
    "retirement-related benefits",
    "funded status",
    "benefit obligation",
)
RETIREMENT_NOTE_PENSION_COLUMN_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bpension\b",
        r"defined[\s\-]benefit",
    )
]
RETIREMENT_NOTE_OTHER_POSTRETIREMENT_COLUMN_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"other\s+benefits",
        r"other\s+postretirement",
        r"postretirement",
    )
]
RETIREMENT_FUNDED_STATUS_LABEL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"funded status",
        r"net amount recognized",
        r"net amount .* recognized",
    )
]
RETIREMENT_DIRECT_LIABILITY_LABEL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"liabilit(?:y|ies)",
        r"accrued .* benefit",
        r"amount recognized .* balance sheet",
    )
]
RETIREMENT_BENEFIT_OBLIGATION_LABEL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"projected benefit obligation",
        r"accumulated benefit obligation",
        r"benefit obligation .* end of year",
        r"benefit obligation",
    )
]
RETIREMENT_PLAN_ASSETS_LABEL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"fair value of plan assets",
        r"plan assets .* end of year",
        r"plan assets",
    )
]


SMART_METRIC_UNITS = {
    "capital_structure.debt_like_obligations_normalized": "usd",
    "capital_structure.net_pension_liability": "usd",
    "capital_structure.other_postretirement_benefit_liability": "usd",
    "capital_structure.combined_retirement_liability": "usd",
    "capital_structure.debt_like_obligations_including_pension": "usd",
    "capital_structure.debt_like_obligations_including_retirement": "usd",
    "liquidity.available_liquidity_normalized": "usd",
    "operating.operating_earnings_normalized": "usd",
    "capital_structure.net_debt_normalized": "usd",
    "capital_structure.net_debt_including_pension": "usd",
    "capital_structure.net_debt_including_retirement": "usd",
    "capital_structure.gross_leverage_normalized": "x",
    "capital_structure.gross_leverage_including_pension": "x",
    "capital_structure.gross_leverage_including_retirement": "x",
    "capital_structure.net_leverage_normalized": "x",
    "capital_structure.net_leverage_including_pension": "x",
    "capital_structure.net_leverage_including_retirement": "x",
}


class _CompanyProcessingTimeout(RuntimeError):
    """Raised when a single-company smart-normalized build exceeds the allowed timeout."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-path", required=True, help="Input snapshot JSONL path")
    parser.add_argument("--metric-registry-path", required=True, help="Smart metric registry JSON")
    parser.add_argument("--component-policy-path", required=True, help="Component inclusion policy JSON")
    parser.add_argument("--source-precedence-path", required=True, help="Source precedence policy JSON")
    parser.add_argument(
        "--companyfacts-root",
        help="Optional SEC companyfacts folder for operating-earnings repair. Defaults to the local canonical companyfacts root when present.",
    )
    parser.add_argument(
        "--sec-filing-cache-root",
        help=(
            "Optional cache root for SEC filing-note HTML used to separate pension from other postretirement "
            "liabilities when companyfacts only exposes combined concepts. Defaults to /tmp/sec_retirement_note_cache."
        ),
    )
    parser.add_argument(
        "--market-availability-overrides-path",
        help="Optional JSON file with explicit market-availability cash adjustments",
    )
    parser.add_argument(
        "--company-processing-timeout-seconds",
        type=float,
        default=15.0,
        help="Fail open on a single company if smart-normalized enrichment exceeds this timeout. Use 0 to disable.",
    )
    parser.add_argument(
        "--resume-if-exists",
        action="store_true",
        help="If the output file already exists, skip completed company ids and append remaining rows.",
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


def _load_completed_company_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            company_id = row.get("company_id")
            if company_id is not None:
                completed.add(str(company_id))
    return completed


def _summarize_output_rows(path: Path) -> Dict[str, Dict[str, int]]:
    counters: Counter[str] = Counter()
    fail_open: Counter[str] = Counter()
    for row in iter_snapshot_rows(path):
        features = row.get("features") or {}
        row_fail_reason = None
        for metric_name in SMART_METRIC_NAMES:
            node = features.get(metric_name) or {}
            support_mode = str(node.get("support_mode") or "unsupported")
            if support_mode not in {"exact", "proxy_missing_component", "unsupported"}:
                support_mode = "unsupported"
            counters[f"{metric_name}:{support_mode}"] += 1
            if support_mode == "unsupported":
                missing_reason = str(node.get("missing_reason") or "")
                if row_fail_reason is None and missing_reason in {"company_processing_timeout", "company_processing_failed"}:
                    row_fail_reason = missing_reason
        if row_fail_reason is not None:
            fail_open[row_fail_reason] += 1

    summary: Dict[str, Dict[str, int]] = {}
    for metric_name in SMART_METRIC_NAMES:
        summary[metric_name] = {
            "exact": counters[f"{metric_name}:exact"],
            "proxy_missing_component": counters[f"{metric_name}:proxy_missing_component"],
            "unsupported": counters[f"{metric_name}:unsupported"],
        }
    summary["row_fail_open"] = {
        "company_processing_timeout": fail_open["company_processing_timeout"],
        "company_processing_failed": fail_open["company_processing_failed"],
    }
    return summary


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


def _feature_template(
    *,
    metric_name: str,
    as_of_time: str,
    computed_at: str,
    support_mode: str,
    value: Any,
    unit: str,
    provenance_sources: list[str],
    missing_reason: str | None,
    component_breakdown: Dict[str, Any] | None,
    quality_flags: list[str] | None,
) -> Dict[str, Any]:
    provenance = []
    for source in provenance_sources:
        provenance.append(
            {
                "artifact_type": "OntologyDerivedMetric",
                "artifact_id": f"smart_metric:{Path(source).name}",
                "source": source,
                "published_at": as_of_time,
                "ingested_at": computed_at,
                "hash": None,
            }
        )

    return {
        "name": metric_name,
        "value": value,
        "unit": unit,
        "computed_at": computed_at,
        "as_of_time": as_of_time,
        "window": None,
        "confidence": 1.0 if value is not None else None,
        "provenance": provenance,
        "missing_reason": missing_reason,
        "fallback_used": None,
        "metric_policy_id": None,
        "market_owner": None,
        "primary_source_basis": "smart_normalized_policy",
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
        "input_source_classification": "smart_normalized_policy",
        "input_source_formula_basis": None,
        "input_source_alignment_status": "aligned",
        "input_source_document_ids": None,
        "definition_requirement": None,
        "definition_requirement_reason": None,
        "methodology_execution_decision": None,
        "methodology_execution_reason": None,
        "input_layer_bucket": "smart_normalized",
        "input_layer_bucket_reason": "policy_governed_normalization",
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


def _node(features: Dict[str, Any], metric_name: str) -> Dict[str, Any]:
    return features.get(metric_name) or {}


def _value(node: Dict[str, Any]) -> float | None:
    value = node.get("value")
    return None if value is None else float(value)


def _exact(node: Dict[str, Any]) -> bool:
    return node.get("support_mode") == "exact"


def _is_supported(node: Dict[str, Any]) -> bool:
    return node.get("support_mode") in {"exact", "proxy_missing_component"}


def _approximately_equal(left: float | None, right: float | None, tolerance: float = RECONCILIATION_TOLERANCE) -> bool:
    return left is not None and right is not None and abs(left - right) <= tolerance


def _parse_iso_date(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value[:10]
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _latest_recursive_timestamp(payload: Any) -> datetime | None:
    latest: datetime | None = None
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"end", "filed", "published_at", "as_of_time", "period_end"}:
                parsed = _parse_iso_date(str(value) if value is not None else None)
                if parsed is not None and (latest is None or parsed > latest):
                    latest = parsed
            nested = _latest_recursive_timestamp(value)
            if nested is not None and (latest is None or nested > latest):
                latest = nested
    elif isinstance(payload, list):
        for item in payload:
            nested = _latest_recursive_timestamp(item)
            if nested is not None and (latest is None or nested > latest):
                latest = nested
    return latest


def _latest_node_timestamp(node: Dict[str, Any]) -> datetime | None:
    latest: datetime | None = None
    for provenance in node.get("provenance") or []:
        parsed = _parse_iso_date(provenance.get("published_at"))
        if parsed is not None and (latest is None or parsed > latest):
            latest = parsed
    component_latest = _latest_recursive_timestamp(node.get("component_breakdown"))
    if component_latest is not None and (latest is None or component_latest > latest):
        latest = component_latest
    return latest


def _companyfacts_candidate_is_fresher(
    *,
    existing_node: Dict[str, Any],
    companyfacts_meta: Dict[str, Any] | None,
    candidate_value: float | None,
) -> bool:
    if candidate_value is None:
        return False
    existing_value = _value(existing_node)
    if existing_value is None:
        return True
    existing_ts = _latest_node_timestamp(existing_node)
    companyfacts_ts = _latest_recursive_timestamp(companyfacts_meta)
    if companyfacts_ts is None:
        return False
    if existing_ts is None:
        return not _approximately_equal(existing_value, candidate_value)
    if companyfacts_ts <= existing_ts:
        return False
    gap_days = (companyfacts_ts - existing_ts).days
    return gap_days >= FRESHER_COMPANYFACTS_OVERRIDE_MIN_GAP_DAYS and not _approximately_equal(
        existing_value,
        candidate_value,
    )


def _load_companyfacts(path: Path | None) -> Dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        completed = subprocess.run(
            ["/bin/cat", str(path)],
            capture_output=True,
            timeout=COMPANYFACTS_LOAD_TIMEOUT_SECONDS,
            check=True,
        )
        return json.loads(completed.stdout)
    except subprocess.TimeoutExpired:
        return None
    except subprocess.CalledProcessError:
        return None
    except _CompanyProcessingTimeout:
        return None
    try:
        with _company_processing_guard(COMPANYFACTS_LOAD_TIMEOUT_SECONDS):
            return json.loads(path.read_text())
    except _CompanyProcessingTimeout:
        return None
    except Exception:  # noqa: BLE001
        return None


def _companyfacts_has_any_concepts(companyfacts: Dict[str, Any] | None, concept_names: list[str]) -> bool:
    if companyfacts is None:
        return False
    for taxonomy in ("us-gaap", "dei", "ifrs-full"):
        facts = (companyfacts.get("facts") or {}).get(taxonomy) or {}
        for concept_name in concept_names:
            if concept_name in facts:
                return True
    return False


def _companyfacts_may_need_retirement_note_split(companyfacts: Dict[str, Any] | None) -> bool:
    if companyfacts is None:
        return False
    has_exact = _companyfacts_has_any_concepts(
        companyfacts,
        NET_PENSION_LIABILITY_EXACT_TOTAL_CONCEPTS
        + NET_PENSION_LIABILITY_EXACT_CURRENT_CONCEPTS
        + NET_PENSION_LIABILITY_EXACT_NONCURRENT_CONCEPTS,
    )
    if has_exact:
        return False
    return _companyfacts_has_any_concepts(
        companyfacts,
        NET_PENSION_LIABILITY_PROXY_TOTAL_CONCEPTS
        + NET_PENSION_LIABILITY_PROXY_CURRENT_CONCEPTS
        + NET_PENSION_LIABILITY_PROXY_NONCURRENT_CONCEPTS,
    )


