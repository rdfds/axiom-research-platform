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


def _sec_helpers_available() -> bool:
    return requests is not None and BeautifulSoup is not None and pd is not None


def _normalize_note_label(text: Any) -> str:
    label = " ".join(str(text or "").replace("\xa0", " ").replace("\u200b", " ").split())
    label = label.replace("’", "'").replace("–", "-").replace("—", "-")
    return label.strip(" :")


def _retirement_note_column_category(header_text: str) -> str | None:
    normalized = _normalize_note_label(header_text).lower()
    if not normalized or normalized in {"nan", "none"}:
        return None
    if any(pattern.search(normalized) for pattern in RETIREMENT_NOTE_OTHER_POSTRETIREMENT_COLUMN_PATTERNS):
        return "other_postretirement"
    if any(pattern.search(normalized) for pattern in RETIREMENT_NOTE_PENSION_COLUMN_PATTERNS):
        return "pension"
    return None


def _flatten_note_column_name(column: Any) -> str:
    if isinstance(column, tuple):
        parts = [_normalize_note_label(part) for part in column if _normalize_note_label(part) not in {"", "nan", "None"}]
        return " ".join(parts)
    return _normalize_note_label(column)


def _coerce_note_numeric(value: Any, *, multiplier: float) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        if value != value:  # NaN
            return None
        return float(value) * multiplier
    if isinstance(value, int):
        return float(value) * multiplier
    text = _normalize_note_label(value)
    if not text or text.lower() in {"nan", "-", "—", "nm", "n/m"}:
        return None
    negative = "(" in text and ")" in text
    stripped = text.replace("$", "").replace(",", "").replace("(", "").replace(")", "").strip()
    try:
        numeric = float(stripped)
    except ValueError:
        return None
    return (-numeric if negative else numeric) * multiplier


def _load_sec_submissions(
    cik: str,
    *,
    session: Any,
    cache_dir: Path | None,
) -> dict[str, Any] | None:
    cache_path = None if cache_dir is None else cache_dir / f"CIK{cik}.json"
    if cache_path is not None and cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:  # noqa: BLE001
            pass
    if os.environ.get("AXIOM_DISABLE_SEC_NETWORK_FALLBACK") == "1":
        return None
    if session is None:
        return None
    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception:  # noqa: BLE001
        return None
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload))
    return payload


def _latest_sec_filing(
    *,
    cik: str,
    as_of_date: date,
    session: Any,
    cache_dir: Path | None,
) -> dict[str, Any] | None:
    filings = _recent_sec_filings(
        cik=cik,
        as_of_date=as_of_date,
        session=session,
        cache_dir=cache_dir,
    )
    return None if not filings else filings[0]


def _recent_sec_filings(
    *,
    cik: str,
    as_of_date: date,
    session: Any,
    cache_dir: Path | None,
) -> list[dict[str, Any]]:
    submissions = _load_sec_submissions(cik, session=session, cache_dir=cache_dir)
    if not submissions:
        return []
    recent = (submissions.get("filings") or {}).get("recent") or {}
    forms = {"10-Q": 2, "10-K": 1}
    filings: list[tuple[date, int, dict[str, Any]]] = []
    for filing_date, form, accession, primary_document in zip(
        recent.get("filingDate", []),
        recent.get("form", []),
        recent.get("accessionNumber", []),
        recent.get("primaryDocument", []),
    ):
        if form not in forms:
            continue
        filed_dt = _parse_iso_date(filing_date)
        if filed_dt is None or filed_dt.date() > as_of_date:
            continue
        record = {
            "cik": cik,
            "filing_date": filing_date,
            "form": form,
            "accession_number": accession,
            "primary_document": primary_document,
        }
        filings.append((filed_dt.date(), forms[form], record))
    filings.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [record for _, _, record in filings]


def _fetch_sec_primary_document(
    filing: dict[str, Any],
    *,
    session: Any,
    cache_dir: Path | None,
) -> str | None:
    accession = str(filing["accession_number"])
    accession_nodash = accession.replace("-", "")
    cik_no_zeros = str(int(filing["cik"]))
    primary_document = str(filing["primary_document"])
    cache_path = None
    if cache_dir is not None:
        safe_name = f"{filing['cik']}_{accession_nodash}_{Path(primary_document).name}"
        cache_path = cache_dir / safe_name
        if cache_path.exists():
            return cache_path.read_text(errors="ignore")
    if os.environ.get("AXIOM_DISABLE_SEC_NETWORK_FALLBACK") == "1":
        return None
    if session is None:
        return None
    url = f"{SEC_ARCHIVES_BASE}/{cik_no_zeros}/{accession_nodash}/{primary_document}"
    try:
        response = session.get(url, timeout=60)
        response.raise_for_status()
        html = response.text
    except Exception:  # noqa: BLE001
        return None
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(html)
    return html


def _retirement_table_multiplier(table_text: str) -> float:
    lower = table_text.lower()
    if "in billions" in lower or "($ in billions)" in lower or "(billions)" in lower:
        return 1_000_000_000.0
    if "in millions" in lower or "($ in millions)" in lower or "(millions)" in lower:
        return 1_000_000.0
    if "in thousands" in lower or "($ in thousands)" in lower or "(thousands)" in lower:
        return 1_000.0
    return 1.0


def _extract_retirement_note_components_from_html(
    *,
    filing: dict[str, Any],
    html: str,
    as_of_time: str,
) -> dict[str, Any] | None:
    if not _sec_helpers_available():
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
        html_tables = soup.find_all("table")
        dataframes = pd.read_html(StringIO(html), displayed_only=False)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return None
    if not dataframes:
        return None
    as_of_year = str(as_of_time[:4])
    best: tuple[tuple[int, float], dict[str, Any]] | None = None
    for table_index, dataframe in enumerate(dataframes):
        table_html = html_tables[table_index] if table_index < len(html_tables) else None
        table_text = ""
        if table_html is not None:
            table_text = " ".join(table_html.get_text(" ", strip=True).split())
        if table_text and not any(cue in table_text.lower() for cue in RETIREMENT_NOTE_TABLE_CUES):
            continue
        frame = dataframe.copy()
        frame.columns = [_flatten_note_column_name(col) for col in frame.columns]
        if frame.empty or len(frame.columns) < 2:
            continue
        label_col = frame.columns[0]
        category_columns: dict[str, list[tuple[int, str]]] = {"pension": [], "other_postretirement": []}
        for idx, column_name in enumerate(frame.columns[1:], start=1):
            category = _retirement_note_column_category(column_name)
            if category is not None:
                category_columns[category].append((idx, column_name))
        if not category_columns["pension"] and not category_columns["other_postretirement"]:
            continue
        for category, cols in list(category_columns.items()):
            year_specific = [(idx, name) for idx, name in cols if as_of_year in name]
            if year_specific:
                category_columns[category] = year_specific
        multiplier = _retirement_table_multiplier(table_text or frame.to_string())
        direct_values: dict[str, tuple[float, dict[str, Any]]] = {}
        obligations: dict[str, tuple[float, dict[str, Any]]] = {}
        assets: dict[str, tuple[float, dict[str, Any]]] = {}

        for _, raw_row in frame.iterrows():
            label = _normalize_note_label(raw_row[label_col])
            if not label or label.lower() in {"nan", "none"}:
                continue
            row_values: dict[str, float] = {}
            row_columns: dict[str, list[str]] = {}
            for category, cols in category_columns.items():
                numeric_values: list[float] = []
                used_headers: list[str] = []
                for idx, header in cols:
                    numeric = _coerce_note_numeric(raw_row.iloc[idx], multiplier=multiplier)
                    if numeric is not None:
                        numeric_values.append(numeric)
                        used_headers.append(header)
                if numeric_values:
                    row_values[category] = float(sum(numeric_values))
                    row_columns[category] = used_headers
            if not row_values:
                continue
            if any(pattern.search(label) for pattern in RETIREMENT_FUNDED_STATUS_LABEL_PATTERNS):
                for category, value in row_values.items():
                    direct_values[category] = (
                        max(-value, 0.0),
                        {
                            "row_label": label,
                            "mode": "funded_status_row",
                            "headers": row_columns.get(category),
                        },
                    )
                continue
            if any(pattern.search(label) for pattern in RETIREMENT_DIRECT_LIABILITY_LABEL_PATTERNS):
                for category, value in row_values.items():
                    direct_values[category] = (
                        max(value, 0.0),
                        {
                            "row_label": label,
                            "mode": "direct_liability_row",
                            "headers": row_columns.get(category),
                        },
                    )
                continue
            if any(pattern.search(label) for pattern in RETIREMENT_BENEFIT_OBLIGATION_LABEL_PATTERNS):
                for category, value in row_values.items():
                    obligations[category] = (
                        value,
                        {
                            "row_label": label,
                            "mode": "benefit_obligation_row",
                            "headers": row_columns.get(category),
                        },
                    )
                continue
            if any(pattern.search(label) for pattern in RETIREMENT_PLAN_ASSETS_LABEL_PATTERNS):
                for category, value in row_values.items():
                    assets[category] = (
                        value,
                        {
                            "row_label": label,
                            "mode": "plan_assets_row",
                            "headers": row_columns.get(category),
                        },
                    )

        components: dict[str, dict[str, Any]] = {}
        for category in ("pension", "other_postretirement"):
            if category in direct_values:
                value, source_meta = direct_values[category]
                components[category] = {
                    "value": value,
                    "source_meta": source_meta,
                }
            elif category in obligations and category in assets:
                obligation_value, obligation_meta = obligations[category]
                asset_value, asset_meta = assets[category]
                components[category] = {
                    "value": max(obligation_value - asset_value, 0.0),
                    "source_meta": {
                        "mode": "benefit_obligation_minus_plan_assets",
                        "benefit_obligation": obligation_meta,
                        "plan_assets": asset_meta,
                    },
                }
        if not components:
            continue
        score = 0
        for component in components.values():
            mode = component["source_meta"]["mode"]
            if mode == "funded_status_row":
                score += 6
            elif mode == "direct_liability_row":
                score += 5
            else:
                score += 3
        if "pension" in components:
            score += 4
        if "other_postretirement" in components:
            score += 2
        pension_value = float(components["pension"]["value"]) if "pension" in components else None
        candidate = {
            "pension_value": pension_value,
            "other_postretirement_value": (
                float(components["other_postretirement"]["value"])
                if "other_postretirement" in components
                else None
            ),
            "component_meta": {
                "mode": "filing_note_retirement_split",
                "table_index": table_index,
                "table_excerpt": table_text[:4000],
                "filing": filing,
                "pension": components.get("pension"),
                "other_postretirement": components.get("other_postretirement"),
            },
        }
        rank = (score, pension_value or 0.0)
        if best is None or rank > best[0]:
            best = (rank, candidate)
    return None if best is None else best[1]


def _retirement_regime_hint_from_html(html: str) -> dict[str, Any] | None:
    lower = " ".join(html.lower().split())
    has_defined_contribution = "defined contribution" in lower or "defined-contribution" in lower
    has_defined_benefit_signal = any(
        cue in lower
        for cue in (
            "defined benefit",
            "funded status",
            "projected benefit obligation",
            "accumulated benefit obligation",
            "plan assets",
            "other postretirement benefit obligation",
            "postretirement benefit obligation",
            "pension liability",
            "net periodic pension",
        )
    )
    if not has_defined_contribution or has_defined_benefit_signal:
        return None
    return {
        "regime_hint": "defined_contribution_only",
        "text_excerpt": lower[:1000],
    }


def _load_retirement_note_components(
    *,
    cik: str | None,
    as_of_time: str,
    session: Any,
    cache_dir: Path | None,
) -> dict[str, Any] | None:
    if cik is None or not _sec_helpers_available():
        return None
    as_of_dt = _parse_iso_date(as_of_time)
    if as_of_dt is None:
        return None
    filings = _recent_sec_filings(
        cik=cik,
        as_of_date=as_of_dt.date(),
        session=session,
        cache_dir=cache_dir,
    )
    if not filings:
        return None
    hinted_regime: dict[str, Any] | None = None
    for filing_index, filing in enumerate(filings[:RETIREMENT_NOTE_MAX_FILINGS_TO_SCAN]):
        filed_dt = _parse_iso_date(filing.get("filing_date"))
        if filed_dt is None:
            continue
        filing_age_days = (as_of_dt - filed_dt).days
        if filing_age_days < 0 or filing_age_days > RETIREMENT_NOTE_CARRYFORWARD_MAX_AGE_DAYS:
            continue
        html = _fetch_sec_primary_document(filing, session=session, cache_dir=cache_dir)
        if not html:
            continue
        parsed = _extract_retirement_note_components_from_html(
            filing=filing,
            html=html,
            as_of_time=as_of_time,
        )
        if parsed is not None:
            component_meta = dict(parsed.get("component_meta") or {})
            component_meta["filing_age_days"] = filing_age_days
            component_meta["carryforward_used"] = filing_index > 0
            component_meta["carryforward_source"] = "prior_filing_note" if filing_index > 0 else "current_filing_note"
            parsed["component_meta"] = component_meta
            parsed["regime_hint"] = "pension_proxy_split_note"
            return parsed
        hint = _retirement_regime_hint_from_html(html)
        if hint is not None and hinted_regime is None:
            hinted_regime = {
                "pension_value": None,
                "other_postretirement_value": None,
                "component_meta": {
                    "mode": "defined_contribution_only_filing_text",
                    "filing": filing,
                    "filing_age_days": filing_age_days,
                    "carryforward_used": filing_index > 0,
                    "carryforward_source": "prior_filing_note" if filing_index > 0 else "current_filing_note",
                    "text_excerpt": hint["text_excerpt"],
                },
                "regime_hint": hint["regime_hint"],
            }
    return hinted_regime


def _registry_metric(registry: Dict[str, Any], metric_key: str) -> Dict[str, str]:
    metrics = registry.get("metrics") if isinstance(registry, dict) else None
    entry = metrics.get(metric_key) if isinstance(metrics, dict) else None
    if not isinstance(entry, dict):
        return {
            "status": "partially_feasible",
            "promotion_rule": "registry_entry_missing_defaulted",
        }
    return {
        "status": str(entry.get("status") or "partially_feasible"),
        "promotion_rule": str(entry.get("promotion_rule") or "registry_entry_missing_defaulted"),
    }


def _load_market_availability_overrides(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {}
    return payload if isinstance(payload, dict) else {}


def _infer_companyfacts_path_from_features(features: Dict[str, Any]) -> Path | None:
    candidate_metric_names = [
        "liquidity.restricted_cash_sec_exact",
        "liquidity.marketable_securities_sec_exact",
        "capital_structure.lease_liabilities_sec_exact",
    ]
    for metric_name in candidate_metric_names:
        node = features.get(metric_name) or {}
        breakdown = node.get("component_breakdown") or {}
        path_text = breakdown.get("companyfacts_path")
        if path_text:
            path = Path(path_text)
            if path.exists():
                return path
        for provenance in node.get("provenance") or []:
            source = provenance.get("source")
            if not source or not str(source).endswith(".json"):
                continue
            path = Path(source)
            if path.exists():
                return path
    return None


def _latest_companyfacts_point_value(
    companyfacts: Dict[str, Any] | None,
    concept_names: list[str],
    *,
    as_of_time: str,
) -> tuple[float | None, Dict[str, Any] | None]:
    if companyfacts is None:
        return None, None
    as_of_date = as_of_time[:10]
    as_of_dt = _parse_iso_date(as_of_time)
    if as_of_dt is None:
        return None, None

    candidates: list[tuple[str, str, float, Dict[str, Any]]] = []
    for taxonomy in ("us-gaap", "dei", "ifrs-full"):
        facts = (companyfacts.get("facts") or {}).get(taxonomy) or {}
        for concept_name in concept_names:
            concept = facts.get(concept_name) or {}
            units = concept.get("units") or {}
            for unit, entries in units.items():
                if str(unit).upper() != "USD":
                    continue
                for entry in entries:
                    end = entry.get("end")
                    filed = entry.get("filed")
                    value = entry.get("val")
                    if end is None or value is None or end > as_of_date:
                        continue
                    if filed is not None and filed > as_of_date:
                        continue
                    end_dt = _parse_iso_date(end)
                    if end_dt is None or (as_of_dt - end_dt).days > 550:
                        continue
                    meta = {
                        "concept": concept_name,
                        "taxonomy": taxonomy,
                        "end": end,
                        "filed": filed,
                        "fy": entry.get("fy"),
                        "fp": entry.get("fp"),
                        "frame": entry.get("frame"),
                        "form": entry.get("form"),
                        "unit": unit,
                    }
                    candidates.append((end, filed or "", float(value), meta))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, _, value, meta = candidates[-1]
    return value, meta


def _effective_net_pension_liability_value(
    companyfacts: Dict[str, Any] | None,
    *,
    as_of_time: str,
    retirement_note_loader: Callable[[], Dict[str, Any] | None] | None = None,
) -> Dict[str, Any]:
    retirement_note_state: Any = _COMPANYFACTS_UNSET

    def _get_retirement_note_components() -> Dict[str, Any] | None:
        nonlocal retirement_note_state
        if retirement_note_state is _COMPANYFACTS_UNSET:
            if retirement_note_loader is None:
                retirement_note_state = None
            else:
                try:
                    with _company_processing_guard(RETIREMENT_NOTE_PARSE_TIMEOUT_SECONDS):
                        retirement_note_state = retirement_note_loader()
                except _CompanyProcessingTimeout:
                    retirement_note_state = None
                except Exception:  # noqa: BLE001
                    retirement_note_state = None
        return retirement_note_state

    def _with_regime(
        payload: Dict[str, Any],
        *,
        regime: str,
        regime_source: str,
        regime_meta: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        enriched = dict(payload)
        enriched["retirement_regime"] = regime
        enriched["retirement_regime_source"] = regime_source
        enriched["retirement_regime_component_meta"] = regime_meta
        return enriched

    if companyfacts is None and retirement_note_loader is None:
        return _with_regime({
            "value": None,
            "exact": False,
            "source_metric": None,
            "formula": "unavailable",
            "support_override": None,
            "component_meta": None,
            "other_postretirement_value": None,
            "other_postretirement_exact": False,
            "other_postretirement_source_metric": None,
            "other_postretirement_formula": "unavailable",
            "other_postretirement_support_override": None,
            "other_postretirement_component_meta": None,
            "combined_retirement_value": None,
            "combined_retirement_exact": False,
            "combined_retirement_source_metric": None,
            "combined_retirement_formula": "unavailable",
            "combined_retirement_support_override": None,
            "combined_retirement_component_meta": None,
        }, regime="retirement_not_surfaced", regime_source="unavailable")

    exact_total_value, exact_total_meta = _latest_companyfacts_point_value(
        companyfacts,
        NET_PENSION_LIABILITY_EXACT_TOTAL_CONCEPTS,
        as_of_time=as_of_time,
    )
    exact_current_value, exact_current_meta = _latest_companyfacts_point_value(
        companyfacts,
        NET_PENSION_LIABILITY_EXACT_CURRENT_CONCEPTS,
        as_of_time=as_of_time,
    )
    exact_noncurrent_value, exact_noncurrent_meta = _latest_companyfacts_point_value(
        companyfacts,
        NET_PENSION_LIABILITY_EXACT_NONCURRENT_CONCEPTS,
        as_of_time=as_of_time,
    )
    proxy_total_value, proxy_total_meta = _latest_companyfacts_point_value(
        companyfacts,
        NET_PENSION_LIABILITY_PROXY_TOTAL_CONCEPTS,
        as_of_time=as_of_time,
    )
    proxy_current_value, proxy_current_meta = _latest_companyfacts_point_value(
        companyfacts,
        NET_PENSION_LIABILITY_PROXY_CURRENT_CONCEPTS,
        as_of_time=as_of_time,
    )
    proxy_noncurrent_value, proxy_noncurrent_meta = _latest_companyfacts_point_value(
        companyfacts,
        NET_PENSION_LIABILITY_PROXY_NONCURRENT_CONCEPTS,
        as_of_time=as_of_time,
    )

    def _normalized_nonnegative(value: float | None) -> tuple[float | None, bool]:
        if value is None:
            return None, False
        if value < 0:
            return 0.0, True
        return float(value), False

    normalized_exact_total, exact_total_clipped = _normalized_nonnegative(exact_total_value)
    if normalized_exact_total is not None:
        return _with_regime({
            "value": normalized_exact_total,
            "exact": True,
            "source_metric": "capital_structure.net_pension_liability_companyfacts_exact",
            "formula": "latest_exact_net_pension_liability_on_or_before_asof",
            "support_override": "overfunded_pension_excluded_from_liability_view" if exact_total_clipped else None,
            "component_meta": {
                "mode": "exact_total_companyfacts",
                "exact_total": exact_total_meta,
            },
            "other_postretirement_value": None,
            "other_postretirement_exact": False,
            "other_postretirement_source_metric": None,
            "other_postretirement_formula": "unavailable",
            "other_postretirement_support_override": None,
            "other_postretirement_component_meta": None,
            "combined_retirement_value": normalized_exact_total,
            "combined_retirement_exact": False,
            "combined_retirement_source_metric": "capital_structure.combined_retirement_liability_from_pension_exact",
            "combined_retirement_formula": "net_pension_liability_exact + 0_assumed_missing_other_postretirement",
            "combined_retirement_support_override": "combined_retirement_uses_pension_only",
            "combined_retirement_component_meta": {
                "mode": "pension_only_exact_companyfacts",
                "pension_exact": exact_total_meta,
                "other_postretirement_missing_assumed_zero": True,
            },
        }, regime="pension_exact", regime_source="companyfacts_exact_total", regime_meta={
            "mode": "pension_exact_companyfacts",
            "exact_total": exact_total_meta,
        })

    if exact_current_value is not None or exact_noncurrent_value is not None:
        current_component, current_clipped = _normalized_nonnegative(exact_current_value)
        noncurrent_component, noncurrent_clipped = _normalized_nonnegative(exact_noncurrent_value)
        value = float((current_component or 0.0) + (noncurrent_component or 0.0))
        exact_ready = exact_current_value is not None or exact_noncurrent_value is not None
        override = None
        if current_clipped or noncurrent_clipped:
            override = "overfunded_pension_excluded_from_liability_view"
            exact_ready = False
        elif exact_current_value is None or exact_noncurrent_value is None:
            override = "single_pension_liability_component"
        return _with_regime({
            "value": value,
            "exact": exact_ready,
            "source_metric": "capital_structure.net_pension_liability_companyfacts_exact",
            "formula": "sum_available_exact_pension_liability_components",
            "support_override": override,
            "component_meta": {
                "mode": "exact_split_companyfacts",
                "current": exact_current_meta,
                "noncurrent": exact_noncurrent_meta,
            },
            "other_postretirement_value": None,
            "other_postretirement_exact": False,
            "other_postretirement_source_metric": None,
            "other_postretirement_formula": "unavailable",
            "other_postretirement_support_override": None,
            "other_postretirement_component_meta": None,
            "combined_retirement_value": value,
            "combined_retirement_exact": False,
            "combined_retirement_source_metric": "capital_structure.combined_retirement_liability_from_pension_exact",
            "combined_retirement_formula": "net_pension_liability_exact + 0_assumed_missing_other_postretirement",
            "combined_retirement_support_override": (
                "combined_retirement_uses_pension_only"
                + (f";{override}" if override else "")
            ),
            "combined_retirement_component_meta": {
                "mode": "pension_only_exact_companyfacts",
                "pension_exact": {
                    "current": exact_current_meta,
                    "noncurrent": exact_noncurrent_meta,
                },
                "other_postretirement_missing_assumed_zero": True,
            },
        }, regime="pension_exact", regime_source="companyfacts_exact_split", regime_meta={
            "mode": "pension_exact_companyfacts",
            "current": exact_current_meta,
            "noncurrent": exact_noncurrent_meta,
            "support_override": override,
        })

    retirement_note_components = _get_retirement_note_components()
    note_pension_value = None if retirement_note_components is None else retirement_note_components.get("pension_value")
    note_other_postretirement_value = (
        None if retirement_note_components is None else retirement_note_components.get("other_postretirement_value")
    )
    note_component_meta = None if retirement_note_components is None else retirement_note_components.get("component_meta")
    note_regime_hint = None if retirement_note_components is None else retirement_note_components.get("regime_hint")

    if note_pension_value is not None:
        combined_retirement_value = float(note_pension_value + (note_other_postretirement_value or 0.0))
        return _with_regime({
            "value": float(note_pension_value),
            "exact": False,
            "source_metric": "capital_structure.net_pension_liability_filing_note_proxy",
            "formula": "latest_filing_note_defined_benefit_pension_liability_on_or_before_asof",
            "support_override": (
                "filing_note_defined_benefit_pension_proxy"
                + (";other_postretirement_excluded_from_pension_metric" if note_other_postretirement_value is not None else "")
            ),
            "component_meta": {
                "mode": "filing_note_split_components",
                "filing_note_reference": note_component_meta,
                "combined_proxy_reference": {
                    "proxy_total": proxy_total_meta,
                    "proxy_current": proxy_current_meta,
                    "proxy_noncurrent": proxy_noncurrent_meta,
                }
                if proxy_total_meta is not None or proxy_current_meta is not None or proxy_noncurrent_meta is not None
                else None,
            },
            "other_postretirement_value": (
                None if note_other_postretirement_value is None else float(note_other_postretirement_value)
            ),
            "other_postretirement_exact": False,
            "other_postretirement_source_metric": (
                "capital_structure.other_postretirement_benefit_liability_filing_note_proxy"
                if note_other_postretirement_value is not None
                else None
            ),
            "other_postretirement_formula": (
                "latest_filing_note_other_postretirement_liability_on_or_before_asof"
                if note_other_postretirement_value is not None
                else "unavailable"
            ),
            "other_postretirement_support_override": (
                "filing_note_other_postretirement_proxy"
                if note_other_postretirement_value is not None
                else None
            ),
            "other_postretirement_component_meta": (
                {
                    "mode": "filing_note_split_components",
                    "filing_note_reference": note_component_meta,
                }
                if note_other_postretirement_value is not None
                else None
            ),
            "combined_retirement_value": combined_retirement_value,
            "combined_retirement_exact": False,
            "combined_retirement_source_metric": "capital_structure.combined_retirement_liability_filing_note_proxy",
            "combined_retirement_formula": (
                "latest_filing_note_defined_benefit_pension_plus_other_postretirement_liability_on_or_before_asof"
                if note_other_postretirement_value is not None
                else "latest_filing_note_defined_benefit_pension_liability_on_or_before_asof + 0_assumed_missing_other_postretirement"
            ),
            "combined_retirement_support_override": (
                "filing_note_combined_retirement_proxy"
                if note_other_postretirement_value is not None
                else "filing_note_combined_retirement_uses_pension_only"
            ),
            "combined_retirement_component_meta": {
                "mode": "filing_note_split_components",
                "filing_note_reference": note_component_meta,
                "proxy_reference": {
                    "proxy_total": proxy_total_meta,
                    "proxy_current": proxy_current_meta,
                    "proxy_noncurrent": proxy_noncurrent_meta,
                }
                if proxy_total_meta is not None or proxy_current_meta is not None or proxy_noncurrent_meta is not None
                else None,
                "other_postretirement_missing_assumed_zero": note_other_postretirement_value is None,
            },
        }, regime="pension_proxy_split_note", regime_source="filing_note_split", regime_meta={
            "mode": "filing_note_split",
            "filing_note_reference": note_component_meta,
            "carryforward_used": bool((note_component_meta or {}).get("carryforward_used")),
            "filing_age_days": (note_component_meta or {}).get("filing_age_days"),
        })

    normalized_proxy_total, proxy_total_clipped = _normalized_nonnegative(proxy_total_value)
    if normalized_proxy_total is not None:
        return _with_regime({
            "value": None,
            "exact": False,
            "source_metric": "capital_structure.net_pension_liability_companyfacts_proxy",
            "formula": "combined_proxy_not_promoted_without_pension_specific_or_split_filing_support",
            "support_override": (
                "combined_pension_and_postretirement_liability_not_separable"
                + (";overfunded_pension_excluded_from_liability_view" if proxy_total_clipped else "")
            ),
            "component_meta": {
                "mode": "proxy_total_companyfacts_unseparated",
                "proxy_total": proxy_total_meta,
                "filing_note_reference": note_component_meta,
            },
            "other_postretirement_value": None,
            "other_postretirement_exact": False,
            "other_postretirement_source_metric": None,
            "other_postretirement_formula": "unavailable",
            "other_postretirement_support_override": "combined_pension_and_postretirement_liability_not_separable",
            "other_postretirement_component_meta": {
                "mode": "proxy_total_companyfacts_unseparated",
                "proxy_total": proxy_total_meta,
                "filing_note_reference": note_component_meta,
            },
            "combined_retirement_value": normalized_proxy_total,
            "combined_retirement_exact": False,
            "combined_retirement_source_metric": "capital_structure.combined_retirement_liability_companyfacts_proxy",
            "combined_retirement_formula": "latest_combined_retirement_liability_on_or_before_asof",
            "combined_retirement_support_override": (
                "combined_retirement_proxy_companyfacts_unseparated"
                + (";overfunded_retirement_liability_excluded_from_liability_view" if proxy_total_clipped else "")
            ),
            "combined_retirement_component_meta": {
                "mode": "proxy_total_companyfacts_unseparated",
                "proxy_total": proxy_total_meta,
                "filing_note_reference": note_component_meta,
            },
        }, regime="combined_retirement_only", regime_source="companyfacts_combined_proxy_total", regime_meta={
            "mode": "combined_retirement_only",
            "proxy_total": proxy_total_meta,
            "filing_note_reference": note_component_meta,
        })

    if proxy_current_value is not None or proxy_noncurrent_value is not None:
        current_component, current_clipped = _normalized_nonnegative(proxy_current_value)
        noncurrent_component, noncurrent_clipped = _normalized_nonnegative(proxy_noncurrent_value)
        combined_proxy_value = float((current_component or 0.0) + (noncurrent_component or 0.0))
        return _with_regime({
            "value": None,
            "exact": False,
            "source_metric": "capital_structure.net_pension_liability_companyfacts_proxy",
            "formula": "combined_proxy_not_promoted_without_pension_specific_or_split_filing_support",
            "support_override": (
                "combined_pension_and_postretirement_liability_not_separable"
                + (";overfunded_pension_excluded_from_liability_view" if current_clipped or noncurrent_clipped else "")
            ),
            "component_meta": {
                "mode": "proxy_split_companyfacts_unseparated",
                "current": proxy_current_meta,
                "noncurrent": proxy_noncurrent_meta,
                "filing_note_reference": note_component_meta,
            },
            "other_postretirement_value": None,
            "other_postretirement_exact": False,
            "other_postretirement_source_metric": None,
            "other_postretirement_formula": "unavailable",
            "other_postretirement_support_override": "combined_pension_and_postretirement_liability_not_separable",
            "other_postretirement_component_meta": {
                "mode": "proxy_split_companyfacts_unseparated",
                "current": proxy_current_meta,
                "noncurrent": proxy_noncurrent_meta,
                "filing_note_reference": note_component_meta,
            },
            "combined_retirement_value": combined_proxy_value,
            "combined_retirement_exact": False,
            "combined_retirement_source_metric": "capital_structure.combined_retirement_liability_companyfacts_proxy",
            "combined_retirement_formula": "sum_available_combined_retirement_liability_components",
            "combined_retirement_support_override": (
                "combined_retirement_proxy_companyfacts_unseparated"
                + (
                    ";overfunded_retirement_liability_excluded_from_liability_view"
                    if current_clipped or noncurrent_clipped
                    else ""
                )
            ),
            "combined_retirement_component_meta": {
                "mode": "proxy_split_companyfacts_unseparated",
                "current": proxy_current_meta,
                "noncurrent": proxy_noncurrent_meta,
                "filing_note_reference": note_component_meta,
            },
        }, regime="combined_retirement_only", regime_source="companyfacts_combined_proxy_split", regime_meta={
            "mode": "combined_retirement_only",
            "current": proxy_current_meta,
            "noncurrent": proxy_noncurrent_meta,
            "filing_note_reference": note_component_meta,
        })

    if note_regime_hint == "defined_contribution_only":
        return _with_regime({
            "value": None,
            "exact": False,
            "source_metric": None,
            "formula": "defined_contribution_only_filing_text_no_liability_components",
            "support_override": "defined_contribution_only_no_retirement_liability_components",
            "component_meta": note_component_meta,
            "other_postretirement_value": None,
            "other_postretirement_exact": False,
            "other_postretirement_source_metric": None,
            "other_postretirement_formula": "unavailable",
            "other_postretirement_support_override": None,
            "other_postretirement_component_meta": None,
            "combined_retirement_value": None,
            "combined_retirement_exact": False,
            "combined_retirement_source_metric": None,
            "combined_retirement_formula": "unavailable",
            "combined_retirement_support_override": None,
            "combined_retirement_component_meta": None,
        }, regime="defined_contribution_only", regime_source="filing_text_hint", regime_meta=note_component_meta)

    return _with_regime({
        "value": None,
        "exact": False,
        "source_metric": None,
        "formula": "unavailable",
        "support_override": None,
        "component_meta": None,
        "other_postretirement_value": None,
        "other_postretirement_exact": False,
        "other_postretirement_source_metric": None,
        "other_postretirement_formula": "unavailable",
        "other_postretirement_support_override": None,
        "other_postretirement_component_meta": None,
        "combined_retirement_value": None,
        "combined_retirement_exact": False,
        "combined_retirement_source_metric": None,
        "combined_retirement_formula": "unavailable",
        "combined_retirement_support_override": None,
        "combined_retirement_component_meta": None,
    }, regime="retirement_not_surfaced", regime_source="no_supported_retirement_liability_path")


def _effective_cash_equivalents_value(
    cash_exact: Dict[str, Any],
    *,
    companyfacts: Dict[str, Any] | None,
    as_of_time: str,
) -> Dict[str, Any]:
    companyfacts_value, companyfacts_meta = _latest_companyfacts_point_value(
        companyfacts,
        CASH_EQ_COMPANYFACTS_CONCEPTS,
        as_of_time=as_of_time,
    )

    if _exact(cash_exact) and _value(cash_exact) is not None:
        if _companyfacts_candidate_is_fresher(
            existing_node=cash_exact,
            companyfacts_meta=companyfacts_meta,
            candidate_value=companyfacts_value,
        ):
            return {
                "value": companyfacts_value,
                "exact": True,
                "source_metric": "liquidity.cash_and_equivalents_companyfacts_exact",
                "formula": "cash_and_equivalents_companyfacts_exact",
                "support_override": "companyfacts_cash_exact_newer_than_provider_direct",
                "component_meta": companyfacts_meta,
            }
        return {
            "value": _value(cash_exact),
            "exact": True,
            "source_metric": "liquidity.cash_and_equivalents_statement_direct",
            "formula": "cash_and_equivalents_statement_direct",
            "support_override": None,
            "component_meta": cash_exact.get("component_breakdown"),
        }

    if companyfacts_value is not None:
        return {
            "value": companyfacts_value,
            "exact": True,
            "source_metric": "liquidity.cash_and_equivalents_companyfacts_exact",
            "formula": "cash_and_equivalents_companyfacts_exact",
            "support_override": "companyfacts_cash_exact_fallback",
            "component_meta": companyfacts_meta,
        }

    return {
        "value": _value(cash_exact),
        "exact": False,
        "source_metric": "liquidity.cash_and_equivalents_statement_direct",
        "formula": "cash_and_equivalents_statement_direct",
        "support_override": None,
        "component_meta": cash_exact.get("component_breakdown"),
    }


def _market_availability_adjustment(
    overrides: Dict[str, Any] | None,
    *,
    company_id: str | None,
    as_of_time: str,
) -> Dict[str, Any] | None:
    if not overrides or not company_id:
        return None
    entries = overrides.get(str(company_id)) or []
    if not isinstance(entries, list):
        return None
    as_of_dt = _parse_iso_date(as_of_time)
    if as_of_dt is None:
        return None

    chosen: Dict[str, Any] | None = None
    chosen_start: datetime | None = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        value = entry.get("not_freely_transferable_cash")
        if value is None:
            continue
        start_dt = _parse_iso_date(entry.get("effective_start") or entry.get("effective_date"))
        end_dt = _parse_iso_date(entry.get("effective_end") or entry.get("valid_through"))
        if start_dt is not None and as_of_dt < start_dt:
            continue
        if end_dt is not None and as_of_dt > end_dt:
            continue
        if chosen is None or (start_dt is not None and (chosen_start is None or start_dt > chosen_start)):
            chosen = entry
            chosen_start = start_dt

    if chosen is None:
        return None

    try:
        adjustment_value = float(chosen["not_freely_transferable_cash"])
    except Exception:  # noqa: BLE001
        return None

    return {
        "value": adjustment_value,
        "label": chosen.get("label"),
        "effective_start": chosen.get("effective_start") or chosen.get("effective_date"),
        "effective_end": chosen.get("effective_end") or chosen.get("valid_through"),
        "reported_period_end": chosen.get("reported_period_end"),
        "filing_date": chosen.get("filing_date"),
        "source": chosen.get("source"),
    }


def _extract_lease_reference_value(reference: Dict[str, Any]) -> Dict[str, Any] | None:
    candidates: list[Dict[str, Any]] = []

    direct_total = reference.get("direct_total_reference")
    if direct_total and direct_total.get("value") is not None:
        direct_total_value = float(direct_total["value"])
        if direct_total_value < 0:
            direct_total_value = None
        end_dt = max(
            (
                _parse_iso_date((component or {}).get("end"))
                for component in (direct_total.get("components") or [None])
            ),
            default=None,
        )
        if end_dt is not None and direct_total_value is not None:
            candidates.append(
                {
                    "value": direct_total_value,
                    "end_dt": end_dt,
                    "source_mode": "direct_total_reference",
                }
            )

    partial = reference.get("partial_component_reference")
    if partial and partial.get("value") is not None:
        partial_value = float(partial["value"])
        if partial_value < 0:
            partial_value = None
        end_dt = max(
            (
                _parse_iso_date(component.get("end"))
                for component in ((partial.get("current_components") or []) + (partial.get("noncurrent_components") or []))
                if component.get("end")
            ),
            default=None,
        )
        if end_dt is not None and partial_value is not None:
            candidates.append(
                {
                    "value": partial_value,
                    "end_dt": end_dt,
                    "source_mode": "partial_component_reference",
                }
            )

    payments_due = reference.get("payments_due_reference")
    if payments_due and payments_due.get("derived_total_value") is not None:
        derived_total_value = float(payments_due["derived_total_value"])
        if derived_total_value < 0:
            derived_total_value = None
        end_dt = _parse_iso_date((payments_due.get("payments_due") or {}).get("end"))
        if end_dt is not None and derived_total_value is not None:
            candidates.append(
                {
                    "value": derived_total_value,
                    "end_dt": end_dt,
                    "source_mode": "payments_due_reference",
                }
            )

    if not candidates:
        return None

    source_priority = {
        "partial_component_reference": 2,
        "direct_total_reference": 1,
        "payments_due_reference": 0,
    }
    candidates.sort(key=lambda candidate: (candidate["end_dt"], source_priority[candidate["source_mode"]]))
    return candidates[-1]


def _fresh_rou_asset_available(reference: Dict[str, Any], as_of_time: str) -> bool:
    rou_reference = reference.get("right_of_use_asset_reference") or {}
    component = (rou_reference.get("components") or [None])[0] or {}
    rou_end_dt = _parse_iso_date(component.get("end"))
    as_of_dt = _parse_iso_date(as_of_time)
    if rou_end_dt is None or as_of_dt is None:
        return False
    return (as_of_dt - rou_end_dt).days <= LEASE_ROU_FRESH_MAX_AGE_DAYS


def _stale_corroborated_lease_reference_value(reference: Dict[str, Any], as_of_time: str) -> Dict[str, Any] | None:
    extracted = _extract_lease_reference_value(reference)
    as_of_dt = _parse_iso_date(as_of_time)
    if extracted is None or extracted["end_dt"] is None or as_of_dt is None:
        return None
    age_days = (as_of_dt - extracted["end_dt"]).days
    if age_days < 0 or age_days > LEASE_STALE_CARRY_FORWARD_MAX_AGE_DAYS:
        return None
    if not _fresh_rou_asset_available(reference, as_of_time):
        return None
    extracted["age_days"] = age_days
    return extracted


def _fresh_lease_reference_value(reference: Dict[str, Any], as_of_time: str) -> Dict[str, Any] | None:
    extracted = _extract_lease_reference_value(reference)
    as_of_dt = _parse_iso_date(as_of_time)
    if extracted is None or extracted["end_dt"] is None or as_of_dt is None:
        return None
    age_days = (as_of_dt - extracted["end_dt"]).days
    if age_days < 0 or age_days > LEASE_ROU_FRESH_MAX_AGE_DAYS:
        return None
    extracted["age_days"] = age_days
    return extracted


def _effective_liquidity_component_values(
    *,
    cash_grouped: Dict[str, Any],
    cash_exact: Dict[str, Any],
    restricted_cash_sec: Dict[str, Any],
    marketable_sec: Dict[str, Any],
    restricted_cash: Dict[str, Any],
    marketable: Dict[str, Any],
) -> Dict[str, Any]:
    grouped_cash_value = _value(cash_grouped)
    cash_exact_value = _value(cash_exact)

    restricted_cash_inferred_zero = (
        restricted_cash_sec.get("support_mode") == "unsupported"
        and restricted_cash_sec.get("missing_reason") == "sec_concept_absent"
    )
    marketable_inferred_zero = (
        marketable_sec.get("support_mode") == "unsupported"
        and marketable_sec.get("missing_reason") == "sec_concept_absent"
    )

    restricted_cash_value = (
        _value(restricted_cash_sec)
        if _exact(restricted_cash_sec)
        else (0.0 if restricted_cash_inferred_zero else (_value(restricted_cash) if _exact(restricted_cash) else None))
    )
    marketable_value = (
        _value(marketable_sec)
        if _exact(marketable_sec)
        else (0.0 if marketable_inferred_zero else (_value(marketable) if _exact(marketable) else None))
    )

    restricted_cash_zero_reconciled = False
    marketable_zero_reconciled = False
    restricted_cash_market_default_zero = False

    if (
        restricted_cash_value is None
        and _exact(cash_grouped)
        and cash_exact_value is None
        and restricted_cash_sec.get("support_mode") == "unsupported"
        and restricted_cash_sec.get("missing_reason") == "sec_concept_unavailable"
    ):
        restricted_cash_value = 0.0
        restricted_cash_inferred_zero = True
        restricted_cash_market_default_zero = True

    if (
        restricted_cash_value is None
        and cash_exact_value is not None
        and marketable_value is not None
        and _approximately_equal(grouped_cash_value, cash_exact_value + marketable_value)
    ):
        restricted_cash_value = 0.0
        restricted_cash_inferred_zero = True
        restricted_cash_zero_reconciled = True

    if (
        marketable_value is None
        and cash_exact_value is not None
        and restricted_cash_value is not None
        and _approximately_equal(grouped_cash_value, cash_exact_value + restricted_cash_value)
    ):
        marketable_value = 0.0
        marketable_inferred_zero = True
        marketable_zero_reconciled = True

    if (
        restricted_cash_value is None
        and marketable_value is None
        and cash_exact_value is not None
        and _approximately_equal(grouped_cash_value, cash_exact_value)
    ):
        restricted_cash_value = 0.0
        marketable_value = 0.0
        restricted_cash_inferred_zero = True
        marketable_inferred_zero = True
        restricted_cash_zero_reconciled = True
        marketable_zero_reconciled = True

    return {
        "restricted_cash_value": restricted_cash_value,
        "marketable_value": marketable_value,
        "restricted_cash_inferred_zero": restricted_cash_inferred_zero,
        "marketable_inferred_zero": marketable_inferred_zero,
        "restricted_cash_zero_reconciled": restricted_cash_zero_reconciled,
        "marketable_zero_reconciled": marketable_zero_reconciled,
        "restricted_cash_market_default_zero": restricted_cash_market_default_zero,
    }


def _companyfacts_priority_ttm(
    companyfacts: Dict[str, Any] | None,
    concepts: list[str],
    *,
    as_of_date: str,
) -> tuple[float | None, Dict[str, Any] | None]:
    if companyfacts is None:
        return None, None
    for concept_name in concepts:
        value, meta = _compute_ttm_from_concept(companyfacts, concept_name, as_of_date)
        if value is not None:
            return value, meta
    return None, None


def _companyfacts_depreciation_ttm(
    companyfacts: Dict[str, Any] | None,
    *,
    as_of_date: str,
) -> tuple[float | None, Dict[str, Any] | None, bool, list[str] | None]:
    if companyfacts is None:
        return None, None, False, None
    for concept_group in DEPRECIATION_TTM_CONCEPT_GROUPS:
        if len(concept_group) == 1:
            value, meta = _compute_ttm_from_concept(companyfacts, concept_group[0], as_of_date)
            if value is not None:
                if concept_group[0] == "Depreciation":
                    return value, meta, False, ["partial_depreciation_without_full_amortization"]
                return value, meta, True, None
        else:
            parts = []
            parts_meta = []
            for concept_name in concept_group:
                part_value, part_meta = _compute_ttm_from_concept(companyfacts, concept_name, as_of_date)
                if part_value is None:
                    parts = []
                    break
                parts.append(part_value)
                parts_meta.append(part_meta)
            if parts:
                return float(sum(parts)), {
                    "mode": "sum_concepts",
                    "components": parts_meta,
                    "formula": "sum_component_ttm_values",
                }, True, None
    return None, None, False, None


def _effective_operating_earnings_baseline(
    *,
    ebitda: Dict[str, Any],
    net_income: Dict[str, Any],
    interest_expense: Dict[str, Any],
    companyfacts: Dict[str, Any] | None,
    as_of_time: str,
) -> Dict[str, Any]:
    ebitda_value = _value(ebitda)
    if ebitda_value is not None:
        return {
            "value": ebitda_value,
            "exact": _exact(ebitda),
            "missing_reason": None,
            "quality_flags": ebitda.get("quality_flags"),
            "component_breakdown": {
                "baseline_source_metric": "operating.ebitda_ltm_provider_direct",
                "baseline_value": ebitda_value,
                "formula": "provider_direct_ebitda_baseline",
            },
        }

    net_income_value = _value(net_income)
    if net_income.get("support_mode") != "exact" or net_income_value is None:
        return {
            "value": None,
            "exact": False,
            "missing_reason": "component_unavailable",
            "quality_flags": ["component_unavailable", "smart_metric_not_promoted"],
            "component_breakdown": {
                "formula": "unavailable",
            },
        }

    as_of_date = as_of_time[:10]
    interest_value = _value(interest_expense) if _exact(interest_expense) else None
    interest_meta = None
    interest_source = None
    if interest_value is not None:
        interest_meta = interest_expense.get("component_breakdown")
        interest_source = "capital_structure.interest_expense_statement_direct"
    else:
        interest_value, interest_meta = _companyfacts_priority_ttm(
            companyfacts,
            OPERATING_EARNINGS_INTEREST_CONCEPTS,
            as_of_date=as_of_date,
        )
        if interest_value is not None:
            interest_source = "sec_companyfacts_interest_expense"

    tax_value, tax_meta = _companyfacts_priority_ttm(
        companyfacts,
        OPERATING_EARNINGS_TAX_CONCEPTS,
        as_of_date=as_of_date,
    )
    depreciation_value, depreciation_meta, depreciation_exact, depreciation_quality_flags = _companyfacts_depreciation_ttm(
        companyfacts,
        as_of_date=as_of_date,
    )

    if interest_value is None or tax_value is None or depreciation_value is None:
        return {
            "value": None,
            "exact": False,
            "missing_reason": "component_unavailable",
            "quality_flags": ["component_unavailable", "smart_metric_not_promoted"],
            "component_breakdown": {
                "net_income_ttm_provider_direct": net_income_value,
                "interest_expense": interest_value,
                "income_tax_expense_benefit_ttm": tax_value,
                "depreciation_amortization_ttm": depreciation_value,
                "formula": "net_income + interest_expense + income_tax_expense_benefit + depreciation_amortization",
            },
        }

    quality_flags = list(depreciation_quality_flags or [])
    exact = depreciation_exact
    return {
        "value": float(net_income_value + interest_value + tax_value + depreciation_value),
        "exact": exact,
        "missing_reason": None,
        "quality_flags": quality_flags or None,
        "component_breakdown": {
            "baseline_source_metric": "earnings.net_income_ttm_provider_direct",
            "baseline_value": net_income_value,
            "net_income_ttm_provider_direct": net_income_value,
            "interest_expense": interest_value,
            "interest_expense_source_metric": interest_source,
            "interest_expense_meta": interest_meta,
            "income_tax_expense_benefit_ttm": tax_value,
            "income_tax_expense_benefit_meta": tax_meta,
            "depreciation_amortization_ttm": depreciation_value,
            "depreciation_amortization_meta": depreciation_meta,
            "earnings_support_override": "net_income_plus_interest_tax_depreciation_amortization",
            "formula": "net_income + interest_expense + income_tax_expense_benefit + depreciation_amortization",
        },
    }


def _grouped_cash_proxy_can_promote_to_exact_cash_baseline(
    *,
    cash_grouped: Dict[str, Any],
    cash_exact: Dict[str, Any],
    marketable_sec: Dict[str, Any],
    marketable: Dict[str, Any],
    marketable_inferred_zero: bool,
) -> bool:
    component_breakdown = cash_grouped.get("component_breakdown") or {}
    return (
        cash_grouped.get("support_mode") == "proxy_missing_component"
        and cash_grouped.get("missing_reason") == "cash_or_sti_component_missing"
        and _value(cash_exact) is None
        and component_breakdown.get("mode") == "partial_cash_stack"
        and component_breakdown.get("cash") is not None
        and component_breakdown.get("short_term_investments") is None
        and marketable_inferred_zero
        and marketable_sec.get("missing_reason") == "sec_concept_absent"
        and marketable.get("support_mode") == "unsupported"
        and marketable.get("missing_reason") == "not_disclosed"
    )


def _grouped_cash_proxy_can_complete_with_exact_marketable_securities(
    *,
    cash_grouped: Dict[str, Any],
    marketable_value: float | None,
) -> bool:
    component_breakdown = cash_grouped.get("component_breakdown") or {}
    return (
        cash_grouped.get("support_mode") == "proxy_missing_component"
        and cash_grouped.get("missing_reason") == "cash_or_sti_component_missing"
        and marketable_value is not None
        and component_breakdown.get("mode") == "partial_cash_stack"
        and component_breakdown.get("cash") is not None
        and component_breakdown.get("short_term_investments") is None
    )


def _effective_total_debt_baseline(
    *,
    total_debt: Dict[str, Any],
    current_debt: Dict[str, Any],
    long_term_debt: Dict[str, Any],
    companyfacts: Dict[str, Any] | None = None,
    as_of_time: str | None = None,
) -> Dict[str, Any]:
    total_debt_value = _value(total_debt)
    total_debt_exact = _exact(total_debt) and total_debt_value is not None
    current_debt_value = _value(current_debt)
    long_term_debt_value = _value(long_term_debt)
    current_debt_exact = _exact(current_debt) and current_debt_value is not None
    long_term_debt_exact = _exact(long_term_debt) and long_term_debt_value is not None
    total_debt_breakdown = total_debt.get("component_breakdown") or {}
    current_debt_breakdown = current_debt.get("component_breakdown") or {}
    overlap_delta = (
        float(total_debt_value - long_term_debt_value)
        if total_debt_value is not None and long_term_debt_value is not None
        else None
    )
    overlap_is_material = (
        overlap_delta is not None
        and overlap_delta > 0.0
        and overlap_delta > max(1_000_000.0, abs(float(total_debt_value)) * 0.005)
    )
    companyfacts_total_debt_value = None
    companyfacts_total_debt_meta = None
    companyfacts_total_debt_exact = False
    if companyfacts is not None and as_of_time is not None:
        companyfacts_total_debt_value, companyfacts_support_mode, _, companyfacts_total_debt_meta, _ = (
            _build_sec_core_metric("capital_structure.total_debt_provider_direct", companyfacts, as_of_time[:10])
        )
        companyfacts_total_debt_exact = (
            companyfacts_support_mode == "exact" and companyfacts_total_debt_value is not None
        )
    if (
        total_debt_exact
        and total_debt_breakdown.get("mode") == "current_plus_noncurrent_debt_plus_short_term_borrowings"
        and long_term_debt_exact
        and overlap_is_material
        and (
            current_debt_breakdown.get("concept") == "LongTermDebtCurrent"
            or (total_debt_breakdown.get("current") or {}).get("concept") == "LongTermDebtCurrent"
        )
    ):
        return {
            "value": long_term_debt_value,
            "exact": True,
            "source_metric": "capital_structure.long_term_debt_statement_direct",
            "formula": "exact_long_term_debt_statement_direct_due_to_current_short_term_overlap",
            "override_reason": "short_term_borrowings_overlap_current_debt",
        }

    # Never let a fresher companyfacts fallback undercut an already-supported
    # debt baseline from the artifact. That would make debt-like obligations
    # smaller than total debt after downstream recomputation.
    if (
        total_debt_value is not None
        and _is_supported(total_debt)
        and companyfacts_total_debt_exact
        and companyfacts_total_debt_value is not None
        and companyfacts_total_debt_value < total_debt_value
        and not _approximately_equal(companyfacts_total_debt_value, total_debt_value)
    ):
        return {
            "value": total_debt_value,
            "exact": total_debt_exact,
            "source_metric": "capital_structure.total_debt_provider_direct",
            "formula": "baseline_total_debt_provider_direct",
            "override_reason": "preserve_supported_total_debt_floor",
        }

    if total_debt_exact and companyfacts_total_debt_exact and _companyfacts_candidate_is_fresher(
        existing_node=total_debt,
        companyfacts_meta=companyfacts_total_debt_meta,
        candidate_value=companyfacts_total_debt_value,
    ):
        return {
            "value": companyfacts_total_debt_value,
            "exact": True,
            "source_metric": "capital_structure.total_debt_companyfacts_exact",
            "formula": "companyfacts_total_debt_provider_direct",
            "override_reason": "fresher_companyfacts_total_debt",
        }

    if total_debt_exact:
        return {
            "value": total_debt_value,
            "exact": True,
            "source_metric": "capital_structure.total_debt_provider_direct",
            "formula": "baseline_total_debt_provider_direct",
            "override_reason": None,
        }

    if companyfacts_total_debt_exact:
        return {
            "value": companyfacts_total_debt_value,
            "exact": True,
            "source_metric": "capital_structure.total_debt_companyfacts_exact",
            "formula": "companyfacts_total_debt_provider_direct",
            "override_reason": "companyfacts_total_debt_fallback",
        }

    if current_debt_exact and long_term_debt_exact:
        return {
            "value": current_debt_value + long_term_debt_value,
            "exact": True,
            "source_metric": (
                "capital_structure.current_debt_statement_direct + "
                "capital_structure.long_term_debt_statement_direct"
            ),
            "formula": "current_debt_statement_direct + long_term_debt_statement_direct",
            "override_reason": None,
        }

    total_debt_missing_reason = total_debt.get("missing_reason")
    total_debt_mode = (total_debt.get("component_breakdown") or {}).get("mode")
    supports_single_component_override = total_debt_missing_reason in {
        "debt_component_missing",
        "debt_component_period_mismatch",
    } and total_debt_mode in {
        "partial_debt_stack",
        "current_plus_noncurrent_debt",
        "current_plus_noncurrent_debt_plus_short_term_borrowings",
    }

    if (
        supports_single_component_override
        and total_debt_value is not None
        and current_debt_exact
        and long_term_debt.get("support_mode") == "unsupported"
        and _approximately_equal(total_debt_value, current_debt_value)
    ):
        return {
            "value": total_debt_value,
            "exact": True,
            "source_metric": "capital_structure.current_debt_statement_direct",
            "formula": "exact_current_debt_statement_direct + 0_inferred_long_term_debt",
            "override_reason": "single_statement_component_matches_total_debt",
        }

    if (
        supports_single_component_override
        and total_debt_value is not None
        and long_term_debt_exact
        and current_debt.get("support_mode") == "unsupported"
        and _approximately_equal(total_debt_value, long_term_debt_value)
    ):
        return {
            "value": total_debt_value,
            "exact": True,
            "source_metric": "capital_structure.long_term_debt_statement_direct",
            "formula": "exact_long_term_debt_statement_direct + 0_inferred_current_debt",
            "override_reason": "single_statement_component_matches_total_debt",
        }

    if total_debt_value is not None:
        return {
            "value": total_debt_value,
            "exact": False,
            "source_metric": "capital_structure.total_debt_provider_direct",
            "formula": "baseline_total_debt_provider_direct",
            "override_reason": None,
        }

    if current_debt_value is not None or long_term_debt_value is not None:
        return {
            "value": float((current_debt_value or 0.0) + (long_term_debt_value or 0.0)),
            "exact": False,
            "source_metric": (
                "capital_structure.current_debt_statement_direct + "
                "capital_structure.long_term_debt_statement_direct"
            ),
            "formula": "sum_available_statement_debt_components",
            "override_reason": None,
        }

    return {
        "value": None,
        "exact": False,
        "source_metric": None,
        "formula": "unavailable",
        "override_reason": None,
    }


def _row_may_need_companyfacts(features: Dict[str, Any]) -> bool:
    cash_exact = _node(features, "liquidity.cash_and_equivalents_statement_direct")
    total_debt = _node(features, "capital_structure.total_debt_provider_direct")
    ebitda = _node(features, "operating.ebitda_ltm_provider_direct")
    net_income = _node(features, "earnings.net_income_ttm_provider_direct")
    pension = _node(features, "capital_structure.net_pension_liability")

    return (
        cash_exact.get("support_mode") != "exact"
        or total_debt.get("support_mode") != "exact"
        or (_value(ebitda) is None and _exact(net_income))
        or not _is_supported(pension)
    )


def _build_fail_open_smart_metrics(
    *,
    as_of_time: str,
    computed_at: str,
    provenance_sources: list[str],
    error_type: str,
    error_message: str,
) -> Dict[str, Dict[str, Any]]:
    missing_reason = "company_processing_timeout" if error_type == "company_processing_timeout" else "company_processing_failed"
    component_breakdown = {
        "error_type": error_type,
        "error_message": str(error_message).strip()[:240],
    }
    return {
        metric_name: _feature_template(
            metric_name=metric_name,
            as_of_time=as_of_time,
            computed_at=computed_at,
            support_mode="unsupported",
            value=None,
            unit=SMART_METRIC_UNITS[metric_name],
            provenance_sources=provenance_sources,
            missing_reason=missing_reason,
            component_breakdown=component_breakdown,
            quality_flags=["smart_metric_fail_open", error_type],
        )
        for metric_name in SMART_METRIC_NAMES
    }


def _effective_lease_liability_value(
    lease_sec: Dict[str, Any],
    *,
    as_of_time: str | None = None,
    total_debt_value: float | None = None,
) -> Dict[str, Any]:
    lease_value = _value(lease_sec)
    if _exact(lease_sec) and lease_value is not None:
        if lease_value < 0:
            return {
                "value": None,
                "exact": False,
                "inferred_zero": False,
                "support_override": "negative_lease_liability_exact_value_ignored",
            }
        return {
            "value": lease_value,
            "exact": True,
            "inferred_zero": False,
            "support_override": None,
        }
    if (
        lease_sec.get("support_mode") == "unsupported"
        and lease_sec.get("missing_reason") == "sec_concept_absent"
    ):
        return {
            "value": 0.0,
            "exact": True,
            "inferred_zero": True,
            "support_override": None,
        }
    if (
        as_of_time is not None
        and lease_sec.get("support_mode") == "unsupported"
        and lease_sec.get("missing_reason") == "sec_concept_unavailable"
    ):
        component_breakdown = lease_sec.get("component_breakdown") or {}
        operating_reference = component_breakdown.get("operating_reference") or {}
        finance_reference = component_breakdown.get("finance_reference") or {}
        aggregate_reference = component_breakdown.get("aggregate_total_reference") or {}
        operating_present = bool(operating_reference.get("present"))
        finance_present = bool(finance_reference.get("present"))
        aggregate_fresh = _fresh_lease_reference_value(aggregate_reference, as_of_time)
        operating_fresh = _fresh_lease_reference_value(operating_reference, as_of_time)
        finance_fresh = _fresh_lease_reference_value(finance_reference, as_of_time)
        operating_stale = _stale_corroborated_lease_reference_value(operating_reference, as_of_time)
        finance_stale = _stale_corroborated_lease_reference_value(finance_reference, as_of_time)

        if aggregate_fresh is not None and not operating_present and not finance_present:
            return {
                "value": float(aggregate_fresh["value"]),
                "exact": True,
                "inferred_zero": False,
                "support_override": "fresh_aggregate_lease_reference",
            }

        if (
            (operating_fresh is not None or operating_stale is not None)
            and (finance_fresh is not None or finance_stale is not None)
        ) or (
            (operating_fresh is not None or operating_stale is not None)
            and not finance_present
        ) or (
            (finance_fresh is not None or finance_stale is not None)
            and not operating_present
        ):
            return {
                "value": float(
                    (operating_fresh or operating_stale or {}).get("value", 0.0)
                    + (finance_fresh or finance_stale or {}).get("value", 0.0)
                ),
                "exact": True,
                "inferred_zero": False,
                "support_override": (
                    "fresh_liability_total_reference"
                    if operating_stale is None and finance_stale is None
                    else (
                        "stale_liability_total_corroborated_by_fresh_rou_asset"
                        if operating_fresh is None and finance_fresh is None
                        else "hybrid_fresh_and_stale_liability_total_reference"
                    )
                ),
            }

        fresh_or_stale_operating = operating_fresh or operating_stale
        fresh_or_stale_finance = finance_fresh or finance_stale
        if total_debt_value is not None and total_debt_value > 0:
            if (
                fresh_or_stale_operating is not None
                and finance_present
                and finance_fresh is None
                and finance_stale is None
                and finance_reference.get("present")
            ):
                finance_extracted = _extract_lease_reference_value(finance_reference)
                finance_value = (finance_extracted or {}).get("value")
                finance_is_immaterial = (
                    finance_value is not None
                    and (
                        finance_value <= LEASE_IMMATERIAL_STALE_COMPONENT_MAX_ABS_USD
                        or (finance_value / float(total_debt_value))
                        <= LEASE_IMMATERIAL_STALE_COMPONENT_MAX_RELATIVE_TO_DEBT
                    )
                )
                if finance_is_immaterial:
                    return {
                        "value": float(fresh_or_stale_operating.get("value", 0.0) + finance_value),
                        "exact": True,
                        "inferred_zero": False,
                        "support_override": "hybrid_fresh_and_immaterial_stale_liability_total_reference",
                    }

            if (
                fresh_or_stale_finance is not None
                and operating_present
                and operating_fresh is None
                and operating_stale is None
                and operating_reference.get("present")
            ):
                operating_extracted = _extract_lease_reference_value(operating_reference)
                operating_value = (operating_extracted or {}).get("value")
                operating_is_immaterial = (
                    operating_value is not None
                    and (
                        operating_value <= LEASE_IMMATERIAL_STALE_COMPONENT_MAX_ABS_USD
                        or (operating_value / float(total_debt_value))
                        <= LEASE_IMMATERIAL_STALE_COMPONENT_MAX_RELATIVE_TO_DEBT
                    )
                )
                if operating_is_immaterial:
                    return {
                        "value": float(operating_value + fresh_or_stale_finance.get("value", 0.0)),
                        "exact": True,
                        "inferred_zero": False,
                        "support_override": "hybrid_fresh_and_immaterial_stale_liability_total_reference",
                    }

        if not operating_present and not finance_present:
            return {
                "value": 0.0,
                "exact": True,
                "inferred_zero": True,
                "support_override": "no_lease_references_present",
            }

        if (
            operating_stale is not None
            and finance_stale is not None
        ) or (
            operating_stale is not None
            and not finance_present
        ) or (
            finance_stale is not None
            and not operating_present
        ):
            return {
                "value": float((operating_stale or {}).get("value", 0.0) + (finance_stale or {}).get("value", 0.0)),
                "exact": True,
                "inferred_zero": False,
                "support_override": "stale_liability_total_corroborated_by_fresh_rou_asset",
            }
    return {
        "value": None,
        "exact": False,
        "inferred_zero": False,
        "support_override": None,
    }


def _smart_value_node(
    *,
    metric_name: str,
    as_of_time: str,
    computed_at: str,
    registry_status: str,
    promotion_rule: str,
    value: float | None,
    unit: str,
    component_breakdown: Dict[str, Any],
    provenance_sources: list[str],
    exact_ready: bool,
    missing_reason: str | None,
) -> Dict[str, Any]:
    components = dict(component_breakdown)
    components["registry_status"] = registry_status
    components["promotion_rule"] = promotion_rule

    if value is None:
        return _feature_template(
            metric_name=metric_name,
            as_of_time=as_of_time,
            computed_at=computed_at,
            support_mode="unsupported",
            value=None,
            unit=unit,
            provenance_sources=provenance_sources,
            missing_reason=missing_reason or "component_unavailable",
            component_breakdown=components,
            quality_flags=[missing_reason or "component_unavailable", "smart_metric_not_promoted"],
        )

    support_mode = "exact" if exact_ready else "proxy_missing_component"
    quality_flags = None if exact_ready else ["smart_metric_partial_feasible"]
    return _feature_template(
        metric_name=metric_name,
        as_of_time=as_of_time,
        computed_at=computed_at,
        support_mode=support_mode,
        value=float(value),
        unit=unit,
        provenance_sources=provenance_sources,
        missing_reason=None,
        component_breakdown=components,
        quality_flags=quality_flags,
    )


def _retirement_regime_feature_node(
    *,
    as_of_time: str,
    computed_at: str,
    provenance_sources: list[str],
    effective_retirement: Dict[str, Any],
) -> Dict[str, Any]:
    regime = str(effective_retirement.get("retirement_regime") or "retirement_not_surfaced")
    component_breakdown = {
        "regime_source": effective_retirement.get("retirement_regime_source") or "unavailable",
        "pension_source_metric": effective_retirement.get("source_metric"),
        "combined_retirement_source_metric": effective_retirement.get("combined_retirement_source_metric"),
    }
    if effective_retirement.get("retirement_regime_component_meta") is not None:
        component_breakdown["classification_reference"] = effective_retirement["retirement_regime_component_meta"]
    return _feature_template(
        metric_name="capital_structure.retirement_obligation_regime",
        as_of_time=as_of_time,
        computed_at=computed_at,
        support_mode="exact",
        value=regime,
        unit="category",
        provenance_sources=provenance_sources,
        missing_reason=None,
        component_breakdown=component_breakdown,
        quality_flags=None,
    )


def materialize_smart_metrics_for_row(
    *,
    row: Dict[str, Any],
    registry: Dict[str, Any],
    computed_at: str,
    provenance_sources: list[str],
    companyfacts: Dict[str, Any] | None = None,
    companyfacts_loader: Callable[[], Dict[str, Any] | None] | None = None,
    retirement_note_loader: Callable[[], Dict[str, Any] | None] | None = None,
    market_availability_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    as_of_time = row["as_of_time"]
    company_id = str(row.get("company_id")) if row.get("company_id") is not None else None
    features = row.setdefault("features", {})
    companyfacts_state: Any = companyfacts if companyfacts is not None else _COMPANYFACTS_UNSET

    def _get_companyfacts() -> Dict[str, Any] | None:
        nonlocal companyfacts_state
        if companyfacts_state is _COMPANYFACTS_UNSET:
            if companyfacts_loader is not None:
                companyfacts_state = companyfacts_loader()
            else:
                companyfacts_state = _load_companyfacts(_infer_companyfacts_path_from_features(features))
        return None if companyfacts_state is _COMPANYFACTS_UNSET else companyfacts_state

    total_debt = _node(features, "capital_structure.total_debt_provider_direct")
    cash_grouped = _node(features, "liquidity.cash_and_short_term_investments_provider_direct")
    cash_exact = _node(features, "liquidity.cash_and_equivalents_statement_direct")
    restricted_cash_sec = _node(features, "liquidity.restricted_cash_sec_exact")
    marketable_sec = _node(features, "liquidity.marketable_securities_sec_exact")
    revolver_sec = _node(features, "liquidity.revolver_undrawn_sec_exact")
    lease_sec = _node(features, "capital_structure.lease_liabilities_sec_exact")
    restricted_cash = _node(features, "liquidity.restricted_cash")
    marketable = _node(features, "liquidity.marketable_securities")
    revolver = _node(features, "liquidity.revolver_undrawn")
    ebitda = _node(features, "operating.ebitda_ltm_provider_direct")
    net_income = _node(features, "earnings.net_income_ttm_provider_direct")
    interest_expense = _node(features, "capital_structure.interest_expense_statement_direct")
    current_debt = _node(features, "capital_structure.current_debt_statement_direct")
    long_term_debt = _node(features, "capital_structure.long_term_debt_statement_direct")
    debt_like_registry = _registry_metric(registry, "debt_like_obligations_normalized")
    pension_registry = _registry_metric(registry, "net_pension_liability")
    other_postretirement_registry = _registry_metric(registry, "other_postretirement_benefit_liability")
    combined_retirement_registry = _registry_metric(registry, "combined_retirement_liability")
    debt_including_pension_registry = _registry_metric(registry, "debt_like_obligations_including_pension")
    debt_including_retirement_registry = _registry_metric(registry, "debt_like_obligations_including_retirement")
    liquidity_registry = _registry_metric(registry, "available_liquidity_normalized")
    earnings_registry = _registry_metric(registry, "operating_earnings_normalized")
    net_debt_registry = _registry_metric(registry, "net_debt_normalized")
    net_debt_including_pension_registry = _registry_metric(registry, "net_debt_including_pension")
    net_debt_including_retirement_registry = _registry_metric(registry, "net_debt_including_retirement")
    gross_leverage_registry = _registry_metric(registry, "gross_leverage_normalized")
    gross_leverage_including_pension_registry = _registry_metric(registry, "gross_leverage_including_pension")
    gross_leverage_including_retirement_registry = _registry_metric(registry, "gross_leverage_including_retirement")
    net_leverage_registry = _registry_metric(registry, "net_leverage_normalized")
    net_leverage_including_pension_registry = _registry_metric(registry, "net_leverage_including_pension")
    net_leverage_including_retirement_registry = _registry_metric(registry, "net_leverage_including_retirement")

    current_debt_value = _value(current_debt)
    long_term_debt_value = _value(long_term_debt)
    effective_total_debt = _effective_total_debt_baseline(
        total_debt=total_debt,
        current_debt=current_debt,
        long_term_debt=long_term_debt,
        companyfacts=_get_companyfacts() if _row_may_need_companyfacts(features) else None,
        as_of_time=as_of_time,
    )
    effective_lease = _effective_lease_liability_value(
        lease_sec,
        as_of_time=as_of_time,
        total_debt_value=effective_total_debt["value"],
    )
    lease_sec_value = effective_lease["value"]
    lease_exact = bool(effective_lease["exact"])
    lease_inferred_zero = bool(effective_lease["inferred_zero"])
    debt_value = effective_total_debt["value"]
    debt_baseline_value = debt_value
    debt_base_formula = effective_total_debt["formula"]
    debt_base_source_metric = effective_total_debt["source_metric"]
    debt_exact_ready = lease_exact and bool(effective_total_debt["exact"])
    raw_lease_sec_value = _value(lease_sec)
    debt_components = {
        "baseline_source_metric": debt_base_source_metric,
        "baseline_value": debt_baseline_value,
        "total_debt_provider_direct": _value(total_debt),
        "current_debt_statement_direct": current_debt_value,
        "long_term_debt_statement_direct": long_term_debt_value,
        "lease_liabilities_sec_exact": lease_sec_value,
        "lease_liabilities_raw_input_value": raw_lease_sec_value,
        "lease_liabilities_inferred_zero": lease_inferred_zero,
        "formula": debt_base_formula
        + (
            " + lease_liabilities_sec_exact"
            if lease_sec_value is not None and not lease_inferred_zero
            else (" + 0_inferred_lease_liabilities" if lease_inferred_zero else "")
        ),
    }
    if effective_total_debt["override_reason"] is not None:
        debt_components["baseline_support_override"] = effective_total_debt["override_reason"]
    if effective_lease.get("support_override") is not None:
        debt_components["lease_support_override"] = effective_lease["support_override"]
    if raw_lease_sec_value is not None and raw_lease_sec_value < 0:
        debt_components["lease_negative_input_ignored"] = True
        debt_exact_ready = False
    if debt_value is not None and lease_sec_value is not None:
        debt_value = debt_value + lease_sec_value
    if debt_baseline_value is not None and debt_value is not None and debt_value < debt_baseline_value:
        debt_value = debt_baseline_value
        debt_components["debt_like_floored_to_total_debt"] = True
        debt_exact_ready = False
    features["capital_structure.debt_like_obligations_normalized"] = _smart_value_node(
        metric_name="capital_structure.debt_like_obligations_normalized",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=debt_like_registry["status"],
        promotion_rule=debt_like_registry["promotion_rule"],
        value=debt_value,
        unit="usd",
        component_breakdown=debt_components,
        provenance_sources=provenance_sources,
        exact_ready=debt_exact_ready,
        missing_reason="component_unavailable" if debt_value is None else None,
    )

    effective_net_pension = _effective_net_pension_liability_value(
        _get_companyfacts(),
        as_of_time=as_of_time,
        retirement_note_loader=retirement_note_loader,
    )
    pension_value = effective_net_pension["value"]
    pension_exact_ready = bool(effective_net_pension["exact"])
    pension_components = {
        "source_metric": effective_net_pension["source_metric"],
        "formula": effective_net_pension["formula"],
    }
    if effective_net_pension.get("component_meta") is not None:
        pension_components["companyfacts_reference"] = effective_net_pension["component_meta"]
    if effective_net_pension.get("support_override") is not None:
        pension_components["support_override"] = effective_net_pension["support_override"]
    features["capital_structure.net_pension_liability"] = _smart_value_node(
        metric_name="capital_structure.net_pension_liability",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=pension_registry["status"],
        promotion_rule=pension_registry["promotion_rule"],
        value=pension_value,
        unit="usd",
        component_breakdown=pension_components,
        provenance_sources=provenance_sources,
        exact_ready=pension_exact_ready,
        missing_reason="component_unavailable" if pension_value is None else None,
    )

    other_postretirement_value = effective_net_pension.get("other_postretirement_value")
    other_postretirement_exact_ready = bool(effective_net_pension.get("other_postretirement_exact"))
    other_postretirement_components = {
        "source_metric": effective_net_pension.get("other_postretirement_source_metric"),
        "formula": effective_net_pension.get("other_postretirement_formula") or "unavailable",
    }
    if effective_net_pension.get("other_postretirement_component_meta") is not None:
        other_postretirement_components["companyfacts_reference"] = effective_net_pension["other_postretirement_component_meta"]
    if effective_net_pension.get("other_postretirement_support_override") is not None:
        other_postretirement_components["support_override"] = effective_net_pension["other_postretirement_support_override"]
    features["capital_structure.other_postretirement_benefit_liability"] = _smart_value_node(
        metric_name="capital_structure.other_postretirement_benefit_liability",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=other_postretirement_registry["status"],
        promotion_rule=other_postretirement_registry["promotion_rule"],
        value=other_postretirement_value,
        unit="usd",
        component_breakdown=other_postretirement_components,
        provenance_sources=provenance_sources,
        exact_ready=other_postretirement_exact_ready,
        missing_reason="component_unavailable" if other_postretirement_value is None else None,
    )

    combined_retirement_value = effective_net_pension.get("combined_retirement_value")
    combined_retirement_exact_ready = bool(effective_net_pension.get("combined_retirement_exact"))
    combined_retirement_components = {
        "source_metric": effective_net_pension.get("combined_retirement_source_metric"),
        "formula": effective_net_pension.get("combined_retirement_formula") or "unavailable",
    }
    if effective_net_pension.get("combined_retirement_component_meta") is not None:
        combined_retirement_components["companyfacts_reference"] = effective_net_pension["combined_retirement_component_meta"]
    if effective_net_pension.get("combined_retirement_support_override") is not None:
        combined_retirement_components["support_override"] = effective_net_pension["combined_retirement_support_override"]
    features["capital_structure.combined_retirement_liability"] = _smart_value_node(
        metric_name="capital_structure.combined_retirement_liability",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=combined_retirement_registry["status"],
        promotion_rule=combined_retirement_registry["promotion_rule"],
        value=combined_retirement_value,
        unit="usd",
        component_breakdown=combined_retirement_components,
        provenance_sources=provenance_sources,
        exact_ready=combined_retirement_exact_ready,
        missing_reason="component_unavailable" if combined_retirement_value is None else None,
    )
    features["capital_structure.retirement_obligation_regime"] = _retirement_regime_feature_node(
        as_of_time=as_of_time,
        computed_at=computed_at,
        provenance_sources=provenance_sources,
        effective_retirement=effective_net_pension,
    )

    debt_including_pension_value = None if debt_value is None else float(debt_value + (pension_value or 0.0))
    debt_including_pension_exact_ready = debt_exact_ready and pension_value is not None and pension_exact_ready
    debt_including_pension_components = {
        "debt_like_obligations_normalized": debt_value,
        "net_pension_liability": pension_value,
        "formula": (
            "debt_like_obligations_normalized + net_pension_liability"
            if pension_value is not None
            else "debt_like_obligations_normalized + 0_assumed_missing_net_pension_liability"
        ),
    }
    if pension_value is None and debt_including_pension_value is not None:
        debt_including_pension_components["pension_missing_assumed_zero"] = True
    if effective_net_pension.get("support_override") is not None:
        debt_including_pension_components["pension_support_override"] = effective_net_pension["support_override"]
    features["capital_structure.debt_like_obligations_including_pension"] = _smart_value_node(
        metric_name="capital_structure.debt_like_obligations_including_pension",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=debt_including_pension_registry["status"],
        promotion_rule=debt_including_pension_registry["promotion_rule"],
        value=debt_including_pension_value,
        unit="usd",
        component_breakdown=debt_including_pension_components,
        provenance_sources=provenance_sources,
        exact_ready=debt_including_pension_exact_ready,
        missing_reason="component_unavailable" if debt_including_pension_value is None else None,
    )

    debt_including_retirement_value = (
        None if debt_value is None else float(debt_value + (combined_retirement_value or 0.0))
    )
    debt_including_retirement_exact_ready = (
        debt_exact_ready and combined_retirement_value is not None and combined_retirement_exact_ready
    )
    debt_including_retirement_components = {
        "debt_like_obligations_normalized": debt_value,
        "combined_retirement_liability": combined_retirement_value,
        "formula": (
            "debt_like_obligations_normalized + combined_retirement_liability"
            if combined_retirement_value is not None
            else "debt_like_obligations_normalized + 0_assumed_missing_combined_retirement_liability"
        ),
    }
    if combined_retirement_value is None and debt_including_retirement_value is not None:
        debt_including_retirement_components["combined_retirement_missing_assumed_zero"] = True
    if effective_net_pension.get("combined_retirement_support_override") is not None:
        debt_including_retirement_components["combined_retirement_support_override"] = (
            effective_net_pension["combined_retirement_support_override"]
        )
    features["capital_structure.debt_like_obligations_including_retirement"] = _smart_value_node(
        metric_name="capital_structure.debt_like_obligations_including_retirement",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=debt_including_retirement_registry["status"],
        promotion_rule=debt_including_retirement_registry["promotion_rule"],
        value=debt_including_retirement_value,
        unit="usd",
        component_breakdown=debt_including_retirement_components,
        provenance_sources=provenance_sources,
        exact_ready=debt_including_retirement_exact_ready,
        missing_reason="component_unavailable" if debt_including_retirement_value is None else None,
    )

    grouped_cash_value = _value(cash_grouped)
    effective_cash_exact = _effective_cash_equivalents_value(
        cash_exact,
        companyfacts=_get_companyfacts() if _row_may_need_companyfacts(features) else None,
        as_of_time=as_of_time,
    )
    cash_exact_value = effective_cash_exact["value"]
    liquidity_components_effective = _effective_liquidity_component_values(
        cash_grouped=cash_grouped,
        cash_exact=cash_exact,
        restricted_cash_sec=restricted_cash_sec,
        marketable_sec=marketable_sec,
        restricted_cash=restricted_cash,
        marketable=marketable,
    )
    restricted_cash_value = liquidity_components_effective["restricted_cash_value"]
    marketable_value = liquidity_components_effective["marketable_value"]
    restricted_cash_inferred_zero = liquidity_components_effective["restricted_cash_inferred_zero"]
    marketable_inferred_zero = liquidity_components_effective["marketable_inferred_zero"]
    restricted_cash_zero_reconciled = liquidity_components_effective["restricted_cash_zero_reconciled"]
    marketable_zero_reconciled = liquidity_components_effective["marketable_zero_reconciled"]
    restricted_cash_market_default_zero = liquidity_components_effective["restricted_cash_market_default_zero"]
    revolver_value = _value(revolver_sec) if _exact(revolver_sec) else (_value(revolver) if _exact(revolver) else None)
    market_availability_adjustment = _market_availability_adjustment(
        market_availability_overrides,
        company_id=company_id,
        as_of_time=as_of_time,
    )
    not_freely_transferable_cash_value = (
        market_availability_adjustment["value"] if market_availability_adjustment is not None else None
    )

    grouped_cash_exact = _exact(cash_grouped) and grouped_cash_value is not None
    grouped_cash_promoted_exact = _grouped_cash_proxy_can_promote_to_exact_cash_baseline(
        cash_grouped=cash_grouped,
        cash_exact=cash_exact,
        marketable_sec=marketable_sec,
        marketable=marketable,
        marketable_inferred_zero=marketable_inferred_zero,
    )
    grouped_cash_completed_with_marketable = _grouped_cash_proxy_can_complete_with_exact_marketable_securities(
        cash_grouped=cash_grouped,
        marketable_value=marketable_value,
    )
    grouped_cash_effectively_exact = grouped_cash_exact or (
        grouped_cash_promoted_exact and grouped_cash_value is not None
    )
    sec_cash_plus_marketable_override_eligible = (
        effective_cash_exact.get("support_override") == "companyfacts_cash_exact_fallback"
        and marketable_value is not None
        and not marketable_inferred_zero
    )
    sec_cash_plus_marketable_value = (
        cash_exact_value + marketable_value
        if effective_cash_exact["exact"] and cash_exact_value is not None and marketable_value is not None
        else None
    )
    grouped_cash_inconsistent_with_sec_cash_stack = (
        grouped_cash_value is not None
        and sec_cash_plus_marketable_value is not None
        and not _approximately_equal(grouped_cash_value, sec_cash_plus_marketable_value)
    )
    liquidity_base_excludes_restricted_cash = False
    cash_basis_source_metric_used = None
    cash_basis_support_override = None

    if sec_cash_plus_marketable_override_eligible and sec_cash_plus_marketable_value is not None and (
        not grouped_cash_effectively_exact or grouped_cash_inconsistent_with_sec_cash_stack
    ):
        liquidity_base = sec_cash_plus_marketable_value
        liquidity_formula = (
            effective_cash_exact["formula"]
            + (
                " + marketable_securities_sec_exact"
                if not marketable_inferred_zero
                else " + 0_inferred_short_term_investments"
            )
        )
        liquidity_base_excludes_restricted_cash = True
        cash_basis_source_metric_used = effective_cash_exact["source_metric"]
        cash_basis_support_override = "sec_cash_and_marketable_override_provider_grouped_cash"
    elif grouped_cash_effectively_exact:
        liquidity_base = grouped_cash_value
        liquidity_formula = "cash_and_short_term_investments_provider_direct"
        cash_basis_source_metric_used = "liquidity.cash_and_short_term_investments_provider_direct"
        if grouped_cash_promoted_exact:
            cash_basis_support_override = "partial_cash_stack_without_short_term_investments"
    elif grouped_cash_completed_with_marketable and grouped_cash_value is not None:
        liquidity_base = grouped_cash_value + marketable_value
        liquidity_formula = (
            "cash_and_short_term_investments_provider_direct_cash_component"
            + (
                " + marketable_securities_sec_exact"
                if not marketable_inferred_zero
                else " + 0_inferred_short_term_investments"
            )
        )
        liquidity_base_excludes_restricted_cash = True
        cash_basis_source_metric_used = "liquidity.cash_and_short_term_investments_provider_direct_cash_component"
        cash_basis_support_override = "partial_cash_stack_completed_by_marketable_securities"
    elif cash_exact_value is not None and marketable_value is not None:
        liquidity_base = cash_exact_value + marketable_value
        liquidity_formula = (
            effective_cash_exact["formula"] + " + marketable_securities_sec_exact"
            if not marketable_inferred_zero
            else effective_cash_exact["formula"] + " + 0_inferred_short_term_investments"
        )
        liquidity_base_excludes_restricted_cash = True
        cash_basis_source_metric_used = effective_cash_exact["source_metric"]
        cash_basis_support_override = effective_cash_exact.get("support_override")
    elif cash_exact_value is not None and grouped_cash_value is not None and abs(grouped_cash_value - cash_exact_value) <= 1.0:
        liquidity_base = cash_exact_value
        liquidity_formula = effective_cash_exact["formula"]
        liquidity_base_excludes_restricted_cash = True
        cash_basis_source_metric_used = effective_cash_exact["source_metric"]
        cash_basis_support_override = effective_cash_exact.get("support_override")
    elif grouped_cash_value is not None:
        liquidity_base = grouped_cash_value
        liquidity_formula = "cash_and_short_term_investments_provider_direct"
        cash_basis_source_metric_used = "liquidity.cash_and_short_term_investments_provider_direct"
    elif cash_exact_value is not None:
        liquidity_base = cash_exact_value
        liquidity_formula = effective_cash_exact["formula"]
        liquidity_base_excludes_restricted_cash = True
        cash_basis_source_metric_used = effective_cash_exact["source_metric"]
        cash_basis_support_override = effective_cash_exact.get("support_override")
    else:
        liquidity_base = None
        liquidity_formula = "unavailable"

    available_liquidity_raw = None
    if liquidity_base is not None:
        available_liquidity_raw = liquidity_base
        if restricted_cash_value is not None and not liquidity_base_excludes_restricted_cash:
            available_liquidity_raw -= restricted_cash_value
        if not_freely_transferable_cash_value is not None:
            available_liquidity_raw -= not_freely_transferable_cash_value
        if revolver_value is not None:
            available_liquidity_raw += revolver_value

    available_liquidity = available_liquidity_raw
    negative_floor_applied = available_liquidity_raw is not None and available_liquidity_raw < 0
    if negative_floor_applied:
        available_liquidity = 0.0

    liquidity_exact_ready = (
        restricted_cash_value is not None
        and (
            grouped_cash_effectively_exact
            or grouped_cash_completed_with_marketable
            or (
                cash_exact_value is not None
                and (
                    marketable_value is not None
                    or (
                        grouped_cash_value is not None
                        and abs(grouped_cash_value - cash_exact_value) <= 1.0
                    )
                )
            )
        )
    )
    if negative_floor_applied:
        liquidity_exact_ready = False
    formula_text = (
        liquidity_formula
        + (
            " - restricted_cash_sec_exact"
            if restricted_cash_value is not None and not restricted_cash_inferred_zero and not liquidity_base_excludes_restricted_cash
            else (
                " - 0_market_default_restricted_cash"
                if restricted_cash_market_default_zero and not liquidity_base_excludes_restricted_cash
                else (
                    " - 0_inferred_restricted_cash"
                    if restricted_cash_inferred_zero and not liquidity_base_excludes_restricted_cash
                    else ""
                )
            )
        )
        + (
            " - not_freely_transferable_cash_disclosed"
            if not_freely_transferable_cash_value is not None
            else ""
        )
        + (" + revolver_undrawn_exact" if revolver_value is not None else "")
    )
    if negative_floor_applied:
        formula_text = f"max(0, {formula_text})"
    liquidity_components = {
        "grouped_cash_provider_direct": grouped_cash_value,
        "cash_and_equivalents_statement_direct": cash_exact_value,
        "cash_basis_source_metric": cash_basis_source_metric_used,
        "restricted_cash_sec_exact": restricted_cash_value,
        "marketable_securities_sec_exact": marketable_value,
        "restricted_cash_inferred_zero": restricted_cash_inferred_zero,
        "marketable_securities_inferred_zero": marketable_inferred_zero,
        "restricted_cash_zero_reconciled": restricted_cash_zero_reconciled,
        "marketable_securities_zero_reconciled": marketable_zero_reconciled,
        "restricted_cash_market_default_zero": restricted_cash_market_default_zero,
        "restricted_cash_already_excluded_from_cash_basis": liquidity_base_excludes_restricted_cash,
        "revolver_undrawn_sec_exact": _value(revolver_sec) if _exact(revolver_sec) else None,
        "revolver_undrawn_exact": revolver_value,
        "not_freely_transferable_cash_disclosed": not_freely_transferable_cash_value,
        "raw_value_before_floor": available_liquidity_raw,
        "formula": formula_text,
    }
    if cash_basis_support_override is not None:
        liquidity_components["cash_basis_support_override"] = cash_basis_support_override
    if (
        effective_cash_exact.get("component_meta") is not None
        and cash_basis_source_metric_used == effective_cash_exact["source_metric"]
    ):
        liquidity_components["cash_basis_component_meta"] = effective_cash_exact["component_meta"]
    if restricted_cash_market_default_zero:
        liquidity_components["restricted_cash_support_override"] = (
            "grouped_cash_market_baseline_without_restricted_cash_disclosure"
        )
    if market_availability_adjustment is not None:
        liquidity_components["market_availability_adjustment"] = market_availability_adjustment
    if negative_floor_applied:
        liquidity_components["exact_guard_reason"] = "negative_available_liquidity"
    features["liquidity.available_liquidity_normalized"] = _smart_value_node(
        metric_name="liquidity.available_liquidity_normalized",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=liquidity_registry["status"],
        promotion_rule=liquidity_registry["promotion_rule"],
        value=available_liquidity,
        unit="usd",
        component_breakdown=liquidity_components,
        provenance_sources=provenance_sources,
        exact_ready=liquidity_exact_ready,
        missing_reason="component_unavailable" if available_liquidity is None else None,
    )

    effective_operating_earnings = _effective_operating_earnings_baseline(
        ebitda=ebitda,
        net_income=net_income,
        interest_expense=interest_expense,
        companyfacts=_get_companyfacts() if (_value(ebitda) is None and _exact(net_income)) else None,
        as_of_time=as_of_time,
    )
    earnings_value = effective_operating_earnings["value"]
    earnings_exact_ready = bool(effective_operating_earnings["exact"])
    features["operating.operating_earnings_normalized"] = _smart_value_node(
        metric_name="operating.operating_earnings_normalized",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=earnings_registry["status"],
        promotion_rule=earnings_registry["promotion_rule"],
        value=earnings_value,
        unit="usd",
        component_breakdown=effective_operating_earnings["component_breakdown"],
        provenance_sources=provenance_sources,
        exact_ready=earnings_exact_ready,
        missing_reason=effective_operating_earnings["missing_reason"],
    )
    if effective_operating_earnings.get("quality_flags") is not None:
        features["operating.operating_earnings_normalized"]["quality_flags"] = effective_operating_earnings["quality_flags"]

    net_debt_value = None if debt_value is None or available_liquidity is None else debt_value - available_liquidity
    net_debt_exact_ready = debt_exact_ready and liquidity_exact_ready
    features["capital_structure.net_debt_normalized"] = _smart_value_node(
        metric_name="capital_structure.net_debt_normalized",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=net_debt_registry["status"],
        promotion_rule=net_debt_registry["promotion_rule"],
        value=net_debt_value,
        unit="usd",
        component_breakdown={
            "debt_like_obligations_normalized": debt_value,
            "available_liquidity_normalized": available_liquidity,
            "formula": "debt_like_obligations_normalized - available_liquidity_normalized",
        },
        provenance_sources=provenance_sources,
        exact_ready=net_debt_exact_ready,
        missing_reason="component_unavailable" if net_debt_value is None else None,
    )

    net_debt_including_pension_value = (
        None
        if debt_including_pension_value is None or available_liquidity is None
        else debt_including_pension_value - available_liquidity
    )
    net_debt_including_pension_exact_ready = debt_including_pension_exact_ready and liquidity_exact_ready
    net_debt_including_pension_components = {
        "debt_like_obligations_including_pension": debt_including_pension_value,
        "available_liquidity_normalized": available_liquidity,
        "formula": "debt_like_obligations_including_pension - available_liquidity_normalized",
    }
    if pension_value is None and debt_including_pension_value is not None:
        net_debt_including_pension_components["pension_missing_assumed_zero"] = True
    features["capital_structure.net_debt_including_pension"] = _smart_value_node(
        metric_name="capital_structure.net_debt_including_pension",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=net_debt_including_pension_registry["status"],
        promotion_rule=net_debt_including_pension_registry["promotion_rule"],
        value=net_debt_including_pension_value,
        unit="usd",
        component_breakdown=net_debt_including_pension_components,
        provenance_sources=provenance_sources,
        exact_ready=net_debt_including_pension_exact_ready,
        missing_reason="component_unavailable" if net_debt_including_pension_value is None else None,
    )

    net_debt_including_retirement_value = (
        None
        if debt_including_retirement_value is None or available_liquidity is None
        else debt_including_retirement_value - available_liquidity
    )
    net_debt_including_retirement_exact_ready = (
        debt_including_retirement_exact_ready and liquidity_exact_ready
    )
    net_debt_including_retirement_components = {
        "debt_like_obligations_including_retirement": debt_including_retirement_value,
        "available_liquidity_normalized": available_liquidity,
        "formula": "debt_like_obligations_including_retirement - available_liquidity_normalized",
    }
    if combined_retirement_value is None and net_debt_including_retirement_value is not None:
        net_debt_including_retirement_components["combined_retirement_missing_assumed_zero"] = True
    features["capital_structure.net_debt_including_retirement"] = _smart_value_node(
        metric_name="capital_structure.net_debt_including_retirement",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=net_debt_including_retirement_registry["status"],
        promotion_rule=net_debt_including_retirement_registry["promotion_rule"],
        value=net_debt_including_retirement_value,
        unit="usd",
        component_breakdown=net_debt_including_retirement_components,
        provenance_sources=provenance_sources,
        exact_ready=net_debt_including_retirement_exact_ready,
        missing_reason="component_unavailable" if net_debt_including_retirement_value is None else None,
    )

    if debt_value is None or earnings_value is None:
        gross_lev_value = None
        gross_missing = "component_unavailable"
    elif earnings_value <= 0:
        gross_lev_value = None
        gross_missing = "non_positive_denominator"
    else:
        gross_lev_value = debt_value / earnings_value
        gross_missing = None
    gross_lev_exact_ready = debt_exact_ready and earnings_exact_ready and gross_lev_value is not None
    features["capital_structure.gross_leverage_normalized"] = _smart_value_node(
        metric_name="capital_structure.gross_leverage_normalized",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=gross_leverage_registry["status"],
        promotion_rule=gross_leverage_registry["promotion_rule"],
        value=gross_lev_value,
        unit="x",
        component_breakdown={
            "debt_like_obligations_normalized": debt_value,
            "operating_earnings_normalized": earnings_value,
            "formula": "debt_like_obligations_normalized / operating_earnings_normalized",
        },
        provenance_sources=provenance_sources,
        exact_ready=gross_lev_exact_ready,
        missing_reason=gross_missing,
    )

    if debt_including_pension_value is None or earnings_value is None:
        gross_lev_including_pension_value = None
        gross_including_pension_missing = "component_unavailable"
    elif earnings_value <= 0:
        gross_lev_including_pension_value = None
        gross_including_pension_missing = "non_positive_denominator"
    else:
        gross_lev_including_pension_value = debt_including_pension_value / earnings_value
        gross_including_pension_missing = None
    gross_lev_including_pension_exact_ready = (
        debt_including_pension_exact_ready and earnings_exact_ready and gross_lev_including_pension_value is not None
    )
    gross_lev_including_pension_components = {
        "debt_like_obligations_including_pension": debt_including_pension_value,
        "operating_earnings_normalized": earnings_value,
        "formula": "debt_like_obligations_including_pension / operating_earnings_normalized",
    }
    if pension_value is None and debt_including_pension_value is not None:
        gross_lev_including_pension_components["pension_missing_assumed_zero"] = True
    features["capital_structure.gross_leverage_including_pension"] = _smart_value_node(
        metric_name="capital_structure.gross_leverage_including_pension",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=gross_leverage_including_pension_registry["status"],
        promotion_rule=gross_leverage_including_pension_registry["promotion_rule"],
        value=gross_lev_including_pension_value,
        unit="x",
        component_breakdown=gross_lev_including_pension_components,
        provenance_sources=provenance_sources,
        exact_ready=gross_lev_including_pension_exact_ready,
        missing_reason=gross_including_pension_missing,
    )

    if debt_including_retirement_value is None or earnings_value is None:
        gross_lev_including_retirement_value = None
        gross_including_retirement_missing = "component_unavailable"
    elif earnings_value <= 0:
        gross_lev_including_retirement_value = None
        gross_including_retirement_missing = "non_positive_denominator"
    else:
        gross_lev_including_retirement_value = debt_including_retirement_value / earnings_value
        gross_including_retirement_missing = None
    gross_lev_including_retirement_exact_ready = (
        debt_including_retirement_exact_ready and earnings_exact_ready and gross_lev_including_retirement_value is not None
    )
    gross_lev_including_retirement_components = {
        "debt_like_obligations_including_retirement": debt_including_retirement_value,
        "operating_earnings_normalized": earnings_value,
        "formula": "debt_like_obligations_including_retirement / operating_earnings_normalized",
    }
    if combined_retirement_value is None and gross_lev_including_retirement_value is not None:
        gross_lev_including_retirement_components["combined_retirement_missing_assumed_zero"] = True
    features["capital_structure.gross_leverage_including_retirement"] = _smart_value_node(
        metric_name="capital_structure.gross_leverage_including_retirement",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=gross_leverage_including_retirement_registry["status"],
        promotion_rule=gross_leverage_including_retirement_registry["promotion_rule"],
        value=gross_lev_including_retirement_value,
        unit="x",
        component_breakdown=gross_lev_including_retirement_components,
        provenance_sources=provenance_sources,
        exact_ready=gross_lev_including_retirement_exact_ready,
        missing_reason=gross_including_retirement_missing,
    )

    if net_debt_value is None or earnings_value is None:
        net_lev_value = None
        net_missing = "component_unavailable"
    elif earnings_value <= 0:
        net_lev_value = None
        net_missing = "non_positive_denominator"
    else:
        net_lev_value = net_debt_value / earnings_value
        net_missing = None
    net_lev_exact_ready = net_debt_exact_ready and earnings_exact_ready and net_lev_value is not None
    features["capital_structure.net_leverage_normalized"] = _smart_value_node(
        metric_name="capital_structure.net_leverage_normalized",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=net_leverage_registry["status"],
        promotion_rule=net_leverage_registry["promotion_rule"],
        value=net_lev_value,
        unit="x",
        component_breakdown={
            "net_debt_normalized": net_debt_value,
            "operating_earnings_normalized": earnings_value,
            "formula": "net_debt_normalized / operating_earnings_normalized",
        },
        provenance_sources=provenance_sources,
        exact_ready=net_lev_exact_ready,
        missing_reason=net_missing,
    )

    if net_debt_including_pension_value is None or earnings_value is None:
        net_lev_including_pension_value = None
        net_including_pension_missing = "component_unavailable"
    elif earnings_value <= 0:
        net_lev_including_pension_value = None
        net_including_pension_missing = "non_positive_denominator"
    else:
        net_lev_including_pension_value = net_debt_including_pension_value / earnings_value
        net_including_pension_missing = None
    net_lev_including_pension_exact_ready = (
        net_debt_including_pension_exact_ready and earnings_exact_ready and net_lev_including_pension_value is not None
    )
    net_lev_including_pension_components = {
        "net_debt_including_pension": net_debt_including_pension_value,
        "operating_earnings_normalized": earnings_value,
        "formula": "net_debt_including_pension / operating_earnings_normalized",
    }
    if pension_value is None and net_debt_including_pension_value is not None:
        net_lev_including_pension_components["pension_missing_assumed_zero"] = True
    features["capital_structure.net_leverage_including_pension"] = _smart_value_node(
        metric_name="capital_structure.net_leverage_including_pension",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=net_leverage_including_pension_registry["status"],
        promotion_rule=net_leverage_including_pension_registry["promotion_rule"],
        value=net_lev_including_pension_value,
        unit="x",
        component_breakdown=net_lev_including_pension_components,
        provenance_sources=provenance_sources,
        exact_ready=net_lev_including_pension_exact_ready,
        missing_reason=net_including_pension_missing,
    )

    if net_debt_including_retirement_value is None or earnings_value is None:
        net_lev_including_retirement_value = None
        net_including_retirement_missing = "component_unavailable"
    elif earnings_value <= 0:
        net_lev_including_retirement_value = None
        net_including_retirement_missing = "non_positive_denominator"
    else:
        net_lev_including_retirement_value = net_debt_including_retirement_value / earnings_value
        net_including_retirement_missing = None
    net_lev_including_retirement_exact_ready = (
        net_debt_including_retirement_exact_ready and earnings_exact_ready and net_lev_including_retirement_value is not None
    )
    net_lev_including_retirement_components = {
        "net_debt_including_retirement": net_debt_including_retirement_value,
        "operating_earnings_normalized": earnings_value,
        "formula": "net_debt_including_retirement / operating_earnings_normalized",
    }
    if combined_retirement_value is None and net_lev_including_retirement_value is not None:
        net_lev_including_retirement_components["combined_retirement_missing_assumed_zero"] = True
    features["capital_structure.net_leverage_including_retirement"] = _smart_value_node(
        metric_name="capital_structure.net_leverage_including_retirement",
        as_of_time=as_of_time,
        computed_at=computed_at,
        registry_status=net_leverage_including_retirement_registry["status"],
        promotion_rule=net_leverage_including_retirement_registry["promotion_rule"],
        value=net_lev_including_retirement_value,
        unit="x",
        component_breakdown=net_lev_including_retirement_components,
        provenance_sources=provenance_sources,
        exact_ready=net_lev_including_retirement_exact_ready,
        missing_reason=net_including_retirement_missing,
    )

    return row


def main() -> None:
    args = parse_args()
    snapshot_path = Path(args.snapshot_path)
    registry_path = Path(args.metric_registry_path)
    component_policy_path = Path(args.component_policy_path)
    source_precedence_path = Path(args.source_precedence_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    registry = json.loads(registry_path.read_text())
    json.loads(component_policy_path.read_text())
    json.loads(source_precedence_path.read_text())
    computed_at = _now_iso()
    provenance_sources = [str(registry_path), str(component_policy_path), str(source_precedence_path)]
    companyfacts_root = Path(args.companyfacts_root) if args.companyfacts_root else (
        DEFAULT_LOCAL_COMPANYFACTS_ROOT if DEFAULT_LOCAL_COMPANYFACTS_ROOT.exists() else None
    )
    companyfacts_cache: dict[str, Dict[str, Any] | None] = {}
    retirement_note_cache_root = (
        Path(args.sec_filing_cache_root)
        if args.sec_filing_cache_root
        else DEFAULT_SEC_RETIREMENT_CACHE_ROOT
    )
    if retirement_note_cache_root is not None:
        retirement_note_cache_root.mkdir(parents=True, exist_ok=True)
    retirement_note_cache: dict[tuple[str, str], Dict[str, Any] | None] = {}
    sec_session = None
    if _sec_helpers_available():
        sec_session = requests.Session()
        sec_session.headers.update({"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    market_availability_overrides_path = (
        Path(args.market_availability_overrides_path)
        if args.market_availability_overrides_path
        else (DEFAULT_MARKET_AVAILABILITY_OVERRIDES_PATH if DEFAULT_MARKET_AVAILABILITY_OVERRIDES_PATH.exists() else None)
    )
    market_availability_overrides = _load_market_availability_overrides(market_availability_overrides_path)
    completed_company_ids = _load_completed_company_ids(out_path) if args.resume_if_exists else set()
    open_mode = "a" if args.resume_if_exists and out_path.exists() else "w"
    if open_mode == "w" and out_path.exists() and not args.resume_if_exists:
        out_path.unlink()

    processed_rows = 0
    with out_path.open(open_mode, buffering=1) as out_handle:
        for row in iter_snapshot_rows(snapshot_path):
            entity_id = str(row.get("company_id")) if row.get("company_id") is not None else None
            if entity_id is not None and entity_id in completed_company_ids:
                continue

            try:
                with _company_processing_guard(args.company_processing_timeout_seconds):
                    companyfacts_loader = None
                    retirement_note_loader = None
                    cached_companyfacts: Dict[str, Any] | None = None
                    if companyfacts_root is not None and entity_id is not None and _row_may_need_companyfacts(row.setdefault("features", {})):
                        def _loader(entity_id: str = entity_id) -> Dict[str, Any] | None:
                            if entity_id not in companyfacts_cache:
                                companyfacts_cache[entity_id] = _load_companyfacts(companyfacts_root / f"CIK{entity_id}.json")
                            return companyfacts_cache[entity_id]
                        companyfacts_loader = _loader
                        if sec_session is not None:
                            cached_companyfacts = _loader()
                    if entity_id is not None and sec_session is not None and _companyfacts_may_need_retirement_note_split(cached_companyfacts):
                        as_of_time = row["as_of_time"]

                        def _retirement_loader(
                            entity_id: str = entity_id,
                            as_of_time: str = as_of_time,
                        ) -> Dict[str, Any] | None:
                            cache_key = (entity_id, as_of_time[:10])
                            if cache_key not in retirement_note_cache:
                                retirement_note_cache[cache_key] = _load_retirement_note_components(
                                    cik=entity_id,
                                    as_of_time=as_of_time,
                                    session=sec_session,
                                    cache_dir=retirement_note_cache_root,
                                )
                            return retirement_note_cache[cache_key]

                        retirement_note_loader = _retirement_loader

                    row = materialize_smart_metrics_for_row(
                        row=row,
                        registry=registry,
                        computed_at=computed_at,
                        provenance_sources=provenance_sources,
                        companyfacts_loader=companyfacts_loader,
                        retirement_note_loader=retirement_note_loader,
                        market_availability_overrides=market_availability_overrides,
                    )
            except _CompanyProcessingTimeout as exc:
                row.setdefault("features", {}).update(
                    _build_fail_open_smart_metrics(
                        as_of_time=row["as_of_time"],
                        computed_at=computed_at,
                        provenance_sources=provenance_sources,
                        error_type="company_processing_timeout",
                        error_message=str(exc),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                row.setdefault("features", {}).update(
                    _build_fail_open_smart_metrics(
                        as_of_time=row["as_of_time"],
                        computed_at=computed_at,
                        provenance_sources=provenance_sources,
                        error_type="company_processing_failed",
                        error_message=str(exc),
                    )
                )

            out_handle.write(json.dumps(row) + "\n")
            processed_rows += 1

    summary = _summarize_output_rows(out_path)
    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(summary, indent=2))

    print(f"Wrote smart-normalized layer -> {out_path}")
    if args.resume_if_exists and completed_company_ids:
        print(f"resumed_from_existing_rows={len(completed_company_ids)}")
    print(f"new_rows_processed={processed_rows}")
    row_fail_open = summary.get("row_fail_open") or {}
    if row_fail_open.get("company_processing_timeout") or row_fail_open.get("company_processing_failed"):
        print(
            "row_fail_open:"
            f" company_processing_timeout={row_fail_open.get('company_processing_timeout', 0)}"
            f" company_processing_failed={row_fail_open.get('company_processing_failed', 0)}"
        )


if __name__ == "__main__":
    main()
