#!/usr/bin/env python3
"""Repair statement-direct debt overrides in an already-materialized artifact."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import signal
from collections import defaultdict
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlparse

import duckdb
import requests
from bs4 import BeautifulSoup

try:
    from backfill_sec_companyfacts_components import _extract_lease_liabilities
except Exception:  # noqa: BLE001
    try:
        from scripts.backfill_sec_companyfacts_components import _extract_lease_liabilities
    except Exception:  # noqa: BLE001
        _extract_lease_liabilities = None


TARGET_MODE = "statement_direct_current_plus_noncurrent_debt"
CURRENT_DEBT_FACT = "financial.debt_current"
LONG_TERM_DEBT_FACT = "financial.debt_long_term"
TOTAL_DEBT_FACT = "financial.total_debt"
MAX_ALIGNMENT_GAP_DAYS = 45
MAX_EXACT_AGE_DAYS = 130
PARTIAL_TOTAL_DEBT_MIN_LIFT = 1.10
PARTIAL_TOTAL_DEBT_MAX_DOWNWARD_REPLACEMENT = 0.75
TOTAL_DEBT_REGRESSION_THRESHOLD = 1.50
SEC_USER_AGENT = "Codex/axiom_v1 research support"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
SEC_EXACT_MAX_AGE_DAYS = 130

CURRENT_TOTAL_LABEL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^short[\s\-]term borrowings$",
        r"^(?:.+\s+)?short[\s\-]term borrowings$",
        r"^short[\s\-]term debt$",
        r"^(?:.+\s+)?short[\s\-]term debt$",
        r"^current portion of long[\s\-]term debt$",
        r"^(?:.+\s+)?current portion of long[\s\-]term debt$",
        r"^current maturities of long[\s\-]term debt$",
        r"^(?:.+\s+)?current maturities of long[\s\-]term debt$",
        r"^long[\s\-]term borrowings due within one year$",
        r"^(?:.+\s+)?long[\s\-]term borrowings due within one year$",
        r"^debt due within one year$",
        r"^(?:.+\s+)?debt due within one year$",
        r"^debt payable within one year$",
        r"^(?:.+\s+)?debt payable within one year$",
        r"^current debt$",
        r"^(?:.+\s+)?current debt$",
    )
]
CURRENT_EXTRA_LABEL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^short[\s\-]term securitization borrowings$",
        r"^(?:.+\s+)?short[\s\-]term securitization borrowings$",
        r"^current securitization borrowings$",
        r"^(?:.+\s+)?current securitization borrowings$",
        r"^securitization borrowings due within one year$",
        r"^(?:.+\s+)?securitization borrowings due within one year$",
    )
]
CURRENT_COMPONENT_LABEL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^commercial paper$",
        r"^notes payable(?: to banks)?$",
        r"^current portion of long[\s\-]term debt$",
        r"^current maturities of long[\s\-]term debt$",
        r"^long[\s\-]term borrowings due within one year$",
        r"^short[\s\-]term securitization borrowings$",
    )
]
LONG_TOTAL_LABEL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^long[\s\-]term borrowings$",
        r"^(?:.+\s+)?long[\s\-]term borrowings$",
        r"^long[\s\-]term debt$",
        r"^(?:.+\s+)?long[\s\-]term debt$",
        r"^long[\s\-]term debt payable after one year$",
        r"^(?:.+\s+)?long[\s\-]term debt payable after one year$",
        r"^long term debt$",
        r"^notes payable and long[\s\-]term debt$",
    )
]
TOTAL_LABEL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^long[\s\-]term debt, including current maturities$",
        r"^debt, including current maturities$",
        r"^total debt$",
        r"^total borrowings$",
    )
]
VEHICLE_PROGRAM_DEBT_LABEL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^debt$",
        r"^debt due to .+$",
        r"^vehicle[\s\-]backed debt$",
        r"^vehicle[\s\-]backed debt due to .+$",
    )
]
BALANCE_SHEET_CUES = (
    "balance sheet",
    "balance sheets",
    "liabilities and stockholders",
    "liabilities and shareholders",
    "liabilities and equity",
    "current liabilities",
)
FAIR_VALUE_CUES = ("fair value", "carrying amount")
VEHICLE_PROGRAM_TABLE_CUES = (
    "liabilities under vehicle programs",
    "debt under vehicle programs",
)


class _CompanyProcessingTimeoutError(RuntimeError):
    """Raised when a single company repair attempt exceeds the allowed time."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--facts-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--enable-sec-filing-fallback", action="store_true")
    parser.add_argument("--sec-cache-dir")
    parser.add_argument("--company-ids", nargs="*")
    parser.add_argument("--company-ids-file")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--company-processing-timeout-seconds", type=int, default=30)
    parser.add_argument("--skip-fact-registry-repair", action="store_true")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _timeout_handler(signum, frame):  # noqa: ANN001, ARG001
    raise _CompanyProcessingTimeoutError("company processing timed out")


@contextmanager
def _company_processing_timeout(seconds: int | None):
    if not seconds or seconds <= 0 or os.name == "nt":
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, float(seconds))
    signal.signal(signal.SIGALRM, _timeout_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _parse_date(value: Any) -> date | None:
    if value in (None, "", "None"):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _node_value(node: dict[str, Any] | None) -> float | None:
    if not node:
        return None
    value = node.get("value")
    return None if value is None else float(value)


def _is_exact(node: dict[str, Any] | None) -> bool:
    return node is not None and node.get("support_mode") == "exact" and _node_value(node) is not None


def _is_supported(node: dict[str, Any] | None) -> bool:
    return node is not None and node.get("support_mode") != "unsupported" and _node_value(node) is not None


def _set_metric(
    template: dict[str, Any] | None,
    *,
    value: float | None,
    support_mode: str,
    missing_reason: str | None,
    component_breakdown: dict[str, Any],
    computed_at: str,
    provenance_source: str,
    unit: str | None = None,
    quality_flags: list[str] | None = None,
) -> dict[str, Any]:
    node = dict(template or {})
    node["value"] = value
    node["support_mode"] = support_mode
    node["missing_reason"] = missing_reason
    node["component_breakdown"] = component_breakdown
    node["computed_at"] = computed_at
    node["provenance_source"] = provenance_source
    if unit is not None:
        node["unit"] = unit
    if quality_flags:
        node["quality_flags"] = quality_flags
    else:
        node["quality_flags"] = []
    return node


def _metric_support_from_components(*nodes: dict[str, Any] | None) -> str:
    if any(not _is_supported(node) for node in nodes):
        return "unsupported"
    if all(_is_exact(node) for node in nodes):
        return "exact"
    return "proxy_missing_component"


def _collect_target_ids(artifact_path: Path) -> tuple[list[str], str | None]:
    entity_ids: list[str] = []
    as_of_time: str | None = None
    with artifact_path.open() as src:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if as_of_time is None:
                as_of_time = row.get("as_of_time")
            total_debt = (row.get("features") or {}).get("capital_structure.total_debt_provider_direct") or {}
            mode = (total_debt.get("component_breakdown") or {}).get("mode")
            if mode == TARGET_MODE or (
                mode == "partial_debt_stack"
                and (total_debt.get("component_breakdown") or {}).get("current")
                and not (total_debt.get("component_breakdown") or {}).get("noncurrent")
            ) or total_debt.get("missing_reason") in {
                "debt_component_missing",
                "long_term_debt_components_missing",
                "debt_component_period_mismatch",
                "statement_debt_pair_stale",
                "statement_debt_pair_unresolved",
            }:
                entity_ids.append(str(row["company_id"]))
    return sorted(set(entity_ids)), as_of_time


def _load_company_ids_file(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    company_ids: list[str] = []
    with path.open() as handle:
        for line in handle:
            text = line.strip()
            if text:
                company_ids.append(text.zfill(10))
    return company_ids


def _load_completed_company_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            company_id = str(row.get("company_id", "")).zfill(10)
            if company_id:
                completed.add(company_id)
    return completed


def _load_candidates(
    *,
    facts_path: Path,
    entity_ids: list[str],
    as_of_time: str,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    if not entity_ids:
        return {}
    entity_sql = ",".join(f"'{entity_id}'" for entity_id in entity_ids)
    as_of_sql = as_of_time.replace("'", "''")
    query = f"""
        SELECT
            entity_id,
            fact_type,
            fact_id,
            source_id,
            source_type,
            raw_pointer,
            unit,
            fact_value,
            CAST(effective_at AS VARCHAR) AS effective_at,
            CAST(fact_time AS VARCHAR) AS fact_time
        FROM parquet_scan('{facts_path.as_posix()}')
        WHERE entity_id IN ({entity_sql})
          AND fact_type IN ('{CURRENT_DEBT_FACT}', '{LONG_TERM_DEBT_FACT}', '{TOTAL_DEBT_FACT}')
          AND fact_value IS NOT NULL
          AND (
                valid_from IS NULL
                OR COALESCE(
                    TRY_CAST(valid_from AS TIMESTAMP),
                    CAST(TRY_CAST(valid_from AS DATE) AS TIMESTAMP)
                ) <= TIMESTAMP '{as_of_sql.replace('Z', '')}'
          )
          AND (
                valid_to IS NULL
                OR COALESCE(
                    TRY_CAST(valid_to AS TIMESTAMP),
                    CAST(TRY_CAST(valid_to AS DATE) AS TIMESTAMP)
                ) > TIMESTAMP '{as_of_sql.replace('Z', '')}'
          )
    """
    rows = duckdb.connect().execute(query).fetchall()
    out: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        (
            entity_id,
            fact_type,
            fact_id,
            source_id,
            source_type,
            raw_pointer,
            unit,
            fact_value,
            effective_at,
            fact_time,
        ) = row
        out[str(entity_id)][str(fact_type)].append(
            {
                "value": float(fact_value),
                "end_dt": _parse_date(effective_at) or _parse_date(fact_time),
                "meta": {
                    "fact_type": fact_type,
                    "fact_id": fact_id,
                    "source_id": source_id,
                    "source_type": source_type,
                    "raw_pointer": raw_pointer,
                    "registry_unit": unit,
                    "effective_at": effective_at,
                    "end": effective_at,
                    "fact_time": fact_time,
                    "formula": "statement_direct_fact",
                },
            }
        )
    return out


def _best_pair(
    current_candidates: list[dict[str, Any]],
    long_term_candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int | None]:
    best: tuple[Any, ...] | None = None
    best_current = None
    best_long_term = None
    best_gap = None
    for current in current_candidates:
        current_end = current.get("end_dt")
        current_source = (current.get("meta") or {}).get("source_type")
        if current_end is None or not current_source:
            continue
        for long_term in long_term_candidates:
            long_term_end = long_term.get("end_dt")
            long_term_source = (long_term.get("meta") or {}).get("source_type")
            if long_term_end is None or current_source != long_term_source:
                continue
            gap_days = abs((current_end - long_term_end).days)
            if gap_days > MAX_ALIGNMENT_GAP_DAYS:
                continue
            latest_end = max(current_end, long_term_end)
            score = (latest_end, -gap_days, current_source == "sec_edgar_xbrl")
            if best is None or score > best:
                best = score
                best_current = current
                best_long_term = long_term
                best_gap = gap_days
    return best_current, best_long_term, best_gap


def _latest_fact(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.get("end_dt") or date.min, (item.get("value") or 0.0)))


def _sec_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": SEC_USER_AGENT})
    return session


def _ensure_cache_dir(cache_dir: Path | None) -> Path | None:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _load_sec_submissions(cik: str, *, session: requests.Session, cache_dir: Path | None) -> dict[str, Any] | None:
    cache_path = None if cache_dir is None else cache_dir / f"CIK{cik}.json"
    if cache_path is not None and cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass
    if os.environ.get("AXIOM_DISABLE_SEC_NETWORK_FALLBACK") == "1":
        return None
    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return None
    if cache_path is not None:
        cache_path.write_text(json.dumps(payload))
    return payload


def _latest_sec_filing(
    *,
    cik: str,
    as_of_date: date,
    session: requests.Session,
    cache_dir: Path | None,
) -> dict[str, Any] | None:
    submissions = _load_sec_submissions(cik, session=session, cache_dir=cache_dir)
    if not submissions:
        return None
    recent = (submissions.get("filings") or {}).get("recent") or {}
    best: tuple[date, int, dict[str, Any]] | None = None
    forms = {"10-Q": 2, "10-K": 1}
    for filing_date, form, accession, primary_document in zip(
        recent.get("filingDate", []),
        recent.get("form", []),
        recent.get("accessionNumber", []),
        recent.get("primaryDocument", []),
    ):
        if form not in forms:
            continue
        filed_dt = _parse_date(filing_date)
        if filed_dt is None or filed_dt > as_of_date:
            continue
        record = {
            "cik": cik,
            "filing_date": filing_date,
            "form": form,
            "accession_number": accession,
            "primary_document": primary_document,
        }
        score = (filed_dt, forms[form], record)
        if best is None or score > best:
            best = score
    return None if best is None else best[2]


def _fetch_sec_primary_document(
    filing: dict[str, Any],
    *,
    session: requests.Session,
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
    url = f"{SEC_ARCHIVES_BASE}/{cik_no_zeros}/{accession_nodash}/{primary_document}"
    try:
        response = session.get(url, timeout=60)
        response.raise_for_status()
        html = response.text
    except requests.RequestException:
        return None
    if cache_path is not None:
        cache_path.write_text(html)
    return html


def _table_multiplier(table_text: str) -> float:
    lower = table_text.lower()
    if "in billions" in lower or "($ in billions)" in lower or "(billions)" in lower:
        return 1_000_000_000.0
    if "in millions" in lower or "($ in millions)" in lower or "(millions)" in lower:
        return 1_000_000.0
    if "in thousands" in lower or "($ in thousands)" in lower or "(thousands)" in lower:
        return 1_000.0
    return 1.0


def _document_multiplier(document_text: str) -> float:
    lower = document_text.lower()
    if "all amounts are presented in billions" in lower or "all dollar amounts are in billions" in lower:
        return 1_000_000_000.0
    if "all amounts are presented in millions" in lower or "all dollar amounts are in millions" in lower:
        return 1_000_000.0
    if "all amounts are presented in thousands" in lower or "all dollar amounts are in thousands" in lower:
        return 1_000.0
    if "in billions" in lower or "(billions)" in lower:
        return 1_000_000_000.0
    if "in millions" in lower or "(millions)" in lower:
        return 1_000_000.0
    if "in thousands" in lower or "(thousands)" in lower:
        return 1_000.0
    return 1.0


def _normalize_label(text: str) -> str:
    label = " ".join(str(text or "").replace("\xa0", " ").replace("\u200b", " ").split())
    label = label.replace("’", "'").replace("–", "-").replace("—", "-")
    return label.strip(" :")


