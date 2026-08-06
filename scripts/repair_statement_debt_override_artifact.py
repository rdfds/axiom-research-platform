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


_NUMBER_RE = re.compile(r"^\(?\s*[\$]?\s*-?\d[\d,]*(?:\.\d+)?\s*\)?$")


def _parse_numeric_cell(text: str, *, multiplier: float) -> float | None:
    cleaned = _normalize_label(text)
    if not cleaned or cleaned in {"$", "—", "-", "nm", "n/m"}:
        return None
    if not _NUMBER_RE.match(cleaned):
        return None
    negative = "(" in cleaned and ")" in cleaned
    stripped = cleaned.replace("$", "").replace(",", "").replace("(", "").replace(")", "").strip()
    try:
        value = float(stripped)
    except ValueError:
        return None
    return (-value if negative else value) * multiplier


def _row_first_numeric(cells: list[str], *, multiplier: float) -> float | None:
    for cell in cells[1:]:
        value = _parse_numeric_cell(cell, multiplier=multiplier)
        if value is not None:
            return value
    return None


def _matches_any(label: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(label) for pattern in patterns)


def _extract_filing_table_debt_candidate(
    *,
    filing: dict[str, Any],
    html: str,
    as_of_date: date,
) -> dict[str, Any] | None:
    soup = BeautifulSoup(html, "html.parser")
    document_multiplier = _document_multiplier(" ".join(soup.get_text(" ", strip=True).split()[:20000]))
    best: tuple[tuple[int, float], dict[str, Any]] | None = None
    for table_index, table in enumerate(soup.find_all("table")):
        table_text = " ".join(table.get_text(" ", strip=True).split())
        lower_table_text = table_text.lower()
        table_multiplier = _table_multiplier(table_text)
        rows: list[tuple[str, float, list[str]]] = []
        for tr in table.find_all("tr"):
            cells = [_normalize_label(cell.get_text(" ", strip=True)) for cell in tr.find_all(["th", "td"])]
            cells = [cell for cell in cells if cell]
            if len(cells) < 2:
                continue
            label = cells[0]
            value = _row_first_numeric(cells, multiplier=1.0)
            if value is None:
                continue
            rows.append((label, value, cells))
        if not rows:
            continue

        sample_max_raw = max(abs(value) for _, value, _ in rows)
        multiplier = table_multiplier
        if multiplier == 1.0 and document_multiplier != 1.0:
            multiplier = 1.0 if sample_max_raw > 1_000_000.0 else document_multiplier

        balance_sheet_cue = any(cue in lower_table_text for cue in BALANCE_SHEET_CUES)
        fair_value_cue = any(cue in lower_table_text for cue in FAIR_VALUE_CUES)
        vehicle_program_table_cue = any(cue in lower_table_text for cue in VEHICLE_PROGRAM_TABLE_CUES)

        current_total = None
        current_extras = []
        current_components = []
        long_total = None
        total_row = None
        vehicle_program_components = []
        matched_rows: list[dict[str, Any]] = []

        for label, raw_value, cells in rows:
            value = raw_value * multiplier
            lower_label = label.lower()
            if _matches_any(label, CURRENT_TOTAL_LABEL_PATTERNS):
                current_total = value if current_total is None else current_total + value
                matched_rows.append({"label": label, "value": value, "cells": cells, "bucket": "current_total"})
            elif _matches_any(label, CURRENT_EXTRA_LABEL_PATTERNS):
                current_extras.append(value)
                matched_rows.append({"label": label, "value": value, "cells": cells, "bucket": "current_extra"})
            elif _matches_any(label, CURRENT_COMPONENT_LABEL_PATTERNS):
                current_components.append(value)
                matched_rows.append({"label": label, "value": value, "cells": cells, "bucket": "current_component"})
            elif _matches_any(label, LONG_TOTAL_LABEL_PATTERNS):
                long_total = value if long_total is None else long_total + value
                matched_rows.append({"label": label, "value": value, "cells": cells, "bucket": "long_total"})
            elif _matches_any(label, TOTAL_LABEL_PATTERNS):
                total_row = value
                matched_rows.append({"label": label, "value": value, "cells": cells, "bucket": "total_row"})
            elif vehicle_program_table_cue and _matches_any(label, VEHICLE_PROGRAM_DEBT_LABEL_PATTERNS):
                vehicle_program_components.append(value)
                matched_rows.append({"label": label, "value": value, "cells": cells, "bucket": "vehicle_program_debt"})

        current_value = None
        current_mode = None
        if current_total is not None:
            current_value = current_total + sum(current_extras)
            current_mode = "current_total_plus_extras" if current_extras else "current_total"
        elif current_components:
            current_value = sum(current_components)
            current_mode = "current_components"

        candidate_value = None
        candidate_mode = None
        if current_value is not None and long_total is not None and vehicle_program_components:
            candidate_value = current_value + long_total + sum(vehicle_program_components)
            candidate_mode = "balance_sheet_current_plus_long_plus_vehicle_program_debt"
        elif current_value is not None and long_total is not None:
            candidate_value = current_value + long_total
            candidate_mode = "balance_sheet_current_plus_long"
        elif total_row is not None and not fair_value_cue:
            candidate_value = total_row
            candidate_mode = "table_total_debt_row"
        elif long_total is not None and current_value is None and total_row is None and balance_sheet_cue:
            candidate_value = long_total
            candidate_mode = "long_term_only"

        if candidate_value is None or candidate_value <= 0:
            continue

        filing_dt = _parse_date(filing.get("filing_date"))
        age_days = None if filing_dt is None else (as_of_date - filing_dt).days
        score = 0
        if balance_sheet_cue:
            score += 10
        if candidate_mode == "balance_sheet_current_plus_long_plus_vehicle_program_debt":
            score += 12
        elif candidate_mode == "balance_sheet_current_plus_long":
            score += 8
        elif candidate_mode == "table_total_debt_row":
            score += 5
        elif candidate_mode == "long_term_only":
            score += 1
        if current_extras:
            score += 2
        if vehicle_program_components:
            score += 3
        if fair_value_cue:
            score -= 8
        if age_days is not None and age_days <= SEC_EXACT_MAX_AGE_DAYS:
            score += 2

        candidate = {
            "value": float(candidate_value),
            "mode": candidate_mode,
            "current_mode": current_mode,
            "balance_sheet_cue": balance_sheet_cue,
            "fair_value_cue": fair_value_cue,
            "table_index": table_index,
            "table_excerpt": table_text[:4000],
            "matched_rows": matched_rows,
            "filing": filing,
            "age_days": age_days,
        }
        rank = (score, candidate_value)
        if best is None or rank > best[0]:
            best = (rank, candidate)
    return None if best is None else best[1]


def _finance_lease_adjustment_from_lease_node(lease_node: dict[str, Any] | None) -> tuple[float, dict[str, Any] | None]:
    if not lease_node or _node_value(lease_node) is None:
        return 0.0, None
    breakdown = lease_node.get("component_breakdown") or {}
    finance_ref = breakdown.get("finance_reference") or {}
    direct = (finance_ref.get("direct_total_reference") or {}).get("value")
    if direct is not None:
        return float(direct), {"mode": "finance_lease_direct_total", "source": finance_ref.get("direct_total_reference")}
    partial = finance_ref.get("partial_component_reference") or {}
    current_value = partial.get("current_value")
    noncurrent_value = partial.get("noncurrent_value")
    if current_value is not None or noncurrent_value is not None:
        return float((current_value or 0.0) + (noncurrent_value or 0.0)), {
            "mode": "finance_lease_partial_components",
            "source": partial,
        }
    return 0.0, None


def _should_override_total_debt_with_filing_candidate(
    *,
    current_value: float | None,
    current_support: str | None,
    parsed_value: float,
) -> bool:
    if current_value is None:
        return True
    if current_support != "exact":
        if parsed_value >= float(current_value):
            return True
        return parsed_value >= float(current_value) * PARTIAL_TOTAL_DEBT_MAX_DOWNWARD_REPLACEMENT
    return parsed_value > float(current_value) * PARTIAL_TOTAL_DEBT_MIN_LIFT


def _repair_total_debt_from_sec_filing(
    *,
    row: dict[str, Any],
    computed_at: str,
    provenance_source: str,
    session: requests.Session,
    cache_dir: Path | None,
    companyfacts: dict[str, Any] | None = None,
) -> bool:
    features = row["features"]
    total_debt = features.get("capital_structure.total_debt_provider_direct")
    if not total_debt:
        return False
    current_value = _node_value(total_debt)
    current_support = total_debt.get("support_mode")
    current_missing = total_debt.get("missing_reason")
    breakdown = total_debt.get("component_breakdown") or {}

    as_of_date = _parse_date(row.get("as_of_time"))
    if as_of_date is None:
        return False
    cik = str(row["company_id"]).zfill(10)
    filing = _latest_sec_filing(cik=cik, as_of_date=as_of_date, session=session, cache_dir=cache_dir)
    if filing is None:
        return False
    html = _fetch_sec_primary_document(filing, session=session, cache_dir=cache_dir)
    if not html:
        return False
    candidate = _extract_filing_table_debt_candidate(filing=filing, html=html, as_of_date=as_of_date)
    if candidate is None:
        return False

    market_cap_node = features.get("market.market_cap_provider_direct")
    revenue_node = features.get("operating.revenue_ttm_provider_direct")
    market_cap_value = _node_value(market_cap_node)
    revenue_value = _node_value(revenue_node)

    def _plausible_debt_value(value: float) -> bool:
        if value <= 0 or value < 1_000_000.0 or value > 1_000_000_000_000.0:
            return False
        if market_cap_value is not None and revenue_value is not None:
            if value > market_cap_value * 10.0 and value > revenue_value * 10.0:
                return False
        elif market_cap_value is not None:
            if value > market_cap_value * 50.0:
                return False
        elif revenue_value is not None:
            if value > revenue_value * 20.0:
                return False
        elif current_value is not None and current_value > 0:
            if value > current_value * 100.0 and value > 10_000_000_000.0:
                return False
        return True

    parsed_value = float(candidate["value"])
    unit_rescale_divisor = 1.0
    if not _plausible_debt_value(parsed_value):
        for divisor in (1_000.0, 1_000_000.0):
            rescaled = parsed_value / divisor
            if _plausible_debt_value(rescaled):
                parsed_value = rescaled
                unit_rescale_divisor = divisor
                break
    if not _plausible_debt_value(parsed_value):
        return False
    should_override = _should_override_total_debt_with_filing_candidate(
        current_value=current_value,
        current_support=current_support,
        parsed_value=parsed_value,
    )
    if not should_override:
        return False

    lease = features.get("capital_structure.lease_liabilities_sec_exact")
    if lease is None and companyfacts is not None and _extract_lease_liabilities is not None:
        lease_value, lease_meta = _extract_lease_liabilities(companyfacts, row.get("as_of_time", "")[:10])
        if lease_value is not None and lease_meta is not None:
            lease = {
                "value": lease_value,
                "component_breakdown": lease_meta,
            }
    finance_lease_adjustment, finance_lease_meta = _finance_lease_adjustment_from_lease_node(lease)
    adjusted_value = parsed_value
    support_mode = "proxy_missing_component"
    missing_reason = "filing_table_total_debt_repair"
    quality_flags = ["filing_table_total_debt_repair"]
    if candidate["mode"] in {
        "balance_sheet_current_plus_long",
        "balance_sheet_current_plus_long_plus_vehicle_program_debt",
    } and not candidate["fair_value_cue"] and candidate["balance_sheet_cue"]:
        support_mode = "exact"
        missing_reason = None
    if candidate.get("age_days") is not None and candidate["age_days"] > SEC_EXACT_MAX_AGE_DAYS:
        support_mode = "proxy_missing_component"
        missing_reason = "filing_table_total_debt_stale"
        quality_flags.append("filing_table_total_debt_stale")
    if unit_rescale_divisor != 1.0:
        support_mode = "proxy_missing_component"
        missing_reason = "filing_table_total_debt_unit_rescaled"
        quality_flags.append(f"filing_table_total_debt_unit_rescaled_{int(unit_rescale_divisor)}")
    if finance_lease_adjustment:
        adjusted_value -= finance_lease_adjustment
        quality_flags.append("finance_lease_adjusted")
    if adjusted_value <= 0:
        return False

    repaired = _set_metric(
        total_debt,
        value=float(adjusted_value),
        support_mode=support_mode,
        missing_reason=missing_reason,
        component_breakdown={
            "mode": "sec_filing_table_total_debt",
            "selected_filing": filing,
            "selected_table_index": candidate["table_index"],
            "selected_table_mode": candidate["mode"],
            "matched_rows": candidate["matched_rows"],
            "unit_rescale_divisor": unit_rescale_divisor,
            "finance_lease_adjustment": finance_lease_meta,
            "finance_lease_adjustment_value": finance_lease_adjustment,
            "source_table_excerpt": candidate["table_excerpt"],
            "prior_total_debt_breakdown": breakdown,
            "formula": "filing_table_debt_components - exact_finance_lease_if_available",
        },
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit=total_debt.get("unit", "usd"),
        quality_flags=quality_flags,
    )
    repaired["input_source_classification"] = "sec_filing_table_repair"
    repaired["input_layer_bucket_reason"] = "sec_filing_table_total_debt_repair"
    repaired["primary_source_basis"] = "sec_filing_table"
    repaired["provenance_artifact_type"] = "SecFilingHtml"
    features["capital_structure.total_debt_provider_direct"] = repaired
    return True


def _repair_partial_total_debt_node(
    *,
    row: dict[str, Any],
    candidates: dict[str, dict[str, list[dict[str, Any]]]],
    computed_at: str,
    provenance_source: str,
) -> bool:
    features = row["features"]
    total_debt = features.get("capital_structure.total_debt_provider_direct")
    if not total_debt:
        return False
    breakdown = total_debt.get("component_breakdown") or {}
    if not (
        breakdown.get("mode") == "partial_debt_stack"
        and breakdown.get("current")
        and not breakdown.get("noncurrent")
    ):
        return False

    entity_id = str(row["company_id"])
    entity_candidates = candidates.get(entity_id, {})
    total_candidates = entity_candidates.get(TOTAL_DEBT_FACT, [])
    current_fact_candidates = entity_candidates.get(CURRENT_DEBT_FACT, [])
    latest_total = _latest_fact(total_candidates)
    latest_current = _latest_fact(current_fact_candidates)
    if latest_total is None:
        return False

    current_value = _node_value(total_debt)
    if current_value is None:
        return False

    current_end = _parse_date((breakdown.get("current") or {}).get("end"))
    selected_fact = None
    repair_reason = None
    latest_total_value = float(latest_total["value"])

    if (
        current_end is not None
        and latest_total.get("end_dt") is not None
        and latest_total["end_dt"] >= current_end
        and latest_total_value > current_value * PARTIAL_TOTAL_DEBT_MIN_LIFT
    ):
        selected_fact = latest_total
        repair_reason = "total_debt_fact_override"
    elif (
        latest_current is not None
        and abs(latest_total_value - float(latest_current["value"])) <= 1.0
    ):
        max_total = max(total_candidates, key=lambda item: float(item["value"]))
        max_total_value = float(max_total["value"])
        if max_total_value > latest_total_value * TOTAL_DEBT_REGRESSION_THRESHOLD:
            selected_fact = max_total
            repair_reason = "total_debt_fact_regression_detected"

    if selected_fact is None or repair_reason is None:
        return False

    as_of_date = _parse_date(row.get("as_of_time"))
    selected_end = selected_fact.get("end_dt")
    age_days = None if as_of_date is None or selected_end is None else (as_of_date - selected_end).days
    quality_flags = [repair_reason]
    if age_days is not None and age_days > MAX_EXACT_AGE_DAYS:
        quality_flags.append("stale_total_debt_fact")

    repaired = _set_metric(
        total_debt,
        value=float(selected_fact["value"]),
        support_mode="proxy_missing_component",
        missing_reason=repair_reason,
        component_breakdown={
            "mode": "fact_registry_total_debt_override",
            "selected_total_debt_fact": selected_fact["meta"],
            "latest_total_debt_fact": latest_total["meta"],
            "latest_debt_current_fact": None if latest_current is None else latest_current["meta"],
            "original_partial_debt_stack": breakdown,
            "age_days": age_days,
            "formula": "override_total_debt_from_fact_registry",
        },
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit=total_debt.get("unit", "usd"),
        quality_flags=quality_flags,
    )
    repaired["input_source_classification"] = "fact_registry_repair"
    repaired["input_layer_bucket_reason"] = "fact_registry_total_debt_repair"
    repaired["primary_source_basis"] = "fact_registry"
    repaired["provenance_artifact_type"] = "FactRegistry"
    features["capital_structure.total_debt_provider_direct"] = repaired
    return True


def _repair_total_debt_node(
    *,
    row: dict[str, Any],
    candidates: dict[str, dict[str, list[dict[str, Any]]]],
    computed_at: str,
    provenance_source: str,
) -> bool:
    features = row["features"]
    total_debt = features.get("capital_structure.total_debt_provider_direct")
    if not total_debt:
        return False
    breakdown = total_debt.get("component_breakdown") or {}
    if breakdown.get("mode") != TARGET_MODE:
        return False

    entity_id = str(row["company_id"])
    entity_candidates = candidates.get(entity_id, {})
    current_candidates = entity_candidates.get(CURRENT_DEBT_FACT, [])
    long_term_candidates = entity_candidates.get(LONG_TERM_DEBT_FACT, [])
    current_fact, long_term_fact, gap_days = _best_pair(current_candidates, long_term_candidates)
    if current_fact is None or long_term_fact is None:
        cur_src = ((breakdown.get("current_statement_debt") or {}).get("source_type"))
        lt_src = ((breakdown.get("long_term_statement_debt") or {}).get("source_type"))
        if total_debt.get("support_mode") == "exact" and cur_src != lt_src:
            downgraded = _set_metric(
                total_debt,
                value=_node_value(total_debt),
                support_mode="proxy_missing_component",
                missing_reason="statement_debt_pair_unresolved",
                component_breakdown={
                    **breakdown,
                    "repair_status": "unresolved_source_mismatch",
                },
                computed_at=computed_at,
                provenance_source=provenance_source,
                unit=total_debt.get("unit", "usd"),
                quality_flags=["statement_debt_pair_unresolved"],
            )
            features["capital_structure.total_debt_provider_direct"] = downgraded
            return True
        return False

    total_value = float(current_fact["value"] + long_term_fact["value"])
    as_of_date = _parse_date(row.get("as_of_time"))
    latest_end = max(current_fact["end_dt"], long_term_fact["end_dt"])
    age_days = None if as_of_date is None else (as_of_date - latest_end).days
    prior_has_capital_lease_overlap = bool(breakdown.get("capital_lease_overlap_detected"))

    support_mode = "exact"
    missing_reason = None
    quality_flags: list[str] = []
    if prior_has_capital_lease_overlap:
        support_mode = "proxy_missing_component"
        missing_reason = "finance_lease_adjustment_unavailable"
        quality_flags.append("finance_lease_adjustment_unavailable")
    elif age_days is not None and age_days > MAX_EXACT_AGE_DAYS:
        support_mode = "proxy_missing_component"
        missing_reason = "statement_debt_pair_stale"
        quality_flags.append("statement_debt_pair_stale")

    repaired = _set_metric(
        total_debt,
        value=total_value,
        support_mode=support_mode,
        missing_reason=missing_reason,
        component_breakdown={
            "mode": TARGET_MODE,
            "current_statement_debt": current_fact["meta"],
            "long_term_statement_debt": long_term_fact["meta"],
            "repaired_prior_breakdown": breakdown,
            "capital_lease_overlap_detected": prior_has_capital_lease_overlap,
            "alignment_gap_days": gap_days,
            "age_days": age_days,
            "formula": "current_debt_statement_direct + long_term_debt_statement_direct",
        },
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit=total_debt.get("unit", "usd"),
        quality_flags=quality_flags,
    )
    repaired["input_source_classification"] = "statement_direct_repair"
    repaired["input_layer_bucket_reason"] = "statement_direct_debt_repair"
    repaired["primary_source_basis"] = "statement_direct"
    repaired["provenance_artifact_type"] = "StatementFact"
    features["capital_structure.total_debt_provider_direct"] = repaired
    return True


def _recompute_standardized(features: dict[str, Any], *, row: dict[str, Any], computed_at: str, provenance_source: str) -> None:
    total_debt = features.get("capital_structure.total_debt_provider_direct")
    cash = features.get("liquidity.cash_and_short_term_investments_provider_direct")
    ebitda = features.get("operating.ebitda_ltm_provider_direct")
    total_debt_value = _node_value(total_debt)
    cash_value = _node_value(cash)
    ebitda_value = _node_value(ebitda)
    as_of_time = row["as_of_time"]

    net_debt_value = None if total_debt_value is None or cash_value is None else total_debt_value - cash_value
    standardized_support = _metric_support_from_components(total_debt, cash)
    features["capital_structure.net_debt_standardized"] = _set_metric(
        features.get("capital_structure.net_debt_standardized"),
        value=net_debt_value,
        support_mode=standardized_support if net_debt_value is not None else "unsupported",
        missing_reason=None if net_debt_value is not None else "component_unavailable",
        component_breakdown={
            "total_debt_provider_direct": total_debt_value,
            "cash_and_short_term_investments_provider_direct": cash_value,
            "formula": "total_debt_provider_direct - cash_and_short_term_investments_provider_direct",
        },
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit="usd",
    )

    if total_debt_value is None or ebitda_value is None:
        gross_value = None
        gross_support = "unsupported"
        gross_missing = "component_unavailable"
    elif ebitda_value <= 0:
        gross_value = None
        gross_support = "unsupported"
        gross_missing = "non_positive_denominator"
    else:
        gross_value = total_debt_value / ebitda_value
        gross_support = _metric_support_from_components(total_debt, ebitda)
        gross_missing = None
    features["capital_structure.gross_leverage_standardized"] = _set_metric(
        features.get("capital_structure.gross_leverage_standardized"),
        value=gross_value,
        support_mode=gross_support,
        missing_reason=gross_missing,
        component_breakdown={
            "total_debt_provider_direct": total_debt_value,
            "ebitda_ltm_provider_direct": ebitda_value,
            "formula": "total_debt_provider_direct / ebitda_ltm_provider_direct",
        },
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit="x",
    )

    if net_debt_value is None or ebitda_value is None:
        net_lev_value = None
        net_lev_support = "unsupported"
        net_lev_missing = "component_unavailable"
    elif ebitda_value <= 0:
        net_lev_value = None
        net_lev_support = "unsupported"
        net_lev_missing = "non_positive_denominator"
    else:
        net_lev_value = net_debt_value / ebitda_value
        net_lev_support = _metric_support_from_components(features["capital_structure.net_debt_standardized"], ebitda)
        net_lev_missing = None
    features["capital_structure.net_leverage_standardized"] = _set_metric(
        features.get("capital_structure.net_leverage_standardized"),
        value=net_lev_value,
        support_mode=net_lev_support,
        missing_reason=net_lev_missing,
        component_breakdown={
            "net_debt_standardized": net_debt_value,
            "ebitda_ltm_provider_direct": ebitda_value,
            "formula": "net_debt_standardized / ebitda_ltm_provider_direct",
        },
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit="x",
    )


def _recompute_smart(features: dict[str, Any], *, row: dict[str, Any], computed_at: str, provenance_source: str) -> None:
    total_debt = features.get("capital_structure.total_debt_provider_direct")
    lease = features.get("capital_structure.lease_liabilities_sec_exact")
    liquidity = features.get("liquidity.available_liquidity_normalized")
    earnings = features.get("operating.operating_earnings_normalized")
    total_debt_value = _node_value(total_debt)
    raw_lease_value = _node_value(lease)
    lease_value = raw_lease_value if raw_lease_value is None or raw_lease_value >= 0 else None
    liquidity_value = _node_value(liquidity)
    earnings_value = _node_value(earnings)

    if total_debt_value is None:
        debt_like_value = None
        debt_like_support = "unsupported"
        debt_like_missing = "component_unavailable"
    else:
        debt_like_value = total_debt_value + (lease_value or 0.0)
        if _is_exact(total_debt) and _is_exact(lease):
            debt_like_support = "exact"
        elif _is_supported(total_debt):
            debt_like_support = "proxy_missing_component"
        else:
            debt_like_support = "unsupported"
        if raw_lease_value is not None and raw_lease_value < 0 and debt_like_support == "exact":
            debt_like_support = "proxy_missing_component"
        debt_like_missing = None if debt_like_support != "unsupported" else "component_unavailable"
        if debt_like_value < total_debt_value:
            debt_like_value = total_debt_value
            if debt_like_support == "exact":
                debt_like_support = "proxy_missing_component"
    features["capital_structure.debt_like_obligations_normalized"] = _set_metric(
        features.get("capital_structure.debt_like_obligations_normalized"),
        value=debt_like_value,
        support_mode=debt_like_support,
        missing_reason=debt_like_missing,
        component_breakdown={
            "baseline_source_metric": "capital_structure.total_debt_provider_direct",
            "baseline_value": total_debt_value,
            "lease_liabilities_sec_exact": lease_value,
            "lease_liabilities_raw_input_value": raw_lease_value,
            "lease_negative_input_ignored": bool(raw_lease_value is not None and raw_lease_value < 0),
            "formula": "total_debt_provider_direct + lease_liabilities_sec_exact",
        },
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit="usd",
    )

    if debt_like_value is None or liquidity_value is None:
        net_debt_value = None
        net_debt_support = "unsupported"
        net_debt_missing = "component_unavailable"
    else:
        net_debt_value = debt_like_value - liquidity_value
        if _is_exact(features["capital_structure.debt_like_obligations_normalized"]) and _is_exact(liquidity):
            net_debt_support = "exact"
        elif _is_supported(features["capital_structure.debt_like_obligations_normalized"]) and _is_supported(liquidity):
            net_debt_support = "proxy_missing_component"
        else:
            net_debt_support = "unsupported"
        net_debt_missing = None if net_debt_support != "unsupported" else "component_unavailable"
    features["capital_structure.net_debt_normalized"] = _set_metric(
        features.get("capital_structure.net_debt_normalized"),
        value=net_debt_value,
        support_mode=net_debt_support,
        missing_reason=net_debt_missing,
        component_breakdown={
            "debt_like_obligations_normalized": debt_like_value,
            "available_liquidity_normalized": liquidity_value,
            "formula": "debt_like_obligations_normalized - available_liquidity_normalized",
        },
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit="usd",
    )

    if debt_like_value is None or earnings_value is None:
        gross_value = None
        gross_support = "unsupported"
        gross_missing = "component_unavailable"
    elif earnings_value <= 0:
        gross_value = None
        gross_support = "unsupported"
        gross_missing = "non_positive_denominator"
    else:
        gross_value = debt_like_value / earnings_value
        if _is_exact(features["capital_structure.debt_like_obligations_normalized"]) and _is_exact(earnings):
            gross_support = "exact"
        elif _is_supported(features["capital_structure.debt_like_obligations_normalized"]) and _is_supported(earnings):
            gross_support = "proxy_missing_component"
        else:
            gross_support = "unsupported"
        gross_missing = None
    features["capital_structure.gross_leverage_normalized"] = _set_metric(
        features.get("capital_structure.gross_leverage_normalized"),
        value=gross_value,
        support_mode=gross_support,
        missing_reason=gross_missing,
        component_breakdown={
            "debt_like_obligations_normalized": debt_like_value,
            "operating_earnings_normalized": earnings_value,
            "formula": "debt_like_obligations_normalized / operating_earnings_normalized",
        },
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit="x",
    )

    if net_debt_value is None or earnings_value is None:
        net_value = None
        net_support = "unsupported"
        net_missing = "component_unavailable"
    elif earnings_value <= 0:
        net_value = None
        net_support = "unsupported"
        net_missing = "non_positive_denominator"
    else:
        net_value = net_debt_value / earnings_value
        if _is_exact(features["capital_structure.net_debt_normalized"]) and _is_exact(earnings):
            net_support = "exact"
        elif _is_supported(features["capital_structure.net_debt_normalized"]) and _is_supported(earnings):
            net_support = "proxy_missing_component"
        else:
            net_support = "unsupported"
        net_missing = None
    features["capital_structure.net_leverage_normalized"] = _set_metric(
        features.get("capital_structure.net_leverage_normalized"),
        value=net_value,
        support_mode=net_support,
        missing_reason=net_missing,
        component_breakdown={
            "net_debt_normalized": net_debt_value,
            "operating_earnings_normalized": earnings_value,
            "formula": "net_debt_normalized / operating_earnings_normalized",
        },
        computed_at=computed_at,
        provenance_source=provenance_source,
        unit="x",
    )


def _process_repair_row(
    *,
    row: dict[str, Any],
    candidates: dict[str, dict[str, list[dict[str, Any]]]],
    computed_at: str,
    provenance_source: str,
    sec_session: requests.Session | None,
    sec_cache_dir: Path,
    timeout_seconds: int | None,
    skip_fact_registry_repair: bool,
) -> tuple[bool, bool, str | None]:
    original_row = copy.deepcopy(row)
    working_row = copy.deepcopy(row)
    working_row.setdefault("features", {})
    row_repaired = False
    sec_row_repaired = False
    try:
        with _company_processing_timeout(timeout_seconds):
            if not skip_fact_registry_repair:
                if _repair_total_debt_node(
                    row=working_row,
                    candidates=candidates,
                    computed_at=computed_at,
                    provenance_source=provenance_source,
                ):
                    row_repaired = True
                elif _repair_partial_total_debt_node(
                    row=working_row,
                    candidates=candidates,
                    computed_at=computed_at,
                    provenance_source=provenance_source,
                ):
                    row_repaired = True
            if sec_session is not None and _repair_total_debt_from_sec_filing(
                row=working_row,
                computed_at=computed_at,
                provenance_source=provenance_source,
                session=sec_session,
                cache_dir=sec_cache_dir,
            ):
                sec_row_repaired = True
                row_repaired = True
            if row_repaired:
                _recompute_standardized(
                    working_row["features"],
                    row=working_row,
                    computed_at=computed_at,
                    provenance_source=provenance_source,
                )
                _recompute_smart(
                    working_row["features"],
                    row=working_row,
                    computed_at=computed_at,
                    provenance_source=provenance_source,
                )
    except _CompanyProcessingTimeoutError:
        row.clear()
        row.update(original_row)
        return False, False, "company_processing_timeout"
    except Exception:  # noqa: BLE001
        row.clear()
        row.update(original_row)
        return False, False, "company_processing_failed"

    row.clear()
    row.update(working_row)
    return row_repaired, sec_row_repaired, None


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)
    out_path = Path(args.out)
    computed_at = _now_iso()
    provenance_source = f"{artifact_path}:statement_debt_override_repair"

    entity_ids, as_of_time = _collect_target_ids(artifact_path)
    requested_company_ids = {str(company_id).zfill(10) for company_id in (args.company_ids or [])}
    requested_company_ids.update(_load_company_ids_file(Path(args.company_ids_file)) if args.company_ids_file else [])
    if requested_company_ids:
        entity_ids = [entity_id for entity_id in entity_ids if str(entity_id).zfill(10) in requested_company_ids]
    if args.batch_size and args.batch_size > 0:
        start = max(args.batch_index, 0) * args.batch_size
        end = start + args.batch_size
        entity_ids = entity_ids[start:end]
    if args.skip_fact_registry_repair:
        candidates = {}
    else:
        candidates = _load_candidates(
            facts_path=Path(args.facts_path),
            entity_ids=entity_ids,
            as_of_time=as_of_time or "2024-12-31T00:00:00Z",
        )

    repaired_count = 0
    sec_repaired_count = 0
    fail_open_counts = {"company_processing_timeout": 0, "company_processing_failed": 0}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_root = _ensure_cache_dir(Path(args.sec_cache_dir)) if args.sec_cache_dir else None
    completed_company_ids = _load_completed_company_ids(out_path) if args.resume else set()
    if args.resume:
        write_mode = "a"
    else:
        out_path.write_text("")
        write_mode = "a"

    with TemporaryDirectory() as temp_cache_dir:
        sec_cache_dir = cache_root or Path(temp_cache_dir)
        sec_session = _sec_session() if args.enable_sec_filing_fallback else None
        with artifact_path.open() as src, out_path.open(write_mode, buffering=1) as dst:
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                company_id = str(row.get("company_id", "")).zfill(10)
                if company_id in completed_company_ids:
                    continue
                if entity_ids and company_id not in entity_ids and (not requested_company_ids or company_id not in requested_company_ids):
                    dst.write(json.dumps(row) + "\n")
                    continue
                row_repaired, sec_row_repaired, failure_reason = _process_repair_row(
                    row=row,
                    candidates=candidates,
                    computed_at=computed_at,
                    provenance_source=provenance_source,
                    sec_session=sec_session,
                    sec_cache_dir=sec_cache_dir,
                    timeout_seconds=args.company_processing_timeout_seconds,
                    skip_fact_registry_repair=args.skip_fact_registry_repair,
                )
                if failure_reason:
                    fail_open_counts[failure_reason] += 1
                if row_repaired:
                    repaired_count += 1
                if sec_row_repaired:
                    sec_repaired_count += 1
                dst.write(json.dumps(row) + "\n")

    print(f"Repaired statement debt overrides ({repaired_count} rows, sec_filing={sec_repaired_count}) -> {out_path}")
    print(
        "row_fail_open:"
        f" company_processing_timeout={fail_open_counts['company_processing_timeout']}"
        f" company_processing_failed={fail_open_counts['company_processing_failed']}"
    )


if __name__ == "__main__":
    main()
