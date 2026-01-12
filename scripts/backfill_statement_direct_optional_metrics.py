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
            end_dt = _parse_iso_date(entry['end'])
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


