#!/usr/bin/env python3
"""Run the provider-direct external audit in chunked parallel batches.

This keeps the exact audit logic from `audit_provider_direct_metrics.py`, but
avoids one large monolithic pass over the full artifact by splitting the
artifact rows into smaller chunks, auditing each chunk independently, and then
merging the partial results back into one exact report.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import audit_provider_direct_metrics as audit_core
import backfill_input_layer_v1_metrics as core
import backfill_market_macro_input_layer_v1 as market_macro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument(
        "--taxonomy-reference-path",
        default=str(ROOT / "data" / "refinitiv" / "fundamentals_all.parquet"),
    )
    parser.add_argument(
        "--entity-identifier-path",
        default=str(ROOT / "data" / "inputs_layer" / "entity_identifier.parquet"),
    )
    parser.add_argument(
        "--companyfacts-root",
        default=str(ROOT / "data" / "sec" / "companyfacts"),
    )
    parser.add_argument(
        "--raw-timeseries-path",
        default=str(ROOT / "data" / "inputs_layer" / "raw_timeseries.parquet"),
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument(
        "--skip-market-cap",
        action="store_true",
        help="Skip market cap in this audit. Useful when the market stack has already been audited separately via CRSP.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=96,
        help="Number of rows to audit per worker chunk.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum worker processes to use.",
    )
    return parser.parse_args()


def _selected_metrics(skip_market_cap: bool) -> tuple[str, ...]:
    if not skip_market_cap:
        return tuple(audit_core.METRICS)
    return tuple(metric for metric in audit_core.METRICS if metric != "market.market_cap_provider_direct")


def _iter_snapshot_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _chunk_rows(rows: List[Dict[str, Any]], chunk_size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(rows), chunk_size):
        yield rows[start : start + chunk_size]


