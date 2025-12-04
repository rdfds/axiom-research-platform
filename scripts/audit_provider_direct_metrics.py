#!/usr/bin/env python3
"""Audit the six core provider-direct metrics against raw-source reconstructions.

This is intentionally a triangulation audit, not a circular "rerun the same
artifact builder and call it validated" pass.

For each metric we compare the artifact value to:
1. the live provider reference row (Refinitiv sidecar)
2. a raw-source reconstruction from SEC companyfacts and/or PIT price history

The output is a compact JSON report with support counts, source-basis counts,
gap stats, and the largest discrepancy examples.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable

import pandas as pd

import backfill_input_layer_v1_metrics as core
import backfill_market_macro_input_layer_v1 as market_macro


METRICS = (
    "market.market_cap_provider_direct",
    "operating.revenue_ttm_provider_direct",
    "operating.ebitda_ltm_provider_direct",
    "earnings.net_income_ttm_provider_direct",
    "liquidity.cash_and_short_term_investments_provider_direct",
    "capital_structure.total_debt_provider_direct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument(
        "--taxonomy-reference-path",
        default=str(Path(__file__).resolve().parents[1] / "data" / "refinitiv" / "fundamentals_all.parquet"),
    )
    parser.add_argument(
        "--entity-identifier-path",
        default=str(Path(__file__).resolve().parents[1] / "data" / "inputs_layer" / "entity_identifier.parquet"),
    )
    parser.add_argument(
        "--companyfacts-root",
        default=str(Path(__file__).resolve().parents[1] / "data" / "sec" / "companyfacts"),
    )
    parser.add_argument(
        "--raw-timeseries-path",
        default=str(Path(__file__).resolve().parents[1] / "data" / "inputs_layer" / "raw_timeseries.parquet"),
    )
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def iter_snapshot_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _pct_gap(left: float | None, right: float | None) :
    if left is None or right is None:
        return None
    denom = max(abs(float(right)), 1.0)
    return abs(float(left) - float(right)) / denom


