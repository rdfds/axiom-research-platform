#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SNAPSHOT_CATALOG_PATH = (
    REPO_ROOT
    / "out/materialized_feedback_20260405/company_state_snapshots_audit_catalog.asof_safe_enriched_v1.jsonl.gz"
)
DEFAULT_OUT_PATH = (
    REPO_ROOT
    / "out/materialized_feedback_20260405/company_state_snapshots_input_complete_catalog.asof_safe_v1.jsonl.gz"
)
DEFAULT_COMPANYFACTS_ROOT = REPO_ROOT / "data/sec/companyfacts"
DEFAULT_ENTITY_IDENTIFIER_PATH = REPO_ROOT / "data/inputs_layer/entity_identifier.parquet"
DEFAULT_CRSP_DAILY_ROOT = REPO_ROOT / "data/wrds/crsp"

TARGET_METRICS = [
    "operating.revenue_ttm_provider_direct",
    "operating.revenue_ttm_lag_1y",
    "operating.ebitda_ltm_provider_direct",
    "cash_flow.free_cash_flow_ttm",
    "capital_structure.total_debt_provider_direct",
    "capital_structure.debt_due_next_24m",
    "liquidity.cash_and_short_term_investments_provider_direct",
    "liquidity.marketable_securities_sec_exact",
    "market.market_cap_provider_direct",
    "market.enterprise_value",
    "market.ev_ebitda",
    "market.fcf_yield",
    "market.volatility_30d",
    "market.volatility_90d",
    "market.drawdown_90d",
]


def _parse_args() :
    parser = argparse.ArgumentParser(
        description="Build an as-of-safe snapshot catalog with completeness enrichment applied universe-wide."
    )
    parser.add_argument("--snapshot-catalog-path", default=str(DEFAULT_SNAPSHOT_CATALOG_PATH))
    parser.add_argument("--companyfacts-root", default=str(DEFAULT_COMPANYFACTS_ROOT))
    parser.add_argument("--entity-identifier-path", default=str(DEFAULT_ENTITY_IDENTIFIER_PATH))
    parser.add_argument("--crsp-daily-root", default=str(DEFAULT_CRSP_DAILY_ROOT))
    parser.add_argument("--crsp-market-cache-path", default="")
    parser.add_argument("--out-path", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--summary-path", default="")
    return parser.parse_args()


