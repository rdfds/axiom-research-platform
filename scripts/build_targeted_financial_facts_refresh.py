#!/usr/bin/env python
"""
Build a tiny SEC-derived financial-facts refresh for a targeted company set.

This avoids the full all-company small-file rebuild when we only need to
validate a handful of companies.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.financial_fact_tags import FINANCIAL_FACT_TAGS
from src.ingestion import compute_version_id
from src.sec_companyfacts_bulk import CompanyFactsBulkSource


def _load_sec_ingest_module():
    path = ROOT / "scripts" / "26_ingest_sec_xbrl.py"
    spec = importlib.util.spec_from_file_location("sec_ingest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build targeted SEC financial-facts refresh.")
    parser.add_argument("--ciks", default=None, help="Comma-separated CIKs/company_ids.")
    parser.add_argument("--ciks-file", default=None, help="Path to newline/comma-delimited CIK/company_id file.")
    parser.add_argument("--years", default="2023,2024")
    parser.add_argument("--companyfacts-zip", default=str(ROOT / "data" / "sec" / "companyfacts.zip"))
    parser.add_argument("--companyfacts-dir", default=str(ROOT / "data" / "sec" / "companyfacts"))
    parser.add_argument("--warehouse-root", default="/tmp/targeted_financial_facts_refresh/warehouse_financials")
    parser.add_argument("--enriched-root", default="/tmp/targeted_financial_facts_refresh/enriched")
    parser.add_argument("--validity-root", default="/tmp/targeted_financial_facts_refresh/validity")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory", default="12GB")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


