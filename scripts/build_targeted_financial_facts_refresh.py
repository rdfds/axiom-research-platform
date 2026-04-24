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


