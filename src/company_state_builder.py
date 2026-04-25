"""
CompanyStateSnapshot builder (World Model spec).

This module assembles a point-in-time company snapshot using only
information available as-of a given timestamp. It is intentionally
conservative and auditable: every feature carries provenance and
confidence metadata. Missing data is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re
import subprocess
import time
import uuid

import duckdb
import numpy as np
import pandas as pd

from scripts.backfill_sec_companyfacts_components import (
    _extract_lease_liabilities as _extract_sec_companyfacts_lease_liabilities,
    _extract_marketable_securities as _extract_sec_companyfacts_marketable_securities,
    _extract_restricted_cash as _extract_sec_companyfacts_restricted_cash,
    _extract_revolver_undrawn as _extract_sec_companyfacts_revolver_undrawn,
)
from scripts.backfill_input_layer_v1_metrics import (
    DEBT_CURRENT_CONCEPTS,
    DEBT_NONCURRENT_CONCEPTS,
    SHORT_TERM_BORROWINGS_CONCEPTS,
    _build_sec_core_metric,
    _instant_candidates,
)
from scripts.backfill_smart_normalized_metrics_v1 import materialize_smart_metrics_for_row

from .company_state_input_source_registry import CompanyStateInputSourceRegistry
from .company_state_components import RegimeClassifier, PeerSetResolver
from .data_paths import resolve_companyfacts_root, resolve_data_path
from .metric_policy import MetricPolicyEngine, TaxonomyContext


ROOT = Path(__file__).resolve().parents[1]
MAX_SEC_FACT_AGE_DAYS = 550
COMPANYFACTS_FRESHER_OVERRIDE_MIN_DAYS = 7
COMPANYFACTS_LOAD_TIMEOUT_SECONDS = 1.5
COMPANYFACTS_CASH_EQ_CONCEPTS = ["CashAndCashEquivalentsAtCarryingValue", "Cash"]
INTEREST_EXPENSE_TTM_EXACT_CONCEPTS = ["InterestExpense"]


def _smart_metric_registry_path() -> Path:
    env = str(os.environ.get("AXIOM_SMART_METRIC_REGISTRY_PATH", "") or "").strip()
    if env:
        return Path(env)
    return ROOT / "out" / "smart_metric_registry_v1.json"


