#!/usr/bin/env python3
"""Repair market-pricing metrics in a materialized company-state artifact.

This is a narrow repair pass for metrics that are conceptually simple but can be
missing in the built artifact even when the underlying normalized inputs are
already present.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


REPAIR_METRICS = [
    "market.enterprise_value",
    "market.ev_ebitda",
    "market.pe_ratio",
    "operating.ebitda_margin_ttm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True, help="Input company-state JSONL artifact")
    parser.add_argument("--out", required=True, help="Output repaired JSONL artifact")
    parser.add_argument("--summary-out", help="Optional summary JSON")
    return parser.parse_args()


