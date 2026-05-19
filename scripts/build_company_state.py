#!/usr/bin/env python
"""
Build a point-in-time CompanyState snapshot.

This is a baseline assembler that joins:
- RawTimeSeriesStore (prices + macro + estimates)
- EventRegistry (corporate actions)
- ExtractedFactRegistry (text-derived signals)
- EntityGraph (ID resolution)

Default output is a per-entity, per-asof snapshot in LONG format for scale.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import duckdb


ROOT = Path(__file__).resolve().parents[1]


def utc_now() :
    return datetime.now(timezone.utc).isoformat()


