#!/usr/bin/env python
"""
Validate Inputs Layer datasets against JSON schemas.

This script performs lightweight checks:
- required columns
- basic dtype compatibility
- timestamp parseability
- timestamp ordering vs ingested_at
- confidence_score bounds

Outputs a DataIntegrityLog parquet.
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_schema(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


