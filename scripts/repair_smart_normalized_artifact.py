#!/usr/bin/env python3
"""Refresh smart-normalized metrics in an already-materialized artifact."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backfill_smart_normalized_metrics_v1 as smart  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--metric-registry-path", required=True)
    parser.add_argument("--component-policy-path", required=True)
    parser.add_argument("--source-precedence-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out")
    return parser.parse_args()


