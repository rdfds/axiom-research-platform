#!/usr/bin/env python3
"""Repair debt-stack metrics in an already-materialized input-layer artifact.

This is intentionally narrow and fast:

1. Recompute `capital_structure.total_debt_provider_direct` from SEC companyfacts
   using the latest debt-stack logic.
2. Recompute the standardized debt/leverage combos that directly depend on it.

We use this repair pass when the underlying debt builder has improved and we
want to refresh existing artifacts without re-running the whole pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backfill_input_layer_v1_metrics as core  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--companyfacts-root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


