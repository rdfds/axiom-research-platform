#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge selected causal-model cells from an overlay artifact into a base artifact.")
    p.add_argument("--base-model", required=True, help="Champion/base causal model artifact JSON.")
    p.add_argument("--overlay-model", required=True, help="Overlay/rescue causal model artifact JSON.")
    p.add_argument(
        "--selection-json",
        required=True,
        help="JSON file describing which objective/cell pairs to copy from the overlay.",
    )
    p.add_argument("--out-model", required=True, help="Output merged causal model artifact JSON.")
    p.add_argument(
        "--out-model-card",
        default="",
        help="Optional merged model-card JSON output. Defaults to <out-model>.model_card.json",
    )
    return p.parse_args()


