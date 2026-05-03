#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.company_state_builder import CompanyStateBuilder
from src.historical_recommendation_eval import _build_historical_alias_overrides, _cached_snapshot_loader


def _parse_args() :
    parser = argparse.ArgumentParser(
        description="Build a keyed snapshot cache for the fixed historical cases in a manifest."
    )
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--companyfacts-root", default="")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _emit(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload), flush=True)


