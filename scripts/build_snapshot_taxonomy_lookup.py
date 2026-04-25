#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def _extract_metric_value(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("value")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact taxonomy lookup from keyed snapshot JSON files.")
    parser.add_argument("--snapshot-root")
    parser.add_argument("--snapshot-catalog-path")
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


