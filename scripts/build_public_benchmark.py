#!/usr/bin/env python3
"""Build the deterministic public benchmark artifact from its committed report."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = Path(
    "examples/hd_market_expectations/"
    "forward_gap_placebo_walk_forward_operating_ex_energy.sample.md"
)
DEFAULT_OUTPUT_PATH = Path("results/public_benchmark.json")


def _required(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not parse {label} from {SOURCE_PATH}")
    return match.group(1)


