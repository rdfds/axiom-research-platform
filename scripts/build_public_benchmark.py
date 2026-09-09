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


def _parse_placebo_rows(text: str) -> list[dict[str, float]]:
    section = _required(
        r"## Placebo Check\n\n([\s\S]*?)(?=\n## )",
        text,
        "placebo table",
    )
    rows: list[dict[str, float]] = []
    for line in section.splitlines():
        if not line.startswith("|") or "---" in line or "Lambda" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 5:
            continue
        values = [float(cell) for cell in cells]
        rows.append(
            {
                "lambda": values[0],
                "actual_mean_mae_improvement": values[1],
                "placebo_mean_mae_improvement": values[2],
                "actual_minus_placebo": values[3],
                "actual_beats_placebo_rate": values[4],
            }
        )
    if not rows:
        raise ValueError(f"Could not parse placebo rows from {SOURCE_PATH}")
    return rows


