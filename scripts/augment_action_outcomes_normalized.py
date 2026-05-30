#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.action_normalization import augment_action_outcomes_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add lossless normalization columns to an existing action outcomes parquet."
    )
    parser.add_argument("--in-path", required=True, help="Input parquet path.")
    parser.add_argument("--out-path", required=True, help="Output parquet path.")
    return parser.parse_args()


