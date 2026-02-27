#!/usr/bin/env python3
"""Overlay WRDS CDS spreads onto the flat quantitative comps export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TIER_RANK = {"SNRFOR": 3, "SUBLT2": 2, "SECDOM": 1}


def _company_id_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d+)")[0].str.zfill(10)


