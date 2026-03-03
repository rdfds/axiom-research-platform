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


def _count_support(series: pd.Series) -> dict[str, int]:
    vc = series.fillna("unsupported").value_counts()
    return {
        "exact": int(vc.get("exact", 0)),
        "proxy_missing_component": int(vc.get("proxy_missing_component", 0)),
        "unsupported": int(vc.get("unsupported", 0)),
    }


def _prepare_cds(cds_path: Path, as_of_date: str) -> pd.DataFrame:
    cds = pd.read_csv(cds_path)
    cds["date"] = pd.to_datetime(cds["date"])
    cds = cds[
        (cds["tenor"] == "5Y")
        & (cds["currency"] == "USD")
        & (cds["date"] <= pd.Timestamp(as_of_date))
        & cds["parspread"].notna()
    ].copy()

    cds["tier_rank"] = cds["tier"].map(TIER_RANK).fillna(0)
    cds["primarycurve_rank"] = (cds["primarycurve"] == "Y").astype(int)
    cds["liq_nonnull"] = cds["curveliquidityscore"].notna().astype(int)

    # Prefer senior curves, then primary curves, then rows with actual liquidity scores.
    cds = cds.sort_values(
        [
            "redcode",
            "date",
            "tier_rank",
            "primarycurve_rank",
            "liq_nonnull",
            "curveliquidityscore",
        ],
        ascending=[True, True, False, False, False, False],
    )
    return cds.groupby(["redcode", "date"], as_index=False).head(1).copy()


