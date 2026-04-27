#!/usr/bin/env python
"""
Export CompanyState from long format to wide format.

This is optional and can be heavy for large datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="data/company_state/company_state.parquet")
    parser.add_argument("--out", default="data/company_state/company_state_wide.parquet")
    args = parser.parse_args()

    inp = ROOT / args.inp
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(inp)
    df["col"] = df["feature_group"] + "::" + df["feature_key"]
    # Prefer numeric where available, else string
    df["value"] = df["value_num"].combine_first(df["value_str"])
    wide = df.pivot_table(index="entity_id", columns="col", values="value", aggfunc="last")
    wide.reset_index().to_parquet(out, index=False)
    print(f"Saved wide CompanyState -> {out} ({len(wide):,} entities)")


if __name__ == "__main__":
    main()
