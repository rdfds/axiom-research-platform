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


def main() -> None:
    args = _parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input parquet: {in_path}")

    df = pd.read_parquet(in_path)
    print(f"[augment_action_outcomes_normalized] loaded {in_path} rows={len(df):,}", flush=True)

    df = augment_action_outcomes_df(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[augment_action_outcomes_normalized] wrote {out_path} rows={len(df):,}", flush=True)


if __name__ == "__main__":
    main()
