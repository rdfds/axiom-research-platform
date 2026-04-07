#!/usr/bin/env python3
"""Inspect candidate dividend provider files for direct dividend fields.

This audit is intentionally read-only. It helps answer whether we have a truly
direct provider field for dividend per share or dividend yield, rather than an
event series we would need to aggregate ourselves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


DEFAULT_FILES = [
    "./data/refinitiv/dividends_complete.parquet",
    "./data/refinitiv/dividends_with_amounts.parquet",
    "./data/dividend_profiles.parquet",
    "./data/dividend_actions.parquet",
    "./data/special_dividends.parquet",
    "./data/special_dividends_linked.parquet",
]

KEYWORDS = ("div", "yield", "special", "regular", "cash", "amount", "trailing", "forward")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="*", default=DEFAULT_FILES, help="Parquet files to inspect")
    parser.add_argument("--sample-rows", type=int, default=3, help="Number of sample rows to print")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    con = duckdb.connect()
    for raw_path in args.files:
        path = Path(raw_path)
        print(f"\nFILE {path}")
        if not path.exists():
            print("  missing")
            continue
        print(f"  size_bytes={path.stat().st_size}")
        try:
            desc = con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{path.as_posix()}')"
            ).fetchall()
        except Exception as exc:
            print(f"  read_error={type(exc).__name__}: {exc}")
            continue

        columns = [row[0] for row in desc]
        candidate_columns = [col for col in columns if any(k in col.lower() for k in KEYWORDS)]
        print(f"  candidate_columns={candidate_columns}")

        if candidate_columns:
            selected = ", ".join([f'"{col}"' for col in candidate_columns[:12]])
            try:
                samples = con.execute(
                    f"SELECT {selected} FROM read_parquet('{path.as_posix()}') LIMIT {args.sample_rows}"
                ).fetchdf()
                print(samples.to_string(index=False))
            except Exception as exc:
                print(f"  sample_error={type(exc).__name__}: {exc}")

    print("\nPromotion rule:")
    print("  green only if there is a single direct provider field for DPS or yield with no required in-house aggregation")


if __name__ == "__main__":
    main()
