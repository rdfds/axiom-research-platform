#!/usr/bin/env python3
"""Build a filtered CRSP daily market cache for the current entity universe.

This scans the large CRSP daily stock export once, keeps only the permnos that
exist in the current entity identifier file, and writes a compact parquet cache
that downstream scripts can query quickly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity-identifier-path", required=True)
    parser.add_argument("--crsp-daily-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--date-from", required=True, help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--date-to", required=True, help="Inclusive end date YYYY-MM-DD")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE target_permnos AS
        SELECT DISTINCT
            CAST(identifier_value AS BIGINT) AS permno
        FROM read_parquet('{args.entity_identifier_path}')
        WHERE lower(CAST(identifier_type AS VARCHAR)) = 'permno'
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT
                CAST(src.PERMNO AS BIGINT) AS permno,
                CAST(src.DlyCalDt AS DATE) AS trade_date,
                CAST(NULLIF(src.DlyClose, '') AS DOUBLE) AS close_price,
                ABS(CAST(NULLIF(src.DlyPrc, '') AS DOUBLE)) AS price_proxy,
                CAST(NULLIF(src.DlyRet, '') AS DOUBLE) AS total_return,
                CAST(NULLIF(src.DlyRetx, '') AS DOUBLE) AS price_return,
                CAST(NULLIF(src.ShrOut, '') AS DOUBLE) AS shares_outstanding,
                CAST(NULLIF(src.DlyCap, '') AS DOUBLE) AS daily_cap,
                CAST(src.DlyDelFlg AS VARCHAR) AS delist_flag
            FROM read_csv_auto(
                '{args.crsp_daily_path}',
                header = TRUE,
                all_varchar = TRUE
            ) AS src
            INNER JOIN target_permnos AS p
                ON CAST(src.PERMNO AS BIGINT) = p.permno
            WHERE CAST(src.DlyCalDt AS DATE) BETWEEN DATE '{args.date_from}' AND DATE '{args.date_to}'
        ) TO '{out_path}' (FORMAT PARQUET)
        """
    )
    print(f"Wrote CRSP market cache -> {out_path}")


if __name__ == "__main__":
    main()
