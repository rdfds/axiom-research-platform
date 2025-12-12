#!/usr/bin/env python
"""
Build a compact FMP fundamentals cache for fast lookup.

Outputs a wide, per-ticker quarterly table with key income/balance/cash items.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb


INCOME_ITEMS = [
    "Revenue",
    "EBITDA",
    "NetIncome",
    "EPS",
    "EPSDiluted",
    "SharesOut",
    "SharesOutDiluted",
]
BALANCE_ITEMS = [
    "Cash",
    "ShortTermInvestments",
    "DebtCurrent",
    "DebtLongTerm",
    "TotalAssets",
]
CASH_ITEMS = [
    "OperatingCashFlow",
    "Capex",
    "FreeCashFlow",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-path",
        default=str(Path("data/warehouse/warehouse_financials/year=*/part_*.parquet")),
        help="Input parquet glob for FMP financials (long format).",
    )
    parser.add_argument(
        "--out-path",
        default=str(Path("data/curated/fmp_fundamentals_cache.parquet")),
        help="Output parquet path for compact FMP fundamentals cache.",
    )
    parser.add_argument(
        "--skip-count",
        action="store_true",
        help="Skip the upfront row count (faster start).",
    )
    parser.add_argument(
        "--yearly",
        action="store_true",
        help="Process one year at a time with progress logging (writes a partitioned output dir).",
    )
    args = parser.parse_args()

    in_path = args.in_path
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    t0 = time.time()
    if args.skip_count:
        print("[build_fmp_cache] Skipping row count (faster start).")
    else:
        print("[build_fmp_cache] Counting FMP rows to estimate runtime...")
        total_rows = con.execute(
            f"""
            select count(*) from read_parquet('{in_path}', hive_partitioning=1)
            where source_system='fmp_financials'
              and statement_type in ('income','balance_sheet','balance','cash_flow')
              and line_item in ({",".join([f"'{i}'" for i in INCOME_ITEMS + BALANCE_ITEMS + CASH_ITEMS])})
            """
        ).fetchone()[0]
        print(f"[build_fmp_cache] Rows to scan: {total_rows:,}")

    income_sql = ",".join([f"'{item}'" for item in INCOME_ITEMS])
    balance_sql = ",".join([f"'{item}'" for item in BALANCE_ITEMS])
    cash_sql = ",".join([f"'{item}'" for item in CASH_ITEMS])
    query = f"""
        select
            upper(company_id) as ticker,
            fiscal_period_end as datadate,
            fiscal_year,
            fiscal_quarter,
            max(case when statement_type='income' and line_item='Revenue' then value end) as revenue,
            max(case when statement_type='income' and line_item='EBITDA' then value end) as ebitda,
            max(case when statement_type='income' and line_item='NetIncome' then value end) as net_income,
            max(case when statement_type='income' and line_item='EPS' then value end) as eps,
            max(case when statement_type='income' and line_item='EPSDiluted' then value end) as eps_diluted,
            max(case when statement_type='income' and line_item='SharesOut' then value end) as shares_out,
            max(case when statement_type='income' and line_item='SharesOutDiluted' then value end) as shares_out_diluted,
            max(case when statement_type in ('balance_sheet','balance') and line_item='Cash' then value end) as cash,
            max(case when statement_type in ('balance_sheet','balance') and line_item='ShortTermInvestments' then value end) as short_term_investments,
            max(case when statement_type in ('balance_sheet','balance') and line_item='DebtCurrent' then value end) as debt_current,
            max(case when statement_type in ('balance_sheet','balance') and line_item='DebtLongTerm' then value end) as debt_long_term,
            max(case when statement_type in ('balance_sheet','balance') and line_item='TotalAssets' then value end) as total_assets,
            max(case when statement_type='cash_flow' and line_item='OperatingCashFlow' then value end) as operating_cash_flow,
            max(case when statement_type='cash_flow' and line_item='Capex' then value end) as capex,
            max(case when statement_type='cash_flow' and line_item='FreeCashFlow' then value end) as free_cash_flow
        from read_parquet('{in_path}', hive_partitioning=1)
        where source_system='fmp_financials'
          and statement_type in ('income','balance_sheet','balance','cash_flow')
          and line_item in ({income_sql}, {balance_sql}, {cash_sql})
        group by ticker, datadate, fiscal_year, fiscal_quarter
    """
    if args.yearly:
        out_dir = out_path if out_path.suffix != ".parquet" else out_path.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        base = Path(in_path.split("year=")[0]) if "year=" in in_path else Path(in_path).parent
        year_dirs = sorted(base.glob("year=*"))
        print(f"[build_fmp_cache] Processing {len(year_dirs)} years into {out_dir}...")
        for year_dir in year_dirs:
            year = year_dir.name.split("=")[-1]
            year_in = str(year_dir / "part_*.parquet")
            year_out_dir = out_dir / f"year={year}"
            year_out_dir.mkdir(parents=True, exist_ok=True)
            year_out = year_out_dir / f"part_{year}.parquet"
            print(f"[build_fmp_cache] Year {year}: building cache...")
            year_query = query.replace(in_path, year_in)
            con.execute(f"COPY ({year_query}) TO '{year_out.as_posix()}' (FORMAT 'parquet');")
            print(f"[build_fmp_cache] Year {year}: done -> {year_out.name}")
        print(f"[build_fmp_cache] Saved partitioned cache -> {out_dir} in {time.time() - t0:.1f}s")
    else:
        print("[build_fmp_cache] Building compact FMP fundamentals cache (this may take a while)...")
        con.execute(f"COPY ({query}) TO '{out_path.as_posix()}' (FORMAT 'parquet');")
        print(f"[build_fmp_cache] Saved -> {out_path} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
