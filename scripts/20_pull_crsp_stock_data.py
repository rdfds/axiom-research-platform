#!/usr/bin/env python
"""
Pull CRSP Stock (US equities) via WRDS
======================================
This script pulls the core CRSP Stock tables for US equities:
  - msenames (security master / name history)
  - msedist (distributions: dividends, splits, spin-offs, etc.)
  - msedelist (delistings)
  - msf (monthly stock file)
  - dsf (daily stock file, optional)
  - ccmxpf_lnkhist (Compustat link table)

Defaults:
  - US common stocks only (shrcd 10/11, exchcd 1/2/3)
  - 2000-01-01 through today
  - monthly + distributions + delistings + names + link table

Usage:
  python -u scripts/20_pull_crsp_stock_data.py [WRDS_USERNAME]

Examples:
  # Smoke test (last 30 days, monthly only)
  python -u scripts/20_pull_crsp_stock_data.py --start 2026-01-01 --end 2026-01-31

  # Full 2000-present monthly + distributions + delistings
  python -u scripts/20_pull_crsp_stock_data.py --start 2000-01-01

  # Re-pull distributions with broader filters (all share codes) and overwrite
  python -u scripts/20_pull_crsp_stock_data.py --start 2000-01-01 --end 2024-12-31 \\
    --no-monthly --no-delistings --no-names --no-link --all-shares --force

  # Include daily (very large)
  python -u scripts/20_pull_crsp_stock_data.py --start 2000-01-01 --daily
"""

import argparse
import json
from datetime import datetime, date
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import wrds


DATA_DIR = Path(__file__).parent.parent / "data" / "wrds" / "crsp"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def year_chunks(start: str, end: str, chunk_years: int) -> Iterable[Tuple[str, str]]:
    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()

    year = start_date.year
    while year <= end_date.year:
        chunk_start = date(year, 1, 1)
        chunk_end_year = min(year + chunk_years - 1, end_date.year)
        chunk_end = date(chunk_end_year, 12, 31)
        if year == start_date.year:
            chunk_start = start_date
        if chunk_end_year == end_date.year:
            chunk_end = end_date
        yield chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        year += chunk_years


def build_permno_cte(common_only: bool, shrcd: List[int], exchcd: List[int]) -> str:
    where = "nameendt >= %(start_date)s AND namedt <= %(end_date)s"
    if common_only:
        where += f" AND shrcd IN ({','.join(str(x) for x in shrcd)})"
        where += f" AND exchcd IN ({','.join(str(x) for x in exchcd)})"
    return f"""
    WITH permnos AS (
        SELECT DISTINCT permno
        FROM crsp.msenames
        WHERE {where}
    )
    """


def write_manifest(entry: dict) -> None:
    manifest_path = DATA_DIR / "manifest.jsonl"
    with open(manifest_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_query(
    db: wrds.Connection,
    query: str,
    params: dict,
    out_path: Path,
    table_label: str,
    chunksize: int,
    force: bool,
) -> int:
    if force:
        if out_path.exists():
            out_path.unlink()
        for part in out_path.parent.glob(out_path.stem + "_part_*.parquet"):
            part.unlink()
    elif out_path.exists():
        return 0

    result = db.raw_sql(query, params=params, chunksize=chunksize if chunksize and chunksize > 0 else None)

    # Some WRDS setups ignore chunksize and return a full DataFrame
    if isinstance(result, pd.DataFrame):
        if result is None or len(result) == 0:
            return 0
        result.to_parquet(out_path, index=False)
        write_manifest({"file": out_path.name, "rows": len(result), "table": table_label, "timestamp": datetime.now().isoformat()})
        return len(result)

    rows = 0
    if result is None:
        return 0

    for idx, chunk in enumerate(result):
        if not isinstance(chunk, pd.DataFrame):
            raise TypeError(f"Unexpected chunk type from WRDS: {type(chunk)}")
        part_path = out_path.with_name(out_path.stem + f"_part_{idx:04d}" + out_path.suffix)
        chunk.to_parquet(part_path, index=False)
        rows += len(chunk)

    if rows > 0:
        write_manifest({"file": out_path.name, "rows": rows, "table": table_label, "timestamp": datetime.now().isoformat()})
    return rows


def pull_msenames(db, start_date, end_date, common_only, shrcd, exchcd, chunksize, force):
    log("Pulling crsp.msenames ...")
    where = "namedt <= %(end_date)s AND nameendt >= %(start_date)s"
    if common_only:
        where += f" AND shrcd IN ({','.join(str(x) for x in shrcd)})"
        where += f" AND exchcd IN ({','.join(str(x) for x in exchcd)})"
    query = f"""
    SELECT *
    FROM crsp.msenames
    WHERE {where}
    """
    out_path = DATA_DIR / f"msenames_{start_date}_to_{end_date}.parquet"
    rows = run_query(
        db,
        query,
        {"start_date": start_date, "end_date": end_date},
        out_path,
        "msenames",
        chunksize,
        force=force,
    )
    log(f"Saved {rows:,} rows -> {out_path.name}" if rows else "No rows returned for msenames.")


def pull_linktable(db, start_date, end_date, common_only, shrcd, exchcd, chunksize, force):
    log("Pulling crsp.ccmxpf_lnkhist ...")
    cte = build_permno_cte(common_only, shrcd, exchcd)
    query = f"""
    {cte}
    SELECT l.*
    FROM crsp.ccmxpf_lnkhist l
    JOIN permnos p ON l.lpermno = p.permno
    """
    out_path = DATA_DIR / "ccmxpf_lnkhist.parquet"
    rows = run_query(
        db,
        query,
        {"start_date": start_date, "end_date": end_date},
        out_path,
        "ccmxpf_lnkhist",
        chunksize,
        force=force,
    )
    log(f"Saved {rows:,} rows -> {out_path.name}" if rows else "No rows returned for link table.")


def pull_table_by_year(
    db,
    table: str,
    date_col: str,
    label: str,
    start_date: str,
    end_date: str,
    common_only: bool,
    shrcd: List[int],
    exchcd: List[int],
    chunksize: int,
    chunk_years: int,
    force: bool,
):
    log(f"Pulling {table} ...")
    cte = build_permno_cte(common_only, shrcd, exchcd)
    base_query = f"""
    {cte}
    SELECT a.*
    FROM {table} a
    JOIN permnos p ON a.permno = p.permno
    WHERE a.{date_col} BETWEEN %(start_date)s AND %(end_date)s
    """

    total_rows = 0
    for chunk_start, chunk_end in year_chunks(start_date, end_date, chunk_years):
        out_path = DATA_DIR / f"{label}_{chunk_start}_to_{chunk_end}.parquet"
        rows = run_query(
            db,
            base_query,
            {"start_date": chunk_start, "end_date": chunk_end},
            out_path,
            label,
            chunksize,
            force=force,
        )
        total_rows += rows
        log(f"  {label} {chunk_start} -> {chunk_end}: {rows:,} rows")

    log(f"Total {label} rows: {total_rows:,}")


def main():
    parser = argparse.ArgumentParser(description="Pull CRSP Stock data from WRDS")
    parser.add_argument("username", nargs="?", default=None, help="WRDS username (optional)")
    parser.add_argument("--start", default="2000-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD); default today")
    parser.add_argument("--daily", action="store_true", help="Include daily stock file (dsf)")
    parser.add_argument("--no-monthly", action="store_true", help="Skip monthly stock file (msf)")
    parser.add_argument("--no-distributions", action="store_true", help="Skip distributions (msedist)")
    parser.add_argument("--no-delistings", action="store_true", help="Skip delistings (msedelist)")
    parser.add_argument("--no-names", action="store_true", help="Skip msenames pull")
    parser.add_argument("--no-link", action="store_true", help="Skip ccmxpf_lnkhist pull")
    parser.add_argument("--all-shares", action="store_true", help="Do not filter by shrcd/exchcd")
    parser.add_argument("--shrcd", default="10,11", help="Share codes to include (comma-separated)")
    parser.add_argument("--exchcd", default="1,2,3", help="Exchange codes to include (comma-separated)")
    parser.add_argument("--chunk-years", type=int, default=1, help="Year chunk size")
    parser.add_argument("--chunksize", type=int, default=0, help="DB fetch chunk size (rows)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files")

    args = parser.parse_args()
    end_date = args.end or datetime.utcnow().strftime("%Y-%m-%d")

    common_only = not args.all_shares
    shrcd = parse_int_list(args.shrcd)
    exchcd = parse_int_list(args.exchcd)

    log("Connecting to WRDS...")
    db = wrds.Connection(wrds_username=args.username) if args.username else wrds.Connection()
    log("Connected.")

    try:
        if not args.no_names:
            pull_msenames(db, args.start, end_date, common_only, shrcd, exchcd, args.chunksize, args.force)

        if not args.no_link:
            pull_linktable(db, args.start, end_date, common_only, shrcd, exchcd, args.chunksize, args.force)

        if not args.no_monthly:
            pull_table_by_year(
                db,
                table="crsp.msf",
                date_col="date",
                label="msf",
                start_date=args.start,
                end_date=end_date,
                common_only=common_only,
                shrcd=shrcd,
                exchcd=exchcd,
                chunksize=args.chunksize,
                chunk_years=args.chunk_years,
                force=args.force,
            )

        if args.daily:
            pull_table_by_year(
                db,
                table="crsp.dsf",
                date_col="date",
                label="dsf",
                start_date=args.start,
                end_date=end_date,
                common_only=common_only,
                shrcd=shrcd,
                exchcd=exchcd,
                chunksize=args.chunksize,
                chunk_years=args.chunk_years,
                force=args.force,
            )

        if not args.no_distributions:
            pull_table_by_year(
                db,
                table="crsp.msedist",
                date_col="exdt",
                label="msedist",
                start_date=args.start,
                end_date=end_date,
                common_only=common_only,
                shrcd=shrcd,
                exchcd=exchcd,
                chunksize=args.chunksize,
                chunk_years=args.chunk_years,
                force=args.force,
            )

        if not args.no_delistings:
            pull_table_by_year(
                db,
                table="crsp.msedelist",
                date_col="dlstdt",
                label="msedelist",
                start_date=args.start,
                end_date=end_date,
                common_only=common_only,
                shrcd=shrcd,
                exchcd=exchcd,
                chunksize=args.chunksize,
                chunk_years=args.chunk_years,
                force=args.force,
            )
    finally:
        db.close()
        log("WRDS connection closed.")


if __name__ == "__main__":
    main()
