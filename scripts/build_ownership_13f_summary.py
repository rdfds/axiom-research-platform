#!/usr/bin/env python
"""
Build issuer-level 13F ownership summary for CompanyState.

Input:
  - data/warehouse/warehouse_13f_holdings (partitioned parquet directory)
  - data/inputs_layer/entity_identifier.parquet (CUSIP -> entity_id map)

Output:
  - data/inputs_layer/ownership_13f_summary.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import duckdb


ROOT = Path(__file__).resolve().parents[1]


def _is_materialized(path: Path) -> bool:
    if not path.exists():
        return False
    st = path.stat()
    if st.st_size <= 0:
        return False
    # Keep this non-blocking; do not trigger iCloud fetch here.
    if hasattr(st, "st_blocks") and st.st_blocks == 0:
        return False
    return True


def _parse_years(arg: Optional[str]) -> Optional[List[int]]:
    if not arg:
        return None
    out: List[int] = []
    for x in arg.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return sorted(set(out))


def _iter_year_dirs(root: Path, years: Optional[List[int]]) -> Iterable[Path]:
    if years is None:
        for p in sorted(root.glob("year=*")):
            if p.is_dir():
                yield p
        return
    for y in years:
        p = root / f"year={y}"
        if p.is_dir():
            yield p


def _readable_year_files(year_dir: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(year_dir.glob("*.parquet")):
        name = p.name.lower()
        if name.startswith(".") or name.startswith("_"):
            continue
        if _is_materialized(p):
            out.append(p)
    return out


def _first_present(cols: set[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build issuer-level 13F ownership summary.")
    parser.add_argument("--holdings-path", default="data/warehouse/warehouse_13f_holdings")
    parser.add_argument("--entity-identifier-path", default="data/inputs_layer/entity_identifier.parquet")
    parser.add_argument("--out", default="data/inputs_layer/ownership_13f_summary.parquet")
    parser.add_argument("--years", default=None, help="Comma-separated years to process (default: all)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    holdings_root = ROOT / args.holdings_path
    entity_identifier_path = ROOT / args.entity_identifier_path
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.overwrite:
        print(f"Output exists: {out_path}")
        return
    if not holdings_root.exists():
        raise FileNotFoundError(f"Missing holdings path: {holdings_root}")
    if not entity_identifier_path.exists():
        raise FileNotFoundError(f"Missing entity_identifier parquet: {entity_identifier_path}")
    if not _is_materialized(entity_identifier_path):
        raise RuntimeError(f"entity_identifier is offloaded/unreadable: {entity_identifier_path}")

    years = _parse_years(args.years)
    tmp_dir = out_path.parent / f".tmp_{out_path.stem}"
    if tmp_dir.exists():
        for p in tmp_dir.glob("*.parquet"):
            p.unlink()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=2")
    con.execute(f"CREATE OR REPLACE TEMP VIEW ids AS SELECT * FROM read_parquet('{entity_identifier_path.as_posix()}')")
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW cusip_map AS
        SELECT
            CAST(entity_id AS VARCHAR) AS entity_id,
            upper(regexp_replace(CAST(identifier_value AS VARCHAR), '[^A-Za-z0-9]', '', 'g')) AS cusip_full,
            left(upper(regexp_replace(CAST(identifier_value AS VARCHAR), '[^A-Za-z0-9]', '', 'g')), 8) AS cusip8
        FROM ids
        WHERE lower(CAST(identifier_type AS VARCHAR)) = 'cusip'
        """
    )

    wrote = 0
    skipped = 0
    for year_dir in _iter_year_dirs(holdings_root, years):
        year = year_dir.name.split("=")[-1]
        files = _readable_year_files(year_dir)
        if not files:
            skipped += 1
            print(f"[skip] year={year} no readable parquet files")
            continue
        schema_df = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{files[0].as_posix()}')").df()
        cols = set(schema_df["column_name"].astype(str))

        filer_col = _first_present(cols, ["company_id", "cik", "filer_cik"])
        cusip_col = _first_present(cols, ["holding_cusip", "security_id", "cusip"])
        report_col = _first_present(cols, ["report_date", "event_time", "rdate"])
        filing_col = _first_present(cols, ["filing_date", "available_time", "fdate"])
        shares_col = _first_present(cols, ["shares", "sshprnamt"])
        value_usd_col = _first_present(cols, ["value_usd"])
        value_k_col = _first_present(cols, ["value_k", "value"])

        if filer_col is None or cusip_col is None or report_col is None or filing_col is None:
            skipped += 1
            print(f"[skip] year={year} missing required columns in holdings schema")
            continue

        shares_expr = f"try_cast({shares_col} AS DOUBLE)" if shares_col else "CAST(NULL AS DOUBLE)"
        if value_usd_col:
            value_expr = f"try_cast({value_usd_col} AS DOUBLE)"
        elif value_k_col:
            value_expr = f"try_cast({value_k_col} AS DOUBLE) * 1000.0"
        else:
            value_expr = "CAST(NULL AS DOUBLE)"

        file_list = ", ".join("'" + f.as_posix().replace("'", "''") + "'" for f in files)
        out_part = tmp_dir / f"year={year}.parquet"
        print(f"[year {year}] files={len(files)} aggregating...")
        query = f"""
        COPY (
            WITH base AS (
                SELECT
                    CAST({filer_col} AS VARCHAR) AS filer_id,
                    upper(regexp_replace(COALESCE(CAST({cusip_col} AS VARCHAR), ''), '[^A-Za-z0-9]', '', 'g')) AS cusip_full,
                    left(upper(regexp_replace(COALESCE(CAST({cusip_col} AS VARCHAR), ''), '[^A-Za-z0-9]', '', 'g')), 8) AS cusip8,
                    try_cast({report_col} AS TIMESTAMP) AS report_date,
                    try_cast({filing_col} AS TIMESTAMP) AS filing_date,
                    {shares_expr} AS shares,
                    {value_expr} AS value_usd
                FROM read_parquet([{file_list}], union_by_name=True)
            ),
            mapped AS (
                SELECT
                    COALESCE(mf.entity_id, m8.entity_id) AS company_id,
                    b.filer_id,
                    b.report_date,
                    b.filing_date,
                    b.shares,
                    b.value_usd
                FROM base b
                LEFT JOIN cusip_map mf
                  ON b.cusip_full = mf.cusip_full
                LEFT JOIN cusip_map m8
                  ON b.cusip8 = m8.cusip8
                WHERE b.filer_id IS NOT NULL
                  AND b.report_date IS NOT NULL
                  AND b.filing_date IS NOT NULL
                  AND b.filing_date >= b.report_date
                  AND COALESCE(mf.entity_id, m8.entity_id) IS NOT NULL
            ),
            filer_agg AS (
                SELECT
                    company_id,
                    report_date,
                    filing_date,
                    filer_id,
                    sum(COALESCE(shares, 0.0)) AS shares,
                    sum(COALESCE(value_usd, 0.0)) AS value_usd
                FROM mapped
                GROUP BY 1,2,3,4
            ),
            ranked AS (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY company_id, report_date, filing_date
                        ORDER BY shares DESC NULLS LAST, filer_id ASC
                    ) AS rn
                FROM filer_agg
            )
            SELECT
                company_id,
                report_date,
                filing_date,
                sum(shares) AS total_13f_shares,
                sum(value_usd) AS total_13f_value_usd,
                sum(CASE WHEN rn <= 5 THEN shares ELSE 0.0 END) AS top5_13f_shares,
                count(*) AS holder_count,
                filing_date AS published_at,
                filing_date AS ingested_at,
                report_date AS effective_at,
                'wrds_13f' AS source_type,
                md5(company_id || '|' || CAST(report_date AS VARCHAR) || '|' || CAST(filing_date AS VARCHAR)) AS artifact_id
            FROM ranked
            GROUP BY 1,2,3
        ) TO '{out_part.as_posix()}' (FORMAT 'parquet');
        """
        con.execute(query)
        wrote += 1
        print(f"[year {year}] wrote {out_part}")

    if wrote == 0:
        raise RuntimeError("No yearly ownership summaries were produced. Materialize 13F holdings files first.")

    con.execute(
        f"""
        COPY (
            SELECT * FROM read_parquet('{(tmp_dir / 'year=*.parquet').as_posix()}', union_by_name=True)
        ) TO '{out_path.as_posix()}' (FORMAT 'parquet');
        """
    )
    total_rows = con.execute(f"SELECT count(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
    print(f"Wrote ownership summary -> {out_path} rows={total_rows} years_written={wrote} years_skipped={skipped}")


if __name__ == "__main__":
    main()
