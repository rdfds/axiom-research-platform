#!/usr/bin/env python
"""
Backfill credit spread change + rating migration outcome columns onto action_outcomes.parquet
without rebuilding the entire outcomes table.

Inputs (defaults):
  data/curated/action_outcomes.parquet
  data/curated/trace_btds_daily_fisduniverse.parquet
  data/curated/bond_issuances_fisd.parquet
  data/inputs_layer/issuer_rating_history.parquet

Output:
  data/curated/action_outcomes_with_credit_ratings.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


def _parse_horizons(raw: str) -> List[int]:
    out: List[int] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        out = [1, 6, 12, 24]
    return sorted(set(out))


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _rating_score_case(expr: str) -> str:
    # Higher score = better rating quality.
    return f"""
    CASE {expr}
      -- S&P / Fitch style
      WHEN 'AAA' THEN 22
      WHEN 'AA+' THEN 21 WHEN 'AA' THEN 20 WHEN 'AA-' THEN 19
      WHEN 'A+' THEN 18 WHEN 'A' THEN 17 WHEN 'A-' THEN 16
      WHEN 'BBB+' THEN 15 WHEN 'BBB' THEN 14 WHEN 'BBB-' THEN 13
      WHEN 'BB+' THEN 12 WHEN 'BB' THEN 11 WHEN 'BB-' THEN 10
      WHEN 'B+' THEN 9 WHEN 'B' THEN 8 WHEN 'B-' THEN 7
      WHEN 'CCC+' THEN 6 WHEN 'CCC' THEN 5 WHEN 'CCC-' THEN 4
      WHEN 'CC' THEN 3 WHEN 'C' THEN 2 WHEN 'D' THEN 1
      -- Moody's style (normalized to upper-case)
      WHEN 'AAA' THEN 22
      WHEN 'AA1' THEN 21 WHEN 'AA2' THEN 20 WHEN 'AA3' THEN 19
      WHEN 'A1' THEN 18 WHEN 'A2' THEN 17 WHEN 'A3' THEN 16
      WHEN 'BAA1' THEN 15 WHEN 'BAA2' THEN 14 WHEN 'BAA3' THEN 13
      WHEN 'BA1' THEN 12 WHEN 'BA2' THEN 11 WHEN 'BA3' THEN 10
      WHEN 'B1' THEN 9 WHEN 'B2' THEN 8 WHEN 'B3' THEN 7
      WHEN 'CAA1' THEN 6 WHEN 'CAA2' THEN 5 WHEN 'CAA3' THEN 4
      WHEN 'CA' THEN 3
      ELSE NULL
    END
    """


def _resolve_col(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {str(c).lower(): str(c) for c in columns}
    for cand in candidates:
        hit = lookup.get(str(cand).lower())
        if hit:
            return hit
    return None


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _run(
    *,
    in_path: Path,
    out_path: Path,
    trace_daily_path: Path,
    fisd_issues_path: Path,
    issuer_ratings_path: Path,
    gvkey_to_cik_path: Path,
    horizons: Iterable[int],
    duckdb_memory: str | None,
    duckdb_threads: int | None,
    duckdb_temp_dir: Path | None,
    duckdb_max_temp_dir_size: str | None,
    chunk_size: int | None,
) -> None:
    if not _exists(in_path):
        raise FileNotFoundError(f"Missing action outcomes: {in_path}")

    credit_enabled = _exists(trace_daily_path) and _exists(fisd_issues_path)
    rating_enabled = _exists(issuer_ratings_path)
    if not credit_enabled and not rating_enabled:
        raise RuntimeError(
            "Need at least one data source: TRACE+FISD for credit spread, or issuer ratings for migration."
        )

    con = duckdb.connect()
    if duckdb_memory:
        con.execute(f"SET memory_limit='{duckdb_memory}'")
    if duckdb_threads:
        con.execute(f"SET threads={int(duckdb_threads)}")
    con.execute("SET preserve_insertion_order=false")
    if duckdb_temp_dir:
        con.execute(f"PRAGMA temp_directory='{duckdb_temp_dir.as_posix()}'")
    if duckdb_max_temp_dir_size:
        con.execute(f"PRAGMA max_temp_directory_size='{duckdb_max_temp_dir_size}'")

    in_cols = [
        r[0] for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{in_path.as_posix()}')"
        ).fetchall()
    ]
    ratings_cols: List[str] = []
    if rating_enabled:
        ratings_cols = [
            r[0] for r in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{issuer_ratings_path.as_posix()}')"
            ).fetchall()
        ]
    out_metric_cols = []
    for h in horizons:
        out_metric_cols.extend([f"credit_spread_change_{h}m", f"rating_migration_{h}m"])
    base_cols = [c for c in in_cols if c not in out_metric_cols]
    base_projection = ",\n        ".join([f"a.{_quote_ident(c)}" for c in base_cols])

    horizon_yield_ctes: List[str] = []
    horizon_rating_ctes: List[str] = []
    horizon_selects: List[str] = []
    for h in horizons:
        if credit_enabled:
            horizon_yield_ctes.append(
                f"""
    yield_h{h} AS (
        SELECT a._row_id, iy.issuer_yield_pct
        FROM actions a
        LEFT JOIN issuer_daily iy
          ON iy.company_id_norm = a._company_id_norm
         AND iy.trade_date <= a._action_date_d + INTERVAL '{h} months'
        QUALIFY row_number() OVER (PARTITION BY a._row_id ORDER BY iy.trade_date DESC) = 1
    )
                """
            )
            horizon_selects.append(
                f"""
        CASE
          WHEN yb.issuer_yield_pct IS NOT NULL AND yield_h{h}.issuer_yield_pct IS NOT NULL
          THEN (yield_h{h}.issuer_yield_pct - yb.issuer_yield_pct) * 100.0
          ELSE NULL
        END AS credit_spread_change_{h}m
                """
            )
        else:
            horizon_selects.append(f"NULL::DOUBLE AS credit_spread_change_{h}m")

        if rating_enabled:
            horizon_rating_ctes.append(
                f"""
    rating_h{h} AS (
        SELECT a._row_id, r.rating_score
        FROM actions a
        LEFT JOIN ratings_norm r
          ON r.company_id_norm = a._company_id_norm
         AND r.rating_date <= a._action_date_d + INTERVAL '{h} months'
        QUALIFY row_number() OVER (PARTITION BY a._row_id ORDER BY r.rating_date DESC) = 1
    )
                """
            )
            horizon_selects.append(
                f"""
        CASE
          WHEN rb.rating_score IS NOT NULL AND rating_h{h}.rating_score IS NOT NULL
          THEN CAST(rating_h{h}.rating_score - rb.rating_score AS DOUBLE)
          ELSE NULL
        END AS rating_migration_{h}m
                """
            )
        else:
            horizon_selects.append(f"NULL::DOUBLE AS rating_migration_{h}m")

    credit_ctes = ""
    credit_joins = ""
    if credit_enabled:
        credit_ctes = f"""
    action_ids AS (
        SELECT DISTINCT _company_id_norm AS company_id_norm
        FROM actions
        WHERE _company_id_norm IS NOT NULL
    ),
    issuer_cusips AS (
        SELECT DISTINCT
            lpad(regexp_extract(CAST(gvkey AS VARCHAR), '[0-9]+', 0), 6, '0') AS company_id_norm,
            substr(upper(trim(coalesce(COMPLETE_CUSIP, ISSUE_CUSIP))), 1, 9) AS cusip9
        FROM read_parquet('{fisd_issues_path.as_posix()}')
        WHERE coalesce(COMPLETE_CUSIP, ISSUE_CUSIP) IS NOT NULL
          AND gvkey IS NOT NULL
    ),
    issuer_cusips_filtered AS (
        SELECT ic.company_id_norm, ic.cusip9
        FROM issuer_cusips ic
        JOIN action_ids aid ON aid.company_id_norm = ic.company_id_norm
    ),
    issuer_daily AS (
        SELECT
            ic.company_id_norm,
            CAST(t.trade_date AS DATE) AS trade_date,
            avg(t.yld_pt_avg) AS issuer_yield_pct
        FROM read_parquet('{trace_daily_path.as_posix()}') t
        JOIN issuer_cusips_filtered ic
          ON ic.cusip9 = upper(trim(t.cusip_id))
        WHERE CAST(t.trade_date AS DATE) IS NOT NULL
          AND t.yld_pt_avg IS NOT NULL
        GROUP BY 1, 2
    ),
    yb AS (
        SELECT a._row_id, iy.issuer_yield_pct
        FROM actions a
        LEFT JOIN issuer_daily iy
          ON iy.company_id_norm = a._company_id_norm
         AND iy.trade_date <= a._action_date_d
        QUALIFY row_number() OVER (PARTITION BY a._row_id ORDER BY iy.trade_date DESC) = 1
    ),
    {",".join(horizon_yield_ctes)}
        """
        credit_joins = "LEFT JOIN yb ON yb._row_id = a._row_id\n"
        for h in horizons:
            credit_joins += f"    LEFT JOIN yield_h{h} ON yield_h{h}._row_id = a._row_id\n"

    rating_ctes = ""
    rating_joins = ""
    if rating_enabled:
        r_gvkey = _resolve_col(ratings_cols, ["gvkey"])
        r_company_id = _resolve_col(ratings_cols, ["company_id"])
        r_date = _resolve_col(ratings_cols, ["rating_date", "RATING_DATE"])
        r_curr = _resolve_col(ratings_cols, ["current_rating_symbol"])
        r_sym = _resolve_col(ratings_cols, ["rating_symbol", "RATING"])
        if not r_date or not (r_curr or r_sym) or not (r_gvkey or r_company_id):
            rating_enabled = False
            print(
                f"[warn] ratings disabled: unsupported schema in {issuer_ratings_path}"
            )
        else:
            rating_expr = (
                f"coalesce({_quote_ident(r_curr)}, {_quote_ident(r_sym)})"
                if r_curr and r_sym
                else _quote_ident(r_curr or r_sym)
            )
            if r_gvkey:
                rating_source_cte = f"""
    ratings_norm AS (
        SELECT
            lpad(regexp_extract(CAST({_quote_ident(r_gvkey)} AS VARCHAR), '[0-9]+', 0), 6, '0') AS company_id_norm,
            CAST({_quote_ident(r_date)} AS DATE) AS rating_date,
            {_rating_score_case(f"upper(replace(trim(cast({rating_expr} AS VARCHAR)), ' ', ''))")} AS rating_score
        FROM read_parquet('{issuer_ratings_path.as_posix()}')
        WHERE CAST({_quote_ident(r_date)} AS DATE) IS NOT NULL
    )
                """
            else:
                cik_map_cte = ""
                map_join = ""
                if _exists(gvkey_to_cik_path):
                    cik_map_cte = f""",
    cik_map AS (
        SELECT
            lpad(regexp_extract(CAST(gvkey AS VARCHAR), '[0-9]+', 0), 6, '0') AS gvkey_norm,
            lpad(regexp_extract(CAST(cik AS VARCHAR), '[0-9]+', 0), 10, '0') AS cik10
        FROM read_csv_auto('{gvkey_to_cik_path.as_posix()}', all_varchar=true)
        WHERE gvkey IS NOT NULL AND cik IS NOT NULL
    )"""
                    map_join = "LEFT JOIN cik_map cm ON lpad(regexp_extract(rr.company_id_raw, '[0-9]+', 0), 10, '0') = cm.cik10"
                else:
                    print(
                        f"[warn] gvkey->cik map not found at {gvkey_to_cik_path}; "
                        "rating join will assume company_id already matches gvkey."
                    )

                rating_source_cte = f"""
    ratings_raw AS (
        SELECT
            CAST({_quote_ident(r_company_id)} AS VARCHAR) AS company_id_raw,
            CAST({_quote_ident(r_date)} AS DATE) AS rating_date,
            {_rating_score_case(f"upper(replace(trim(cast({rating_expr} AS VARCHAR)), ' ', ''))")} AS rating_score
        FROM read_parquet('{issuer_ratings_path.as_posix()}')
        WHERE CAST({_quote_ident(r_date)} AS DATE) IS NOT NULL
    ){cik_map_cte},
    ratings_norm AS (
        SELECT
            COALESCE(
                CASE
                    WHEN length(regexp_extract(rr.company_id_raw, '[0-9]+', 0)) <= 6
                    THEN lpad(regexp_extract(rr.company_id_raw, '[0-9]+', 0), 6, '0')
                    ELSE NULL
                END,
                cm.gvkey_norm
            ) AS company_id_norm,
            rr.rating_date,
            rr.rating_score
        FROM ratings_raw rr
        {map_join}
    )
                """

        rating_ctes = f"""
    {rating_source_cte.strip()},
    rb AS (
        SELECT a._row_id, r.rating_score
        FROM actions a
        LEFT JOIN ratings_norm r
          ON r.company_id_norm = a._company_id_norm
         AND r.rating_date <= a._action_date_d
         AND r.rating_score IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY a._row_id ORDER BY r.rating_date DESC) = 1
    ),
    {",".join(horizon_rating_ctes)}
        """
        rating_joins = "LEFT JOIN rb ON rb._row_id = a._row_id\n"
        for h in horizons:
            rating_joins += f"    LEFT JOIN rating_h{h} ON rating_h{h}._row_id = a._row_id\n"

    actions_cte = f"""
    actions AS (
        SELECT
            row_number() OVER () AS _row_id,
            *,
            lpad(regexp_extract(CAST(company_id AS VARCHAR), '[0-9]+', 0), 6, '0') AS _company_id_norm,
            CAST(action_date AS DATE) AS _action_date_d
        FROM read_parquet('{in_path.as_posix()}')
    )
    """.strip()
    cte_parts = [actions_cte]
    if credit_ctes:
        cte_parts.append(credit_ctes.strip())
    if rating_ctes:
        cte_parts.append(rating_ctes.strip())
    ctes = ",\n".join(cte_parts)

    select_metrics = ",\n        ".join(horizon_selects)
    select_query_core = f"""
    WITH {ctes}
    SELECT
        a._row_id AS __row_id,
        {base_projection},
        {select_metrics}
    FROM actions a
    {credit_joins}{rating_joins}
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    if chunk_size and int(chunk_size) > 0:
        total_rows = int(
            con.execute(f"SELECT count(*) FROM read_parquet('{in_path.as_posix()}')").fetchone()[0]
        )
        writer: pq.ParquetWriter | None = None
        try:
            for start in range(1, total_rows + 1, int(chunk_size)):
                end = min(total_rows, start + int(chunk_size) - 1)
                chunk_query = f"""
                SELECT * FROM (
                {select_query_core}
                )
                WHERE __row_id BETWEEN {start} AND {end}
                """
                df_chunk = con.execute(chunk_query).fetch_df()
                if df_chunk.empty:
                    continue
                if "__row_id" in df_chunk.columns:
                    df_chunk = df_chunk.drop(columns=["__row_id"])
                table = pa.Table.from_pandas(df_chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path.as_posix(), table.schema, compression="zstd")
                else:
                    table = pa.Table.from_pandas(df_chunk, schema=writer.schema, preserve_index=False)
                writer.write_table(table)
                print(
                    f"[chunk] wrote rows {start:,}-{end:,} chunk_rows={len(df_chunk):,}",
                    flush=True,
                )
        finally:
            if writer is not None:
                writer.close()
    else:
        select_query = f"""
        SELECT * EXCLUDE (__row_id) FROM (
        {select_query_core}
        )
        """
        con.execute(f"COPY ({select_query}) TO '{out_path.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd')")

    total = con.execute(f"SELECT count(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
    print(f"Wrote outcomes -> {out_path} rows={total}")
    for h in horizons:
        c_col = f"credit_spread_change_{h}m"
        r_col = f"rating_migration_{h}m"
        c_n = con.execute(
            f"SELECT count(*) FROM read_parquet('{out_path.as_posix()}') WHERE {c_col} IS NOT NULL"
        ).fetchone()[0]
        r_n = con.execute(
            f"SELECT count(*) FROM read_parquet('{out_path.as_posix()}') WHERE {r_col} IS NOT NULL"
        ).fetchone()[0]
        print(f"[coverage] {c_col}={c_n} {r_col}={r_n}")
    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill action_outcomes with credit spread and rating migration columns.")
    parser.add_argument("--in-path", default="data/curated/action_outcomes.parquet")
    parser.add_argument(
        "--out-path",
        default="data/curated/action_outcomes_with_credit_ratings.parquet",
        help="Output parquet path.",
    )
    parser.add_argument("--trace-daily-path", default="data/curated/trace_btds_daily_fisduniverse.parquet")
    parser.add_argument("--fisd-issues-path", default="data/curated/bond_issuances_fisd.parquet")
    parser.add_argument("--issuer-ratings-path", default="data/inputs_layer/issuer_rating_history.parquet")
    parser.add_argument("--gvkey-to-cik-path", default="data/wrds/compustat/cik_gvkey.csv.gz")
    parser.add_argument("--horizons", default="1,6,12,24")
    parser.add_argument("--duckdb-memory", default="3GB")
    parser.add_argument("--duckdb-threads", type=int, default=1)
    parser.add_argument("--duckdb-temp-dir", default="/tmp/axiom_duckdb")
    parser.add_argument("--duckdb-max-temp-dir-size", default="20GiB")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Rows per chunk for low-memory backfill. Set 0 to use one-shot COPY.",
    )
    args = parser.parse_args()

    in_path = (ROOT / args.in_path).resolve()
    out_path = (ROOT / args.out_path).resolve()
    trace_daily_path = (ROOT / args.trace_daily_path).resolve()
    fisd_issues_path = (ROOT / args.fisd_issues_path).resolve()
    issuer_ratings_path = (ROOT / args.issuer_ratings_path).resolve()
    gvkey_to_cik_path = (ROOT / args.gvkey_to_cik_path).resolve()
    temp_dir = (ROOT / args.duckdb_temp_dir).resolve() if args.duckdb_temp_dir else None
    if temp_dir:
        temp_dir.mkdir(parents=True, exist_ok=True)

    _run(
        in_path=in_path,
        out_path=out_path,
        trace_daily_path=trace_daily_path,
        fisd_issues_path=fisd_issues_path,
        issuer_ratings_path=issuer_ratings_path,
        gvkey_to_cik_path=gvkey_to_cik_path,
        horizons=_parse_horizons(args.horizons),
        duckdb_memory=args.duckdb_memory,
        duckdb_threads=args.duckdb_threads,
        duckdb_temp_dir=temp_dir,
        duckdb_max_temp_dir_size=args.duckdb_max_temp_dir_size,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
