#!/usr/bin/env python
"""
Build contradiction / expiration metadata for ExtractedFactRegistry.

Outputs a new dataset with:
  - valid_from (fact_time)
  - valid_to (next fact_time in same group)
  - is_superseded
  - superseded_by_fact_id
  - contradiction_group_id (if multiple distinct values in group)
  - value_norm (normalized fact_value for grouping)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-path",
        default="data/inputs_layer/extracted_fact_registry_enriched",
        help="Input ExtractedFactRegistry (directory or parquet).",
    )
    parser.add_argument(
        "--out",
        default="data/inputs_layer/extracted_fact_registry_validity",
        help="Output dataset (directory or parquet).",
    )
    parser.add_argument(
        "--years",
        default=None,
        help="Comma-separated years to process (when in-path is a directory).",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory", default="6GB")
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="DuckDB temp directory (use a path on a disk with free space).",
    )
    parser.add_argument(
        "--max-temp",
        default=None,
        help="DuckDB max temp directory size (e.g., '5GiB').",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip outputs that already exist (even if overwrite is set).",
    )
    parser.add_argument(
        "--latest-years",
        type=int,
        default=None,
        help="Process only the most recent N years with data.",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=256,
        help="Skip parquet parts smaller than this size (bytes).",
    )
    parser.add_argument(
        "--skip-bad-files",
        action="store_true",
        help="Skip unreadable parquet parts (e.g., corrupt files).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List years and file counts, then exit.",
    )
    args = parser.parse_args()

    in_path = ROOT / args.in_path
    out_path = ROOT / args.out
    if out_path.suffix == ".parquet":
        out_is_dir = False
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_is_dir = True
        out_path.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET memory_limit='{args.memory}'")
    con.execute("SET preserve_insertion_order=false")
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{temp_dir.as_posix()}'")
    if args.max_temp:
        con.execute(f"PRAGMA max_temp_directory_size='{args.max_temp}'")

    def _collect_parquet_files(input_path: Path) -> list[Path]:
        if input_path.is_file():
            return [input_path]
        files = sorted(input_path.glob("*.parquet"))
        keep: list[Path] = []
        for f in files:
            name = f.name.lower()
            if "tmp_part" in name or name.startswith("tmp_"):
                continue
            try:
                if f.stat().st_size < args.min_bytes:
                    continue
            except OSError:
                continue
            keep.append(f)
        return keep

    def build_for_path(input_path: Path, output_file: Path) -> None:
        if not input_path.exists():
            print(f"[skip] input missing: {input_path}")
            return
        files = _collect_parquet_files(input_path)
        if not files:
            print(f"[skip] no readable parquet parts in: {input_path}")
            return
        if output_file.exists() and (args.resume or not args.overwrite):
            print(f"Skipping output exists: {output_file}")
            return
        t0 = time.time()
        source_sql = f"SELECT * FROM read_parquet('{input_path.as_posix()}', union_by_name=True)"
        tz_files: list[str] = []
        non_tz_files: list[str] = []
        bad_files: list[str] = []
        for f in files:
            try:
                pf = pq.ParquetFile(f)
                has_tz = False
                for col in ("published_at", "effective_at", "ingested_at"):
                    try:
                        field = pf.schema_arrow.field(col)
                    except Exception:
                        continue
                    if str(field.type).startswith("timestamp") and getattr(field.type, "tz", None):
                        has_tz = True
                        break
                if has_tz:
                    tz_files.append(f.as_posix())
                else:
                    non_tz_files.append(f.as_posix())
            except Exception:
                bad_files.append(f.as_posix())
        if args.skip_bad_files and bad_files:
            print(f"Skipping {len(bad_files)} unreadable parquet parts for {input_path}")
        if args.skip_bad_files:
            if not tz_files and not non_tz_files:
                print(f"[skip] no readable parquet parts in: {input_path}")
                return

        if tz_files:
            quoted_tz = ", ".join([("'" + p + "'") for p in tz_files])
            tz_expr = f"[{quoted_tz}]"
        else:
            tz_expr = None
        if non_tz_files:
            quoted_ntz = ", ".join([("'" + p + "'") for p in non_tz_files])
            ntz_expr = f"[{quoted_ntz}]"
        else:
            ntz_expr = None

        reads = []
        if tz_expr:
            reads.append(f"""
                SELECT
                    fact_id, document_id, entity_id, fact_type, fact_value, unit, context,
                    confidence_score, citation_span, paragraph_index, speaker, transcript_timestamp, is_qa,
                    source_id, source_type,
                    CAST(published_at AS TIMESTAMP) AS published_at,
                    CAST(effective_at AS TIMESTAMP) AS effective_at,
                    CAST(ingested_at AS TIMESTAMP) AS ingested_at,
                    raw_pointer
                FROM read_parquet({tz_expr}, union_by_name=True)
            """)
        if ntz_expr:
            reads.append(f"""
                SELECT
                    fact_id, document_id, entity_id, fact_type, fact_value, unit, context,
                    confidence_score, citation_span, paragraph_index, speaker, transcript_timestamp, is_qa,
                    source_id, source_type,
                    CAST(published_at AS TIMESTAMP) AS published_at,
                    CAST(effective_at AS TIMESTAMP) AS effective_at,
                    CAST(ingested_at AS TIMESTAMP) AS ingested_at,
                    raw_pointer
                FROM read_parquet({ntz_expr}, union_by_name=True)
            """)
        if reads:
            source_sql = " UNION ALL ".join(reads)

        query = f"""
        WITH base AS (
            SELECT
                *,
                coalesce(
                    try_cast(published_at AS TIMESTAMPTZ),
                    try_cast(effective_at AS TIMESTAMPTZ),
                    try_cast(ingested_at AS TIMESTAMPTZ)
                ) AS fact_time,
                lower(trim(cast(fact_value AS VARCHAR))) AS value_norm,
                coalesce(cast(context AS VARCHAR), '') AS context_norm
            FROM ({source_sql}) AS src
            WHERE fact_id IS NOT NULL
              AND entity_id IS NOT NULL
              AND fact_type IS NOT NULL
        ),
        ordered AS (
            SELECT
                *,
                lead(fact_time) OVER (
                    PARTITION BY entity_id, fact_type, context_norm
                    ORDER BY fact_time NULLS LAST, fact_id
                ) AS next_time,
                lead(fact_id) OVER (
                    PARTITION BY entity_id, fact_type, context_norm
                    ORDER BY fact_time NULLS LAST, fact_id
                ) AS next_fact_id
            FROM base
        ),
        groups AS (
            SELECT
                entity_id,
                fact_type,
                context_norm,
                CASE WHEN count(distinct value_norm) > 1 THEN 1 ELSE 0 END AS has_contradiction
            FROM base
            GROUP BY 1,2,3
        )
        SELECT
            o.*,
            o.fact_time AS valid_from,
            o.next_time AS valid_to,
            o.next_fact_id AS superseded_by_fact_id,
            CASE WHEN o.next_fact_id IS NULL THEN FALSE ELSE TRUE END AS is_superseded,
            CASE WHEN g.has_contradiction = 1
                 THEN md5(o.entity_id || '|' || o.fact_type || '|' || o.context_norm)
                 ELSE NULL
            END AS contradiction_group_id
        FROM ordered o
        LEFT JOIN groups g
          ON o.entity_id = g.entity_id
         AND o.fact_type = g.fact_type
         AND o.context_norm = g.context_norm
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        con.execute(f"COPY ({query}) TO '{output_file.as_posix()}' (FORMAT 'parquet');")
        dt = time.time() - t0
        print(f"Wrote fact validity -> {output_file} ({dt:.1f}s)")

    if in_path.is_dir():
        year_dirs = []
        for p in sorted(in_path.glob("year=*")):
            try:
                y = int(p.name.split("=")[1])
            except Exception:
                continue
            year_dirs.append((y, p))
        if args.years:
            years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
            year_dirs = [item for item in year_dirs if item[0] in years]
        if args.latest_years:
            year_dirs = sorted(year_dirs)[-int(args.latest_years) :]
        if args.dry_run:
            for y, p in year_dirs:
                files = _collect_parquet_files(p)
                print(f"{y}: files={len(files)} path={p}")
            return
        for y, p in year_dirs:
            input_path = p
            output_file = out_path / f"year={y}" / "part.parquet" if out_is_dir else out_path
            files = _collect_parquet_files(input_path)
            print(f"[year={y}] starting | files={len(files)} | out={output_file}")
            build_for_path(input_path, output_file)
        return

    if out_path.exists() and not args.overwrite:
        print(f"Output exists: {out_path}")
        return

    build_for_path(in_path, out_path)


if __name__ == "__main__":
    main()
