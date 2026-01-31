#!/usr/bin/env python
"""
Build a RawDocument versioning stream from RawDocumentStore.

Outputs a compact table with version numbers and prior_document_id per source_id.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-docs-path",
        default="data/inputs_layer/raw_documents",
        help="Path to RawDocumentStore (directory or parquet).",
    )
    parser.add_argument(
        "--years",
        default=None,
        help="Comma-separated years to process (when raw-docs-path is a directory).",
    )
    parser.add_argument(
        "--out",
        default="data/inputs_layer/raw_document_versions.parquet",
        help="Output parquet path for version stream.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory", default="6GB")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-bad-files",
        action="store_true",
        help="Skip unreadable parquet parts (e.g., corrupt files).",
    )
    args = parser.parse_args()

    raw_docs_path = ROOT / args.raw_docs_path
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET memory_limit='{args.memory}'")

    def build_for_path(input_path: Path, out_file: Path) -> None:
        if out_file.exists() and not args.overwrite:
            print(f"Skipping output exists: {out_file}")
            return
        source_expr = f"'{input_path.as_posix()}'"
        if args.skip_bad_files and input_path.is_dir():
            files = sorted(input_path.glob("**/part_*.parquet"))
            ok_files: list[str] = []
            bad_files: list[str] = []
            for f in files:
                try:
                    pq.ParquetFile(f)
                    ok_files.append(f.as_posix())
                except Exception:
                    bad_files.append(f.as_posix())
            if bad_files:
                print(f"Skipping {len(bad_files)} unreadable parquet parts for {input_path}")
            if not ok_files:
                raise RuntimeError(f"No readable parquet parts found in {input_path}")
            quoted = ", ".join([f"'{p}'" for p in ok_files])
            source_expr = f"[{quoted}]"

        query = f"""
        WITH base AS (
            SELECT
                document_id,
                source_id,
                source_type,
                published_at,
                ingested_at,
                content_hash
            FROM read_parquet({source_expr}, union_by_name=True)
            WHERE document_id IS NOT NULL
              AND source_id IS NOT NULL
        ),
        ordered AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY source_type, source_id
                    ORDER BY published_at NULLS LAST, ingested_at NULLS LAST, content_hash NULLS LAST, document_id
                ) AS version_num,
                lag(document_id) OVER (
                    PARTITION BY source_type, source_id
                    ORDER BY published_at NULLS LAST, ingested_at NULLS LAST, content_hash NULLS LAST, document_id
                ) AS prior_document_id,
                lead(document_id) OVER (
                    PARTITION BY source_type, source_id
                    ORDER BY published_at NULLS LAST, ingested_at NULLS LAST, content_hash NULLS LAST, document_id
                ) AS next_document_id
            FROM base
        )
        SELECT
            document_id,
            source_id,
            source_type,
            published_at,
            ingested_at,
            content_hash,
            version_num,
            prior_document_id,
            CASE WHEN next_document_id IS NULL THEN TRUE ELSE FALSE END AS is_latest
        FROM ordered
        """
        out_file.parent.mkdir(parents=True, exist_ok=True)
        con.execute(f"COPY ({query}) TO '{out_file.as_posix()}' (FORMAT 'parquet');")
        print(f"Wrote RawDocumentVersion stream -> {out_file}")

    if raw_docs_path.is_dir() and args.years:
        years = [y.strip() for y in args.years.split(",") if y.strip()]
        out_dir = out_path if out_path.suffix != ".parquet" else out_path.parent / "raw_document_versions"
        for y in years:
            input_path = raw_docs_path / f"year={y}"
            out_file = out_dir / f"year={y}" / "part.parquet"
            build_for_path(input_path, out_file)
        return

    if out_path.exists() and not args.overwrite:
        print(f"Output exists: {out_path}")
        return

    build_for_path(raw_docs_path, out_path)


if __name__ == "__main__":
    main()
