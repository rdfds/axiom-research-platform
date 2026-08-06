#!/usr/bin/env python
"""
Build a year-level chunk lookup table filtered to only the chunk_ids
referenced by facts for that year. This makes enrichment much faster.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import time

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True, help="Year to process (e.g., 2026).")
    parser.add_argument("--facts-root", default="data/inputs_layer/extracted_fact_registry")
    parser.add_argument("--chunks-root", default="data/warehouse/warehouse_doc_chunks")
    parser.add_argument("--out-root", default="data/inputs_layer/chunk_lookup")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory", default="6GB")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="If set, process chunk files in batches of this size and write part_*.parquet files.",
    )
    parser.add_argument(
        "--skip-bad-files",
        action="store_true",
        help="Skip unreadable chunk parquet files.",
    )
    args = parser.parse_args()

    year = args.year.strip()
    facts_root = ROOT / args.facts_root / f"year={year}"
    chunks_root = ROOT / args.chunks_root / f"year={year}"
    out_root = ROOT / args.out_root / f"year={year}"
    out_root.mkdir(parents=True, exist_ok=True)
    out_file = out_root / "part.parquet"

    if out_file.exists() and not args.overwrite:
        print(f"Skipping year={year}, output exists: {out_file}")
        return

    con = duckdb.connect()
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET memory_limit='{args.memory}'")
    con.execute("SET preserve_insertion_order=false")

    facts_glob = (facts_root / "*.parquet").as_posix()
    chunks_glob = (chunks_root / "*.parquet").as_posix()
    fact_files = sorted(facts_root.glob("*.parquet"))
    chunk_files = sorted(chunks_root.glob("*.parquet"))
    print(f"[chunk_lookup] year={year} facts_files={len(fact_files)} chunk_files={len(chunk_files)}")

    # Precompute chunk_id set once (persist to disk for resume)
    print("[chunk_lookup] building fact_chunk_ids...")
    start_ids = time.time()
    fact_ids_dir = out_root / "fact_chunk_ids"
    fact_ids_file = fact_ids_dir / "part.parquet"
    if fact_ids_file.exists() and not args.overwrite:
        print(f"[chunk_lookup] reusing fact_chunk_ids from {fact_ids_file}")
    else:
        fact_ids_dir.mkdir(parents=True, exist_ok=True)
        facts_source = f"read_parquet('{facts_glob}', union_by_name=True)"
        ok_facts = []
        if args.skip_bad_files and fact_files:
            for f in fact_files:
                try:
                    pq.ParquetFile(f)
                    ok_facts.append(f.as_posix())
                except Exception:
                    continue
            print(f"[chunk_lookup] readable fact_files={len(ok_facts)}/{len(fact_files)}")

        if ok_facts:
            fact_list = ", ".join([("'" + p + "'") for p in ok_facts])
            facts_source = f"read_parquet([{fact_list}], union_by_name=True)"
            con.execute(
                f"""
                COPY (
                    SELECT DISTINCT citation_span AS chunk_id
                    FROM {facts_source}
                    WHERE citation_span IS NOT NULL
                ) TO '{fact_ids_file.as_posix()}' (FORMAT 'parquet');
                """
            )
        else:
            # Fallback: build chunk ids file-by-file to tolerate flaky files
            con.execute("CREATE TEMP TABLE fact_chunk_ids_raw(chunk_id VARCHAR)")
            for idx, f in enumerate(fact_files):
                if idx % 50 == 0:
                    print(f"[chunk_lookup] scanning facts {idx}/{len(fact_files)}")
                try:
                    con.execute(
                        f"""
                        INSERT INTO fact_chunk_ids_raw
                        SELECT citation_span AS chunk_id
                        FROM read_parquet('{f.as_posix()}', union_by_name=True)
                        WHERE citation_span IS NOT NULL
                        """
                    )
                except Exception:
                    continue
            con.execute(
                f"""
                COPY (
                    SELECT DISTINCT chunk_id FROM fact_chunk_ids_raw
                ) TO '{fact_ids_file.as_posix()}' (FORMAT 'parquet');
                """
            )
    # load persisted chunk ids into temp table
    con.execute(
        f"""
        CREATE TEMP TABLE fact_chunk_ids AS
        SELECT * FROM read_parquet('{fact_ids_file.as_posix()}')
        """
    )
    count_ids = con.execute("SELECT count(*) FROM fact_chunk_ids").fetchone()[0]
    print(f"[chunk_lookup] fact_chunk_ids={count_ids} built in {time.time() - start_ids:.1f}s")

    def _write_batch(file_list: list[str], out_path: Path) -> None:
        file_list_sql = ", ".join([("'" + p + "'") for p in file_list])
        query = f"""
        WITH chunks AS (
            SELECT
                chunk_id,
                CAST(document_id AS VARCHAR) AS document_id,
                CAST(chunk_index AS BIGINT) AS chunk_index,
                CAST(speaker AS VARCHAR) AS speaker,
                CAST(section_type AS VARCHAR) AS section_type
            FROM read_parquet([{file_list_sql}], union_by_name=True)
            WHERE chunk_id IN (SELECT chunk_id FROM fact_chunk_ids)
        )
        SELECT * FROM chunks
        """
        con.execute(f"COPY ({query}) TO '{out_path.as_posix()}' (FORMAT 'parquet');")

    # Optionally skip unreadable chunk files
    if args.skip_bad_files and chunk_files:
        ok_files = []
        for f in chunk_files:
            try:
                pq.ParquetFile(f)
                ok_files.append(f)
            except Exception:
                continue
        print(f"[chunk_lookup] readable chunk_files={len(ok_files)}")
        chunk_files = ok_files

    start = time.time()
    if args.batch_size and chunk_files:
        # write multiple parts for progress visibility
        for i in range(0, len(chunk_files), args.batch_size):
            batch = chunk_files[i : i + args.batch_size]
            if not batch:
                continue
            out_part = out_root / f"part_{i//args.batch_size:05d}.parquet"
            if out_part.exists() and not args.overwrite:
                continue
            _write_batch([b.as_posix() for b in batch], out_part)
            print(f"[chunk_lookup] wrote batch {i//args.batch_size} ({i+len(batch)}/{len(chunk_files)})")
    elif chunk_files:
        # single pass
        query = f"""
        WITH chunks AS (
            SELECT
                chunk_id,
                CAST(document_id AS VARCHAR) AS document_id,
                CAST(chunk_index AS BIGINT) AS chunk_index,
                CAST(speaker AS VARCHAR) AS speaker,
                CAST(section_type AS VARCHAR) AS section_type
            FROM read_parquet('{chunks_glob}', union_by_name=True)
            WHERE chunk_id IN (SELECT chunk_id FROM fact_chunk_ids)
        )
        SELECT * FROM chunks
        """
        con.execute(f"COPY ({query}) TO '{out_file.as_posix()}' (FORMAT 'parquet');")
        print(f"Wrote chunk lookup year={year} -> {out_file}")
    else:
        # Fallback: scan chunk files one-by-one via DuckDB to bypass unreadable files
        print("[chunk_lookup] fallback: scanning chunks file-by-file")
        out_part = out_root / "part_00000.parquet"
        chunk_list = sorted((chunks_root).glob("*.parquet"))
        total_chunks = len(chunk_list)
        # Create an empty file with the right schema without reading any chunk files
        if (not out_part.exists()) or args.overwrite:
            con.execute(
                f"""
                COPY (
                    SELECT
                        CAST(NULL AS VARCHAR) AS chunk_id,
                        CAST(NULL AS VARCHAR) AS document_id,
                        CAST(NULL AS BIGINT) AS chunk_index,
                        CAST(NULL AS VARCHAR) AS speaker,
                        CAST(NULL AS VARCHAR) AS section_type
                    WHERE FALSE
                ) TO '{out_part.as_posix()}' (FORMAT 'parquet');
                """
            )
        for i, f in enumerate(chunk_list):
            if i % 50 == 0:
                print(f"[chunk_lookup] scanning chunks {i}/{total_chunks}")
            try:
                con.execute(
                    f"""
                    COPY (
                        SELECT
                            chunk_id,
                            CAST(document_id AS VARCHAR) AS document_id,
                            CAST(chunk_index AS BIGINT) AS chunk_index,
                            CAST(speaker AS VARCHAR) AS speaker,
                            CAST(section_type AS VARCHAR) AS section_type
                        FROM read_parquet('{f.as_posix()}', union_by_name=True)
                        WHERE chunk_id IN (SELECT chunk_id FROM fact_chunk_ids)
                    ) TO '{out_part.as_posix()}' (FORMAT 'parquet', APPEND TRUE);
                    """
                )
            except Exception:
                continue
    elapsed = time.time() - start
    print(f"[chunk_lookup] done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
