#!/usr/bin/env python
"""
Enrich ExtractedFactRegistry with document_id, paragraph_index, speaker, is_qa
by joining to warehouse_doc_chunks on chunk_id (citation_span).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import duckdb
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts-root", default="data/inputs_layer/extracted_fact_registry")
    parser.add_argument("--chunks-root", default="data/warehouse/warehouse_doc_chunks")
    parser.add_argument("--out", default="data/inputs_layer/extracted_fact_registry_enriched")
    parser.add_argument("--years", default=None, help="Comma-separated years (default: all).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory", default="6GB")
    parser.add_argument("--filewise", action="store_true", help="Process each fact part individually.")
    parser.add_argument("--skip-errors", action="store_true", help="Skip unreadable files in filewise mode.")
    parser.add_argument(
        "--skip-errors-fast",
        action="store_true",
        help="Fast mode: avoid pre-scanning; on error, drop bad files and retry.",
    )
    args = parser.parse_args()

    facts_root = ROOT / args.facts_root
    chunks_root = ROOT / args.chunks_root
    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    out_is_dir = True

    con = duckdb.connect()
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET memory_limit='{args.memory}'")

    # resolve years
    if args.years:
        years = [y.strip() for y in args.years.split(",") if y.strip()]
    else:
        years = []
        for p in facts_root.glob("year=*"):
            m = re.search(r"year=(\d{4})", p.as_posix())
            if m:
                years.append(m.group(1))
        years = sorted(set(years))

    for y in years:
        out_dir = out_root / f"year={y}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "part.parquet"
        if (not args.filewise) and out_file.exists() and not args.overwrite:
            print(f"Skipping year={y}, output exists.")
            continue

        print(f"[enrich] year={y} | filewise={args.filewise} | skip_errors={args.skip_errors}")

        facts_glob = (facts_root / f"year={y}" / "*.parquet").as_posix()
        chunks_glob = (chunks_root / f"year={y}" / "*.parquet").as_posix()

        def build_query(facts_source_sql: str, chunks_source_sql: str) -> str:
            return f"""
        WITH facts AS (
            SELECT *
            FROM {facts_source_sql}
        ),
        fact_chunk_ids AS (
            SELECT DISTINCT citation_span AS chunk_id
            FROM facts
            WHERE citation_span IS NOT NULL
        ),
        chunks AS (
            SELECT
                chunk_id,
                CAST(document_id AS VARCHAR) AS doc_document_id,
                CAST(chunk_index AS BIGINT) AS chunk_index,
                CAST(speaker AS VARCHAR) AS speaker,
                CAST(section_type AS VARCHAR) AS section_type
            FROM {chunks_source_sql}
            WHERE chunk_id IN (SELECT chunk_id FROM fact_chunk_ids)
        )
        SELECT
            f.fact_id,
            coalesce(CAST(f.document_id AS VARCHAR), c.doc_document_id) AS document_id,
            f.entity_id,
            f.fact_type,
            f.fact_value,
            f.unit,
            f.context,
            f.confidence_score,
            f.citation_span,
            coalesce(try_cast(f.paragraph_index AS BIGINT), c.chunk_index) AS paragraph_index,
            coalesce(CAST(f.speaker AS VARCHAR), c.speaker) AS speaker,
            f.transcript_timestamp,
            coalesce(
                try_cast(f.is_qa AS BOOLEAN),
                CAST(CASE
                    WHEN c.section_type IS NULL THEN NULL
                    WHEN lower(c.section_type) LIKE '%qa%' THEN TRUE
                    ELSE FALSE
                END AS BOOLEAN)
            ) AS is_qa,
            f.source_id,
            f.source_type,
            f.published_at,
            f.effective_at,
            f.ingested_at,
            f.raw_pointer
        FROM facts f
        LEFT JOIN chunks c
          ON f.citation_span = c.chunk_id
        """

        if args.filewise:
            fact_dir = facts_root / f"year={y}"
            files = sorted(fact_dir.glob("*.parquet"))
            if not files:
                print(f"No files for year={y}")
                continue
            year_out_dir = out_root / f"year={y}"
            year_out_dir.mkdir(parents=True, exist_ok=True)
            for idx, fpath in enumerate(files):
                out_part = out_file if out_is_dir is False else (year_out_dir / f"part_{idx:05d}.parquet")
                if out_part.exists() and not args.overwrite:
                    continue
                facts_source = f"read_parquet('{fpath.as_posix()}', union_by_name=True)"
                chunks_source = f"read_parquet('{chunks_glob}', union_by_name=True)"
                if idx % 50 == 0:
                    print(f"[enrich] year={y} part={idx}/{len(files)}")
                query = build_query(facts_source, chunks_source)
                try:
                    con.execute(f"COPY ({query}) TO '{out_part.as_posix()}' (FORMAT 'parquet');")
                except Exception as exc:
                    if args.skip_errors:
                        print(f"[skip] {fpath.name}: {exc}")
                        continue
                    raise
            print(f"Enriched year={y} -> {year_out_dir}")
        else:
            facts_source = f"read_parquet('{facts_glob}', union_by_name=True)"
            chunks_source = f"read_parquet('{chunks_glob}', union_by_name=True)"
            if args.skip_errors and not args.skip_errors_fast:
                # filter unreadable fact/chunk files to avoid full job failure
                def _filter_readable(file_list: list[Path]) -> list[str]:
                    ok = []
                    for fp in file_list:
                        try:
                            pq.ParquetFile(fp)
                            ok.append(fp.as_posix())
                        except Exception:
                            continue
                    return ok

                fact_files = sorted((facts_root / f"year={y}").glob("*.parquet"))
                chunk_files = sorted((chunks_root / f"year={y}").glob("*.parquet"))
                fact_ok = _filter_readable(fact_files)
                chunk_ok = _filter_readable(chunk_files)
                print(f"[enrich] year={y} readable facts={len(fact_ok)}/{len(fact_files)} chunks={len(chunk_ok)}/{len(chunk_files)}")
                if fact_ok:
                    fact_list = ", ".join([("'" + p + "'") for p in fact_ok])
                    facts_source = f"read_parquet([{fact_list}], union_by_name=True)"
                if chunk_ok:
                    chunk_list = ", ".join([("'" + p + "'") for p in chunk_ok])
                    chunks_source = f"read_parquet([{chunk_list}], union_by_name=True)"

            def _run_with_retry(facts_source_sql: str, chunks_source_sql: str) -> None:
                attempt = 0
                bad = set()
                while True:
                    attempt += 1
                    query = build_query(facts_source_sql, chunks_source_sql)
                    try:
                        con.execute(f"COPY ({query}) TO '{out_file.as_posix()}' (FORMAT 'parquet');")
                        print(f"Enriched year={y} -> {out_file}")
                        return
                    except duckdb.Error as e:
                        msg = str(e)
                        m = re.search(r"file '([^']+\\.parquet)'", msg)
                        if not m:
                            raise
                        bad_file = m.group(1)
                        if bad_file in bad or not args.skip_errors:
                            raise
                        bad.add(bad_file)
                        print(f"[skip] dropping bad file: {bad_file}")
                        # rebuild file lists excluding bad files
                        fact_files = sorted((facts_root / f"year={y}").glob("*.parquet"))
                        chunk_files = sorted((chunks_root / f"year={y}").glob("*.parquet"))
                        fact_list = [f.as_posix() for f in fact_files if f.as_posix() not in bad]
                        chunk_list = [f.as_posix() for f in chunk_files if f.as_posix() not in bad]
                        if not fact_list:
                            raise
                        fact_list_sql = ", ".join([("'" + p + "'") for p in fact_list])
                        facts_source_sql = f"read_parquet([{fact_list_sql}], union_by_name=True)"
                        if chunk_list:
                            chunk_list_sql = ", ".join([("'" + p + "'") for p in chunk_list])
                            chunks_source_sql = f"read_parquet([{chunk_list_sql}], union_by_name=True)"
                        if attempt > 10:
                            raise

            _run_with_retry(facts_source, chunks_source)

    print(f"Saved enriched ExtractedFactRegistry -> {out_root}")


if __name__ == "__main__":
    main()
