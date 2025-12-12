#!/usr/bin/env python
"""
Build per-year document text map from warehouse_doc_chunks.

Outputs: data/inputs_layer/doc_text_map/year=YYYY/part.parquet
Columns: document_id, raw_text, content_hash
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-root", default="data/warehouse/warehouse_doc_chunks")
    parser.add_argument("--out-root", default="data/inputs_layer/doc_text_map")
    parser.add_argument("--facts-root", default="data/inputs_layer/extracted_fact_registry_enriched")
    parser.add_argument("--limit-to-facts", action="store_true")
    parser.add_argument("--years", default=None, help="Comma-separated years to process.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory", default="6GB")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--text-separator", default="\\n")
    args = parser.parse_args()

    chunks_root = ROOT / args.chunks_root
    out_root = ROOT / args.out_root
    facts_root = ROOT / args.facts_root
    out_root.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET memory_limit='{args.memory}'")

    if args.years:
        years = [y.strip() for y in args.years.split(",") if y.strip()]
    else:
        years = sorted({p.name.split('=')[1] for p in chunks_root.glob('year=*')})

    for y in years:
        year_dir = out_root / f"year={y}"
        out_file = year_dir / "part.parquet"
        if out_file.exists() and not args.overwrite:
            print(f"Skipping year={y}, output exists.")
            continue
        year_dir.mkdir(parents=True, exist_ok=True)

        chunks_dir = chunks_root / f"year={y}"
        if not chunks_dir.exists() or not any(chunks_dir.glob("*.parquet")):
            print(f"Skipping year={y}, no chunk files found.")
            continue

        chunks_glob = (chunks_dir / "*.parquet").as_posix()
        facts_path = facts_root / f"year={y}" / "part.parquet"
        if args.limit_to_facts and facts_path.exists():
            query = f"""
            WITH fact_docs AS (
                SELECT DISTINCT document_id
                FROM read_parquet('{facts_path.as_posix()}')
                WHERE document_id IS NOT NULL
            ),
            chunks AS (
                SELECT document_id, chunk_index, text
                FROM read_parquet('{chunks_glob}', union_by_name=True)
                WHERE document_id IN (SELECT document_id FROM fact_docs)
            )
            SELECT
                document_id,
                string_agg(text, '{args.text_separator}' ORDER BY chunk_index) AS raw_text,
                sha256(string_agg(text, '{args.text_separator}' ORDER BY chunk_index)) AS content_hash
            FROM chunks
            GROUP BY 1
            """
        else:
            query = f"""
            WITH chunks AS (
                SELECT document_id, chunk_index, text
                FROM read_parquet('{chunks_glob}', union_by_name=True)
                WHERE document_id IS NOT NULL
            )
            SELECT
                document_id,
                string_agg(text, '{args.text_separator}' ORDER BY chunk_index) AS raw_text,
                sha256(string_agg(text, '{args.text_separator}' ORDER BY chunk_index)) AS content_hash
            FROM chunks
            GROUP BY 1
            """
        con.execute(f"COPY ({query}) TO '{out_file.as_posix()}' (FORMAT 'parquet');")
        print(f"Wrote doc text map year={y} -> {out_file}")

    print(f"Saved doc text map -> {out_root}")


if __name__ == "__main__":
    main()
