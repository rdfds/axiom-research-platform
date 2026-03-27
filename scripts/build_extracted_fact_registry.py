#!/usr/bin/env python
"""
Build ExtractedFactRegistry from warehouse text signals + document chunks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import duckdb


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--signals-root",
        default="data/warehouse/warehouse_text_signals",
        help="Root directory for warehouse_text_signals (hive-partitioned by year).",
    )
    parser.add_argument(
        "--chunks-root",
        default="data/warehouse/warehouse_doc_chunks",
        help="Root directory for warehouse_doc_chunks (hive-partitioned by year).",
    )
    parser.add_argument(
        "--out",
        default="data/inputs_layer/extracted_fact_registry",
        help="Output directory for ExtractedFactRegistry parquet dataset.",
    )
    parser.add_argument(
        "--years",
        default=None,
        help="Comma-separated years to process (e.g., 2018,2019). Defaults to all years found.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-year outputs if present.",
    )
    parser.add_argument(
        "--use-chunks",
        action="store_true",
        help="Join to doc_chunks for document_id/speaker/paragraph info (slower).",
    )
    parser.add_argument(
        "--filewise",
        action="store_true",
        help="Process each parquet file individually to avoid schema conflicts.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip files that fail to read when running in --filewise mode.",
    )
    args = parser.parse_args()

    signals_root = ROOT / args.signals_root
    chunks_root = ROOT / args.chunks_root
    signals_glob = (signals_root / "**/*.parquet").as_posix()
    chunks_glob = (chunks_root / "**/*.parquet").as_posix()
    out_path = (ROOT / args.out)
    if out_path.suffix == ".parquet":
        # treat as file path
        out_is_dir = False
        out_path = out_path
    else:
        out_is_dir = True
        out_path.mkdir(parents=True, exist_ok=True)
    out_path_str = out_path.as_posix()

    con = duckdb.connect()

    # resolve years
    if args.years:
        years = [y.strip() for y in args.years.split(",") if y.strip()]
    else:
        years = []
        for p in signals_root.glob("year=*"):
            m = re.search(r"year=(\d{4})", p.as_posix())
            if m:
                years.append(m.group(1))
        years = sorted(set(years))

    if not years:
        raise RuntimeError("No years found to process.")

    for y in years:
        if out_is_dir:
            year_dir = out_path / f"year={y}"
            year_dir.mkdir(parents=True, exist_ok=True)
            out_file = year_dir / "part.parquet"
        else:
            out_file = out_path

        if out_file.exists() and not args.overwrite:
            print(f"Skipping year={y}, output exists: {out_file}")
            continue

        year_dir = signals_root / f"year={y}"
        signals_glob_year = (year_dir / "*.parquet").as_posix()
        chunks_glob_year = (chunks_root / f"year={y}" / "*.parquet").as_posix()

        if args.filewise:
            files = sorted(year_dir.glob("*.parquet"))
            if not files:
                print(f"No files for year={y}")
                continue
            year_out_dir = out_path / f"year={y}"
            year_out_dir.mkdir(parents=True, exist_ok=True)
            for idx, fpath in enumerate(files):
                out_part = out_file if not out_is_dir else (year_out_dir / f"part_{idx:05d}.parquet")
                if out_part.exists() and not args.overwrite:
                    continue
                query = f"""
                SELECT
                    'fact_' || md5(coalesce(version_id, '') || '|' || coalesce(signal_name, '') || '|' || coalesce(list_extract(supporting_chunk_ids, 1), '')) AS fact_id,
                    'doc:' || coalesce(version_id, raw_payload_hash, signal_name) AS document_id,
                    entity_id,
                    signal_name AS fact_type,
                    value AS fact_value,
                    NULL::VARCHAR AS unit,
                    NULL::VARCHAR AS context,
                    coalesce(confidence, 1.0) AS confidence_score,
                    list_extract(supporting_chunk_ids, 1) AS citation_span,
                    NULL::INTEGER AS paragraph_index,
                    NULL::VARCHAR AS speaker,
                    NULL::VARCHAR AS transcript_timestamp,
                    NULL::BOOLEAN AS is_qa,
                    coalesce(version_id, raw_payload_hash, signal_name) AS source_id,
                    source_system AS source_type,
                    CAST(event_time AS TIMESTAMPTZ) AS published_at,
                    CAST(event_time AS TIMESTAMPTZ) AS effective_at,
                    CAST(ingestion_time AS TIMESTAMPTZ) AS ingested_at,
                    'data/warehouse/warehouse_text_signals' || '#version_id=' || coalesce(version_id, '') || '|signal=' || coalesce(signal_name, '') AS raw_pointer
                FROM read_parquet('{fpath.as_posix()}', union_by_name=True)
                WHERE signal_name IS NOT NULL AND entity_id IS NOT NULL
                """
                try:
                    con.execute(f"COPY ({query}) TO '{out_part.as_posix()}' (FORMAT 'parquet');")
                except Exception as exc:
                    if args.skip_errors:
                        print(f"[skip] {fpath.name}: {exc}")
                        continue
                    raise
            print(f"Saved ExtractedFactRegistry year={y} -> {year_out_dir}")
            continue

        if args.use_chunks:
            query = f"""
            WITH signals AS (
                SELECT
                    source_system,
                    entity_id,
                    event_time,
                    ingestion_time,
                    version_id,
                    raw_payload_hash,
                    signal_name,
                    value,
                    confidence,
                    list_extract(supporting_chunk_ids, 1) AS chunk_id
                FROM read_parquet('{signals_glob_year}', hive_partitioning=1, union_by_name=True)
                WHERE signal_name IS NOT NULL
            ),
            chunks AS (
                SELECT
                    chunk_id,
                    document_id,
                    chunk_index,
                    speaker,
                    section_type
                FROM read_parquet('{chunks_glob_year}', hive_partitioning=1, union_by_name=True)
            )
            SELECT
                'fact_' || md5(coalesce(s.version_id, '') || '|' || coalesce(s.signal_name, '') || '|' || coalesce(s.chunk_id, '')) AS fact_id,
                coalesce(c.document_id, 'doc_unknown:' || md5(coalesce(s.version_id, '') || '|' || coalesce(s.signal_name, ''))) AS document_id,
                s.entity_id,
                s.signal_name AS fact_type,
                s.value AS fact_value,
                NULL::VARCHAR AS unit,
                NULL::VARCHAR AS context,
                coalesce(s.confidence, 1.0) AS confidence_score,
                s.chunk_id AS citation_span,
                c.chunk_index AS paragraph_index,
                c.speaker AS speaker,
                NULL::VARCHAR AS transcript_timestamp,
                CASE
                    WHEN c.section_type IS NULL THEN NULL
                    WHEN lower(c.section_type) LIKE '%qa%' THEN TRUE
                    ELSE FALSE
                END AS is_qa,
                coalesce(s.version_id, s.raw_payload_hash, s.signal_name) AS source_id,
                s.source_system AS source_type,
                CAST(s.event_time AS TIMESTAMPTZ) AS published_at,
                CAST(s.event_time AS TIMESTAMPTZ) AS effective_at,
                CAST(s.ingestion_time AS TIMESTAMPTZ) AS ingested_at,
                'data/warehouse/warehouse_text_signals' || '#version_id=' || coalesce(s.version_id, '') || '|signal=' || coalesce(s.signal_name, '') AS raw_pointer
            FROM signals s
            LEFT JOIN chunks c USING (chunk_id)
            WHERE s.entity_id IS NOT NULL
            """
        else:
            query = f"""
            SELECT
                'fact_' || md5(coalesce(version_id, '') || '|' || coalesce(signal_name, '') || '|' || coalesce(list_extract(supporting_chunk_ids, 1), '')) AS fact_id,
                'doc:' || coalesce(version_id, raw_payload_hash, signal_name) AS document_id,
                entity_id,
                signal_name AS fact_type,
                value AS fact_value,
                NULL::VARCHAR AS unit,
                NULL::VARCHAR AS context,
                coalesce(confidence, 1.0) AS confidence_score,
                list_extract(supporting_chunk_ids, 1) AS citation_span,
                NULL::INTEGER AS paragraph_index,
                NULL::VARCHAR AS speaker,
                NULL::VARCHAR AS transcript_timestamp,
                NULL::BOOLEAN AS is_qa,
                coalesce(version_id, raw_payload_hash, signal_name) AS source_id,
                source_system AS source_type,
                CAST(event_time AS TIMESTAMPTZ) AS published_at,
                CAST(event_time AS TIMESTAMPTZ) AS effective_at,
                CAST(ingestion_time AS TIMESTAMPTZ) AS ingested_at,
                'data/warehouse/warehouse_text_signals' || '#version_id=' || coalesce(version_id, '') || '|signal=' || coalesce(signal_name, '') AS raw_pointer
            FROM read_parquet('{signals_glob_year}', hive_partitioning=1, union_by_name=True)
            WHERE signal_name IS NOT NULL AND entity_id IS NOT NULL
            """

        con.execute(f"COPY ({query}) TO '{out_file.as_posix()}' (FORMAT 'parquet');")
        print(f"Saved ExtractedFactRegistry year={y} -> {out_file}")

    if out_is_dir:
        print(f"Saved ExtractedFactRegistry dataset -> {out_path_str}")
    else:
        print(f"Saved ExtractedFactRegistry -> {out_path_str}")


if __name__ == "__main__":
    main()
