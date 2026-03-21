#!/usr/bin/env python
"""
Build RawDocumentStore from warehouse_documents.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--documents-root",
        default="data/warehouse/warehouse_documents",
        help="Root directory for warehouse_documents (hive-partitioned by year).",
    )
    parser.add_argument(
        "--out",
        default="data/inputs_layer/raw_documents",
        help="Output RawDocumentStore (directory for partitioned dataset).",
    )
    parser.add_argument("--years", default=None, help="Comma-separated years to process.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory", default="6GB")
    parser.add_argument("--filewise", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--include-raw-text", action="store_true")
    parser.add_argument("--chunks-root", default="data/warehouse/warehouse_doc_chunks")
    parser.add_argument("--text-map-root", default="data/inputs_layer/doc_text_map")
    parser.add_argument("--text-map-only", action="store_true")
    parser.add_argument("--text-separator", default="\\n")
    args = parser.parse_args()

    docs_root = ROOT / args.documents_root
    chunks_root = ROOT / args.chunks_root
    text_map_root = ROOT / args.text_map_root
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

    # resolve years
    if args.years:
        years = [y.strip() for y in args.years.split(",") if y.strip()]
    else:
        years = sorted({p.name.split("=")[1] for p in docs_root.glob("year=*")})

    for y in years:
        docs_year_dir = docs_root / f"year={y}"
        docs_glob = (docs_year_dir / "*.parquet").as_posix()
        if out_is_dir:
            out_year_dir = out_path / f"year={y}"
            out_year_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_year_dir / "part.parquet"
        else:
            out_year_dir = None
            out_file = out_path
        if out_file.exists() and not args.overwrite:
            print(f"Skipping year={y}, output exists.")
            continue

        if args.filewise:
            files = sorted(docs_year_dir.glob("*.parquet"))
            for idx, fpath in enumerate(files):
                out_part = out_file if not out_is_dir else (out_year_dir / f"part_{idx:05d}.parquet")
                if out_part.exists() and not args.overwrite:
                    continue
                chunks_cte = ""
                chunks_join = ""
                raw_text_expr = "NULL::VARCHAR"
                content_hash_expr = "NULL::VARCHAR"
                if args.include_raw_text:
                    text_map_file = text_map_root / f"year={y}" / "part.parquet"
                    if text_map_file.exists():
                        chunks_cte = f"""
                        , text_src AS (
                            SELECT document_id, raw_text, content_hash
                            FROM read_parquet('{text_map_file.as_posix()}')
                        )
                        """
                        chunks_join = "LEFT JOIN text_src USING (document_id)"
                        raw_text_expr = "text_src.raw_text"
                        content_hash_expr = "text_src.content_hash"
                    elif args.text_map_only:
                        # No text map available; leave raw_text/content_hash NULL
                        raw_text_expr = "NULL::VARCHAR"
                        content_hash_expr = "NULL::VARCHAR"
                    else:
                        chunks_glob = (chunks_root / f"year={y}" / "*.parquet").as_posix()
                        chunks_cte = f"""
                        , text_src AS (
                            SELECT
                                document_id,
                                string_agg(text, '{args.text_separator}' ORDER BY chunk_index) AS raw_text,
                                sha256(string_agg(text, '{args.text_separator}' ORDER BY chunk_index)) AS content_hash
                            FROM read_parquet('{chunks_glob}', union_by_name=True)
                            WHERE document_id IN (SELECT document_id FROM docs)
                            GROUP BY 1
                        )
                        """
                        chunks_join = "LEFT JOIN text_src USING (document_id)"
                        raw_text_expr = "text_src.raw_text"
                        content_hash_expr = "text_src.content_hash"
                query = f"""
                WITH docs AS (
                    SELECT
                        source_system,
                        entity_id,
                        event_time,
                        available_time,
                        ingestion_time,
                        version_id,
                        raw_payload_hash,
                        document_id,
                        document_type,
                        title,
                        publisher,
                        analyst,
                        rating,
                        price_target,
                        call_date,
                        publish_date,
                        presentation_date,
                        release_date,
                        source_url
                    FROM read_parquet('{fpath.as_posix()}', union_by_name=True)
                    WHERE document_id IS NOT NULL
                )
                {chunks_cte}
                ,
                norm AS (
                    SELECT
                        document_id,
                        coalesce(version_id, raw_payload_hash, document_id) AS source_id,
                        source_system AS source_type,
                        entity_id,
                        'entity_id' AS entity_id_type,
                        coalesce(
                            CAST(available_time AS TIMESTAMPTZ),
                            CAST(publish_date AS TIMESTAMPTZ),
                            CAST(release_date AS TIMESTAMPTZ),
                            CAST(event_time AS TIMESTAMPTZ)
                        ) AS published_at,
                        coalesce(
                            CAST(release_date AS TIMESTAMPTZ),
                            CAST(publish_date AS TIMESTAMPTZ),
                            CAST(event_time AS TIMESTAMPTZ)
                        ) AS effective_at,
                        CAST(ingestion_time AS TIMESTAMPTZ) AS ingested_at,
                        1.0 AS confidence_score,
                        'data/warehouse/warehouse_documents#document_id=' || document_id AS raw_pointer,
                        document_type AS doc_type,
                        NULL::VARCHAR AS doc_subtype,
                        NULL::VARCHAR AS doc_format,
                        NULL::VARCHAR AS language,
                        title,
                        source_url AS url,
                        raw_payload_hash AS sha256,
                        'data/warehouse/warehouse_doc_chunks#document_id=' || document_id AS content_pointer
                        {f", {raw_text_expr} AS raw_text" if args.include_raw_text else ""}
                        {f", {content_hash_expr} AS content_hash" if args.include_raw_text else ""}
                        {", NULL::VARCHAR AS raw_html" if args.include_raw_text else ""}
                        {", NULL::VARCHAR AS raw_pdf_hash" if args.include_raw_text else ""}
                        {", NULL::INTEGER AS version" if args.include_raw_text else ""}
                        {", NULL::VARCHAR AS metadata" if args.include_raw_text else ""}
                    FROM docs
                    {chunks_join}
                )
                SELECT * FROM norm
                """
                try:
                    con.execute(f"COPY ({query}) TO '{out_part.as_posix()}' (FORMAT 'parquet');")
                except Exception as exc:
                    if args.skip_errors:
                        print(f"[skip] {fpath.name}: {exc}")
                        continue
                    raise
            print(f"Saved RawDocumentStore year={y} -> {out_year_dir if out_is_dir else out_file}")
            continue

        chunks_cte = ""
        chunks_join = ""
        raw_text_expr = "NULL::VARCHAR"
        content_hash_expr = "NULL::VARCHAR"
        if args.include_raw_text:
            text_map_file = text_map_root / f"year={y}" / "part.parquet"
            if text_map_file.exists():
                chunks_cte = f"""
                , text_src AS (
                    SELECT document_id, raw_text, content_hash
                    FROM read_parquet('{text_map_file.as_posix()}')
                )
                """
                chunks_join = "LEFT JOIN text_src USING (document_id)"
                raw_text_expr = "text_src.raw_text"
                content_hash_expr = "text_src.content_hash"
            elif args.text_map_only:
                raw_text_expr = "NULL::VARCHAR"
                content_hash_expr = "NULL::VARCHAR"
            else:
                chunks_glob = (chunks_root / f"year={y}" / "*.parquet").as_posix()
                chunks_cte = f"""
                , text_src AS (
                    SELECT
                        document_id,
                        string_agg(text, '{args.text_separator}' ORDER BY chunk_index) AS raw_text,
                        sha256(string_agg(text, '{args.text_separator}' ORDER BY chunk_index)) AS content_hash
                    FROM read_parquet('{chunks_glob}', union_by_name=True)
                    GROUP BY 1
                )
                """
                chunks_join = "LEFT JOIN text_src USING (document_id)"
                raw_text_expr = "text_src.raw_text"
                content_hash_expr = "text_src.content_hash"

        query = f"""
        WITH docs AS (
            SELECT
                source_system,
                entity_id,
                event_time,
                available_time,
                ingestion_time,
                version_id,
                raw_payload_hash,
                document_id,
                document_type,
                title,
                publisher,
                analyst,
                rating,
                price_target,
                call_date,
                publish_date,
                presentation_date,
                release_date,
                source_url
            FROM read_parquet('{docs_glob}', union_by_name=True)
            WHERE document_id IS NOT NULL
        )
        {chunks_cte}
        ,
        norm AS (
            SELECT
                document_id,
                coalesce(version_id, raw_payload_hash, document_id) AS source_id,
                source_system AS source_type,
                entity_id,
                'entity_id' AS entity_id_type,
                coalesce(
                    CAST(available_time AS TIMESTAMPTZ),
                    CAST(publish_date AS TIMESTAMPTZ),
                    CAST(release_date AS TIMESTAMPTZ),
                    CAST(event_time AS TIMESTAMPTZ)
                ) AS published_at,
                coalesce(
                    CAST(release_date AS TIMESTAMPTZ),
                    CAST(publish_date AS TIMESTAMPTZ),
                    CAST(event_time AS TIMESTAMPTZ)
                ) AS effective_at,
                CAST(ingestion_time AS TIMESTAMPTZ) AS ingested_at,
                1.0 AS confidence_score,
                'data/warehouse/warehouse_documents#document_id=' || document_id AS raw_pointer,
                document_type AS doc_type,
                NULL::VARCHAR AS doc_subtype,
                NULL::VARCHAR AS doc_format,
                NULL::VARCHAR AS language,
                title,
                source_url AS url,
                raw_payload_hash AS sha256,
                'data/warehouse/warehouse_doc_chunks#document_id=' || document_id AS content_pointer
                {f", {raw_text_expr} AS raw_text" if args.include_raw_text else ""}
                {f", {content_hash_expr} AS content_hash" if args.include_raw_text else ""}
                {", NULL::VARCHAR AS raw_html" if args.include_raw_text else ""}
                {", NULL::VARCHAR AS raw_pdf_hash" if args.include_raw_text else ""}
                {", NULL::INTEGER AS version" if args.include_raw_text else ""}
                {", NULL::VARCHAR AS metadata" if args.include_raw_text else ""}
            FROM docs
            {chunks_join}
        ),
        ranked AS (
            SELECT *, row_number() OVER (PARTITION BY document_id ORDER BY published_at DESC NULLS LAST) AS rn
            FROM norm
        )
        SELECT
            document_id,
            source_id,
            source_type,
            entity_id,
            entity_id_type,
            published_at,
            effective_at,
            ingested_at,
            confidence_score,
            raw_pointer,
            doc_type,
            doc_subtype,
            doc_format,
            language,
            title,
            url,
            sha256,
            content_pointer
            {", raw_text" if args.include_raw_text else ""}
            {", content_hash" if args.include_raw_text else ""}
            {", raw_html" if args.include_raw_text else ""}
            {", raw_pdf_hash" if args.include_raw_text else ""}
            {", version" if args.include_raw_text else ""}
            {", metadata" if args.include_raw_text else ""}
        FROM ranked
        WHERE rn = 1
        """

        con.execute(f"COPY ({query}) TO '{out_file.as_posix()}' (FORMAT 'parquet');")
        print(f"Saved RawDocumentStore year={y} -> {out_file}")

    print(f"Saved RawDocumentStore -> {out_path}")


if __name__ == "__main__":
    main()
