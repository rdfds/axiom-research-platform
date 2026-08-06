#!/usr/bin/env python
"""
Build EntityRelationship edges from M&A deals (acquirer -> target).
Uses warehouse_mna_deals.parquet (Refinitiv feed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mna-path", default="data/warehouse/warehouse_mna_deals.parquet")
    parser.add_argument("--out", default="data/inputs_layer/entity_relationship.parquet")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory", default="6GB")
    args = parser.parse_args()

    mna_path = ROOT / args.mna_path
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.overwrite:
        print(f"Output exists: {out_path}")
        return

    con = duckdb.connect()
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET memory_limit='{args.memory}'")

    query = f"""
    WITH mna AS (
        SELECT
            deal_id,
            source_system,
            raw_payload_hash,
            version_id,
            available_time,
            event_time,
            ingestion_time,
            announcement_date,
            close_date,
            try_cast(acquirer_company_id AS BIGINT) AS acquirer_id,
            try_cast(target_company_id AS BIGINT) AS target_id
        FROM read_parquet('{mna_path.as_posix()}')
    ),
    norm AS (
        SELECT
            CAST(acquirer_id AS VARCHAR) AS parent_entity_id,
            CAST(target_id AS VARCHAR) AS child_entity_id,
            'acquirer' AS relationship_type,
            CAST(announcement_date AS TIMESTAMPTZ) AS valid_from,
            try_cast(close_date AS TIMESTAMPTZ) AS valid_to,
            deal_id AS source_id,
            source_system AS source_type,
            coalesce(
                CAST(announcement_date AS TIMESTAMPTZ),
                CAST(available_time AS TIMESTAMPTZ),
                CAST(event_time AS TIMESTAMPTZ)
            ) AS published_at,
            try_cast(close_date AS TIMESTAMPTZ) AS effective_at,
            CAST(ingestion_time AS TIMESTAMPTZ) AS ingested_at,
            1.0 AS confidence_score,
            '{mna_path.as_posix()}#deal_id=' || deal_id AS raw_pointer
        FROM mna
        WHERE acquirer_id IS NOT NULL
          AND target_id IS NOT NULL
          AND acquirer_id != target_id
    )
    SELECT * FROM norm
    """

    con.execute(f"COPY ({query}) TO '{out_path.as_posix()}' (FORMAT 'parquet');")
    print(f"Wrote EntityRelationship (M&A) -> {out_path}")


if __name__ == "__main__":
    main()
