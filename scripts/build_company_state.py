#!/usr/bin/env python
"""
Build a point-in-time CompanyState snapshot.

This is a baseline assembler that joins:
- RawTimeSeriesStore (prices + macro + estimates)
- EventRegistry (corporate actions)
- ExtractedFactRegistry (text-derived signals)
- EntityGraph (ID resolution)

Default output is a per-entity, per-asof snapshot in LONG format for scale.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import duckdb


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asof", required=True, help="As-of date (YYYY-MM-DD).")
    parser.add_argument("--out", default="data/company_state/company_state.parquet")
    parser.add_argument("--format", choices=["long", "wide"], default="long")
    parser.add_argument("--engine", choices=["duckdb", "pandas"], default="duckdb")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory", default="6GB")
    parser.add_argument("--timeseries-path", default="data/inputs_layer/raw_timeseries.parquet")
    parser.add_argument("--events-path", default="data/inputs_layer/event_registry_enriched.parquet")
    parser.add_argument("--facts-cache-dir", default="data/company_state/_facts_yearly")
    parser.add_argument("--rebuild-facts-cache", action="store_true")
    parser.add_argument("--facts-year-start", type=int, default=2005)
    parser.add_argument("--facts-year-end", type=int, default=None)
    parser.add_argument("--cache-only", action="store_true", help="Build facts cache only, skip final CompanyState.")
    parser.add_argument("--sample-facts", type=int, default=0, help="Limit facts rows (0 = all).")
    args = parser.parse_args()

    asof = pd.to_datetime(args.asof, utc=True)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.engine == "duckdb":
        con = duckdb.connect()
        con.execute(f"SET threads={args.threads}")
        con.execute(f"SET memory_limit='{args.memory}'")

        ts_path = (ROOT / args.timeseries_path).as_posix()
        ev_path = (ROOT / args.events_path).as_posix()
        facts_path = (ROOT / "data/inputs_layer/extracted_fact_registry_enriched").as_posix()
        graph_path = (ROOT / "data/inputs_layer/entity_graph.parquet").as_posix()
        cache_dir = ROOT / args.facts_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        if args.format == "long":
            asof_year = asof.year
            year_start = args.facts_year_start
            year_end = args.facts_year_end if args.facts_year_end else asof_year
            # Build per-year facts cache to speed up full aggregation
            for y in range(year_start, min(year_end, asof_year) + 1):
                year_dir = cache_dir / f"year={y}"
                out_file = year_dir / "part.parquet"
                if out_file.exists() and not args.rebuild_facts_cache:
                    continue
                year_dir.mkdir(parents=True, exist_ok=True)
                year_fact_dir = Path(facts_path) / f"year={y}"
                src_parts = []
                text_path = year_fact_dir / "part.parquet"
                fin_path = year_fact_dir / "part_financials.parquet"
                if text_path.exists():
                    src_parts.append(
                        f"""
                        SELECT entity_id, fact_type, fact_value,
                               CAST(published_at AS TIMESTAMPTZ) AS published_at
                        FROM read_parquet('{text_path.as_posix()}')
                        """
                    )
                if fin_path.exists():
                    src_parts.append(
                        f"""
                        SELECT entity_id, fact_type, fact_value,
                               CAST(published_at AS TIMESTAMPTZ) AS published_at
                        FROM read_parquet('{fin_path.as_posix()}')
                        """
                    )
                if not src_parts:
                    continue
                union_src = " UNION ALL ".join(src_parts)
                facts_year_query = f"""
                SELECT
                    entity_id,
                    fact_type,
                    arg_max(fact_value, published_at) AS value,
                    max(published_at) AS published_at
                FROM ({union_src})
                WHERE published_at <= TIMESTAMPTZ '{asof.isoformat()}'
                GROUP BY 1,2
                """
                con.execute(f"COPY ({facts_year_query}) TO '{out_file.as_posix()}' (FORMAT 'parquet');")
            if args.cache_only:
                print("Facts cache built; skipping final CompanyState (cache-only).")
                return

            query = f"""
            WITH ts AS (
                SELECT
                    entity_id,
                    series_id,
                    arg_max(value, published_at) AS value,
                    max(published_at) AS published_at
                FROM read_parquet('{ts_path}')
                WHERE published_at <= TIMESTAMPTZ '{asof.isoformat()}'
                GROUP BY 1,2
            ),
            facts AS (
                SELECT
                    entity_id,
                    fact_type,
                    arg_max(value, published_at) AS value,
                    max(published_at) AS published_at
                FROM read_parquet('{cache_dir.as_posix()}/**/*.parquet', hive_partitioning=1, union_by_name=True)
                GROUP BY 1,2
            ),
            events AS (
                SELECT
                    company_id AS entity_id,
                    event_type AS action_type,
                    max(coalesce(announced_at, effective_at, created_at)) AS published_at
                FROM read_parquet('{ev_path}')
                WHERE coalesce(announced_at, effective_at, created_at) <= TIMESTAMPTZ '{asof.isoformat()}'
                  AND company_id IS NOT NULL
                GROUP BY 1,2
            )
            SELECT
                entity_id,
                'ts' AS feature_group,
                series_id AS feature_key,
                try_cast(value AS DOUBLE) AS value_num,
                CAST(value AS VARCHAR) AS value_str,
                NULL::TIMESTAMPTZ AS value_ts,
                published_at,
                TIMESTAMPTZ '{asof.isoformat()}' AS asof,
                TIMESTAMPTZ '{utc_now()}' AS built_at
            FROM ts
            UNION ALL
            SELECT
                entity_id,
                'fact' AS feature_group,
                fact_type AS feature_key,
                try_cast(value AS DOUBLE) AS value_num,
                CAST(value AS VARCHAR) AS value_str,
                NULL::TIMESTAMPTZ AS value_ts,
                published_at,
                TIMESTAMPTZ '{asof.isoformat()}' AS asof,
                TIMESTAMPTZ '{utc_now()}' AS built_at
            FROM facts
            UNION ALL
            SELECT
                e.entity_id,
                'event' AS feature_group,
                e.action_type AS feature_key,
                NULL::DOUBLE AS value_num,
                NULL::VARCHAR AS value_str,
                e.published_at AS value_ts,
                e.published_at AS published_at,
                TIMESTAMPTZ '{asof.isoformat()}' AS asof,
                TIMESTAMPTZ '{utc_now()}' AS built_at
            FROM events e
            """
            con.execute(f"COPY ({query}) TO '{out_path.as_posix()}' (FORMAT 'parquet');")
            print(f"Saved CompanyState (long) -> {out_path}")
        else:
            raise NotImplementedError("Wide format not supported with duckdb engine yet.")
        return

    # pandas fallback (slow)
    ts = pd.read_parquet(ROOT / args.timeseries_path)
    ev = pd.read_parquet(ROOT / args.events_path)
    facts = pd.read_parquet(ROOT / "data/inputs_layer/extracted_fact_registry_enriched")
    if args.sample_facts and len(facts) > args.sample_facts:
        facts = facts.sample(args.sample_facts, random_state=42)
    graph = pd.read_parquet(ROOT / "data/inputs_layer/entity_graph.parquet")

    ts["published_at"] = pd.to_datetime(ts["published_at"], utc=True, errors="coerce")
    if "published_at" in ev.columns:
        ev["published_at"] = pd.to_datetime(ev["published_at"], utc=True, errors="coerce")
    else:
        ev["published_at"] = pd.to_datetime(
            ev.get("announced_at").combine_first(ev.get("effective_at")).combine_first(ev.get("created_at")),
            utc=True,
            errors="coerce",
        )
    facts["published_at"] = pd.to_datetime(facts["published_at"], utc=True, errors="coerce")
    ts = ts[ts["published_at"] <= asof]
    ev = ev[ev["published_at"] <= asof]
    facts = facts[facts["published_at"] <= asof]

    latest_ts = ts.sort_values("published_at").groupby(["entity_id", "series_id"]).tail(1)
    ev["action_type"] = ev.get("event_type").combine_first(ev.get("action_type_norm")).combine_first(ev.get("action_type"))
    latest_ev = ev.sort_values("published_at").groupby(["company_id", "action_type"]).tail(1)
    latest_facts = facts.sort_values("published_at").groupby(["entity_id", "fact_type"]).tail(1)

    ts_wide = latest_ts.pivot_table(index="entity_id", columns="series_id", values="value", aggfunc="last")
    facts_wide = latest_facts.pivot_table(index="entity_id", columns="fact_type", values="fact_value", aggfunc="last")

    ts_wide.columns = [f"ts::{c}" for c in ts_wide.columns]
    facts_wide.columns = [f"fact::{c}" for c in facts_wide.columns]

    ev_latest = latest_ev.rename(columns={"company_id": "entity_id"})
    ev_wide = ev_latest.pivot_table(index="entity_id", columns="action_type", values="published_at", aggfunc="last")
    ev_wide.columns = [f"event::{c}::last_date" for c in ev_wide.columns]

    df = ts_wide.join(facts_wide, how="outer").join(ev_wide, how="outer")
    df["asof"] = asof
    df["built_at"] = utc_now()

    df.reset_index().to_parquet(out_path, index=False)
    print(f"Saved CompanyState -> {out_path} ({len(df):,} entities)")


if __name__ == "__main__":
    main()
