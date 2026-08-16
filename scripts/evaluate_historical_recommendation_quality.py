#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from src.historical_recommendation_eval import (
    build_historical_recommendation_report,
    render_historical_recommendation_markdown,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run historical ex-post validation for recommendation quality.")
    parser.add_argument("--runs-root", required=True, help="Output root for generated historical recommendation runs.")
    parser.add_argument("--outcomes-path", required=True, help="Normalized outcomes parquet used for both case selection and precedent matching.")
    parser.add_argument("--entity-graph-path", default="data/inputs_layer/entity_graph.parquet")
    parser.add_argument("--entity-identifier-path", default="data/inputs_layer/entity_identifier.parquet")
    parser.add_argument("--entity-table-path", default="data/inputs_layer/entity.parquet")
    parser.add_argument("--raw-timeseries-path", default="data/inputs_layer/raw_timeseries.parquet")
    parser.add_argument("--macro-timeseries-path")
    parser.add_argument("--event-store-path", default="data/inputs_layer/event_store.parquet")
    parser.add_argument("--facts-path", default="data/inputs_layer/extracted_fact_registry_validity")
    parser.add_argument("--ownership-summary-path", default="data/inputs_layer/ownership_13f_summary.parquet")
    parser.add_argument("--issuer-ratings-path", default="data/inputs_layer/issuer_rating_history.parquet")
    parser.add_argument(
        "--case-count",
        type=int,
        help="Requested supported cases. With --fixed-cases-json, omitting this replays every manifest case.",
    )
    parser.add_argument("--lookback-days", type=int, default=120)
    parser.add_argument("--alignment-horizon-days", type=int, default=365)
    parser.add_argument("--families", nargs="*", help="Optional normalized action families to include.")
    parser.add_argument("--max-cases-per-company", type=int, default=1)
    parser.add_argument("--top-plans", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--strict-evidence", action="store_true")
    parser.add_argument("--precedent-top-k", type=int, default=0)
    parser.add_argument("--planner-random-seed", type=int, default=7)
    parser.add_argument("--limit", type=int, help="Optional row limit before stratified case selection.")
    parser.add_argument("--skip-timeseries", action="store_true")
    parser.add_argument("--skip-macro", action="store_true")
    parser.add_argument("--skip-events", action="store_true")
    parser.add_argument("--skip-peers", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-cache-facts", action="store_true")
    parser.add_argument("--no-cache-events", action="store_true")
    parser.add_argument("--no-cache-timeseries", action="store_true")
    parser.add_argument("--no-cache-ownership", action="store_true")
    parser.add_argument("--no-cache-ratings", action="store_true")
    parser.add_argument("--snapshot-cache-dir", help="Optional persistent cache directory for built historical snapshots.")
    parser.add_argument("--quiet-progress", action="store_true", help="Suppress per-case progress logs on stderr.")
    parser.add_argument("--min-non-missing-core-features", type=int, default=3, help="Minimum non-missing core snapshot features required for a case to be scored.")
    parser.add_argument("--selection-multiplier", type=int, default=10, help="How many candidate historical cases to scan per requested supported case.")
    parser.add_argument("--max-candidate-cases", type=int, help="Absolute cap on candidate historical cases to scan before stopping.")
    parser.add_argument("--no-historical-backfill-mode", action="store_true", help="Respect ingestion/create timestamps instead of historical backfill mode.")
    parser.add_argument(
        "--exclude-report-json",
        nargs="*",
        help="Optional historical evaluation report JSONs whose anchor cases should be excluded from case selection.",
    )
    parser.add_argument(
        "--fixed-cases-json",
        nargs="*",
        help="Optional report/manifest JSONs whose `cases` entries should be replayed exactly instead of reselecting cases.",
    )
    parser.add_argument(
        "--action-support-manifest-path",
        default="./configs/action_data_support_manifest.json",
        help="Optional action-data support manifest. If missing or stale, evaluation recomputes support from outcomes.",
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md")
    return parser.parse_args()


