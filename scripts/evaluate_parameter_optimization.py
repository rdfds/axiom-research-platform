#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.parameter_backtest import build_parameter_backtest_report, render_parameter_backtest_markdown


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate historical support for parameter optimization outputs.")
    parser.add_argument("--runs-roots", nargs="+", required=True, help="One or more recommendation runs roots.")
    parser.add_argument("--snapshot-root", required=True, help="Snapshot root used to rebuild dossiers.")
    parser.add_argument("--outcomes-path", required=True, help="Historical outcomes parquet with normalized action columns.")
    parser.add_argument("--run-ids-file", help="Optional file of run IDs to restrict evaluation.")
    parser.add_argument("--out-json", required=True, help="Output JSON path.")
    parser.add_argument("--out-md", help="Optional output Markdown path.")
    parser.add_argument("--review-count", type=int, default=25, help="How many cases to include in the review queue.")
    parser.add_argument("--limit", type=int, help="Optional limit on runs analyzed.")
    parser.add_argument("--min-bucket-samples", type=int, default=25, help="Minimum historical rows required per size bucket.")
    return parser.parse_args()


