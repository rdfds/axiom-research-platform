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


def main() -> None:
    args = _parse_args()
    run_ids = None
    if args.run_ids_file:
        run_ids = [line.strip() for line in Path(args.run_ids_file).read_text().splitlines() if line.strip()]

    report = build_parameter_backtest_report(
        runs_roots=args.runs_roots,
        snapshot_root=args.snapshot_root,
        outcomes_path=args.outcomes_path,
        run_ids=run_ids,
        review_count=args.review_count,
        limit=args.limit,
        min_bucket_samples=args.min_bucket_samples,
    )
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))

    out_md = None
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(render_parameter_backtest_markdown(report))

    print(
        json.dumps(
            {
                "ok": True,
                "out_json": str(out_json),
                "out_md": str(out_md) if out_md else None,
                "runs_analyzed": int(report.get("runs_analyzed", 0) or 0),
                "historical_coverage_rate": float(((report.get("aggregate", {}) or {}).get("historical_coverage_rate", 0.0) or 0.0)),
                "mean_alignment_score": float(((report.get("aggregate", {}) or {}).get("mean_alignment_score", 0.0) or 0.0)),
            }
        )
    )


if __name__ == "__main__":
    main()
