#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from src.head_to_head_eval import build_head_to_head_report, export_blinded_packets


def _parse_run_ids(path: Path | None) -> List[str] | None:
    if path is None:
        return None
    run_ids: List[str] = []
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text:
            continue
        run_ids.append(text.split()[-1])
    return run_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Build blinded model-vs-baseline benchmark packets.")
    parser.add_argument("--runs-roots", nargs="+", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--realized-outcomes-path")
    parser.add_argument("--alignment-horizon-days", type=int, default=540)
    parser.add_argument("--run-ids-file")
    parser.add_argument("--review-count", type=int, default=50)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--packets-out-dir")
    parser.add_argument("--answer-key-out")
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    run_ids = _parse_run_ids(Path(args.run_ids_file)) if args.run_ids_file else None
    report = build_head_to_head_report(
        runs_roots=args.runs_roots,
        snapshot_root=args.snapshot_root,
        baseline_dir=args.baseline_dir,
        realized_outcomes_path=args.realized_outcomes_path,
        alignment_horizon_days=args.alignment_horizon_days,
        run_ids=run_ids,
        review_count=args.review_count,
        limit=args.limit,
    )
    out_json = Path(args.out_json)
    out_json.write_text(json.dumps(report, indent=2))
    export_summary = None
    if args.packets_out_dir:
        export_summary = export_blinded_packets(
            report=report,
            packets_out_dir=args.packets_out_dir,
            answer_key_out=args.answer_key_out,
        )
    print(
        json.dumps(
            {
                "ok": True,
                "out_json": str(out_json),
                "runs_analyzed": report.get("runs_analyzed", 0),
                "review_queue": len(report.get("review_queue", []) or []),
                "model_win_rate": (report.get("aggregate", {}) or {}).get("model_win_rate"),
                "baseline_win_rate": (report.get("aggregate", {}) or {}).get("baseline_win_rate"),
                "sign_test_p_value": (report.get("aggregate", {}) or {}).get("sign_test_p_value"),
                "ex_post_coverage_rate": ((report.get("aggregate", {}) or {}).get("ex_post", {}) or {}).get("coverage_rate"),
                "exported_packets": (export_summary or {}).get("exported_packets"),
            }
        )
    )


if __name__ == "__main__":
    main()
