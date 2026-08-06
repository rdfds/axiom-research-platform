#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from src.dossier_eval import build_dossier_eval_report, render_dossier_eval_markdown


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
    parser = argparse.ArgumentParser(description="Evaluate board-ready dossier quality across recommendation runs.")
    parser.add_argument("--runs-roots", nargs="+", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--run-ids-file")
    parser.add_argument("--review-count", type=int, default=50)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-postures-json")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md")
    args = parser.parse_args()

    run_ids = _parse_run_ids(Path(args.run_ids_file)) if args.run_ids_file else None
    expected_postures = json.loads(Path(args.expected_postures_json).read_text()) if args.expected_postures_json else None
    report = build_dossier_eval_report(
        runs_roots=args.runs_roots,
        snapshot_root=args.snapshot_root,
        run_ids=run_ids,
        review_count=args.review_count,
        limit=args.limit,
        expected_postures=expected_postures,
    )

    out_json = Path(args.out_json)
    out_json.write_text(json.dumps(report, indent=2))

    if args.out_md:
        out_md = Path(args.out_md)
        out_md.write_text(render_dossier_eval_markdown(report))

    print(
        json.dumps(
            {
                "ok": True,
                "out_json": str(out_json),
                "out_md": str(args.out_md) if args.out_md else None,
                "runs_analyzed": report.get("runs_analyzed", 0),
                "review_queue": len(report.get("review_queue", []) or []),
                "heuristic_overall_mean": (report.get("aggregate", {}) or {}).get("heuristic_overall_mean"),
            }
        )
    )


if __name__ == "__main__":
    main()
