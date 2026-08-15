#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.head_to_head_eval import render_head_to_head_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a head-to-head benchmark report into a human review markdown.")
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    report = json.loads(Path(args.report_json).read_text())
    out_md = Path(args.out_md)
    out_md.write_text(render_head_to_head_markdown(report))
    print(
        json.dumps(
            {
                "ok": True,
                "report_json": str(args.report_json),
                "out_md": str(out_md),
                "runs_analyzed": report.get("runs_analyzed", 0),
            }
        )
    )


if __name__ == "__main__":
    main()
