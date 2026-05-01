#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metric_goldens import DEFAULT_GOLDENS_PATH, validate_metric_goldens


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate market-metric goldens against the company-state builder.")
    parser.add_argument("--goldens-path", default=str(DEFAULT_GOLDENS_PATH), help="Path to metric goldens JSON file.")
    parser.add_argument("--case-id", action="append", default=[], help="Optional case id filter. Repeatable.")
    parser.add_argument("--out", help="Optional JSON output path.")
    args = parser.parse_args()

    report = validate_metric_goldens(args.goldens_path, case_ids=args.case_id or None)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered)
    print(rendered)
    return 0 if report["summary"]["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
