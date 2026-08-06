#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.named_company_snapshot_builder import (
    DEFAULT_FACTS_PATH,
    DEFAULT_FRESH_SNAPSHOT_ROOT,
    build_named_company_snapshots,
)
from src.named_company_metric_benchmarks import DEFAULT_TARGETS_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fresh named-company snapshots with the current market metric engine.")
    parser.add_argument("--targets-path", default=str(DEFAULT_TARGETS_PATH), help="Path to named-company target config.")
    parser.add_argument("--snapshot-root", default=str(DEFAULT_FRESH_SNAPSHOT_ROOT), help="Output keyed snapshot root.")
    parser.add_argument("--facts-path", default=str(DEFAULT_FACTS_PATH), help="Facts directory or parquet file.")
    parser.add_argument("--facts-lookback-years", type=int, default=5, help="Inclusive fact-year window ending at the as-of year.")
    parser.add_argument("--case-id", action="append", default=[], help="Optional case id filter. Repeatable.")
    parser.add_argument("--force", action="store_true", help="Rebuild snapshots even if a readable output already exists.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose builder logging.")
    parser.add_argument("--out", help="Optional JSON output path.")
    args = parser.parse_args()

    report = build_named_company_snapshots(
        args.targets_path,
        snapshot_root=args.snapshot_root,
        facts_path=args.facts_path,
        facts_lookback_years=args.facts_lookback_years,
        case_ids=args.case_id or None,
        force=args.force,
        debug=args.debug,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
