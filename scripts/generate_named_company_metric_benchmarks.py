#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.named_company_metric_benchmarks import (
    DEFAULT_FUNDAMENTALS_PATH,
    DEFAULT_SNAPSHOT_ROOT,
    DEFAULT_TARGETS_PATH,
    generate_named_company_metric_benchmarks,
)


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate named-company metric benchmark packets.')
    parser.add_argument('--targets-path', default=str(DEFAULT_TARGETS_PATH), help='Path to named company target config.')
    parser.add_argument('--snapshot-root', default=str(DEFAULT_SNAPSHOT_ROOT), help='Snapshot root to read benchmark packets from.')
    parser.add_argument('--fundamentals-path', default=str(DEFAULT_FUNDAMENTALS_PATH), help='Fallback fundamentals parquet for taxonomy context.')
    parser.add_argument('--case-id', action='append', default=[], help='Optional case id filter. Repeatable.')
    parser.add_argument('--out', help='Optional JSON output path.')
    args = parser.parse_args()

    report = generate_named_company_metric_benchmarks(
        args.targets_path,
        snapshot_root=args.snapshot_root,
        fundamentals_path=args.fundamentals_path,
        case_ids=args.case_id or None,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered)
    print(rendered)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
