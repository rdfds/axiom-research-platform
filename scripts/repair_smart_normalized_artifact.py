#!/usr/bin/env python3
"""Refresh smart-normalized metrics in an already-materialized artifact."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backfill_smart_normalized_metrics_v1 as smart  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--metric-registry-path", required=True)
    parser.add_argument("--component-policy-path", required=True)
    parser.add_argument("--source-precedence-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    registry_path = Path(args.metric_registry_path)
    component_policy_path = Path(args.component_policy_path)
    source_precedence_path = Path(args.source_precedence_path)

    registry = json.loads(registry_path.read_text())
    json.loads(component_policy_path.read_text())
    json.loads(source_precedence_path.read_text())

    computed_at = smart._now_iso()
    provenance_sources = [
        str(registry_path),
        str(component_policy_path),
        str(source_precedence_path),
    ]
    counters: Counter[str] = Counter()

    with artifact_path.open() as src, out_path.open("w") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            row = smart.materialize_smart_metrics_for_row(
                row=row,
                registry=registry,
                computed_at=computed_at,
                provenance_sources=provenance_sources,
            )
            for metric_name in smart.SMART_METRIC_NAMES:
                counters[f"{metric_name}:{row['features'][metric_name]['support_mode']}"] += 1
            dst.write(json.dumps(row) + "\n")

    if args.summary_out:
        summary = {}
        for metric_name in smart.SMART_METRIC_NAMES:
            summary[metric_name] = {
                "exact": counters[f"{metric_name}:exact"],
                "proxy_missing_component": counters[f"{metric_name}:proxy_missing_component"],
                "unsupported": counters[f"{metric_name}:unsupported"],
            }
        Path(args.summary_out).write_text(json.dumps(summary, indent=2))

    print(f"Repaired smart-normalized metrics -> {out_path}")


if __name__ == "__main__":
    main()
