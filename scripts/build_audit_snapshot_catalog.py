#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FULL_SNAPSHOT_PATH = (
    REPO_ROOT
    / "out/materialized_feedback_20260405/company_state_snapshots_asof=2024-12-31.input_layer_v1_smart_normalized_with_sec.feedback_pipeline.full_inputs_v3.jsonl.gz"
)
DEFAULT_REPLAY_SNAPSHOT_ROOT = (
    REPO_ROOT / "out/manual_replay_bundle_20260405_localized/reports/snapshot_cache/keyed"
)
DEFAULT_OUT_PATH = (
    REPO_ROOT
    / "out/materialized_feedback_20260405/company_state_snapshots_audit_catalog.asof_safe_enriched_v1.jsonl.gz"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an audit-friendly snapshot catalog from durable and replay-safe sources.")
    parser.add_argument("--full-snapshot-path", default=str(DEFAULT_FULL_SNAPSHOT_PATH))
    parser.add_argument("--replay-snapshot-root", default=str(DEFAULT_REPLAY_SNAPSHOT_ROOT))
    parser.add_argument("--out-path", default=str(DEFAULT_OUT_PATH))
    return parser.parse_args()


def _iter_full_rows(snapshot_path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(snapshot_path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            row["snapshot_catalog_source"] = "full_inputs_v3"
            yield row


def _iter_replay_rows(snapshot_root: Path) -> Iterable[Dict[str, Any]]:
    for path in sorted(snapshot_root.rglob("*.json")):
        row = json.loads(path.read_text())
        row["snapshot_catalog_source"] = "replay_snapshot_cache"
        row["snapshot_catalog_path"] = str(path)
        yield row


def _dedupe_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return str(row.get("company_id") or ""), str(row.get("as_of_time") or "")


def _feature_count(row: Dict[str, Any]) -> int:
    features = row.get("features") or {}
    if not isinstance(features, dict):
        return 0
    return sum(1 for value in features.values() if isinstance(value, dict) and value.get("value") is not None)


def _prefer_row(existing: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    existing_source = str(existing.get("snapshot_catalog_source") or "")
    candidate_source = str(candidate.get("snapshot_catalog_source") or "")
    if existing_source != candidate_source:
        if existing_source == "full_inputs_v3":
            return existing
        if candidate_source == "full_inputs_v3":
            return candidate
    if _feature_count(candidate) > _feature_count(existing):
        return candidate
    return existing


def main() -> None:
    args = _parse_args()
    full_snapshot_path = Path(args.full_snapshot_path)
    replay_snapshot_root = Path(args.replay_snapshot_root)
    out_path = Path(args.out_path)

    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in _iter_full_rows(full_snapshot_path):
        rows[_dedupe_key(row)] = row
    for row in _iter_replay_rows(replay_snapshot_root):
        key = _dedupe_key(row)
        if key in rows:
            rows[key] = _prefer_row(rows[key], row)
        else:
            rows[key] = row

    ordered_rows = sorted(
        rows.values(),
        key=lambda row: (
            str(row.get("company_id") or ""),
            str(row.get("as_of_time") or ""),
        ),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as handle:
        for row in ordered_rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

    summary = {
        "out_path": str(out_path),
        "row_count": len(ordered_rows),
        "full_inputs_v3_rows": sum(1 for row in ordered_rows if row.get("snapshot_catalog_source") == "full_inputs_v3"),
        "replay_snapshot_rows": sum(1 for row in ordered_rows if row.get("snapshot_catalog_source") == "replay_snapshot_cache"),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
