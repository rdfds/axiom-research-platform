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


