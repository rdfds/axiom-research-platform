#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.replay_snapshot_enrichment import enrich_snapshot_with_revenue_growth_inputs

DEFAULT_ENTITY_IDENTIFIER_PATH = ROOT / "out/manual_replay_bundle_20260405_localized/inputs/entity_identifier.parquet"
DEFAULT_CRSP_DAILY_ROOT = ROOT / "data/wrds/crsp"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich replay snapshots with as-of-safe matching inputs (revenue growth plus derived buyback/dividend support metrics)."
    )
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--snapshot-cache-root", required=True)
    parser.add_argument("--companyfacts-root", required=True)
    parser.add_argument("--entity-identifier-path", default=str(DEFAULT_ENTITY_IDENTIFIER_PATH))
    parser.add_argument("--crsp-daily-root", default=str(DEFAULT_CRSP_DAILY_ROOT))
    parser.add_argument("--crsp-market-cache-path", default="")
    parser.add_argument("--summary-path", default="")
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _modern_snapshot_path(snapshot_cache_root: Path, *, company_id: str, as_of_time: str) -> Path:
    as_of_date = str(as_of_time).split("T", 1)[0]
    return snapshot_cache_root / f"as_of_date={as_of_date}" / f"company_id={company_id}.json"


def _legacy_snapshot_path(snapshot_cache_root: Path, *, company_id: str, as_of_time: str) -> Path:
    ts = str(as_of_time).replace("-", "").replace(":", "")
    if ts.endswith("+0000"):
        ts = ts[:-5]
    if ts.endswith("+00:00"):
        ts = ts[:-6]
    ts = ts.replace("T", "T").replace("Z", "")
    date_part, _, time_part = str(as_of_time).partition("T")
    time_part = (time_part or "00:00:00+00:00").split("+", 1)[0].replace(":", "")
    return snapshot_cache_root / f"company_id={company_id}" / f"snapshot_as_of={date_part.replace('-', '')}T{time_part}Z.json"


def _snapshot_path(snapshot_cache_root: Path, *, company_id: str, as_of_time: str) :
    modern = _modern_snapshot_path(snapshot_cache_root, company_id=company_id, as_of_time=as_of_time)
    if modern.exists():
        return modern
    legacy = _legacy_snapshot_path(snapshot_cache_root, company_id=company_id, as_of_time=as_of_time)
    if legacy.exists():
        return legacy
    return modern


