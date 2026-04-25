#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def _extract_metric_value(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("value")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact taxonomy lookup from keyed snapshot JSON files.")
    parser.add_argument("--snapshot-root")
    parser.add_argument("--snapshot-catalog-path")
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def _parse_snapshot_taxonomy(path: Path) -> Dict[str, str] | None:
    company_id = path.stem.split("company_id=", 1)[-1].strip()
    if not company_id:
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    features = payload.get("features") if isinstance(payload, dict) else None
    features = features if isinstance(features, dict) else {}
    sector_name = str(_extract_metric_value(features['taxonomy.sector']) or "").strip()
    subsector_name = str(_extract_metric_value(features.get("taxonomy.subsector")) or "").strip()
    if not sector_name and not subsector_name:
        return None
    return {
        "company_id": company_id,
        "taxonomy.sector": sector_name,
        "taxonomy.subsector": subsector_name,
    }


