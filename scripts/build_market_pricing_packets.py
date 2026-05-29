#!/usr/bin/env python3
"""Build human-readable scorecard packets from the market-pricing scorecard artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True, help="Scorecard JSONL artifact")
    parser.add_argument("--out", required=True, help="Output JSON packet path")
    parser.add_argument("--companyfacts-root", help="Optional SEC companyfacts directory for issuer names")
    parser.add_argument("--limit", type=int, default=12, help="Rows per packet")
    return parser.parse_args()


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _node_value(features: Dict[str, Any], name: str) -> float | None:
    node = features.get(name) or {}
    value = node.get("value")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return None


