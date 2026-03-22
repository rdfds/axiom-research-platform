#!/usr/bin/env python3
"""Repair cash-flow-based metrics in a materialized company-state artifact."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

try:
    from repair_operating_history_artifact import (
        _companyfacts_priority_ttm,
        _load_companyfacts,
    )
except Exception:  # noqa: BLE001
    from scripts.repair_operating_history_artifact import (  # type: ignore
        _companyfacts_priority_ttm,
        _load_companyfacts,
    )


REPAIR_METRICS = [
    "market.fcf_yield",
    "operating.fcf_conversion",
]

OPERATING_CASH_FLOW_TTM_CONCEPTS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
]
CAPEX_TTM_CONCEPTS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PurchaseOfPropertyPlantAndEquipment",
    "PropertyPlantAndEquipmentAdditions",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True, help="Input company-state JSONL artifact")
    parser.add_argument("--companyfacts-root", required=True, help="SEC companyfacts folder")
    parser.add_argument("--out", required=True, help="Output repaired JSONL artifact")
    parser.add_argument("--summary-out", help="Optional summary JSON")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


