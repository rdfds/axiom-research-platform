#!/usr/bin/env python
"""
Backfill provenance coverage + transform lineage for existing CompanyState JSONL snapshots
without rebuilding features.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_ref(
    artifact_type: str,
    artifact_id: str,
    source: str | None,
    as_of_time: str | None,
) -> Dict[str, Any]:
    return {
        "artifact_type": artifact_type,
        "artifact_id": artifact_id,
        "source": source,
        "published_at": as_of_time,
        "ingested_at": as_of_time,
        "hash": None,
    }


