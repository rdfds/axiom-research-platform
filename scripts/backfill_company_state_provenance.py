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


