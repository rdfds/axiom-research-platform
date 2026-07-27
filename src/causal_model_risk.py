"""Causal model risk diagnostics for run-level governance.

This module summarizes causal coverage, support, quality, OOS rates, and
fallback behavior from FeasibilityResults action-candidate payloads.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


