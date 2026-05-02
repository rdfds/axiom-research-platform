#!/usr/bin/env python
"""
Fast delta updater for ownership concentration + issuer rating features.

Use this to enrich an existing CompanyState snapshot JSONL without rerunning
full feature computation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb
import numpy as np


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


