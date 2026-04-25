#!/usr/bin/env python
"""
Fast post-processing updater to close remaining spec gaps without re-running
full snapshot computation.

Adds/refreshes:
  - market.ev_ebitda_vs_peer_z
  - market.fcf_yield_percentile_peers
  - operating.guidance_revision_direction
  - operating.cyclicality_macro_beta_proxy
  - operating.revenue_sensitivity_proxy
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        x = float(v)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None


