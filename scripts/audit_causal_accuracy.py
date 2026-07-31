#!/usr/bin/env python
"""Audit causal coverage/strict-gate quality from completed recommendation runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


_CAUSAL_DRIVER_NAMES = {
    "causal_model_blend_weight",
    "causal_model_quality",
    "causal_model_support_score",
    "causal_model_mode",
}


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
    except Exception:
        return float(default)
    if out != out:  # NaN
        return float(default)
    return float(out)


def _load_json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(path.read_text()) or {})


