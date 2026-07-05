#!/usr/bin/env python
"""Benchmark targeted causal routing on an existing recommendation run."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))


def _default_model_path() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    return str(repo_root / "data" / "models" / "causal_impact_model_v5_5_hybrid.json")


DEFAULT_PRESET: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("platform_acquisition", ("mna.platform_acquisition",)),
    ("tuck_in_acquisition", ("mna.tuck_in_acquisition",)),
    ("special_dividend", ("capital_return.special_dividend",)),
    ("dividend_initiate", ("capital_return.dividend_initiate",)),
)


