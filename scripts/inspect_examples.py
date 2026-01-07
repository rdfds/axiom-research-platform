#!/usr/bin/env python3
"""Print a compact summary of the committed Axiom examples."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: str) :
    return json.loads((ROOT / path).read_text())


