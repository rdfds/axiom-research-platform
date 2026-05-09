#!/usr/bin/env python3
"""Build the committed Home Depot market-expectations sample as static HTML.

The original Axiom workspace renders a richer interactive surface from local
artifacts. This public builder keeps the same evidence shape but uses only the
small committed sample so anyone can reproduce the view without private data.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path


def _load(build_dir: Path, name: str) -> dict:
    return json.loads((build_dir / name).read_text())


