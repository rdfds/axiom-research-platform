#!/usr/bin/env python
"""Create and persist a RecommendationRun."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_json_arg(raw: str | None, file_path: str | None) -> dict | None:
    if raw and file_path:
        raise ValueError("Provide only one of inline JSON or file path")
    if raw:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("JSON argument must decode to object")
        return obj
    if file_path:
        obj = json.loads(Path(file_path).read_text())
        if not isinstance(obj, dict):
            raise ValueError("JSON file must decode to object")
        return obj
    return None


