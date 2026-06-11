#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) :
    return json.loads(path.read_text())


def _load_existing_runtime_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = _load_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


