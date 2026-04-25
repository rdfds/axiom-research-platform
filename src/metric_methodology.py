from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json


ROOT = Path(__file__).resolve().parents[1]


def _default_methodology_registry_path() -> Path:
    env = str(os.environ.get("AXIOM_METHODOLOGY_REGISTRY_PATH", "") or "").strip()
    if env:
        return Path(env)
    return ROOT / "configs" / "metric_methodologies" / "consumer_industrials_canonical_registry_v1.json"


