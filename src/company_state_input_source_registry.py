from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]


def _default_company_state_input_source_registry_path() -> Path:
    env = str(os.environ.get("AXIOM_INPUT_SOURCE_REGISTRY_PATH", "") or "").strip()
    if env:
        return Path(env)
    return ROOT / "configs" / "metric_methodologies" / "company_state_input_source_registry_v1.json"

