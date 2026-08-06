from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "pipeline_v1.json"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Missing pipeline config: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)
