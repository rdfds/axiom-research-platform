from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.named_company_metric_benchmarks import generate_named_company_metric_benchmarks


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


