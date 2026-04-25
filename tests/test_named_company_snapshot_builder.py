from __future__ import annotations

import json
from pathlib import Path

from src.named_company_snapshot_builder import (
    build_named_company_snapshots,
    required_fact_years,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_bytes(path: Path, payload: bytes = b"par1") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


