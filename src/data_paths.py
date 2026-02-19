from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data"
DATA_ROOT_ENV = "AXIOM_DATA_ROOT"
COMPANYFACTS_ROOT_ENV = "AXIOM_COMPANYFACTS_ROOT"


def configured_data_root() :
    raw = str(os.environ.get(DATA_ROOT_ENV, "") or "").strip()
    if not raw:
        return DEFAULT_DATA_ROOT
    return Path(raw).expanduser()


def _absolute_no_symlink(path: Path) -> Path:
    return Path(os.path.abspath(str(path.expanduser())))


