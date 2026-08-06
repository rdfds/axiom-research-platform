from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data"
DATA_ROOT_ENV = "AXIOM_DATA_ROOT"
COMPANYFACTS_ROOT_ENV = "AXIOM_COMPANYFACTS_ROOT"


def configured_data_root() -> Path:
    raw = str(os.environ.get(DATA_ROOT_ENV, "") or "").strip()
    if not raw:
        return DEFAULT_DATA_ROOT
    return Path(raw).expanduser()


def _absolute_no_symlink(path: Path) -> Path:
    return Path(os.path.abspath(str(path.expanduser())))


def _relative_under_default_data_root(path: Path) -> Optional[Path]:
    try:
        return _absolute_no_symlink(path).relative_to(_absolute_no_symlink(DEFAULT_DATA_ROOT))
    except Exception:
        return None


def resolve_data_path(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    data_root = configured_data_root()

    if candidate.is_absolute():
        rel = _relative_under_default_data_root(candidate)
        if rel is not None:
            return data_root / rel
        return candidate

    parts = candidate.parts
    if parts[:1] == ("data",):
        rel = Path(*parts[1:]) if len(parts) > 1 else Path()
        return data_root / rel

    return candidate


def resolve_companyfacts_root(path: Path | str | None = None) -> Optional[Path]:
    raw_override = str(os.environ.get(COMPANYFACTS_ROOT_ENV, "") or "").strip()
    if raw_override:
        return Path(raw_override).expanduser()
    if path is None:
        return None
    return resolve_data_path(path)
