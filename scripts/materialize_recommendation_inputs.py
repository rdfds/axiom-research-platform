#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional


def _copy_file(src: Path, dst: Path) -> Path:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _materialize_keyed_snapshots(snapshot_root: Path, as_of: str, companies: List[str], dest_root: Path) -> Path:
    dest_snapshot_root = dest_root / "snapshot_root"
    dest_keyed_dir = dest_snapshot_root / "keyed" / f"as_of_date={as_of}"
    dest_keyed_dir.mkdir(parents=True, exist_ok=True)
    for company_id in companies:
        src = snapshot_root / "keyed" / f"as_of_date={as_of}" / f"company_id={company_id}.json"
        dst = dest_keyed_dir / f"company_id={company_id}.json"
        _copy_file(src, dst)
    return dest_snapshot_root


def _maybe_copy(path_value: Optional[str], dest_dir: Path) -> Optional[str]:
    if not path_value:
        return None
    src = Path(path_value)
    dst = dest_dir / src.name
    return str(_copy_file(src, dst))


