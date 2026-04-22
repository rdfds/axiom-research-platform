import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.company_state_builder import CompanyStateBuilder
from src.company_state_delta import update_snapshot
from src.company_state_store import SnapshotStore


def _parse_company_ids(arg: Optional[str]) :
    if not arg:
        return []
    if "," in arg:
        return [x.strip() for x in arg.split(",") if x.strip()]
    return [arg.strip()]


def _load_company_ids_file(path: Optional[str]) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--company-ids-file not found: {path}")
    ids: List[str] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [x.strip() for x in line.replace(",", " ").split() if x.strip()]
            ids.extend(parts)
    return ids


