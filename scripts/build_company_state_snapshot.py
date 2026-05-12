import argparse
import faulthandler
import json
import os
import sys
import time
import zlib
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Allow trace-hang before heavy imports (set TRACE_HANG=1)
if os.environ.get("TRACE_HANG") == "1":
    faulthandler.dump_traceback_later(30, repeat=True)

import pandas as pd
import duckdb

from src.company_state_builder import CompanyStateBuilder
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
            # allow comma-separated or whitespace-separated
            parts = [x.strip() for x in line.replace(",", " ").split() if x.strip()]
            ids.extend(parts)
    return ids

def _apply_shard(ids: List[str], shard: Optional[int], shard_count: Optional[int]) -> List[str]:
    if shard is None or shard_count is None:
        return ids
    if shard < 0 or shard_count <= 0 or shard >= shard_count:
        raise SystemExit("--shard must be in [0, shard_count)")
    out: List[str] = []
    for cid in ids:
        h = zlib.crc32(cid.encode("utf-8")) % shard_count
        if h == shard:
            out.append(cid)
    return out

def _count_jsonl_rows(path: Path) -> int:
    n = 0
    with path.open("r") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


