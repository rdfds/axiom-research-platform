#!/usr/bin/env python
"""
Scan parquet files and move any unreadable/corrupt files to a quarantine folder.

Default target:
  data/warehouse/warehouse_press_releases

Usage:
  python -u scripts/41_quarantine_corrupt_parquet.py
  python -u scripts/41_quarantine_corrupt_parquet.py --path data/warehouse/warehouse_prices_daily
  python -u scripts/41_quarantine_corrupt_parquet.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pyarrow.parquet as pq


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def iter_files(base: Path, pattern: str) -> List[Path]:
    return sorted(base.rglob(pattern))


def is_parquet_ok(path: Path) -> bool:
    try:
        if path.stat().st_size == 0:
            return False
        pf = pq.ParquetFile(path)
        _ = pf.metadata
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/warehouse/warehouse_press_releases")
    parser.add_argument("--pattern", default="part_*.parquet")
    parser.add_argument("--quarantine", default="data/warehouse/_corrupt")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    base = Path(args.path)
    if not base.exists():
        raise FileNotFoundError(f"Path not found: {base}")

    quarantine_base = Path(args.quarantine)
    files = iter_files(base, args.pattern)
    log(f"Scanning {len(files):,} files under {base}")

    moved = 0
    checked = 0
    start = time.perf_counter()

    for idx, path in enumerate(files, start=1):
        checked += 1
        ok = is_parquet_ok(path)
        if not ok:
            rel = path.relative_to(base)
            dest = quarantine_base / base.name / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                suffix = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                dest = dest.with_name(dest.stem + f"_{suffix}" + dest.suffix)
            if args.dry_run:
                log(f"[dry-run] Would move corrupt file: {path} -> {dest}")
            else:
                shutil.move(str(path), str(dest))
                log(f"Moved corrupt file: {path} -> {dest}")
            moved += 1
            if args.limit and moved >= args.limit:
                break

        if args.log_every and idx % args.log_every == 0:
            elapsed = time.perf_counter() - start
            log(f"Progress: {idx}/{len(files)} checked | {moved} moved | elapsed {elapsed/60:.1f}m")

    elapsed = time.perf_counter() - start
    log(f"Done. Checked {checked:,} files; moved {moved:,}. Elapsed {elapsed/60:.1f}m")


if __name__ == "__main__":
    main()
