#!/usr/bin/env python
"""
Compact press release warehouse files to reduce per-file overhead.

Reads:
  data/warehouse/warehouse_press_releases/year=*/part_*.parquet

Writes:
  data/warehouse/warehouse_press_releases/year=*/part_compact.parquet

Options (env):
  PR_COMPACT_START_YEAR=2000
  PR_COMPACT_END_YEAR=YYYY
  PR_COMPACT_BACKUP=1      (move old part_*.parquet to backup)
  PR_COMPACT_DELETE=0      (delete old part_*.parquet instead of backup)
  PR_COMPACT_SKIP_EXISTING=1 (skip if part_compact already exists)
  PR_COMPACT_LOG_EVERY=50  (file progress per year)
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


DATA_DIR = Path(__file__).parent.parent / "data"
WAREHOUSE_DIR = DATA_DIR / "warehouse"

PR_COMPACT_START_YEAR = int(os.getenv("PR_COMPACT_START_YEAR", "2000"))
PR_COMPACT_END_YEAR = int(os.getenv("PR_COMPACT_END_YEAR", datetime.utcnow().year))
PR_COMPACT_BACKUP = os.getenv("PR_COMPACT_BACKUP", "1") == "1"
PR_COMPACT_DELETE = os.getenv("PR_COMPACT_DELETE", "0") == "1"
PR_COMPACT_SKIP_EXISTING = os.getenv("PR_COMPACT_SKIP_EXISTING", "1") == "1"
PR_COMPACT_LOG_EVERY = int(os.getenv("PR_COMPACT_LOG_EVERY", "50"))


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def normalize_list(value: Optional[object]) -> List[str]:
    if value is None:
        return []
    try:
        import numpy as np
    except Exception:
        np = None  # type: ignore

    if np is not None and isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        return [str(v) for v in value.tolist() if v is not None]
    if isinstance(value, pd.Series):
        if value.empty:
            return []
        return [str(v) for v in value.tolist() if v is not None]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v is not None and not (isinstance(v, float) and pd.isna(v))]
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    return [str(value)]


def iter_year_dirs(base: Path) -> Iterable[Path]:
    for year_dir in sorted(base.glob("year=*")):
        try:
            year = int(year_dir.name.split("=")[1])
        except Exception:
            continue
        if year < PR_COMPACT_START_YEAR or year > PR_COMPACT_END_YEAR:
            continue
        yield year_dir


def read_parts(parts: List[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for idx, path in enumerate(parts, start=1):
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            log(f"  Skipping unreadable parquet {path}: {exc}")
            continue
        if df.empty:
            continue
        frames.append(df)
        if PR_COMPACT_LOG_EVERY and idx % PR_COMPACT_LOG_EVERY == 0:
            log(f"  Read {idx}/{len(parts)} files")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    base = WAREHOUSE_DIR / "warehouse_press_releases"
    if not base.exists():
        raise FileNotFoundError(f"Missing press releases warehouse: {base}")

    backup_root = None
    if PR_COMPACT_BACKUP and not PR_COMPACT_DELETE:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_root = WAREHOUSE_DIR / "_backup" / "warehouse_press_releases" / stamp
        backup_root.mkdir(parents=True, exist_ok=True)

    for year_dir in iter_year_dirs(base):
        parts = sorted(year_dir.glob("part_*.parquet"))
        if not parts:
            continue

        compact_path = year_dir / "part_compact.parquet"
        compact_exists = compact_path.exists()
        compact_mtime = compact_path.stat().st_mtime if compact_exists else None
        non_compact_all = [p for p in parts if p.name != "part_compact.parquet"]

        # If a compact file exists, only include new parts since it was written.
        if compact_exists:
            if compact_mtime is not None:
                new_parts = [p for p in non_compact_all if p.stat().st_mtime > compact_mtime]
            else:
                new_parts = non_compact_all

            if PR_COMPACT_SKIP_EXISTING and not new_parts:
                log(f"Skipping {year_dir.name}: compact file already exists and no new parts.")
                continue

            # Rebuild using the existing compact + only new parts (avoids duplication).
            read_parts_list = [compact_path] + new_parts
            cleanup_parts = non_compact_all
        else:
            read_parts_list = parts
            cleanup_parts = parts

        log(f"Compacting {year_dir.name}: {len(read_parts_list)} files")
        t0 = time.perf_counter()
        df = read_parts(read_parts_list)
        if df.empty:
            log(f"  No rows found for {year_dir.name}. Skipping.")
            continue

        for col in ("quality_flags", "upstream_version_ids"):
            if col in df.columns:
                df[col] = df[col].apply(normalize_list)

        if "text" in df.columns:
            df["text"] = df["text"].where(df["text"].notna(), None)

        temp_path = compact_path.with_suffix(".parquet.tmp")
        df.to_parquet(temp_path, index=False)
        temp_path.replace(compact_path)
        t1 = time.perf_counter()
        log(f"  Wrote {len(df):,} rows -> {compact_path.name} in {t1 - t0:.1f}s")

        if PR_COMPACT_DELETE:
            for path in cleanup_parts:
                try:
                    path.unlink()
                except Exception as exc:
                    log(f"  Failed to delete {path}: {exc}")
        elif backup_root is not None:
            year_backup = backup_root / year_dir.name
            year_backup.mkdir(parents=True, exist_ok=True)
            for path in cleanup_parts:
                try:
                    shutil.move(str(path), str(year_backup / path.name))
                except Exception as exc:
                    log(f"  Failed to move {path}: {exc}")

    log("Done.")


if __name__ == "__main__":
    main()
