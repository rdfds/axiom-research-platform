#!/usr/bin/env python
"""
Backfill macro features onto action_outcomes via merge_asof.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.config import load_config


DATA_DIR = Path(__file__).parent.parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-path", default=str(DATA_DIR / "curated" / "action_outcomes.parquet"))
    parser.add_argument("--out-path", default=None)
    parser.add_argument("--macro-path", default=str(DATA_DIR / "warehouse" / "warehouse_macro.parquet"))
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    out_path = Path(args.out_path) if args.out_path else in_path

    config = load_config(args.config)
    macro_series: Dict[str, str] = config.get("macro_series", {})
    if not macro_series:
        raise RuntimeError("No macro_series found in config.")

    print("[backfill_macro] loading action_outcomes...", flush=True)
    df = pd.read_parquet(in_path)
    if df.empty:
        raise RuntimeError("No rows in action_outcomes.")
    df["action_date"] = pd.to_datetime(df["action_date"], errors="coerce").astype("datetime64[ns]")
    df = df.sort_values("action_date")

    print("[backfill_macro] loading macro table...", flush=True)
    macro = pd.read_parquet(Path(args.macro_path), columns=["entity_id", "event_time", "value"])
    macro["event_time"] = pd.to_datetime(macro["event_time"], errors="coerce").astype("datetime64[ns]")
    macro = macro.sort_values("event_time")

    for name, series_id in macro_series.items():
        print(f"[backfill_macro] merging series {name} ({series_id})...", flush=True)
        m = macro[macro["entity_id"] == series_id][["event_time", "value"]].copy()
        if m.empty:
            df[f"macro_{name}"] = None
            continue
        m = m.rename(columns={"event_time": "macro_time", "value": f"macro_{name}"})
        m = m.sort_values("macro_time")
        df = pd.merge_asof(
            df,
            m,
            left_on="action_date",
            right_on="macro_time",
            direction="backward",
        )
        df = df.drop(columns=["macro_time"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[backfill_macro] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
