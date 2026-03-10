#!/usr/bin/env python
"""
Build master datasets (best/largest per type) and a Russell 3000 proxy universe.

Outputs:
  data/curated/universe_r3000_proxy.parquet
  data/curated/corporate_actions_master.parquet
  data/curated/prices_master.parquet
  data/curated/prices_master_full.parquet
  data/curated/fundamentals_master.parquet
  data/curated/buybacks_master.parquet
  data/curated/mna_master.parquet
  data/curated/master_summary.csv
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
import pyarrow.dataset as ds

try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None


DATA_DIR = Path(__file__).parent.parent / "data"
CRSP_DIR = DATA_DIR / "wrds" / "crsp"
CURATED_DIR = DATA_DIR / "curated"
CURATED_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(pattern: str):
    files = list(CRSP_DIR.glob(pattern))
    if not files:
        return None
    return ds.dataset(files, format="parquet")


def build_r3000_proxy(max_rank: Optional[int] = 3000):
    dataset = load_dataset("msf_*.parquet")
    if dataset is None:
        raise FileNotFoundError("CRSP msf files not found. Run scripts/20_pull_crsp_stock_data.py first.")

    if max_rank and max_rank > 0:
        table = dataset.to_table(columns=["permno", "permco", "date", "prc", "shrout"])
        df = table.to_pandas()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["prc"] = pd.to_numeric(df["prc"], errors="coerce")
        df["shrout"] = pd.to_numeric(df["shrout"], errors="coerce")
        df["mktcap"] = df["prc"].abs() * df["shrout"]

        df = df.dropna(subset=["date", "permno"])
        df = df.dropna(subset=["mktcap"])
        df = df.sort_values(["date", "mktcap"], ascending=[True, False])
        df["rank"] = df.groupby("date")["mktcap"].rank(method="first", ascending=False)
        universe = df[df["rank"] <= max_rank].copy()
    else:
        # Use msenames to include all CRSP-listed permno-months (not just msf coverage).
        names_dataset = load_dataset("msenames_*.parquet")
        if names_dataset is None:
            raise FileNotFoundError("CRSP msenames files not found. Run scripts/20_pull_crsp_stock_data.py first.")
        if duckdb is None:
            raise RuntimeError("duckdb is required to build full universe from msenames.")
        pattern = str(CRSP_DIR / "msenames_*.parquet")
        con = duckdb.connect(database=":memory:")
        query = f"""
        WITH names AS (
            SELECT
                permno,
                CAST(namedt AS DATE) AS namedt,
                CAST(nameendt AS DATE) AS nameendt
            FROM read_parquet('{pattern}')
            WHERE namedt IS NOT NULL AND nameendt IS NOT NULL
        ),
        expanded AS (
            SELECT
                permno,
                (date_trunc('month', gs.value) + INTERVAL '1 month' - INTERVAL '1 day')::DATE AS date
            FROM names
            CROSS JOIN generate_series(
                date_trunc('month', namedt),
                date_trunc('month', nameendt),
                INTERVAL '1 month'
            ) AS gs(value)
        )
        SELECT DISTINCT permno, date
        FROM expanded
        """
        universe = con.execute(query).df()

    # Ensure uniqueness per permno-month to avoid merge inflation downstream.
    universe = universe.drop_duplicates(subset=["date", "permno"])

    out_path = CURATED_DIR / "universe_r3000_proxy.parquet"
    universe.to_parquet(out_path, index=False)
    return universe


