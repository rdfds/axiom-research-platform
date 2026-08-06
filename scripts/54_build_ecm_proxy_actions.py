"""
Build ECM proxy actions from Compustat fundamentals (share count changes).

Output: data/curated/equity_offerings_proxy.parquet

Env vars:
  FUNDAMENTALS_PATH=data/fundamentals_quarterly.parquet
  ECM_PROXY_OUT_PATH=data/curated/equity_offerings_proxy.parquet
  ECM_MIN_SHARE_CHANGE=0.03
  ECM_MAX_SHARE_CHANGE=0.50
  ECM_REQUIRE_PRICE=1
  ECM_USE_RDQ=1
  ECM_MIN_MKT_CAP=0
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DATA_DIR = Path(__file__).parent.parent / "data"
FUND_PATH = Path(os.getenv("FUNDAMENTALS_PATH", DATA_DIR / "fundamentals_quarterly.parquet"))
OUT_PATH = Path(os.getenv("ECM_PROXY_OUT_PATH", DATA_DIR / "curated" / "equity_offerings_proxy.parquet"))

MIN_SHARE_CHANGE = float(os.getenv("ECM_MIN_SHARE_CHANGE", "0.03"))
MAX_SHARE_CHANGE = float(os.getenv("ECM_MAX_SHARE_CHANGE", "0.50"))
REQUIRE_PRICE = os.getenv("ECM_REQUIRE_PRICE", "1") == "1"
USE_RDQ = os.getenv("ECM_USE_RDQ", "1") == "1"
MIN_MKT_CAP = float(os.getenv("ECM_MIN_MKT_CAP", "0"))


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    if not FUND_PATH.exists():
        raise FileNotFoundError(f"Missing fundamentals file: {FUND_PATH}")

    log("Loading Compustat fundamentals...")
    cols = [
        "gvkey",
        "datadate",
        "rdq",
        "cshoq",
        "prccq",
        "mkvaltq",
        "tic",
        "conm",
        "sic",
    ]
    schema_cols = set(pq.ParquetFile(FUND_PATH).schema.names)
    cols = [c for c in cols if c in schema_cols]
    df = pd.read_parquet(FUND_PATH, columns=cols)

    for col in ["cshoq", "prccq", "mkvaltq"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
    if "rdq" in df.columns:
        df["rdq"] = pd.to_datetime(df["rdq"], errors="coerce")

    df = df.sort_values(["gvkey", "datadate"])
    df["cshoq_prev"] = df.groupby("gvkey")["cshoq"].shift(1)

    df["share_change"] = df["cshoq"] - df["cshoq_prev"]
    df["share_change_pct"] = df["share_change"] / df["cshoq_prev"]

    mask = df["share_change_pct"].notna() & (df["share_change_pct"] >= MIN_SHARE_CHANGE)
    if MAX_SHARE_CHANGE > 0:
        mask &= df["share_change_pct"] <= MAX_SHARE_CHANGE
    if REQUIRE_PRICE and "prccq" in df.columns:
        mask &= df["prccq"].notna()
    if MIN_MKT_CAP > 0 and "mkvaltq" in df.columns:
        mask &= df["mkvaltq"].notna() & (df["mkvaltq"] >= MIN_MKT_CAP)

    out = df.loc[mask].copy()
    out["action_date"] = out["rdq"] if USE_RDQ and "rdq" in out.columns else out["datadate"]
    out["action_date"] = out["action_date"].fillna(out["datadate"])

    # Estimate issuance amount using price * share change
    out["amount"] = np.where(
        out["prccq"].notna(),
        out["share_change"] * out["prccq"],
        np.nan,
    )

    out["source"] = "compustat_proxy"
    out["action_type"] = "share_issuance_proxy"
    out["action_subtype"] = "share_issuance_proxy"
    out["confidence"] = np.where(out["share_change_pct"] >= 0.05, "high", "medium")

    keep_cols = [
        "gvkey",
        "action_date",
        "amount",
        "share_change",
        "share_change_pct",
        "prccq",
        "mkvaltq",
        "tic",
        "conm",
        "sic",
        "source",
        "action_type",
        "action_subtype",
        "confidence",
    ]
    out = out[[c for c in keep_cols if c in out.columns]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    log(f"Saved {len(out):,} proxy ECM actions -> {OUT_PATH}")


if __name__ == "__main__":
    main()
