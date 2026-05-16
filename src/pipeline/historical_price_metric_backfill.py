from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, Optional

import duckdb
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RAW_TIMESERIES_PATH = REPO_ROOT / "data/inputs_layer/raw_timeseries.parquet"
_PRICE_BACKFILL_COLS = (
    "base_volatility_30d",
    "base_volatility_90d",
    "base_drawdown_90d",
    "base_momentum_60d",
)
_STALE_STATE_VECTOR_COLS = (
    "state_vector_v1.market_stress",
    "state_vector_v1.market_access",
)


def _normalize_company_id_for_price_join(value: object) -> str:
    digits = re.search(r"[0-9]+", str(value or ""))
    if digits is None:
        return ""
    return digits.group(0).rjust(6, "0")[:6]


def default_raw_timeseries_path() -> Path:
    return _DEFAULT_RAW_TIMESERIES_PATH


def _missing_metric_mask(frame: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    mask = pd.Series(False, index=frame.index, dtype=bool)
    for col in cols:
        if col not in frame.columns:
            mask = mask | True
            continue
        mask = mask | pd.to_numeric(frame[col], errors="coerce").isna()
    return mask


