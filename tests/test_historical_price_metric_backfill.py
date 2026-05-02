from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.pipeline.historical_price_metric_backfill import backfill_historical_price_window_metrics
from src.pipeline.precedent_brain import augment_precedent_state_vector_columns


def _write_price_history_parquet(path: Path, *, company_id: str) -> None:
    dates = pd.bdate_range("2023-09-01", periods=160, tz="UTC")
    steps = np.arange(len(dates), dtype=float)
    prices = 100.0 + (0.25 * steps) + (4.0 * np.sin(steps / 7.0))
    frame = pd.DataFrame(
        {
            "company_id": [company_id] * len(dates),
            "series_type": ["price"] * len(dates),
            "trade_date": dates.tz_convert(None),
            "adjusted_close": prices,
            "close": prices,
        }
    )
    frame.to_parquet(path, index=False)


def _write_sparse_monthly_price_history_parquet(path: Path, *, company_id: str) -> None:
    dates = pd.to_datetime(
        [
            "2020-01-31",
            "2020-02-29",
            "2020-03-31",
            "2020-04-30",
        ]
    )
    prices = [100.0, 92.0, 80.0, 88.0]
    frame = pd.DataFrame(
        {
            "company_id": [company_id] * len(dates),
            "series_type": ["price"] * len(dates),
            "trade_date": dates,
            "adjusted_close": prices,
            "close": prices,
        }
    )
    frame.to_parquet(path, index=False)


