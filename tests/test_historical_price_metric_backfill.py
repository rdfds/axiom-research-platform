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


def test_backfill_historical_price_window_metrics_recovers_price_features_and_clears_stale_state_vectors() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "raw_timeseries.parquet"
        _write_price_history_parquet(raw_path, company_id="001234")

        hist = pd.DataFrame(
            [
                {
                    "company_id": "001234",
                    "action_date": "2024-04-15T00:00:00+00:00",
                    "base_volatility_30d": np.nan,
                    "base_volatility_90d": np.nan,
                    "base_drawdown_90d": np.nan,
                    "base_momentum_60d": np.nan,
                    "state_vector_v1.market_stress": 0.91,
                    "state_vector_v1.market_access": 0.09,
                    "base_credit_window_proxy": 0.72,
                    "base_equity_window_proxy": 0.63,
                    "base_credit_spread_level": 0.035,
                    "base_market_cap": 2500.0,
                    "base_total_debt": 400.0,
                    "base_cash": 250.0,
                    "base_ebitda_ttm": 150.0,
                }
            ]
        )

        backfilled = backfill_historical_price_window_metrics(hist, raw_timeseries_path=raw_path)
        row = backfilled.iloc[0]

        assert pd.notna(row["base_volatility_30d"])
        assert pd.notna(row["base_volatility_90d"])
        assert pd.notna(row["base_drawdown_90d"])
        assert pd.notna(row["base_momentum_60d"])
        assert pd.isna(row["state_vector_v1.market_stress"])
        assert pd.isna(row["state_vector_v1.market_access"])

        augmented = augment_precedent_state_vector_columns(backfilled)
        augmented_row = augmented.iloc[0]
        expected_market_stress = (row["base_volatility_90d"] * 0.6) + (abs(row["base_drawdown_90d"]) * 0.4)

        assert np.isclose(augmented_row["state_vector_v1.market_stress"], expected_market_stress)
        assert pd.notna(augmented_row["state_vector_v1.market_access"])


def test_backfill_historical_price_window_metrics_uses_sparse_monthly_fallback_for_90d_stress_inputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "raw_timeseries.parquet"
        _write_sparse_monthly_price_history_parquet(raw_path, company_id="001239")

        hist = pd.DataFrame(
            [
                {
                    "company_id": "001239",
                    "action_date": "2020-05-15T00:00:00+00:00",
                    "base_volatility_30d": np.nan,
                    "base_volatility_90d": np.nan,
                    "base_drawdown_90d": np.nan,
                    "base_momentum_60d": np.nan,
                    "state_vector_v1.market_stress": 0.77,
                    "state_vector_v1.market_access": 0.23,
                    "base_credit_window_proxy": 0.55,
                    "base_equity_window_proxy": 0.48,
                    "base_credit_spread_level": 0.045,
                    "base_market_cap": 300.0,
                    "base_total_debt": 200.0,
                    "base_cash": 40.0,
                    "base_ebitda_ttm": 50.0,
                }
            ]
        )

        backfilled = backfill_historical_price_window_metrics(hist, raw_timeseries_path=raw_path)
        row = backfilled.iloc[0]

        assert pd.isna(row["base_volatility_30d"])
        assert pd.notna(row["base_volatility_90d"])
        assert pd.notna(row["base_drawdown_90d"])
        assert pd.notna(row["base_momentum_60d"])
        assert pd.isna(row["state_vector_v1.market_stress"])
        assert pd.isna(row["state_vector_v1.market_access"])


def test_backfill_historical_price_window_metrics_preserves_existing_price_metrics() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "raw_timeseries.parquet"
        _write_price_history_parquet(raw_path, company_id="001234")

        hist = pd.DataFrame(
            [
                {
                    "company_id": "001234",
                    "action_date": "2024-04-15T00:00:00+00:00",
                    "base_volatility_30d": 0.11,
                    "base_volatility_90d": 0.22,
                    "base_drawdown_90d": -0.18,
                    "base_momentum_60d": 0.05,
                    "state_vector_v1.market_stress": 0.31,
                    "state_vector_v1.market_access": 0.69,
                }
            ]
        )

        backfilled = backfill_historical_price_window_metrics(hist, raw_timeseries_path=raw_path)
        row = backfilled.iloc[0]

        assert row["base_volatility_30d"] == 0.11
        assert row["base_volatility_90d"] == 0.22
        assert row["base_drawdown_90d"] == -0.18
        assert row["base_momentum_60d"] == 0.05
        assert row["state_vector_v1.market_stress"] == 0.31
        assert row["state_vector_v1.market_access"] == 0.69
