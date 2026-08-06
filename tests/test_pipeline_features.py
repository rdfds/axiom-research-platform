from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.pipeline.features import FeatureBuilder


class _FakeWarehouse:
    def __init__(self, series_frames: dict[str, pd.DataFrame]):
        self.series_frames = series_frames

    def query(self, table_name: str, as_of=None, columns=None, where=None, limit=None, prefer_gvkey=False):
        assert table_name == "warehouse_macro"
        series_id = str(where or "").split("'")[1]
        df = self.series_frames.get(series_id, pd.DataFrame()).copy()
        if df.empty:
            return df
        if as_of is not None:
            cutoff = pd.Timestamp(as_of)
            df = df[pd.to_datetime(df["available_time"], errors="coerce") <= cutoff]
        return df


def test_compute_macro_features_emits_fed_funds_and_real_gdp_growth():
    warehouse = _FakeWarehouse(
        {
            "DGS10": pd.DataFrame(
                [
                    {"event_time": "2025-02-28", "available_time": "2025-02-28", "value": 4.40},
                    {"event_time": "2025-03-31", "available_time": "2025-03-31", "value": 4.58},
                ]
            ),
            "DFF": pd.DataFrame(
                [
                    {"event_time": "2025-03-31", "available_time": "2025-03-31", "value": 4.33},
                ]
            ),
            "GDPC1": pd.DataFrame(
                [
                    {"event_time": "2024-03-31", "available_time": "2024-04-25", "value": 100.0},
                    {"event_time": "2024-06-30", "available_time": "2024-07-25", "value": 102.0},
                    {"event_time": "2024-09-30", "available_time": "2024-10-30", "value": 104.0},
                    {"event_time": "2024-12-31", "available_time": "2025-01-30", "value": 106.0},
                    {"event_time": "2025-03-31", "available_time": "2025-04-30", "value": 108.0},
                ]
            ),
        }
    )
    builder = FeatureBuilder(warehouse=warehouse)
    builder.link_table = None

    out = builder.compute_macro_features(
        datetime(2025, 6, 30),
        {
            "rate_10y": "DGS10",
            "fed_funds_effective": "DFF",
            "real_gdp": "GDPC1",
        },
    )

    assert out["macro_rate_10y"] == 4.58
    assert out["macro_fed_funds_effective"] == 4.33
    assert out["macro_real_gdp"] == 108.0
    assert out["macro_real_gdp_growth_yoy"] == pytest.approx(0.08)
