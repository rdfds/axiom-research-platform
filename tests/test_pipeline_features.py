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


