import time
from pathlib import Path

import pandas as pd
import pytest

from scripts.backfill_market_macro_input_layer_v1 import (
    _CompanyProcessingTimeout,
    _build_fail_open_market_cap_metric,
    _build_macro_metrics,
    _build_price_metrics_from_crsp,
    _build_fail_open_macro_metrics,
    _build_fail_open_market_metrics,
    _company_processing_guard,
    _load_macro_history,
    _load_crsp_daily_from_repo,
    _load_price_history_for_batch,
    _load_price_history_for_row,
)


def test_load_crsp_daily_from_repo_returns_exact_daily_shape(tmp_path: Path):
    crsp_root = tmp_path / "crsp"
    crsp_root.mkdir()
    dsf_path = crsp_root / "dsf_2024-01-01_to_2024-12-31.parquet"
    pd.DataFrame(
        {
            "permno": [10001, 10001],
            "date": pd.to_datetime(["2024-12-30", "2024-12-31"]),
            "prc": [-10.5, -11.0],
            "ret": [0.01, 0.02],
            "retx": [0.009, 0.018],
            "shrout": [1000.0, 1000.0],
        }
    ).to_parquet(dsf_path, index=False)

    loaded = _load_crsp_daily_from_repo(
        crsp_root,
        ["10001"],
        min_asof_date=pd.Timestamp("2024-12-31", tz="UTC"),
        max_asof_date=pd.Timestamp("2024-12-31", tz="UTC"),
    )

    assert list(loaded["permno"]) == ["10001", "10001"]
    assert list(loaded["close_price"]) == [10.5, 11.0]
    assert list(loaded["price_proxy"]) == [10.5, 11.0]
    assert list(loaded["total_return"]) == [0.01, 0.02]
    assert list(loaded["price_return"]) == [0.009, 0.018]
    assert list(loaded["shares_outstanding"]) == [1000.0, 1000.0]
    assert list(loaded["daily_cap"]) == [10500.0, 11000.0]
    assert list(loaded["delist_flag"]) == [False, False]


