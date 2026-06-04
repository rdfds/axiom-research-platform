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


def test_load_price_history_for_row_uses_single_permno_crsp_slice(tmp_path: Path):
    crsp_root = tmp_path / "crsp"
    crsp_root.mkdir()
    dsf_path = crsp_root / "dsf_2024-01-01_to_2024-12-31.parquet"
    pd.DataFrame(
        {
            "permno": [10001, 10001, 20002],
            "date": pd.to_datetime(["2024-12-30", "2024-12-31", "2024-12-31"]),
            "prc": [-10.5, -11.0, -99.0],
            "ret": [0.01, 0.02, 0.03],
            "retx": [0.009, 0.018, 0.02],
            "shrout": [1000.0, 1000.0, 500.0],
        }
    ).to_parquet(dsf_path, index=False)

    loaded = _load_price_history_for_row(
        permno="10001",
        as_of_time="2024-12-31T00:00:00+00:00",
        crsp_market_cache_path=None,
        crsp_daily_root=crsp_root,
        raw_timeseries_path=tmp_path / "unused.parquet",
        allow_monthly_market_proxy=False,
    )

    assert list(loaded["permno"]) == ["10001", "10001"]
    assert list(loaded["close_price"]) == [10.5, 11.0]


def test_load_price_history_for_batch_groups_by_permno(tmp_path: Path):
    crsp_root = tmp_path / "crsp"
    crsp_root.mkdir()
    dsf_path = crsp_root / "dsf_2024-01-01_to_2024-12-31.parquet"
    pd.DataFrame(
        {
            "permno": [10001, 10001, 20002],
            "date": pd.to_datetime(["2024-12-30", "2024-12-31", "2024-12-31"]),
            "prc": [-10.5, -11.0, -99.0],
            "ret": [0.01, 0.02, 0.03],
            "retx": [0.009, 0.018, 0.02],
            "shrout": [1000.0, 1000.0, 500.0],
        }
    ).to_parquet(dsf_path, index=False)

    loaded = _load_price_history_for_batch(
        permnos=["10001", "20002"],
        as_of_times=["2024-12-31T00:00:00+00:00", "2024-12-31T00:00:00+00:00"],
        crsp_market_cache_path=None,
        crsp_daily_root=crsp_root,
        raw_timeseries_path=tmp_path / "unused.parquet",
        allow_monthly_market_proxy=False,
    )

    assert sorted(loaded) == ["10001", "20002"]
    assert list(loaded["10001"]["close_price"]) == [10.5, 11.0]
    assert list(loaded["20002"]["close_price"]) == [99.0]


def test_build_price_metrics_from_crsp_marks_daily_returns_exact():
    dates = pd.bdate_range("2023-10-02", periods=340, tz="UTC")
    returns = pd.Series([0.001] * len(dates), dtype=float)
    prices = 100.0 * (1.0 + returns).cumprod()
    history = pd.DataFrame(
        {
            "permno": "10001",
            "trade_date": dates.normalize(),
            "date_key": dates.normalize(),
            "close_price": prices,
            "price_proxy": prices,
            "total_return": returns,
            "price_return": returns,
            "shares_outstanding": 1000.0,
            "daily_cap": prices * 1000.0,
            "delist_flag": False,
        }
    )

    metrics = _build_price_metrics_from_crsp(
        permno="10001",
        price_history=history,
        as_of_time="2024-12-31T00:00:00+00:00",
        computed_at="2026-03-29T00:00:00+00:00",
        provenance_source="/tmp/crsp",
    )

    assert metrics["market.price_spot"]["support_mode"] == "exact"
    assert metrics["market.total_return_1m_standardized"]["support_mode"] == "exact"
    assert metrics["market.total_return_3m_standardized"]["support_mode"] == "exact"
    assert metrics["market.total_return_6m_standardized"]["support_mode"] == "exact"
    assert metrics["market.total_return_12m_standardized"]["support_mode"] == "exact"


def test_company_processing_guard_times_out():
    with pytest.raises(_CompanyProcessingTimeout):
        with _company_processing_guard(0.05):
            time.sleep(0.2)


