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


def test_fail_open_market_metrics_mark_market_stack_unsupported():
    metrics = _build_fail_open_market_metrics(
        as_of_time="2024-12-31T00:00:00+00:00",
        computed_at="2026-03-30T00:00:00+00:00",
        provenance_source="/tmp/companyfacts/CIK0000000001.json",
        error_type="company_processing_timeout",
        error_message="timed out",
    )

    assert metrics["market.price_spot"]["support_mode"] == "unsupported"
    assert metrics["market.total_return_12m_standardized"]["missing_reason"] == "company_processing_timeout"
    assert metrics["market.market_cap_provider_direct"]["support_mode"] == "unsupported"
    assert metrics["market.market_cap_provider_direct"]["component_breakdown"]["error_type"] == "company_processing_timeout"


def test_market_cap_fail_open_preserves_exact_price_metrics():
    dates = pd.bdate_range("2024-11-01", periods=45, tz="UTC")
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
        computed_at="2026-03-30T00:00:00+00:00",
        provenance_source="/tmp/crsp",
    )
    metrics["market.market_cap_provider_direct"] = _build_fail_open_market_cap_metric(
        as_of_time="2024-12-31T00:00:00+00:00",
        computed_at="2026-03-30T00:00:00+00:00",
        provenance_source="/tmp/companyfacts/CIK0000000001.json",
        error_type="company_processing_timeout",
        error_message="timed out",
    )

    assert metrics["market.price_spot"]["support_mode"] == "exact"
    assert metrics["market.total_return_1m_standardized"]["support_mode"] == "exact"
    assert metrics["market.market_cap_provider_direct"]["support_mode"] == "unsupported"
    assert metrics["market.market_cap_provider_direct"]["missing_reason"] == "company_processing_timeout"


def test_fail_open_macro_metrics_mark_macro_stack_unsupported():
    metrics = _build_fail_open_macro_metrics(
        as_of_time="2024-12-31T00:00:00+00:00",
        computed_at="2026-03-30T00:00:00+00:00",
        provenance_source="/tmp/raw_timeseries.parquet",
        error_type="macro_build_failed",
        error_message="bad macro row",
    )

    assert metrics["macro.sofr_or_fed_funds"]["support_mode"] == "unsupported"
    assert metrics["macro.ust_10y_yield"]["missing_reason"] == "macro_build_failed"
    assert metrics["macro.curve_2s10s"]["component_breakdown"]["error_type"] == "macro_build_failed"
    assert metrics["macro.cpi_yoy"]["support_mode"] == "unsupported"


def test_build_macro_metrics_emits_explicit_fed_funds_sofr_and_real_gdp_growth():
    macro_history = pd.DataFrame(
        {
            "instrument_id": [
                "SOFR",
                "DFF",
                "DGS2",
                "DGS10",
                "CPIAUCSL",
                "CPIAUCSL",
                "RSAFS",
                "RSAFS",
                "GDPC1",
                "GDPC1",
                "GDPC1",
                "GDPC1",
                "GDPC1",
            ],
            "event_date": pd.to_datetime(
                [
                    "2024-12-31",
                    "2024-12-31",
                    "2024-12-31",
                    "2024-12-31",
                    "2024-12-31",
                    "2023-12-31",
                    "2024-12-31",
                    "2023-12-31",
                    "2023-12-31",
                    "2024-03-31",
                    "2024-06-30",
                    "2024-09-30",
                    "2024-12-31",
                ],
                utc=True,
            ).normalize(),
            "value": [
                4.6,
                4.4,
                4.2,
                4.6,
                300.0,
                285.0,
                200.0,
                180.0,
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
            ],
            "units": ["pct"] * 13,
        }
    )
    macro_history["date_key"] = macro_history["event_date"]

    metrics = _build_macro_metrics(
        macro_history=macro_history,
        as_of_time="2024-12-31T00:00:00+00:00",
        computed_at="2026-04-01T00:00:00+00:00",
        provenance_source="/tmp/raw_timeseries.parquet",
    )

    assert metrics["macro.fed_funds_effective"]["support_mode"] == "exact"
    assert metrics["macro.fed_funds_effective"]["value"] == 4.4
    assert metrics["macro.sofr"]["support_mode"] == "exact"
    assert metrics["macro.sofr"]["value"] == 4.6
    assert metrics["macro.sofr_or_fed_funds"]["value"] == 4.6
    assert metrics["macro.sofr_or_fed_funds"]["component_breakdown"]["selected_instrument"] == "SOFR"
    assert metrics["macro.real_gdp_growth_yoy"]["support_mode"] == "exact"
    assert round(metrics["macro.real_gdp_growth_yoy"]["value"], 6) == 0.04
    assert (
        metrics["macro.real_gdp_growth_yoy"]["component_breakdown"]["formula"]
        == "(current_value / value_4_observations_prior) - 1"
    )


def test_load_macro_history_keeps_enough_gdp_release_history_for_yoy(tmp_path: Path):
    raw_path = tmp_path / "raw_timeseries.parquet"
    pd.DataFrame(
        {
            "series_type": ["macro"] * 5,
            "instrument_id": ["GDPC1"] * 5,
            "event_time": pd.to_datetime(
                [
                    "2023-10-26",
                    "2024-01-25",
                    "2024-04-25",
                    "2024-07-25",
                    "2024-10-30",
                ],
                utc=True,
            ),
            "value": [100.0, 101.0, 102.0, 103.0, 104.0],
            "units": ["usd"] * 5,
        }
    ).to_parquet(raw_path, index=False)

    loaded = _load_macro_history(
        raw_path,
        min_asof_date=pd.Timestamp("2024-12-31", tz="UTC"),
        max_asof_date=pd.Timestamp("2024-12-31", tz="UTC"),
    )

    gdp = loaded[loaded["instrument_id"] == "GDPC1"].sort_values("event_date").reset_index(drop=True)
    assert len(gdp) == 5
    assert gdp.iloc[0]["event_date"] < pd.Timestamp("2023-11-01", tz="UTC")
