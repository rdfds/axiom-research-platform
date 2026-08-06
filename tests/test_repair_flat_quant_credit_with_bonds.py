from pathlib import Path

import pandas as pd

from scripts.repair_flat_quant_credit_with_bonds import build_bond_credit_overlay


def test_build_bond_credit_overlay(tmp_path: Path) -> None:
    flat = pd.DataFrame(
        [
            {"company_id": "0000006201", "company_name": "American Airlines", "market__credit_spread_level__support_mode": "proxy_missing_component"},
            {"company_id": "0000009999", "company_name": "No Bonds Inc", "market__credit_spread_level__support_mode": "proxy_missing_component"},
        ]
    )
    flat_path = tmp_path / "flat.parquet"
    flat.to_parquet(flat_path, index=False)

    identifiers = pd.DataFrame(
        [
            {"entity_id": "0000006201", "identifier_type": "permno", "identifier_value": "6201"},
            {"entity_id": "0000009999", "identifier_type": "permno", "identifier_value": "9999"},
        ]
    )
    identifier_path = tmp_path / "identifiers.parquet"
    identifiers.to_parquet(identifier_path, index=False)

    issues = pd.DataFrame(
        [
            {
                "permno": 6201,
                "COMPLETE_CUSIP": "00123AB45",
                "offering_date": "2023-01-01",
                "maturity_date": "2027-01-01",
                "amount": 500_000_000.0,
                "offering_amt_k": 500_000.0,
                "CONVERTIBLE": "N",
                "ASSET_BACKED": "N",
                "PERPETUAL": "N",
                "PRIVATE_PLACEMENT": "N",
            }
        ]
    )
    issues_path = tmp_path / "issues.parquet"
    issues.to_parquet(issues_path, index=False)

    trace = pd.DataFrame(
        [
            {
                "cusip_id": "00123AB45",
                "trade_date": "2024-12-20",
                "yld_pt_avg": 5.0,
                "price_avg": 100.0,
                "volume_total": 1_000_000.0,
                "trades": 3,
            },
            {
                "cusip_id": "00123AB45",
                "trade_date": "2024-12-10",
                "yld_pt_avg": 4.0,
                "price_avg": 101.0,
                "volume_total": 500_000.0,
                "trades": 2,
            },
        ]
    )
    trace_path = tmp_path / "trace.parquet"
    trace.to_parquet(trace_path, index=False)

    raw_timeseries = pd.DataFrame(
        [
            {
                "instrument_id": "DGS2",
                "event_time": "2024-12-19",
                "value": 3.0,
            },
            {
                "instrument_id": "DGS2",
                "event_time": "2024-12-10",
                "value": 2.0,
            },
        ]
    )
    raw_path = tmp_path / "raw.parquet"
    raw_timeseries.to_parquet(raw_path, index=False)

    overlay = build_bond_credit_overlay(
        flat_path=flat_path,
        entity_identifier_path=identifier_path,
        trace_daily_path=trace_path,
        bond_issuances_path=issues_path,
        raw_timeseries_path=raw_path,
        as_of_date="2024-12-31",
        current_lookback_days=90,
        history_lookback_days=730,
    )

    assert len(overlay) == 1
    row = overlay.iloc[0]
    assert row["company_id"] == "0000006201"
    assert abs(row["credit_spread_level"] - 0.02) < 1e-9
    assert abs(row["spread_percentile_2y"] - 1.0) < 1e-9
    assert row["current_issue_count"] == 1
    assert row["history_obs"] == 2
