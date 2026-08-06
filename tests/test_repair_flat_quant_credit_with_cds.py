from pathlib import Path

import pandas as pd

from scripts.repair_flat_quant_credit_with_cds import overlay_cds


def test_overlay_cds_prefers_primary_senior_curve_and_overrides_proxy(tmp_path: Path) -> None:
    flat = pd.DataFrame(
        [
            {
                "company_id": "0000000001",
                "company_name": "Example Co",
                "market__credit_spread_level__value": 0.03,
                "market__credit_spread_level__unit": "spread",
                "market__credit_spread_level__support_mode": "proxy_missing_component",
                "market__credit_spread_level__fallback_used": "heuristic",
                "market__credit_spread_percentile_2y__value": None,
                "market__credit_spread_percentile_2y__unit": "percentile_0_1",
                "market__credit_spread_percentile_2y__support_mode": "unsupported",
                "market__credit_spread_percentile_2y__fallback_used": None,
                "market__credit_window_proxy__value": 0.5,
                "market__credit_window_proxy__unit": "index_0_1",
                "market__credit_window_proxy__support_mode": "proxy_missing_component",
                "market__credit_window_proxy__fallback_used": "heuristic",
            }
        ]
    )
    flat_path = tmp_path / "flat.parquet"
    flat.to_parquet(flat_path, index=False)

    mapping = pd.DataFrame(
        [
            {
                "company_id": 1,
                "company_name": "Example Co",
                "equity_ticker": "EX",
                "cds_ticker": "EXAMPLE",
                "redcode": "ABC123",
                "shortname": "Example Co",
                "match_type": "exact_ticker",
            }
        ]
    )
    mapping_path = tmp_path / "map.csv"
    mapping.to_csv(mapping_path, index=False)

    cds = pd.DataFrame(
        [
            {
                "redcode": "ABC123",
                "date": "2024-12-30",
                "ticker": "EXAMPLE",
                "shortname": "Example Co",
                "tier": "SNRFOR",
                "currency": "USD",
                "docclause": "XR14",
                "primarycurve": "Y",
                "tenor": "5Y",
                "parspread": 0.012,
                "upfront": 0.0,
                "carriedforward": 0,
                "compositedepth5y": 3,
                "compositecurverating": "A",
                "curveliquidityscore": 4.0,
            },
            {
                "redcode": "ABC123",
                "date": "2024-12-30",
                "ticker": "EXAMPLE",
                "shortname": "Example Co",
                "tier": "SUBLT2",
                "currency": "USD",
                "docclause": "XR14",
                "primarycurve": "N",
                "tenor": "5Y",
                "parspread": 0.05,
                "upfront": 0.0,
                "carriedforward": 0,
                "compositedepth5y": 3,
                "compositecurverating": "A",
                "curveliquidityscore": None,
            },
            {
                "redcode": "ABC123",
                "date": "2024-12-31",
                "ticker": "EXAMPLE",
                "shortname": "Example Co",
                "tier": "SNRFOR",
                "currency": "USD",
                "docclause": "XR14",
                "primarycurve": "Y",
                "tenor": "5Y",
                "parspread": 0.02,
                "upfront": 0.0,
                "carriedforward": 0,
                "compositedepth5y": 3,
                "compositecurverating": "A",
                "curveliquidityscore": 4.0,
            },
        ]
    )
    cds_path = tmp_path / "cds.csv.gz"
    cds.to_csv(cds_path, index=False, compression="gzip")

    repaired, summary = overlay_cds(flat_path, cds_path, mapping_path, "2024-12-31")
    row = repaired.iloc[0]

    assert row["market__credit_spread_level__value"] == 0.02
    assert row["market__credit_spread_level__support_mode"] == "exact"
    assert row["market__credit_spread_percentile_2y__value"] == 1.0
    assert row["market__credit_window_proxy__value"] == 0.0
    assert row["market__cds_redcode"] == "ABC123"
    assert row["market__credit_truth_tier"] == "cds_exact"
    assert row["market__credit_truth_tier_rank__value"] == 3
    assert summary["cds_exact_matches"] == 1
    assert summary["market.credit_truth_tier"]["cds_exact"] == 1
