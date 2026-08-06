import json
from pathlib import Path

import pandas as pd

from scripts.repair_flat_quant_ebitda_with_refinitiv import main


def test_refinitiv_overlay_replaces_market_grade_metrics(tmp_path, monkeypatch):
    flat_path = tmp_path / "flat.parquet"
    provider_path = tmp_path / "provider.parquet"
    identifier_path = tmp_path / "identifiers.parquet"
    ratings_path = tmp_path / "ratings.parquet"
    out_parquet = tmp_path / "out.parquet"
    out_csv = tmp_path / "out.csv"
    summary_out = tmp_path / "summary.json"

    pd.DataFrame(
        [
            {
                "company_id": "0000006201",
                "company_name": "AMERICAN AIRLINES GROUP INC.",
                "operating__ebitda_margin_ttm__value": 0.041,
                "operating__ebitda_margin_ttm__support_mode": "proxy_missing_component",
                "operating__ebitda_margin_ttm__fallback_used": "old_fallback",
                "market__fcf_yield__value": 0.12,
                "market__fcf_yield__support_mode": "proxy_missing_component",
                "market__fcf_yield__fallback_used": "old_fallback",
                "operating__fcf_conversion__value": 0.8,
                "operating__fcf_conversion__support_mode": "proxy_missing_component",
                "operating__fcf_conversion__fallback_used": "old_fallback",
                "market__ev_ebitda__value": 11.2,
                "market__ev_ebitda__support_mode": "proxy_missing_component",
                "market__ev_ebitda__fallback_used": "old_fallback",
            }
        ]
    ).to_parquet(flat_path, index=False)

    pd.DataFrame(
        [
            {
                "Instrument": "AAL.OQ",
                "Company Common Name": "American Airlines Group Inc",
                "Revenue": 54_633_000_000.0,
                "EBITDA": 3_845_000_000.0,
                "Free Cash Flow": -1_449_000_000.0,
                "Cash and Short Term Investments": 5_836_000_000.0,
                "Company Market Cap": 8_782_004_649.0,
                "Enterprise Value To EBITDA (Daily Time Series Ratio)": 8.329772,
            }
        ]
    ).to_parquet(provider_path, index=False)

    pd.DataFrame(
        [
            {
                "entity_id": "0000006201",
                "identifier_type": "ticker",
                "identifier_value": "AAL",
            }
        ]
    ).to_parquet(identifier_path, index=False)

    pd.DataFrame(
        [
            {
                "company_id": "0000006201",
                "rating_date": "2024-10-24T00:00:00Z",
                "rating_symbol": "B+",
                "current_rating_symbol": "B+",
                "outlook": "Stable",
                "creditwatch": None,
                "source_type": "ciq_ratings",
                "artifact_id": "ciq:aal:bplus",
                "effective_at": "2024-10-24T00:00:00Z",
                "published_at": "2024-10-24T00:00:00Z",
                "ingested_at": "2024-10-24T00:00:00Z",
            }
        ]
    ).to_parquet(ratings_path, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "repair_flat_quant_ebitda_with_refinitiv.py",
            "--flat-path",
            str(flat_path),
            "--provider-reference-path",
            str(provider_path),
            "--entity-identifier-path",
            str(identifier_path),
            "--ratings-path",
            str(ratings_path),
            "--out-parquet",
            str(out_parquet),
            "--out-csv",
            str(out_csv),
            "--summary-out",
            str(summary_out),
        ],
    )

    main()

    repaired = pd.read_parquet(out_parquet)
    row = repaired.iloc[0]
    assert row["operating__ebitda_margin_ttm__support_mode"] == "exact"
    assert row["market__fcf_yield__support_mode"] == "exact"
    assert row["operating__fcf_conversion__support_mode"] == "exact"
    assert row["market__ev_ebitda__support_mode"] == "exact"
    assert abs(row["operating__ebitda_margin_ttm__value"] - (3_845_000_000.0 / 54_633_000_000.0)) < 1e-12
    assert abs(row["market__fcf_yield__value"] - (-1_449_000_000.0 / 8_782_004_649.0)) < 1e-12
    assert abs(row["operating__fcf_conversion__value"] - (-1_449_000_000.0 / 3_845_000_000.0)) < 1e-12
    assert row["market__ev_ebitda__value"] == 8.329772
    assert row["operating__ebitda_provider_direct__value"] == 3_845_000_000.0
    assert row["operating__revenue_provider_direct__value"] == 54_633_000_000.0
    assert row["operating__free_cash_flow_provider_direct__value"] == -1_449_000_000.0
    assert row["liquidity__cash_and_short_term_investments_provider_direct__value"] == 5_836_000_000.0
    assert row["market__market_cap_provider_direct__value"] == 8_782_004_649.0
    assert row["capital_structure__rating_state__rating"] == "B+"
    assert row["capital_structure__rating_state__rating_support_mode"] == "exact"
    assert row["capital_structure__rating_state__outlook"] == "Stable"
    assert row["capital_structure__rating_state__score__value"] == 14.0
    assert row["capital_structure__rating_state__score__support_mode"] == "exact"

    summary = json.loads(summary_out.read_text())
    assert summary["provider_ebitda_coverage"] == 1
    assert summary["provider_free_cash_flow_coverage"] == 1
    assert summary["provider_cash_and_short_term_investments_coverage"] == 1
    assert summary["provider_market_cap_coverage"] == 1
    assert summary["ebitda_margin_overrides"] == 1
    assert summary["fcf_yield_overrides"] == 1
    assert summary["fcf_conversion_overrides"] == 1
    assert summary["ev_ebitda_overrides"] == 1
    assert summary["rating_matches"] == 1
    assert summary["capital_structure.rating_state.rating"]["exact"] == 1
    assert summary["capital_structure.rating_state.score"]["exact"] == 1
