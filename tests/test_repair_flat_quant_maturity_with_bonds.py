from pathlib import Path

import pandas as pd

from scripts.repair_flat_quant_maturity_with_bonds import main


def test_maturity_overlay_with_bond_schedule_and_redemptions(tmp_path: Path, monkeypatch) -> None:
    flat = pd.DataFrame(
        [
            {
                "company_id": "0000001111",
                "company_name": "Bond Proxy Co",
                "capital_structure__debt_due_0_12m__value": None,
                "capital_structure__debt_due_0_12m__unit": "usd",
                "capital_structure__debt_due_0_12m__support_mode": "unsupported",
                "capital_structure__debt_due_0_12m__fallback_used": None,
                "capital_structure__debt_like_obligations_normalized__value": 1_500_000_000.0,
                "capital_structure__debt_like_obligations_normalized__unit": "usd",
                "capital_structure__debt_like_obligations_normalized__support_mode": "exact",
                "capital_structure__maturity_wall_ratio_24m__value": None,
                "capital_structure__maturity_wall_ratio_24m__unit": "ratio",
                "capital_structure__maturity_wall_ratio_24m__support_mode": "unsupported",
                "capital_structure__maturity_wall_ratio_24m__fallback_used": None,
            },
            {
                "company_id": "0000002222",
                "company_name": "Statement Due Co",
                "capital_structure__debt_due_0_12m__value": 200_000_000.0,
                "capital_structure__debt_due_0_12m__unit": "usd",
                "capital_structure__debt_due_0_12m__support_mode": "exact",
                "capital_structure__debt_due_0_12m__fallback_used": None,
                "capital_structure__debt_like_obligations_normalized__value": 1_000_000_000.0,
                "capital_structure__debt_like_obligations_normalized__unit": "usd",
                "capital_structure__debt_like_obligations_normalized__support_mode": "exact",
                "capital_structure__maturity_wall_ratio_24m__value": 0.2,
                "capital_structure__maturity_wall_ratio_24m__unit": "ratio",
                "capital_structure__maturity_wall_ratio_24m__support_mode": "proxy_missing_component",
                "capital_structure__maturity_wall_ratio_24m__fallback_used": "old_lower_bound",
            },
        ]
    )
    flat_path = tmp_path / "flat.parquet"
    flat.to_parquet(flat_path, index=False)

    identifiers = pd.DataFrame(
        [
            {"entity_id": "0000001111", "identifier_type": "permno", "identifier_value": "1001"},
            {"entity_id": "0000002222", "identifier_type": "permno", "identifier_value": "1002"},
        ]
    )
    identifier_path = tmp_path / "identifiers.parquet"
    identifiers.to_parquet(identifier_path, index=False)

    issues = pd.DataFrame(
        [
            {
                "ISSUE_ID": "A1",
                "permno": 1001,
                "offering_date": "2024-01-01",
                "maturity_date": "2025-06-01",
                "amount": 500_000_000.0,
                "offering_amt_k": 500_000.0,
                "currency": "N",
                "CONVERTIBLE": "N",
                "ASSET_BACKED": "N",
                "PERPETUAL": "N",
                "PRIVATE_PLACEMENT": "N",
            },
            {
                "ISSUE_ID": "A2",
                "permno": 1001,
                "offering_date": "2024-01-01",
                "maturity_date": "2026-06-01",
                "amount": 400_000_000.0,
                "offering_amt_k": 400_000.0,
                "currency": "N",
                "CONVERTIBLE": "N",
                "ASSET_BACKED": "N",
                "PERPETUAL": "N",
                "PRIVATE_PLACEMENT": "N",
            },
            {
                "ISSUE_ID": "B1",
                "permno": 1002,
                "offering_date": "2024-01-01",
                "maturity_date": "2026-08-01",
                "amount": 300_000_000.0,
                "offering_amt_k": 300_000.0,
                "currency": "USD",
                "CONVERTIBLE": "N",
                "ASSET_BACKED": "N",
                "PERPETUAL": "N",
                "PRIVATE_PLACEMENT": "N",
            },
        ]
    )
    issues_path = tmp_path / "issues.parquet"
    issues.to_parquet(issues_path, index=False)

    redemptions = pd.DataFrame(
        [
            {"ISSUE_ID": "A2", "action_date": "2024-12-01", "amount": 100_000.0},
        ]
    )
    redemptions_path = tmp_path / "redemptions.parquet"
    redemptions.to_parquet(redemptions_path, index=False)

    out_path = tmp_path / "out.parquet"
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "repair_flat_quant_maturity_with_bonds.py",
            "--flat-path",
            str(flat_path),
            "--entity-identifier-path",
            str(identifier_path),
            "--bond-issuances-path",
            str(issues_path),
            "--bond-redemptions-path",
            str(redemptions_path),
            "--out-parquet",
            str(out_path),
            "--summary-out",
            str(summary_path),
        ],
    )
    main()

    out = pd.read_parquet(out_path)
    row1 = out.loc[out["company_id"] == "0000001111"].iloc[0]
    assert row1["capital_structure__public_bond_outstanding__value"] == 800_000_000.0
    assert row1["capital_structure__public_bond_due_0_12m__value"] == 500_000_000.0
    assert row1["capital_structure__public_bond_due_12_24m__value"] == 300_000_000.0
    assert row1["capital_structure__debt_due_0_12m__value"] == 500_000_000.0
    assert row1["capital_structure__debt_due_0_12m__support_mode"] == "proxy_missing_component"
    assert row1["capital_structure__debt_due_12_24m__value"] == 300_000_000.0
    assert row1["capital_structure__debt_due_12_24m__support_mode"] == "proxy_missing_component"
    assert abs(row1["capital_structure__maturity_wall_ratio_24m__value"] - (800_000_000.0 / 1_500_000_000.0)) < 1e-9

    row2 = out.loc[out["company_id"] == "0000002222"].iloc[0]
    assert row2["capital_structure__public_bond_due_0_12m__value"] == 0.0
    assert row2["capital_structure__public_bond_due_12_24m__value"] == 300_000_000.0
    assert row2["capital_structure__debt_due_0_12m__value"] == 200_000_000.0
    assert row2["capital_structure__debt_due_0_12m__support_mode"] == "exact"
    assert row2["capital_structure__debt_due_12_24m__value"] == 300_000_000.0
    assert row2["capital_structure__debt_due_12_24m__support_mode"] == "proxy_missing_component"
    assert abs(row2["capital_structure__maturity_wall_ratio_24m__value"] - 0.5) < 1e-9
