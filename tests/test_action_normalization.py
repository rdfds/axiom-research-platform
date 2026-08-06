from src.action_normalization import augment_action_outcomes_df, normalize_action_record


def test_normalize_platform_acquisition_family_scale():
    row = {
        "action_type": "acquisition",
        "action_subtype": "Disclosed Dollar Value Deal",
        "action_size": 2_000_000_000.0,
        "base_market_cap": 10_000_000_000.0,
    }
    out = normalize_action_record(row)
    assert out["normalized_action_family"] == "mna"
    assert out["normalized_action_subfamily"] == "platform_disclosed"
    assert out["normalized_action_id"] == "mna.platform_acquisition"
    assert out["normalization_level"] == "family_scale"
    assert out["family_scale_bucket"] == "medium"


def test_normalize_exact_dividend():
    row = {
        "action_type": "dividend_increase",
        "action_subtype": "dividend_increase",
    }
    out = normalize_action_record(row)
    assert out["normalized_action_id"] == "capital_return.dividend_increase"
    assert out["normalization_level"] == "exact"


def test_normalize_dividend_initiate_and_lbo_exact():
    dividend = normalize_action_record(
        {
            "action_type": "dividend_initiate",
            "action_subtype": "dividend_initiate",
        }
    )
    lbo = normalize_action_record(
        {
            "action_type": "acquisition",
            "action_subtype": "acquisition_lbo",
            "action_size": 5_000_000_000.0,
            "base_market_cap": 6_000_000_000.0,
        }
    )
    assert dividend["normalized_action_id"] == "capital_return.dividend_initiate"
    assert dividend["normalization_level"] == "exact"
    assert lbo["normalized_action_id"] == "mna.go_private_lbo"
    assert lbo["normalized_action_subfamily"] == "platform_lbo"
    assert lbo["family_scale_bucket"] == "large"


def test_normalize_stock_split_and_reverse_split_exact():
    split = normalize_action_record(
        {
            "action_type": "stock_split",
            "action_subtype": "stock_split",
            "action_size": 2.0,
        }
    )
    reverse = normalize_action_record(
        {
            "action_type": "reverse_split",
            "action_subtype": "reverse_split",
            "action_size": 0.1,
        }
    )
    assert split["normalized_action_family"] == "governance"
    assert split["normalized_action_id"] == "governance.stock_split"
    assert split["normalization_level"] == "exact"
    assert reverse["normalized_action_family"] == "governance"
    assert reverse["normalized_action_id"] == "governance.reverse_split"
    assert reverse["normalization_level"] == "exact"


def test_normalize_mna_backfills_canonical_action_ids_by_scale():
    tuck_in = normalize_action_record(
        {
            "action_type": "acquisition",
            "action_subtype": "Undisclosed Dollar Value Deal",
            "action_size": 100_000_000.0,
            "base_market_cap": 5_000_000_000.0,
        }
    )
    transformational = normalize_action_record(
        {
            "action_type": "acquisition",
            "action_subtype": "acquisition_merger",
            "action_size": 4_000_000_000.0,
            "base_market_cap": 10_000_000_000.0,
        }
    )
    tender = normalize_action_record(
        {
            "action_type": "acquisition",
            "action_subtype": "acquisition_tender",
            "action_size": 800_000_000.0,
            "base_market_cap": 8_000_000_000.0,
        }
    )
    assert tuck_in["normalized_action_id"] == "mna.tuck_in_acquisition"
    assert transformational["normalized_action_id"] == "mna.transformational_acquisition"
    assert tender["normalized_action_id"] == "mna.platform_acquisition"


def test_normalize_divestiture_backfills_portfolio_action_ids():
    full = normalize_action_record(
        {
            "action_type": "divestiture",
            "action_subtype": "asset sale",
            "percent_divested": 1.0,
        }
    )
    partial = normalize_action_record(
        {
            "action_type": "divestiture",
            "action_subtype": "asset sale",
            "percent_divested": 0.4,
        }
    )
    fallback = normalize_action_record(
        {
            "action_type": "divestiture",
            "action_subtype": "asset sale",
        }
    )
    assert full["normalized_action_id"] == "portfolio.divestiture_full"
    assert partial["normalized_action_id"] == "portfolio.divestiture_partial"
    assert fallback["normalized_action_id"] == "portfolio.asset_sale"


def test_normalize_divestiture_subtype_backfills_partial_and_full_action_ids():
    stake_sale = normalize_action_record(
        {
            "action_type": "divestiture",
            "action_subtype": "Stake Purchases Deal",
        }
    )
    acquired_business_sale = normalize_action_record(
        {
            "action_type": "divestiture",
            "action_subtype": "acquisition_tender",
        }
    )
    assert stake_sale["normalized_action_id"] == "portfolio.divestiture_partial"
    assert acquired_business_sale["normalized_action_id"] == "portfolio.divestiture_full"


def test_augment_action_outcomes_df_adds_columns():
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "action_type": "loan_issuance",
                "action_subtype": "Revolver/Line >= 1 Yr.",
                "action_size": 100.0,
                "base_market_cap": 1_000.0,
            }
        ]
    )
    out = augment_action_outcomes_df(df)
    assert "raw_action_type" in out.columns
    assert out.loc[0, "normalized_action_family"] == "capital_structure"
    assert out.loc[0, "normalized_action_subfamily"] == "revolver"
    assert out.loc[0, "normalized_action_id"] == "capital_structure.revolver_draw_or_resize"


def test_augment_action_outcomes_df_backfills_mna_action_ids():
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "action_type": "acquisition",
                "action_subtype": "acquisition_tender",
                "action_size": 900_000_000.0,
                "base_market_cap": 9_000_000_000.0,
            }
        ]
    )
    out = augment_action_outcomes_df(df)
    assert out.loc[0, "normalized_action_family"] == "mna"
    assert out.loc[0, "normalized_action_id"] == "mna.platform_acquisition"


def test_normalize_buyback_and_bond_issuance_backfill_exact_action_ids():
    buyback = normalize_action_record(
        {
            "action_type": "buyback",
            "action_subtype": "buyback",
            "action_size": 500_000_000.0,
            "base_market_cap": 10_000_000_000.0,
        }
    )
    bond = normalize_action_record(
        {
            "action_type": "bond_issuance",
            "action_subtype": None,
            "action_size": 750_000_000.0,
            "base_market_cap": 12_000_000_000.0,
        }
    )
    term_loan = normalize_action_record(
        {
            "action_type": "loan_issuance",
            "action_subtype": "Term Loan",
            "action_size": 200_000_000.0,
            "base_market_cap": 5_000_000_000.0,
        }
    )
    assert buyback["normalized_action_id"] == "capital_return.open_market_buyback"
    assert bond["normalized_action_id"] == "capital_structure.new_debt_issuance"
    assert term_loan["normalized_action_id"] == "capital_structure.new_debt_issuance"


def test_normalize_loan_refinancing_backfills_refinancing_action_id():
    refinance = normalize_action_record(
        {
            "action_type": "loan_refinancing",
            "action_subtype": "Term Loan B",
            "action_size": 350_000_000.0,
            "base_market_cap": 4_000_000_000.0,
        }
    )
    assert refinance["normalized_action_family"] == "capital_structure"
    assert refinance["normalized_action_subfamily"] == "refinancing_term_loan_family"
    assert refinance["normalized_action_id"] == "capital_structure.refinancing"
    assert refinance["normalization_level"] == "family_scale"


def test_augment_action_outcomes_df_keeps_split_action_size_from_ratio():
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "action_type": "stock_split",
                "action_subtype": "stock_split",
                "action_size": 2.0,
            },
            {
                "action_type": "reverse_split",
                "action_subtype": "reverse_split",
                "action_size": 0.2,
            },
        ]
    )
    out = augment_action_outcomes_df(df)
    assert out.loc[0, "normalized_action_id"] == "governance.stock_split"
    assert out.loc[1, "normalized_action_id"] == "governance.reverse_split"


def test_augment_action_outcomes_df_maps_loan_refinancing():
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "action_type": "loan_refinancing",
                "action_subtype": "Revolver/Line >= 1 Yr.",
                "action_size": 150.0,
                "base_market_cap": 1_500.0,
            }
        ]
    )
    out = augment_action_outcomes_df(df)
    assert out.loc[0, "normalized_action_family"] == "capital_structure"
    assert out.loc[0, "normalized_action_subfamily"] == "refinancing_revolver_family"
    assert out.loc[0, "normalized_action_id"] == "capital_structure.refinancing"


def test_normalize_loan_refinancing_maps_bond_family():
    refinance = normalize_action_record(
        {
            "action_type": "loan_refinancing",
            "action_subtype": "Fixed-Rate Bond",
            "action_size": 120_000_000.0,
            "base_market_cap": 2_000_000_000.0,
        }
    )
    assert refinance["normalized_action_subfamily"] == "refinancing_bond_family"
    assert refinance["normalized_action_id"] == "capital_structure.refinancing"
