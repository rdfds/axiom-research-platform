from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "57_ingest_wrds_dealscan_loanconnector.py"
    spec = importlib.util.spec_from_file_location("ingest_wrds_dealscan_loanconnector", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_active_artifact_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "58_build_wrds_dealscan_active_revolver_artifact.py"
    spec = importlib.util.spec_from_file_location("build_wrds_dealscan_active_revolver_artifact", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_snapshot_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_company_state_snapshot.py"
    spec = importlib.util.spec_from_file_location("build_company_state_snapshot", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_normalized_outputs_creates_revolver_proxy_dataset(tmp_path: Path):
    module = _load_script_module()

    facilities_path = tmp_path / "facilities.csv"
    company_map_path = tmp_path / "company_map.csv"
    id_map_path = tmp_path / "id_map.csv"
    covenants_path = tmp_path / "covenants.csv"
    out_root = tmp_path / "out"

    pd.DataFrame(
        [
            {
                "Borrower_Name": "Example Corp",
                "Borrower_Id": "101",
                "Ticker": "ABC",
                "Parent": "Example Corp",
                "LPC_Deal_ID": "9001",
                "Deal_Active_Date": "2024-01-15",
                "Deal_Remark": "",
                "LPC_Tranche_ID": "7001",
                "Tranche_Type": "Revolver/Line >= 1 Yr.",
                "Tranche_Active_Date": "2024-01-15",
                "Tranche_Maturity_Date": "2029-01-15",
                "Tranche_Amount": "2500",
                "Tranche_Amount_Converted": "2500",
                "Tranche_Remark": "",
                "All_in_Spread_Undrawn_bps": "5",
                "Annual_Fee_bps": "",
                "Commitment_Fee_bps": "5",
                "Utilization_Fee_bps": "",
                "Covenants": "Yes",
                "All_Covenants_Financial": "Max Leverage Ratio: Value is 3.50",
                "Letter_Of_Credit": "",
                "Swingline": "250",
            },
            {
                "Borrower_Name": "Example Corp",
                "Borrower_Id": "101",
                "Ticker": "ABC",
                "Parent": "Example Corp",
                "LPC_Deal_ID": "9001",
                "Deal_Active_Date": "2024-01-15",
                "Deal_Remark": "",
                "LPC_Tranche_ID": "7002",
                "Tranche_Type": "Term Loan B",
                "Tranche_Active_Date": "2024-01-15",
                "Tranche_Maturity_Date": "2031-01-15",
                "Tranche_Amount": "1000",
                "Tranche_Amount_Converted": "1000",
                "Tranche_Remark": "",
                "All_in_Spread_Undrawn_bps": "",
                "Annual_Fee_bps": "",
                "Commitment_Fee_bps": "",
                "Utilization_Fee_bps": "",
                "Covenants": "Yes",
                "All_Covenants_Financial": "",
                "Letter_Of_Credit": "",
                "Swingline": "",
            },
        ]
    ).to_csv(facilities_path, index=False)

    pd.DataFrame(
        [{"company_name": "Example Corp", "loanconnector_company_id": "101", "lpc_company_id": "88.0"}]
    ).to_csv(company_map_path, index=False)

    pd.DataFrame(
        [
            {
                "loanconnector_deal_id": "9001",
                "wrds_package_id": "111.0",
                "loanconnector_tranche_id": "7001",
                "wrds_facility_id": "222.0",
            }
        ]
    ).to_csv(id_map_path, index=False)

    pd.DataFrame(
        [
            {
                "lpc_deal_id": "9001",
                "lpc_tranche_id": "7001",
                "deal_active_date": "2024-01-15",
                "deal_input_date": "2024-01-16",
                "tranche_active_date": "2024-01-15",
                "tranche_o_a": "Origination",
                "all_covenants_financial": "Max Leverage Ratio: Value is 3.50",
                "max_leverage_ratio": "3.50:1",
            }
        ]
    ).to_csv(covenants_path, index=False)

    outputs = module.build_normalized_outputs(
        facilities_path=facilities_path,
        company_map_path=company_map_path,
        id_map_path=id_map_path,
        covenants_path=covenants_path,
        out_root=out_root,
    )

    facilities = pd.read_parquet(outputs["facilities"])
    revolvers = pd.read_parquet(outputs["revolver_facilities"])

    assert len(facilities) == 2
    assert len(revolvers) == 1
    assert revolvers.iloc[0]["ticker"] == "ABC"
    assert revolvers.iloc[0]["tranche_amount_converted_usd"] == 2_500_000_000.0
    assert revolvers.iloc[0]["wrds_facility_id"] == "222"
    assert revolvers.iloc[0]["max_leverage_ratio"] == "3.50:1"


