import importlib.util
from pathlib import Path

import pandas as pd


_SCRIPT_PATH = Path("./scripts/51_build_action_outcomes.py")
_SPEC = importlib.util.spec_from_file_location("build_action_outcomes", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_resolved_action_size_prefers_positive_split_ratio_over_zero_amount():
    row = pd.Series(
        {
            "action_type": "reverse_split",
            "size": 0.0,
            "amount": 0.0,
            "ratio": 0.05,
            "split_factor": None,
            "facpr": 0.05,
        }
    )

    assert _MODULE._resolved_action_size(row) == 0.05


def test_build_action_outcomes_fast_uses_positive_split_ratio(tmp_path: Path):
    actions_path = tmp_path / "actions.parquet"
    fundamentals_path = tmp_path / "fundamentals.parquet"
    out_path = tmp_path / "outcomes.parquet"

    pd.DataFrame(
        [
            {
                "company_id": "123456",
                "action_type": "reverse_split",
                "action_subtype": "forward",
                "action_date": pd.Timestamp("2024-06-01"),
                "size": 0.0,
                "amount": 0.0,
                "ratio": 0.05,
                "ticker": "TEST",
            }
        ]
    ).to_parquet(actions_path, index=False)

    pd.DataFrame(
        [
            {
                "gvkey": "123456",
                "datadate": pd.Timestamp("2024-03-31"),
                "revtq": 100.0,
                "oibdpq": 10.0,
                "niq": 5.0,
                "epspxq": 1.0,
                "cshoq": 10.0,
                "cheq": 5.0,
                "dlttq": 20.0,
                "dlcq": 2.0,
                "atq": 200.0,
                "oancfy": 12.0,
                "capxy": 4.0,
                "prccq": 10.0,
                "mkvaltq": 100.0,
            }
        ]
    ).to_parquet(fundamentals_path, index=False)

    _MODULE.build_action_outcomes_fast(
        actions_path=actions_path,
        out_path=out_path,
        action_types=["reverse_split"],
        start_date=None,
        end_date=None,
        date_field="auto",
        config_path=None,
        fundamentals_path=str(fundamentals_path),
        horizons_override=[3],
        include_macro=False,
        duckdb_memory=None,
        duckdb_threads=1,
        duckdb_preserve_order=True,
    )

    out = pd.read_parquet(out_path)
    assert len(out) == 1
    assert out.loc[0, "action_type"] == "reverse_split"
    assert abs(float(out.loc[0, "action_size"]) - 0.05) < 1e-9

