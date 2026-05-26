import importlib.util
from pathlib import Path

import pandas as pd


_SCRIPT_PATH = Path("./scripts/augment_action_outcomes_richer_contract.py")
_SPEC = importlib.util.spec_from_file_location("augment_action_outcomes_richer_contract", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_numeric_series_returns_aligned_nans_when_column_missing():
    df = pd.DataFrame({"base_total_debt": [10.0, 20.0, 30.0]})

    series = _MODULE._numeric_series(df, "macro_ig_oas")

    assert isinstance(series, pd.Series)
    assert list(series.index) == list(df.index)
    assert int(series.notna().sum()) == 0


def test_numeric_series_coerces_existing_column_to_numeric():
    df = pd.DataFrame({"macro_ig_oas": ["1.2", None, "bad"]})

    series = _MODULE._numeric_series(df, "macro_ig_oas")

    assert float(series.iloc[0]) == 1.2
    assert pd.isna(series.iloc[1])
    assert pd.isna(series.iloc[2])
