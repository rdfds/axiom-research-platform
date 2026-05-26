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


