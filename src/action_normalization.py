from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


NORMALIZATION_RULES_VERSION = "v2_lossless_20260317"


def _norm_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().lower()


