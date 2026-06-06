from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .precedent_brain import (
    _STATE_VECTOR_BASE_WEIGHTS,
    _STATE_VECTOR_MATCHING_COLS,
    augment_precedent_state_vector_columns,
)


_OUTCOME_SPECS: Dict[str, Dict[str, float]] = {
    "ALL": {
        "outcome_pe_6m": 0.18,
        "outcome_pe_12m": 0.22,
        "outcome_ev_ebitda_6m": 0.14,
        "outcome_ev_ebitda_12m": 0.18,
        "credit_spread_change_6m": 0.10,
        "credit_spread_change_12m": 0.10,
        "rating_migration_6m": 0.03,
        "rating_migration_12m": 0.03,
        "leverage_delta": 0.12,
        "fcf_margin_delta": 0.10,
    },
    "capital_return": {
        "outcome_pe_6m": 0.22,
        "outcome_pe_12m": 0.28,
        "outcome_ev_ebitda_6m": 0.16,
        "outcome_ev_ebitda_12m": 0.18,
        "leverage_delta": 0.06,
        "fcf_margin_delta": 0.10,
    },
    "capital_structure": {
        "credit_spread_change_6m": 0.20,
        "credit_spread_change_12m": 0.20,
        "rating_migration_6m": 0.10,
        "rating_migration_12m": 0.10,
        "leverage_delta": 0.20,
        "fcf_margin_delta": 0.08,
        "outcome_ev_ebitda_12m": 0.07,
        "outcome_pe_12m": 0.05,
    },
    "mna": {
        "outcome_pe_6m": 0.18,
        "outcome_pe_12m": 0.24,
        "outcome_ev_ebitda_6m": 0.14,
        "outcome_ev_ebitda_12m": 0.18,
        "leverage_delta": 0.10,
        "fcf_margin_delta": 0.16,
    },
    "portfolio": {
        "outcome_pe_6m": 0.18,
        "outcome_pe_12m": 0.24,
        "outcome_ev_ebitda_6m": 0.14,
        "outcome_ev_ebitda_12m": 0.18,
        "leverage_delta": 0.12,
        "fcf_margin_delta": 0.14,
    },
}


def _clean_scope_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _state_feature_names() -> Tuple[str, ...]:
    return tuple(_STATE_VECTOR_MATCHING_COLS)


