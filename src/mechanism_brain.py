"""Mechanism Brain: evaluates action candidates without ranking.

Transforms ActionCandidateDraft-like records into ActionCandidate evaluations with:
- feasibility gating
- mechanism activation
- probabilistic counterfactual impact
- structural sanity flags
- risks and assumptions
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .causal_impact_model import (
    CausalImpactModel,
    get_causal_action_policy,
    load_default_causal_impact_model,
)
from .model_feature_bundle import feature_view_from_snapshot
from .runtime_feature_adapter import resolve_feature_value


_HARD_ACTIONS_HIGH_COMPLEXITY = {
    "mna.go_private_lbo",
    "mna.transformational_acquisition",
    "portfolio.spin_off",
    "portfolio.carve_out_ipo",
    "restructuring.chapter_pathway",
    "restructuring.out_of_court_restructuring",
}

_CASH_CONSUMING_ACTION_PREFIXES = (
    "capital_return.",
    "mna.",
)

_DEBT_REQUIRING_ACTION_IDS = {
    "capital_structure.new_debt_issuance",
    "capital_structure.refinancing",
    "capital_structure.tender_offer_debt",
    "capital_structure.exchange_offer",
    "capital_structure.liability_management_exercise",
    "capital_structure.revolver_draw_or_resize",
    "capital_structure.convertible_issuance",
}

_EQUITY_REQUIRING_ACTION_IDS = {
    "capital_structure.equity_issuance",
    "capital_structure.convertible_issuance",
    "capital_structure.preferred_issuance",
    "portfolio.carve_out_ipo",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _to_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    if isinstance(v, bool):
        return float(v)
    try:
        out = float(v)
    except Exception:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _feature_value(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw.get("value")
    return raw


def _extract_feature(features: Dict[str, Any], name: str, default: Any = None) -> Any:
    return resolve_feature_value(features, name, default=default)


