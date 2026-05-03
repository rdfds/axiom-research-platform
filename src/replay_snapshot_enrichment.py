from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import pandas as pd


MetricLoader = Callable[[Path], Optional[dict]]
MetricBuilder = Callable[[str, dict, str], Tuple[Optional[float], str, Optional[str], Optional[Dict[str, Any]], Optional[list[str]]]]

_COMPANYFACTS_METRICS: tuple[tuple[str, str], ...] = (
    ("operating.revenue_ttm_provider_direct", "usd"),
    ("operating.revenue_ttm_lag_1y", "usd"),
    ("liquidity.cash_and_short_term_investments_provider_direct", "usd"),
    ("capital_structure.total_debt_provider_direct", "usd"),
)
_EXACTISH_SUPPORT_MODES = {"exact", "exact_not_applicable", "exact_structural_zero"}
_MAX_SAFE_PRICE_STALENESS_DAYS = 21.0
_MAX_SAFE_REFERENCE_EV_RATIO = 2.0
_MAX_DAILY_ANCHOR_GAP_DAYS = 7


def _sec_metric_builders() -> tuple[MetricLoader, MetricBuilder]:
    try:
        from scripts.backfill_input_layer_v1_metrics import _build_sec_core_metric, _load_companyfacts
    except Exception:
        from backfill_input_layer_v1_metrics import _build_sec_core_metric, _load_companyfacts
    return _load_companyfacts, _build_sec_core_metric


def _price_history_loaders():
    try:
        from scripts.backfill_market_macro_input_layer_v1 import (
            _load_crsp_daily_from_repo,
            _load_crsp_market_cache,
        )
    except Exception:
        from backfill_market_macro_input_layer_v1 import (
            _load_crsp_daily_from_repo,
            _load_crsp_market_cache,
        )
    return _load_crsp_daily_from_repo, _load_crsp_market_cache


def _feature_record_needs_enrichment(raw: Any) -> bool:
    if raw is None:
        return True
    if not isinstance(raw, dict):
        return raw is None
    if raw.get("value") is None:
        return True
    support_mode = str(raw.get("support_mode") or "").strip().lower()
    return support_mode == "unsupported"


def _feature_value(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw.get("value")
    return raw


def _safe_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _support_mode(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    value = raw.get("support_mode")
    return str(value).strip().lower() if value is not None else None


