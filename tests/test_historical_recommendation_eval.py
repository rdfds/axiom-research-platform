from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from src.historical_recommendation_eval import (
    _aggregate_historical_cases,
    _cached_snapshot_loader,
    _filter_excluded_historical_cases,
    _historical_case_key,
    _load_excluded_historical_case_keys,
    _load_fixed_historical_cases,
    _normalize_fixed_historical_case,
    _prefilter_support_is_eligible,
    _prioritize_historical_cases,
    build_historical_recommendation_report,
    render_historical_recommendation_markdown,
    _resolve_supported_historical_entities,
    _score_ex_post_alignment,
    _select_historical_cases_from_frame,
    _snapshot_coverage_summary,
    _snapshot_has_meaningful_coverage,
    _summarize_historical_selection_pool,
    _summarize_case_support_by_family,
)
from src.model_feature_bundle import feature_view_from_snapshot


@dataclass
class _DummySnapshot:
    company_id: str
    as_of_time: str
    features: dict


