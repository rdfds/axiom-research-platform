import time

import pytest

from scripts.backfill_input_layer_v1_metrics import (
    _CompanyProcessingTimeout,
    _build_legacy_provider_metric,
    _build_sec_core_metric,
    _build_fail_open_metric_set,
    _company_processing_guard,
    _compute_ttm_from_concept,
    _select_preferred_direct_metric,
)


def _instant_fact(value: float):
    return {
        "val": value,
        "end": "2024-09-30",
        "filed": "2024-10-31",
        "fy": 2024,
        "fp": "Q3",
        "form": "10-Q",
        "frame": "CY2024Q3I",
    }


def _duration_fact(value: float, *, start: str, end: str, filed: str, fy: int, fp: str, form: str = "10-Q", frame=None):
    fact = {
        "val": value,
        "start": start,
        "end": end,
        "filed": filed,
        "fy": fy,
        "fp": fp,
        "form": form,
    }
    if frame is not None:
        fact["frame"] = frame
    return fact


def _metric_node(*, value, support_mode, primary_source_basis, missing_reason=None, quality_flags=None, breakdown=None):
    return {
        "value": value,
        "support_mode": support_mode,
        "primary_source_basis": primary_source_basis,
        "missing_reason": missing_reason,
        "quality_flags": quality_flags,
        "component_breakdown": breakdown or {},
    }


