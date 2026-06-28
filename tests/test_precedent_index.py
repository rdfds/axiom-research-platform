from __future__ import annotations

from src.pipeline.precedent_index import build_precedent_index, query_precedent_index


def _dist(sample_size: int = 12) -> dict:
    return {
        "mean": 0.1,
        "median": 0.08,
        "p10": -0.2,
        "p25": -0.05,
        "p75": 0.2,
        "p90": 0.35,
        "sample_size": sample_size,
    }


def _metric_set(sample_size: int = 12) -> dict:
    return {
        "valuation_multiple_change": _dist(sample_size),
        "equity_return_vs_sector": _dist(sample_size),
        "credit_spread_change": _dist(sample_size),
        "rating_migration": _dist(sample_size),
        "leverage_change": _dist(sample_size),
        "fcf_change": _dist(sample_size),
        "volatility_change": _dist(sample_size),
    }


def _pack() -> dict:
    outcomes = {
        "horizon_1m": _metric_set(11),
        "horizon_6m": _metric_set(12),
        "horizon_12m": _metric_set(13),
        "horizon_24m": _metric_set(14),
    }
    return {
        "cohorts": [
            {
                "precedent_id": "p1",
                "company_id": "0001",
                "key_state_features": {"base_sector": "TECH"},
            }
        ],
        "outcome_distributions": outcomes,
        "regime_splits": [
            {"regime_label": "credit_tight", "outcome_distributions": outcomes},
            {"regime_label": "risk_on", "outcome_distributions": outcomes},
        ],
        "mismatch_diagnostics": {"out_of_sample_flag": False, "cohort_size": 30},
        "precedent_confidence": 0.77,
    }


