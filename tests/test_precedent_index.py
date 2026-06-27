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


