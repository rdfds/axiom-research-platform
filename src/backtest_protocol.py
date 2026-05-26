from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class BacktestProtocol:
    key: str
    label: str
    benchmark_mode: str = "reference_metrics"
    weighting_mode: str = "equal_weight"
    rebalance_mode: str = "event_driven"
    score_field: str = "historical_alignment.score"
    holding_period_days: int = 120
    turnover_fraction: float = 1.0
    top_actions_per_case: int = 1
    strong_alignment_threshold: float = 0.60
    min_case_count: int = 10
    cost_model_key: str = "manual_replay_event_equal_weight_v1"
    notes: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "benchmark_mode": self.benchmark_mode,
            "weighting_mode": self.weighting_mode,
            "rebalance_mode": self.rebalance_mode,
            "score_field": self.score_field,
            "holding_period_days": int(self.holding_period_days),
            "turnover_fraction": float(self.turnover_fraction),
            "top_actions_per_case": int(self.top_actions_per_case),
            "strong_alignment_threshold": float(self.strong_alignment_threshold),
            "min_case_count": int(self.min_case_count),
            "cost_model_key": self.cost_model_key,
            "notes": list(self.notes),
        }


