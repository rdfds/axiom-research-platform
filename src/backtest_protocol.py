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


def default_backtest_protocols() -> Dict[str, BacktestProtocol]:
    return {
        "manual_replay_default_v1": BacktestProtocol(
            key="manual_replay_default_v1",
            label="Manual Replay Canonical OOS",
            notes=(
                "Event-driven equal-weight protocol on frozen historical replay cases.",
                "Gross and net strategy scorecards are based on support-adjusted alignment scores.",
            ),
        ),
        "capital_return_holdout_v1": BacktestProtocol(
            key="capital_return_holdout_v1",
            label="Capital Return Holdout OOS",
            turnover_fraction=0.80,
            cost_model_key="manual_replay_event_equal_weight_v1",
            notes=(
                "Capital return actions are a little less implementation-heavy than M&A or portfolio moves.",
            ),
        ),
        "capital_structure_holdout_v1": BacktestProtocol(
            key="capital_structure_holdout_v1",
            label="Capital Structure Holdout OOS",
            turnover_fraction=1.00,
            cost_model_key="manual_replay_conservative_v1",
            notes=(
                "Uses the conservative friction profile because financing actions tend to be more path-dependent.",
            ),
        ),
    }


def infer_default_protocol_key(benchmark_key: Optional[str]) -> str:
    text = str(benchmark_key or "").strip().lower()
    if "capital_return" in text:
        return "capital_return_holdout_v1"
    if "capital_structure" in text:
        return "capital_structure_holdout_v1"
    return "manual_replay_default_v1"


def resolve_backtest_protocol(
    *,
    protocol_key: Optional[str] = None,
    benchmark_key: Optional[str] = None,
) -> BacktestProtocol:
    protocols = default_backtest_protocols()
    key = str(protocol_key or infer_default_protocol_key(benchmark_key))
    if key not in protocols:
        raise KeyError(f"Unknown backtest protocol '{key}'. Available: {sorted(protocols)}")
    return protocols[key]


__all__ = [
    "BacktestProtocol",
    "default_backtest_protocols",
    "infer_default_protocol_key",
    "resolve_backtest_protocol",
]
