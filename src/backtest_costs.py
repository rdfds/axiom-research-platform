from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


_DEFAULT_FAMILY_MULTIPLIERS: Dict[str, float] = {
    "capital_return": 0.85,
    "capital_structure": 1.00,
    "mna": 1.25,
    "portfolio": 1.10,
    "restructuring": 0.95,
}


@dataclass(frozen=True)
class TransactionCostModel:
    key: str
    label: str
    commission_bps: float = 1.0
    half_spread_bps: float = 4.0
    slippage_bps: float = 4.0
    market_impact_bps: float = 3.0
    annual_short_borrow_bps: float = 30.0
    annual_financing_bps: float = 0.0
    family_multipliers: Mapping[str, float] = field(default_factory=dict)
    description: str = ""

    def multiplier_for_family(self, action_family: Optional[str]) -> float:
        family = str(action_family or "").strip()
        if not family:
            return 1.0
        if family in self.family_multipliers:
            return float(self.family_multipliers[family])
        return float(_DEFAULT_FAMILY_MULTIPLIERS.get(family, 1.0))

    def estimate_case_cost(
        self,
        *,
        action_family: Optional[str],
        holding_period_days: int,
        turnover_fraction: float = 1.0,
        short_exposure_fraction: float = 0.0,
        gross_exposure_multiple: float = 1.0,
    ) -> Dict[str, Any]:
        turnover = max(0.0, float(turnover_fraction or 0.0))
        holding_days = max(1, int(holding_period_days or 1))
        short_fraction = max(0.0, min(1.0, float(short_exposure_fraction or 0.0)))
        gross_multiple = max(1.0, float(gross_exposure_multiple or 1.0))
        family_multiplier = self.multiplier_for_family(action_family)

        # Round-trip trading friction on an event-driven rebalance.
        commission_component = 2.0 * float(self.commission_bps)
        spread_component = 2.0 * float(self.half_spread_bps)
        slippage_component = float(self.slippage_bps)
        impact_component = float(self.market_impact_bps)
        immediate_cost_bps = turnover * (
            commission_component
            + spread_component
            + slippage_component
            + impact_component
        )

        hold_years = float(holding_days) / 365.0
        borrow_component = hold_years * float(self.annual_short_borrow_bps) * short_fraction
        financing_component = hold_years * float(self.annual_financing_bps) * max(0.0, gross_multiple - 1.0)
        carry_cost_bps = borrow_component + financing_component

        total_cost_bps = family_multiplier * (immediate_cost_bps + carry_cost_bps)
        return {
            "model_key": self.key,
            "action_family": str(action_family or ""),
            "holding_period_days": holding_days,
            "turnover_fraction": round(turnover, 6),
            "short_exposure_fraction": round(short_fraction, 6),
            "gross_exposure_multiple": round(gross_multiple, 6),
            "family_multiplier": round(family_multiplier, 6),
            "components_bps": {
                "commission": round(family_multiplier * commission_component * turnover, 6),
                "bid_ask_spread": round(family_multiplier * spread_component * turnover, 6),
                "slippage": round(family_multiplier * slippage_component * turnover, 6),
                "market_impact": round(family_multiplier * impact_component * turnover, 6),
                "short_borrow_carry": round(family_multiplier * borrow_component, 6),
                "financing_carry": round(family_multiplier * financing_component, 6),
            },
            "total_cost_bps": round(total_cost_bps, 6),
            "total_cost_fraction": round(total_cost_bps / 10000.0, 8),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "commission_bps": float(self.commission_bps),
            "half_spread_bps": float(self.half_spread_bps),
            "slippage_bps": float(self.slippage_bps),
            "market_impact_bps": float(self.market_impact_bps),
            "annual_short_borrow_bps": float(self.annual_short_borrow_bps),
            "annual_financing_bps": float(self.annual_financing_bps),
            "family_multipliers": dict(self.family_multipliers or {}),
            "description": self.description,
        }


def default_transaction_cost_models() -> Dict[str, TransactionCostModel]:
    return {
        "manual_replay_event_equal_weight_v1": TransactionCostModel(
            key="manual_replay_event_equal_weight_v1",
            label="Manual Replay Event-Driven Equal-Weight Costs",
            commission_bps=1.0,
            half_spread_bps=4.0,
            slippage_bps=4.0,
            market_impact_bps=3.0,
            annual_short_borrow_bps=30.0,
            annual_financing_bps=0.0,
            description=(
                "Lightweight default friction model for historical recommendation replay. "
                "Designed to keep us honest about turnover and implementation drag without "
                "pretending we already have a full execution simulator."
            ),
        ),
        "manual_replay_conservative_v1": TransactionCostModel(
            key="manual_replay_conservative_v1",
            label="Manual Replay Conservative Friction",
            commission_bps=1.5,
            half_spread_bps=5.0,
            slippage_bps=6.0,
            market_impact_bps=4.0,
            annual_short_borrow_bps=60.0,
            annual_financing_bps=20.0,
            description="More conservative cost assumption for higher-friction governance and capital-structure moves.",
        ),
    }


def resolve_transaction_cost_model(model_key: Optional[str] = None) -> TransactionCostModel:
    models = default_transaction_cost_models()
    key = str(model_key or "manual_replay_event_equal_weight_v1")
    if key not in models:
        raise KeyError(f"Unknown transaction cost model '{key}'. Available: {sorted(models)}")
    return models[key]


__all__ = [
    "TransactionCostModel",
    "default_transaction_cost_models",
    "resolve_transaction_cost_model",
]
