from __future__ import annotations

from typing import Any, Dict

from .types import ActionCandidate


_ACTION_EFFECT_ALIASES: Dict[str, str] = {
    "capital_return.open_market_buyback": "buyback",
    "capital_return.accelerated_share_repurchase": "buyback",
    "capital_return.tender_offer_buyback": "buyback",
    "capital_return.dividend_increase": "dividend",
    "capital_return.dividend_cut": "dividend",
    "capital_return.dividend_initiate": "dividend",
    "capital_return.special_dividend": "dividend",
    "capital_structure.new_debt_issuance": "debt_issuance",
    "capital_structure.refinancing": "refinancing",
    "capital_structure.equity_issuance": "equity_issuance",
    "capital_structure.convertible_issuance": "equity_issuance",
    "mna.tuck_in_acquisition": "acquisition",
    "mna.platform_acquisition": "acquisition",
    "mna.go_private_lbo": "acquisition",
    "mna.transformational_acquisition": "acquisition",
    "mna.minority_investment": "acquisition",
    "portfolio.divestiture_full": "divestiture",
    "portfolio.divestiture_partial": "divestiture",
    "portfolio.asset_sale": "asset_sale",
    "portfolio.spin_off": "spin_off",
    "governance.stock_split": "stock_split",
    "restructuring.cost_program": "cost_program",
}


def build_change_vector(action: ActionCandidate, config: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert an ActionCandidate into an expected metric change vector.
    Uses config-defined action_effects and scales by action params.
    """
    action_effects = config.get("action_effects", {})
    effect_keys = []
    if action.action_id:
        effect_keys.append(action.action_id)
        alias = _ACTION_EFFECT_ALIASES.get(action.action_id)
        if alias:
            effect_keys.append(alias)
    if action.action_subtype:
        effect_keys.append(action.action_subtype)
    effect_keys.append(action.action_type)

    effects = {}
    for key in effect_keys:
        if key in action_effects:
            effects = action_effects[key]
            break
    if not effects:
        # Allow direct metric_deltas on params for ad-hoc scenarios
        return {k: float(v) for k, v in action.params.get("metric_deltas", {}).items()}

    scale = float(action.params.get("size", 1.0))
    change = {}
    for metric, base_delta in effects.items():
        try:
            change[metric] = float(base_delta) * scale
        except Exception:
            continue
    return change
