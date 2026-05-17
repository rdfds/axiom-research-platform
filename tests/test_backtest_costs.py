from src.backtest_costs import resolve_transaction_cost_model


def test_estimate_case_cost_increases_for_higher_friction_family_and_short_borrow():
    model = resolve_transaction_cost_model("manual_replay_event_equal_weight_v1")

    capital_return = model.estimate_case_cost(
        action_family="capital_return",
        holding_period_days=120,
        turnover_fraction=1.0,
    )
    mna = model.estimate_case_cost(
        action_family="mna",
        holding_period_days=120,
        turnover_fraction=1.0,
    )
    shorted = model.estimate_case_cost(
        action_family="capital_structure",
        holding_period_days=120,
        turnover_fraction=1.0,
        short_exposure_fraction=0.5,
    )

    assert capital_return["total_cost_bps"] > 0.0
    assert mna["total_cost_bps"] > capital_return["total_cost_bps"]
    assert shorted["components_bps"]["short_borrow_carry"] > 0.0


def test_resolve_conservative_cost_model_exposes_expected_metadata():
    model = resolve_transaction_cost_model("manual_replay_conservative_v1")

    payload = model.to_dict()

    assert payload["key"] == "manual_replay_conservative_v1"
    assert payload["annual_financing_bps"] == 20.0
    assert "conservative" in payload["label"].lower()
