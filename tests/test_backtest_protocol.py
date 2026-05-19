from src.backtest_protocol import infer_default_protocol_key, resolve_backtest_protocol


def test_infer_default_protocol_key_maps_family_specific_benchmarks():
    assert infer_default_protocol_key("capital_return_holdout") == "capital_return_holdout_v1"
    assert infer_default_protocol_key("capital_structure_holdout") == "capital_structure_holdout_v1"
    assert infer_default_protocol_key("something_else") == "manual_replay_default_v1"


def test_resolve_backtest_protocol_uses_benchmark_when_protocol_not_supplied():
    protocol = resolve_backtest_protocol(benchmark_key="capital_structure_holdout")

    assert protocol.key == "capital_structure_holdout_v1"
    assert protocol.cost_model_key == "manual_replay_conservative_v1"
    assert protocol.turnover_fraction == 1.0
