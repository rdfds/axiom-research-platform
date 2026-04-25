from src.metric_methodology import MetricMethodologyRegistry
from src.metric_policy import MetricPolicyEngine


CORE_FITCH_METRICS = [
    "capital_structure.total_debt",
    "capital_structure.net_debt",
    "capital_structure.gross_leverage",
    "capital_structure.net_leverage",
    "capital_structure.interest_coverage",
    "capital_structure.fixed_charge_coverage",
    "liquidity.usable_cash",
]


def test_metric_methodology_registry_is_well_formed():
    registry = MetricMethodologyRegistry()
    policy = MetricPolicyEngine()
    errors = registry.validate(expected_metric_ids=policy.policy.get("metrics", {}).keys())
    assert errors == []


