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


def test_metric_methodology_registry_uses_fitch_for_core_credit_metrics():
    registry = MetricMethodologyRegistry()
    for metric_id in CORE_FITCH_METRICS:
        entry = registry.metric(metric_id)
        assert entry["canonical_owner_id"] == "fitch_ratings"
        assert entry["market_layer_status"] == "keep"
        assert entry["canonical_classification"] == "canonical_external"


def test_metric_methodology_registry_demotes_internal_only_metrics():
    registry = MetricMethodologyRegistry()

    available = registry.metric("liquidity.available_for_actions")
    assert available["canonical_owner_id"] == "axiom_internal"
    assert available["market_layer_status"] == "rename"
    assert available["recommended_metric_name"] == "liquidity.readily_available_liquidity"

    runway = registry.metric("liquidity.runway_months")
    assert runway["canonical_classification"] == "internal_only"
    assert runway["market_layer_status"] == "retire"

    refi = registry.metric("capital_structure.refi_pressure_flag")
    assert refi["canonical_classification"] == "internal_only"
    assert refi["market_layer_status"] == "retire"


def test_metric_methodology_registry_uses_filing_native_for_maturity_wall():
    registry = MetricMethodologyRegistry()
    entry = registry.metric("capital_structure.maturity_wall_ratio_24m")
    assert entry["canonical_owner_id"] == "issuer_filing_native"
    assert entry["canonical_classification"] == "filing_native_external"
    assert entry["market_layer_status"] == "keep"
