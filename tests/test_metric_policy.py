from src.metric_policy import MetricPolicyEngine


def test_metric_policy_resolves_subsector_taxonomy():
    engine = MetricPolicyEngine()
    taxonomy = engine.resolve_taxonomy(
        "ABC",
        entity_row={
            "sector": "Consumer Discretionary",
            "subsector": "Specialty Retail",
            "sic": "5331",
        },
        fingerprints={},
    )
    assert taxonomy.archetype == "lease_heavy"
    assert taxonomy.override_level_applied == "subsector"
    assert taxonomy.support_mode == "exact"


def test_metric_policy_marks_financial_leverage_unsupported():
    engine = MetricPolicyEngine()
    taxonomy = engine.resolve_taxonomy(
        "BANK",
        entity_row={
            "sector": "Financials",
            "subsector": "Regional Banks",
            "sic": "6021",
        },
        fingerprints={},
    )
    assert taxonomy.archetype == "financial_institution"
    assert engine.resolve_applicability("capital_structure.net_leverage", taxonomy) == "unsupported"
    meta = engine.metric_metadata(
        "capital_structure.net_leverage",
        taxonomy,
        view_type="decision",
    )
    assert meta["support_mode"] == "unsupported"
    assert meta["applicability_status"] == "unsupported"


def test_metric_policy_resolves_consumer_staples_subsector():
    engine = MetricPolicyEngine()
    taxonomy = engine.resolve_taxonomy(
        "FOOD",
        entity_row={
            "sector": "Consumer Staples",
            "subsector": "Packaged Foods",
            "sic": "2090",
        },
        fingerprints={},
    )
    assert taxonomy.archetype == "consumer_branded_staples"
    assert taxonomy.override_level_applied == "subsector"


def test_metric_policy_prefers_gics_distribution_retail_over_broad_consumer_staples():
    engine = MetricPolicyEngine()
    taxonomy = engine.resolve_taxonomy(
        "WMT",
        entity_row={
            "gics_sector": "Consumer Staples",
            "gics_sub_industry": "Consumer Staples Distribution & Retail",
            "sector": "Consumer Staples",
            "subsector": "Consumer Staples Distribution & Retail",
            "sic": "5331",
        },
        fingerprints={},
    )
    assert taxonomy.archetype == "consumer_grocery_retail"
    assert taxonomy.override_level_applied == "subsector"
    assert taxonomy.support_mode == "exact"


