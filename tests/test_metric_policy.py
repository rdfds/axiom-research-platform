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


def test_metric_policy_resolves_transport_logistics_ahead_of_broad_transport():
    engine = MetricPolicyEngine()
    taxonomy = engine.resolve_taxonomy(
        "TRNS",
        entity_row={
            "sector": "Industrials",
            "subsector": "Air Freight & Logistics",
            "sic": "4213",
        },
        fingerprints={},
    )
    assert taxonomy.archetype == "transport_logistics"
    assert taxonomy.override_level_applied == "subsector"


def test_metric_policy_resolves_aerospace_defense():
    engine = MetricPolicyEngine()
    taxonomy = engine.resolve_taxonomy(
        "AERO",
        entity_row={
            "sector": "Industrials",
            "subsector": "Aerospace & Defense",
            "sic": "3721",
        },
        fingerprints={},
    )
    assert taxonomy.archetype == "aerospace_defense"
    assert taxonomy.override_level_applied == "subsector"


def test_metric_policy_resolves_automotive_oem():
    engine = MetricPolicyEngine()
    taxonomy = engine.resolve_taxonomy(
        "AUTO",
        entity_row={
            "sector": "Consumer Discretionary",
            "subsector": "Automobiles",
            "sic": "3711",
        },
        fingerprints={},
    )
    assert taxonomy.archetype == "automotive_oem"


def test_metric_policy_resolves_professional_services():
    engine = MetricPolicyEngine()
    taxonomy = engine.resolve_taxonomy(
        "PROF",
        entity_row={
            "sector": "Industrials",
            "subsector": "Professional Services",
            "sic": "7370",
        },
        fingerprints={},
    )
    assert taxonomy.archetype == "professional_services"


def test_metric_policy_falls_back_when_policy_file_read_times_out(tmp_path, monkeypatch):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("{}")

    original_read_text = type(policy_path).read_text

    def _boom(self, *args, **kwargs):
        if self == policy_path:
            raise TimeoutError("dataless placeholder")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(type(policy_path), "read_text", _boom)

    engine = MetricPolicyEngine(policy_path=policy_path)
    taxonomy = engine.resolve_taxonomy(
        "ABC",
        entity_row={"sector": "Industrials", "subsector": "Machinery"},
        fingerprints={},
    )

    assert engine.policy_id == "market_metric_policy_v1"
    assert engine.metric_definition("capital_structure.net_leverage") == {}
    assert engine.resolve_applicability("capital_structure.net_leverage", taxonomy) == "primary"
    assert engine.sector_field_candidates()
