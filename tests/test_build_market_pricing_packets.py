from scripts.build_market_pricing_packets import build_packets


def _node(name, value):
    return {"name": name, "value": value}


def _row(company_id, **features):
    return {
        "company_id": company_id,
        "features": {name: _node(name, value) for name, value in features.items()},
    }


def test_build_packets_assigns_expected_lists():
    rows = [
        _row(
            "0001",
            **{
                "market.comp_overall_score": 75.0,
                "market.valuation_gap_score": 40.0,
                "market.value_score": 30.0,
                "market.quality_score": 80.0,
                "market.balance_sheet_score": 78.0,
                "market.risk_score": 72.0,
            },
        ),
        _row(
            "0002",
            **{
                "market.comp_overall_score": 42.0,
                "market.valuation_gap_score": -10.0,
                "market.value_score": 65.0,
                "market.quality_score": 18.0,
                "market.balance_sheet_score": 20.0,
                "market.risk_score": 25.0,
            },
        ),
        _row(
            "0003",
            **{
                "market.comp_overall_score": 60.0,
                "market.valuation_gap_score": 32.0,
                "market.value_score": 28.0,
                "market.quality_score": 72.0,
                "market.balance_sheet_score": 66.0,
                "market.risk_score": 58.0,
            },
        ),
    ]

    packets = build_packets(rows, companyfacts_root=None, limit=3)

    assert packets["top_longs"][0]["company_id"] == "0001"
    assert packets["fragile_shorts"][0]["company_id"] == "0002"
    ids = [row["company_id"] for row in packets["mispriced_quality"]]
    assert ids[:2] == ["0001", "0003"]
