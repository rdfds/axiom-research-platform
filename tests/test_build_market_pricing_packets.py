from scripts.build_market_pricing_packets import build_packets


def _node(name, value):
    return {"name": name, "value": value}


def _row(company_id, **features):
    return {
        "company_id": company_id,
        "features": {name: _node(name, value) for name, value in features.items()},
    }


