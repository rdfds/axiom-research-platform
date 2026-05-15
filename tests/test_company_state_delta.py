from src.company_state_delta import update_snapshot


class DummyBuilder:
    def _load_facts(self, cid, asof):
        return None

    def _load_timeseries(self, cid, asof):
        return None

    def _load_macro(self, asof):
        return None

    def _compute_market(self, ts, facts, asof):
        return {
            "market.market_cap": {
                "name": "market.market_cap",
                "value": 123.0,
                "as_of_time": asof.isoformat(),
                "computed_at": asof.isoformat(),
                "window": None,
                "confidence": None,
                "provenance": [],
                "unit": "usd",
                "missing_reason": None,
                "fallback_used": None,
            }
        }

    def _compute_regime(self, macro, asof):
        return {"credit_regime": "neutral"}


def _snapshot():
    return {
        "company_id": "001",
        "as_of_time": "2026-02-01T00:00:00Z",
        "features": {},
        "regime": {},
    }


def test_update_market_only():
    snap = _snapshot()
    out = update_snapshot(snap, DummyBuilder(), "2026-02-28", mode="market")
    assert "market.market_cap" in out["features"]
    assert out["regime"] == {}


def test_update_regime_only():
    snap = _snapshot()
    out = update_snapshot(snap, DummyBuilder(), "2026-02-28", mode="regime")
    assert out["regime"]["credit_regime"] == "neutral"
    assert out["features"] == {}


def test_update_both():
    snap = _snapshot()
    out = update_snapshot(snap, DummyBuilder(), "2026-02-28", mode="both")
    assert "market.market_cap" in out["features"]
    assert out["regime"]["credit_regime"] == "neutral"
