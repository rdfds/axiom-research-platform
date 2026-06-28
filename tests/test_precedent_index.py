from __future__ import annotations

from src.pipeline.precedent_index import build_precedent_index, query_precedent_index


def _dist(sample_size: int = 12) -> dict:
    return {
        "mean": 0.1,
        "median": 0.08,
        "p10": -0.2,
        "p25": -0.05,
        "p75": 0.2,
        "p90": 0.35,
        "sample_size": sample_size,
    }


def _metric_set(sample_size: int = 12) -> dict:
    return {
        "valuation_multiple_change": _dist(sample_size),
        "equity_return_vs_sector": _dist(sample_size),
        "credit_spread_change": _dist(sample_size),
        "rating_migration": _dist(sample_size),
        "leverage_change": _dist(sample_size),
        "fcf_change": _dist(sample_size),
        "volatility_change": _dist(sample_size),
    }


def _pack() -> dict:
    outcomes = {
        "horizon_1m": _metric_set(11),
        "horizon_6m": _metric_set(12),
        "horizon_12m": _metric_set(13),
        "horizon_24m": _metric_set(14),
    }
    return {
        "cohorts": [
            {
                "precedent_id": "p1",
                "company_id": "0001",
                "key_state_features": {"base_sector": "TECH"},
            }
        ],
        "outcome_distributions": outcomes,
        "regime_splits": [
            {"regime_label": "credit_tight", "outcome_distributions": outcomes},
            {"regime_label": "risk_on", "outcome_distributions": outcomes},
        ],
        "mismatch_diagnostics": {"out_of_sample_flag": False, "cohort_size": 30},
        "precedent_confidence": 0.77,
    }


def test_build_precedent_index_creates_candidate_and_distribution_rows():
    matches = [
        {
            "candidate": {
                "candidate_id": "c1",
                "action_type": "capital_return",
                "action_subtype": "open_market_buyback",
                "action_id": "capital_return.open_market_buyback",
            },
            "precedent_pack": _pack(),
        }
    ]
    idx = build_precedent_index(run_id="run-1", precedent_matches=matches)
    assert idx.get("index_version")
    assert idx["counts"]["candidates"] == 1
    assert idx["counts"]["distribution_rows"] > 0


def test_query_precedent_index_filters_by_action_regime_horizon():
    matches = [
        {
            "candidate": {
                "candidate_id": "c1",
                "action_type": "capital_return",
                "action_subtype": "open_market_buyback",
                "action_id": "capital_return.open_market_buyback",
            },
            "precedent_pack": _pack(),
        },
        {
            "candidate": {
                "candidate_id": "c2",
                "action_type": "capital_structure",
                "action_subtype": "refinancing",
                "action_id": "capital_structure.refinancing",
            },
            "precedent_pack": _pack(),
        },
    ]
    idx = build_precedent_index(run_id="run-2", precedent_matches=matches)
    out = query_precedent_index(
        idx,
        action_type="capital_return",
        regime="credit_tight",
        time_horizon="12m",
        limit=50,
    )
    assert out["count"] > 0
    assert all(r["action_type"] == "capital_return" for r in out["rows"])
    assert all(r["regime_label"] == "credit_tight" for r in out["rows"])
    assert all(r["time_horizon"] == "12m" for r in out["rows"])


def test_query_precedent_index_supports_min_sample_size_and_oos_filter():
    matches = [
        {
            "candidate": {
                "candidate_id": "c1",
                "action_type": "capital_return",
                "action_subtype": "open_market_buyback",
                "action_id": "capital_return.open_market_buyback",
            },
            "precedent_pack": _pack(),
        },
        {
            "candidate": {
                "candidate_id": "c2",
                "action_type": "capital_return",
                "action_subtype": "open_market_buyback",
                "action_id": "capital_return.open_market_buyback",
            },
            "precedent_pack": {
                **_pack(),
                "mismatch_diagnostics": {"out_of_sample_flag": True, "cohort_size": 5},
            },
        },
    ]
    idx = build_precedent_index(run_id="run-3", precedent_matches=matches)

    out = query_precedent_index(
        idx,
        action_type="capital_return",
        regime="all",
        time_horizon="12m",
        min_sample_size=13,
        exclude_out_of_sample=True,
        limit=200,
    )
    assert out["count"] > 0
    assert all(int(r["sample_size"]) >= 13 for r in out["rows"])
    assert all(bool(r["out_of_sample_flag"]) is False for r in out["rows"])


def test_sector_enriched_from_cohort_company_ids(monkeypatch):
    import src.pipeline.precedent_index as pi

    monkeypatch.setattr(
        pi,
        "_load_gvkey_sector_map",
        lambda: {"012421": "MANUFACTURING"},
    )

    matches = [
        {
            "candidate": {
                "candidate_id": "c1",
                "action_type": "capital_return",
                "action_subtype": "open_market_buyback",
                "action_id": "capital_return.open_market_buyback",
            },
            "precedent_pack": {
                **_pack(),
                "cohorts": [
                    {"company_id": "12421", "key_state_features": {}},
                    {"company_id": "012421", "key_state_features": {}},
                ],
            },
        }
    ]
    idx = build_precedent_index(run_id="run-4", precedent_matches=matches)
    assert idx["candidate_rows"][0]["sector"] == "MANUFACTURING"


def test_query_precedent_index_ranks_high_confidence_rows_first():
    high = _pack()
    high["precedent_confidence"] = 0.90
    low = _pack()
    low["precedent_confidence"] = 0.20
    low["mismatch_diagnostics"] = {"out_of_sample_flag": True, "cohort_size": 30}

    idx = build_precedent_index(
        run_id="run-5",
        precedent_matches=[
            {
                "candidate": {
                    "candidate_id": "c1",
                    "action_type": "capital_return",
                    "action_subtype": "open_market_buyback",
                    "action_id": "capital_return.open_market_buyback",
                },
                "precedent_pack": low,
            },
            {
                "candidate": {
                    "candidate_id": "c2",
                    "action_type": "capital_return",
                    "action_subtype": "open_market_buyback",
                    "action_id": "capital_return.open_market_buyback",
                },
                "precedent_pack": high,
            },
        ],
    )

    out = query_precedent_index(
        idx,
        action_type="capital_return",
        regime="all",
        time_horizon="12m",
        limit=10,
    )
    assert out["count"] > 0
    assert float(out["rows"][0]["precedent_confidence"]) >= float(out["rows"][-1]["precedent_confidence"])
    assert float(out["rows"][0]["query_score"]) >= float(out["rows"][-1]["query_score"])


def test_query_precedent_index_supports_min_precedent_confidence():
    weak = _pack()
    weak["precedent_confidence"] = 0.12
    strong = _pack()
    strong["precedent_confidence"] = 0.82
    idx = build_precedent_index(
        run_id="run-6",
        precedent_matches=[
            {
                "candidate": {
                    "candidate_id": "c1",
                    "action_type": "capital_return",
                    "action_subtype": "open_market_buyback",
                    "action_id": "capital_return.open_market_buyback",
                },
                "precedent_pack": weak,
            },
            {
                "candidate": {
                    "candidate_id": "c2",
                    "action_type": "capital_return",
                    "action_subtype": "open_market_buyback",
                    "action_id": "capital_return.open_market_buyback",
                },
                "precedent_pack": strong,
            },
        ],
    )
    out = query_precedent_index(
        idx,
        action_type="capital_return",
        regime="all",
        time_horizon="12m",
        min_precedent_confidence=0.5,
        limit=50,
    )
    assert out["count"] > 0
    assert all(float(row["precedent_confidence"]) >= 0.5 for row in out["rows"])
