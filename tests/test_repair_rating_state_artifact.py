import pandas as pd

from scripts.repair_rating_state_artifact import (
    build_rating_index,
    repair_rating_state,
)


def _node(name, value, *, support_mode="unsupported", unit="rating"):
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "computed_at": "2026-03-23T00:00:00+00:00",
        "as_of_time": "2024-12-31T00:00:00+00:00",
        "window": None,
        "confidence": None,
        "provenance": [],
        "missing_reason": None if value is not None else "not_disclosed",
        "fallback_used": None,
        "support_mode": support_mode,
        "component_breakdown": None,
        "quality_flags": None,
    }


def test_build_rating_index_prefers_fitch_rows_and_latest_date():
    df = pd.DataFrame(
        [
            {
                "company_id": "0000001111",
                "rating_symbol": "BB",
                "outlook": "Stable",
                "creditwatch": "N",
                "source_type": "moodys",
                "rating_date": "2024-12-20",
            },
            {
                "company_id": "0000001111",
                "rating_symbol": "BB-",
                "outlook": "Negative",
                "creditwatch": "Y",
                "source_type": "fitch",
                "rating_date": "2024-11-20",
            },
        ]
    )
    df["rating_date"] = pd.to_datetime(df["rating_date"], utc=True)

    index = build_rating_index(df)

    payload = index["0000001111"]["payload"]
    assert payload["rating"] == "BB-"
    assert payload["outlook"] == "Negative"
    assert payload["watchlist"] is True
    assert payload["score"] == 12.0


def test_repair_rating_state_from_index():
    features = {
        "capital_structure.rating_state": _node("capital_structure.rating_state", None),
    }
    df = pd.DataFrame(
        [
            {
                "company_id": "3197",
                "rating_symbol": "BBB-",
                "outlook": "Stable",
                "creditwatch": "N",
                "source_type": "fitch",
                "rating_date": pd.Timestamp("2024-12-15", tz="UTC"),
                "artifact_id": "issuer_rating:3197:2024-12-15",
            }
        ]
    )
    index = build_rating_index(df)

    repaired = repair_rating_state(
        features=features,
        company_id="0000003197",
        rating_index=index,
        computed_at="2026-03-23T00:00:00+00:00",
    )

    assert repaired is True
    node = features["capital_structure.rating_state"]
    assert node["support_mode"] == "exact"
    assert node["value"]["rating"] == "BBB-"
    assert node["value"]["score"] == 10.0
    assert node["component_breakdown"]["selection_rule"] == "latest_rating_prefer_fitch"
