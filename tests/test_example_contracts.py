from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def test_company_state_example_has_provenance_rich_features() -> None:
    sample = _load("examples/company_state_snapshot/company_state_hd.sample.json")
    assert sample["ticker"] == "HD"
    assert len(sample["features"]) >= 10
    for feature in sample["features"].values():
        assert feature["as_of_time"]
        assert feature["computed_at"]
        assert feature["provenance"]
        assert "confidence" in feature


