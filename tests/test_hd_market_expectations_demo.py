from __future__ import annotations

import json
from pathlib import Path

from scripts.build_hd_market_expectations_demo import build_demo


def test_hd_market_expectations_sample_rebuilds_static_demo(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    sample_dir = root / "examples" / "hd_market_expectations"
    html = build_demo(sample_dir, tmp_path / "hd_demo")

    payload = json.loads((sample_dir / "valuation_driver_data.sample.json").read_text())
    assert [company["ticker"] for company in payload["companies"]] == ["HD"]
    assert payload["companies"][0]["forward_expectations"]["forward_grade"] == "strong"

    rendered = html.read_text()
    assert "Home Depot" in rendered
    assert "Market expectations" in rendered
    assert "const DATA=" in rendered
