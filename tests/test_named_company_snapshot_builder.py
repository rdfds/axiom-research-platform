from __future__ import annotations

import json
from pathlib import Path

from src.named_company_snapshot_builder import (
    build_named_company_snapshots,
    required_fact_years,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_bytes(path: Path, payload: bytes = b"par1") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_required_fact_years_uses_inclusive_lookback():
    assert required_fact_years("2026-02-28", 5) == [2022, 2023, 2024, 2025, 2026]


def test_snapshot_builder_blocks_when_required_inputs_are_unreadable(tmp_path: Path):
    targets_path = tmp_path / "targets.json"
    facts_path = tmp_path / "facts"
    entity_path = tmp_path / "entity.parquet"
    taxonomy_path = tmp_path / "fundamentals.parquet"
    ratings_path = tmp_path / "issuer_rating_history.parquet"

    _write_json(
        targets_path,
        {
            "metadata": {},
            "targets": [
                {
                    "case_id": "walmart",
                    "company_id": "0000104169",
                    "ticker": "WMT",
                    "display_name": "Walmart Inc",
                    "as_of_date": "2026-02-28",
                    "expected_archetype": "consumer_grocery_retail",
                }
            ],
        },
    )
    _write_bytes(entity_path)
    _write_bytes(taxonomy_path)
    _write_bytes(ratings_path)
    # Leave fact shards absent so the builder reports a materialization blocker.

    report = build_named_company_snapshots(
        targets_path,
        snapshot_root=tmp_path / "snapshots",
        facts_path=facts_path,
        entity_table_path=entity_path,
        taxonomy_reference_path=taxonomy_path,
        issuer_ratings_path=ratings_path,
    )

    assert report["summary"]["blocked_unmaterialized_inputs"] == 1
    result = report["results"][0]
    assert result["build_status"] == "blocked_unmaterialized_inputs"
    assert any("year=2026/part.parquet" in path for path in result["blocked_input_paths"])


def test_snapshot_builder_writes_fresh_snapshot_and_excerpts_metrics(tmp_path: Path, monkeypatch):
    targets_path = tmp_path / "targets.json"
    facts_path = tmp_path / "facts"
    entity_path = tmp_path / "entity.parquet"
    taxonomy_path = tmp_path / "fundamentals.parquet"
    ratings_path = tmp_path / "issuer_rating_history.parquet"
    snapshot_root = tmp_path / "snapshots"

    _write_json(
        targets_path,
        {
            "metadata": {},
            "targets": [
                {
                    "case_id": "costco",
                    "company_id": "0000909832",
                    "ticker": "COST",
                    "display_name": "Costco Wholesale Corp",
                    "as_of_date": "2026-02-28",
                    "expected_archetype": "consumer_grocery_retail",
                }
            ],
        },
    )
    for year in [2022, 2023, 2024, 2025, 2026]:
        _write_bytes(facts_path / f"year={year}" / "part.parquet")
    _write_bytes(entity_path)
    _write_bytes(taxonomy_path)
    _write_bytes(ratings_path)

    class FakeBuilder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def build(self, company_id, as_of_date, extra_aliases=None):
            assert company_id == "0000909832"
            assert as_of_date == "2026-02-28"
            assert extra_aliases == ["COST"]
            return object()

    monkeypatch.setattr(
        "src.named_company_snapshot_builder.CompanyStateBuilder",
        FakeBuilder,
    )
    monkeypatch.setattr(
        "src.named_company_snapshot_builder.snapshot_to_json",
        lambda _: {
            "company_id": "0000909832",
            "as_of_time": "2026-02-28T00:00:00+00:00",
            "features": {
                "capital_structure.total_debt_market": {
                    "value": 123.0,
                    "unit": "usd",
                    "support_mode": "exact",
                    "applicability_status": "primary",
                    "missing_reason": None,
                    "fallback_used": None,
                    "quality_flags": [],
                    "component_breakdown": {"reported_debt": 100.0, "lease_weight": 0.75},
                }
            },
            "provenance": {
                "market_metric_context": {
                    "archetype": "consumer_grocery_retail",
                    "support_mode": "exact",
                }
            },
        },
    )

    report = build_named_company_snapshots(
        targets_path,
        snapshot_root=snapshot_root,
        facts_path=facts_path,
        entity_table_path=entity_path,
        taxonomy_reference_path=taxonomy_path,
        issuer_ratings_path=ratings_path,
    )

    assert report["summary"]["built"] == 1
    result = report["results"][0]
    assert result["build_status"] == "built"
    assert result["actual_archetype"] == "consumer_grocery_retail"
    assert result["archetype_match"] is True
    assert result["metrics"]["capital_structure.total_debt_market"]["value"] == 123.0

    snapshot_path = snapshot_root / "as_of_date=2026-02-28" / "company_id=0000909832.json"
    assert snapshot_path.exists()
