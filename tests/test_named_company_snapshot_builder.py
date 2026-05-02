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


