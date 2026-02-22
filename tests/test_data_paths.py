from __future__ import annotations

from pathlib import Path

from src.company_state_builder import CompanyStateBuilder
from src.data_paths import resolve_companyfacts_root, resolve_data_path
from src.metric_goldens import _default_companyfacts_root as metric_goldens_companyfacts_root
from src.named_company_snapshot_builder import (
    _default_companyfacts_root as named_builder_companyfacts_root,
    _default_entity_table_path,
    _default_facts_path,
)


def test_resolve_data_path_rewrites_repo_relative_data_path(monkeypatch):
    monkeypatch.setenv("AXIOM_DATA_ROOT", "/tmp/axiom_data_root")

    resolved = resolve_data_path("data/inputs_layer/raw_timeseries.parquet")

    assert resolved == Path("/tmp/axiom_data_root/inputs_layer/raw_timeseries.parquet")


def test_resolve_data_path_rewrites_absolute_repo_data_path(monkeypatch):
    monkeypatch.setenv("AXIOM_DATA_ROOT", "/tmp/axiom_data_root")
    repo_data_path = Path("./data/curated/action_outcomes.parquet")

    resolved = resolve_data_path(repo_data_path)

    assert resolved == Path("/tmp/axiom_data_root/curated/action_outcomes.parquet")


