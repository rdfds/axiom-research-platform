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


def test_company_state_builder_uses_axiom_data_root(monkeypatch):
    monkeypatch.setenv("AXIOM_DATA_ROOT", "/tmp/axiom_data_root")
    monkeypatch.setenv("AXIOM_COMPANYFACTS_ROOT", "/tmp/companyfacts_override")

    builder = CompanyStateBuilder()

    assert builder.raw_timeseries_path == Path("/tmp/axiom_data_root/inputs_layer/raw_timeseries.parquet")
    assert builder.facts_path == Path("/tmp/axiom_data_root/inputs_layer/extracted_fact_registry_validity")
    assert builder.taxonomy_reference_path == Path("/tmp/axiom_data_root/refinitiv/fundamentals_all.parquet")
    assert builder.companyfacts_root == Path("/tmp/companyfacts_override")


def test_named_builder_defaults_follow_axiom_data_root(monkeypatch):
    monkeypatch.setenv("AXIOM_DATA_ROOT", "/tmp/axiom_data_root")
    monkeypatch.setenv("AXIOM_COMPANYFACTS_ROOT", "/tmp/companyfacts_override")

    assert _default_facts_path() == Path("/tmp/axiom_data_root/inputs_layer/extracted_fact_registry_validity")
    assert _default_entity_table_path() == Path("/tmp/axiom_data_root/inputs_layer/entity.parquet")
    assert named_builder_companyfacts_root() == Path("/tmp/companyfacts_override")


def test_metric_goldens_companyfacts_root_follows_override(monkeypatch):
    monkeypatch.setenv("AXIOM_COMPANYFACTS_ROOT", "/tmp/companyfacts_override")

    assert resolve_companyfacts_root("data/sec/companyfacts") == Path("/tmp/companyfacts_override")
    assert metric_goldens_companyfacts_root() == Path("/tmp/companyfacts_override")
