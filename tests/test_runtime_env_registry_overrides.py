from __future__ import annotations

import json

import pandas as pd

from src.company_state_builder import CompanyStateBuilder
from src.company_state_input_source_registry import CompanyStateInputSourceRegistry
from src.metric_methodology import MetricMethodologyRegistry
from src.metric_policy import MetricPolicyEngine


def test_metric_methodology_registry_honors_late_env_override(tmp_path, monkeypatch):
    registry_path = tmp_path / "methodology.json"
    registry_path.write_text(json.dumps({"registry_id": "late_override", "metrics": {}}))
    monkeypatch.setenv("AXIOM_METHODOLOGY_REGISTRY_PATH", str(registry_path))

    registry = MetricMethodologyRegistry()

    assert registry.registry_path == registry_path
    assert registry.registry_id == "late_override"


def test_input_source_registry_honors_late_env_override(tmp_path, monkeypatch):
    registry_path = tmp_path / "input_sources.json"
    registry_path.write_text(json.dumps({"registry_id": "input_override", "owners": {}, "metrics": {}}))
    monkeypatch.setenv("AXIOM_INPUT_SOURCE_REGISTRY_PATH", str(registry_path))

    registry = CompanyStateInputSourceRegistry()

    assert registry.registry_path == registry_path
    assert registry.registry_id == "input_override"


def test_metric_policy_engine_honors_late_env_override(tmp_path, monkeypatch):
    policy_path = tmp_path / "policy.json"
    methodology_path = tmp_path / "methodology.json"
    policy_path.write_text(json.dumps({"policy_id": "policy_override", "taxonomy": {"archetypes": {}}, "metrics": {}}))
    methodology_path.write_text(json.dumps({"registry_id": "methodology_override", "metrics": {}}))
    monkeypatch.setenv("AXIOM_METRIC_POLICY_PATH", str(policy_path))
    monkeypatch.setenv("AXIOM_METHODOLOGY_REGISTRY_PATH", str(methodology_path))

    engine = MetricPolicyEngine()

    assert engine.policy_path == policy_path
    assert engine.policy_id == "policy_override"
    assert engine.methodology_registry.registry_path == methodology_path


