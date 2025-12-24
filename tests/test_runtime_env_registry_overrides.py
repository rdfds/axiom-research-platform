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


