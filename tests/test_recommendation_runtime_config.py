from __future__ import annotations

from src.causal_impact_model import (
    DEFAULT_CAUSAL_IMPACT_MODEL_ARTIFACT,
    DEFAULT_CAUSAL_ROUTING_CONFIG_PATH,
    _default_model_path,
)
from src.recommendation_runtime_config import (
    DEFAULT_PRECEDENT_RETRIEVAL_VERSION,
    capture_runtime_env_config,
)


def test_default_causal_model_path_points_to_current_champion(monkeypatch):
    monkeypatch.delenv("CAUSAL_IMPACT_MODEL_PATH", raising=False)

    assert _default_model_path() == DEFAULT_CAUSAL_IMPACT_MODEL_ARTIFACT


def test_capture_runtime_env_config_uses_current_champion_default(monkeypatch):
    monkeypatch.delenv("CAUSAL_IMPACT_MODEL_PATH", raising=False)
    monkeypatch.delenv("CAUSAL_ROUTING_CONFIG_PATH", raising=False)

    config = capture_runtime_env_config()

    assert config["causal"]["model"]["path"] == str(DEFAULT_CAUSAL_IMPACT_MODEL_ARTIFACT)
    assert config["causal"]["routing"]["path"] == str(DEFAULT_CAUSAL_ROUTING_CONFIG_PATH)


def test_capture_runtime_env_config_uses_compact_precedent_default(monkeypatch):
    monkeypatch.delenv("PRECEDENT_RETRIEVAL_VERSION", raising=False)

    config = capture_runtime_env_config()

    assert config["precedent"]["retrieval_version"] == DEFAULT_PRECEDENT_RETRIEVAL_VERSION
