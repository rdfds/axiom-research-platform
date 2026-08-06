from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json


ROOT = Path(__file__).resolve().parents[1]


def _default_methodology_registry_path() -> Path:
    env = str(os.environ.get("AXIOM_METHODOLOGY_REGISTRY_PATH", "") or "").strip()
    if env:
        return Path(env)
    return ROOT / "configs" / "metric_methodologies" / "consumer_industrials_canonical_registry_v1.json"


class MetricMethodologyRegistry:
    def __init__(self, registry_path: Path | str | None = None) -> None:
        self.registry_path = Path(registry_path) if registry_path is not None else _default_methodology_registry_path()
        self.registry = {
            "registry_id": "consumer_industrials_metric_methodology_registry_v1",
            "version": 1,
            "metrics": {},
        }
        try:
            if self.registry_path.exists():
                self.registry = json.loads(self.registry_path.read_text())
        except Exception:
            # Some synced config files show up as dataless placeholders locally.
            # Fall back to an empty registry rather than failing snapshot builds.
            self.registry = {
                "registry_id": "consumer_industrials_metric_methodology_registry_v1",
                "version": 1,
                "metrics": {},
            }

    @property
    def registry_id(self) -> str:
        return str(self.registry.get("registry_id") or "consumer_industrials_metric_methodology_registry_v1")

    def canonical_owners(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.registry.get("canonical_owners") or {})

    def metrics(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.registry.get("metrics") or {})

    def metric(self, metric_id: str) -> Dict[str, Any]:
        return dict(self.metrics().get(metric_id) or {})

    def validate(self, expected_metric_ids: Optional[Iterable[str]] = None) -> List[str]:
        errors: List[str] = []
        registry = self.registry
        if not registry.get("registry_id"):
            errors.append("missing registry_id")
        if not isinstance(registry.get("metrics"), dict) or not registry.get("metrics"):
            errors.append("missing metrics")
            return errors

        owners = self.canonical_owners()
        metrics = self.metrics()
        required_keys = {
            "label",
            "canonical_owner_id",
            "canonical_classification",
            "market_layer_status",
            "current_alignment_status",
            "current_axiom_formula",
            "canonical_formula",
            "code_anchors",
        }
        valid_classifications = {"canonical_external", "filing_native_external", "internal_only"}
        valid_market_status = {"keep", "rename", "retire"}
        valid_alignment = {"aligned", "partial_proxy", "needs_rebuild", "internal_metric_only"}

        for metric_id, entry in metrics.items():
            missing = sorted(key for key in required_keys if key not in entry)
            if missing:
                errors.append(f"{metric_id}: missing keys {missing}")
            owner_id = entry.get("canonical_owner_id")
            if owner_id not in owners:
                errors.append(f"{metric_id}: unknown canonical_owner_id {owner_id}")
            if entry.get("canonical_classification") not in valid_classifications:
                errors.append(f"{metric_id}: invalid canonical_classification {entry.get('canonical_classification')}")
            if entry.get("market_layer_status") not in valid_market_status:
                errors.append(f"{metric_id}: invalid market_layer_status {entry.get('market_layer_status')}")
            if entry.get("current_alignment_status") not in valid_alignment:
                errors.append(f"{metric_id}: invalid current_alignment_status {entry.get('current_alignment_status')}")
            if not isinstance(entry.get("code_anchors"), list) or not entry.get("code_anchors"):
                errors.append(f"{metric_id}: code_anchors must be a non-empty list")
            if entry.get("canonical_owner_id") == "fitch_ratings" and not entry.get("primary_source_document_id"):
                errors.append(f"{metric_id}: Fitch-owned metric missing primary_source_document_id")
            if entry.get("canonical_classification") == "internal_only" and entry.get("market_layer_status") == "keep":
                errors.append(f"{metric_id}: internal_only metric cannot be marked keep")

        if expected_metric_ids is not None:
            expected = {str(metric_id) for metric_id in expected_metric_ids}
            missing_metrics = sorted(expected - set(metrics))
            if missing_metrics:
                errors.append(f"registry missing expected metrics: {missing_metrics}")
        return errors
