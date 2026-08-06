from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]


def _default_company_state_input_source_registry_path() -> Path:
    env = str(os.environ.get("AXIOM_INPUT_SOURCE_REGISTRY_PATH", "") or "").strip()
    if env:
        return Path(env)
    return ROOT / "configs" / "metric_methodologies" / "company_state_input_source_registry_v1.json"

_INPUT_LAYER_BUCKET_BY_DECISION = {
    "adopt_exact_external_methodology": "strict_market_defined",
    "keep_externally_anchored_house_formula": "secondary_externally_anchored",
    "retain_internal_inference": "internal_inference",
}

_INPUT_LAYER_BUCKET_REASON = {
    "strict_market_defined": (
        "This metric belongs to the strict market-defined input layer because it follows a named trusted "
        "external methodology, filing-native definition, or standardized public-market definition."
    ),
    "secondary_externally_anchored": (
        "This metric remains in the secondary externally anchored layer because it uses trusted external raw "
        "data but still requires a documented deterministic Axiom normalization step."
    ),
    "internal_inference": (
        "This metric remains in the internal inference layer because no single trusted external definition "
        "exists for the final metric."
    ),
}


class CompanyStateInputSourceRegistry:
    def __init__(self, registry_path: Path | str | None = None) -> None:
        self.registry_path = (
            Path(registry_path)
            if registry_path is not None
            else _default_company_state_input_source_registry_path()
        )
        self.registry = {
            "registry_id": "company_state_input_source_registry_v1",
            "version": "1.0.0",
            "owners": {},
            "metrics": {},
        }
        try:
            if self.registry_path.exists():
                self.registry = json.loads(self.registry_path.read_text())
        except Exception:
            # Synced placeholder files can time out on read; keep the builder alive.
            self.registry = {
                "registry_id": "company_state_input_source_registry_v1",
                "version": "1.0.0",
                "owners": {},
                "metrics": {},
            }

    @property
    def registry_id(self) -> str:
        return str(self.registry.get("registry_id") or "company_state_input_source_registry_v1")

    def owners(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.registry.get("owners") or {})

    def metrics(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.registry.get("metrics") or {})

    def metric(self, metric_id: str) -> Dict[str, Any]:
        return dict((self.registry.get("metrics") or {}).get(metric_id) or {})

    def owner(self, owner_id: str | None) -> Dict[str, Any]:
        if owner_id is None:
            return {}
        return dict((self.registry.get("owners") or {}).get(owner_id) or {})

    def metric_ids_by_execution_decision(self, decision: str) -> List[str]:
        return sorted(
            metric_id
            for metric_id, record in self.metrics().items()
            if record.get("methodology_execution_decision") == decision
        )

    def strict_market_defined_metric_ids(self) -> List[str]:
        return self.metric_ids_by_execution_decision("adopt_exact_external_methodology")

    def secondary_externally_anchored_metric_ids(self) -> List[str]:
        return self.metric_ids_by_execution_decision("keep_externally_anchored_house_formula")

    def internal_inference_metric_ids(self) -> List[str]:
        return self.metric_ids_by_execution_decision("retain_internal_inference")

    def input_layer_bucket(self, metric_id: str) -> str | None:
        record = self.metric(metric_id)
        decision = record.get("methodology_execution_decision")
        return _INPUT_LAYER_BUCKET_BY_DECISION.get(str(decision)) if decision is not None else None

    def input_layer_bucket_reason(self, metric_id: str) -> str | None:
        bucket = self.input_layer_bucket(metric_id)
        if bucket is None:
            return None
        return _INPUT_LAYER_BUCKET_REASON[bucket]

    def input_layer_summary(self) -> Dict[str, Dict[str, Any]]:
        return {
            "strict_market_defined": {
                "registry_metric_count": len(self.strict_market_defined_metric_ids()),
                "registry_metric_ids": self.strict_market_defined_metric_ids(),
            },
            "secondary_externally_anchored": {
                "registry_metric_count": len(self.secondary_externally_anchored_metric_ids()),
                "registry_metric_ids": self.secondary_externally_anchored_metric_ids(),
            },
            "internal_inference": {
                "registry_metric_count": len(self.internal_inference_metric_ids()),
                "registry_metric_ids": self.internal_inference_metric_ids(),
            },
        }
