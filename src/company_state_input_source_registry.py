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


