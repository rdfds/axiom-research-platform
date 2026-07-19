#!/usr/bin/env python
"""Build a hybrid causal model artifact from champion/challenger models.

The hybrid keeps champion cells by default and selectively upgrades cells from
the challenger when they are enabled and meet configured OOS quality floors.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple


class _RidgePredictor:
    """Compatibility wrapper for legacy pickled ridge predictors."""

    def __init__(self, beta: Any = None) -> None:
        self.beta = beta

    def predict(self, X: Any) -> list[float]:  # noqa: N803
        beta_raw = self.beta
        if beta_raw is None:
            return [0.0 for _ in list(X or [])]
        try:
            beta = [float(v) for v in list(beta_raw)]
        except Exception:
            return [0.0 for _ in list(X or [])]
        if not beta:
            return [0.0 for _ in list(X or [])]
        out: list[float] = []
        for row_raw in list(X or []):
            try:
                row = [float(v) for v in list(row_raw)]
            except Exception:
                row = []
            y = beta[0]
            width = min(len(row), max(0, len(beta) - 1))
            for idx in range(width):
                y += beta[idx + 1] * row[idx]
            out.append(float(y))
        return out


class _BundleUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        # Legacy training artifacts may pickle _RidgePredictor under __main__.
        if module == "__main__" and name == "_RidgePredictor":
            return _RidgePredictor
        return super().find_class(module, name)


def _to_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _load_bundle(model_path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    bundle_rel = str(payload.get("model_bundle_path", "")).strip()
    if not bundle_rel:
        return {}
    bundle_path = Path(bundle_rel)
    if not bundle_path.is_absolute():
        bundle_path = model_path.parent / bundle_path
    try:
        with open(bundle_path, "rb") as fh:
            loaded = pickle.load(fh)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        with open(bundle_path, "rb") as fh:
            loaded = _BundleUnpickler(fh).load()
        return loaded if isinstance(loaded, dict) else {}


def _build_model_card(payload: Dict[str, Any]) -> Dict[str, Any]:
    card: Dict[str, Any] = {
        "version": str(payload.get("version", "")),
        "trained_at": str(payload.get("trained_at", "")),
        "dataset_rows": int(payload.get("training_rows", 0) or 0),
        "training_split": dict(payload.get("training_split", {}) or {}),
        "model_family": str(payload.get("model_family", "")),
        "cell_level": str(payload.get("cell_level", "")),
        "feature_transform_spec": dict(payload.get("feature_transform_spec", {}) or {}),
        "objectives": {},
    }
    objectives = dict(payload.get("objectives", {}) or {})
    for objective, objective_payload in objectives.items():
        dr_models = dict((objective_payload or {}).get("dr_models", {}) or {})
        objective_card = {"actions": {}, "enabled_actions": 0}
        for action_name, model in dr_models.items():
            enabled = bool((model or {}).get("enabled", True))
            if enabled:
                objective_card["enabled_actions"] += 1
            objective_card["actions"][action_name] = {
                "method": str((model or {}).get("method", "")),
                "n_train": int((model or {}).get("n_train", 0) or 0),
                "n_valid": int((model or {}).get("n_valid", 0) or 0),
                "treated_rows": int((model or {}).get("treated_rows", 0) or 0),
                "control_rows": int((model or {}).get("control_rows", 0) or 0),
                "oos_r2": (model or {}).get("oos_r2"),
                "residual_std": (model or {}).get("residual_std"),
                "enabled": enabled,
                "gate_reason": str((model or {}).get("gate_reason", "")),
            }
        card["objectives"][str(objective)] = objective_card
    return card


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build hybrid causal model from champion/challenger artifacts.")
    p.add_argument("--champion-model", required=True, help="Path to champion model JSON")
    p.add_argument("--challenger-model", required=True, help="Path to challenger model JSON")
    p.add_argument("--out-path", required=True, help="Output hybrid model JSON path")
    p.add_argument("--model-card-out", default="", help="Optional output path for model_card JSON")
    p.add_argument(
        "--challenger-min-oos-r2",
        type=float,
        default=0.08,
        help="Minimum challenger cell OOS R2 required to replace/append a cell.",
    )
    p.add_argument(
        "--replace-min-delta-oos-r2",
        type=float,
        default=0.0,
        help="Required challenger OOS improvement over champion for replacement when champion is enabled.",
    )
    return p.parse_args()


def _validate_compatibility(champion: Dict[str, Any], challenger: Dict[str, Any]) -> None:
    fields = [
        "feature_order",
        "model_family",
        "cell_level",
    ]
    for f in fields:
        if champion.get(f) != challenger.get(f):
            raise ValueError(f"incompatible field {f}: champion={champion.get(f)!r} challenger={challenger.get(f)!r}")
    if dict(champion.get("objectives", {}) or {}).keys() != dict(challenger.get("objectives", {}) or {}).keys():
        raise ValueError("objective sets differ between champion and challenger")


def _pick_source(
    champion_model: Dict[str, Any] | None,
    challenger_model: Dict[str, Any] | None,
    challenger_min_oos_r2: float,
    replace_min_delta_oos_r2: float,
) -> str:
    c = champion_model if isinstance(champion_model, dict) else None
    h = challenger_model if isinstance(challenger_model, dict) else None

    if h is None:
        return "champion"

    h_enabled = bool(h.get("enabled", True))
    h_oos = _to_float(h.get("oos_r2"))
    h_eligible = h_enabled and h_oos is not None and float(h_oos) >= float(challenger_min_oos_r2)
    if not h_eligible:
        return "champion"

    if c is None:
        return "challenger"

    c_enabled = bool(c.get("enabled", True))
    if not c_enabled:
        return "challenger"

    c_oos = _to_float(c.get("oos_r2"))
    if c_oos is None:
        return "challenger"
    if float(h_oos) >= float(c_oos) + float(replace_min_delta_oos_r2):
        return "challenger"
    return "champion"


