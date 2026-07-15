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


