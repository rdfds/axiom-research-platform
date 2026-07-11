#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge selected causal-model cells from an overlay artifact into a base artifact.")
    p.add_argument("--base-model", required=True, help="Champion/base causal model artifact JSON.")
    p.add_argument("--overlay-model", required=True, help="Overlay/rescue causal model artifact JSON.")
    p.add_argument(
        "--selection-json",
        required=True,
        help="JSON file describing which objective/cell pairs to copy from the overlay.",
    )
    p.add_argument("--out-model", required=True, help="Output merged causal model artifact JSON.")
    p.add_argument(
        "--out-model-card",
        default="",
        help="Optional merged model-card JSON output. Defaults to <out-model>.model_card.json",
    )
    return p.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return obj


def _resolve_bundle_path(model_path: Path, payload: Dict[str, Any]) -> Path | None:
    raw = str(payload.get("model_bundle_path", "") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = model_path.parent / path
    return path


def _load_bundle(model_path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    bundle_path = _resolve_bundle_path(model_path, payload)
    if bundle_path is None or not bundle_path.exists():
        return {}
    with bundle_path.open("rb") as fh:
        loaded = pickle.load(fh)
    return loaded if isinstance(loaded, dict) else {}


def _parse_selection(path: Path) -> List[Tuple[str, str]]:
    obj = _load_json(path)
    raw = obj.get("replace_dr_models", obj)
    out: List[Tuple[str, str]] = []
    if isinstance(raw, list):
        for item in raw:
            text = str(item or "").strip()
            if "::" not in text:
                continue
            objective, cell = text.split("::", 1)
            pair = (str(objective).strip(), str(cell).strip())
            if pair not in out and pair[0] and pair[1]:
                out.append(pair)
        return out
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid selection JSON at {path}")
    for objective, cells in raw.items():
        objective_name = str(objective or "").strip()
        if not objective_name:
            continue
        for cell in list(cells or []):
            cell_name = str(cell or "").strip()
            if not cell_name:
                continue
            pair = (objective_name, cell_name)
            if pair not in out:
                out.append(pair)
    return out


def _recount_enabled_actions(model_card: Dict[str, Any]) -> None:
    for objective_payload in list((model_card.get("objectives") or {}).values()):
        if not isinstance(objective_payload, dict):
            continue
        actions = dict(objective_payload.get("actions", {}) or {})
        objective_payload["enabled_actions"] = int(
            sum(1 for info in actions.values() if isinstance(info, dict) and bool(info.get("enabled")))
        )


def _assert_feature_contract_compatible(
    *,
    base_payload: Dict[str, Any],
    overlay_payload: Dict[str, Any],
    base_model_path: Path,
    overlay_model_path: Path,
) :
    base_feature_order = list(base_payload.get("feature_order", []) or [])
    overlay_feature_order = list(overlay_payload.get("feature_order", []) or [])
    if base_feature_order and overlay_feature_order and base_feature_order != overlay_feature_order:
        raise ValueError(
            "Cannot merge causal overlays with incompatible feature_order values: "
            f"{base_model_path} has {len(base_feature_order)} features while "
            f"{overlay_model_path} has {len(overlay_feature_order)} features."
        )

    base_transform = dict(base_payload.get("feature_transform_spec", {}) or {})
    overlay_transform = dict(overlay_payload.get("feature_transform_spec", {}) or {})
    if base_transform and overlay_transform and base_transform != overlay_transform:
        raise ValueError(
            "Cannot merge causal overlays with incompatible feature_transform_spec metadata: "
            f"{base_model_path} and {overlay_model_path} do not share the same transform contract."
        )


