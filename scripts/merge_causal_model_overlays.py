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
) -> None:
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


def merge_models(
    *,
    base_model_path: Path,
    overlay_model_path: Path,
    selection_path: Path,
    out_model_path: Path,
    out_model_card_path: Path,
) -> Dict[str, Any]:
    base_payload = _load_json(base_model_path)
    overlay_payload = _load_json(overlay_model_path)
    selections = _parse_selection(selection_path)
    if not selections:
        raise ValueError(f"No valid overlay selections found in {selection_path}")
    _assert_feature_contract_compatible(
        base_payload=base_payload,
        overlay_payload=overlay_payload,
        base_model_path=base_model_path,
        overlay_model_path=overlay_model_path,
    )

    merged_payload = copy.deepcopy(base_payload)
    merged_card = copy.deepcopy(dict(base_payload.get("model_card", {}) or {}))
    overlay_card = dict(overlay_payload.get("model_card", {}) or {})

    base_bundle = _load_bundle(base_model_path, base_payload)
    overlay_bundle = _load_bundle(overlay_model_path, overlay_payload)
    merged_bundle = dict(base_bundle)

    replaced: List[Dict[str, str]] = []
    for objective_name, cell_name in selections:
        base_objective = dict((merged_payload.get("objectives") or {}).get(objective_name, {}) or {})
        overlay_objective = dict((overlay_payload.get("objectives") or {}).get(objective_name, {}) or {})
        base_dr = dict(base_objective.get("dr_models", {}) or {})
        overlay_dr = dict(overlay_objective.get("dr_models", {}) or {})
        overlay_model = overlay_dr.get(cell_name)
        if not isinstance(overlay_model, dict):
            raise KeyError(f"Overlay model missing objective={objective_name} cell={cell_name}")
        base_dr[cell_name] = copy.deepcopy(overlay_model)
        base_objective["dr_models"] = base_dr
        merged_payload.setdefault("objectives", {})[objective_name] = base_objective

        bundle_key = str(overlay_model.get("bundle_key", "") or "")
        if bundle_key:
            if bundle_key not in overlay_bundle:
                raise KeyError(f"Overlay bundle missing key={bundle_key} for objective={objective_name} cell={cell_name}")
            merged_bundle[bundle_key] = overlay_bundle[bundle_key]

        merged_objective_card = dict((merged_card.get("objectives") or {}).get(objective_name, {}) or {})
        overlay_objective_card = dict((overlay_card.get("objectives") or {}).get(objective_name, {}) or {})
        merged_actions_card = dict(merged_objective_card.get("actions", {}) or {})
        overlay_actions_card = dict(overlay_objective_card.get("actions", {}) or {})
        if cell_name in overlay_actions_card:
            merged_actions_card[cell_name] = copy.deepcopy(overlay_actions_card[cell_name])
        else:
            merged_actions_card[cell_name] = {
                "enabled": bool(overlay_model.get("enabled")),
                "oos_r2": overlay_model.get("oos_r2"),
                "gate_reason": overlay_model.get("gate_reason"),
                "method": overlay_model.get("method"),
                "n_train": overlay_model.get("n_train"),
                "n_valid": overlay_model.get("n_valid"),
                "treated_rows": overlay_model.get("treated_rows"),
                "control_rows": overlay_model.get("control_rows"),
                "residual_std": overlay_model.get("residual_std"),
            }
        merged_objective_card["actions"] = merged_actions_card
        merged_card.setdefault("objectives", {})[objective_name] = merged_objective_card

        replaced.append(
            {
                "objective": objective_name,
                "cell": cell_name,
                "bundle_key": bundle_key,
            }
        )

    if merged_bundle:
        out_bundle_path = out_model_path.with_suffix(".bundle.pkl")
        out_bundle_path.parent.mkdir(parents=True, exist_ok=True)
        with out_bundle_path.open("wb") as fh:
            pickle.dump(merged_bundle, fh)
        merged_payload["model_bundle_path"] = out_bundle_path.name

    _recount_enabled_actions(merged_card)
    merged_payload["model_card"] = merged_card
    merged_payload["merge_metadata"] = {
        "merged_at": datetime.now(timezone.utc).isoformat(),
        "base_model_path": str(base_model_path),
        "overlay_model_path": str(overlay_model_path),
        "selection_path": str(selection_path),
        "replaced_cells": replaced,
    }

    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    out_model_path.write_text(json.dumps(merged_payload, indent=2))
    out_model_card_path.parent.mkdir(parents=True, exist_ok=True)
    out_model_card_path.write_text(json.dumps(merged_card, indent=2))
    return {
        "ok": True,
        "out_model_path": str(out_model_path),
        "out_model_card_path": str(out_model_card_path),
        "replaced_cells": replaced,
    }


def main() -> None:
    args = _parse_args()
    out_model_path = Path(str(args.out_model))
    out_model_card_path = (
        Path(str(args.out_model_card))
        if str(args.out_model_card or "").strip()
        else out_model_path.with_suffix(".model_card.json")
    )
    payload = merge_models(
        base_model_path=Path(str(args.base_model)),
        overlay_model_path=Path(str(args.overlay_model)),
        selection_path=Path(str(args.selection_json)),
        out_model_path=out_model_path,
        out_model_card_path=out_model_card_path,
    )
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
