from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from scripts.merge_causal_model_overlays import merge_models


def _write_bundle(path: Path, payload: dict) -> None:
    with path.open("wb") as fh:
        pickle.dump(payload, fh)


def test_merge_models_replaces_selected_cell_and_bundle(tmp_path: Path):
    base_model_path = tmp_path / "base.json"
    overlay_model_path = tmp_path / "overlay.json"
    selection_path = tmp_path / "selection.json"
    out_model_path = tmp_path / "merged.json"
    out_model_card_path = tmp_path / "merged.model_card.json"

    base_bundle_path = tmp_path / "base.bundle.pkl"
    overlay_bundle_path = tmp_path / "overlay.bundle.pkl"
    _write_bundle(base_bundle_path, {"value_creation::capital_return::buyback": "base-model"})
    _write_bundle(overlay_bundle_path, {"value_creation::capital_return::buyback": "overlay-model"})

    base_payload = {
        "version": "base",
        "model_bundle_path": base_bundle_path.name,
        "objectives": {
            "value_creation": {
                "models": {},
                "dr_models": {
                    "capital_return::buyback": {
                        "model_family": "hgb",
                        "bundle_key": "value_creation::capital_return::buyback",
                        "enabled": False,
                        "oos_r2": -0.1,
                    }
                },
            }
        },
        "model_card": {
            "objectives": {
                "value_creation": {
                    "enabled_actions": 0,
                    "actions": {
                        "capital_return::buyback": {
                            "enabled": False,
                            "oos_r2": -0.1,
                        }
                    },
                }
            }
        },
    }
    overlay_payload = {
        "version": "overlay",
        "model_bundle_path": overlay_bundle_path.name,
        "objectives": {
            "value_creation": {
                "models": {},
                "dr_models": {
                    "capital_return::buyback": {
                        "model_family": "hgb",
                        "bundle_key": "value_creation::capital_return::buyback",
                        "enabled": True,
                        "oos_r2": 0.2,
                    }
                },
            }
        },
        "model_card": {
            "objectives": {
                "value_creation": {
                    "enabled_actions": 1,
                    "actions": {
                        "capital_return::buyback": {
                            "enabled": True,
                            "oos_r2": 0.2,
                        }
                    },
                }
            }
        },
    }
    selection_payload = {"replace_dr_models": {"value_creation": ["capital_return::buyback"]}}

    base_model_path.write_text(json.dumps(base_payload))
    overlay_model_path.write_text(json.dumps(overlay_payload))
    selection_path.write_text(json.dumps(selection_payload))

    result = merge_models(
        base_model_path=base_model_path,
        overlay_model_path=overlay_model_path,
        selection_path=selection_path,
        out_model_path=out_model_path,
        out_model_card_path=out_model_card_path,
    )

    assert result["ok"] is True
    merged_payload = json.loads(out_model_path.read_text())
    merged_card = json.loads(out_model_card_path.read_text())
    assert (
        merged_payload["objectives"]["value_creation"]["dr_models"]["capital_return::buyback"]["enabled"] is True
    )
    assert merged_card["objectives"]["value_creation"]["enabled_actions"] == 1

    merged_bundle_path = out_model_path.with_suffix(".bundle.pkl")
    with merged_bundle_path.open("rb") as fh:
        merged_bundle = pickle.load(fh)
    assert merged_bundle["value_creation::capital_return::buyback"] == "overlay-model"


def test_merge_models_rejects_feature_order_mismatch(tmp_path: Path):
    base_model_path = tmp_path / "base.json"
    overlay_model_path = tmp_path / "overlay.json"
    selection_path = tmp_path / "selection.json"
    out_model_path = tmp_path / "merged.json"
    out_model_card_path = tmp_path / "merged.model_card.json"

    base_bundle_path = tmp_path / "base.bundle.pkl"
    overlay_bundle_path = tmp_path / "overlay.bundle.pkl"
    _write_bundle(base_bundle_path, {"rating_preservation::capital_return::dividend_initiate": "base-model"})
    _write_bundle(overlay_bundle_path, {"rating_preservation::capital_return::dividend_initiate": "overlay-model"})

    base_payload = {
        "version": "base",
        "feature_order": ["a", "b"],
        "feature_transform_spec": {"usd_millions_features": ["a"]},
        "model_bundle_path": base_bundle_path.name,
        "objectives": {
            "rating_preservation": {
                "models": {},
                "dr_models": {
                    "capital_return::dividend_initiate": {
                        "model_family": "hgb",
                        "bundle_key": "rating_preservation::capital_return::dividend_initiate",
                        "enabled": False,
                    }
                },
            }
        },
        "model_card": {"objectives": {}},
    }
    overlay_payload = {
        "version": "overlay",
        "feature_order": ["a", "b", "c"],
        "feature_transform_spec": {"usd_millions_features": ["a", "c"]},
        "model_bundle_path": overlay_bundle_path.name,
        "objectives": {
            "rating_preservation": {
                "models": {},
                "dr_models": {
                    "capital_return::dividend_initiate": {
                        "model_family": "hgb",
                        "bundle_key": "rating_preservation::capital_return::dividend_initiate",
                        "enabled": True,
                    }
                },
            }
        },
        "model_card": {"objectives": {}},
    }
    selection_payload = {"replace_dr_models": {"rating_preservation": ["capital_return::dividend_initiate"]}}

    base_model_path.write_text(json.dumps(base_payload))
    overlay_model_path.write_text(json.dumps(overlay_payload))
    selection_path.write_text(json.dumps(selection_payload))

    with pytest.raises(ValueError, match="feature_order"):
        merge_models(
            base_model_path=base_model_path,
            overlay_model_path=overlay_model_path,
            selection_path=selection_path,
            out_model_path=out_model_path,
            out_model_card_path=out_model_card_path,
        )
