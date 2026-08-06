#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_causal_rescue_plan import build_causal_rescue_plan  # noqa: E402


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_path(*parts: str) -> str:
    return str(_REPO_ROOT.joinpath(*parts))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a targeted causal rescue model for blocked actions.")
    p.add_argument("--audit-json", default="", help="Optional ML audit JSON used to generate rescue actions.")
    p.add_argument("--rescue-actions-file", default="", help="Optional newline-delimited rescue action ids.")
    p.add_argument(
        "--generated-rescue-actions-out",
        default="/tmp/causal_rescue_actions_train.txt",
        help="Where to materialize rescue actions when --audit-json is used.",
    )
    p.add_argument("--strict-pass-threshold", type=float, default=0.50)
    p.add_argument("--min-action-rows", type=int, default=100)
    p.add_argument("--low-row-blocklist-threshold", type=int, default=50)
    p.add_argument(
        "--mapping-path",
        default=_default_path("config", "causal_rescue_action_mapping.json"),
        help="JSON mapping from recommendation actions to trainable dataset patterns.",
    )
    p.add_argument(
        "--outcomes-path",
        default=_default_path("data", "curated", "action_outcomes_with_credit_ratings.parquet"),
    )
    p.add_argument(
        "--out-path",
        default=_default_path("data", "models", "causal_impact_model_rescue_hgb.json"),
    )
    p.add_argument(
        "--model-card-out",
        default=_default_path("data", "models", "causal_impact_model_rescue_hgb.model_card.json"),
    )
    p.add_argument("--train-end-date", default="2023-12-31")
    p.add_argument("--validation-start-date", default="2024-01-01")
    p.add_argument("--model-family", default="hgb")
    p.add_argument("--cell-level", default="action_subtype")
    p.add_argument("--crossfit-folds", type=int, default=3)
    p.add_argument("--dr-min-treated-rows", type=int, default=1500)
    p.add_argument("--dr-min-control-rows", type=int, default=20000)
    p.add_argument("--min-validation-rows", type=int, default=300)
    p.add_argument("--propensity-clip", type=float, default=0.03)
    p.add_argument("--gate-min-oos-r2", type=float, default=0.0)
    p.add_argument("--gate-min-train-rows", type=int, default=8000)
    p.add_argument("--gate-min-treated-rows", type=int, default=1500)
    p.add_argument("--gate-min-control-rows", type=int, default=20000)
    p.add_argument("--progress-every-cells", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _read_action_ids(path: Path) -> List[str]:
    action_ids: List[str] = []
    for raw in path.read_text().splitlines():
        item = raw.strip()
        if item and not item.startswith("#") and item not in action_ids:
            action_ids.append(item)
    return action_ids


def _load_action_mapping(path: Path) -> Dict[str, Dict[str, object]]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid mapping JSON: {path}")
    out: Dict[str, Dict[str, object]] = {}
    for key, value in obj.items():
        if not isinstance(value, dict):
            continue
        out[str(key)] = dict(value)
    return out


def _expand_recommendation_actions_to_train_patterns(
    action_ids: List[str],
    mapping: Dict[str, Dict[str, object]],
) -> Tuple[List[str], List[str], Dict[str, Dict[str, object]]]:
    patterns: List[str] = []
    unresolved: List[str] = []
    coverage: Dict[str, Dict[str, object]] = {}
    for action_id in action_ids:
        payload = dict(mapping.get(str(action_id), {}) or {})
        mapped = [str(x) for x in (payload.get("train_patterns") or []) if str(x).strip()]
        coverage[str(action_id)] = {
            "status": str(payload.get("status", "unsupported")),
            "notes": str(payload.get("notes", "")),
            "train_patterns": mapped,
        }
        if not mapped:
            unresolved.append(str(action_id))
            continue
        for pattern in mapped:
            if pattern not in patterns:
                patterns.append(pattern)
    return patterns, unresolved, coverage


def materialize_rescue_action_ids(
    args: argparse.Namespace,
) -> Tuple[List[str], Path, List[str], List[str], Dict[str, Dict[str, object]]]:
    if str(args.rescue_actions_file or "").strip():
        source_path = Path(str(args.rescue_actions_file))
        recommendation_action_ids = _read_action_ids(source_path)
    else:
        if not str(args.audit_json or "").strip():
            raise ValueError("Either --audit-json or --rescue-actions-file is required.")

        audit = json.loads(Path(str(args.audit_json)).read_text())
        plan = build_causal_rescue_plan(
            audit=audit,
            strict_pass_threshold=float(args.strict_pass_threshold),
            min_action_rows=int(args.min_action_rows),
            low_row_blocklist_threshold=int(args.low_row_blocklist_threshold),
        )
        recommendation_action_ids = []
        for row in plan.get("rescue_actions") or []:
            action_id = str((row or {}).get("action_id", "")).strip()
            if action_id and action_id not in recommendation_action_ids:
                recommendation_action_ids.append(action_id)

    mapping = _load_action_mapping(Path(str(args.mapping_path)))
    train_patterns, unresolved, coverage = _expand_recommendation_actions_to_train_patterns(
        recommendation_action_ids,
        mapping,
    )
    out_path = Path(str(args.generated_rescue_actions_out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(f"{pattern}\n" for pattern in train_patterns))
    return train_patterns, out_path, recommendation_action_ids, unresolved, coverage


def build_rescue_train_command(args: argparse.Namespace, action_ids_path: Path) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "train_causal_impact_model.py"),
        "--outcomes-path",
        str(args.outcomes_path),
        "--out-path",
        str(args.out_path),
        "--model-card-out",
        str(args.model_card_out),
        "--train-end-date",
        str(args.train_end_date),
        "--validation-start-date",
        str(args.validation_start_date),
        "--model-family",
        str(args.model_family),
        "--cell-level",
        str(args.cell_level),
        "--crossfit-folds",
        str(int(args.crossfit_folds)),
        "--dr-min-treated-rows",
        str(int(args.dr_min_treated_rows)),
        "--dr-min-control-rows",
        str(int(args.dr_min_control_rows)),
        "--min-validation-rows",
        str(int(args.min_validation_rows)),
        "--propensity-clip",
        str(float(args.propensity_clip)),
        "--gate-min-oos-r2",
        str(float(args.gate_min_oos_r2)),
        "--gate-min-train-rows",
        str(int(args.gate_min_train_rows)),
        "--gate-min-treated-rows",
        str(int(args.gate_min_treated_rows)),
        "--gate-min-control-rows",
        str(int(args.gate_min_control_rows)),
        "--progress-every-cells",
        str(int(args.progress_every_cells)),
        "--action-id-allowlist-file",
        str(action_ids_path),
        "--subtype-target-normalize",
    ]
    if bool(args.quiet):
        cmd.append("--quiet")
    return cmd


def _build_runtime_env() -> dict:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    return env


def main() -> None:
    args = _parse_args()
    train_patterns, action_ids_path, recommendation_action_ids, unresolved, coverage = materialize_rescue_action_ids(args)
    if not train_patterns:
        raise SystemExit(
            "No trainable rescue patterns resolved from rescue actions. Review dataset taxonomy or mapping table."
        )

    Path(str(args.out_path)).parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.model_card_out)).parent.mkdir(parents=True, exist_ok=True)

    cmd = build_rescue_train_command(args, action_ids_path)
    payload = {
        "ok": True,
        "event": "rescue_training_configured",
        "recommendation_action_count": len(recommendation_action_ids),
        "recommendation_action_ids": recommendation_action_ids,
        "train_pattern_count": len(train_patterns),
        "train_patterns": train_patterns,
        "unmapped_recommendation_actions": unresolved,
        "mapping_path": str(args.mapping_path),
        "mapping_coverage": coverage,
        "action_ids_path": str(action_ids_path),
        "out_path": str(args.out_path),
        "model_card_out": str(args.model_card_out),
        "command": cmd,
    }
    print(json.dumps(payload), flush=True)
    if bool(args.dry_run):
        return

    result = subprocess.run(cmd, env=_build_runtime_env(), check=False)
    raise SystemExit(int(result.returncode))


if __name__ == "__main__":
    main()
