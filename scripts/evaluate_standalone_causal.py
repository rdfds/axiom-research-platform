#!/usr/bin/env python
"""Evaluate standalone causal impact for a single company/action input.

Example:
  python -u scripts/evaluate_standalone_causal.py \
    --company-id 0000320193 \
    --as-of 2026-02-28 \
    --action-id capital_return.open_market_buyback \
    --param size_pct_market_cap=0.05 \
    --param funding_mix.cash=1 \
    --snapshot-root data/company_state_snapshots/final_run_2026-02-28 \
    --entity-identifier-path data/inputs_layer/entity_identifier.parquet
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.causal_impact_model import load_default_causal_impact_model
from src.recommendation_run import _resolve_snapshot, _snapshot_company_aliases


def _parse_param_values(items: list[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid --param format: {raw}. Expected key=value.")
        key, val = raw.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            raise ValueError(f"Invalid --param key: {raw}")
        if val.lower() in {"true", "false"}:
            parsed: Any = val.lower() == "true"
        else:
            try:
                parsed = float(val) if "." in val else int(val)
            except Exception:
                parsed = val

        cur = out
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cur or not isinstance(cur[part], dict):
                cur[part] = {}
            cur = cur[part]
        cur[parts[-1]] = parsed
    return out


def _as_of_datetime(raw: str) -> datetime:
    s = str(raw).strip()
    if "T" not in s:
        s = s + "T00:00:00+00:00"
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--company-id", required=True)
    p.add_argument("--as-of", required=True, help="YYYY-MM-DD or full ISO timestamp")
    p.add_argument("--action-id", required=True)
    p.add_argument("--action-type", default=None, help="Optional override; defaults to action_id root")
    p.add_argument("--param", action="append", default=[], help="key=value, dotted keys allowed")
    p.add_argument("--snapshot-root", default="data/company_state_snapshots/final_run_2026-02-28")
    p.add_argument("--snapshot-path", default=None)
    p.add_argument("--entity-identifier-path", default="data/inputs_layer/entity_identifier.parquet")
    p.add_argument("--model-path", default=None, help="Optional causal artifact path")
    p.add_argument("--out", default=None, help="Optional output JSON path")
    args = p.parse_args()

    as_of = _as_of_datetime(args.as_of)
    snapshot_root = Path(args.snapshot_root) if args.snapshot_root else None
    snapshot_path = Path(args.snapshot_path) if args.snapshot_path else None
    entity_identifier_path = Path(args.entity_identifier_path)
    aliases = _snapshot_company_aliases(str(args.company_id), entity_identifier_path)
    snapshot = _resolve_snapshot(
        company_id=str(args.company_id),
        as_of_time=as_of,
        snapshot_root=snapshot_root,
        snapshot_path=snapshot_path,
        snapshot_builder=None,
        snapshot_loader=None,
        aliases=aliases,
    )
    features = dict(snapshot.get("features", {}) or {})
    regime = dict(snapshot.get("regime", {}) or {})
    params = _parse_param_values(args.param)
    action_id = str(args.action_id)
    action_type = str(args.action_type or action_id.split(".", 1)[0])
    action_subtype = str(action_id.split(".", 1)[1]) if "." in action_id else ""

    model = load_default_causal_impact_model(path_str=args.model_path) if args.model_path else load_default_causal_impact_model()
    if model is None:
        raise ValueError("No causal model artifact found. Set --model-path or CAUSAL_IMPACT_MODEL_PATH.")

    pred = model.predict(
        action_id=action_id,
        action_type=action_type,
        params=params,
        features=features,
        regime=regime,
        action_subtype=action_subtype,
    )
    if pred is None:
        raise ValueError("Causal model could not score this candidate (missing required features/model).")

    out = {
        "ok": True,
        "company_id": str(args.company_id),
        "as_of_time": as_of.isoformat(),
        "action_id": action_id,
        "action_type": action_type,
        "action_subtype": action_subtype,
        "parameters": params,
        "objectives": pred.objectives,
        "diagnostics": {
            "model_version": pred.model_version,
            "n_train": pred.n_train,
            "model_quality": pred.model_quality,
            "coverage_score": pred.coverage_score,
            "support_score": pred.support_score,
            "out_of_sample_flag": pred.out_of_sample_flag,
            "blend_weight_if_used": pred.blend_weight,
        },
    }
    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)


if __name__ == "__main__":
    main()
