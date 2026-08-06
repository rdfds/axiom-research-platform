#!/usr/bin/env python
"""Create a targeted causal rescue plan from full ML audit output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build causal rescue/blocklist plan from ML audit JSON.")
    p.add_argument("--audit-json", required=True, help="Path to audit_full_ml_status.py output JSON.")
    p.add_argument(
        "--strict-pass-threshold",
        type=float,
        default=0.50,
        help="Actions below this strict pass rate are flagged for rescue/blocklist.",
    )
    p.add_argument(
        "--min-action-rows",
        type=int,
        default=100,
        help="Minimum action rows required for rescue prioritization.",
    )
    p.add_argument(
        "--low-row-blocklist-threshold",
        type=int,
        default=50,
        help="Include low-row actions in blocklist suggestions when they clear this minimum support threshold.",
    )
    p.add_argument(
        "--out-json",
        default="/tmp/causal_rescue_plan.json",
        help="Output plan JSON.",
    )
    p.add_argument(
        "--out-blocklist",
        default="/tmp/causal_action_blocklist_suggested.txt",
        help="Output newline-delimited action blocklist.",
    )
    p.add_argument(
        "--out-rescue-actions",
        default="/tmp/causal_action_rescue_candidates.txt",
        help="Output newline-delimited action rescue candidates.",
    )
    return p.parse_args()


def _f(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
    except Exception:
        return float(default)
    if out != out:
        return float(default)
    return float(out)


def _classify_action(action: Dict[str, Any], strict_thr: float) -> Dict[str, Any]:
    row = dict(action)
    strict_rate = _f(row.get("strict_pass_rate"))
    causal_rate = _f(row.get("causal_rate"))
    rows = int(row.get("rows", 0) or 0)

    if causal_rate <= 0.0:
        row["rescue_reason"] = "no_causal_coverage"
        row["recommended_action"] = "keep_blocked_collect_more_signal"
    elif strict_rate <= 0.0:
        row["rescue_reason"] = "strict_gate_total_failure"
        row["recommended_action"] = "targeted_retrain_and_rebenchmark"
    elif strict_rate < strict_thr:
        row["rescue_reason"] = "strict_gate_under_threshold"
        row["recommended_action"] = "recalibrate_and_rebenchmark"
    else:
        row["rescue_reason"] = "healthy"
        row["recommended_action"] = "no_action"

    if rows <= 0:
        row["rescue_reason"] = "no_support"
        row["recommended_action"] = "keep_blocked_collect_more_signal"
    return row


def build_causal_rescue_plan(
    *,
    audit: Dict[str, Any],
    strict_pass_threshold: float,
    min_action_rows: int,
    low_row_blocklist_threshold: int,
) -> Dict[str, Any]:
    actions: List[Dict[str, Any]] = list(audit.get("actions") or [])

    strict_thr = float(strict_pass_threshold)
    min_rows = int(min_action_rows)
    low_row_thr = int(low_row_blocklist_threshold)

    rescue = [
        _classify_action(a, strict_thr)
        for a in actions
        if int(a.get("rows", 0) or 0) >= min_rows and _f(a.get("strict_pass_rate")) < strict_thr
    ]
    rescue.sort(key=lambda a: (_f(a.get("strict_pass_rate")), -int(a.get("rows", 0) or 0), str(a.get("action_id", ""))))

    low_row_block_only = [
        _classify_action(a, strict_thr)
        for a in actions
        if low_row_thr <= int(a.get("rows", 0) or 0) < min_rows
        and _f(a.get("strict_pass_rate")) < strict_thr
    ]
    low_row_block_only.sort(
        key=lambda a: (_f(a.get("strict_pass_rate")), -int(a.get("rows", 0) or 0), str(a.get("action_id", "")))
    )

    blocklist = []
    for row in rescue + low_row_block_only:
        action_id = str(row.get("action_id", "")).strip()
        if action_id and action_id not in blocklist:
            blocklist.append(action_id)

    zero_causal = [a for a in rescue if _f(a.get("causal_rate")) <= 0.0]
    causal_but_failing = [a for a in rescue if _f(a.get("causal_rate")) > 0.0]

    return {
        "thresholds": {
            "strict_pass_threshold": strict_thr,
            "min_action_rows": min_rows,
            "low_row_blocklist_threshold": low_row_thr,
        },
        "summary": {
            "actions_total": len(actions),
            "rescue_actions": len(rescue),
            "low_row_blocklist_actions": len(low_row_block_only),
            "zero_causal_actions": len(zero_causal),
            "causal_but_low_strict_actions": len(causal_but_failing),
        },
        "rescue_actions": rescue,
        "low_row_blocklist_actions": low_row_block_only,
        "zero_causal_actions": zero_causal,
        "causal_but_low_strict_actions": causal_but_failing,
        "suggested_blocklist": blocklist,
    }


def main() -> None:
    args = _parse_args()
    audit = json.loads(Path(args.audit_json).read_text())
    out = {
        "source_audit_json": str(args.audit_json),
        **build_causal_rescue_plan(
            audit=audit,
            strict_pass_threshold=float(args.strict_pass_threshold),
            min_action_rows=int(args.min_action_rows),
            low_row_blocklist_threshold=int(args.low_row_blocklist_threshold),
        ),
    }

    Path(args.out_json).write_text(json.dumps(out, indent=2))
    blocklist = list(out.get("suggested_blocklist") or [])
    rescue = list(out.get("rescue_actions") or [])
    Path(args.out_blocklist).write_text("".join(f"{x}\n" for x in blocklist))
    Path(args.out_rescue_actions).write_text(
        "".join(f"{str(row.get('action_id', '')).strip()}\n" for row in rescue if str(row.get("action_id", "")).strip())
    )

    print(
        json.dumps(
            {
                "ok": True,
                "out_json": str(args.out_json),
                "out_blocklist": str(args.out_blocklist),
                "out_rescue_actions": str(args.out_rescue_actions),
                "rescue_actions": len(rescue),
            }
        )
    )


if __name__ == "__main__":
    main()
