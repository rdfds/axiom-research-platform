#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_full_ml_status import build_ml_status_audit  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gate recommendation canary runs against regression thresholds.")
    p.add_argument("--runs-roots", nargs="+", required=True)
    p.add_argument("--run-ids-file", required=True)
    p.add_argument("--out", default="/tmp/recommendation_canary_gate.json")
    p.add_argument("--min-action-rows", type=int, default=50)
    p.add_argument("--min-causal-rate-mean", type=float, default=0.75)
    p.add_argument("--min-strict-all-mean", type=float, default=0.70)
    p.add_argument("--min-strict-causal-mean", type=float, default=0.90)
    p.add_argument("--min-precedent-conf-mean", type=float, default=0.35)
    p.add_argument("--max-precedent-oos-mean", type=float, default=0.90)
    return p.parse_args()


def _mean(payload: Dict[str, Any], *keys: str) -> float:
    cur: Any = payload
    for key in keys:
        cur = (cur or {}).get(key) if isinstance(cur, dict) else None
    try:
        out = float(cur)
    except Exception:
        return 0.0
    if out != out:
        return 0.0
    return float(out)


def evaluate_canary_gate(audit: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    status_counts = dict(audit.get("status_counts", {}) or {})
    completed = int(status_counts.get("completed", 0) or 0)
    failed = int(status_counts.get("failed", 0) or 0)
    runs_analyzed = int(audit.get("runs_analyzed", 0) or 0)
    causal_rate_mean = _mean(audit, "causal_summary", "causal_rate", "mean")
    strict_all_mean = _mean(audit, "causal_summary", "strict_pass_rate_among_all", "mean")
    strict_causal_mean = _mean(audit, "causal_summary", "strict_pass_rate_among_causal", "mean")
    precedent_conf_mean = _mean(audit, "precedent_summary", "precedent_confidence_mean", "mean")
    precedent_oos_mean = _mean(audit, "precedent_summary", "out_of_sample_rate", "mean")

    checks = [
        {
            "name": "no_failed_runs",
            "pass": failed == 0,
            "actual": failed,
            "expected": 0,
        },
        {
            "name": "all_runs_completed",
            "pass": completed == runs_analyzed,
            "actual": completed,
            "expected": runs_analyzed,
        },
        {
            "name": "causal_rate_mean",
            "pass": causal_rate_mean >= float(args.min_causal_rate_mean),
            "actual": causal_rate_mean,
            "expected_min": float(args.min_causal_rate_mean),
        },
        {
            "name": "strict_all_mean",
            "pass": strict_all_mean >= float(args.min_strict_all_mean),
            "actual": strict_all_mean,
            "expected_min": float(args.min_strict_all_mean),
        },
        {
            "name": "strict_causal_mean",
            "pass": strict_causal_mean >= float(args.min_strict_causal_mean),
            "actual": strict_causal_mean,
            "expected_min": float(args.min_strict_causal_mean),
        },
        {
            "name": "precedent_conf_mean",
            "pass": precedent_conf_mean >= float(args.min_precedent_conf_mean),
            "actual": precedent_conf_mean,
            "expected_min": float(args.min_precedent_conf_mean),
        },
        {
            "name": "precedent_oos_mean",
            "pass": precedent_oos_mean <= float(args.max_precedent_oos_mean),
            "actual": precedent_oos_mean,
            "expected_max": float(args.max_precedent_oos_mean),
        },
    ]

    return {
        "gate_pass": all(bool(c.get("pass")) for c in checks),
        "metrics": {
            "runs_analyzed": runs_analyzed,
            "status_counts": status_counts,
            "causal_rate_mean": causal_rate_mean,
            "strict_all_mean": strict_all_mean,
            "strict_causal_mean": strict_causal_mean,
            "precedent_conf_mean": precedent_conf_mean,
            "precedent_oos_mean": precedent_oos_mean,
        },
        "checks": checks,
    }


def main() -> None:
    args = _parse_args()
    run_ids: set[str] = set()
    for line in Path(args.run_ids_file).read_text().splitlines():
        parts = [x for x in line.strip().split() if x]
        if parts:
            run_ids.add(parts[-1])

    audit = build_ml_status_audit(
        runs_roots=args.runs_roots,
        include_run_ids=run_ids,
        min_action_rows=int(args.min_action_rows),
    )
    gate = evaluate_canary_gate(audit, args)
    out = {
        "audit": audit,
        "gate": gate,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps({"ok": bool(gate["gate_pass"]), "out": str(out_path), **gate}))
    raise SystemExit(0 if gate["gate_pass"] else 1)


if __name__ == "__main__":
    main()
