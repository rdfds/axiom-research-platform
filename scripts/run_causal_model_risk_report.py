#!/usr/bin/env python
"""Generate/attach a causal model risk report for an existing run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.causal_model_risk import build_causal_model_risk_report
from src.recommendation_run import RecommendationRunStore


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--runs-root", default="data/recommendation_runs")
    p.add_argument("--attach", action="store_true", help="Attach report under run artifacts as CausalModelRiskReport")
    p.add_argument("--out", default=None, help="Optional explicit output json path")
    args = p.parse_args()

    store = RecommendationRunStore(root=args.runs_root)
    run = store.get_run(args.run_id)
    if run is None:
        raise ValueError(f"Run not found: {args.run_id}")
    artifacts = dict(run.metadata.get("artifacts", {}) or {})
    feas_path = artifacts.get("FeasibilityResults")
    if not feas_path:
        raise ValueError("FeasibilityResults artifact missing for run")
    snap_path = artifacts.get("Snapshot")
    snapshot = {}
    if snap_path and Path(snap_path).exists():
        snapshot = json.loads(Path(snap_path).read_text())

    feas = json.loads(Path(feas_path).read_text())
    rows = list(feas.get("results", []) or [])
    report = build_causal_model_risk_report(run=run, snapshot=snapshot, feasibility_results=rows)

    if args.attach:
        out_path = store.attach_artifact(run.run_id, "CausalModelRiskReport", report)
    else:
        out_path = Path(args.out) if args.out else Path(f"/tmp/causal_model_risk_report_{run.run_id}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))

    print(json.dumps({"ok": True, "run_id": run.run_id, "out": str(out_path)}))


if __name__ == "__main__":
    main()

