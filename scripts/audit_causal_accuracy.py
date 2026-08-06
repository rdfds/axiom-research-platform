#!/usr/bin/env python
"""Audit causal coverage/strict-gate quality from completed recommendation runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


_CAUSAL_DRIVER_NAMES = {
    "causal_model_blend_weight",
    "causal_model_quality",
    "causal_model_support_score",
    "causal_model_mode",
}


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
    except Exception:
        return float(default)
    if out != out:  # NaN
        return float(default)
    return float(out)


def _load_json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(path.read_text()) or {})


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit causal accuracy/coverage from completed run artifacts.")
    p.add_argument(
        "--runs-roots",
        nargs="+",
        default=["/tmp/recommendation_runs_v4_clean", "/tmp/recommendation_runs_fresh"],
        help="Run roots to scan (each must contain runs/ and artifacts/).",
    )
    p.add_argument(
        "--run-ids-file",
        default="",
        help="Optional text file with run IDs to include (one run_id per line, or 'CIK RUN_ID' pairs).",
    )
    p.add_argument("--out", default="/tmp/causal_accuracy_audit.json", help="Output JSON path.")
    p.add_argument("--min-action-rows", type=int, default=50, help="Minimum rows to report action-level stats.")
    return p.parse_args()


def _is_causal_row(driver_names: set[str]) -> bool:
    return bool(driver_names & _CAUSAL_DRIVER_NAMES)


def _strict_gate_pass(driver_map: Dict[str, Dict[str, Any]]) -> bool:
    bw = _to_float((driver_map.get("causal_model_blend_weight") or {}).get("contribution"), 0.0)
    mode = _to_float((driver_map.get("causal_model_mode") or {}).get("contribution"), 0.0)
    return (bw > 0.0) or (mode > 0.0)


def main() -> None:
    args = _parse_args()
    roots = [Path(x) for x in args.runs_roots]
    include_run_ids: set[str] = set()
    if str(args.run_ids_file).strip():
        fp = Path(str(args.run_ids_file).strip())
        if fp.exists():
            for line in fp.read_text().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                include_run_ids.add(parts[-1])

    run_rows: List[Dict[str, Any]] = []
    action_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"rows": 0, "causal_rows": 0, "strict_pass_rows": 0, "run_ids": set()}
    )
    run_mtime: Dict[str, float] = {}

    for root in roots:
        runs_dir = root / "runs"
        if not runs_dir.exists():
            continue

        for run_path in runs_dir.glob("run_id=*.json"):
            run_mtime[run_path.stem.replace("run_id=", "")] = run_path.stat().st_mtime
            try:
                run = _load_json(run_path)
            except Exception:
                continue

            if str(run.get("status", "")) != "completed":
                continue

            run_id = str(run.get("run_id", run_path.stem.replace("run_id=", "")))
            if include_run_ids and run_id not in include_run_ids:
                continue
            artifacts = dict((run.get("metadata", {}) or {}).get("artifacts", {}) or {})
            feasibility_path = artifacts.get("FeasibilityResults")
            if not feasibility_path:
                continue

            try:
                feasibility = _load_json(Path(str(feasibility_path)))
            except Exception:
                continue
            results = list(feasibility.get("results") or [])

            risk_summary: Dict[str, Any] = {}
            risk_path = artifacts.get("CausalModelRiskReport")
            if risk_path and Path(str(risk_path)).exists():
                try:
                    risk_summary = dict((_load_json(Path(str(risk_path))).get("summary") or {}))
                except Exception:
                    risk_summary = {}

            total_rows = len(results)
            causal_rows = 0
            strict_rows = 0
            for row in results:
                action_candidate = dict(row.get("action_candidate") or row.get("candidate") or {})
                action_id = str(action_candidate.get("action_id", ""))
                drivers = list(((action_candidate.get("impact_distribution") or {}).get("key_drivers") or []))
                driver_map = {
                    str(d.get("driver_name")): d
                    for d in drivers
                    if isinstance(d, dict) and d.get("driver_name")
                }
                names = set(driver_map.keys())
                is_causal = _is_causal_row(names)
                strict_pass = is_causal and _strict_gate_pass(driver_map)
                if is_causal:
                    causal_rows += 1
                if strict_pass:
                    strict_rows += 1

                s = action_stats[action_id]
                s["rows"] += 1
                if is_causal:
                    s["causal_rows"] += 1
                if strict_pass:
                    s["strict_pass_rows"] += 1
                s["run_ids"].add(run_id)

            run_rows.append(
                {
                    "run_id": run_id,
                    "status": "completed",
                    "runs_root": str(root),
                    "mechanism_model_version": str((run.get("model_versions") or {}).get("mechanism_model_version", "")),
                    "total_rows": int(total_rows),
                    "causal_rows": int(causal_rows),
                    "causal_rate": float(causal_rows / total_rows) if total_rows else 0.0,
                    "strict_pass_rows": int(strict_rows),
                    "strict_pass_rate_among_all": float(strict_rows / total_rows) if total_rows else 0.0,
                    "strict_pass_rate_among_causal": float(strict_rows / causal_rows) if causal_rows else 0.0,
                    "risk_summary": risk_summary,
                }
            )

    run_rows.sort(key=lambda x: run_mtime.get(str(x.get("run_id", "")), 0.0), reverse=True)

    action_report = []
    for action_id, stat in action_stats.items():
        rows = int(stat["rows"])
        if rows < int(args.min_action_rows):
            continue
        causal_rows = int(stat["causal_rows"])
        strict_rows = int(stat["strict_pass_rows"])
        action_report.append(
            {
                "action_id": action_id,
                "rows": rows,
                "causal_rows": causal_rows,
                "causal_rate": float(causal_rows / rows) if rows else 0.0,
                "strict_pass_rows": strict_rows,
                "strict_pass_rate": float(strict_rows / rows) if rows else 0.0,
                "run_count": int(len(stat["run_ids"])),
            }
        )
    action_report.sort(key=lambda x: x["rows"], reverse=True)

    out_payload = {
        "runs_analyzed": int(len(run_rows)),
        "actions_reported": int(len(action_report)),
        "runs": run_rows,
        "actions": action_report,
    }
    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2))

    print(json.dumps({"ok": True, "out": str(out_path), "runs_analyzed": len(run_rows), "actions_reported": len(action_report)}))


if __name__ == "__main__":
    main()
