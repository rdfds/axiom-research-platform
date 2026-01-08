#!/usr/bin/env python
"""Aggregate causal + precedent + latency diagnostics for recommendation runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_CAUSAL_DRIVER_NAMES = {
    "causal_model_blend_weight",
    "causal_model_quality",
    "causal_model_support_score",
    "causal_model_mode",
}


def _to_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        out = float(v)
    except Exception:
        return default
    if out != out:  # NaN
        return default
    return float(out)


def _safe_ratio(num: float, den: float) -> float:
    den_f = float(den)
    if den_f <= 0.0:
        return 0.0
    return float(num) / den_f


def _load_json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(path.read_text()) or {})


def _driver_map(action_candidate: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    drivers = (
        ((action_candidate.get("impact_distribution", {}) or {}).get("key_drivers"))
        or []
    )
    out: Dict[str, Dict[str, Any]] = {}
    for row in drivers:
        if not isinstance(row, dict):
            continue
        name = str(row.get("driver_name", "")).strip()
        if not name:
            continue
        out[name] = row
    return out


def _parse_dt(v: str) -> Optional[datetime]:
    raw = str(v or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _audit_ts(audit_log: List[Dict[str, Any]], event_type: str) -> Optional[datetime]:
    for row in audit_log:
        if str(row.get("event_type", "")) != event_type:
            continue
        ts = _parse_dt(str(row.get("timestamp", "")))
        if ts is not None:
            return ts
    return None


def _duration_seconds(
    audit_log: List[Dict[str, Any]],
    event_start: str,
    event_end: str,
) -> Optional[float]:
    start = _audit_ts(audit_log, event_start)
    end = _audit_ts(audit_log, event_end)
    if start is None or end is None:
        return None
    return round((end - start).total_seconds(), 6)


def _quantile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    ordered = sorted(float(x) for x in xs)
    idx = int(round((len(ordered) - 1) * float(q)))
    idx = min(max(idx, 0), len(ordered) - 1)
    return float(ordered[idx])


def _summary_stats(rows: List[float]) -> Dict[str, Optional[float]]:
    vals = [float(x) for x in rows]
    if not vals:
        return {"mean": None, "p10": None, "p50": None, "p90": None, "min": None, "max": None}
    return {
        "mean": round(sum(vals) / len(vals), 6),
        "p10": round(float(_quantile(vals, 0.10) or 0.0), 6),
        "p50": round(float(_quantile(vals, 0.50) or 0.0), 6),
        "p90": round(float(_quantile(vals, 0.90) or 0.0), 6),
        "min": round(min(vals), 6),
        "max": round(max(vals), 6),
    }


def _read_run_ids(path_value: str) -> set[str]:
    run_ids: set[str] = set()
    raw = str(path_value or "").strip()
    if not raw:
        return run_ids
    p = Path(raw)
    if not p.exists():
        return run_ids
    for line in p.read_text().splitlines():
        parts = [x for x in str(line).strip().split() if x]
        if not parts:
            continue
        run_ids.add(parts[-1])
    return run_ids


def _iter_run_paths(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        runs_dir = root / "runs"
        if not runs_dir.exists():
            continue
        for run_path in runs_dir.glob("run_id=*.json"):
            yield run_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit full ML status across recommendation runs.")
    p.add_argument(
        "--runs-roots",
        nargs="+",
        default=["/tmp/recommendation_runs_v4_clean", "/tmp/recommendation_runs_fresh"],
        help="Run roots to scan (must contain runs/ and artifacts/).",
    )
    p.add_argument(
        "--run-ids-file",
        default="",
        help="Optional run-id filter (one run_id per line, or 'CIK RUN_ID').",
    )
    p.add_argument("--out", default="/tmp/ml_status_audit.json", help="Output JSON path.")
    p.add_argument("--min-action-rows", type=int, default=50, help="Min rows for action-level output.")
    return p.parse_args()


def build_ml_status_audit(
    *,
    runs_roots: Iterable[str | Path],
    include_run_ids: Optional[set[str]] = None,
    min_action_rows: int = 50,
) -> Dict[str, Any]:
    roots = [Path(x) for x in runs_roots]
    include_run_ids = set(include_run_ids or set())
    status_counts: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()

    run_rows: List[Dict[str, Any]] = []
    action_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"rows": 0, "causal_rows": 0, "strict_pass_rows": 0, "run_ids": set()}
    )

    for run_path in _iter_run_paths(roots):
        try:
            run = _load_json(run_path)
        except Exception:
            continue

        run_id = str(run.get("run_id", run_path.stem.replace("run_id=", "")))
        if include_run_ids and run_id not in include_run_ids:
            continue

        status = str(run.get("status", "")).strip() or "unknown"
        status_counts[status] += 1
        audit_log = list(run.get("audit_log") or [])

        if status == "failed":
            reason = ""
            for ev in reversed(audit_log):
                if str(ev.get("event_type", "")) == "run_failed":
                    reason = str((ev.get("details") or {}).get("error", "")).strip()
                    break
            failure_reasons[reason or "unknown_error"] += 1
            continue

        if status != "completed":
            continue

        artifacts = dict((run.get("metadata", {}) or {}).get("artifacts", {}) or {})
        feasibility_path = Path(str(artifacts.get("FeasibilityResults", "")))
        precedent_path = Path(str(artifacts.get("PrecedentMatches", "")))
        risk_path = Path(str(artifacts.get("CausalModelRiskReport", "")))

        feasibility_rows = []
        if feasibility_path.exists():
            try:
                feasibility_rows = list((_load_json(feasibility_path).get("results") or []))
            except Exception:
                feasibility_rows = []

        total = len(feasibility_rows)
        causal_present = 0
        strict_pass = 0

        for row in feasibility_rows:
            action_candidate = dict(row.get("action_candidate", {}) or {})
            action_id = str(action_candidate.get("action_id", "unknown"))
            drivers = _driver_map(action_candidate)
            names = set(drivers.keys())
            has_causal = bool(names & _CAUSAL_DRIVER_NAMES)
            if has_causal:
                causal_present += 1
                mode = _to_float((drivers.get("causal_model_mode") or {}).get("contribution"), 0.0) or 0.0
                blend = _to_float((drivers.get("causal_model_blend_weight") or {}).get("contribution"), 0.0) or 0.0
                if mode >= 0.5 or blend > 0.0:
                    strict_pass += 1

            bucket = action_stats[action_id]
            bucket["rows"] += 1
            if has_causal:
                bucket["causal_rows"] += 1
                if (
                    (_to_float((drivers.get("causal_model_mode") or {}).get("contribution"), 0.0) or 0.0) >= 0.5
                    or (_to_float((drivers.get("causal_model_blend_weight") or {}).get("contribution"), 0.0) or 0.0) > 0.0
                ):
                    bucket["strict_pass_rows"] += 1
            bucket["run_ids"].add(run_id)

        risk_summary: Dict[str, Any] = {}
        if risk_path.exists():
            try:
                risk_summary = dict((_load_json(risk_path).get("summary") or {}))
            except Exception:
                risk_summary = {}

        precedent_conf: List[float] = []
        precedent_oos: List[float] = []
        precedent_low_cov: List[float] = []
        precedent_regime_non_empty: List[float] = []
        precedent_tails_per_candidate: List[float] = []
        precedent_candidates = 0
        if precedent_path.exists():
            try:
                precedent = _load_json(precedent_path)
                matches = list(precedent.get("results") or [])
            except Exception:
                matches = []
            precedent_candidates = len(matches)
            for row in matches:
                pack = dict(row.get("precedent_pack", {}) or {})
                md = dict(pack.get("mismatch_diagnostics", {}) or {})
                precedent_conf.append(float(_to_float(pack.get("precedent_confidence"), 0.0) or 0.0))
                precedent_oos.append(1.0 if bool(md.get("out_of_sample_flag")) else 0.0)
                precedent_low_cov.append(1.0 if bool(md.get("low_precedent_coverage")) else 0.0)
                precedent_regime_non_empty.append(1.0 if len(pack.get("regime_splits") or []) > 0 else 0.0)
                tails = pack.get("tail_events") or pack.get("tails") or []
                precedent_tails_per_candidate.append(float(len(tails)))

        run_rows.append(
            {
                "run_id": run_id,
                "company_id": str(run.get("company_id", "")),
                "status": status,
                "counts": {
                    "candidate_rows": total,
                    "causal_rows": causal_present,
                    "strict_pass_rows": strict_pass,
                    "precedent_candidates": precedent_candidates,
                },
                "causal": {
                    "causal_rate": round(_safe_ratio(causal_present, max(1, total)), 6),
                    "strict_pass_rate_among_all": round(_safe_ratio(strict_pass, max(1, total)), 6),
                    "strict_pass_rate_among_causal": round(_safe_ratio(strict_pass, max(1, causal_present)), 6),
                },
                "precedent": {
                    "precedent_confidence_mean": round(sum(precedent_conf) / len(precedent_conf), 6)
                    if precedent_conf
                    else None,
                    "out_of_sample_rate": round(sum(precedent_oos) / len(precedent_oos), 6) if precedent_oos else None,
                    "low_coverage_rate": round(sum(precedent_low_cov) / len(precedent_low_cov), 6)
                    if precedent_low_cov
                    else None,
                    "regime_non_empty_rate": round(
                        sum(precedent_regime_non_empty) / len(precedent_regime_non_empty), 6
                    )
                    if precedent_regime_non_empty
                    else None,
                    "tails_per_candidate_mean": round(
                        sum(precedent_tails_per_candidate) / len(precedent_tails_per_candidate), 6
                    )
                    if precedent_tails_per_candidate
                    else None,
                },
                "latency_seconds": {
                    "queue_delay": _duration_seconds(audit_log, "run_created", "candidate_generation_started"),
                    "candidate_generation": _duration_seconds(
                        audit_log, "candidate_generation_started", "candidate_generation_completed"
                    ),
                    "feasibility_eval": _duration_seconds(audit_log, "feasibility_eval_started", "feasibility_eval_completed"),
                    "precedent_retrieval": _duration_seconds(
                        audit_log, "precedent_retrieval_started", "precedent_retrieval_completed"
                    ),
                    "planning": _duration_seconds(audit_log, "planning_started", "planning_completed"),
                    "active_pipeline": _duration_seconds(audit_log, "candidate_generation_started", "run_completed"),
                    "total_run": _duration_seconds(audit_log, "run_created", "run_completed"),
                },
                "risk_summary": risk_summary,
            }
        )

    action_rows = []
    for action_id, vals in action_stats.items():
        rows = int(vals["rows"])
        if rows < int(min_action_rows):
            continue
        causal_rows = int(vals["causal_rows"])
        strict_rows = int(vals["strict_pass_rows"])
        action_rows.append(
            {
                "action_id": action_id,
                "rows": rows,
                "causal_rows": causal_rows,
                "causal_rate": round(_safe_ratio(causal_rows, rows), 6),
                "strict_pass_rows": strict_rows,
                "strict_pass_rate": round(_safe_ratio(strict_rows, rows), 6),
                "run_count": len(vals["run_ids"]),
            }
        )
    action_rows.sort(key=lambda x: (x["strict_pass_rate"], x["causal_rate"], x["rows"]))

    causal_rates = [float((r.get("causal", {}) or {}).get("causal_rate") or 0.0) for r in run_rows]
    strict_all = [float((r.get("causal", {}) or {}).get("strict_pass_rate_among_all") or 0.0) for r in run_rows]
    strict_causal = [float((r.get("causal", {}) or {}).get("strict_pass_rate_among_causal") or 0.0) for r in run_rows]
    precedent_conf = [
        float((r.get("precedent", {}) or {}).get("precedent_confidence_mean") or 0.0)
        for r in run_rows
        if (r.get("precedent", {}) or {}).get("precedent_confidence_mean") is not None
    ]
    precedent_oos = [
        float((r.get("precedent", {}) or {}).get("out_of_sample_rate") or 0.0)
        for r in run_rows
        if (r.get("precedent", {}) or {}).get("out_of_sample_rate") is not None
    ]

    def _latency_vector(k: str) -> List[float]:
        out = []
        for row in run_rows:
            val = ((row.get("latency_seconds", {}) or {}).get(k))
            if val is None:
                continue
            out.append(float(val))
        return out

    out = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "runs_analyzed": len(run_rows),
        "actions_reported": len(action_rows),
        "status_counts": dict(status_counts),
        "failure_reasons": [{"reason": k, "count": v} for k, v in failure_reasons.most_common()],
        "causal_summary": {
            "causal_rate": _summary_stats(causal_rates),
            "strict_pass_rate_among_all": _summary_stats(strict_all),
            "strict_pass_rate_among_causal": _summary_stats(strict_causal),
        },
        "precedent_summary": {
            "precedent_confidence_mean": _summary_stats(precedent_conf),
            "out_of_sample_rate": _summary_stats(precedent_oos),
        },
        "latency_summary_seconds": {
            "queue_delay": _summary_stats(_latency_vector("queue_delay")),
            "candidate_generation": _summary_stats(_latency_vector("candidate_generation")),
            "feasibility_eval": _summary_stats(_latency_vector("feasibility_eval")),
            "precedent_retrieval": _summary_stats(_latency_vector("precedent_retrieval")),
            "planning": _summary_stats(_latency_vector("planning")),
            "active_pipeline": _summary_stats(_latency_vector("active_pipeline")),
            "total_run": _summary_stats(_latency_vector("total_run")),
        },
        "actions": action_rows,
        "runs": run_rows,
    }
    return out


def main() -> None:
    args = _parse_args()
    include_run_ids = _read_run_ids(args.run_ids_file)
    out = build_ml_status_audit(
        runs_roots=args.runs_roots,
        include_run_ids=include_run_ids,
        min_action_rows=int(args.min_action_rows),
    )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, indent=2))
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(out_path),
                "runs_analyzed": int(out.get("runs_analyzed", 0)),
                "actions_reported": int(out.get("actions_reported", 0)),
                "status_counts": dict(out.get("status_counts", {}) or {}),
            }
        )
    )


if __name__ == "__main__":
    main()
