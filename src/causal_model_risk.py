"""Causal model risk diagnostics for run-level governance.

This module summarizes causal coverage, support, quality, OOS rates, and
fallback behavior from FeasibilityResults action-candidate payloads.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    if isinstance(v, bool):
        return float(v)
    try:
        out = float(v)
    except Exception:
        return default
    if out != out:  # nan
        return default
    return out


def _driver_map(action_candidate: Dict[str, Any]) -> Dict[str, float]:
    impact = dict(action_candidate.get("impact_distribution", {}) or {})
    drivers = list(impact.get("key_drivers", []) or [])
    out: Dict[str, float] = {}
    for d in drivers:
        if not isinstance(d, dict):
            continue
        name = str(d.get("driver_name", "")).strip()
        if not name:
            continue
        val = _to_float(d.get("contribution"))
        if val is None:
            continue
        out[name] = float(val)
    return out


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    qq = max(0.0, min(1.0, float(q)))
    pos = qq * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    w = pos - lo
    return float(xs[lo] * (1.0 - w) + xs[hi] * w)


def _safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def build_causal_model_risk_report(
    run: Any,
    snapshot: Dict[str, Any],
    feasibility_results: List[Dict[str, Any]],
    previous_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rows = list(feasibility_results or [])
    total = len(rows)
    feasible_total = sum(1 for r in rows if bool(r.get("feasible")))

    causal_present = 0
    standalone_applied = 0
    standalone_fallback = 0
    oos_penalty = 0
    low_quality = 0
    low_support = 0
    low_oos = 0
    low_treated = 0
    low_control = 0
    strict_gate_pass = 0
    strict_gate_fail = 0
    quality_vals: List[float] = []
    support_vals: List[float] = []
    oos_vals: List[float] = []
    treated_vals: List[float] = []
    control_vals: List[float] = []
    uncertainty_vals: List[float] = []
    min_oos_floor = _to_float(os.environ.get("CAUSAL_STRICT_MIN_OOS_R2"), 0.0) or 0.0
    min_treated_floor = int(_to_float(os.environ.get("CAUSAL_STRICT_MIN_TREATED_ROWS"), 1000.0) or 1000.0)
    min_control_floor = int(_to_float(os.environ.get("CAUSAL_STRICT_MIN_CONTROL_ROWS"), 5000.0) or 5000.0)

    by_action: Dict[str, Dict[str, int]] = {}

    for r in rows:
        ac = dict(r.get("action_candidate", {}) or {})
        aid = str(ac.get("action_id", "unknown"))
        if aid not in by_action:
            by_action[aid] = {
                "count": 0,
                "causal_present": 0,
                "standalone_applied": 0,
                "standalone_fallback": 0,
                "oos_penalty": 0,
            }
        by_action[aid]["count"] += 1

        drivers = _driver_map(ac)
        has_causal = any(
            k in drivers
            for k in (
                "causal_model_mode",
                "causal_model_quality",
                "causal_model_support_score",
                "causal_model_blend_weight",
            )
        )
        if has_causal:
            causal_present += 1
            by_action[aid]["causal_present"] += 1

        mode = _to_float(drivers.get("causal_model_mode"))
        if mode is not None and mode >= 0.5:
            standalone_applied += 1
            by_action[aid]["standalone_applied"] += 1
        elif mode is not None:
            standalone_fallback += 1
            by_action[aid]["standalone_fallback"] += 1

        blend_weight = _to_float(drivers.get("causal_model_blend_weight"))
        if has_causal:
            if (mode is not None and mode >= 0.5) or (blend_weight is not None and blend_weight > 0.0):
                strict_gate_pass += 1
            else:
                strict_gate_fail += 1

        if "causal_model_oos_penalty" in drivers:
            oos_penalty += 1
            by_action[aid]["oos_penalty"] += 1

        q = _to_float(drivers.get("causal_model_quality"))
        if q is not None:
            quality_vals.append(float(q))
            if q < 0.0:
                low_quality += 1

        s = _to_float(drivers.get("causal_model_support_score"))
        if s is not None:
            support_vals.append(float(s))
            if s < 0.25:
                low_support += 1

        min_oos = _to_float(drivers.get("causal_model_min_oos_r2"))
        if min_oos is not None:
            oos_vals.append(float(min_oos))
            if float(min_oos) < float(min_oos_floor):
                low_oos += 1

        min_treated = _to_float(drivers.get("causal_model_min_treated_rows"))
        if min_treated is not None:
            treated_vals.append(float(min_treated))
            if int(min_treated) < int(min_treated_floor):
                low_treated += 1

        min_control = _to_float(drivers.get("causal_model_min_control_rows"))
        if min_control is not None:
            control_vals.append(float(min_control))
            if int(min_control) < int(min_control_floor):
                low_control += 1

        u = _to_float(((ac.get("impact_distribution", {}) or {}).get("uncertainty_score")))
        if u is not None:
            uncertainty_vals.append(float(u))

    quality_n = len(quality_vals)
    support_n = len(support_vals)
    oos_n = len(oos_vals)
    treated_n = len(treated_vals)
    control_n = len(control_vals)

    summary = {
        "total_candidates": total,
        "feasible_candidates": feasible_total,
        "causal_present_rate": round(_safe_ratio(causal_present, total), 6),
        "standalone_applied_rate": round(_safe_ratio(standalone_applied, max(1, causal_present)), 6),
        "standalone_fallback_rate": round(_safe_ratio(standalone_fallback, max(1, causal_present)), 6),
        "strict_gate_pass_rate": round(_safe_ratio(strict_gate_pass, max(1, causal_present)), 6),
        "strict_gate_fail_rate": round(_safe_ratio(strict_gate_fail, max(1, causal_present)), 6),
        "oos_penalty_rate": round(_safe_ratio(oos_penalty, total), 6),
        "low_quality_rate": round(_safe_ratio(low_quality, max(1, quality_n)), 6),
        "low_support_rate": round(_safe_ratio(low_support, max(1, support_n)), 6),
        "low_min_oos_r2_rate": round(_safe_ratio(low_oos, max(1, oos_n)), 6),
        "low_treated_coverage_rate": round(_safe_ratio(low_treated, max(1, treated_n)), 6),
        "low_control_coverage_rate": round(_safe_ratio(low_control, max(1, control_n)), 6),
        "avg_model_quality": round(sum(quality_vals) / quality_n, 6) if quality_n else None,
        "avg_support_score": round(sum(support_vals) / support_n, 6) if support_n else None,
        "avg_min_oos_r2": round(sum(oos_vals) / oos_n, 6) if oos_n else None,
        "uncertainty_p50": round(_quantile(uncertainty_vals, 0.50), 6) if uncertainty_vals else None,
        "uncertainty_p90": round(_quantile(uncertainty_vals, 0.90), 6) if uncertainty_vals else None,
    }

    sensitivity_suite = {
        "quality_floor_0": {
            "pass_count": sum(1 for v in quality_vals if v >= 0.0),
            "total_scored": quality_n,
            "pass_rate": round(_safe_ratio(sum(1 for v in quality_vals if v >= 0.0), max(1, quality_n)), 6),
        },
        "support_floor_025": {
            "pass_count": sum(1 for v in support_vals if v >= 0.25),
            "total_scored": support_n,
            "pass_rate": round(_safe_ratio(sum(1 for v in support_vals if v >= 0.25), max(1, support_n)), 6),
        },
        "strict_gate_quality0_support025": {
            "pass_count": min(
                sum(1 for v in quality_vals if v >= 0.0),
                sum(1 for v in support_vals if v >= 0.25),
            ),
            "total_candidates": total,
            "pass_rate_vs_candidates": round(
                _safe_ratio(
                    min(
                        sum(1 for v in quality_vals if v >= 0.0),
                        sum(1 for v in support_vals if v >= 0.25),
                    ),
                    max(1, total),
                ),
                6,
            ),
        },
        "strict_gate_full": {
            "pass_count": int(strict_gate_pass),
            "total_causal_candidates": int(causal_present),
            "pass_rate": round(_safe_ratio(strict_gate_pass, max(1, causal_present)), 6),
        },
    }

    challenger_monitoring: Dict[str, Any] = {}
    if isinstance(previous_report, dict):
        prev_summary = dict(previous_report.get("summary", {}) or {})
        challenger_monitoring = {
            "has_previous_report": True,
            "previous_generated_at": previous_report.get("generated_at"),
            "delta_low_quality_rate": round(
                float(summary["low_quality_rate"]) - float(prev_summary.get("low_quality_rate", 0.0) or 0.0), 6
            ),
            "delta_low_support_rate": round(
                float(summary["low_support_rate"]) - float(prev_summary.get("low_support_rate", 0.0) or 0.0), 6
            ),
            "delta_oos_penalty_rate": round(
                float(summary["oos_penalty_rate"]) - float(prev_summary.get("oos_penalty_rate", 0.0) or 0.0), 6
            ),
        }
    else:
        challenger_monitoring = {"has_previous_report": False}

    alerts: List[str] = []
    if summary["oos_penalty_rate"] > 0.20:
        alerts.append("High OOS penalty rate: consider narrowing candidate scope or retraining causal models.")
    if summary["low_quality_rate"] > 0.50:
        alerts.append("More than half of causal-scored candidates have negative model quality.")
    if summary["standalone_fallback_rate"] > 0.30:
        alerts.append("Standalone causal fallback rate is elevated; investigate support/quality thresholds.")

    action_breakdown = []
    for aid, vals in sorted(by_action.items()):
        cnt = int(vals["count"])
        action_breakdown.append(
            {
                "action_id": aid,
                "count": cnt,
                "causal_present_rate": round(_safe_ratio(int(vals["causal_present"]), cnt), 6),
                "standalone_applied_rate": round(_safe_ratio(int(vals["standalone_applied"]), max(1, int(vals["causal_present"]))), 6),
                "standalone_fallback_rate": round(_safe_ratio(int(vals["standalone_fallback"]), max(1, int(vals["causal_present"]))), 6),
                "oos_penalty_rate": round(_safe_ratio(int(vals["oos_penalty"]), cnt), 6),
            }
        )

    return {
        "run_id": str(getattr(run, "run_id", "")),
        "company_id": str(getattr(run, "company_id", "")),
        "as_of_time": str(getattr(run, "as_of_time", "")),
        "generated_at": _now_iso(),
        "snapshot_hash": str(getattr(getattr(run, "frozen_state", None), "snapshot_hash", "")),
        "mechanism_model_version": str(getattr(getattr(run, "model_versions", None), "mechanism_model_version", "")),
        "causal_mode": "standalone" if "mode_standalone" in str(getattr(getattr(run, "model_versions", None), "mechanism_model_version", "")) else "blend",
        "summary": summary,
        "sensitivity_suite": sensitivity_suite,
        "challenger_monitoring": challenger_monitoring,
        "action_breakdown": action_breakdown,
        "alerts": alerts,
        "snapshot_regime": dict(snapshot.get("regime", {}) or {}) if isinstance(snapshot, dict) else {},
    }


__all__ = ["build_causal_model_risk_report"]
