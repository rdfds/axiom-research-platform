from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .action_ontology import build_default_action_schema_registry
from .planner_brain import build_plan_set
from .recommendation_run import RecommendationRun


def build_planner_eval_report(
    runs_roots: Sequence[str | Path],
    run_ids: Optional[Sequence[str]] = None,
    review_count: int = 50,
    limit: Optional[int] = None,
    rebuild_plan_set: bool = True,
) -> Dict[str, Any]:
    resolved_roots = [Path(root) for root in runs_roots]
    selected_run_ids = _resolve_run_ids(runs_roots=resolved_roots, run_ids=run_ids, limit=limit)

    cases: List[Dict[str, Any]] = []
    missing_artifacts: List[Dict[str, Any]] = []
    for run_id, runs_root in selected_run_ids:
        try:
            cases.append(_build_case_report(runs_root=runs_root, run_id=run_id, rebuild_plan_set=rebuild_plan_set))
        except FileNotFoundError as exc:
            missing_artifacts.append(
                {
                    "run_id": run_id,
                    "runs_root": str(runs_root),
                    "error": str(exc),
                }
            )

    aggregate = _aggregate_cases(cases=cases, missing_artifacts=missing_artifacts)
    review_queue = _select_review_queue(cases=cases, review_count=review_count)
    return {
        "ok": True,
        "runs_analyzed": len(cases),
        "missing_artifacts": missing_artifacts,
        "aggregate": aggregate,
        "review_queue": review_queue,
        "cases": cases,
    }


def render_planner_eval_markdown(report: Dict[str, Any]) -> str:
    aggregate = dict(report.get("aggregate", {}) or {})
    lines: List[str] = []
    lines.append("# Planner Evaluation Report")
    lines.append("")
    lines.append(f"- Runs analyzed: `{report.get('runs_analyzed', 0)}`")
    lines.append(f"- Missing artifacts: `{len(report.get('missing_artifacts', []) or [])}`")
    lines.append(f"- Heuristic overall mean: `{aggregate.get('heuristic_overall_mean', 0.0):.3f}`")
    lines.append(f"- Positive top-plan raw-score rate: `{aggregate.get('positive_top_plan_rate', 0.0):.3f}`")
    lines.append(f"- Supported top-plan step rate: `{aggregate.get('supported_top_plan_rate', 0.0):.3f}`")
    lines.append(f"- Explanation completeness rate: `{aggregate.get('explanation_complete_rate', 0.0):.3f}`")
    lines.append("")

    bucket_counts = dict(aggregate.get("bucket_counts", {}) or {})
    if bucket_counts:
        lines.append("## Bucket Mix")
        lines.append("")
        for bucket, count in sorted(bucket_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{bucket}`: `{count}`")
        lines.append("")

    flag_counts = dict(aggregate.get("flag_counts", {}) or {})
    if flag_counts:
        lines.append("## Heuristic Flags")
        lines.append("")
        for flag, count in sorted(flag_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{flag}`: `{count}`")
        lines.append("")

    lines.append("## Human Review Queue")
    lines.append("")
    for idx, case in enumerate(report.get("review_queue", []) or [], start=1):
        lines.extend(_render_case_markdown(case=case, index=idx))
    return "\n".join(lines).strip() + "\n"


def _render_case_markdown(case: Dict[str, Any], index: int) -> List[str]:
    lines: List[str] = []
    top_plan = dict(case.get("top_plan", {}) or {})
    top_three = list(case.get("top_three_paths", []) or [])
    support = dict(case.get("top_plan_support", {}) or {})
    heuristic = dict(case.get("heuristic", {}) or {})

    lines.append(f"### {index}. `{case.get('company_id')}` / `{case.get('bucket')}`")
    lines.append("")
    lines.append(f"- Run: `{case.get('run_id')}`")
    lines.append(f"- Top plan: `{top_plan.get('action_path', '')}`")
    lines.append(f"- Top plan raw score: `{top_plan.get('raw_total_score', 0.0):.3f}`")
    lines.append(f"- Heuristic overall score: `{heuristic.get('overall_score', 0.0):.3f}`")
    lines.append(f"- Flags: `{', '.join(heuristic.get('flags', []) or ['none'])}`")
    lines.append(f"- Summary: {top_plan.get('summary_explanation') or 'missing'}")
    lines.append(f"- Top 3: `{'; '.join(top_three)}`")
    lines.append(
        "- Support: "
        f"`precedent_mean={support.get('avg_precedent_confidence', 0.0):.3f}` "
        f"`causal_step_rate={support.get('causal_step_rate', 0.0):.3f}` "
        f"`pass_prob_mean={support.get('avg_pass_probability', 0.0):.3f}`"
    )
    lines.append("")
    lines.append("- [ ] Top-1 plan is strategically sensible")
    lines.append("- [ ] Top-3 contains no obvious nonsense")
    lines.append("- [ ] Explanation is persuasive and numbers-backed")
    lines.append("- [ ] Risks / triggers / branches are useful")
    lines.append("")
    for step in list(case.get("top_plan_step_cards", []) or []):
        lines.append(
            f"- Step `{step.get('action_id')}`: "
            f"`pass={step.get('pass_probability', 0.0):.3f}` "
            f"`eval={step.get('evaluation_confidence', 0.0):.3f}` "
            f"`precedent={step.get('precedent_confidence', 0.0):.3f}` "
            f"`causal={step.get('has_causal')}` "
            f"`impact={step.get('impact_snapshot')}`"
        )
    lines.append("")
    return lines


def _resolve_run_ids(
    runs_roots: Sequence[Path],
    run_ids: Optional[Sequence[str]],
    limit: Optional[int],
) -> List[Tuple[str, Path]]:
    by_id: List[Tuple[str, Path]] = []
    explicit = list(run_ids or [])
    if explicit:
        for run_id in explicit:
            found = False
            for runs_root in runs_roots:
                run_path = runs_root / "runs" / f"run_id={run_id}.json"
                if run_path.exists():
                    by_id.append((run_id, runs_root))
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"run_id={run_id} not found under any runs root")
    else:
        for runs_root in runs_roots:
            for run_path in sorted((runs_root / "runs").glob("run_id=*.json")):
                run_id = run_path.stem.replace("run_id=", "", 1)
                by_id.append((run_id, runs_root))
    if limit is not None:
        by_id = by_id[: max(0, int(limit))]
    return by_id


def _build_case_report(runs_root: Path, run_id: str, rebuild_plan_set: bool) -> Dict[str, Any]:
    run_payload = json.loads((runs_root / "runs" / f"run_id={run_id}.json").read_text())
    artifacts_root = runs_root / "artifacts" / f"run_id={run_id}"
    feasibility = json.loads((artifacts_root / "FeasibilityResults.json").read_text())
    precedent = json.loads((artifacts_root / "PrecedentMatches.json").read_text())
    if rebuild_plan_set:
        recommendation_run = RecommendationRun.from_dict(run_payload)
        feasible_candidates = [
            row.get("action_candidate") or row.get("candidate") or {}
            for row in list(feasibility.get("results", []) or [])
            if row.get("feasible")
        ]
        stored_plan_set = json.loads((artifacts_root / "PlanSet.json").read_text()) if (artifacts_root / "PlanSet.json").exists() else {}
        top_plans = max(3, len(list(stored_plan_set.get("plans", []) or [])))
        plan_set = build_plan_set(
            run=recommendation_run,
            feasible_candidates=feasible_candidates,
            precedent_matches=list(precedent.get("results", []) or []),
            registry=build_default_action_schema_registry(),
            top_plans=top_plans,
        )
    else:
        plan_set = json.loads((artifacts_root / "PlanSet.json").read_text())

    feasible_rows = [row for row in list(feasibility.get("results", []) or []) if row.get("feasible")]
    support_by_action = _best_support_by_action(feasible_rows=feasible_rows, precedent_rows=list(precedent.get("results", []) or []))
    plans = list(plan_set.get("plans", []) or [])
    top_plan = dict(plans[0] or {}) if plans else {}

    bucket = _infer_bucket(feasible_rows=feasible_rows, top_plan=top_plan)
    top_plan_steps = list(top_plan.get("steps", []) or [])
    top_plan_support = _top_plan_support(step_actions=[step.get("action_id") for step in top_plan_steps], support_by_action=support_by_action)
    explanation = _explanation_score(top_plan=top_plan)
    structural = _structural_score(top_plan=top_plan)
    top_three_quality = _top_three_quality(plans=plans, support_by_action=support_by_action)
    heuristic = _heuristic_summary(
        top_plan=top_plan,
        top_plan_support=top_plan_support,
        structural=structural,
        explanation=explanation,
        top_three_quality=top_three_quality,
    )

    return {
        "run_id": run_id,
        "runs_root": str(runs_root),
        "company_id": run_payload.get("company_id"),
        "bucket": bucket,
        "plan_count": len(plans),
        "feasible_action_count": len(feasible_rows),
        "top_plan": {
            "action_path": " -> ".join(step.get("action_id", "") for step in top_plan_steps),
            "score": float(top_plan.get("score", 0.0) or 0.0),
            "raw_total_score": float(((top_plan.get("score_components", {}) or {}).get("raw_total_score", top_plan.get("score", 0.0)) or 0.0)),
            "summary_explanation": top_plan.get("summary_explanation"),
        },
        "top_three_paths": [
            " -> ".join(step.get("action_id", "") for step in list(plan.get("steps", []) or []))
            for plan in plans[:3]
        ],
        "top_plan_support": top_plan_support,
        "top_plan_step_cards": [
            _step_card(action_id=step.get("action_id"), support=support_by_action.get(step.get("action_id")))
            for step in top_plan_steps
        ],
        "heuristic": heuristic,
    }


def _best_support_by_action(feasible_rows: Sequence[Dict[str, Any]], precedent_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    precedent_by_candidate_id: Dict[str, float] = {}
    precedent_by_action_id: Dict[str, float] = {}
    for row in precedent_rows:
        candidate = dict(row.get("candidate", {}) or {})
        action_id = str(candidate.get("action_id", "") or "")
        candidate_id = str(candidate.get("candidate_id", "") or "")
        precedent_pack = dict(row.get("precedent_pack", {}) or {})
        confidence = float(
            precedent_pack.get("precedent_confidence")
            or precedent_pack.get("calibration_confidence")
            or 0.0
        )
        if candidate_id:
            precedent_by_candidate_id[candidate_id] = max(confidence, precedent_by_candidate_id.get(candidate_id, 0.0))
        if action_id:
            precedent_by_action_id[action_id] = max(confidence, precedent_by_action_id.get(action_id, 0.0))

    best: Dict[str, Dict[str, Any]] = {}
    for row in feasible_rows:
        candidate = dict(row.get("action_candidate") or row.get("candidate") or {})
        action_id = str(candidate.get("action_id", "") or "")
        if not action_id:
            continue
        candidate_id = str(candidate.get("candidate_id", "") or "")
        impact = dict(candidate.get("impact_distribution", {}) or {})
        objectives = dict(impact.get("objectives", {}) or {})
        entry = {
            "action_id": action_id,
            "pass_probability": float((row.get("pass_probability") or (candidate.get("feasibility", {}) or {}).get("pass_probability") or 0.0)),
            "evaluation_confidence": float(candidate.get("evaluation_confidence", 0.0) or 0.0),
            "precedent_confidence": float(precedent_by_candidate_id.get(candidate_id) or precedent_by_action_id.get(action_id) or 0.0),
            "has_causal": _has_causal(candidate),
            "impact_snapshot": {
                key: round(float((objectives.get(key, {}) or {}).get("median", 0.0) or 0.0), 3)
                for key in ["value_creation", "risk_reduction", "growth", "rating_preservation", "optionality"]
            },
        }
        score = (
            entry["evaluation_confidence"]
            + entry["precedent_confidence"]
            + entry["pass_probability"]
            + (0.1 if entry["has_causal"] else 0.0)
        )
        current = best.get(action_id)
        if current is None or score > current["_selection_score"]:
            entry["_selection_score"] = score
            best[action_id] = entry

    for payload in best.values():
        payload.pop("_selection_score", None)
    return best


def _infer_bucket(feasible_rows: Sequence[Dict[str, Any]], top_plan: Dict[str, Any]) -> str:
    feasible_actions = {
        str((row.get("action_candidate") or row.get("candidate") or {}).get("action_id", "") or "")
        for row in feasible_rows
    }
    if any(action_id.startswith("mna.") for action_id in feasible_actions):
        return "acquisition"
    if any(action_id in {"portfolio.divestiture_partial", "portfolio.divestiture_full", "portfolio.asset_sale"} for action_id in feasible_actions):
        return "divestiture"
    if (
        any(action_id in {"capital_structure.new_debt_issuance", "capital_structure.refinancing"} for action_id in feasible_actions)
        and any(action_id in {"capital_return.open_market_buyback", "capital_return.accelerated_share_repurchase", "capital_return.tender_offer_buyback"} for action_id in feasible_actions)
    ):
        return "buyback_refi"
    top_steps = list(top_plan.get("steps", []) or [])
    if top_steps:
        return str(top_steps[0].get("action_id", "other")).split(".", 1)[0]
    return "other"


def _top_plan_support(step_actions: Sequence[str], support_by_action: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    rows = [support_by_action[action_id] for action_id in step_actions if action_id in support_by_action]
    if not rows:
        return {
            "avg_precedent_confidence": 0.0,
            "causal_step_rate": 0.0,
            "avg_pass_probability": 0.0,
            "all_steps_supported": False,
        }
    return {
        "avg_precedent_confidence": round(sum(row["precedent_confidence"] for row in rows) / len(rows), 6),
        "causal_step_rate": round(sum(1.0 for row in rows if row["has_causal"]) / len(rows), 6),
        "avg_pass_probability": round(sum(row["pass_probability"] for row in rows) / len(rows), 6),
        "all_steps_supported": all((row["precedent_confidence"] > 0.0) or row["has_causal"] for row in rows),
    }


def _explanation_score(top_plan: Dict[str, Any]) -> Dict[str, Any]:
    top_steps = list(top_plan.get("steps", []) or [])
    summary_present = bool(str(top_plan.get("summary_explanation", "") or "").strip())
    complete_steps = 0
    for step in top_steps:
        explanation = dict(step.get("explanation", {}) or {})
        if (
            str(explanation.get("problem_statement", "") or "").strip()
            and str(explanation.get("why_this_action", "") or "").strip()
            and str(explanation.get("why_now", "") or "").strip()
        ):
            complete_steps += 1
    completeness = 0.0
    if top_steps:
        completeness = complete_steps / len(top_steps)
    score = ((0.4 if summary_present else 0.0) + (0.6 * completeness))
    return {
        "summary_present": summary_present,
        "complete_step_rate": round(completeness, 6),
        "score": round(score, 6),
    }


def _structural_score(top_plan: Dict[str, Any]) -> Dict[str, Any]:
    top_steps = list(top_plan.get("steps", []) or [])
    action_ids = [str(step.get("action_id", "") or "") for step in top_steps]
    duplicates = len(set(action_ids)) != len(action_ids)
    prerequisite_ok = True
    seen: set[str] = set()
    for step in top_steps:
        prereqs = set(step.get("prerequisites", []) or [])
        if not prereqs.issubset(seen):
            prerequisite_ok = False
            break
        seen.add(str(step.get("action_id", "") or ""))
    raw_total = float(((top_plan.get("score_components", {}) or {}).get("raw_total_score", top_plan.get("score", 0.0)) or 0.0))
    components = [
        1.0 if top_steps else 0.0,
        1.0 if not duplicates else 0.0,
        1.0 if prerequisite_ok else 0.0,
        1.0 if raw_total > 0.0 else 0.0,
    ]
    return {
        "duplicates": duplicates,
        "prerequisite_ok": prerequisite_ok,
        "raw_total_score": round(raw_total, 6),
        "score": round(sum(components) / len(components), 6),
    }


def _top_three_quality(plans: Sequence[Dict[str, Any]], support_by_action: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    top_three = list(plans[:3] or [])
    if not top_three:
        return {
            "positive_rate": 0.0,
            "unique_path_rate": 0.0,
            "unsupported_rate": 1.0,
            "score": 0.0,
        }
    paths = []
    positive = 0
    unsupported = 0
    for plan in top_three:
        steps = list(plan.get("steps", []) or [])
        paths.append(tuple(step.get("action_id") for step in steps))
        raw_total = float(((plan.get("score_components", {}) or {}).get("raw_total_score", plan.get("score", 0.0)) or 0.0))
        if raw_total > 0.0:
            positive += 1
        if any(
            not ((support_by_action.get(step.get("action_id"), {}) or {}).get("precedent_confidence", 0.0) > 0.0 or (support_by_action.get(step.get("action_id"), {}) or {}).get("has_causal", False))
            for step in steps
        ):
            unsupported += 1
    positive_rate = positive / len(top_three)
    unique_path_rate = len(set(paths)) / len(top_three)
    unsupported_rate = unsupported / len(top_three)
    score = (0.45 * positive_rate) + (0.25 * unique_path_rate) + (0.30 * (1.0 - unsupported_rate))
    return {
        "positive_rate": round(positive_rate, 6),
        "unique_path_rate": round(unique_path_rate, 6),
        "unsupported_rate": round(unsupported_rate, 6),
        "score": round(score, 6),
    }


def _heuristic_summary(
    top_plan: Dict[str, Any],
    top_plan_support: Dict[str, Any],
    structural: Dict[str, Any],
    explanation: Dict[str, Any],
    top_three_quality: Dict[str, Any],
) -> Dict[str, Any]:
    support_score = (
        (0.35 * float(top_plan_support.get("avg_precedent_confidence", 0.0) or 0.0))
        + (0.25 * float(top_plan_support.get("avg_pass_probability", 0.0) or 0.0))
        + (0.20 * float(top_plan_support.get("causal_step_rate", 0.0) or 0.0))
        + (0.20 * (1.0 if top_plan_support.get("all_steps_supported") else 0.0))
    )
    overall = (
        0.30 * float(structural.get("score", 0.0) or 0.0)
        + 0.30 * float(explanation.get("score", 0.0) or 0.0)
        + 0.20 * float(top_three_quality.get("score", 0.0) or 0.0)
        + 0.20 * support_score
    )
    flags: List[str] = []
    if float(structural.get("raw_total_score", 0.0) or 0.0) <= 0.0:
        flags.append("top_plan_nonpositive")
    if not top_plan_support.get("all_steps_supported"):
        flags.append("top_plan_unsupported_step")
    if not explanation.get("summary_present"):
        flags.append("top_plan_summary_missing")
    if float(explanation.get("complete_step_rate", 0.0) or 0.0) < 1.0:
        flags.append("step_explanation_incomplete")
    if float(top_three_quality.get("positive_rate", 0.0) or 0.0) < 1.0:
        flags.append("top3_contains_nonpositive")
    if float(top_three_quality.get("unique_path_rate", 0.0) or 0.0) < 1.0:
        flags.append("top3_duplicate_paths")

    return {
        "structural_score": round(float(structural.get("score", 0.0) or 0.0), 6),
        "support_score": round(support_score, 6),
        "explanation_score": round(float(explanation.get("score", 0.0) or 0.0), 6),
        "top3_score": round(float(top_three_quality.get("score", 0.0) or 0.0), 6),
        "overall_score": round(overall, 6),
        "flags": flags,
    }


def _step_card(action_id: Optional[str], support: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    support = dict(support or {})
    return {
        "action_id": action_id,
        "pass_probability": float(support.get("pass_probability", 0.0) or 0.0),
        "evaluation_confidence": float(support.get("evaluation_confidence", 0.0) or 0.0),
        "precedent_confidence": float(support.get("precedent_confidence", 0.0) or 0.0),
        "has_causal": bool(support.get("has_causal", False)),
        "impact_snapshot": dict(support.get("impact_snapshot", {}) or {}),
    }


def _has_causal(candidate: Dict[str, Any]) -> bool:
    impact = dict(candidate.get("impact_distribution", {}) or {})
    for driver in list(impact.get("key_drivers", []) or []):
        if str(driver.get("driver_name", "") or "").startswith("causal_model_"):
            return True
    return False


def _aggregate_cases(cases: Sequence[Dict[str, Any]], missing_artifacts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    bucket_counts = Counter()
    flag_counts = Counter()
    overall_scores: List[float] = []
    positive_top = 0
    supported_top = 0
    explanation_complete = 0

    for case in cases:
        bucket_counts[str(case.get("bucket") or "other")] += 1
        heuristic = dict(case.get("heuristic", {}) or {})
        for flag in list(heuristic.get("flags", []) or []):
            flag_counts[str(flag)] += 1
        overall_scores.append(float(heuristic.get("overall_score", 0.0) or 0.0))
        if float(((case.get("top_plan", {}) or {}).get("raw_total_score", 0.0) or 0.0) > 0.0):
            positive_top += 1
        if bool(((case.get("top_plan_support", {}) or {}).get("all_steps_supported", False))):
            supported_top += 1
        if "step_explanation_incomplete" not in list(heuristic.get("flags", []) or []) and "top_plan_summary_missing" not in list(heuristic.get("flags", []) or []):
            explanation_complete += 1

    denom = max(1, len(cases))
    return {
        "bucket_counts": dict(bucket_counts),
        "flag_counts": dict(flag_counts),
        "heuristic_overall_mean": round(sum(overall_scores) / denom, 6),
        "positive_top_plan_rate": round(positive_top / denom, 6),
        "supported_top_plan_rate": round(supported_top / denom, 6),
        "explanation_complete_rate": round(explanation_complete / denom, 6),
        "missing_artifact_rate": round(len(missing_artifacts) / max(1, len(cases) + len(missing_artifacts)), 6),
    }


def _select_review_queue(cases: Sequence[Dict[str, Any]], review_count: int) -> List[Dict[str, Any]]:
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_bucket[str(case.get("bucket") or "other")].append(case)
    for bucket_cases in by_bucket.values():
        bucket_cases.sort(key=lambda item: (float((item.get("heuristic", {}) or {}).get("overall_score", 0.0) or 0.0), str(item.get("company_id", ""))))

    ordered_buckets = sorted(by_bucket, key=lambda key: (len(by_bucket[key]), key))
    selected: List[Dict[str, Any]] = []
    while ordered_buckets and len(selected) < max(1, int(review_count)):
        next_round: List[str] = []
        for bucket in ordered_buckets:
            if len(selected) >= max(1, int(review_count)):
                break
            bucket_cases = by_bucket[bucket]
            if not bucket_cases:
                continue
            selected.append(bucket_cases.pop(0))
            if bucket_cases:
                next_round.append(bucket)
        ordered_buckets = next_round
    return selected


__all__ = [
    "build_planner_eval_report",
    "render_planner_eval_markdown",
]
