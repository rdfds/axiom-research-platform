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


