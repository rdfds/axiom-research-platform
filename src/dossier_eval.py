from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .action_ontology import build_default_action_schema_registry
from .board_ready_dossier import build_board_ready_dossier
from .planner_brain import build_plan_set
from .recommendation_run import RecommendationRun


_RAW_ACTION_ID_RE = re.compile(r"\b[a-z_]+\.[a-z0-9_]+\b")


def build_dossier_eval_report(
    runs_roots: Sequence[str | Path],
    snapshot_root: str | Path,
    run_ids: Optional[Sequence[str]] = None,
    review_count: int = 50,
    limit: Optional[int] = None,
    expected_postures: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_roots = [Path(root) for root in runs_roots]
    snapshot_root_path = Path(snapshot_root)
    selected_run_ids = _resolve_run_ids(runs_roots=resolved_roots, run_ids=run_ids, limit=limit)
    registry = build_default_action_schema_registry()

    cases: List[Dict[str, Any]] = []
    missing_artifacts: List[Dict[str, Any]] = []
    for run_id, runs_root in selected_run_ids:
        try:
            cases.append(
                _build_case_report(
                    runs_root=runs_root,
                    snapshot_root=snapshot_root_path,
                    run_id=run_id,
                    registry=registry,
                    expected_postures=expected_postures or {},
                )
            )
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


def render_dossier_eval_markdown(report: Dict[str, Any]) -> str:
    aggregate = dict(report.get("aggregate", {}) or {})
    lines: List[str] = []
    lines.append("# Board Dossier Evaluation Report")
    lines.append("")
    lines.append(f"- Runs analyzed: `{report.get('runs_analyzed', 0)}`")
    lines.append(f"- Missing artifacts: `{len(report.get('missing_artifacts', []) or [])}`")
    lines.append(f"- Heuristic overall mean: `{aggregate.get('heuristic_overall_mean', 0.0):.3f}`")
    lines.append(f"- Completeness rate: `{aggregate.get('completeness_rate', 0.0):.3f}`")
    lines.append(f"- Humanized language rate: `{aggregate.get('humanized_language_rate', 0.0):.3f}`")
    lines.append(f"- Specific timing rate: `{aggregate.get('specific_timing_rate', 0.0):.3f}`")
    lines.append(f"- Alternatives present rate: `{aggregate.get('alternatives_present_rate', 0.0):.3f}`")
    lines.append(f"- Risk specificity rate: `{aggregate.get('risk_specificity_rate', 0.0):.3f}`")
    lines.append(f"- Status-quo comparison rate: `{aggregate.get('status_quo_comparison_rate', 0.0):.3f}`")
    lines.append(f"- Sizing specificity rate: `{aggregate.get('sizing_specificity_rate', 0.0):.3f}`")
    lines.append(f"- Parameter optimization rate: `{aggregate.get('parameter_optimization_rate', 0.0):.3f}`")
    lines.append(f"- Regret analysis rate: `{aggregate.get('regret_analysis_rate', 0.0):.3f}`")
    lines.append(f"- Scenario sizing rate: `{aggregate.get('scenario_sizing_rate', 0.0):.3f}`")
    lines.append(f"- Rating analysis rate: `{aggregate.get('rating_analysis_rate', 0.0):.3f}`")
    lines.append(f"- Signaling analysis rate: `{aggregate.get('signaling_analysis_rate', 0.0):.3f}`")
    if aggregate.get("expected_posture_coverage_rate") is not None:
        lines.append(f"- Expected-posture coverage rate: `{aggregate.get('expected_posture_coverage_rate', 0.0):.3f}`")
        lines.append(f"- Posture match rate: `{aggregate.get('posture_match_rate', 0.0):.3f}`")
        negative_case_accuracy = aggregate.get("negative_case_accuracy")
        lines.append(
            f"- Negative-case accuracy: `{negative_case_accuracy:.3f}`"
            if negative_case_accuracy is not None
            else "- Negative-case accuracy: `n/a`"
        )
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
    dossier = dict(case.get("dossier", {}) or {})
    thesis = dict(dossier.get("recommendation_thesis", {}) or {})
    risk_case = dict(dossier.get("risk_case", {}) or {})
    lines: List[str] = []
    lines.append(f"### {index}. `{case.get('company_id')}` / `{case.get('top_action')}`")
    lines.append("")
    lines.append(f"- Run: `{case.get('run_id')}`")
    lines.append(f"- Heuristic overall score: `{case.get('heuristic', {}).get('overall_score', 0.0):.3f}`")
    lines.append(f"- Flags: `{', '.join(case.get('heuristic', {}).get('flags', []) or ['none'])}`")
    lines.append(f"- Confidence posture: `{dossier.get('confidence_posture', 'unknown')}`")
    lines.append(f"- Recommended posture: `{dossier.get('status_quo_view', {}).get('recommended_posture', 'unknown')}`")
    lines.append(f"- Executive summary: {dossier.get('executive_summary') or 'missing'}")
    lines.append(f"- Why now: {thesis.get('why_now') or 'missing'}")
    lines.append(f"- Why wait: {dossier.get('status_quo_view', {}).get('why_wait') or 'missing'}")
    lines.append(f"- Sizing: {dossier.get('sizing_guidance', {}).get('recommended_range') or 'missing'}")
    lines.append(f"- Parameter summary: {dossier.get('parameter_optimization', {}).get('summary') or 'missing'}")
    lines.append(f"- Regret balance: {dossier.get('regret_analysis', {}).get('regret_balance') or 'missing'}")
    lines.append(f"- Rating constraint: {dossier.get('rating_cliff_analysis', {}).get('constraint_posture') or 'missing'}")
    lines.append(f"- Signal posture: {dossier.get('signaling_analysis', {}).get('signal_posture') or 'missing'}")
    lines.append(f"- Why not alternatives: `{len(dossier.get('alternative_analysis', []) or [])}` alternatives")
    lines.append(f"- Kill criteria: `{'; '.join(risk_case.get('kill_criteria', []) or []) or 'missing'}`")
    if case.get("expected_posture"):
        lines.append(f"- Expected posture: `{case.get('expected_posture')}` / matched: `{case.get('posture_match')}`")
    lines.append("")
    lines.append("- [ ] Diagnosis is specific to the company")
    lines.append("- [ ] Recommendation thesis is persuasive")
    lines.append("- [ ] Why now is concrete and not generic")
    lines.append("- [ ] Alternatives are rebutted credibly")
    lines.append("- [ ] Risks and kill criteria are decision-useful")
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
            for runs_root in runs_roots:
                run_path = runs_root / "runs" / f"run_id={run_id}.json"
                if run_path.exists():
                    by_id.append((run_id, runs_root))
                    break
            else:
                raise FileNotFoundError(f"run_id={run_id} not found under any runs root")
    else:
        for runs_root in runs_roots:
            for run_path in sorted((runs_root / "runs").glob("run_id=*.json")):
                by_id.append((run_path.stem.replace("run_id=", "", 1), runs_root))
    if limit is not None:
        by_id = by_id[: max(0, int(limit))]
    return by_id


def _build_case_report(
    *,
    runs_root: Path,
    snapshot_root: Path,
    run_id: str,
    registry: Any,
    expected_postures: Dict[str, Any],
) -> Dict[str, Any]:
    run_payload = json.loads((runs_root / "runs" / f"run_id={run_id}.json").read_text())
    recommendation_run = RecommendationRun.from_dict(run_payload)
    artifacts_root = runs_root / "artifacts" / f"run_id={run_id}"
    feasibility = json.loads((artifacts_root / "FeasibilityResults.json").read_text())
    precedent = json.loads((artifacts_root / "PrecedentMatches.json").read_text())
    feasible_candidates = [
        row.get("action_candidate") or row.get("candidate") or {}
        for row in list(feasibility.get("results", []) or [])
        if row.get("feasible")
    ]
    plan_set = build_plan_set(
        run=recommendation_run,
        feasible_candidates=feasible_candidates,
        precedent_matches=list(precedent.get("results", []) or []),
        registry=registry,
        top_plans=5,
    )
    snapshot = _load_snapshot(snapshot_root=snapshot_root, company_id=str(recommendation_run.company_id), as_of_time=str(recommendation_run.as_of_time))
    dossier = build_board_ready_dossier(
        run=recommendation_run,
        snapshot=snapshot,
        plan_set=plan_set,
        feasible_candidates=feasible_candidates,
        precedent_matches=list(precedent.get("results", []) or []),
        registry=registry,
    )
    heuristic = _heuristic_summary(dossier=dossier)
    top_steps = list(((plan_set.get("plans", []) or [{}])[0].get("steps", []) or []))
    top_action = str((top_steps[0].get("action_id", "") if top_steps else ""))
    expectation = _resolve_expected_posture(
        expected_postures=expected_postures,
        run_id=run_id,
        company_id=str(recommendation_run.company_id),
    )
    predicted_posture = str(((dossier.get("status_quo_view", {}) or {}).get("recommended_posture", "")) or "")
    return {
        "run_id": run_id,
        "runs_root": str(runs_root),
        "company_id": recommendation_run.company_id,
        "top_action": top_action,
        "predicted_posture": predicted_posture,
        "expected_posture": expectation,
        "posture_match": (predicted_posture == expectation) if expectation else None,
        "heuristic": heuristic,
        "dossier": {
            "confidence_posture": dossier.get("confidence_posture"),
            "executive_summary": dossier.get("executive_summary"),
            "recommendation_thesis": dossier.get("recommendation_thesis"),
            "status_quo_view": dossier.get("status_quo_view"),
            "sizing_guidance": dossier.get("sizing_guidance"),
            "parameter_optimization": dossier.get("parameter_optimization"),
            "regret_analysis": dossier.get("regret_analysis"),
            "rating_cliff_analysis": dossier.get("rating_cliff_analysis"),
            "signaling_analysis": dossier.get("signaling_analysis"),
            "ranked_action_views": dossier.get("ranked_action_views"),
            "alternative_analysis": dossier.get("alternative_analysis"),
            "risk_case": dossier.get("risk_case"),
            "monitoring": dossier.get("monitoring"),
            "scorecard": dossier.get("scorecard"),
        },
    }


def _load_snapshot(*, snapshot_root: Path, company_id: str, as_of_time: str) -> Dict[str, Any]:
    as_of_date = as_of_time[:10]
    path = snapshot_root / "keyed" / f"as_of_date={as_of_date}" / f"company_id={company_id}.json"
    return json.loads(path.read_text())


def _heuristic_summary(dossier: Dict[str, Any]) -> Dict[str, Any]:
    thesis = dict(dossier.get("recommendation_thesis", {}) or {})
    risk_case = dict(dossier.get("risk_case", {}) or {})
    monitoring = dict(dossier.get("monitoring", {}) or {})
    status_quo_view = dict(dossier.get("status_quo_view", {}) or {})
    sizing_guidance = dict(dossier.get("sizing_guidance", {}) or {})
    parameter_optimization = dict(dossier.get("parameter_optimization", {}) or {})
    regret_analysis = dict(dossier.get("regret_analysis", {}) or {})
    rating_cliff_analysis = dict(dossier.get("rating_cliff_analysis", {}) or {})
    signaling_analysis = dict(dossier.get("signaling_analysis", {}) or {})
    supporting_evidence = list(dossier.get("supporting_evidence", []) or [])
    alternative_analysis = list(dossier.get("alternative_analysis", []) or [])

    flags: List[str] = []
    completeness = all(
        [
            dossier.get("executive_summary"),
            thesis.get("problem_statement"),
            thesis.get("why_this_plan"),
            thesis.get("why_now"),
            thesis.get("what_has_to_be_true"),
            thesis.get("what_would_change_our_mind"),
            risk_case.get("kill_criteria"),
        ]
    )
    if not completeness:
        flags.append("missing_core_thesis")

    humanized_ok = not _contains_raw_action_ids(
        [
            dossier.get("executive_summary"),
            thesis.get("why_this_plan"),
            thesis.get("why_now"),
            *list(thesis.get("what_would_change_our_mind", []) or []),
            *list(risk_case.get("main_failure_modes", []) or []),
            *[item.get("condition") for item in list(monitoring.get("triggers", []) or [])],
            *[item.get("branch_condition") for item in list(monitoring.get("branches", []) or [])],
        ]
    )
    if not humanized_ok:
        flags.append("raw_action_id_leak")

    specific_timing = _specific_timing_score(str(thesis.get("why_now", "") or "")) >= 0.6
    if not specific_timing:
        flags.append("generic_why_now")

    evidence_quality = _evidence_quality_score(supporting_evidence)
    if evidence_quality < 0.6:
        flags.append("weak_evidence_stack")

    alternatives_present = len(alternative_analysis) > 0
    if not alternatives_present:
        flags.append("missing_alternative_rebuttal")

    alternative_depth = any(
        "lower expected utility" in str(item.get("why_not_preferred", "") or "")
        or "weaker empirical support" in str(item.get("why_not_preferred", "") or "")
        or "higher tail risk" in str(item.get("why_not_preferred", "") or "")
        or "value arrives later" in str(item.get("why_not_preferred", "") or "")
        or "more dilution" in str(item.get("why_not_preferred", "") or "")
        or "less fresh capacity" in str(item.get("why_not_preferred", "") or "")
        or "less decisive capital-return mechanism" in str(item.get("why_not_preferred", "") or "")
        or "does not unlock the planned return-of-capital step" in str(item.get("why_not_preferred", "") or "")
        or "addresses the capacity problem less directly" in str(item.get("why_not_preferred", "") or "")
        or "does not address the external-growth problem" in str(item.get("why_not_preferred", "") or "")
        for item in alternative_analysis
    )
    if alternatives_present and not alternative_depth:
        flags.append("shallow_alternative_rebuttal")

    risk_specificity = _risk_specificity_score(risk_case)
    if risk_specificity < 0.6:
        flags.append("generic_risk_case")

    monitoring_quality = bool(monitoring.get("triggers")) and bool(risk_case.get("kill_criteria"))
    if not monitoring_quality:
        flags.append("weak_monitoring")

    status_quo_comparison = all(
        [
            status_quo_view.get("recommended_posture"),
            status_quo_view.get("why_act_now"),
            status_quo_view.get("why_wait"),
            status_quo_view.get("case_for_action"),
            status_quo_view.get("case_for_wait"),
        ]
    )
    if not status_quo_comparison:
        flags.append("missing_status_quo_comparison")

    sizing_specificity = _sizing_specificity_score(sizing_guidance)
    if sizing_specificity < 0.6:
        flags.append("generic_sizing_guidance")

    parameter_optimization_score = 1.0 if parameter_optimization.get("summary") and dict(parameter_optimization.get("recommended_parameters", {}) or {}) else 0.0
    if parameter_optimization_score < 1.0:
        flags.append("missing_parameter_optimization")

    regret_quality = 1.0 if regret_analysis.get("if_we_act_and_are_wrong") and regret_analysis.get("if_we_wait_and_are_wrong") else 0.0
    if regret_quality < 1.0:
        flags.append("missing_regret_analysis")

    scenario_sizing = list(sizing_guidance.get("scenario_overrides", []) or [])
    scenario_sizing_score = 1.0 if len(scenario_sizing) >= 2 else 0.0
    if scenario_sizing_score < 1.0:
        flags.append("missing_scenario_sizing")

    rating_analysis_score = 1.0 if rating_cliff_analysis.get("constraint_posture") and list(rating_cliff_analysis.get("constraints_to_watch", []) or []) else 0.0
    if rating_analysis_score < 1.0:
        flags.append("missing_rating_analysis")

    signaling_score = 1.0 if signaling_analysis.get("signal_posture") and list(signaling_analysis.get("what_market_has_to_believe", []) or []) else 0.0
    if signaling_score < 1.0:
        flags.append("missing_signaling_analysis")

    completeness_score = 1.0 if completeness else 0.0
    humanized_score = 1.0 if humanized_ok else 0.0
    timing_score = _specific_timing_score(str(thesis.get("why_now", "") or ""))
    alternatives_score = 1.0 if alternatives_present and alternative_depth else (0.5 if alternatives_present else 0.0)
    risk_score = risk_specificity
    status_quo_score = 1.0 if status_quo_comparison else 0.0
    overall = round(
        (
            completeness_score
            + humanized_score
            + timing_score
            + evidence_quality
            + alternatives_score
            + risk_score
            + status_quo_score
            + sizing_specificity
            + parameter_optimization_score
            + regret_quality
            + scenario_sizing_score
            + rating_analysis_score
            + signaling_score
        )
        / 13.0,
        6,
    )
    return {
        "flags": flags,
        "overall_score": overall,
        "completeness_score": completeness_score,
        "humanized_score": humanized_score,
        "timing_score": round(timing_score, 6),
        "evidence_score": round(evidence_quality, 6),
        "alternatives_score": alternatives_score,
        "risk_score": round(risk_score, 6),
        "status_quo_score": status_quo_score,
        "sizing_score": round(sizing_specificity, 6),
        "parameter_optimization_score": parameter_optimization_score,
        "regret_score": regret_quality,
        "scenario_sizing_score": scenario_sizing_score,
        "rating_analysis_score": rating_analysis_score,
        "signaling_score": signaling_score,
    }


def _contains_raw_action_ids(values: Sequence[Any]) -> bool:
    for value in values:
        text = str(value or "")
        if _RAW_ACTION_ID_RE.search(text):
            return True
    return False


def _specific_timing_score(text: str) -> float:
    if not text:
        return 0.0
    score = 0.0
    lower = text.lower()
    if any(token in lower for token in ["lead time", "maturity", "window", "waiting", "urgent", "supportive now", "front-of-plan", "after "]):
        score += 0.4
    if bool(re.search(r"\b\d+(\.\d+)?\b", text)):
        score += 0.3
    if any(token in lower for token in ["credit", "equity", "liquidity", "market value"]):
        score += 0.3
    return min(score, 1.0)


def _evidence_quality_score(evidence: Sequence[Dict[str, Any]]) -> float:
    if not evidence:
        return 0.0
    snapshot_count = sum(1 for item in evidence if str(item.get("source", "")) == "snapshot")
    precedent_count = sum(1 for item in evidence if str(item.get("source", "")) == "precedent")
    modeled_count = sum(1 for item in evidence if str(item.get("source", "")) in {"causal", "causal_and_precedent", "precedent", "model_only", "feasibility"})
    numeric_count = sum(1 for item in evidence if item.get("formatted_value"))
    score = 0.0
    if len(evidence) >= 4:
        score += 0.3
    if snapshot_count >= 2:
        score += 0.3
    if precedent_count >= 1:
        score += 0.2
    if modeled_count >= 1:
        score += 0.1
    if numeric_count >= 3:
        score += 0.1
    return min(score, 1.0)


def _risk_specificity_score(risk_case: Dict[str, Any]) -> float:
    failure_modes = [str(x or "") for x in list(risk_case.get("main_failure_modes", []) or [])]
    kill_criteria = [str(x or "") for x in list(risk_case.get("kill_criteria", []) or [])]
    why_acceptable = [str(x or "") for x in list(risk_case.get("why_risks_acceptable", []) or [])]
    score = 0.0
    if failure_modes:
        score += 0.3
    if any("Adverse tail" in x or bool(re.search(r"\(\+|-|\d", x)) for x in failure_modes):
        score += 0.3
    if len(kill_criteria) >= 2:
        score += 0.2
    if why_acceptable:
        score += 0.2
    return min(score, 1.0)


def _sizing_specificity_score(sizing_guidance: Dict[str, Any]) -> float:
    if not sizing_guidance:
        return 0.0
    score = 0.0
    if sizing_guidance.get("recommended_range"):
        score += 0.3
    if list(sizing_guidance.get("rationale", []) or []):
        score += 0.3
    if sizing_guidance.get("why_not_larger"):
        score += 0.2
    if sizing_guidance.get("why_not_smaller"):
        score += 0.2
    return min(score, 1.0)


def _aggregate_cases(cases: Sequence[Dict[str, Any]], missing_artifacts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not cases:
        return {
            "heuristic_overall_mean": 0.0,
            "completeness_rate": 0.0,
            "humanized_language_rate": 0.0,
            "specific_timing_rate": 0.0,
            "alternatives_present_rate": 0.0,
            "risk_specificity_rate": 0.0,
            "status_quo_comparison_rate": 0.0,
            "sizing_specificity_rate": 0.0,
            "parameter_optimization_rate": 0.0,
            "regret_analysis_rate": 0.0,
            "scenario_sizing_rate": 0.0,
            "rating_analysis_rate": 0.0,
            "signaling_analysis_rate": 0.0,
            "expected_posture_coverage_rate": None,
            "posture_match_rate": None,
            "negative_case_accuracy": None,
            "missing_artifact_rate": 1.0 if missing_artifacts else 0.0,
            "flag_counts": {},
        }
    flag_counts = Counter()
    for case in cases:
        for flag in list((case.get("heuristic", {}) or {}).get("flags", []) or []):
            flag_counts[str(flag)] += 1
    count = float(len(cases))
    posture_cases = [case for case in cases if case.get("expected_posture")]
    negative_cases = [case for case in posture_cases if case.get("expected_posture") == "wait"]
    return {
        "heuristic_overall_mean": round(sum(float((case.get("heuristic", {}) or {}).get("overall_score", 0.0) or 0.0) for case in cases) / count, 6),
        "completeness_rate": round(sum(1.0 for case in cases if float((case.get("heuristic", {}) or {}).get("completeness_score", 0.0) or 0.0) >= 1.0) / count, 6),
        "humanized_language_rate": round(sum(1.0 for case in cases if float((case.get("heuristic", {}) or {}).get("humanized_score", 0.0) or 0.0) >= 1.0) / count, 6),
        "specific_timing_rate": round(sum(1.0 for case in cases if float((case.get("heuristic", {}) or {}).get("timing_score", 0.0) or 0.0) >= 0.6) / count, 6),
        "alternatives_present_rate": round(sum(1.0 for case in cases if float((case.get("heuristic", {}) or {}).get("alternatives_score", 0.0) or 0.0) >= 0.5) / count, 6),
        "risk_specificity_rate": round(sum(1.0 for case in cases if float((case.get("heuristic", {}) or {}).get("risk_score", 0.0) or 0.0) >= 0.6) / count, 6),
        "status_quo_comparison_rate": round(sum(1.0 for case in cases if float((case.get("heuristic", {}) or {}).get("status_quo_score", 0.0) or 0.0) >= 1.0) / count, 6),
        "sizing_specificity_rate": round(sum(1.0 for case in cases if float((case.get("heuristic", {}) or {}).get("sizing_score", 0.0) or 0.0) >= 0.6) / count, 6),
        "parameter_optimization_rate": round(sum(1.0 for case in cases if float((case.get("heuristic", {}) or {}).get("parameter_optimization_score", 0.0) or 0.0) >= 1.0) / count, 6),
        "regret_analysis_rate": round(sum(1.0 for case in cases if float((case.get("heuristic", {}) or {}).get("regret_score", 0.0) or 0.0) >= 1.0) / count, 6),
        "scenario_sizing_rate": round(sum(1.0 for case in cases if float((case.get("heuristic", {}) or {}).get("scenario_sizing_score", 0.0) or 0.0) >= 1.0) / count, 6),
        "rating_analysis_rate": round(sum(1.0 for case in cases if float((case.get("heuristic", {}) or {}).get("rating_analysis_score", 0.0) or 0.0) >= 1.0) / count, 6),
        "signaling_analysis_rate": round(sum(1.0 for case in cases if float((case.get("heuristic", {}) or {}).get("signaling_score", 0.0) or 0.0) >= 1.0) / count, 6),
        "expected_posture_coverage_rate": round(len(posture_cases) / count, 6) if posture_cases else None,
        "posture_match_rate": round(sum(1.0 for case in posture_cases if case.get("posture_match")) / len(posture_cases), 6) if posture_cases else None,
        "negative_case_accuracy": round(sum(1.0 for case in negative_cases if case.get("posture_match")) / len(negative_cases), 6) if negative_cases else None,
        "missing_artifact_rate": round(len(missing_artifacts) / (len(cases) + len(missing_artifacts)), 6) if (cases or missing_artifacts) else 0.0,
        "flag_counts": dict(flag_counts),
    }


def _select_review_queue(cases: Sequence[Dict[str, Any]], review_count: int) -> List[Dict[str, Any]]:
    ranked = sorted(
        cases,
        key=lambda case: (
            -len(list((case.get("heuristic", {}) or {}).get("flags", []) or [])),
            float((case.get("heuristic", {}) or {}).get("overall_score", 0.0) or 0.0),
            str(case.get("company_id", "")),
        ),
    )
    return ranked[: max(0, int(review_count))]


def _resolve_expected_posture(
    *,
    expected_postures: Dict[str, Any],
    run_id: str,
    company_id: str,
) -> Optional[str]:
    if not expected_postures:
        return None
    for key in (run_id, company_id):
        if key not in expected_postures:
            continue
        value = expected_postures[key]
        if isinstance(value, dict):
            text = str(value.get("expected_posture", "") or "").strip()
            return text or None
        text = str(value or "").strip()
        return text or None
    return None
