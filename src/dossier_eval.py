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


