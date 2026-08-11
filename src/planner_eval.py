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
    support = dict(case['top_plan_support'] or {})
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


