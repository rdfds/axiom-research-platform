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


