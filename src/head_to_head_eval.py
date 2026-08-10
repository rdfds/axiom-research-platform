from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .action_ontology import build_default_action_schema_registry
from .board_ready_dossier import build_board_ready_dossier
from .planner_brain import build_plan_set
from .recommendation_run import RecommendationRun


_SECTION_RE = re.compile(r"^\s{0,3}#{1,3}\s+(.+?)\s*$", re.MULTILINE)
_RAW_ACTION_ID_RE = re.compile(r"\b[a-z_]+\.[a-z0-9_]+\b")
_NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?(?:%|x|m|mm|b|bn)?\b", re.IGNORECASE)
_GENERIC_ALT_PHRASES = {
    "it addresses the problem less directly",
    "lower expected utility",
    "weaker empirical support",
    "higher tail risk",
    "value arrives later",
}
_SNAPSHOT_METRIC_PATTERNS = (
    "net leverage",
    "maturity wall",
    "liquidity",
    "market value",
    "market cap",
    "credit window",
    "equity window",
    "revenue growth",
    "fcf conversion",
)
_ACTION_HINTS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:open[- ]market )?(?:buyback|repurchase)s?\b", re.IGNORECASE), "capital_return.open_market_buyback"),
    (re.compile(r"\btender offer buyback\b", re.IGNORECASE), "capital_return.tender_offer_buyback"),
    (re.compile(r"\bspecial dividend\b", re.IGNORECASE), "capital_return.special_dividend"),
    (re.compile(r"\bdividend increase\b", re.IGNORECASE), "capital_return.dividend_increase"),
    (re.compile(r"\bdividend cut\b", re.IGNORECASE), "capital_return.dividend_cut"),
    (re.compile(r"\bdividend initiat(?:e|ion)\b", re.IGNORECASE), "capital_return.dividend_initiate"),
    (re.compile(r"\brefinanc(?:e|ing)\b|\brefi\b", re.IGNORECASE), "capital_structure.refinancing"),
    (re.compile(r"\b(?:new )?debt issuance\b|\bbond issuance\b|\bissue debt\b", re.IGNORECASE), "capital_structure.new_debt_issuance"),
    (re.compile(r"\bequity issuance\b|\bshare issuance\b|\bissue equity\b", re.IGNORECASE), "capital_structure.equity_issuance"),
    (re.compile(r"\brevolver\b|\bcredit line\b", re.IGNORECASE), "capital_structure.revolver_draw_or_resize"),
    (re.compile(r"\btuck[- ]in acquisition\b", re.IGNORECASE), "mna.tuck_in_acquisition"),
    (re.compile(r"\bplatform acquisition\b", re.IGNORECASE), "mna.platform_acquisition"),
    (re.compile(r"\bgo[- ]private\b|\blbo\b|\bleveraged buyout\b", re.IGNORECASE), "mna.go_private_lbo"),
    (re.compile(r"\bacquisition\b|\bm&a\b", re.IGNORECASE), "mna.platform_acquisition"),
    (re.compile(r"\bdivestiture\b|\basset sale\b|\bsell (?:a )?business\b", re.IGNORECASE), "portfolio.divestiture_partial"),
)


@dataclass(frozen=True)
class CanonicalPacket:
    packet_id: str
    company_id: str
    as_of_time: str
    source_type: str
    source_label: str
    primary_recommendation: str
    action_path: List[str]
    problem_statement: str
    recommendation_thesis: str
    why_now: str
    alternatives: List[str]
    risks: List[str]
    kill_criteria: List[str]
    evidence_points: List[str]
    confidence_posture: str
    baseline_type: str
    task_match: str
    raw_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "company_id": self.company_id,
            "as_of_time": self.as_of_time,
            "source_type": self.source_type,
            "source_label": self.source_label,
            "primary_recommendation": self.primary_recommendation,
            "action_path": self.action_path,
            "problem_statement": self.problem_statement,
            "recommendation_thesis": self.recommendation_thesis,
            "why_now": self.why_now,
            "alternatives": self.alternatives,
            "risks": self.risks,
            "kill_criteria": self.kill_criteria,
            "evidence_points": self.evidence_points,
            "confidence_posture": self.confidence_posture,
            "baseline_type": self.baseline_type,
            "task_match": self.task_match,
            "raw_text": self.raw_text,
        }


def build_head_to_head_report(
    *,
    runs_roots: Sequence[str | Path],
    snapshot_root: str | Path,
    baseline_dir: str | Path,
    realized_outcomes_path: Optional[str | Path] = None,
    alignment_horizon_days: int = 540,
    run_ids: Optional[Sequence[str]] = None,
    review_count: int = 50,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    resolved_roots = [Path(root) for root in runs_roots]
    snapshot_root_path = Path(snapshot_root)
    baseline_dir_path = Path(baseline_dir)
    registry = build_default_action_schema_registry()
    outcomes_lookup = _load_realized_outcomes_lookup(Path(realized_outcomes_path)) if realized_outcomes_path else None

    selected = _resolve_run_ids(runs_roots=resolved_roots, run_ids=run_ids, limit=limit)
    cases: List[Dict[str, Any]] = []
    missing_inputs: List[Dict[str, Any]] = []

    for run_id, runs_root in selected:
        try:
            cases.append(
                _build_case(
                    run_id=run_id,
                    runs_root=runs_root,
                    snapshot_root=snapshot_root_path,
                    baseline_dir=baseline_dir_path,
                    registry=registry,
                    outcomes_lookup=outcomes_lookup,
                    alignment_horizon_days=alignment_horizon_days,
                )
            )
        except FileNotFoundError as exc:
            missing_inputs.append(
                {
                    "run_id": run_id,
                    "runs_root": str(runs_root),
                    "error": str(exc),
                }
            )

    aggregate = _aggregate_cases(cases=cases, missing_inputs=missing_inputs)
    review_queue = _select_review_queue(cases=cases, review_count=review_count)
    return {
        "ok": True,
        "runs_analyzed": len(cases),
        "missing_inputs": missing_inputs,
        "aggregate": aggregate,
        "review_queue": review_queue,
        "cases": cases,
    }


def render_head_to_head_markdown(report: Dict[str, Any]) -> str:
    aggregate = dict(report.get("aggregate", {}) or {})
    lines: List[str] = []
    lines.append("# Head-To-Head Benchmark Report")
    lines.append("")
    lines.append(f"- Runs analyzed: `{report.get('runs_analyzed', 0)}`")
    lines.append(f"- Missing inputs: `{len(report.get('missing_inputs', []) or [])}`")
    lines.append(f"- Model mean score: `{aggregate.get('model_mean_score', 0.0):.3f}`")
    lines.append(f"- Baseline mean score: `{aggregate.get('baseline_mean_score', 0.0):.3f}`")
    lines.append(f"- Model win rate: `{aggregate.get('model_win_rate', 0.0):.3f}`")
    lines.append(f"- Baseline win rate: `{aggregate.get('baseline_win_rate', 0.0):.3f}`")
    lines.append(f"- Tie rate: `{aggregate.get('tie_rate', 0.0):.3f}`")
    lines.append(f"- Sign-test p-value: `{_fmt_optional(aggregate.get('sign_test_p_value'))}`")
    lines.append(f"- Model win-rate 95% CI: `{_fmt_interval(aggregate.get('model_win_rate_ci_95'))}`")
    lines.append("")

    component_deltas = dict(aggregate.get("component_delta_means", {}) or {})
    if component_deltas:
        lines.append("## Mean Component Deltas (Model - Baseline)")
        lines.append("")
        for key, value in sorted(component_deltas.items()):
            lines.append(f"- `{key}`: `{value:+.3f}`")
        lines.append("")

    by_task_match = dict(aggregate.get("by_task_match", {}) or {})
    if by_task_match:
        lines.append("## By Task Match")
        lines.append("")
        for label, bucket in sorted(by_task_match.items()):
            lines.append(f"### `{label}`")
            lines.append(f"- Cases: `{bucket.get('case_count', 0)}`")
            lines.append(f"- Model win rate: `{_fmt_optional(bucket.get('model_win_rate'))}`")
            lines.append(f"- Baseline win rate: `{_fmt_optional(bucket.get('baseline_win_rate'))}`")
            lines.append(f"- Mean score delta: `{_fmt_optional(bucket.get('mean_score_delta'))}`")
            lines.append(f"- Sign-test p-value: `{_fmt_optional(bucket.get('sign_test_p_value'))}`")
            lines.append("")

    ex_post = dict(aggregate.get("ex_post", {}) or {})
    if ex_post:
        lines.append("## Ex-Post Alignment")
        lines.append("")
        lines.append(f"- Coverage: `{ex_post.get('coverage_rate', 0.0):.3f}`")
        lines.append(f"- Model mean alignment: `{_fmt_optional(ex_post.get('model_mean_score'))}`")
        lines.append(f"- Baseline mean alignment: `{_fmt_optional(ex_post.get('baseline_mean_score'))}`")
        lines.append(f"- Model ex-post win rate: `{_fmt_optional(ex_post.get('model_win_rate'))}`")
        lines.append(f"- Baseline ex-post win rate: `{_fmt_optional(ex_post.get('baseline_win_rate'))}`")
        lines.append(f"- Ex-post sign-test p-value: `{_fmt_optional(ex_post.get('sign_test_p_value'))}`")
        lines.append("")

    lines.append("## Review Queue")
    lines.append("")
    for idx, case in enumerate(report.get("review_queue", []) or [], start=1):
        lines.extend(_render_case_markdown(case=case, index=idx))
    return "\n".join(lines).strip() + "\n"


def _render_case_markdown(case: Dict[str, Any], index: int) -> List[str]:
    comparison = dict(case.get("comparison", {}) or {})
    model = dict(case.get("model_packet", {}) or {})
    baseline = dict(case.get("baseline_packet", {}) or {})
    blinded = dict(case.get("blinded_review", {}) or {})
    lines: List[str] = []
    lines.append(f"### {index}. `{case.get('company_id')}`")
    lines.append("")
    lines.append(f"- Run: `{case.get('run_id')}`")
    lines.append(f"- Winner: `{comparison.get('winner')}`")
    lines.append(f"- Model score: `{comparison.get('model_score', 0.0):.3f}`")
    lines.append(f"- Baseline score: `{comparison.get('baseline_score', 0.0):.3f}`")
    lines.append(f"- Model recommendation: `{model.get('primary_recommendation')}`")
    lines.append(f"- Baseline recommendation: `{baseline.get('primary_recommendation')}`")
    lines.append(f"- Baseline type: `{baseline.get('baseline_type') or 'unknown'}`")
    lines.append(f"- Task match: `{baseline.get('task_match') or 'unknown'}`")
    lines.append(f"- Blinded order: `{blinded.get('order')}`")
    lines.append(f"- Model thesis: {model.get('recommendation_thesis') or 'missing'}")
    lines.append(f"- Baseline thesis: {baseline.get('recommendation_thesis') or 'missing'}")
    lines.append("")
    lines.append("- [ ] Which packet has the better idea?")
    lines.append("- [ ] Which packet has the better why-now logic?")
    lines.append("- [ ] Which packet handles alternatives and risks better?")
    lines.append("- [ ] Which packet would you show to a CEO?")
    lines.append("")
    return lines


