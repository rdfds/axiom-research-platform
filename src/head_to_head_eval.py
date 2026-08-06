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


def _build_case(
    *,
    run_id: str,
    runs_root: Path,
    snapshot_root: Path,
    baseline_dir: Path,
    registry: Any,
    outcomes_lookup: Optional[Dict[str, List[Tuple[Any, str, str]]]],
    alignment_horizon_days: int,
) -> Dict[str, Any]:
    run_payload = json.loads((runs_root / "runs" / f"run_id={run_id}.json").read_text())
    recommendation_run = RecommendationRun.from_dict(run_payload)
    company_id = str(recommendation_run.company_id)
    as_of_time = str(recommendation_run.as_of_time)
    snapshot = _load_snapshot(snapshot_root=snapshot_root, company_id=company_id, as_of_time=as_of_time)
    model_packet = build_model_packet(
        run_id=run_id,
        runs_root=runs_root,
        snapshot=snapshot,
        registry=registry,
        recommendation_run=recommendation_run,
    )
    baseline_packet = load_baseline_packet(
        company_id=company_id,
        as_of_time=as_of_time,
        baseline_dir=baseline_dir,
    )
    comparison = compare_packets(model_packet=model_packet, baseline_packet=baseline_packet, snapshot=snapshot)
    blinded = _build_blinded_review(model_packet=model_packet, baseline_packet=baseline_packet, run_id=run_id)
    ex_post = _build_ex_post_comparison(
        company_id=company_id,
        as_of_time=as_of_time,
        model_packet=model_packet,
        baseline_packet=baseline_packet,
        outcomes_lookup=outcomes_lookup,
        alignment_horizon_days=alignment_horizon_days,
    )
    return {
        "run_id": run_id,
        "company_id": company_id,
        "comparison": comparison,
        "model_packet": model_packet.to_dict(),
        "baseline_packet": baseline_packet.to_dict(),
        "blinded_review": blinded,
        "ex_post": ex_post,
    }


def build_model_packet(
    *,
    run_id: str,
    runs_root: Path,
    snapshot: Dict[str, Any],
    registry: Any,
    recommendation_run: RecommendationRun,
) -> CanonicalPacket:
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
    dossier = build_board_ready_dossier(
        run=recommendation_run,
        snapshot=snapshot,
        plan_set=plan_set,
        feasible_candidates=feasible_candidates,
        precedent_matches=list(precedent.get("results", []) or []),
        registry=registry,
    )
    thesis = dict(dossier.get("recommendation_thesis", {}) or {})
    action_path = [str(step.get("action_id", "") or "") for step in list(((plan_set.get("plans", []) or [{}])[0].get("steps", []) or []))]
    alternatives = [
        str(item.get("why_not_preferred", "") or "")
        for item in list(dossier.get("alternative_analysis", []) or [])
        if str(item.get("why_not_preferred", "") or "").strip()
    ]
    risk_case = dict(dossier.get("risk_case", {}) or {})
    evidence_points = [
        str(item.get("text", "") or "")
        for item in list(dossier.get("supporting_evidence", []) or [])
        if str(item.get("text", "") or "").strip()
    ]
    raw_text_parts = [
        str(dossier.get("executive_summary", "") or ""),
        str(thesis.get("problem_statement", "") or ""),
        str(thesis.get("why_this_plan", "") or ""),
        str(thesis.get("why_now", "") or ""),
        *alternatives,
        *[str(x or "") for x in list(risk_case.get("main_failure_modes", []) or [])],
        *[str(x or "") for x in list(risk_case.get("kill_criteria", []) or [])],
        *evidence_points,
    ]
    return CanonicalPacket(
        packet_id=f"{run_id}:model",
        company_id=str(recommendation_run.company_id),
        as_of_time=str(recommendation_run.as_of_time),
        source_type="model",
        source_label="board_ready_dossier",
        primary_recommendation=_humanize_action(action_path[0] if action_path else ""),
        action_path=action_path,
        problem_statement=str(thesis.get("problem_statement", "") or ""),
        recommendation_thesis=str(thesis.get("why_this_plan", "") or ""),
        why_now=str(thesis.get("why_now", "") or ""),
        alternatives=alternatives,
        risks=[str(x or "") for x in list(risk_case.get("main_failure_modes", []) or [])],
        kill_criteria=[str(x or "") for x in list(risk_case.get("kill_criteria", []) or [])],
        evidence_points=evidence_points,
        confidence_posture=str(dossier.get("confidence_posture", "") or ""),
        baseline_type="model",
        task_match="direct",
        raw_text="\n".join(part for part in raw_text_parts if part),
    )


def load_baseline_packet(*, company_id: str, as_of_time: str, baseline_dir: Path) -> CanonicalPacket:
    candidates = [
        baseline_dir / f"company_id={company_id}.json",
        baseline_dir / f"company_id={company_id}.md",
        baseline_dir / f"{company_id}.json",
        baseline_dir / f"{company_id}.md",
    ]
    for path in candidates:
        if path.exists():
            if path.suffix.lower() == ".json":
                return _load_baseline_json(path=path, company_id=company_id, as_of_time=as_of_time)
            return _load_baseline_markdown(path=path, company_id=company_id, as_of_time=as_of_time)
    raise FileNotFoundError(f"baseline memo not found for company_id={company_id} under {baseline_dir}")


def _load_baseline_json(*, path: Path, company_id: str, as_of_time: str) -> CanonicalPacket:
    payload = json.loads(path.read_text())
    action_path = [str(x or "") for x in list(payload.get("action_path", []) or [])]
    alternatives = [str(x or "") for x in list(payload.get("alternatives", []) or [])]
    risks = [str(x or "") for x in list(payload.get("risks", []) or [])]
    kill_criteria = [str(x or "") for x in list(payload.get("kill_criteria", []) or [])]
    evidence_points = [str(x or "") for x in list(payload.get("evidence_points", []) or [])]
    raw_text = payload.get("raw_text") or "\n".join(
        [
            str(payload.get("problem_statement", "") or ""),
            str(payload.get("recommendation_thesis", "") or ""),
            str(payload.get("why_now", "") or ""),
            *alternatives,
            *risks,
            *kill_criteria,
            *evidence_points,
        ]
    )
    if not action_path:
        action_path = _infer_action_path(
            " ".join(
                [
                    str(payload.get("primary_recommendation", "") or ""),
                    str(payload.get("recommendation_thesis", "") or ""),
                ]
            )
        )
    baseline_type = str(payload.get("baseline_type", "") or "").strip() or _infer_baseline_type(str(payload))
    task_match = str(payload.get("task_match", "") or "").strip() or _infer_task_match(baseline_type=baseline_type)
    return CanonicalPacket(
        packet_id=f"{company_id}:baseline:{path.name}",
        company_id=company_id,
        as_of_time=as_of_time,
        source_type="baseline",
        source_label=str(payload.get("source_label", path.name)),
        primary_recommendation=str(payload.get("primary_recommendation", "") or _humanize_action(action_path[0] if action_path else "")),
        action_path=action_path,
        problem_statement=str(payload.get("problem_statement", "") or ""),
        recommendation_thesis=str(payload.get("recommendation_thesis", "") or ""),
        why_now=str(payload.get("why_now", "") or ""),
        alternatives=alternatives,
        risks=risks,
        kill_criteria=kill_criteria,
        evidence_points=evidence_points,
        confidence_posture=str(payload.get("confidence_posture", "") or ""),
        baseline_type=baseline_type,
        task_match=task_match,
        raw_text=str(raw_text or ""),
    )


def _load_baseline_markdown(*, path: Path, company_id: str, as_of_time: str) -> CanonicalPacket:
    original_text = path.read_text()
    metadata, text = _parse_markdown_preamble_metadata(original_text)
    sections = _parse_markdown_sections(text)
    recommendation_text = sections.get("recommendation", "") or sections.get("recommendation thesis", "")
    action_path = _extract_action_path(recommendation_text) or _infer_action_path(recommendation_text)
    alternatives = _split_bullets(sections.get("alternatives", ""))
    risks = _split_bullets(sections.get("risks", ""))
    kill_criteria = _split_bullets(sections.get("kill criteria", "")) or _split_bullets(sections.get("decision boundaries", ""))
    evidence_points = _split_bullets(sections.get("evidence", ""))
    primary_recommendation = _extract_primary_recommendation(sections.get("recommendation", "") or sections.get("recommendation thesis", ""))
    baseline_type = str(metadata.get("baseline_type", "") or "").strip() or _infer_baseline_type(original_text)
    task_match = str(metadata.get("task_match", "") or "").strip() or _infer_task_match(baseline_type=baseline_type)
    return CanonicalPacket(
        packet_id=f"{company_id}:baseline:{path.name}",
        company_id=company_id,
        as_of_time=as_of_time,
        source_type="baseline",
        source_label=path.name,
        primary_recommendation=primary_recommendation,
        action_path=action_path,
        problem_statement=sections.get("problem", "") or sections.get("problem statement", ""),
        recommendation_thesis=sections.get("recommendation", "") or sections.get("recommendation thesis", ""),
        why_now=sections.get("why now", ""),
        alternatives=alternatives,
        risks=risks,
        kill_criteria=kill_criteria,
        evidence_points=evidence_points,
        confidence_posture="",
        baseline_type=baseline_type,
        task_match=task_match,
        raw_text=text,
    )


def compare_packets(*, model_packet: CanonicalPacket, baseline_packet: CanonicalPacket, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    model_scores = _score_packet(packet=model_packet, snapshot=snapshot)
    baseline_scores = _score_packet(packet=baseline_packet, snapshot=snapshot)
    model_overall = float(model_scores.get("overall_score", 0.0) or 0.0)
    baseline_overall = float(baseline_scores.get("overall_score", 0.0) or 0.0)
    delta = round(model_overall - baseline_overall, 6)
    if delta > 0.05:
        winner = "model"
    elif delta < -0.05:
        winner = "baseline"
    else:
        winner = "tie"
    component_deltas = {
        key: round(float(model_scores.get(key, 0.0) or 0.0) - float(baseline_scores.get(key, 0.0) or 0.0), 6)
        for key in ["completeness_score", "grounding_score", "timing_score", "alternatives_score", "risk_score", "language_score"]
    }
    return {
        "winner": winner,
        "model_score": model_overall,
        "baseline_score": baseline_overall,
        "score_delta": delta,
        "component_deltas": component_deltas,
        "model_scores": model_scores,
        "baseline_scores": baseline_scores,
    }


def export_blinded_packets(
    *,
    report: Dict[str, Any],
    packets_out_dir: str | Path,
    answer_key_out: Optional[str | Path] = None,
) -> Dict[str, Any]:
    out_dir = Path(packets_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    answer_key: Dict[str, Dict[str, Any]] = {}
    exported = 0
    for case in list(report.get("cases", []) or []):
        run_id = str(case.get("run_id", "") or "")
        company_id = str(case.get("company_id", "") or "")
        blinded = dict(case.get("blinded_review", {}) or {})
        if not run_id or not blinded:
            continue
        payload = {
            "run_id": run_id,
            "company_id": company_id,
            "judge_prompt": blinded.get("judge_prompt", ""),
            "packet_A": blinded.get("packet_A", {}),
            "packet_B": blinded.get("packet_B", {}),
        }
        (out_dir / f"run_id={run_id}.json").write_text(json.dumps(payload, indent=2))
        answer_key[run_id] = {
            "company_id": company_id,
            "order": blinded.get("order"),
            "comparison": case.get("comparison", {}),
            "ex_post": case.get("ex_post", {}),
        }
        exported += 1
    if answer_key_out:
        Path(answer_key_out).write_text(json.dumps(answer_key, indent=2))
    return {"exported_packets": exported, "packets_out_dir": str(out_dir), "answer_key_out": str(answer_key_out) if answer_key_out else None}


def _score_packet(*, packet: CanonicalPacket, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    completeness_score = 1.0 if all(
        [
            packet.problem_statement,
            packet.recommendation_thesis,
            packet.why_now,
            packet.alternatives,
            packet.risks,
            packet.kill_criteria,
        ]
    ) else 0.0
    grounding_score = _grounding_score(packet=packet, snapshot=snapshot)
    timing_score = _timing_score(packet.why_now)
    alternatives_score = _alternatives_score(packet.alternatives)
    risk_score = _risk_score(packet.risks, packet.kill_criteria)
    language_score = 0.0 if _RAW_ACTION_ID_RE.search(packet.raw_text) else 1.0
    overall_score = round(
        (
            completeness_score
            + grounding_score
            + timing_score
            + alternatives_score
            + risk_score
            + language_score
        )
        / 6.0,
        6,
    )
    return {
        "overall_score": overall_score,
        "completeness_score": completeness_score,
        "grounding_score": grounding_score,
        "timing_score": timing_score,
        "alternatives_score": alternatives_score,
        "risk_score": risk_score,
        "language_score": language_score,
    }


def _grounding_score(*, packet: CanonicalPacket, snapshot: Dict[str, Any]) -> float:
    text = "\n".join([packet.problem_statement, packet.recommendation_thesis, packet.why_now, *packet.evidence_points])
    metric_hits = sum(1 for token in _SNAPSHOT_METRIC_PATTERNS if token in text.lower())
    numeric_hits = len(_NUMERIC_RE.findall(text))
    evidence_count = len(packet.evidence_points)
    score = 0.0
    if metric_hits >= 2:
        score += 0.4
    if numeric_hits >= 3:
        score += 0.3
    if evidence_count >= 3:
        score += 0.3
    return min(score, 1.0)


def _timing_score(text: str) -> float:
    lower = str(text or "").lower()
    score = 0.0
    if any(token in lower for token in ["window", "waiting", "lead time", "urgent", "supportive now", "front-of-plan", "after "]):
        score += 0.5
    if _NUMERIC_RE.search(text or ""):
        score += 0.25
    if any(token in lower for token in ["credit", "equity", "liquidity", "maturity", "market value"]):
        score += 0.25
    return min(score, 1.0)


def _alternatives_score(alternatives: Sequence[str]) -> float:
    if not alternatives:
        return 0.0
    specific = 0
    for item in alternatives:
        lower = str(item or "").lower()
        if any(phrase in lower for phrase in _GENERIC_ALT_PHRASES):
            continue
        if len(lower.split()) >= 6:
            specific += 1
    if specific >= 2:
        return 1.0
    if specific == 1:
        return 0.75
    return 0.5


def _risk_score(risks: Sequence[str], kill_criteria: Sequence[str]) -> float:
    score = 0.0
    if risks:
        score += 0.4
    if any(_NUMERIC_RE.search(item or "") or "tail" in str(item or "").lower() for item in risks):
        score += 0.3
    if len(list(kill_criteria or [])) >= 2:
        score += 0.3
    return min(score, 1.0)


def _build_blinded_review(*, model_packet: CanonicalPacket, baseline_packet: CanonicalPacket, run_id: str) -> Dict[str, Any]:
    rng = random.Random(str(run_id))
    order = ["A", "B"]
    rng.shuffle(order)
    mapping = {order[0]: model_packet.to_dict(), order[1]: baseline_packet.to_dict()}
    if mapping["A"]["source_type"] == mapping["B"]["source_type"]:
        mapping["A"] = model_packet.to_dict()
        mapping["B"] = baseline_packet.to_dict()
        order = ["A=model", "B=baseline"]
    else:
        order = [f"A={mapping['A']['source_type']}", f"B={mapping['B']['source_type']}"]
    return {
        "order": order,
        "packet_A": _blind_packet(mapping["A"]),
        "packet_B": _blind_packet(mapping["B"]),
        "judge_prompt": (
            "Compare Packet A and Packet B. Score which has the better problem diagnosis, recommendation, why-now logic, "
            "alternative analysis, and risk framing. Do not infer the source."
        ),
    }


def _blind_packet(packet: Dict[str, Any]) -> Dict[str, Any]:
    blinded = dict(packet)
    blinded.pop("source_type", None)
    blinded.pop("source_label", None)
    blinded.pop("confidence_posture", None)
    blinded.pop("baseline_type", None)
    blinded.pop("task_match", None)
    return blinded


def _aggregate_cases(cases: Sequence[Dict[str, Any]], missing_inputs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not cases:
        return {
            "model_mean_score": 0.0,
            "baseline_mean_score": 0.0,
            "model_win_rate": 0.0,
            "baseline_win_rate": 0.0,
            "tie_rate": 0.0,
            "mean_score_delta": 0.0,
            "median_score_delta": 0.0,
            "comparable_case_count": 0,
            "sign_test_p_value": None,
            "model_win_rate_ci_95": None,
            "component_delta_means": {},
            "by_task_match": {},
            "by_baseline_type": {},
            "ex_post": {},
            "missing_input_rate": 1.0 if missing_inputs else 0.0,
        }
    count = float(len(cases))
    model_wins = sum(1 for case in cases if str((case.get("comparison", {}) or {}).get("winner", "")) == "model")
    baseline_wins = sum(1 for case in cases if str((case.get("comparison", {}) or {}).get("winner", "")) == "baseline")
    ties = sum(1 for case in cases if str((case.get("comparison", {}) or {}).get("winner", "")) == "tie")
    score_deltas = [float((case.get("comparison", {}) or {}).get("score_delta", 0.0) or 0.0) for case in cases]
    component_keys = ["completeness_score", "grounding_score", "timing_score", "alternatives_score", "risk_score", "language_score"]
    component_delta_means = {
        key: round(
            sum(float(((case.get("comparison", {}) or {}).get("component_deltas", {}) or {}).get(key, 0.0) or 0.0) for case in cases) / count,
            6,
        )
        for key in component_keys
    }
    significance = _compute_significance(model_wins=model_wins, baseline_wins=baseline_wins)
    return {
        "model_mean_score": round(sum(float((case.get("comparison", {}) or {}).get("model_score", 0.0) or 0.0) for case in cases) / count, 6),
        "baseline_mean_score": round(sum(float((case.get("comparison", {}) or {}).get("baseline_score", 0.0) or 0.0) for case in cases) / count, 6),
        "model_win_rate": round(model_wins / count, 6),
        "baseline_win_rate": round(baseline_wins / count, 6),
        "tie_rate": round(ties / count, 6),
        "mean_score_delta": round(sum(score_deltas) / count, 6),
        "median_score_delta": round(_median(score_deltas), 6),
        "comparable_case_count": significance["comparable_case_count"],
        "sign_test_p_value": significance["sign_test_p_value"],
        "model_win_rate_ci_95": significance["model_win_rate_ci_95"],
        "component_delta_means": component_delta_means,
        "by_task_match": _aggregate_cases_by_bucket(cases=cases, field="task_match"),
        "by_baseline_type": _aggregate_cases_by_bucket(cases=cases, field="baseline_type"),
        "ex_post": _aggregate_ex_post(cases),
        "missing_input_rate": round(len(missing_inputs) / (len(cases) + len(missing_inputs)), 6) if (cases or missing_inputs) else 0.0,
    }


def _select_review_queue(cases: Sequence[Dict[str, Any]], review_count: int) -> List[Dict[str, Any]]:
    ranked = sorted(
        cases,
        key=lambda case: (
            float((case.get("comparison", {}) or {}).get("score_delta", 0.0) or 0.0),
            abs(float((case.get("comparison", {}) or {}).get("score_delta", 0.0) or 0.0)),
            str(case.get("company_id", "")),
        ),
    )
    return ranked[: max(0, int(review_count))]


def _parse_markdown_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(text or ""))
    if not matches:
        return {"recommendation": text.strip()}
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        key = match.group(1).strip().lower()
        sections[key] = text[start:end].strip()
    return sections


def _parse_markdown_preamble_metadata(text: str) -> Tuple[Dict[str, str], str]:
    lines = str(text or "").splitlines()
    metadata: Dict[str, str] = {}
    body_start = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            body_start = idx
            break
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            metadata[key.strip().lower().replace("-", "_").replace(" ", "_")] = value.strip()
            body_start = idx + 1
            continue
        body_start = idx
        break
    return metadata, "\n".join(lines[body_start:]).lstrip()


def _split_bullets(text: str) -> List[str]:
    out: List[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("- ", "* ", "+ ")):
            stripped = stripped[2:].strip()
        out.append(stripped)
    return out


def _extract_action_path(text: str) -> List[str]:
    return _RAW_ACTION_ID_RE.findall(str(text or ""))


def _extract_primary_recommendation(text: str) -> str:
    action_path = _extract_action_path(text)
    if action_path:
        return _humanize_action(action_path[0])
    inferred = _infer_action_path(text)
    if inferred:
        return _humanize_action(inferred[0])
    stripped = " ".join(str(text or "").split())
    return stripped[:120]


def _infer_action_path(text: str) -> List[str]:
    out: List[str] = []
    haystack = str(text or "")
    for pattern, action_id in _ACTION_HINTS:
        if pattern.search(haystack):
            out.append(action_id)
    seen: set[str] = set()
    ordered: List[str] = []
    for action_id in out:
        if action_id not in seen:
            ordered.append(action_id)
            seen.add(action_id)
    return ordered


def _infer_baseline_type(text: str) -> str:
    lower = str(text or "").lower()
    if any(token in lower for token in ["activist", "engine no. 1", "elliott", "starboard", "third point", "jana", "mantle ridge"]):
        return "activist"
    if "investor letter" in lower or "shareholder letter" in lower or "fund letter" in lower:
        return "investor_letter"
    if any(token in lower for token in ["moody", "s&p", "fitch", "rating rationale", "credit opinion"]):
        return "rating_note"
    if any(token in lower for token in ["joint statement", "proxy", "sec filing", "def 14a", "dfan14a"]):
        return "investor_campaign"
    return "management_ir"


def _infer_task_match(*, baseline_type: str) -> str:
    normalized = str(baseline_type or "").lower()
    if normalized in {"activist", "investor_letter", "investor_campaign"}:
        return "direct"
    if normalized in {"rating_note", "transaction_rationale"}:
        return "partial"
    return "weak"


def _resolve_run_ids(
    *,
    runs_roots: Sequence[Path],
    run_ids: Optional[Sequence[str]],
    limit: Optional[int],
) -> List[Tuple[str, Path]]:
    by_id: List[Tuple[str, Path]] = []
    explicit = list(run_ids or [])
    if explicit:
        for run_id in explicit:
            for runs_root in runs_roots:
                if (runs_root / "runs" / f"run_id={run_id}.json").exists():
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


def _load_snapshot(*, snapshot_root: Path, company_id: str, as_of_time: str) -> Dict[str, Any]:
    as_of_date = as_of_time[:10]
    return json.loads((snapshot_root / "keyed" / f"as_of_date={as_of_date}" / f"company_id={company_id}.json").read_text())


def _humanize_action(action_id: str) -> str:
    if not action_id:
        return ""
    return action_id.split(".")[-1].replace("_", " ")


def _load_realized_outcomes_lookup(path: Path) -> Dict[str, List[Tuple[Any, str, str]]]:
    import pandas as pd

    frame = pd.read_parquet(
        path,
        columns=["company_id", "action_date", "normalized_action_id", "normalized_action_family"],
    )
    frame["action_date"] = pd.to_datetime(frame["action_date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["company_id", "action_date"])
    lookup: Dict[str, List[Tuple[Any, str, str]]] = {}
    for company_id, group in frame.groupby("company_id", sort=False):
        ordered = group.sort_values("action_date")
        lookup[str(company_id)] = [
            (
                row.action_date,
                str(row.normalized_action_id or ""),
                str(row.normalized_action_family or ""),
            )
            for row in ordered.itertuples(index=False)
        ]
    return lookup


def _build_ex_post_comparison(
    *,
    company_id: str,
    as_of_time: str,
    model_packet: CanonicalPacket,
    baseline_packet: CanonicalPacket,
    outcomes_lookup: Optional[Dict[str, List[Tuple[Any, str, str]]]],
    alignment_horizon_days: int,
) -> Dict[str, Any]:
    if outcomes_lookup is None:
        return {}
    model_alignment = _score_ex_post_alignment(
        company_id=company_id,
        as_of_time=as_of_time,
        packet=model_packet,
        outcomes_lookup=outcomes_lookup,
        alignment_horizon_days=alignment_horizon_days,
    )
    baseline_alignment = _score_ex_post_alignment(
        company_id=company_id,
        as_of_time=as_of_time,
        packet=baseline_packet,
        outcomes_lookup=outcomes_lookup,
        alignment_horizon_days=alignment_horizon_days,
    )
    model_score = model_alignment.get("score")
    baseline_score = baseline_alignment.get("score")
    winner: Optional[str]
    if model_score is None or baseline_score is None:
        winner = None
    elif float(model_score) > float(baseline_score):
        winner = "model"
    elif float(model_score) < float(baseline_score):
        winner = "baseline"
    else:
        winner = "tie"
    return {
        "model": model_alignment,
        "baseline": baseline_alignment,
        "winner": winner,
    }


def _score_ex_post_alignment(
    *,
    company_id: str,
    as_of_time: str,
    packet: CanonicalPacket,
    outcomes_lookup: Dict[str, List[Tuple[Any, str, str]]],
    alignment_horizon_days: int,
) -> Dict[str, Any]:
    import pandas as pd

    targets = list(packet.action_path or _infer_action_path(" ".join([packet.primary_recommendation, packet.recommendation_thesis])))
    target_families = sorted({_action_family(action_id) for action_id in targets if _action_family(action_id)})
    company_events = list(outcomes_lookup.get(company_id, []) or [])
    if not company_events:
        return {"score": None, "reason": "no_company_events"}
    as_of_ts = pd.Timestamp(as_of_time)
    end_ts = as_of_ts + pd.Timedelta(days=max(1, int(alignment_horizon_days)))
    future_events = [
        (ts, action_id, family)
        for ts, action_id, family in company_events
        if ts > as_of_ts and ts <= end_ts
    ]
    if not future_events:
        return {"score": None, "reason": "no_future_events_within_horizon"}
    if not targets and not target_families:
        return {"score": None, "reason": "no_inferred_action_target"}
    future_ids = {action_id for _, action_id, _ in future_events if action_id}
    future_families = {family for _, _, family in future_events if family}
    primary = targets[0] if targets else ""
    primary_family = _action_family(primary) if primary else ""
    exact_primary = bool(primary and primary in future_ids)
    exact_any = bool(set(targets) & future_ids)
    family_primary = bool(primary_family and primary_family in future_families)
    family_any = bool(set(target_families) & future_families)
    if exact_primary:
        score = 1.0
    elif exact_any:
        score = 0.85
    elif family_primary:
        score = 0.6
    elif family_any:
        score = 0.4
    else:
        score = 0.0
    return {
        "score": round(score, 6),
        "reason": "matched" if score > 0.0 else "no_alignment",
        "primary_exact_match": exact_primary,
        "any_exact_match": exact_any,
        "primary_family_match": family_primary,
        "any_family_match": family_any,
        "future_action_ids_sample": sorted(future_ids)[:5],
        "future_action_families_sample": sorted(future_families)[:5],
    }


def _action_family(action_id: str) -> str:
    text = str(action_id or "")
    if not text or "." not in text:
        return ""
    return text.split(".", 1)[0]


def _aggregate_cases_by_bucket(*, cases: Sequence[Dict[str, Any]], field: str) -> Dict[str, Any]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for case in cases:
        baseline = dict(case.get("baseline_packet", {}) or {})
        label = str(baseline.get(field, "") or "unknown")
        buckets.setdefault(label, []).append(case)
    out: Dict[str, Any] = {}
    for label, bucket_cases in sorted(buckets.items()):
        count = float(len(bucket_cases))
        model_wins = sum(1 for case in bucket_cases if str((case.get("comparison", {}) or {}).get("winner", "")) == "model")
        baseline_wins = sum(1 for case in bucket_cases if str((case.get("comparison", {}) or {}).get("winner", "")) == "baseline")
        ties = sum(1 for case in bucket_cases if str((case.get("comparison", {}) or {}).get("winner", "")) == "tie")
        significance = _compute_significance(model_wins=model_wins, baseline_wins=baseline_wins)
        deltas = [float((case.get("comparison", {}) or {}).get("score_delta", 0.0) or 0.0) for case in bucket_cases]
        out[label] = {
            "case_count": int(count),
            "model_win_rate": round(model_wins / count, 6) if count else None,
            "baseline_win_rate": round(baseline_wins / count, 6) if count else None,
            "tie_rate": round(ties / count, 6) if count else None,
            "mean_score_delta": round(sum(deltas) / count, 6) if count else None,
            "median_score_delta": round(_median(deltas), 6) if count else None,
            "sign_test_p_value": significance["sign_test_p_value"],
            "model_win_rate_ci_95": significance["model_win_rate_ci_95"],
        }
    return out


def _aggregate_ex_post(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    comparable: List[Tuple[float, float, str]] = []
    coverage = 0
    for case in cases:
        ex_post = dict(case.get("ex_post", {}) or {})
        model_score = ((ex_post.get("model", {}) or {}).get("score"))
        baseline_score = ((ex_post.get("baseline", {}) or {}).get("score"))
        if model_score is not None or baseline_score is not None:
            coverage += 1
        if model_score is None or baseline_score is None:
            continue
        comparable.append((float(model_score), float(baseline_score), str(ex_post.get("winner", "") or "tie")))
    if not comparable and not coverage:
        return {}
    model_wins = sum(1 for _, _, winner in comparable if winner == "model")
    baseline_wins = sum(1 for _, _, winner in comparable if winner == "baseline")
    ties = sum(1 for _, _, winner in comparable if winner == "tie")
    significance = _compute_significance(model_wins=model_wins, baseline_wins=baseline_wins)
    return {
        "coverage_rate": round(float(coverage) / float(len(cases)), 6) if cases else 0.0,
        "comparable_case_count": len(comparable),
        "model_mean_score": round(sum(x for x, _, _ in comparable) / len(comparable), 6) if comparable else None,
        "baseline_mean_score": round(sum(y for _, y, _ in comparable) / len(comparable), 6) if comparable else None,
        "model_win_rate": round(float(model_wins) / len(comparable), 6) if comparable else None,
        "baseline_win_rate": round(float(baseline_wins) / len(comparable), 6) if comparable else None,
        "tie_rate": round(float(ties) / len(comparable), 6) if comparable else None,
        "sign_test_p_value": significance["sign_test_p_value"],
        "model_win_rate_ci_95": significance["model_win_rate_ci_95"],
    }


def _compute_significance(*, model_wins: int, baseline_wins: int) -> Dict[str, Any]:
    comparable = int(model_wins) + int(baseline_wins)
    if comparable <= 0:
        return {
            "comparable_case_count": 0,
            "sign_test_p_value": None,
            "model_win_rate_ci_95": None,
        }
    win_rate = float(model_wins) / float(comparable)
    return {
        "comparable_case_count": comparable,
        "sign_test_p_value": round(_exact_two_sided_sign_test(k=max(model_wins, baseline_wins), n=comparable), 6),
        "model_win_rate_ci_95": tuple(round(x, 6) for x in _wilson_interval(successes=model_wins, n=comparable, z=1.96)),
        "model_win_rate": round(win_rate, 6),
    }


def _exact_two_sided_sign_test(*, k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / float(2**n)
    return min(1.0, 2.0 * tail)


def _wilson_interval(*, successes: int, n: int, z: float) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = float(successes) / float(n)
    denom = 1.0 + (z * z) / float(n)
    center = (p + (z * z) / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt((p * (1.0 - p) / float(n)) + ((z * z) / (4.0 * n * n)))
    return (max(0.0, center - spread), min(1.0, center + spread))


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(x) for x in values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _fmt_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _fmt_interval(value: Any) -> str:
    if not value:
        return "n/a"
    left, right = value
    return f"{left:.3f}, {right:.3f}"
