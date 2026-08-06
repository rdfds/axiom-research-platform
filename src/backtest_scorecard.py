from __future__ import annotations

import math
from statistics import median
from typing import Any, Dict, Iterable, List, Optional

from .backtest_costs import TransactionCostModel
from .backtest_protocol import BacktestProtocol


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out):
        return None
    return float(out)


def _mean(values: Iterable[float]) -> Optional[float]:
    xs = [float(value) for value in values]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _stdev(values: Iterable[float]) -> Optional[float]:
    xs = [float(value) for value in values]
    if len(xs) < 2:
        return None
    mean_val = sum(xs) / len(xs)
    variance = sum((value - mean_val) ** 2 for value in xs) / (len(xs) - 1)
    return math.sqrt(max(0.0, variance))


def _nested_lookup(payload: Dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for token in str(dotted_path or "").split("."):
        if not token:
            continue
        if not isinstance(current, dict):
            return None
        current = current.get(token)
    return current


def _case_primary_family(case: Dict[str, Any]) -> str:
    action_ids = list(case.get("top_action_ids", []) or [])
    if action_ids:
        primary = str(action_ids[0] or "")
        if "." in primary:
            return primary.split(".", 1)[0]
    return str(case.get("anchor_action_family") or "")


def _count_values(values: Iterable[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        token = str(value or "").strip()
        if not token:
            continue
        out[token] = out.get(token, 0) + 1
    return dict(sorted(out.items()))


def _support_mode(case: Dict[str, Any]) -> str:
    support = list(case.get("recommended_action_support", []) or [])
    if support:
        return str((support[0] or {}).get("support_mode") or "")
    return ""


def build_portfolio_strategy_scorecard(
    report: Dict[str, Any],
    *,
    protocol: BacktestProtocol,
    cost_model: TransactionCostModel,
) -> Dict[str, Any]:
    cases = list(report.get("cases", []) or [])
    reference_metrics = dict(report.get("reference_metrics", {}) or {})
    scored_cases: List[Dict[str, Any]] = []
    error_cases = 0
    unsupported_cases = 0

    for case in cases:
        if case.get("error"):
            error_cases += 1
            continue
        if case.get("unsupported_reason"):
            unsupported_cases += 1
            continue
        gross_score = _safe_float(_nested_lookup(case, protocol.score_field))
        if gross_score is None:
            continue
        action_family = _case_primary_family(case)
        estimated_cost = cost_model.estimate_case_cost(
            action_family=action_family,
            holding_period_days=protocol.holding_period_days,
            turnover_fraction=protocol.turnover_fraction,
        )
        net_score = max(0.0, float(gross_score) - float(estimated_cost["total_cost_fraction"]))
        scored_cases.append(
            {
                "company_id": str(case.get("source_company_id") or case.get("company_id") or ""),
                "action_family": action_family,
                "gross_score": round(float(gross_score), 6),
                "net_score": round(net_score, 6),
                "estimated_cost_bps": float(estimated_cost["total_cost_bps"]),
                "support_mode": _support_mode(case),
                "reason": str(((case.get("historical_alignment", {}) or {}).get("reason")) or ""),
            }
        )

    scored_count = len(scored_cases)
    gross_scores = [float(case["gross_score"]) for case in scored_cases]
    net_scores = [float(case["net_score"]) for case in scored_cases]
    cost_bps = [float(case["estimated_cost_bps"]) for case in scored_cases]

    gross_mean = _mean(gross_scores)
    net_mean = _mean(net_scores)
    gross_stdev = _stdev(gross_scores)
    net_stdev = _stdev(net_scores)
    strong_threshold = float(protocol.strong_alignment_threshold)

    family_counts = _count_values(case["action_family"] for case in scored_cases)
    family_weights = {
        family: round(count / scored_count, 6)
        for family, count in family_counts.items()
        if scored_count > 0
    }
    max_family_weight = max(family_weights.values()) if family_weights else 0.0
    support_mode_counts = _count_values(case["support_mode"] for case in scored_cases)

    flags: List[str] = []
    if scored_count < int(protocol.min_case_count):
        flags.append("insufficient_scored_cases")
    if max_family_weight > 0.60:
        flags.append("family_concentration_high")
    if cost_bps and (_mean(cost_bps) or 0.0) > 25.0:
        flags.append("cost_drag_high")
    reference_mean = _safe_float(reference_metrics.get("mean_alignment_score"))
    if reference_mean is not None and net_mean is not None and net_mean + 1e-12 < float(reference_mean):
        flags.append("net_mean_alignment_below_reference")

    benchmark_comparison = {
        "reference_mean_alignment_score": reference_mean,
        "delta_mean_alignment_score": round((gross_mean or 0.0) - float(reference_mean), 6)
        if reference_mean is not None and gross_mean is not None
        else None,
        "delta_net_alignment_score": round((net_mean or 0.0) - float(reference_mean), 6)
        if reference_mean is not None and net_mean is not None
        else None,
        "reference_anchor_primary_exact_rate": _safe_float(reference_metrics.get("anchor_primary_exact_rate")),
        "reference_anchor_primary_family_rate": _safe_float(reference_metrics.get("anchor_primary_family_rate")),
        "reference_unsupported_case_count": int(reference_metrics.get("unsupported_case_count", 0) or 0),
        "delta_unsupported_case_count": int(unsupported_cases) - int(reference_metrics.get("unsupported_case_count", 0) or 0),
    }

    gross_sharpe_like = round((gross_mean or 0.0) / gross_stdev, 6) if gross_mean is not None and gross_stdev not in (None, 0.0) else None
    net_sharpe_like = round((net_mean or 0.0) / net_stdev, 6) if net_mean is not None and net_stdev not in (None, 0.0) else None

    return {
        "protocol": protocol.to_dict(),
        "cost_model": cost_model.to_dict(),
        "case_counts": {
            "requested_cases": len(cases),
            "scored_cases": scored_count,
            "unsupported_cases": int(unsupported_cases),
            "error_cases": int(error_cases),
        },
        "portfolio_proxy": {
            "weighting_mode": protocol.weighting_mode,
            "rebalance_mode": protocol.rebalance_mode,
            "holding_period_days": int(protocol.holding_period_days),
            "turnover_fraction": float(protocol.turnover_fraction),
            "gross_mean_alignment_score": round(gross_mean, 6) if gross_mean is not None else None,
            "gross_median_alignment_score": round(float(median(gross_scores)), 6) if gross_scores else None,
            "gross_score_stdev": round(gross_stdev, 6) if gross_stdev is not None else None,
            "gross_strong_alignment_rate": round(sum(score >= strong_threshold for score in gross_scores) / scored_count, 6) if scored_count else 0.0,
            "net_mean_alignment_score": round(net_mean, 6) if net_mean is not None else None,
            "net_median_alignment_score": round(float(median(net_scores)), 6) if net_scores else None,
            "net_score_stdev": round(net_stdev, 6) if net_stdev is not None else None,
            "net_strong_alignment_rate": round(sum(score >= strong_threshold for score in net_scores) / scored_count, 6) if scored_count else 0.0,
            "average_estimated_cost_bps": round(_mean(cost_bps) or 0.0, 6),
            "median_estimated_cost_bps": round(float(median(cost_bps)), 6) if cost_bps else 0.0,
            "gross_sharpe_like": gross_sharpe_like,
            "net_sharpe_like": net_sharpe_like,
        },
        "coverage": {
            "support_mode_counts": support_mode_counts,
            "recommended_family_counts": family_counts,
            "recommended_family_weights": family_weights,
            "max_family_weight": round(max_family_weight, 6),
        },
        "benchmark_comparison": benchmark_comparison,
        "flags": flags,
        "scored_case_sample": scored_cases[:20],
    }


def render_portfolio_strategy_scorecard_markdown(scorecard: Dict[str, Any]) -> str:
    portfolio = dict(scorecard.get("portfolio_proxy", {}) or {})
    case_counts = dict(scorecard.get("case_counts", {}) or {})
    coverage = dict(scorecard.get("coverage", {}) or {})
    benchmark = dict(scorecard.get("benchmark_comparison", {}) or {})
    lines: List[str] = []
    lines.append("# Canonical Backtest Scorecard")
    lines.append("")
    lines.append(f"- Scored cases: `{case_counts.get('scored_cases', 0)}` / `{case_counts.get('requested_cases', 0)}`")
    lines.append(f"- Unsupported cases: `{case_counts.get('unsupported_cases', 0)}`")
    lines.append(f"- Error cases: `{case_counts.get('error_cases', 0)}`")
    lines.append(f"- Gross mean alignment score: `{portfolio.get('gross_mean_alignment_score')}`")
    lines.append(f"- Net mean alignment score: `{portfolio.get('net_mean_alignment_score')}`")
    lines.append(f"- Average estimated cost (bps): `{portfolio.get('average_estimated_cost_bps')}`")
    lines.append(f"- Gross strong alignment rate: `{portfolio.get('gross_strong_alignment_rate')}`")
    lines.append(f"- Net strong alignment rate: `{portfolio.get('net_strong_alignment_rate')}`")
    lines.append("")

    if benchmark:
        lines.append("## Benchmark Comparison")
        lines.append("")
        for key in (
            "reference_mean_alignment_score",
            "delta_mean_alignment_score",
            "delta_net_alignment_score",
            "reference_anchor_primary_exact_rate",
            "reference_anchor_primary_family_rate",
            "reference_unsupported_case_count",
            "delta_unsupported_case_count",
        ):
            if key in benchmark:
                lines.append(f"- `{key}`: `{benchmark.get(key)}`")
        lines.append("")

    family_counts = dict(coverage.get("recommended_family_counts", {}) or {})
    if family_counts:
        lines.append("## Family Mix")
        lines.append("")
        for family, count in family_counts.items():
            lines.append(f"- `{family}`: `{count}`")
        lines.append("")

    flags = list(scorecard.get("flags", []) or [])
    lines.append("## Flags")
    lines.append("")
    if flags:
        for flag in flags:
            lines.append(f"- `{flag}`")
    else:
        lines.append("- `none`")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "build_portfolio_strategy_scorecard",
    "render_portfolio_strategy_scorecard_markdown",
]
