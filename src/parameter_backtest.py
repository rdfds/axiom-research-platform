from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .action_ontology import build_default_action_schema_registry
from .board_ready_dossier import build_board_ready_dossier
from .planner_brain import build_plan_set
from .recommendation_run import RecommendationRun


_POSITIVE_METRICS: Tuple[str, ...] = (
    "revenue_delta",
    "margin_delta",
    "eps_delta",
    "roic_delta",
    "fcf_margin_delta",
    "outcome_pe_12m",
    "outcome_ev_ebitda_12m",
    "rating_migration_12m",
)
_NEGATIVE_METRICS: Tuple[str, ...] = (
    "leverage_delta",
    "credit_spread_change_12m",
)
_SIZE_PARAM_PRIORITY: Tuple[str, ...] = (
    "size_pct_market_cap",
    "size_absolute_usd",
    "amount_refinanced_usd",
    "amount_usd",
    "draw_amount_usd",
    "resize_amount_usd",
    "target_size_pct_ev",
    "percent_divested",
    "estimated_ev_usd",
)


def build_parameter_backtest_report(
    runs_roots: Sequence[str | Path],
    snapshot_root: str | Path,
    outcomes_path: str | Path,
    run_ids: Optional[Sequence[str]] = None,
    review_count: int = 50,
    limit: Optional[int] = None,
    min_bucket_samples: int = 25,
) -> Dict[str, Any]:
    resolved_roots = [Path(root) for root in runs_roots]
    snapshot_root_path = Path(snapshot_root)
    selected_run_ids = _resolve_run_ids(runs_roots=resolved_roots, run_ids=run_ids, limit=limit)
    registry = build_default_action_schema_registry()
    backtester = HistoricalParameterBacktester(
        outcomes_path=Path(outcomes_path),
        min_bucket_samples=min_bucket_samples,
    )

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
                    backtester=backtester,
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

    aggregate = _aggregate_cases(
        cases=cases,
        missing_artifacts=missing_artifacts,
        cohort_priors=backtester.cohort_priors(),
    )
    review_queue = _select_review_queue(cases=cases, review_count=review_count)
    return {
        "ok": True,
        "runs_analyzed": len(cases),
        "missing_artifacts": missing_artifacts,
        "aggregate": aggregate,
        "review_queue": review_queue,
        "cases": cases,
    }


def render_parameter_backtest_markdown(report: Dict[str, Any]) -> str:
    aggregate = dict(report.get("aggregate", {}) or {})
    lines: List[str] = []
    lines.append("# Parameter Backtest Report")
    lines.append("")
    lines.append(f"- Runs analyzed: `{report.get('runs_analyzed', 0)}`")
    lines.append(f"- Missing artifacts: `{len(report.get('missing_artifacts', []) or [])}`")
    lines.append(f"- Historical coverage rate: `{aggregate.get('historical_coverage_rate', 0.0):.3f}`")
    lines.append(f"- Mean alignment score: `{aggregate.get('mean_alignment_score', 0.0):.3f}`")
    lines.append(f"- Bucket match rate: `{aggregate.get('bucket_match_rate', 0.0):.3f}`")
    lines.append(f"- Strong support rate: `{aggregate.get('strong_support_rate', 0.0):.3f}`")
    lines.append(f"- Missing artifact rate: `{aggregate.get('missing_artifact_rate', 0.0):.3f}`")
    lines.append("")

    flag_counts = dict(aggregate.get("flag_counts", {}) or {})
    if flag_counts:
        lines.append("## Flags")
        lines.append("")
        for flag, count in sorted(flag_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{flag}`: `{count}`")
        lines.append("")

    priors = dict(aggregate.get("cohort_priors", {}) or {})
    if priors:
        lines.append("## Historical Cohort Priors")
        lines.append("")
        for cohort_key, payload in sorted(priors.items()):
            lines.append(f"- `{cohort_key}`: best bucket `{payload.get('best_bucket')}` from `{payload.get('best_bucket_n')}` cases")
        lines.append("")

    lines.append("## Review Queue")
    lines.append("")
    for idx, case in enumerate(report.get("review_queue", []) or [], start=1):
        lines.extend(_render_case_markdown(case=case, index=idx))
    return "\n".join(lines).strip() + "\n"


class HistoricalParameterBacktester:
    def __init__(
        self,
        *,
        outcomes_path: Path,
        min_bucket_samples: int = 25,
    ) -> None:
        self.outcomes_path = outcomes_path
        self.min_bucket_samples = max(1, int(min_bucket_samples))
        self.frame = _load_outcomes_frame(outcomes_path)
        self._cohort_cache: Dict[str, Dict[str, Any]] = {}

    def score_dossier(
        self,
        *,
        dossier: Dict[str, Any],
        snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        parameter_optimization = dict(dossier.get("parameter_optimization", {}) or {})
        action_id = str(parameter_optimization.get("action_id", "") or "")
        if not action_id:
            return {"supported": False, "reasons": ["missing_parameter_optimization"]}

        parameter_name, parameter_payload = _select_backtest_parameter(parameter_optimization)
        if not parameter_name:
            return {"supported": False, "reasons": ["no_backtestable_parameter"]}

        cohort_spec = _cohort_spec_for_action(action_id)
        if cohort_spec is None:
            return {
                "supported": False,
                "reasons": [f"no_historical_cohort_for:{action_id}"],
                "parameter_name": parameter_name,
            }

        cohort = self._cohort_summary(cohort_spec)
        if not cohort.get("supported"):
            return {
                "supported": False,
                "reasons": list(cohort.get("reasons", []) or []),
                "parameter_name": parameter_name,
                "cohort_key": cohort_spec["cohort_key"],
            }

        recommended_bucket = _recommended_bucket(
            parameter_name=parameter_name,
            parameter_payload=parameter_payload,
            snapshot=snapshot,
        )
        if not recommended_bucket:
            return {
                "supported": False,
                "reasons": [f"unable_to_bucket:{parameter_name}"],
                "parameter_name": parameter_name,
                "cohort_key": cohort_spec["cohort_key"],
            }

        bucket_stats = list(cohort.get("bucket_stats", []) or [])
        bucket_by_name = {str(item.get("bucket")): item for item in bucket_stats}
        recommended_stats = bucket_by_name.get(recommended_bucket)
        best_bucket = str(cohort.get("best_bucket", "") or "")
        best_stats = bucket_by_name.get(best_bucket, {})
        if recommended_stats is None:
            return {
                "supported": False,
                "reasons": [f"bucket_not_supported:{recommended_bucket}"],
                "parameter_name": parameter_name,
                "cohort_key": cohort_spec["cohort_key"],
                "available_buckets": sorted(bucket_by_name),
            }

        recommended_score = float(recommended_stats.get("median_score", 0.0) or 0.0)
        best_score = float(best_stats.get("median_score", 0.0) or 0.0)
        if best_score > 0.0:
            alignment_score = max(0.0, min(1.0, recommended_score / best_score))
        else:
            alignment_score = 0.0

        if recommended_bucket == best_bucket:
            tuning_suggestion = "keep_current_bucket"
        else:
            tuning_suggestion = f"shift_toward_{best_bucket}"

        notes = [
            f"Recommended bucket `{recommended_bucket}` has median historical composite `{recommended_score:.3f}`.",
            f"Best historical bucket is `{best_bucket}` at `{best_score:.3f}` across `{best_stats.get('n', 0)}` cases.",
        ]
        if recommended_bucket != best_bucket:
            notes.append(f"Historical evidence suggests biasing the parameter toward `{best_bucket}` instead of `{recommended_bucket}`.")

        return {
            "supported": True,
            "action_id": action_id,
            "cohort_key": cohort_spec["cohort_key"],
            "parameter_name": parameter_name,
            "parameter_summary": str(parameter_optimization.get("summary", "") or ""),
            "recommended_bucket": recommended_bucket,
            "best_bucket": best_bucket,
            "recommended_bucket_score": round(recommended_score, 6),
            "best_bucket_score": round(best_score, 6),
            "alignment_score": round(alignment_score, 6),
            "bucket_match": recommended_bucket == best_bucket,
            "strong_support": alignment_score >= 0.8,
            "tuning_suggestion": tuning_suggestion,
            "bucket_stats": bucket_stats,
            "notes": notes,
        }

    def _cohort_summary(self, cohort_spec: Dict[str, Any]) -> Dict[str, Any]:
        cohort_key = str(cohort_spec.get("cohort_key", "") or "")
        cached = self._cohort_cache.get(cohort_key)
        if cached is not None:
            return cached

        frame = self.frame
        exact_id = cohort_spec.get("normalized_action_id")
        if exact_id:
            cohort = frame[frame["normalized_action_id"] == exact_id].copy()
        else:
            family = cohort_spec.get("normalized_action_family")
            subfamilies = list(cohort_spec.get("normalized_action_subfamilies", []) or [])
            cohort = frame[frame["normalized_action_family"] == family].copy()
            if subfamilies:
                cohort = cohort[cohort["normalized_action_subfamily"].isin(subfamilies)].copy()

        if cohort.empty:
            out = {"supported": False, "reasons": [f"no_historical_rows:{cohort_key}"]}
            self._cohort_cache[cohort_key] = out
            return out

        scored = _score_historical_rows(cohort)
        if scored.empty:
            out = {"supported": False, "reasons": [f"no_scored_rows:{cohort_key}"]}
            self._cohort_cache[cohort_key] = out
            return out

        bucket_stats: List[Dict[str, Any]] = []
        for bucket in ("small", "medium", "large"):
            bucket_frame = scored[scored["size_bucket"] == bucket]
            if len(bucket_frame) < self.min_bucket_samples:
                continue
            bucket_stats.append(
                {
                    "bucket": bucket,
                    "n": int(len(bucket_frame)),
                    "median_score": round(float(bucket_frame["composite_score"].median()), 6),
                    "mean_score": round(float(bucket_frame["composite_score"].mean()), 6),
                }
            )

        if not bucket_stats:
            out = {"supported": False, "reasons": [f"insufficient_bucket_samples:{cohort_key}"]}
            self._cohort_cache[cohort_key] = out
            return out

        bucket_stats.sort(key=lambda item: (-float(item.get("median_score", 0.0) or 0.0), -int(item.get("n", 0) or 0), str(item.get("bucket", ""))))
        best_bucket = str(bucket_stats[0]["bucket"])
        out = {
            "supported": True,
            "cohort_key": cohort_key,
            "sample_size": int(len(scored)),
            "best_bucket": best_bucket,
            "best_bucket_n": int(bucket_stats[0]["n"]),
            "bucket_stats": bucket_stats,
        }
        self._cohort_cache[cohort_key] = out
        return out

    def cohort_priors(self) -> Dict[str, Any]:
        priors: Dict[str, Any] = {}
        for action_id in _SUPPORTED_COHORT_ACTIONS:
            spec = _cohort_spec_for_action(action_id)
            if spec is None:
                continue
            summary = self._cohort_summary(spec)
            if summary.get("supported"):
                priors[spec["cohort_key"]] = {
                    "best_bucket": summary.get("best_bucket"),
                    "best_bucket_n": summary.get("best_bucket_n"),
                }
        return priors


def _load_outcomes_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=[
            "normalized_action_id",
            "normalized_action_family",
            "normalized_action_subfamily",
            "family_scale_bucket",
            "action_size",
            "base_market_cap",
            *_POSITIVE_METRICS,
            *_NEGATIVE_METRICS,
        ],
    )
    for column in ("normalized_action_id", "normalized_action_family", "normalized_action_subfamily", "family_scale_bucket"):
        frame[column] = frame[column].astype("string")
    frame["size_ratio"] = pd.to_numeric(frame["action_size"], errors="coerce") / pd.to_numeric(frame["base_market_cap"], errors="coerce")
    frame["size_bucket"] = frame["family_scale_bucket"].fillna("")
    mask_missing_bucket = frame["size_bucket"].eq("") | frame["size_bucket"].isna()
    frame.loc[mask_missing_bucket, "size_bucket"] = frame.loc[mask_missing_bucket, "size_ratio"].map(_size_bucket)
    return frame


def _size_bucket(value: Any) -> Optional[str]:
    try:
        ratio = float(value)
    except Exception:
        return None
    if pd.isna(ratio) or ratio <= 0.0:
        return None
    if ratio < 0.05:
        return "small"
    if ratio < 0.25:
        return "medium"
    return "large"


def _score_historical_rows(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    metric_scores: List[pd.Series] = []
    for column in _POSITIVE_METRICS:
        series = pd.to_numeric(scored.get(column), errors="coerce")
        if series.notna().sum() < 5:
            continue
        metric_scores.append(series.rank(pct=True, method="average"))
    for column in _NEGATIVE_METRICS:
        series = pd.to_numeric(scored.get(column), errors="coerce")
        if series.notna().sum() < 5:
            continue
        metric_scores.append(1.0 - series.rank(pct=True, method="average"))
    if not metric_scores:
        return pd.DataFrame()
    score_table = pd.concat(metric_scores, axis=1)
    scored["composite_score"] = score_table.mean(axis=1, skipna=True)
    scored = scored[scored["composite_score"].notna()].copy()
    return scored


