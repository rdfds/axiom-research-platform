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


def _cohort_spec_for_action(action_id: str) -> Optional[Dict[str, Any]]:
    aid = str(action_id or "")
    if not aid:
        return None
    exact = {
        "capital_structure.equity_issuance",
        "capital_return.dividend_increase",
        "capital_return.dividend_cut",
        "capital_return.dividend_initiate",
        "capital_return.special_dividend",
        "mna.go_private_lbo",
    }
    if aid in exact:
        return {"cohort_key": aid, "normalized_action_id": aid}
    if aid in {
        "capital_return.open_market_buyback",
        "capital_return.accelerated_share_repurchase",
        "capital_return.tender_offer_buyback",
    }:
        return {
            "cohort_key": "capital_return.buyback",
            "normalized_action_family": "capital_return",
            "normalized_action_subfamilies": ["buyback"],
        }
    if aid in {
        "capital_structure.refinancing",
        "capital_structure.new_debt_issuance",
        "capital_structure.revolver_draw_or_resize",
        "capital_structure.tender_offer_debt",
        "capital_structure.exchange_offer",
        "capital_structure.liability_management_exercise",
    }:
        return {
            "cohort_key": "capital_structure.debt_family",
            "normalized_action_family": "capital_structure",
            "normalized_action_subfamilies": ["debt_bond", "debt_loan", "revolver"],
        }
    if aid in {
        "capital_structure.convertible_issuance",
        "capital_structure.preferred_issuance",
    }:
        return {"cohort_key": "capital_structure.equity_issuance", "normalized_action_id": "capital_structure.equity_issuance"}
    if aid in {"mna.platform_acquisition", "mna.tuck_in_acquisition", "mna.transformational_acquisition"}:
        return {
            "cohort_key": "mna.acquisition_family",
            "normalized_action_family": "mna",
        }
    if aid.startswith("portfolio."):
        return {
            "cohort_key": "portfolio.divestiture_family",
            "normalized_action_family": "portfolio",
        }
    return None


_SUPPORTED_COHORT_ACTIONS: Tuple[str, ...] = (
    "capital_return.open_market_buyback",
    "capital_return.dividend_increase",
    "capital_return.dividend_cut",
    "capital_return.dividend_initiate",
    "capital_return.special_dividend",
    "capital_structure.refinancing",
    "capital_structure.new_debt_issuance",
    "capital_structure.revolver_draw_or_resize",
    "capital_structure.equity_issuance",
    "capital_structure.convertible_issuance",
    "capital_structure.preferred_issuance",
    "mna.platform_acquisition",
    "mna.tuck_in_acquisition",
    "mna.go_private_lbo",
    "portfolio.divestiture_partial",
)


def _select_backtest_parameter(parameter_optimization: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    parameters = dict(parameter_optimization.get("recommended_parameters", {}) or {})
    for parameter_name in _SIZE_PARAM_PRIORITY:
        payload = dict(parameters.get(parameter_name, {}) or {})
        if payload:
            return parameter_name, payload
    return "", {}


def _recommended_bucket(
    *,
    parameter_name: str,
    parameter_payload: Dict[str, Any],
    snapshot: Dict[str, Any],
) -> Optional[str]:
    market_cap = _feature_float(snapshot, "market.market_cap")
    recommended_value = parameter_payload.get("recommended_value")
    ratio: Optional[float] = None
    if parameter_name in {"size_pct_market_cap", "target_size_pct_ev", "percent_divested"}:
        ratio = _safe_float(recommended_value)
    elif parameter_name in {"size_absolute_usd", "amount_refinanced_usd", "amount_usd", "draw_amount_usd", "resize_amount_usd", "estimated_ev_usd"}:
        amount = _safe_float(recommended_value)
        if amount is not None and market_cap not in (None, 0.0):
            ratio = amount / market_cap
    return _size_bucket(ratio)


def _feature_float(snapshot: Dict[str, Any], key: str) -> Optional[float]:
    features = dict(snapshot.get("features", {}) or {})
    value = features.get(key)
    if isinstance(value, dict):
        value = value.get("value")
    return _safe_float(value)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        value = float(value)
        if pd.isna(value):
            return None
        return value
    except Exception:
        return None


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
    backtester: HistoricalParameterBacktester,
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
    top_steps = list(((plan_set.get("plans", []) or [{}])[0].get("steps", []) or []))
    top_action = str((top_steps[0].get("action_id", "") if top_steps else ""))
    historical = backtester.score_dossier(dossier=dossier, snapshot=snapshot)
    return {
        "run_id": run_id,
        "runs_root": str(runs_root),
        "company_id": recommendation_run.company_id,
        "top_action": top_action,
        "parameter_summary": str((dossier.get("parameter_optimization", {}) or {}).get("summary", "") or ""),
        "historical": historical,
        "dossier": {
            "recommended_posture": ((dossier.get("status_quo_view", {}) or {}).get("recommended_posture")),
            "parameter_optimization": dossier.get("parameter_optimization"),
            "sizing_guidance": dossier.get("sizing_guidance"),
            "executive_summary": dossier.get("executive_summary"),
        },
    }


def _load_snapshot(*, snapshot_root: Path, company_id: str, as_of_time: str) -> Dict[str, Any]:
    as_of_date = as_of_time[:10]
    path = snapshot_root / "keyed" / f"as_of_date={as_of_date}" / f"company_id={company_id}.json"
    return json.loads(path.read_text())


def _aggregate_cases(
    *,
    cases: Sequence[Dict[str, Any]],
    missing_artifacts: Sequence[Dict[str, Any]],
    cohort_priors: Dict[str, Any],
) -> Dict[str, Any]:
    if not cases:
        return {
            "historical_coverage_rate": 0.0,
            "mean_alignment_score": 0.0,
            "bucket_match_rate": 0.0,
            "strong_support_rate": 0.0,
            "missing_artifact_rate": 1.0 if missing_artifacts else 0.0,
            "flag_counts": {},
            "cohort_priors": cohort_priors,
        }
    flags = Counter()
    supported = [case for case in cases if (case.get("historical", {}) or {}).get("supported")]
    for case in cases:
        historical = dict(case.get("historical", {}) or {})
        if not historical.get("supported"):
            for reason in list(historical.get("reasons", []) or []):
                flags[str(reason)] += 1
        elif not historical.get("bucket_match"):
            flags["bucket_mismatch"] += 1
    count = float(len(cases))
    supported_count = float(len(supported))
    return {
        "historical_coverage_rate": round(supported_count / count, 6),
        "mean_alignment_score": round(sum(float((case.get("historical", {}) or {}).get("alignment_score", 0.0) or 0.0) for case in supported) / supported_count, 6) if supported else 0.0,
        "bucket_match_rate": round(sum(1.0 for case in supported if (case.get("historical", {}) or {}).get("bucket_match")) / supported_count, 6) if supported else 0.0,
        "strong_support_rate": round(sum(1.0 for case in supported if (case.get("historical", {}) or {}).get("strong_support")) / supported_count, 6) if supported else 0.0,
        "missing_artifact_rate": round(len(missing_artifacts) / (len(cases) + len(missing_artifacts)), 6) if (cases or missing_artifacts) else 0.0,
        "flag_counts": dict(flags),
        "cohort_priors": cohort_priors,
    }


def _select_review_queue(cases: Sequence[Dict[str, Any]], review_count: int) -> List[Dict[str, Any]]:
    ranked = sorted(
        cases,
        key=lambda case: (
            0 if (case.get("historical", {}) or {}).get("supported") else 1,
            float((case.get("historical", {}) or {}).get("alignment_score", 0.0) or 0.0),
            str(case.get("company_id", "")),
        ),
    )
    return ranked[: max(0, int(review_count))]


def _render_case_markdown(case: Dict[str, Any], index: int) -> List[str]:
    historical = dict(case.get("historical", {}) or {})
    lines: List[str] = []
    lines.append(f"### {index}. `{case.get('company_id')}` / `{case.get('top_action')}`")
    lines.append("")
    lines.append(f"- Run: `{case.get('run_id')}`")
    lines.append(f"- Parameter summary: {case.get('parameter_summary') or 'missing'}")
    lines.append(f"- Supported: `{historical.get('supported', False)}`")
    if historical.get("supported"):
        lines.append(f"- Cohort: `{historical.get('cohort_key')}`")
        lines.append(f"- Parameter: `{historical.get('parameter_name')}`")
        lines.append(f"- Recommended bucket: `{historical.get('recommended_bucket')}`")
        lines.append(f"- Best historical bucket: `{historical.get('best_bucket')}`")
        lines.append(f"- Alignment score: `{historical.get('alignment_score', 0.0):.3f}`")
        lines.append(f"- Tuning suggestion: `{historical.get('tuning_suggestion')}`")
    else:
        lines.append(f"- Reasons: `{', '.join(historical.get('reasons', []) or ['unknown'])}`")
    lines.append("")
    return lines
