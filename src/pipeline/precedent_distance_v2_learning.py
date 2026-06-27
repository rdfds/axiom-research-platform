from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .precedent_brain import (
    _STATE_VECTOR_CORE_CRITICAL_FEATURES,
    _STATE_VECTOR_GROUPS,
    _STATE_VECTOR_MATCHING_COLS,
    _STATE_VECTOR_V2_DEFAULT_BLEND_WEIGHTS,
    _STATE_VECTOR_V2_DEFAULT_FEATURE_RELATIVE_WEIGHTS,
    _STATE_VECTOR_V2_DEFAULT_GATES,
    _STATE_VECTOR_V2_DEFAULT_GROUP_MULTIPLIERS,
    _STATE_VECTOR_V2_DEFAULT_GROUP_WEIGHTS,
    _STATE_VECTOR_V2_DEFAULT_PENALTIES,
    _WEIGHTED_DISTANCE_V2_VERSION,
)


def load_precedent_distance_v2_objective(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("Objective config must be a JSON object")
    return payload


def _scope_multipliers(scope_key: str) -> Dict[str, float]:
    scope = str(scope_key or "").strip().lower()
    if scope.startswith("capital_return.dividend"):
        return dict(_STATE_VECTOR_V2_DEFAULT_GROUP_MULTIPLIERS.get("capital_return.dividend", {}))
    if scope in {
        "capital_return.open_market_buyback",
        "capital_return.accelerated_share_repurchase",
        "capital_return.buyback",
    }:
        return dict(_STATE_VECTOR_V2_DEFAULT_GROUP_MULTIPLIERS.get("capital_return.buyback", {}))
    return dict(_STATE_VECTOR_V2_DEFAULT_GROUP_MULTIPLIERS.get(scope, {}))


def default_scope_configuration(scope_key: str) -> Dict[str, Any]:
    scope = str(scope_key or "").strip().lower()
    group_weights = dict(_STATE_VECTOR_V2_DEFAULT_GROUP_WEIGHTS)
    for group_name, multiplier in _scope_multipliers(scope).items():
        group_weights[group_name] = float(group_weights.get(group_name, 1.0)) * float(multiplier)
    feature_relative_weights = dict(_STATE_VECTOR_V2_DEFAULT_FEATURE_RELATIVE_WEIGHTS)
    gates = dict(_STATE_VECTOR_V2_DEFAULT_GATES)
    penalties = dict(_STATE_VECTOR_V2_DEFAULT_PENALTIES)
    critical = set(_STATE_VECTOR_CORE_CRITICAL_FEATURES)
    subtype_text = scope.split(".", 1)[1] if "." in scope else scope
    if scope.startswith("capital_structure.") or "debt" in scope or "refinanc" in subtype_text:
        gates["max_size_gap"] = 1.30
        penalties["sector_penalty_weight"] = 0.22
        critical.update({"state_vector_v1.market_access", "state_vector_v1.credit_spread"})
    elif scope.startswith("capital_return.dividend") or subtype_text.startswith("dividend"):
        gates["max_size_gap"] = 1.05
        gates["soft_burden_gap"] = 1.10
        critical.update({"state_vector_v1.cash_generation"})
    elif "buyback" in scope or "repurchase" in scope or "buyback" in subtype_text:
        feature_relative_weights["state_vector_v1.valuation_multiple"] = 1.35
        feature_relative_weights["state_vector_v1.cash_generation"] = 1.20
        gates["max_size_gap"] = 1.15
    return {
        "scope_key": scope,
        "group_weights": group_weights,
        "feature_relative_weights": feature_relative_weights,
        "gates": gates,
        "penalties": penalties,
        "blend_weights": dict(_STATE_VECTOR_V2_DEFAULT_BLEND_WEIGHTS),
        "critical_features": list(critical),
        "use_in_runtime": True,
    }


def build_precedent_distance_v2_payload(
    *,
    scopes: Dict[str, Dict[str, Any]],
    objective_config: Optional[Dict[str, Any]] = None,
    benchmark_key: str = "",
    notes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "version": "precedent_distance_weights_v2",
        "state_distance_version": _WEIGHTED_DISTANCE_V2_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_key": str(benchmark_key or ""),
        "objective": objective_config or {},
        "notes": notes or {},
        "scopes": scopes,
    }


def extract_report_aggregate(report: Dict[str, Any]) -> Dict[str, Any]:
    aggregate = dict(report.get("aggregate", {}) or {})
    aggregate["runs_analyzed"] = int(report.get("runs_analyzed", 0) or 0)
    aggregate["supported_case_count"] = int(report.get("supported_case_count", 0) or 0)
    aggregate["case_count_requested"] = int(report.get("case_count_requested", 0) or 0)
    return aggregate


def is_better_report_aggregate(
    candidate: Dict[str, Any],
    incumbent: Optional[Dict[str, Any]],
    objective_config: Dict[str, Any],
    *,
    tolerance: float = 1e-9,
) -> bool:
    if not incumbent:
        return True
    objective_order = list(objective_config.get("objective_order", []) or [])
    for spec in objective_order:
        metric = str(spec.get("metric") or "").strip()
        direction = str(spec.get("direction") or "").strip().lower()
        if not metric or direction not in {"minimize", "maximize"}:
            continue
        cand_value = candidate.get(metric)
        inc_value = incumbent.get(metric)
        if cand_value is None and inc_value is None:
            continue
        if cand_value is None:
            return False
        if inc_value is None:
            return True
        cand_value = float(cand_value)
        inc_value = float(inc_value)
        if abs(cand_value - inc_value) <= float(tolerance):
            continue
        if direction == "minimize":
            return cand_value < inc_value
        return cand_value > inc_value
    return False


def _parameter_grid(objective_config: Dict[str, Any]) -> List[Tuple[Tuple[str, ...], List[float]]]:
    def _dedupe(values: Sequence[float]) -> List[float]:
        seen: List[float] = []
        for value in values:
            float_value = float(value)
            if any(abs(float_value - existing) <= 1e-12 for existing in seen):
                continue
            seen.append(float_value)
        return seen

    search_space = dict(objective_config.get("search_space", {}) or {})
    group_names = list(search_space.get("group_weights", []) or list(_STATE_VECTOR_GROUPS.keys()))
    raw_group_grid = list(search_space.get("group_weight_grid_values", []) or [])
    group_grid_values = _dedupe([float(value) for value in raw_group_grid]) if raw_group_grid else [0.60, 0.80, 1.00, 1.20, 1.40, 1.70]
    grids: List[Tuple[Tuple[str, ...], List[float]]] = []
    if bool(search_space.get("optimize_group_weights", True)):
        for group_name in group_names:
            grids.append((("group_weights", str(group_name)), list(group_grid_values)))
    if bool(search_space.get("optimize_feature_relative_weights", True)):
        within_group = dict(search_space.get("within_group_relative_weights", {}) or {})
        rel_min = float(within_group.get("min", 0.50) or 0.50)
        rel_max = float(within_group.get("max", 2.00) or 2.00)
        raw_feature_grid = list(search_space.get("feature_relative_weight_grid_values", []) or [])
        if raw_feature_grid:
            rel_values = _dedupe([float(value) for value in raw_feature_grid if rel_min <= float(value) <= rel_max])
        else:
            rel_values = _dedupe([value for value in [rel_min, 0.80, 1.00, 1.20, 1.50, rel_max] if rel_min <= value <= rel_max])
        requested_features = list(search_space.get("feature_relative_weight_features", []) or [])
        candidate_features = requested_features if requested_features else list(_STATE_VECTOR_MATCHING_COLS)
        seen_features: List[str] = []
        for feature_name in candidate_features:
            if feature_name in seen_features:
                continue
            seen_features.append(feature_name)
            grids.append((("feature_relative_weights", feature_name), rel_values))
    if bool(search_space.get("optimize_gates", True)):
        grids.extend(
            [
                (("gates", "min_weighted_coverage"), [0.70, 0.75, 0.80, 0.85]),
                (("gates", "min_critical_coverage"), [0.70, 0.80, 0.90]),
                (("gates", "max_size_gap"), [1.00, 1.15, 1.30, 1.50]),
            ]
        )
    if bool(search_space.get("optimize_penalties", True)):
        grids.extend(
            [
                (("penalties", "missing_penalty_weight"), [0.20, 0.45, 0.70]),
                (("penalties", "critical_missing_penalty_weight"), [0.40, 0.90, 1.40]),
                (("penalties", "sector_penalty_weight"), [0.00, 0.15, 0.30, 0.45, 0.60]),
                (("penalties", "regime_rate_penalty_weight"), [0.00, 0.20, 0.40, 0.60]),
                (("penalties", "regime_credit_penalty_weight"), [0.00, 0.25, 0.45, 0.70]),
            ]
        )
    if bool(search_space.get("optimize_blend_weights", True)):
        grids.extend(
            [
                (("blend_weights", "state"), [0.48, 0.58, 0.68]),
                (("blend_weights", "regime"), [0.06, 0.12, 0.18]),
                (("blend_weights", "sector"), [0.06, 0.10, 0.14]),
            ]
        )
    return grids


def _get_nested(config: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _set_nested(config: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    cursor = config
    for key in path[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[key] = next_value
        cursor = next_value
    cursor[path[-1]] = value


def coordinate_search_scope_configuration(
    *,
    scope_key: str,
    objective_config: Dict[str, Any],
    evaluate_scope_config: Callable[[Dict[str, Any]], Dict[str, Any]],
    max_rounds: int = 1,
) -> Dict[str, Any]:
    best_config = default_scope_configuration(scope_key)
    best_report = evaluate_scope_config(best_config)
    best_metrics = extract_report_aggregate(best_report)
    history: List[Dict[str, Any]] = [
        {
            "stage": "seed",
            "config": copy.deepcopy(best_config),
            "aggregate": dict(best_metrics),
        }
    ]
    parameter_grid = _parameter_grid(objective_config)

    for round_index in range(max(1, int(max_rounds))):
        improved = False
        for path, grid_values in parameter_grid:
            current_value = _get_nested(best_config, path)
            path_best_config: Optional[Dict[str, Any]] = None
            path_best_report: Optional[Dict[str, Any]] = None
            path_best_metrics: Optional[Dict[str, Any]] = None
            for grid_value in grid_values:
                if current_value is not None and abs(float(current_value) - float(grid_value)) <= 1e-12:
                    continue
                candidate_config = copy.deepcopy(best_config)
                _set_nested(candidate_config, path, float(grid_value))
                candidate_report = evaluate_scope_config(candidate_config)
                candidate_metrics = extract_report_aggregate(candidate_report)
                history.append(
                    {
                        "stage": f"round_{round_index + 1}",
                        "parameter_path": ".".join(path),
                        "parameter_value": float(grid_value),
                        "aggregate": dict(candidate_metrics),
                    }
                )
                incumbent_metrics = path_best_metrics if path_best_metrics is not None else best_metrics
                if is_better_report_aggregate(candidate_metrics, incumbent_metrics, objective_config):
                    path_best_config = candidate_config
                    path_best_report = candidate_report
                    path_best_metrics = candidate_metrics
            if path_best_metrics is not None and is_better_report_aggregate(path_best_metrics, best_metrics, objective_config):
                best_config = path_best_config if path_best_config is not None else best_config
                best_report = path_best_report if path_best_report is not None else best_report
                best_metrics = path_best_metrics
                improved = True
        if not improved:
            break

    return {
        "scope_key": str(scope_key or "").strip().lower(),
        "best_config": best_config,
        "best_report": best_report,
        "best_aggregate": best_metrics,
        "history": history,
    }


def write_precedent_distance_v2_payload(payload: Dict[str, Any], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
