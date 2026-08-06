from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .types import ImpactDistribution, PrecedentPack


_COMPACT_PROFILE_FEATURES = [
    "state_vector_v1.size_log_revenue",
    "state_vector_v1.profitability",
    "state_vector_v1.growth",
    "state_vector_v1.net_obligation_burden",
    "state_vector_v1.liquidity_flexibility",
    "state_vector_v1.interest_coverage",
    "state_vector_v1.valuation_multiple",
    "state_vector_v1.cash_generation",
    "state_vector_v1.market_stress",
    "state_vector_v1.market_access",
    "state_vector_v1.rates_level",
    "state_vector_v1.credit_spread",
]


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if np.isnan(out):
        return None
    return out


def _safe_log10(value: Optional[float]) -> Optional[float]:
    if value is None or value <= 0:
        return None
    return float(np.log10(value))


def _weighted_average(parts: List[Tuple[Optional[float], float]]) -> Optional[float]:
    numer = 0.0
    denom = 0.0
    for value, weight in parts:
        if value is None:
            continue
        numer += float(weight) * float(value)
        denom += float(weight)
    if denom <= 0:
        return None
    return numer / denom


def _state_vector_baseline_value(baseline: Dict[str, Any], key: str) -> Optional[float]:
    if key == "state_vector_v1.size_log_revenue":
        return _safe_float(baseline.get(key)) or _safe_log10(_safe_float(baseline.get("revenue_ttm")))
    if key == "state_vector_v1.profitability":
        return _safe_float(baseline.get(key)) or _safe_float(baseline.get("ebitda_margin"))
    if key == "state_vector_v1.growth":
        return (
            _safe_float(baseline.get(key))
            or _safe_float(baseline.get("revenue_yoy_last_q"))
            or _safe_float(baseline.get("revenue_yoy"))
        )
    if key == "state_vector_v1.gross_obligation_burden":
        return (
            _safe_float(baseline.get(key))
            or _safe_float(baseline.get("gross_leverage_including_retirement"))
            or _safe_float(baseline.get("gross_obligation_burden"))
        )
    if key == "state_vector_v1.net_obligation_burden":
        return (
            _safe_float(baseline.get(key))
            or _safe_float(baseline.get("net_leverage_including_retirement"))
            or _safe_float(baseline.get("leverage_net_debt_ebitda"))
        )
    if key == "state_vector_v1.liquidity_flexibility":
        direct = _safe_float(baseline.get(key))
        if direct is not None:
            return direct
        numer = (
            _safe_float(baseline.get("available_liquidity_normalized"))
            or _safe_float(baseline.get("available_for_actions"))
            or _safe_float(baseline.get("cash"))
        )
        denom = (
            _safe_float(baseline.get("debt_due_next_24m"))
            or _safe_float(baseline.get("debt_due_0_12m"))
            or _safe_float(baseline.get("current_debt"))
        )
        if numer is None or denom is None or denom <= 0:
            return None
        return numer / denom
    if key == "state_vector_v1.interest_coverage":
        direct = _safe_float(baseline.get(key)) or _safe_float(baseline.get("interest_coverage"))
        if direct is not None:
            return direct
        ebitda = _safe_float(baseline.get("ebitda_ttm")) or _safe_float(baseline.get("ebitda_ltm"))
        interest = _safe_float(baseline.get("interest_expense"))
        if ebitda is None or interest is None or interest <= 0:
            return None
        return ebitda / interest
    if key == "state_vector_v1.valuation_multiple":
        return (
            _safe_float(baseline.get(key))
            or _safe_float(baseline.get("ev_ebitda"))
            or _safe_float(baseline.get("base_ev_ebitda"))
        )
    if key == "state_vector_v1.cash_generation":
        return (
            _safe_float(baseline.get(key))
            or _safe_float(baseline.get("fcf_yield"))
            or _safe_float(baseline.get("fcf_margin"))
        )
    if key == "state_vector_v1.market_stress":
        direct = _safe_float(baseline.get(key))
        if direct is not None:
            return direct
        vol = _safe_float(baseline.get("volatility_90d"))
        draw = _safe_float(baseline.get("drawdown_90d"))
        return _weighted_average([(vol, 0.6), (abs(draw) if draw is not None else None, 0.4)])
    if key == "state_vector_v1.market_access":
        direct = _safe_float(baseline.get(key))
        if direct is not None:
            return direct
        spread = _safe_float(baseline.get("credit_spread_level"))
        spread_access = None if spread is None else max(0.0, min(1.0, 1.0 - spread / 0.08))
        return _weighted_average(
            [
                (_safe_float(baseline.get("credit_window_proxy")), 0.4),
                (_safe_float(baseline.get("equity_window_proxy")), 0.4),
                (spread_access, 0.2),
            ]
        )
    if key == "state_vector_v1.rates_level":
        return (
            _safe_float(baseline.get(key))
            or _safe_float(baseline.get("fed_funds_effective"))
            or _safe_float(baseline.get("macro_rate_10y"))
        )
    if key == "state_vector_v1.credit_spread":
        return (
            _safe_float(baseline.get(key))
            or _safe_float(baseline.get("hy_oas"))
            or _safe_float(baseline.get("macro_hy_oas"))
        )
    return None


def _first_present_column(df: pd.DataFrame, columns: List[str]) -> Optional[str]:
    for col in columns:
        if col in df.columns and _safe_numeric(df[col]).notna().any():
            return col
    return None


def learn_feature_weights(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> pd.Series:
    """
    Learn feature weights using absolute correlation with the target outcome.
    Falls back to uniform weights when correlations are undefined.
    """
    weights = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        x = _safe_numeric(df[col])
        y = _safe_numeric(df[target_col]) if target_col in df.columns else None
        if y is None or y.dropna().empty or x.dropna().empty:
            weights[col] = 1.0
            continue
        corr = x.corr(y)
        if corr is None or np.isnan(corr):
            weights[col] = 1.0
        else:
            weights[col] = abs(corr)
    if not weights:
        return pd.Series(dtype=float)
    w = pd.Series(weights)
    if w.sum() == 0:
        w[:] = 1.0
    return w / w.sum()


def weighted_mahalanobis_distances(
    X: np.ndarray,
    x: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Compute weighted Mahalanobis distances.
    """
    if X.size == 0:
        return np.array([])
    w = np.diag(weights)
    cov = np.cov(X, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[cov]])
    cov_w = w @ cov @ w
    inv_cov = np.linalg.pinv(cov_w)
    diff = X - x
    d = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    return np.sqrt(np.maximum(d, 0))


def bucket_filter(series: pd.Series, value: float, window: int = 1) -> pd.Series:
    if series.dropna().empty or value is None or np.isnan(value):
        return pd.Series([True] * len(series), index=series.index)
    quantiles = series.quantile([0.2, 0.4, 0.6, 0.8]).values
    bucket = np.digitize([value], quantiles)[0]
    low = max(0, bucket - window)
    high = min(4, bucket + window)
    return series.apply(lambda v: low <= np.digitize([v], quantiles)[0] <= high if pd.notna(v) else False)


def stage1_filter(
    df: pd.DataFrame,
    baseline: Dict[str, Any],
    config: Dict[str, Any],
) -> pd.DataFrame:
    out = df.copy()
    stage1 = config.get("stage1", {})

    sector_mode = stage1.get("sector_mode", "weighted")
    if sector_mode == "strict" and "sector" in df.columns and baseline.get("sector"):
        out = out[out["sector"] == baseline["sector"]]

    size_window = stage1.get("size_decile_window", 1)
    size_col = _first_present_column(out, ["state_vector_v1.size_log_revenue", "base_market_cap"])
    size_baseline = _state_vector_baseline_value(baseline, "state_vector_v1.size_log_revenue")
    if size_col == "base_market_cap":
        size_baseline = _safe_float(baseline.get("market_cap"))
    if size_col and size_baseline is not None:
        mask = bucket_filter(out[size_col], size_baseline, size_window)
        out = out[mask]

    lev_window = stage1.get("leverage_band_window", 1)
    leverage_col = _first_present_column(
        out,
        [
            "state_vector_v1.net_obligation_burden",
            "state_vector_v1.gross_obligation_burden",
            "base_leverage",
        ],
    )
    leverage_baseline = (
        _state_vector_baseline_value(baseline, "state_vector_v1.net_obligation_burden")
        or _state_vector_baseline_value(baseline, "state_vector_v1.gross_obligation_burden")
        or _safe_float(baseline.get("leverage_net_debt_ebitda"))
    )
    if leverage_col and leverage_baseline is not None:
        mask = bucket_filter(out[leverage_col], leverage_baseline, lev_window)
        out = out[mask]

    margin_window = stage1.get("margin_band_window", 1)
    margin_col = _first_present_column(out, ["state_vector_v1.profitability", "base_margin"])
    margin_baseline = _state_vector_baseline_value(baseline, "state_vector_v1.profitability") or _safe_float(baseline.get("ebitda_margin"))
    if margin_col and margin_baseline is not None:
        mask = bucket_filter(out[margin_col], margin_baseline, margin_window)
        out = out[mask]

    return out


def _baseline_value(baseline: Dict[str, Any], col: str) -> Optional[float]:
    if col.startswith("state_vector_v1."):
        return _state_vector_baseline_value(baseline, col)
    key = col[5:] if col.startswith("base_") else col
    val = baseline.get(key)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except Exception:
        return None


def _extract_features(
    df: pd.DataFrame,
    requested: List[str],
    baseline: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[float]]:
    cols: List[str] = []
    vals: List[float] = []
    for key in requested:
        if key not in df.columns:
            continue
        if baseline is None:
            cols.append(key)
            vals.append(0.0)
            continue
        base_val = _baseline_value(baseline, key)
        if base_val is None:
            continue
        cols.append(key)
        vals.append(base_val)
    return cols, vals


def match_precedents(
    df: pd.DataFrame,
    change_vector: Dict[str, float],
    baseline: Dict[str, Any],
    config: Dict[str, Any],
    target_col: str,
    top_n: int = 50,
) -> Tuple[pd.DataFrame, pd.Series]:
    if df.empty:
        return df, pd.Series(dtype=float)

    filtered = stage1_filter(df, baseline, config)
    min_candidates = config.get("stage1", {}).get("min_candidates")
    if filtered.empty or (min_candidates and len(filtered) < min_candidates):
        filtered = df

    change_cols = []
    change_vals = []
    for key, val in change_vector.items():
        for candidate_col in (f"delta_{key}", f"change_{key}", key):
            if candidate_col in filtered.columns:
                change_cols.append(candidate_col)
                change_vals.append(val)
                break

    profile_features = config.get("profile_features") or list(_COMPACT_PROFILE_FEATURES)
    profile_cols, profile_vals = _extract_features(filtered, profile_features, baseline=baseline)

    change_weights = pd.Series(dtype=float)
    profile_weights = pd.Series(dtype=float)
    change_dist = None
    profile_dist = None

    if change_cols:
        X_change = (
            filtered[change_cols]
            .apply(_safe_numeric)
            .fillna(filtered[change_cols].median())
            .to_numpy()
        )
        x_change = np.array(change_vals, dtype=float)
        change_weights = learn_feature_weights(filtered, change_cols, target_col)
        if change_weights.empty:
            change_weights = pd.Series([1.0] * len(change_cols), index=change_cols)
        change_weights = change_weights.reindex(change_cols).fillna(1.0)
        change_dist = weighted_mahalanobis_distances(X_change, x_change, change_weights.to_numpy())

    if profile_cols:
        X_profile = (
            filtered[profile_cols]
            .apply(_safe_numeric)
            .fillna(filtered[profile_cols].median())
            .to_numpy()
        )
        x_profile = np.array(profile_vals, dtype=float)
        profile_weights = learn_feature_weights(filtered, profile_cols, target_col)
        if profile_weights.empty:
            profile_weights = pd.Series([1.0] * len(profile_cols), index=profile_cols)
        profile_weights = profile_weights.reindex(profile_cols).fillna(1.0)
        profile_dist = weighted_mahalanobis_distances(X_profile, x_profile, profile_weights.to_numpy())

    if change_dist is None and profile_dist is None:
        return filtered.head(top_n), pd.Series(dtype=float)

    stage2 = config.get("stage2", {})
    change_weight = float(stage2.get("change_weight", 0.7))
    profile_weight = float(stage2.get("profile_weight", 0.3))
    total_weight = change_weight + profile_weight
    if total_weight == 0:
        total_weight = 1.0
    change_weight /= total_weight
    profile_weight /= total_weight

    if change_dist is None:
        distances = profile_dist
    elif profile_dist is None:
        distances = change_dist
    else:
        distances = (change_weight * change_dist) + (profile_weight * profile_dist)

    filtered = filtered.copy()
    filtered["distance"] = distances
    filtered = filtered.sort_values("distance").head(top_n)
    weights = pd.concat(
        [
            change_weights.rename(lambda c: f"change::{c}"),
            profile_weights.rename(lambda c: f"profile::{c}"),
        ]
    )
    return filtered, weights


def summarize_outcomes(df: pd.DataFrame, outcome_cols: List[str]) -> List[ImpactDistribution]:
    distributions = []
    for col in outcome_cols:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            distributions.append(ImpactDistribution(metric=col, horizon_months=0, p25=None, p50=None, p75=None, n=0))
            continue
        p25, p50, p75 = np.percentile(values, [25, 50, 75])
        distributions.append(
            ImpactDistribution(
                metric=col,
                horizon_months=0,
                p25=float(p25),
                p50=float(p50),
                p75=float(p75),
                n=len(values),
            )
        )
    return distributions


def build_precedent_pack(
    df: pd.DataFrame,
    change_vector: Dict[str, float],
    baseline: Dict[str, Any],
    config: Dict[str, Any],
    target_col: str,
    outcome_cols: List[str],
    top_n: int = 50,
) -> PrecedentPack:
    matches, weights = match_precedents(df, change_vector, baseline, config, target_col, top_n=top_n)
    distributions = summarize_outcomes(matches, outcome_cols)
    mismatch = {}
    if matches.empty:
        mismatch["reason"] = "no_matches"
    return PrecedentPack(
        matches=matches.to_dict(orient="records"),
        distributions=distributions,
        mismatch_diagnostics={"weights": weights.to_dict(), **mismatch},
    )
