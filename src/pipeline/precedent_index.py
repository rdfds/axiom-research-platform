from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


_HORIZON_KEYS = ["horizon_1m", "horizon_6m", "horizon_12m", "horizon_24m"]
_METRIC_KEYS = [
    "valuation_multiple_change",
    "equity_return_vs_sector",
    "credit_spread_change",
    "rating_migration",
    "leverage_change",
    "fcf_change",
    "volatility_change",
]
INDEX_VERSION = "v4_calibrated_query"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_str(v: Any) -> str:
    return str(v or "").strip()


def _norm_lower(v: Any) -> str:
    return _norm_str(v).lower()


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _query_rank_score(
    *,
    precedent_confidence: float,
    sample_size: int,
    out_of_sample_flag: bool,
    source: str,
    retrieval_tier: str,
    low_precedent_coverage: bool,
    exact_support_ratio: float,
    top_similarity_mean: float,
    top_action_match_score: float,
) -> float:
    sample_factor = min(1.0, math.log1p(max(0, int(sample_size))) / math.log(51.0))
    oos_penalty = 0.82 if bool(out_of_sample_flag) else 1.0
    source_penalty = 1.0 if str(source or "") == "overall" else 0.96
    tier = str(retrieval_tier or "")
    if tier == "exact":
        tier_factor = 1.0
    elif tier == "sibling_type":
        tier_factor = 0.93
    else:
        tier_factor = 0.78
    coverage_factor = 0.88 + 0.12 * max(0.0, min(1.0, float(exact_support_ratio)))
    if bool(low_precedent_coverage) and tier == "global":
        coverage_factor *= 0.88
    similarity_factor = 0.85 + 0.15 * max(0.0, min(1.0, float(top_similarity_mean)))
    action_factor = 0.90 + 0.10 * max(0.0, min(1.0, float(top_action_match_score)))
    return round(
        float(precedent_confidence)
        * sample_factor
        * oos_penalty
        * source_penalty
        * tier_factor
        * coverage_factor
        * similarity_factor
        * action_factor,
        6,
    )


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _norm_gvkey(v: Any) -> str:
    s = _norm_str(v)
    if not s:
        return ""
    if s.isdigit():
        return s.zfill(6)
    return s


def _sic2_to_sector_label(sic2: int) -> str:
    if 1 <= sic2 <= 9:
        return "AGRICULTURE"
    if 10 <= sic2 <= 14:
        return "ENERGY"
    if 15 <= sic2 <= 17:
        return "CONSTRUCTION"
    if 20 <= sic2 <= 39:
        return "MANUFACTURING"
    if 40 <= sic2 <= 47:
        return "TRANSPORT_COMM"
    if 48 <= sic2 <= 49:
        return "UTILITIES"
    if 50 <= sic2 <= 51:
        return "WHOLESALE"
    if 52 <= sic2 <= 59:
        return "RETAIL"
    if 60 <= sic2 <= 67:
        return "FINANCIALS"
    if 70 <= sic2 <= 79:
        return "SERVICES"
    if 80 <= sic2 <= 89:
        return "HEALTH_EDU_SERVICES"
    if 90 <= sic2 <= 99:
        return "PUBLIC_OTHER"
    return "UNKNOWN"


@lru_cache(maxsize=1)
def _load_gvkey_sector_map() -> Dict[str, str]:
    path = os.environ.get("PRECEDENT_SECTOR_MAP_PATH", "").strip()
    if path:
        p = Path(path)
    else:
        p = _REPO_ROOT / "data" / "curated" / "bond_issuances_fisd.parquet"
    if not p.exists():
        return {}
    try:
        df = pd.read_parquet(p, columns=["gvkey", "SIC_CODE"])
    except Exception:
        return {}
    if df.empty:
        return {}
    df["gvkey"] = df["gvkey"].map(_norm_gvkey)
    df["sic"] = pd.to_numeric(df["SIC_CODE"], errors="coerce")
    df = df[df["gvkey"] != ""]
    df = df[df["sic"].notna()]
    if df.empty:
        return {}
    df["sic2"] = (df["sic"] // 100).astype("Int64")
    df = df[df["sic2"].notna()]
    if df.empty:
        return {}
    # Mode SIC2 per gvkey.
    mode = (
        df.groupby(["gvkey", "sic2"], as_index=False)
        .size()
        .sort_values(["gvkey", "size"], ascending=[True, False])
        .drop_duplicates(subset=["gvkey"], keep="first")
    )
    out: Dict[str, str] = {}
    for _, row in mode.iterrows():
        gv = _norm_gvkey(row.get("gvkey"))
        try:
            sic2 = int(row.get("sic2"))
        except Exception:
            continue
        out[gv] = _sic2_to_sector_label(sic2)
    return out


def _candidate_sector(candidate: Dict[str, Any], pack: Dict[str, Any]) -> str:
    direct = _norm_str(
        candidate.get("sector")
        or candidate.get("gics_sector")
        or candidate.get("sector_name")
    )
    if direct:
        return direct
    cohorts = pack.get("cohorts") if isinstance(pack.get("cohorts"), list) else pack.get("retrieved_cohorts", [])
    sectors: List[str] = []
    cohort_company_ids: List[str] = []
    for c in cohorts or []:
        if not isinstance(c, dict):
            continue
        cid = _norm_gvkey(c.get("company_id"))
        if cid:
            cohort_company_ids.append(cid)
        ksf = c.get("key_state_features", {}) if isinstance(c.get("key_state_features"), dict) else {}
        s = _norm_str(ksf.get("base_sector"))
        if s:
            sectors.append(s)
    if sectors:
        mode = Counter(sectors).most_common(1)
        if mode:
            return mode[0][0]

    # Fallback: infer sector from cohort gvkeys via SIC->sector mapping.
    sec_map = _load_gvkey_sector_map()
    mapped = [sec_map.get(g, "") for g in cohort_company_ids]
    mapped = [m for m in mapped if m and m != "UNKNOWN"]
    if mapped:
        mode = Counter(mapped).most_common(1)
        if mode:
            return mode[0][0]
    return ""


def _pack_outcomes(pack: Dict[str, Any]) -> Dict[str, Any]:
    d = pack.get("outcome_distributions")
    if isinstance(d, dict):
        return d
    d = pack.get("distributions")
    if isinstance(d, dict):
        return d
    return {}


def _iter_distribution_rows(
    *,
    run_id: str,
    candidate: Dict[str, Any],
    pack: Dict[str, Any],
    regime_label: str,
    outcome_distributions: Dict[str, Any],
    source: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    candidate_id = _norm_str(candidate.get("candidate_id"))
    action_type = _norm_str(candidate.get("action_type"))
    action_id = _norm_str(candidate.get("action_id"))
    action_subtype = _norm_str(candidate.get("action_subtype"))
    sector = _candidate_sector(candidate, pack)
    confidence = _to_float(pack.get("precedent_confidence"))
    if confidence is None:
        confidence = _to_float(pack.get("calibration_confidence")) or 0.0
    mismatch = pack.get("mismatch_diagnostics", {}) if isinstance(pack.get("mismatch_diagnostics"), dict) else {}
    out_of_sample = bool(mismatch.get("out_of_sample_flag", False))
    retrieval_tier = _norm_str(mismatch.get("retrieval_tier"))
    low_precedent_coverage = bool(mismatch.get("low_precedent_coverage", False))
    exact_support_ratio = float(_to_float(mismatch.get("exact_support_ratio")) or 0.0)
    top_similarity_mean = float(_to_float(mismatch.get("top_similarity_mean")) or 0.0)
    top_action_match_score = float(_to_float(mismatch.get("top_action_match_score")) or 0.0)

    for horizon_key in _HORIZON_KEYS:
        metric_set = outcome_distributions.get(horizon_key, {}) if isinstance(outcome_distributions, dict) else {}
        if not isinstance(metric_set, dict):
            continue
        for metric in _METRIC_KEYS:
            dist = metric_set.get(metric, {})
            if not isinstance(dist, dict):
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "candidate_id": candidate_id,
                    "action_type": action_type,
                    "action_subtype": action_subtype,
                    "action_id": action_id,
                    "sector": sector,
                    "regime_label": _norm_str(regime_label),
                    "time_horizon": horizon_key.replace("horizon_", ""),
                    "metric": metric,
                    "mean": _to_float(dist.get("mean")),
                    "median": _to_float(dist.get("median")),
                    "p10": _to_float(dist.get("p10")),
                    "p25": _to_float(dist.get("p25")),
                    "p75": _to_float(dist.get("p75")),
                    "p90": _to_float(dist.get("p90")),
                    "sample_size": int(dist.get("sample_size", 0) or 0),
                    "precedent_confidence": float(confidence),
                    "out_of_sample_flag": out_of_sample,
                    "low_precedent_coverage": low_precedent_coverage,
                    "retrieval_tier": retrieval_tier,
                    "exact_support_ratio": exact_support_ratio,
                    "top_similarity_mean": top_similarity_mean,
                    "top_action_match_score": top_action_match_score,
                    "source": source,
                    "query_score": _query_rank_score(
                        precedent_confidence=float(confidence),
                        sample_size=int(dist.get("sample_size", 0) or 0),
                        out_of_sample_flag=out_of_sample,
                        source=source,
                        retrieval_tier=retrieval_tier,
                        low_precedent_coverage=low_precedent_coverage,
                        exact_support_ratio=exact_support_ratio,
                        top_similarity_mean=top_similarity_mean,
                        top_action_match_score=top_action_match_score,
                    ),
                }
            )
    return rows


def build_precedent_index(run_id: str, precedent_matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    for item in precedent_matches or []:
        candidate = item.get("candidate", {}) if isinstance(item, dict) else {}
        pack = item.get("precedent_pack", {}) if isinstance(item, dict) else {}
        if not isinstance(candidate, dict) or not isinstance(pack, dict):
            continue

        mismatch = pack.get("mismatch_diagnostics", {}) if isinstance(pack.get("mismatch_diagnostics"), dict) else {}
        conf = _to_float(pack.get("precedent_confidence"))
        if conf is None:
            conf = _to_float(pack.get("calibration_confidence")) or 0.0

        candidate_rows.append(
            {
                "run_id": _norm_str(run_id),
                "candidate_id": _norm_str(candidate.get("candidate_id")),
                "action_type": _norm_str(candidate.get("action_type")),
                "action_subtype": _norm_str(candidate.get("action_subtype")),
                "action_id": _norm_str(candidate.get("action_id")),
                "sector": _candidate_sector(candidate, pack),
                "precedent_confidence": float(conf),
                "out_of_sample_flag": bool(mismatch.get("out_of_sample_flag", False)),
                "cohort_size": int(mismatch.get("cohort_size", 0) or 0),
                "low_precedent_coverage": bool(mismatch.get("low_precedent_coverage", False)),
                "retrieval_tier": _norm_str(mismatch.get("retrieval_tier")),
                "exact_match_count": int(mismatch.get("exact_match_count", 0) or 0),
                "minimum_exact_support": int(mismatch.get("minimum_exact_support", 0) or 0),
                "exact_support_ratio": float(_to_float(mismatch.get("exact_support_ratio")) or 0.0),
                "top_similarity_mean": float(_to_float(mismatch.get("top_similarity_mean")) or 0.0),
                "top_action_match_score": float(_to_float(mismatch.get("top_action_match_score")) or 0.0),
            }
        )

        overall = _pack_outcomes(pack)
        rows.extend(
            _iter_distribution_rows(
                run_id=_norm_str(run_id),
                candidate=candidate,
                pack=pack,
                regime_label="all",
                outcome_distributions=overall,
                source="overall",
            )
        )
        for rs in pack.get("regime_splits", []) if isinstance(pack.get("regime_splits"), list) else []:
            if not isinstance(rs, dict):
                continue
            rows.extend(
                _iter_distribution_rows(
                    run_id=_norm_str(run_id),
                    candidate=candidate,
                    pack=pack,
                    regime_label=_norm_str(rs.get("regime_label")),
                    outcome_distributions=rs.get("outcome_distributions", {}) if isinstance(rs.get("outcome_distributions"), dict) else {},
                    source="regime_split",
                )
            )

    return {
        "run_id": _norm_str(run_id),
        "index_version": INDEX_VERSION,
        "generated_at": _now_iso(),
        "candidate_rows": candidate_rows,
        "distribution_rows": rows,
        "counts": {
            "candidates": len(candidate_rows),
            "distribution_rows": len(rows),
        },
    }


def query_precedent_index(
    index: Dict[str, Any],
    *,
    action_type: Optional[str] = None,
    action_id: Optional[str] = None,
    regime: Optional[str] = None,
    sector: Optional[str] = None,
    time_horizon: Optional[str] = None,
    min_sample_size: int = 0,
    min_precedent_confidence: float = 0.0,
    exclude_out_of_sample: bool = False,
    limit: int = 200,
) -> Dict[str, Any]:
    rows = index.get("distribution_rows", []) if isinstance(index.get("distribution_rows"), list) else []
    out = []
    f_action_type = _norm_lower(action_type)
    f_action_id = _norm_lower(action_id)
    f_regime = _norm_lower(regime)
    f_sector = _norm_lower(sector)
    f_horizon = _norm_lower(time_horizon)
    min_n = max(0, int(min_sample_size))
    min_conf = max(0.0, float(min_precedent_confidence or 0.0))
    excl_oos = bool(exclude_out_of_sample)

    for row in rows:
        if not isinstance(row, dict):
            continue
        if f_action_type and _norm_lower(row.get("action_type")) != f_action_type:
            continue
        if f_action_id and _norm_lower(row.get("action_id")) != f_action_id:
            continue
        if f_regime and _norm_lower(row.get("regime_label")) != f_regime:
            continue
        if f_sector and _norm_lower(row.get("sector")) != f_sector:
            continue
        if f_horizon and _norm_lower(row.get("time_horizon")) != f_horizon:
            continue
        if int(row.get("sample_size", 0) or 0) < min_n:
            continue
        if float(_to_float(row.get("precedent_confidence")) or 0.0) < min_conf:
            continue
        if excl_oos and bool(row.get("out_of_sample_flag", False)):
            continue
        out.append(row)
    out.sort(
        key=lambda row: (
            -float(_to_float(row.get("query_score")) or 0.0),
            -float(_to_float(row.get("precedent_confidence")) or 0.0),
            -int(row.get("sample_size", 0) or 0),
            _norm_str(row.get("action_id")),
            _norm_str(row.get("regime_label")),
            _norm_str(row.get("time_horizon")),
            _norm_str(row.get("metric")),
        )
    )
    out = out[: max(1, int(limit))]

    return {
        "run_id": _norm_str(index.get("run_id")),
        "query": {
            "action_type": action_type,
            "action_id": action_id,
            "regime": regime,
            "sector": sector,
            "time_horizon": time_horizon,
            "min_sample_size": min_n,
            "min_precedent_confidence": min_conf,
            "exclude_out_of_sample": excl_oos,
            "limit": max(1, int(limit)),
        },
        "count": len(out),
        "rows": out,
    }


__all__ = ["INDEX_VERSION", "build_precedent_index", "query_precedent_index"]
