from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import gzip
import json
import math
import os
from pathlib import Path
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..action_normalization import _refinancing_subfamily
from ..model_feature_bundle import build_model_feature_bundle
from ..causal_impact_model import action_id_to_outcomes_action_type, action_subtype_to_outcomes_subtype
from .historical_stores import (
    HistoricalCompanyStateSnapshotStore,
    HistoricalEventStore,
    HistoricalOutcomeStore,
    RegimeHistory,
    materialize_historical_frame,
)
from .latent_regime_model import latent_regime_memberships, latent_regime_similarity
from .types import (
    DistributionStats,
    FollowOnOutcome,
    ImpactDistribution,
    MetricDistributionSet,
    MismatchDiagnostics,
    OutcomeDistributions,
    PrecedentCase,
    PrecedentPack,
    RegimeDistribution,
    SimilarityScore,
    TailEvent,
    FeatureMismatch,
)


_LEGACY_EMBEDDING_COLS: Tuple[str, ...] = (
    "base_leverage",
    "base_margin",
    "base_market_cap",
    "base_revenue_ttm",
    "base_roic",
    "base_fcf_margin",
)

_STATE_VECTOR_MATCHING_COLS: Tuple[str, ...] = (
    "state_vector_v1.size_log_revenue",
    "state_vector_v1.profitability",
    "state_vector_v1.growth",
    "state_vector_v1.gross_obligation_burden",
    "state_vector_v1.net_obligation_burden",
    "state_vector_v1.liquidity_flexibility",
    "state_vector_v1.interest_coverage",
    "state_vector_v1.valuation_multiple",
    "state_vector_v1.cash_generation",
    "state_vector_v1.market_stress",
    "state_vector_v1.market_access",
    "state_vector_v1.rates_level",
    "state_vector_v1.credit_spread",
)

_WEIGHTED_DISTANCE_V1_VERSION = "weighted_distance_v1"
_WEIGHTED_DISTANCE_V2_VERSION = "weighted_distance_v2"
_SECOND_STAGE_RERANKER_FEATURES: Tuple[str, ...] = (
    "base_state_similarity",
    "unweighted_state_similarity",
    "weighted_feature_coverage",
    "critical_feature_coverage",
    "size_guardrail_similarity",
    "burden_guardrail_similarity",
    "regime_similarity",
    "parameter_similarity",
    "sector_similarity",
    "action_match_score",
    "borrower_quality_similarity",
    "financing_pressure_similarity",
    "market_regime_similarity",
    "stress_alignment_similarity",
    "compatibility_penalty_factor",
    "debt_archetype_similarity",
    "debt_style_similarity",
    "debt_archetype_gate",
)
_OUTCOME_AWARE_RERANKER_FEATURES: Tuple[str, ...] = (
    "current_similarity_score",
    "outcome_equity_score",
    "outcome_valuation_score",
    "outcome_credit_score",
    "outcome_balance_sheet_score",
    "outcome_support_score",
)
_OUTCOME_AWARE_RERANKER_GROUPS: Dict[str, Tuple[Tuple[str, bool], ...]] = {
    "outcome_equity_score": (
        ("outcome_pe_6m", True),
        ("outcome_pe_12m", True),
    ),
    "outcome_valuation_score": (
        ("outcome_ev_ebitda_6m", True),
        ("outcome_ev_ebitda_12m", True),
    ),
    "outcome_credit_score": (
        ("credit_spread_change_1m", False),
        ("credit_spread_change_6m", False),
        ("credit_spread_change_12m", False),
        ("credit_spread_change_24m", False),
        ("rating_migration_1m", True),
        ("rating_migration_6m", True),
        ("rating_migration_12m", True),
        ("rating_migration_24m", True),
    ),
    "outcome_balance_sheet_score": (
        ("leverage_delta", False),
        ("fcf_margin_delta", True),
    ),
}

_MARKET_CAP_MILLION_HEURISTIC_MAX = 5_000_000.0
_HISTORICAL_MONETARY_UNIT_SCALE = 1_000_000.0

_STATE_VECTOR_BASE_WEIGHTS: Dict[str, float] = {
    "state_vector_v1.size_log_revenue": 0.90,
    "state_vector_v1.profitability": 1.00,
    "state_vector_v1.growth": 0.80,
    "state_vector_v1.gross_obligation_burden": 1.45,
    "state_vector_v1.net_obligation_burden": 1.55,
    "state_vector_v1.liquidity_flexibility": 1.45,
    "state_vector_v1.interest_coverage": 1.30,
    "state_vector_v1.valuation_multiple": 1.00,
    "state_vector_v1.cash_generation": 1.10,
    "state_vector_v1.market_stress": 0.80,
    "state_vector_v1.market_access": 0.95,
    "state_vector_v1.rates_level": 0.45,
    "state_vector_v1.credit_spread": 0.55,
}

_STATE_VECTOR_GROUPS: Dict[str, Tuple[str, ...]] = {
    "identity": (
        "state_vector_v1.size_log_revenue",
        "state_vector_v1.profitability",
        "state_vector_v1.growth",
    ),
    "capital_structure": (
        "state_vector_v1.gross_obligation_burden",
        "state_vector_v1.net_obligation_burden",
    ),
    "liquidity": (
        "state_vector_v1.liquidity_flexibility",
        "state_vector_v1.interest_coverage",
    ),
    "valuation": (
        "state_vector_v1.valuation_multiple",
        "state_vector_v1.cash_generation",
    ),
    "market": (
        "state_vector_v1.market_stress",
        "state_vector_v1.market_access",
    ),
    "macro_regime": (
        "state_vector_v1.rates_level",
        "state_vector_v1.credit_spread",
    ),
}
_STATE_VECTOR_FEATURE_GROUP: Dict[str, str] = {
    feature_name: group_name
    for group_name, feature_names in _STATE_VECTOR_GROUPS.items()
    for feature_name in feature_names
}
_STATE_VECTOR_V2_DEFAULT_GROUP_WEIGHTS: Dict[str, float] = {
    "identity": 1.05,
    "capital_structure": 1.35,
    "liquidity": 1.30,
    "valuation": 1.10,
    "market": 0.90,
    "macro_regime": 0.60,
}
_STATE_VECTOR_V2_DEFAULT_GROUP_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "capital_return.dividend": {
        "liquidity": 1.20,
        "capital_structure": 1.10,
        "valuation": 0.90,
        "market": 0.85,
    },
    "capital_return.buyback": {
        "identity": 1.05,
        "capital_structure": 1.10,
        "liquidity": 1.05,
        "valuation": 1.35,
        "market": 0.90,
    },
    "capital_structure": {
        "capital_structure": 1.35,
        "liquidity": 1.25,
        "valuation": 0.80,
        "market": 1.15,
        "macro_regime": 1.20,
    },
    "mna": {
        "identity": 1.15,
        "valuation": 1.20,
        "market": 1.05,
    },
    "portfolio": {
        "identity": 1.10,
        "valuation": 1.05,
    },
}
_STATE_VECTOR_V2_DEFAULT_FEATURE_RELATIVE_WEIGHTS: Dict[str, float] = {
    feature_name: 1.0
    for feature_name in _STATE_VECTOR_MATCHING_COLS
}
_STATE_VECTOR_V2_DEFAULT_FEATURE_RELATIVE_WEIGHTS.update(
    {
        "state_vector_v1.profitability": 1.15,
        "state_vector_v1.valuation_multiple": 1.20,
        "state_vector_v1.market_access": 1.10,
        "state_vector_v1.credit_spread": 1.10,
    }
)
_STATE_VECTOR_V2_DEFAULT_FEATURE_TRANSFORMS: Dict[str, Dict[str, float]] = {
    # Preserve economically meaningful local differences, but compress heavy tails so
    # "very safe" stops overwhelming business-model identity.
    "state_vector_v1.growth": {"kind": "signed_asinh", "scale": 0.15},
    "state_vector_v1.gross_obligation_burden": {"kind": "signed_log1p_cap", "cap": 10.0},
    "state_vector_v1.net_obligation_burden": {"kind": "signed_log1p_cap", "cap": 10.0},
    "state_vector_v1.liquidity_flexibility": {"kind": "signed_log1p_cap", "cap": 10.0},
    "state_vector_v1.interest_coverage": {"kind": "signed_log1p_cap", "cap": 30.0},
    "state_vector_v1.valuation_multiple": {"kind": "signed_log1p_cap", "cap": 25.0},
    "state_vector_v1.cash_generation": {"kind": "signed_asinh", "scale": 0.05},
}
_STATE_VECTOR_V2_DEFAULT_FEATURE_TRANSFORM_MODE = "default"
_STATE_VECTOR_V2_DEFAULT_BLEND_WEIGHTS: Dict[str, float] = {
    "state": 0.58,
    "regime": 0.12,
    "param": 0.12,
    "sector": 0.10,
    "action": 0.08,
}
_STATE_VECTOR_V2_DEFAULT_GATES: Dict[str, float] = {
    "min_weighted_coverage": 0.75,
    "min_critical_coverage": 0.80,
    "max_size_gap": 1.15,
    "soft_size_gap": 0.35,
    "soft_burden_gap": 1.25,
}
_STATE_VECTOR_V2_DEFAULT_PENALTIES: Dict[str, float] = {
    "distance_scale": 0.55,
    "missing_penalty_weight": 0.45,
    "critical_missing_penalty_weight": 0.90,
    "size_penalty_weight": 1.15,
    "burden_penalty_weight": 0.40,
    "sector_penalty_weight": 0.60,
    "regime_rate_gap_threshold": 1.00,
    "regime_rate_penalty_weight": 0.40,
    "regime_credit_gap_threshold": 1.25,
    "regime_credit_penalty_weight": 0.45,
}

_STATE_VECTOR_CORE_CRITICAL_FEATURES: Tuple[str, ...] = (
    "state_vector_v1.size_log_revenue",
    "state_vector_v1.gross_obligation_burden",
    "state_vector_v1.net_obligation_burden",
    "state_vector_v1.liquidity_flexibility",
    "state_vector_v1.interest_coverage",
    "state_vector_v1.valuation_multiple",
)

_DEFAULT_PRECEDENT_DISTANCE_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "curated" / "precedent_distance_weights_v1.json"
)
_PRECEDENT_DISTANCE_WEIGHTS_CACHE: Dict[str, Any] = {}
_DEFAULT_PRECEDENT_DISTANCE_V2_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "curated" / "precedent_distance_weights_v2.json"
)
_PRECEDENT_DISTANCE_V2_WEIGHTS_CACHE: Dict[str, Any] = {}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_REFINITIV_TAXONOMY_REFERENCE_PATH = _REPO_ROOT / "data" / "refinitiv" / "fundamentals_all.parquet"
_SEC_TICKER_CIK_PATH = _REPO_ROOT / "data" / "mappings" / "sec_ticker_cik.parquet"
_SEC_COMPANY_TICKERS_JSON_PATH = _REPO_ROOT / "data" / "sec" / "company_tickers.json"
_SEC_SUBMISSIONS_ROOT = _REPO_ROOT / "data" / "sec" / "submissions"
_SEC_SUBMISSION_HEADER_BYTES = 16384
_SEC_TICKER_TAXONOMY_OVERRIDES: Dict[str, Tuple[str, str]] = {
    # Official SEC filing pages for these uncovered small-cap equity issuers
    # expose stable SIC labels even when the local submissions cache is absent.
    # We keep the override set intentionally tiny and use it only in the
    # explicitly opt-in identity-heuristic path.
    "BACK": ("Health Care", "Health Care Providers & Services"),
    "DGLY": ("Information Technology", "Communications Equipment"),
    "INHD": ("Materials", "Metals & Mining"),
    "MOBX": ("Information Technology", "Semiconductors & Semiconductor Equipment"),
    "UPXI": ("Financials", "Capital Markets"),
    "YCBD": ("Consumer Staples", "Personal Care Products"),
}
_SNAPSHOT_TAXONOMY_ROOT = (
    _REPO_ROOT
    / "data"
    / "company_state_snapshots"
    / "final_run_2026-02-28"
    / "keyed"
    / "as_of_date=2026-02-28"
)
_SNAPSHOT_TAXONOMY_LOOKUP_PATH = (
    _REPO_ROOT / "data" / "curated" / "snapshot_taxonomy_lookup_2026-02-28.parquet"
)
_SNAPSHOT_TAXONOMY_CATALOG_FALLBACK_PATH = (
    _REPO_ROOT / "out" / "materialized_feedback_20260405" / "company_state_snapshots_input_complete_catalog.asof_safe_v1.jsonl.gz"
)


def _load_precedent_distance_weights() -> Dict[str, Any]:
    if str(os.environ.get("PRECEDENT_DISABLE_LEARNED_DISTANCE_WEIGHTS", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return {}
    path = Path(os.environ.get("PRECEDENT_DISTANCE_WEIGHTS_PATH", _DEFAULT_PRECEDENT_DISTANCE_WEIGHTS_PATH))
    cache_key = str(path)
    try:
        stat = path.stat()
    except FileNotFoundError:
        _PRECEDENT_DISTANCE_WEIGHTS_CACHE.clear()
        return {}
    mtime = float(stat.st_mtime)
    cached = _PRECEDENT_DISTANCE_WEIGHTS_CACHE.get(cache_key)
    if isinstance(cached, dict) and float(cached.get("mtime", -1.0)) == mtime:
        payload = cached.get("payload")
        return payload if isinstance(payload, dict) else {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    _PRECEDENT_DISTANCE_WEIGHTS_CACHE.clear()
    _PRECEDENT_DISTANCE_WEIGHTS_CACHE[cache_key] = {"mtime": mtime, "payload": payload}
    return payload if isinstance(payload, dict) else {}


def _load_precedent_distance_v2_weights() -> Dict[str, Any]:
    if str(os.environ.get("PRECEDENT_DISABLE_DISTANCE_V2", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return {}
    path = Path(os.environ.get("PRECEDENT_DISTANCE_V2_WEIGHTS_PATH", _DEFAULT_PRECEDENT_DISTANCE_V2_WEIGHTS_PATH))
    cache_key = str(path)
    try:
        stat = path.stat()
    except FileNotFoundError:
        _PRECEDENT_DISTANCE_V2_WEIGHTS_CACHE.clear()
        return {}
    mtime = float(stat.st_mtime)
    cached = _PRECEDENT_DISTANCE_V2_WEIGHTS_CACHE.get(cache_key)
    if isinstance(cached, dict) and float(cached.get("mtime", -1.0)) == mtime:
        payload = cached.get("payload")
        return payload if isinstance(payload, dict) else {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    _PRECEDENT_DISTANCE_V2_WEIGHTS_CACHE.clear()
    _PRECEDENT_DISTANCE_V2_WEIGHTS_CACHE[cache_key] = {"mtime": mtime, "payload": payload}
    return payload if isinstance(payload, dict) else {}


def _extract_metric_value(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("value")
    return value


def _normalize_market_cap_to_dollars(value: Any, *, prefer_source_units: bool = False) -> Optional[float]:
    numeric = _to_float(value, None)
    if numeric is None:
        return None
    if not math.isfinite(numeric):
        return None
    if prefer_source_units and abs(float(numeric)) <= float(_MARKET_CAP_MILLION_HEURISTIC_MAX):
        return float(numeric) * float(_HISTORICAL_MONETARY_UNIT_SCALE)
    return float(numeric)


def _normalize_market_cap_series_to_dollars(
    series: pd.Series,
    *,
    prefer_source_units: bool = False,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if not prefer_source_units:
        return numeric
    needs_scale = numeric.abs().le(float(_MARKET_CAP_MILLION_HEURISTIC_MAX)) & numeric.notna()
    return numeric.where(~needs_scale, numeric * float(_HISTORICAL_MONETARY_UNIT_SCALE))


def _candidate_market_cap(candidate_features: Dict[str, Any]) -> Optional[float]:
    features = candidate_features if isinstance(candidate_features, dict) else {}
    for key, prefer_source_units in (
        ("market_cap", True),
        ("scale.market_cap", False),
        ("market.market_cap_provider_direct", False),
        ("market.market_cap", False),
        ("base_market_cap", True),
    ):
        value = _extract_metric_value(features.get(key))
        market_cap = _normalize_market_cap_to_dollars(value, prefer_source_units=prefer_source_units)
        if market_cap is not None and market_cap > 0:
            return float(market_cap)
    return None


def _normalize_ticker_key(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return text


def _normalize_instrument_root(value: Any) -> str:
    text = _normalize_ticker_key(value)
    if not text:
        return ""
    return text.split(".", 1)[0].strip()


def _decode_json_string_fragment(value: str) -> str:
    text = str(value or "")
    if not text:
        return ""
    try:
        return str(json.loads(f'"{text}"'))
    except Exception:
        return text


def _normalize_sic_description(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return re.sub(r"\s+", " ", text)


@lru_cache(maxsize=8192)
def _read_sec_submission_identity(path_text: str) -> Dict[str, Any]:
    path = Path(path_text)
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            head = handle.read(int(_SEC_SUBMISSION_HEADER_BYTES))
    except Exception:
        return {}
    text = head.decode("utf-8", errors="ignore")
    if not text:
        return {}

    def _match_string(field_name: str) -> str:
        match = re.search(rf'"{re.escape(field_name)}"\s*:\s*"((?:\\.|[^"])*)"', text)
        if not match:
            return ""
        return _decode_json_string_fragment(match.group(1)).strip()

    def _match_array(field_name: str) -> List[str]:
        match = re.search(rf'"{re.escape(field_name)}"\s*:\s*\[(.*?)\]', text, flags=re.S)
        if not match:
            return []
        raw_items = re.findall(r'"((?:\\.|[^"])*)"', match.group(1))
        out: List[str] = []
        for item in raw_items:
            ticker_text = _normalize_ticker_key(_decode_json_string_fragment(item))
            if ticker_text:
                out.append(ticker_text)
        return out

    cik_text = _match_string("cik")
    if cik_text.isdigit():
        cik_text = cik_text.zfill(10)
    elif path.stem.startswith("CIK"):
        inferred = path.stem.replace("CIK", "", 1).strip()
        cik_text = inferred.zfill(10) if inferred.isdigit() else inferred
    if not cik_text:
        return {}

    tickers = _match_array("tickers")
    primary_ticker = tickers[0] if tickers else ""
    return {
        "cik": cik_text,
        "sic": _match_string("sic"),
        "sic_description": _match_string("sicDescription"),
        "name": _match_string("name"),
        "owner_org": _match_string("ownerOrg"),
        "primary_ticker": primary_ticker,
        "tickers": tickers,
    }


@lru_cache(maxsize=1)
def _load_sec_submission_identity_index() -> Dict[str, Dict[str, Dict[str, Any]]]:
    by_ticker: Dict[str, Dict[str, Any]] = {}
    by_cik: Dict[str, Dict[str, Any]] = {}
    root = _SEC_SUBMISSIONS_ROOT
    if not root.exists():
        return {"by_ticker": by_ticker, "by_cik": by_cik}
    for path in sorted(root.glob("CIK*.json")):
        record = dict(_read_sec_submission_identity(str(path)) or {})
        cik_text = str(record.get("cik") or "").strip()
        if not cik_text:
            continue
        by_cik[cik_text] = record
        for ticker_text in list(record.get("tickers") or []):
            normalized_ticker = _normalize_ticker_key(ticker_text)
            if normalized_ticker and normalized_ticker not in by_ticker:
                by_ticker[normalized_ticker] = record
    return {"by_ticker": by_ticker, "by_cik": by_cik}


@lru_cache(maxsize=1)
def _load_sec_submission_sic_taxonomy_lookup() -> Dict[str, Tuple[str, str]]:
    index = _load_sec_submission_identity_index()
    by_cik = dict(index.get("by_cik") or {})
    if not by_cik:
        return {}
    snapshot_lookup = _load_snapshot_taxonomy_lookup()
    refinitiv_lookup = _load_refinitiv_taxonomy_lookup()
    votes: Dict[str, Counter[Tuple[str, str]]] = {}
    for record in by_cik.values():
        sector_name = ""
        subsector_name = ""
        cik_text = str(record.get("cik") or "").strip()
        if cik_text:
            sector_name, subsector_name = snapshot_lookup.get(cik_text, ("", ""))
        if not sector_name and not subsector_name:
            ticker_key = _normalize_instrument_root(record.get("primary_ticker"))
            if ticker_key:
                sector_name, subsector_name = refinitiv_lookup.get(ticker_key, ("", ""))
        sector_name = str(sector_name or "").strip()
        subsector_name = str(subsector_name or "").strip()
        if not sector_name and not subsector_name:
            continue
        keys: List[str] = []
        sic_text = str(record.get("sic") or "").strip()
        sic_desc = _normalize_sic_description(record.get("sic_description"))
        if sic_text:
            keys.append(f"sic:{sic_text}")
        if sic_desc:
            keys.append(f"sicdesc:{sic_desc}")
        for key in keys:
            votes.setdefault(key, Counter())[(sector_name, subsector_name)] += 1

    out: Dict[str, Tuple[str, str]] = {}
    for key_text, counter in votes.items():
        if not counter:
            continue
        (sector_name, subsector_name), best_count = counter.most_common(1)[0]
        total = int(sum(counter.values()))
        if str(key_text).startswith("sicdesc:"):
            if best_count < 1 or best_count != total:
                continue
        else:
            if best_count < 2 or float(best_count) / float(max(total, 1)) < 0.75:
                continue
        out[key_text] = (str(sector_name or "").strip(), str(subsector_name or "").strip())
    return out


@lru_cache(maxsize=1)
def _load_refinitiv_taxonomy_lookup() -> Dict[str, Tuple[str, str]]:
    path = _REFINITIV_TAXONOMY_REFERENCE_PATH
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(
            path,
            columns=["Instrument", "GICS Sector Name", "GICS Industry Name"],
        )
    except Exception:
        return {}
    if df.empty:
        return {}
    out: Dict[str, Tuple[str, str]] = {}
    instrument_roots = df.get("Instrument", pd.Series("", index=df.index)).astype(str).map(_normalize_instrument_root)
    sector_series = df.get("GICS Sector Name", pd.Series("", index=df.index)).fillna("").astype(str)
    industry_series = df.get("GICS Industry Name", pd.Series("", index=df.index)).fillna("").astype(str)
    for ticker_key, sector_name, industry_name in zip(instrument_roots, sector_series, industry_series):
        ticker_key = str(ticker_key or "").strip()
        if not ticker_key:
            continue
        sector_name = str(sector_name or "").strip()
        industry_name = str(industry_name or "").strip()
        if not sector_name and not industry_name:
            continue
        existing = out.get(ticker_key)
        if existing and existing[0] and existing[1]:
            continue
        out[ticker_key] = (sector_name, industry_name)
    return out


@lru_cache(maxsize=1)
def _load_sec_ticker_cik_lookup() -> Dict[str, str]:
    out: Dict[str, str] = {}
    path = _SEC_TICKER_CIK_PATH
    if path.exists():
        try:
            df = pd.read_parquet(path, columns=["ticker", "cik"])
        except Exception:
            df = pd.DataFrame()
        for ticker, cik in zip(df.get("ticker", pd.Series("", index=df.index)), df.get("cik", pd.Series("", index=df.index))):
            ticker_key = _normalize_ticker_key(ticker)
            cik_text = str(cik or "").strip()
            if ticker_key and cik_text and ticker_key not in out:
                out[ticker_key] = cik_text
    json_path = _SEC_COMPANY_TICKERS_JSON_PATH
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text())
        except Exception:
            payload = {}
        for row in (payload.values() if isinstance(payload, dict) else []):
            if not isinstance(row, dict):
                continue
            ticker_key = _normalize_ticker_key(row.get("ticker"))
            cik_text = str(row.get("cik_str") or "").strip()
            if cik_text.endswith(".0"):
                cik_text = cik_text[:-2]
            if cik_text.isdigit():
                cik_text = cik_text.zfill(10)
            if ticker_key and cik_text and ticker_key not in out:
                out[ticker_key] = cik_text
    return out


@lru_cache(maxsize=1)
def _load_sec_company_ticker_metadata_lookup() -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    json_path = _SEC_COMPANY_TICKERS_JSON_PATH
    if not json_path.exists():
        return out
    try:
        payload = json.loads(json_path.read_text())
    except Exception:
        payload = {}
    for row in (payload.values() if isinstance(payload, dict) else []):
        if not isinstance(row, dict):
            continue
        ticker_key = _normalize_ticker_key(row.get("ticker"))
        if not ticker_key:
            continue
        cik_text = str(row.get("cik_str") or "").strip()
        if cik_text.endswith(".0"):
            cik_text = cik_text[:-2]
        if cik_text.isdigit():
            cik_text = cik_text.zfill(10)
        out[ticker_key] = {
            "ticker": ticker_key,
            "cik": cik_text,
            "title": str(row.get("title") or "").strip(),
        }
    return out


