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


@lru_cache(maxsize=4096)
def _sec_submission_identity_for_cik(cik: str) -> Dict[str, Any]:
    cik_text = str(cik or "").strip()
    if not cik_text:
        return {}
    if cik_text.isdigit():
        cik_text = cik_text.zfill(10)
    path = _SEC_SUBMISSIONS_ROOT / f"CIK{cik_text}.json"
    return dict(_read_sec_submission_identity(str(path)) or {})


def _taxonomy_from_sec_identity_texts(*, title: Any = "", sic_description: Any = "") -> Tuple[str, str]:
    title_text = str(title or "").strip().lower()
    sic_desc_text = _normalize_sic_description(sic_description)
    combined_text = " ".join(part for part in (title_text, sic_desc_text) if part).strip()
    if not combined_text:
        return ("", "")

    def _has_any(*phrases: str) -> bool:
        return any(str(phrase or "").strip().lower() in combined_text for phrase in phrases)

    if _has_any("therapeutics", "biotechnology", "biotech", "pharma", "pharmaceutical", "gene "):
        return ("Health Care", "Biotechnology")
    if _has_any(
        "medical technologies",
        "medical technology",
        "medical",
        "heartsciences",
        "heart sciences",
        "health sciences",
        "diagnostic",
        "diagnostics",
    ):
        return ("Health Care", "Health Care Equipment & Supplies")
    if _has_any("optical cable", "communications equipment", "telecommunications equipment"):
        return ("Information Technology", "Communications Equipment")
    if _has_any("electrical industrial apparatus", "fuelcell", "fuel cell", "ocean power", "power technologies"):
        return ("Industrials", "Electrical Equipment")
    if _has_any("steel pipe", "steel pipes", "steel tube", "steel tubes"):
        return ("Materials", "Metals & Mining")
    if _has_any("outpatient facilities", "outpatient facility"):
        return ("Health Care", "Health Care Providers & Services")
    if _has_any("perfumes", "cosmetics", "toilet preparations", "personal care"):
        return ("Consumer Staples", "Personal Care Products")
    if _has_any("semiconductors", "semiconductor"):
        return ("Information Technology", "Semiconductors & Semiconductor Equipment")
    if _has_any("finance services"):
        return ("Financials", "Capital Markets")
    return ("", "")


@lru_cache(maxsize=4096)
def _snapshot_taxonomy_for_cik(cik: str) -> Tuple[str, str]:
    cik_text = str(cik or "").strip()
    if not cik_text:
        return ("", "")
    bulk_lookup = _load_snapshot_taxonomy_lookup()
    bulk_hit = bulk_lookup.get(cik_text)
    if bulk_hit:
        return bulk_hit
    return ("", "")


@lru_cache(maxsize=1)
def _load_snapshot_taxonomy_lookup() -> Dict[str, Tuple[str, str]]:
    lookup_path = _SNAPSHOT_TAXONOMY_LOOKUP_PATH
    if lookup_path.exists():
        try:
            df = pd.read_parquet(lookup_path, columns=["company_id", "taxonomy.sector", "taxonomy.subsector"])
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            out: Dict[str, Tuple[str, str]] = {}
            company_ids = df.get("company_id", pd.Series("", index=df.index)).astype(str)
            sector_series = df.get("taxonomy.sector", pd.Series("", index=df.index)).fillna("").astype(str)
            subsector_series = df.get("taxonomy.subsector", pd.Series("", index=df.index)).fillna("").astype(str)
            for company_id, sector_name, subsector_name in zip(company_ids, sector_series, subsector_series):
                company_key = str(company_id or "").strip()
                if not company_key:
                    continue
                sector_name = str(sector_name or "").strip()
                subsector_name = str(subsector_name or "").strip()
                if not sector_name and not subsector_name:
                    continue
                out[company_key] = (sector_name, subsector_name)
            if out:
                return out
    catalog_path = _SNAPSHOT_TAXONOMY_CATALOG_FALLBACK_PATH
    if catalog_path.exists():
        out: Dict[str, Tuple[str, str]] = {}
        scores: Dict[str, Tuple[int, float, str]] = {}
        try:
            with gzip.open(catalog_path, "rt") as handle:
                for line in handle:
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    company_key = str(payload.get("company_id") or "").strip()
                    if not company_key:
                        continue
                    features = payload.get("features") if isinstance(payload, dict) else None
                    features = features if isinstance(features, dict) else {}
                    sector_record = features.get("taxonomy.sector")
                    subsector_record = features.get("taxonomy.subsector")
                    sector_name = str(_extract_metric_value(sector_record) or "").strip()
                    subsector_name = str(_extract_metric_value(subsector_record) or "").strip()
                    if not sector_name and not subsector_name:
                        continue
                    confidence = 0.0
                    for record in (sector_record, subsector_record):
                        try:
                            confidence = max(confidence, float((record or {}).get("confidence") or 0.0))
                        except Exception:
                            continue
                    support_mode = str(
                        (sector_record or {}).get("support_mode")
                        or (subsector_record or {}).get("support_mode")
                        or ""
                    ).strip().lower()
                    score = (
                        int(bool(sector_name) and bool(subsector_name)) + (1 if support_mode == "exact" else 0),
                        float(confidence),
                        str(payload.get("as_of_time") or ""),
                    )
                    existing_score = scores.get(company_key)
                    if existing_score is not None and existing_score >= score:
                        continue
                    scores[company_key] = score
                    out[company_key] = (sector_name, subsector_name)
        except Exception:
            out = {}
        if out:
            return out
    # The frozen snapshot bundle does not reliably carry taxonomy features, so
    # scanning thousands of per-company JSON files here only adds cold-start IO
    # without improving coverage. Historical identity enrichment should come
    # from the curated lookup or the reference taxonomy fallbacks instead.
    return {}


@lru_cache(maxsize=8192)
def _historical_taxonomy_for_ticker(ticker: str, allow_sec_identity_heuristics: bool = False) -> Dict[str, str]:
    ticker_key = _normalize_ticker_key(ticker)
    if not ticker_key:
        return {}

    def _format_taxonomy(sector_name: Any, subsector_name: Any) -> Dict[str, str]:
        sector_text = str(sector_name or "").strip()
        subsector_text = str(subsector_name or "").strip()
        if not sector_text and not subsector_text:
            return {}
        return {
            "taxonomy.sector": sector_text,
            "taxonomy.subsector": subsector_text,
        }

    refinitiv_lookup = _load_refinitiv_taxonomy_lookup()
    refinitiv_hit = refinitiv_lookup.get(_normalize_instrument_root(ticker_key))
    if refinitiv_hit:
        direct_taxonomy = _format_taxonomy(*refinitiv_hit)
        if direct_taxonomy:
            return direct_taxonomy

    cik_lookup = _load_sec_ticker_cik_lookup()
    sec_company_metadata = dict(_load_sec_company_ticker_metadata_lookup().get(ticker_key) or {})
    cik_text = str(cik_lookup.get(ticker_key) or sec_company_metadata.get("cik") or "").strip()

    if cik_text:
        snapshot_taxonomy = _format_taxonomy(*_snapshot_taxonomy_for_cik(cik_text))
        if snapshot_taxonomy:
            return snapshot_taxonomy
    if not allow_sec_identity_heuristics:
        return {}

    override_taxonomy = _format_taxonomy(*_SEC_TICKER_TAXONOMY_OVERRIDES.get(ticker_key, ("", "")))
    if override_taxonomy:
        return override_taxonomy

    submission_record: Dict[str, Any] = {}
    if cik_text:
        submission_record = _sec_submission_identity_for_cik(cik_text)

    if submission_record:
        candidate_tickers: List[str] = []
        primary_ticker = _normalize_ticker_key(submission_record.get("primary_ticker"))
        if primary_ticker:
            candidate_tickers.append(primary_ticker)
        for candidate in list(submission_record.get("tickers") or []):
            candidate_ticker = _normalize_ticker_key(candidate)
            if candidate_ticker and candidate_ticker not in candidate_tickers:
                candidate_tickers.append(candidate_ticker)
        for candidate_ticker in candidate_tickers:
            refinitiv_taxonomy = _format_taxonomy(*refinitiv_lookup.get(_normalize_instrument_root(candidate_ticker), ("", "")))
            if refinitiv_taxonomy:
                return refinitiv_taxonomy

        sec_identity_taxonomy = _format_taxonomy(
            *_taxonomy_from_sec_identity_texts(
                title=submission_record.get("name"),
                sic_description=submission_record.get("sic_description"),
            )
        )
        if sec_identity_taxonomy:
            return sec_identity_taxonomy

    sec_metadata_taxonomy = _format_taxonomy(
        *_taxonomy_from_sec_identity_texts(
            title=sec_company_metadata.get("title"),
        )
    )
    if sec_metadata_taxonomy:
        return sec_metadata_taxonomy
    return {}


def _enrich_missing_historical_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ticker" not in df.columns:
        return df.copy()
    disable_snapshot_lookup = str(os.environ.get("PRECEDENT_DISABLE_HISTORICAL_TAXONOMY_LOOKUP", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    out = df.copy()
    sector_existing = _first_text_series(
        out,
        ("taxonomy.sector", "base_sector", "sector", "gics_sector", "sector_name", "sic", "base_sic"),
    )
    subsector_existing = _first_text_series(
        out,
        ("taxonomy.subsector", "subsector", "industry", "base_industry"),
    )
    need_sector = ~sector_existing.astype(str).str.strip().astype(bool)
    need_subsector = ~subsector_existing.astype(str).str.strip().astype(bool)
    if not bool((need_sector | need_subsector).any()):
        return out
    ticker_series = out.get("ticker", pd.Series("", index=out.index)).astype(str)
    lookup_keys = ticker_series.where(need_sector | need_subsector, "").map(_normalize_ticker_key)
    if disable_snapshot_lookup:
        refinitiv_lookup = _load_refinitiv_taxonomy_lookup()
        taxonomy_map = {}
        for key in lookup_keys.unique():
            key_text = str(key or "").strip()
            if not key_text:
                continue
            sector_name, subsector_name = refinitiv_lookup.get(_normalize_instrument_root(key_text), ("", ""))
            taxonomy_map[key_text] = {
                "taxonomy.sector": str(sector_name or "").strip(),
                "taxonomy.subsector": str(subsector_name or "").strip(),
            }
    else:
        taxonomy_map = {key: _historical_taxonomy_for_ticker(key) for key in lookup_keys.unique() if key}
    mapped_sector = lookup_keys.map(
        lambda key: str((taxonomy_map.get(key) or {}).get("taxonomy.sector") or "").strip()
    )
    mapped_subsector = lookup_keys.map(
        lambda key: str((taxonomy_map.get(key) or {}).get("taxonomy.subsector") or "").strip()
    )
    for col in ("taxonomy.sector", "sector", "gics_sector", "base_sector"):
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].where(~need_sector, mapped_sector.where(mapped_sector.astype(bool), out[col]))
    for col in ("taxonomy.subsector", "subsector", "industry", "base_industry"):
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].where(~need_subsector, mapped_subsector.where(mapped_subsector.astype(bool), out[col]))
    return out


def _flatten_matching_feature_payload(features: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in dict(features or {}).items():
        out[str(key)] = _extract_metric_value(value)
    return out


def _candidate_state_feature_weight_multipliers(
    candidate_features: Dict[str, Any],
    *,
    action_id: str,
    action_subtype: str,
) -> Dict[str, float]:
    if not isinstance(candidate_features, dict) or not candidate_features:
        return {}
    try:
        bundle = build_model_feature_bundle(
            {"features": dict(candidate_features)},
            action_id=action_id,
            action_type=action_subtype,
        )
    except Exception:
        return {}
    support_map = dict((bundle.get("state_vector_v1", {}) or {}).get("support", {}) or {})
    reliability_map = dict((bundle.get("state_vector_v1", {}) or {}).get("reliability", {}) or {})
    values_map = dict((bundle.get("state_vector_v1", {}) or {}).get("values", {}) or {})
    multipliers: Dict[str, float] = {}
    is_capital_structure = str(action_id or "").startswith("capital_structure.")
    for feature_name in _STATE_VECTOR_MATCHING_COLS:
        meta = dict(support_map.get(feature_name) or {})
        support_mode = str(meta.get("support_mode") or "").strip().lower()
        if not support_mode:
            continue
        reliability = _to_float(reliability_map.get(feature_name), 1.0)
        multiplier = 1.0
        if support_mode == "proxy_missing_component":
            multiplier *= 0.80
        elif support_mode not in {"exact", "exact_not_applicable", "exact_structural_zero"}:
            multiplier *= 0.90
        if reliability is not None and reliability < 1.0:
            multiplier *= max(0.55, float(reliability))
        quality_flags = {str(flag) for flag in list(meta.get("quality_flags") or [])}
        feature_value = _to_float(values_map.get(feature_name), None)
        if feature_name == "state_vector_v1.liquidity_flexibility" and support_mode == "proxy_missing_component":
            if "current_debt_fallback" in quality_flags:
                multiplier *= 0.20
            elif "debt_due_0_12m_fallback" in quality_flags:
                multiplier *= 0.40
            if "marketable_securities_missing_assumed_zero" in quality_flags:
                multiplier *= 0.80
            if "revolver_undrawn_missing_assumed_zero" in quality_flags:
                multiplier *= 0.85
            if is_capital_structure and feature_value is not None and feature_value > 25.0:
                multiplier *= 0.50
        if feature_name in {"state_vector_v1.rates_level", "state_vector_v1.credit_spread"}:
            if support_mode == "proxy_missing_component":
                multiplier *= 0.85
            if is_capital_structure:
                multiplier *= 1.10
        multiplier = float(max(0.05, min(1.50, multiplier)))
        if abs(multiplier - 1.0) > 1e-9:
            multipliers[feature_name] = multiplier
    return multipliers


_DEBT_ISSUANCE_ARCHETYPE_LABELS: Tuple[str, ...] = (
    "distressed_borrower",
    "refinancing_pressure",
    "opportunistic_issuer",
)


def _is_debt_support_action(action_id_text: str) -> bool:
    return str(action_id_text or "").strip().lower() in {
        "capital_structure.new_debt_issuance",
        "capital_structure.revolver_draw_or_resize",
    }


def _bounded_sigmoid(value: float) -> float:
    clipped = max(-12.0, min(12.0, float(value)))
    return 1.0 / (1.0 + math.exp(-clipped))


def _debt_issuance_runtime_archetype_profile(
    compact_features: Dict[str, Any],
    *,
    action_id_text: str = "capital_structure.new_debt_issuance",
    action_scale: Optional[float] = None,
) -> Dict[str, Any]:
    action_text = str(action_id_text or "").strip().lower()
    profitability = _to_float(compact_features.get("state_vector_v1.profitability"), None)
    cash_generation = _to_float(compact_features.get("state_vector_v1.cash_generation"), None)
    growth = _to_float(compact_features.get("state_vector_v1.growth"), None)
    gross_burden = _to_float(compact_features.get("state_vector_v1.gross_obligation_burden"), None)
    net_burden = _to_float(compact_features.get("state_vector_v1.net_obligation_burden"), None)
    interest_coverage = _to_float(compact_features.get("state_vector_v1.interest_coverage"), None)
    valuation_multiple = _to_float(compact_features.get("state_vector_v1.valuation_multiple"), None)
    liquidity_flexibility = _to_float(compact_features.get("state_vector_v1.liquidity_flexibility"), None)
    market_access = _to_float(compact_features.get("state_vector_v1.market_access"), None)
    market_stress = _to_float(compact_features.get("state_vector_v1.market_stress"), None)
    credit_spread = _to_float(compact_features.get("state_vector_v1.credit_spread"), None)
    scale_value = _to_float(action_scale, None)

    def _maybe_score(
        value: Optional[float],
        *,
        threshold: float,
        scale: float,
        lower_is_worse: bool,
    ) -> Optional[float]:
        if value is None:
            return None
        signed = (threshold - float(value)) if lower_is_worse else (float(value) - threshold)
        return _bounded_sigmoid(signed / max(float(scale), 1e-9))

    if action_text == "capital_structure.revolver_draw_or_resize":
        distressed_components = [
            (_maybe_score(profitability, threshold=0.10, scale=0.06, lower_is_worse=True), 1.00),
            (_maybe_score(cash_generation, threshold=0.00, scale=0.04, lower_is_worse=True), 1.10),
            (_maybe_score(interest_coverage, threshold=3.00, scale=1.50, lower_is_worse=True), 1.15),
            (_maybe_score(net_burden, threshold=1.60, scale=0.95, lower_is_worse=False), 1.10),
            (_maybe_score(gross_burden, threshold=2.50, scale=1.05, lower_is_worse=False), 0.95),
            (_maybe_score(liquidity_flexibility, threshold=1.10, scale=0.60, lower_is_worse=True), 1.45),
            (_maybe_score(market_access, threshold=0.66, scale=0.12, lower_is_worse=True), 1.20),
            (_maybe_score(market_stress, threshold=0.22, scale=0.08, lower_is_worse=False), 1.10),
            (_maybe_score(credit_spread, threshold=3.60, scale=0.85, lower_is_worse=False), 1.00),
        ]
        distressed_numer = sum(score * weight for score, weight in distressed_components if score is not None)
        distressed_denom = sum(weight for score, weight in distressed_components if score is not None)
        distressed_score = float(distressed_numer / distressed_denom) if distressed_denom > 0.0 else 0.5

        refinancing_components = [
            (_maybe_score(liquidity_flexibility, threshold=1.55, scale=0.85, lower_is_worse=True), 1.30),
            (_maybe_score(gross_burden, threshold=1.90, scale=0.95, lower_is_worse=False), 1.00),
            (_maybe_score(net_burden, threshold=1.10, scale=0.85, lower_is_worse=False), 1.05),
            (_maybe_score(interest_coverage, threshold=4.00, scale=2.00, lower_is_worse=True), 0.80),
            (_maybe_score(market_access, threshold=0.76, scale=0.15, lower_is_worse=True), 0.90),
            (_maybe_score(market_stress, threshold=0.18, scale=0.08, lower_is_worse=False), 0.75),
        ]
        if scale_value is not None:
            refinancing_components.append(
                (_maybe_score(scale_value, threshold=0.08, scale=0.05, lower_is_worse=False), 1.10)
            )
        refi_numer = sum(score * weight for score, weight in refinancing_components if score is not None)
        refi_denom = sum(weight for score, weight in refinancing_components if score is not None)
        refinancing_pressure_score = float(refi_numer / refi_denom) if refi_denom > 0.0 else 0.5

        opportunistic_components = [
            (_maybe_score(profitability, threshold=0.16, scale=0.07, lower_is_worse=False), 1.15),
            (_maybe_score(cash_generation, threshold=0.01, scale=0.04, lower_is_worse=False), 1.10),
            (_maybe_score(growth, threshold=0.05, scale=0.12, lower_is_worse=False), 0.70),
            (_maybe_score(interest_coverage, threshold=5.50, scale=2.50, lower_is_worse=False), 1.00),
            (_maybe_score(liquidity_flexibility, threshold=1.80, scale=1.00, lower_is_worse=False), 1.00),
            (_maybe_score(market_access, threshold=0.80, scale=0.12, lower_is_worse=False), 1.15),
            (_maybe_score(market_stress, threshold=0.16, scale=0.08, lower_is_worse=True), 1.00),
            (_maybe_score(credit_spread, threshold=3.20, scale=0.75, lower_is_worse=True), 0.90),
            (_maybe_score(net_burden, threshold=2.20, scale=1.20, lower_is_worse=True), 0.75),
        ]
        opp_numer = sum(score * weight for score, weight in opportunistic_components if score is not None)
        opp_denom = sum(weight for score, weight in opportunistic_components if score is not None)
        opportunistic_score = float(opp_numer / opp_denom) if opp_denom > 0.0 else 0.5

        scores = {
            "distressed_borrower": distressed_score,
            "refinancing_pressure": refinancing_pressure_score,
            "opportunistic_issuer": opportunistic_score,
        }
        if distressed_score >= 0.60 and distressed_score >= opportunistic_score + 0.06:
            label = "distressed_borrower"
        elif opportunistic_score >= 0.60 and opportunistic_score >= distressed_score + 0.06:
            label = "opportunistic_issuer"
        else:
            label = max(scores.items(), key=lambda item: item[1])[0]
        return {"label": str(label), "scores": scores}

    distressed_components = [
        (_maybe_score(profitability, threshold=0.12, scale=0.06, lower_is_worse=True), 1.10),
        (_maybe_score(cash_generation, threshold=0.00, scale=0.04, lower_is_worse=True), 1.20),
        (_maybe_score(interest_coverage, threshold=3.00, scale=1.50, lower_is_worse=True), 1.20),
        (_maybe_score(net_burden, threshold=1.50, scale=1.00, lower_is_worse=False), 1.20),
        (_maybe_score(gross_burden, threshold=2.40, scale=1.10, lower_is_worse=False), 1.05),
        (_maybe_score(market_access, threshold=0.70, scale=0.14, lower_is_worse=True), 1.15),
        (_maybe_score(market_stress, threshold=0.20, scale=0.10, lower_is_worse=False), 0.85),
        (_maybe_score(credit_spread, threshold=3.00, scale=0.90, lower_is_worse=False), 0.85),
        (_maybe_score(valuation_multiple, threshold=7.00, scale=4.00, lower_is_worse=True), 0.60),
    ]
    distressed_numer = sum(score * weight for score, weight in distressed_components if score is not None)
    distressed_denom = sum(weight for score, weight in distressed_components if score is not None)
    distressed_score = float(distressed_numer / distressed_denom) if distressed_denom > 0.0 else 0.5

    refinancing_components = [
        (_maybe_score(liquidity_flexibility, threshold=1.50, scale=0.75, lower_is_worse=True), 1.25),
        (_maybe_score(gross_burden, threshold=2.00, scale=1.00, lower_is_worse=False), 1.05),
        (_maybe_score(net_burden, threshold=1.00, scale=0.90, lower_is_worse=False), 1.10),
        (_maybe_score(interest_coverage, threshold=4.00, scale=2.00, lower_is_worse=True), 0.80),
        (_maybe_score(market_access, threshold=0.78, scale=0.16, lower_is_worse=True), 0.70),
    ]
    if scale_value is not None:
        refinancing_components.append(
            (_maybe_score(scale_value, threshold=0.08, scale=0.05, lower_is_worse=False), 1.20)
        )
    refi_numer = sum(score * weight for score, weight in refinancing_components if score is not None)
    refi_denom = sum(weight for score, weight in refinancing_components if score is not None)
    refinancing_pressure_score = float(refi_numer / refi_denom) if refi_denom > 0.0 else 0.5

    opportunistic_components = [
        (_maybe_score(profitability, threshold=0.18, scale=0.07, lower_is_worse=False), 1.10),
        (_maybe_score(cash_generation, threshold=0.01, scale=0.04, lower_is_worse=False), 1.05),
        (_maybe_score(growth, threshold=0.08, scale=0.12, lower_is_worse=False), 1.05),
        (_maybe_score(interest_coverage, threshold=6.00, scale=3.00, lower_is_worse=False), 0.95),
        (_maybe_score(market_access, threshold=0.82, scale=0.12, lower_is_worse=False), 1.20),
        (_maybe_score(market_stress, threshold=0.14, scale=0.10, lower_is_worse=True), 0.85),
        (_maybe_score(credit_spread, threshold=3.00, scale=0.80, lower_is_worse=True), 0.95),
        (_maybe_score(net_burden, threshold=2.50, scale=1.40, lower_is_worse=True), 0.80),
        (_maybe_score(valuation_multiple, threshold=18.0, scale=10.0, lower_is_worse=False), 1.15),
        (_maybe_score(liquidity_flexibility, threshold=2.00, scale=1.20, lower_is_worse=False), 0.45),
    ]
    opp_numer = sum(score * weight for score, weight in opportunistic_components if score is not None)
    opp_denom = sum(weight for score, weight in opportunistic_components if score is not None)
    opportunistic_score = float(opp_numer / opp_denom) if opp_denom > 0.0 else 0.5

    scores = {
        "distressed_borrower": distressed_score,
        "refinancing_pressure": refinancing_pressure_score,
        "opportunistic_issuer": opportunistic_score,
    }
    label = "refinancing_pressure"
    if distressed_score >= 0.60 and distressed_score >= opportunistic_score + 0.08:
        label = "distressed_borrower"
    elif opportunistic_score >= 0.60 and opportunistic_score >= distressed_score + 0.08:
        label = "opportunistic_issuer"
    else:
        label = max(scores.items(), key=lambda item: item[1])[0]
    return {
        "label": str(label),
        "scores": scores,
    }


def _debt_issuance_target_feature_multipliers(
    target_profile: Dict[str, Any],
    *,
    action_id_text: str = "capital_structure.new_debt_issuance",
) -> Dict[str, float]:
    action_text = str(action_id_text or "").strip().lower()
    label = str((target_profile or {}).get("label") or "")
    if action_text == "capital_structure.revolver_draw_or_resize":
        if label == "distressed_borrower":
            return {
                "state_vector_v1.profitability": 1.10,
                "state_vector_v1.cash_generation": 1.15,
                "state_vector_v1.gross_obligation_burden": 1.15,
                "state_vector_v1.net_obligation_burden": 1.20,
                "state_vector_v1.interest_coverage": 1.15,
                "state_vector_v1.liquidity_flexibility": 1.35,
                "state_vector_v1.market_access": 1.20,
                "state_vector_v1.market_stress": 1.35,
                "state_vector_v1.credit_spread": 1.25,
                "state_vector_v1.rates_level": 0.95,
                "state_vector_v1.growth": 0.80,
                "state_vector_v1.valuation_multiple": 0.55,
            }
        if label == "opportunistic_issuer":
            return {
                "state_vector_v1.profitability": 1.20,
                "state_vector_v1.cash_generation": 1.15,
                "state_vector_v1.growth": 1.00,
                "state_vector_v1.market_access": 1.25,
                "state_vector_v1.market_stress": 1.15,
                "state_vector_v1.credit_spread": 1.10,
                "state_vector_v1.liquidity_flexibility": 0.95,
                "state_vector_v1.gross_obligation_burden": 0.85,
                "state_vector_v1.net_obligation_burden": 0.80,
                "state_vector_v1.interest_coverage": 1.00,
                "state_vector_v1.valuation_multiple": 0.50,
            }
        return {
            "state_vector_v1.gross_obligation_burden": 1.15,
            "state_vector_v1.net_obligation_burden": 1.15,
            "state_vector_v1.liquidity_flexibility": 1.30,
            "state_vector_v1.interest_coverage": 1.10,
            "state_vector_v1.market_access": 1.15,
            "state_vector_v1.market_stress": 1.25,
            "state_vector_v1.credit_spread": 1.20,
            "state_vector_v1.rates_level": 0.95,
            "state_vector_v1.growth": 0.80,
            "state_vector_v1.valuation_multiple": 0.55,
        }
    if label == "distressed_borrower":
        return {
            "state_vector_v1.profitability": 1.20,
            "state_vector_v1.cash_generation": 1.25,
            "state_vector_v1.gross_obligation_burden": 1.15,
            "state_vector_v1.net_obligation_burden": 1.20,
            "state_vector_v1.interest_coverage": 1.25,
            "state_vector_v1.market_access": 1.15,
            "state_vector_v1.credit_spread": 1.25,
            "state_vector_v1.rates_level": 1.10,
            "state_vector_v1.growth": 0.80,
            "state_vector_v1.valuation_multiple": 0.75,
            "state_vector_v1.liquidity_flexibility": 0.85,
        }
    if label == "opportunistic_issuer":
        return {
            "state_vector_v1.profitability": 1.10,
            "state_vector_v1.cash_generation": 1.05,
            "state_vector_v1.growth": 1.35,
            "state_vector_v1.market_access": 1.30,
            "state_vector_v1.credit_spread": 1.15,
            "state_vector_v1.rates_level": 1.10,
            "state_vector_v1.valuation_multiple": 1.75,
            "state_vector_v1.gross_obligation_burden": 0.80,
            "state_vector_v1.net_obligation_burden": 0.75,
            "state_vector_v1.liquidity_flexibility": 0.75,
            "state_vector_v1.interest_coverage": 0.95,
        }
    return {
        "state_vector_v1.gross_obligation_burden": 1.15,
        "state_vector_v1.net_obligation_burden": 1.15,
        "state_vector_v1.liquidity_flexibility": 1.05,
        "state_vector_v1.interest_coverage": 1.10,
        "state_vector_v1.market_access": 1.10,
        "state_vector_v1.credit_spread": 1.20,
        "state_vector_v1.rates_level": 1.20,
        "state_vector_v1.growth": 0.85,
        "state_vector_v1.valuation_multiple": 0.85,
    }


def _debt_issuance_runtime_archetype_features(
    *,
    emb_raw: np.ndarray,
    candidate_vec_raw: np.ndarray,
    embedding_cols: Sequence[str],
    action_id_text: str = "capital_structure.new_debt_issuance",
    target_action_scale: Optional[float] = None,
    row_action_scales: Optional[Sequence[float] | np.ndarray] = None,
    borrower_quality_similarity: Optional[np.ndarray] = None,
    financing_pressure_similarity: Optional[np.ndarray] = None,
    market_regime_similarity: Optional[np.ndarray] = None,
    stress_alignment_similarity: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    n_rows = int(emb_raw.shape[0]) if emb_raw.ndim == 2 else 0
    if n_rows <= 0:
        empty_float = np.empty(0, dtype=float)
        empty_obj = np.empty(0, dtype=object)
        empty_bool = np.empty(0, dtype=bool)
        return {
            "target_label": "",
            "row_labels": empty_obj,
            "archetype_similarity": empty_float,
            "style_similarity": empty_float,
            "gate": empty_float,
            "preferred_mask": empty_bool,
            "fallback_mask": empty_bool,
        }

    score_keys = tuple(_DEBT_ISSUANCE_ARCHETYPE_LABELS)
    feature_index = {str(col): idx for idx, col in enumerate(embedding_cols)}
    target_compact_values = {
        str(col): float(candidate_vec_raw[idx])
        for idx, col in enumerate(embedding_cols)
        if col in _STATE_VECTOR_MATCHING_COLS and np.isfinite(candidate_vec_raw[idx])
    }
    target_debt_profile = _debt_issuance_runtime_archetype_profile(
        target_compact_values,
        action_id_text=action_id_text,
        action_scale=target_action_scale,
    )
    target_label = str(target_debt_profile.get("label") or "")
    is_revolver_action = str(action_id_text or "").strip().lower() == "capital_structure.revolver_draw_or_resize"
    target_scores = {
        key: float(value)
        for key, value in dict(target_debt_profile.get("scores") or {}).items()
        if key in score_keys and _to_float(value, None) is not None
    }
    growth_idx = feature_index.get("state_vector_v1.growth")
    valuation_idx = feature_index.get("state_vector_v1.valuation_multiple")
    access_idx = feature_index.get("state_vector_v1.market_access")
    target_growth = _to_float(candidate_vec_raw[growth_idx], None) if growth_idx is not None else None
    target_valuation = _to_float(candidate_vec_raw[valuation_idx], None) if valuation_idx is not None else None
    target_access = _to_float(candidate_vec_raw[access_idx], None) if access_idx is not None else None
    row_scale_arr = (
        np.asarray(row_action_scales, dtype=float)
        if row_action_scales is not None
        else np.full(n_rows, np.nan, dtype=float)
    )
    borrower_quality_similarity_arr = (
        np.asarray(borrower_quality_similarity, dtype=float)
        if borrower_quality_similarity is not None
        else np.ones(n_rows, dtype=float)
    )
    financing_pressure_similarity_arr = (
        np.asarray(financing_pressure_similarity, dtype=float)
        if financing_pressure_similarity is not None
        else np.ones(n_rows, dtype=float)
    )
    market_regime_similarity_arr = (
        np.asarray(market_regime_similarity, dtype=float)
        if market_regime_similarity is not None
        else np.ones(n_rows, dtype=float)
    )
    stress_alignment_similarity_arr = (
        np.asarray(stress_alignment_similarity, dtype=float)
        if stress_alignment_similarity is not None
        else np.ones(n_rows, dtype=float)
    )
    cross_label_penalty = {
        "distressed_borrower": {
            "distressed_borrower": 1.0,
            "refinancing_pressure": 0.72,
            "opportunistic_issuer": 0.28,
        },
        "refinancing_pressure": {
            "distressed_borrower": 0.68,
            "refinancing_pressure": 1.0,
            "opportunistic_issuer": 0.35,
        },
        "opportunistic_issuer": {
            "distressed_borrower": 0.22,
            "refinancing_pressure": 0.45,
            "opportunistic_issuer": 1.0,
        },
    }
    row_labels: List[str] = []
    archetype_similarity = np.ones(n_rows, dtype=float)
    style_similarity = np.ones(n_rows, dtype=float)
    gate = np.ones(n_rows, dtype=float)
    for row_idx in range(n_rows):
        row_compact = {
            str(col): float(emb_raw[row_idx, idx])
            for idx, col in enumerate(embedding_cols)
            if col in _STATE_VECTOR_MATCHING_COLS and np.isfinite(emb_raw[row_idx, idx])
        }
        row_profile = _debt_issuance_runtime_archetype_profile(
            row_compact,
            action_id_text=action_id_text,
            action_scale=_to_float(row_scale_arr[row_idx], None),
        )
        row_label = str(row_profile.get("label") or "")
        row_labels.append(row_label)
        row_scores = {
            key: float(value)
            for key, value in dict(row_profile.get("scores") or {}).items()
            if key in score_keys and _to_float(value, None) is not None
        }
        shared_keys = [key for key in score_keys if key in target_scores and key in row_scores]
        if shared_keys:
            archetype_distance = float(
                np.mean([abs(float(target_scores[key]) - float(row_scores[key])) for key in shared_keys])
            )
            archetype_similarity[row_idx] = float(np.exp(-2.60 * archetype_distance))
        label_factor = float((cross_label_penalty.get(target_label) or {}).get(row_label, 0.35))
        gate[row_idx] = float(
            label_factor * np.exp(-2.40 * max(0.72 - archetype_similarity[row_idx], 0.0))
        )
        if target_label == "opportunistic_issuer":
            style_components: List[float] = []
            row_growth = _to_float(emb_raw[row_idx, growth_idx], None) if growth_idx is not None else None
            row_valuation = _to_float(emb_raw[row_idx, valuation_idx], None) if valuation_idx is not None else None
            row_access = _to_float(emb_raw[row_idx, access_idx], None) if access_idx is not None else None
            if is_revolver_action:
                liquidity_idx = feature_index.get("state_vector_v1.liquidity_flexibility")
                stress_idx = feature_index.get("state_vector_v1.market_stress")
                credit_idx = feature_index.get("state_vector_v1.credit_spread")
                target_liquidity = (
                    _to_float(candidate_vec_raw[liquidity_idx], None) if liquidity_idx is not None else None
                )
                row_liquidity = _to_float(emb_raw[row_idx, liquidity_idx], None) if liquidity_idx is not None else None
                target_stress = _to_float(candidate_vec_raw[stress_idx], None) if stress_idx is not None else None
                row_stress = _to_float(emb_raw[row_idx, stress_idx], None) if stress_idx is not None else None
                target_credit = _to_float(candidate_vec_raw[credit_idx], None) if credit_idx is not None else None
                row_credit = _to_float(emb_raw[row_idx, credit_idx], None) if credit_idx is not None else None
                if target_liquidity is not None and row_liquidity is not None:
                    style_components.append(float(np.exp(-abs(float(row_liquidity) - float(target_liquidity)) / 0.85)))
                if target_access is not None and row_access is not None:
                    style_components.append(float(np.exp(-abs(float(row_access) - float(target_access)) / 0.12)))
                if target_stress is not None and row_stress is not None:
                    style_components.append(float(np.exp(-abs(float(row_stress) - float(target_stress)) / 0.08)))
                if target_credit is not None and row_credit is not None:
                    style_components.append(float(np.exp(-abs(float(row_credit) - float(target_credit)) / 0.70)))
                if target_growth is not None and row_growth is not None:
                    style_components.append(float(np.exp(-abs(float(row_growth) - float(target_growth)) / 0.20)))
            else:
                if target_growth is not None and row_growth is not None:
                    style_components.append(float(np.exp(-abs(float(row_growth) - float(target_growth)) / 0.16)))
                if target_valuation is not None and row_valuation is not None:
                    style_components.append(
                        float(
                            np.exp(
                                -abs(float(row_valuation) - float(target_valuation))
                                / max(8.0, 0.22 * abs(float(target_valuation)) + 2.0)
                            )
                        )
                    )
                if target_access is not None and row_access is not None:
                    style_components.append(float(np.exp(-abs(float(row_access) - float(target_access)) / 0.14)))
            if style_components:
                style_similarity[row_idx] = float(
                    np.exp(np.mean(np.log(np.clip(style_components, 1e-9, 1.0))))
                )
            else:
                style_similarity[row_idx] = archetype_similarity[row_idx]
            gate[row_idx] *= float(
                np.exp(-2.60 * max(0.70 - style_similarity[row_idx], 0.0))
                * np.exp(-2.00 * max(0.68 - market_regime_similarity_arr[row_idx], 0.0))
            )
        elif target_label == "distressed_borrower":
            style_similarity[row_idx] = float(borrower_quality_similarity_arr[row_idx])
            stress_factor = (
                1.0
                if float(stress_alignment_similarity_arr[row_idx]) >= 0.999
                else (0.72 if float(stress_alignment_similarity_arr[row_idx]) >= 0.70 else 0.38)
            )
            gate[row_idx] *= float(
                stress_factor
                * np.exp(-2.30 * max(0.72 - float(borrower_quality_similarity_arr[row_idx]), 0.0))
                * np.exp(-1.80 * max(0.62 - float(market_regime_similarity_arr[row_idx]), 0.0))
            )
        else:
            style_similarity[row_idx] = float(financing_pressure_similarity_arr[row_idx])
            gate[row_idx] *= float(
                np.exp(-2.50 * max(0.74 - float(financing_pressure_similarity_arr[row_idx]), 0.0))
                * np.exp(-2.10 * max(0.68 - float(market_regime_similarity_arr[row_idx]), 0.0))
            )
    gate = np.clip(gate, 0.05, 1.0)
    row_labels_arr = np.asarray(row_labels, dtype=object)
    same_archetype_mask = row_labels_arr == target_label
    if target_label == "opportunistic_issuer":
        preferred_mask = (
            same_archetype_mask
            & (archetype_similarity >= 0.66)
            & (style_similarity >= (0.62 if is_revolver_action else 0.60))
            & (market_regime_similarity_arr >= (0.60 if is_revolver_action else 0.58))
        )
        fallback_mask = (
            (gate >= (0.40 if is_revolver_action else 0.38))
            & (style_similarity >= (0.56 if is_revolver_action else 0.54))
            & (market_regime_similarity_arr >= (0.57 if is_revolver_action else 0.55))
        )
    elif target_label == "distressed_borrower":
        preferred_mask = (
            same_archetype_mask
            & (borrower_quality_similarity_arr >= (0.64 if is_revolver_action else 0.62))
            & (stress_alignment_similarity_arr >= (0.74 if is_revolver_action else 0.70))
            & (market_regime_similarity_arr >= (0.56 if is_revolver_action else 0.54))
        )
        fallback_mask = (
            (gate >= (0.42 if is_revolver_action else 0.40))
            & (borrower_quality_similarity_arr >= (0.61 if is_revolver_action else 0.60))
            & (market_regime_similarity_arr >= (0.54 if is_revolver_action else 0.52))
        )
    else:
        preferred_mask = (
            same_archetype_mask
            & (financing_pressure_similarity_arr >= (0.64 if is_revolver_action else 0.62))
            & (market_regime_similarity_arr >= (0.58 if is_revolver_action else 0.56))
        )
        fallback_mask = (
            (gate >= (0.44 if is_revolver_action else 0.42))
            & (financing_pressure_similarity_arr >= (0.60 if is_revolver_action else 0.58))
            & (market_regime_similarity_arr >= (0.56 if is_revolver_action else 0.54))
        )
    return {
        "target_label": target_label,
        "row_labels": row_labels_arr,
        "archetype_similarity": np.asarray(archetype_similarity, dtype=float),
        "style_similarity": np.asarray(style_similarity, dtype=float),
        "gate": np.asarray(gate, dtype=float),
        "preferred_mask": np.asarray(preferred_mask, dtype=bool),
        "fallback_mask": np.asarray(fallback_mask, dtype=bool),
    }


def _debt_support_core_score(values: Sequence[Optional[float]]) -> float:
    valid = [
        float(value)
        for value in values
        if value is not None and np.isfinite(float(value))
    ]
    if not valid:
        return 0.45
    clipped = np.clip(np.asarray(valid, dtype=float), 1e-6, 1.0)
    return float(np.exp(np.mean(np.log(clipped))))


def _apply_debt_support_routing(
    cohort: pd.DataFrame,
    *,
    target_company_id: str,
    target_label: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if cohort.empty:
        return cohort, {
            "applied": False,
            "target_label": str(target_label or ""),
            "lane_counts": {},
            "primary_support_count": 0,
            "same_company_primary_count": 0,
        }

    routed = cohort.copy()
    base_similarity = pd.to_numeric(routed.get("similarity_score"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    borrower_quality = pd.to_numeric(routed.get("borrower_quality_similarity"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    financing_pressure = pd.to_numeric(routed.get("financing_pressure_similarity"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    market_regime = pd.to_numeric(routed.get("market_regime_similarity"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    stress_alignment = pd.to_numeric(routed.get("stress_alignment_similarity"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    archetype_similarity = pd.to_numeric(routed.get("debt_archetype_similarity"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    style_similarity = pd.to_numeric(routed.get("debt_style_similarity"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    gate = pd.to_numeric(routed.get("debt_archetype_gate"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    rate_gap = pd.to_numeric(routed.get("rate_gap"), errors="coerce").fillna(np.inf).to_numpy(dtype=float)
    credit_gap = pd.to_numeric(routed.get("credit_gap"), errors="coerce").fillna(np.inf).to_numpy(dtype=float)
    row_labels = routed.get("debt_archetype_label", pd.Series("", index=routed.index)).fillna("").astype(str).to_numpy(dtype=object)
    same_company = routed.get("company_id", pd.Series("", index=routed.index)).fillna("").astype(str).eq(str(target_company_id or "")).to_numpy(dtype=bool)
    same_archetype = row_labels == str(target_label or "")

    lane_labels = np.full(len(routed), "context", dtype=object)
    lane_priority = np.full(len(routed), 4, dtype=int)
    lane_floor = np.full(len(routed), 0.26, dtype=float)
    lane_core = np.zeros(len(routed), dtype=float)
    lane_base_weight = np.full(len(routed), 0.40, dtype=float)

    for idx in range(len(routed)):
        if str(target_label or "") == "distressed_borrower":
            peer_primary = (
                (not same_company[idx])
                and same_archetype[idx]
                and borrower_quality[idx] >= 0.60
                and stress_alignment[idx] >= 0.70
                and market_regime[idx] >= 0.52
                and rate_gap[idx] <= 1.85
                and credit_gap[idx] <= 1.65
            )
            self_primary = (
                same_company[idx]
                and borrower_quality[idx] >= 0.56
                and stress_alignment[idx] >= 0.70
                and gate[idx] >= 0.32
                and (
                    market_regime[idx] >= 0.42
                    or (rate_gap[idx] <= 2.45 and credit_gap[idx] <= 2.20)
                )
            )
            peer_secondary = (
                (not same_company[idx])
                and gate[idx] >= 0.34
                and borrower_quality[idx] >= 0.56
                and market_regime[idx] >= 0.46
                and rate_gap[idx] <= 2.60
                and credit_gap[idx] <= 2.25
            )
            self_secondary = (
                same_company[idx]
                and gate[idx] >= 0.24
                and borrower_quality[idx] >= 0.50
            )
            core = _debt_support_core_score(
                (
                    borrower_quality[idx],
                    borrower_quality[idx],
                    stress_alignment[idx],
                    market_regime[idx],
                    archetype_similarity[idx],
                    gate[idx],
                )
            )
            lane_base_weight[idx] = 0.38
            if peer_primary:
                lane_labels[idx] = "peer_primary"
                lane_priority[idx] = 0
                lane_floor[idx] = 0.72
            elif self_primary:
                lane_labels[idx] = "same_company_history_primary"
                lane_priority[idx] = 1
                lane_floor[idx] = 0.68
            elif peer_secondary:
                lane_labels[idx] = "peer_secondary"
                lane_priority[idx] = 2
                lane_floor[idx] = 0.50
            elif self_secondary:
                lane_labels[idx] = "same_company_history_secondary"
                lane_priority[idx] = 3
                lane_floor[idx] = 0.46
        elif str(target_label or "") == "opportunistic_issuer":
            peer_primary = (
                (not same_company[idx])
                and same_archetype[idx]
                and style_similarity[idx] >= 0.68
                and market_regime[idx] >= 0.60
                and gate[idx] >= 0.40
                and rate_gap[idx] <= 1.30
                and credit_gap[idx] <= 1.20
            )
            self_primary = (
                same_company[idx]
                and style_similarity[idx] >= 0.62
                and gate[idx] >= 0.34
                and market_regime[idx] >= 0.46
            )
            peer_secondary = (
                (not same_company[idx])
                and gate[idx] >= 0.36
                and style_similarity[idx] >= 0.60
                and market_regime[idx] >= 0.52
                and rate_gap[idx] <= 1.95
                and credit_gap[idx] <= 1.70
            )
            self_secondary = same_company[idx] and gate[idx] >= 0.26 and style_similarity[idx] >= 0.52
            core = _debt_support_core_score(
                (
                    style_similarity[idx],
                    style_similarity[idx],
                    market_regime[idx],
                    archetype_similarity[idx],
                    gate[idx],
                )
            )
            lane_base_weight[idx] = 0.28
            if peer_primary:
                lane_labels[idx] = "peer_primary"
                lane_priority[idx] = 0
                lane_floor[idx] = 0.72
            elif self_primary:
                lane_labels[idx] = "same_company_history_primary"
                lane_priority[idx] = 1
                lane_floor[idx] = 0.68
            elif peer_secondary:
                lane_labels[idx] = "peer_secondary"
                lane_priority[idx] = 2
                lane_floor[idx] = 0.50
            elif self_secondary:
                lane_labels[idx] = "same_company_history_secondary"
                lane_priority[idx] = 3
                lane_floor[idx] = 0.46
        else:
            peer_primary = (
                (not same_company[idx])
                and same_archetype[idx]
                and financing_pressure[idx] >= 0.60
                and market_regime[idx] >= 0.56
                and rate_gap[idx] <= 1.70
                and credit_gap[idx] <= 1.55
            )
            self_primary = (
                same_company[idx]
                and financing_pressure[idx] >= 0.56
                and gate[idx] >= 0.32
                and (
                    market_regime[idx] >= 0.44
                    or (rate_gap[idx] <= 2.30 and credit_gap[idx] <= 2.10)
                )
            )
            peer_secondary = (
                (not same_company[idx])
                and gate[idx] >= 0.36
                and financing_pressure[idx] >= 0.54
                and market_regime[idx] >= 0.50
                and rate_gap[idx] <= 2.40
                and credit_gap[idx] <= 2.10
            )
            self_secondary = same_company[idx] and gate[idx] >= 0.24 and financing_pressure[idx] >= 0.50
            core = _debt_support_core_score(
                (
                    financing_pressure[idx],
                    financing_pressure[idx],
                    market_regime[idx],
                    archetype_similarity[idx],
                    gate[idx],
                )
            )
            lane_base_weight[idx] = 0.34
            if peer_primary:
                lane_labels[idx] = "peer_primary"
                lane_priority[idx] = 0
                lane_floor[idx] = 0.72
            elif self_primary:
                lane_labels[idx] = "same_company_history_primary"
                lane_priority[idx] = 1
                lane_floor[idx] = 0.68
            elif peer_secondary:
                lane_labels[idx] = "peer_secondary"
                lane_priority[idx] = 2
                lane_floor[idx] = 0.50
            elif self_secondary:
                lane_labels[idx] = "same_company_history_secondary"
                lane_priority[idx] = 3
                lane_floor[idx] = 0.46
        lane_core[idx] = core

    routed["pre_debt_support_similarity_score"] = base_similarity
    routed["debt_support_lane"] = lane_labels
    routed["debt_support_priority"] = lane_priority
    routed["debt_support_same_company"] = same_company
    routed["debt_support_core"] = lane_core
    routed_similarity = np.clip(
        lane_floor + 0.22 * (lane_base_weight * base_similarity + (1.0 - lane_base_weight) * lane_core),
        0.0,
        0.999999,
    )
    routed["similarity_score"] = routed_similarity
    routed = routed.sort_values(
        ["debt_support_priority", "similarity_score", "action_date", "company_id"],
        ascending=[True, False, False, True],
    )
    lane_counts = {
        str(label): int(count)
        for label, count in routed["debt_support_lane"].value_counts(dropna=False).to_dict().items()
    }
    return routed, {
        "applied": True,
        "target_label": str(target_label or ""),
        "lane_counts": lane_counts,
        "primary_support_count": int(np.count_nonzero(np.isin(lane_labels, ("peer_primary", "same_company_history_primary")))),
        "same_company_primary_count": int(np.count_nonzero(lane_labels == "same_company_history_primary")),
    }


def _debt_issuance_pairwise_compatibility(
    *,
    emb_raw: np.ndarray,
    candidate_vec_raw: np.ndarray,
    embedding_cols: Sequence[str],
    action_id_text: str = "capital_structure.new_debt_issuance",
    feature_weight_multipliers: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    n_rows = int(emb_raw.shape[0]) if emb_raw.ndim == 2 else 0
    empty = np.empty(0, dtype=float)
    if n_rows <= 0:
        return {
            "borrower_quality_similarity": empty,
            "financing_pressure_similarity": empty,
            "market_regime_similarity": empty,
            "stress_alignment_similarity": empty,
            "compatibility_penalty_factor": empty,
        }

    feature_index = {str(name): idx for idx, name in enumerate(embedding_cols)}
    is_revolver_action = str(action_id_text or "").strip().lower() == "capital_structure.revolver_draw_or_resize"

    def _candidate_value(feature_name: str) -> Optional[float]:
        idx = feature_index.get(feature_name)
        if idx is None:
            return None
        value = _to_float(candidate_vec_raw[idx], None)
        if value is None or not np.isfinite(value):
            return None
        return float(value)

    def _row_array(feature_name: str) -> np.ndarray:
        idx = feature_index.get(feature_name)
        if idx is None:
            return np.full(n_rows, np.nan, dtype=float)
        return np.asarray(emb_raw[:, idx], dtype=float)

    def _logistic(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(values, dtype=float), -12.0, 12.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _quality_distance_similarity() -> np.ndarray:
        if is_revolver_action:
            specs = (
                ("state_vector_v1.profitability", 0.07, 1.20),
                ("state_vector_v1.cash_generation", 0.04, 1.20),
                ("state_vector_v1.gross_obligation_burden", 1.20, 1.00),
                ("state_vector_v1.net_obligation_burden", 1.00, 1.20),
                ("state_vector_v1.interest_coverage", 2.75, 1.20),
                ("state_vector_v1.liquidity_flexibility", 0.85, 1.45),
                ("state_vector_v1.market_access", 0.12, 1.30),
                ("state_vector_v1.market_stress", 0.08, 1.10),
                ("state_vector_v1.credit_spread", 0.80, 1.05),
            )
        else:
            specs = (
                ("state_vector_v1.profitability", 0.08, 1.25),
                ("state_vector_v1.cash_generation", 0.05, 1.10),
                ("state_vector_v1.gross_obligation_burden", 1.50, 1.15),
                ("state_vector_v1.net_obligation_burden", 1.25, 1.25),
                ("state_vector_v1.interest_coverage", 4.00, 1.15),
                ("state_vector_v1.valuation_multiple", 12.00, 0.95),
                ("state_vector_v1.market_access", 0.15, 1.10),
            )
        numer = np.zeros(n_rows, dtype=float)
        denom = np.zeros(n_rows, dtype=float)
        for feature_name, scale, weight in specs:
            cand_value = _candidate_value(feature_name)
            if cand_value is None:
                continue
            row_values = _row_array(feature_name)
            valid = np.isfinite(row_values)
            if not bool(np.any(valid)):
                continue
            diff = np.abs(row_values - float(cand_value)) / max(float(scale), 1e-9)
            numer[valid] += float(weight) * diff[valid]
            denom[valid] += float(weight)
        distance = np.full(n_rows, np.nan, dtype=float)
        valid = denom > 1e-12
        distance[valid] = numer[valid] / denom[valid]
        return np.where(np.isfinite(distance), np.exp(-1.15 * np.clip(distance, 0.0, 20.0)), 1.0)

    liquidity_target_multiplier = float(
        _to_float((feature_weight_multipliers or {}).get("state_vector_v1.liquidity_flexibility"), 1.0) or 1.0
    )

    def _stress_score(prefix_rows: bool) -> Tuple[np.ndarray, np.ndarray]:
        base = np.zeros(n_rows, dtype=float) if prefix_rows else np.zeros(1, dtype=float)
        denom = np.zeros(n_rows, dtype=float) if prefix_rows else np.zeros(1, dtype=float)

        def _add_component(feature_name: str, *, threshold: float, scale: float, weight: float, lower_is_worse: bool) -> None:
            nonlocal base, denom
            if prefix_rows:
                values = _row_array(feature_name)
                valid = np.isfinite(values)
                if not bool(np.any(valid)):
                    return
                transformed = _logistic(((threshold - values) if lower_is_worse else (values - threshold)) / max(scale, 1e-9))
                base[valid] += float(weight) * transformed[valid]
                denom[valid] += float(weight)
            else:
                value = _candidate_value(feature_name)
                if value is None:
                    return
                transformed = _logistic(
                    np.asarray(
                        [((threshold - float(value)) if lower_is_worse else (float(value) - threshold)) / max(scale, 1e-9)],
                        dtype=float,
                    )
                )
                base += float(weight) * transformed
                denom += float(weight)

        if is_revolver_action:
            _add_component("state_vector_v1.profitability", threshold=0.10, scale=0.06, weight=1.00, lower_is_worse=True)
            _add_component("state_vector_v1.cash_generation", threshold=0.00, scale=0.04, weight=1.10, lower_is_worse=True)
            _add_component("state_vector_v1.interest_coverage", threshold=3.00, scale=1.50, weight=1.25, lower_is_worse=True)
            _add_component("state_vector_v1.net_obligation_burden", threshold=1.60, scale=0.95, weight=1.15, lower_is_worse=False)
            _add_component("state_vector_v1.gross_obligation_burden", threshold=2.50, scale=1.05, weight=0.95, lower_is_worse=False)
            _add_component("state_vector_v1.market_access", threshold=0.66, scale=0.12, weight=1.20, lower_is_worse=True)
            _add_component("state_vector_v1.market_stress", threshold=0.22, scale=0.08, weight=1.10, lower_is_worse=False)
            _add_component("state_vector_v1.credit_spread", threshold=3.60, scale=0.85, weight=1.00, lower_is_worse=False)
            liquidity_weight = 1.25 if liquidity_target_multiplier >= 0.35 else 0.90
            liquidity_threshold = 1.10
            liquidity_scale = 0.60
            liquidity_cap = 4.5
        else:
            _add_component("state_vector_v1.profitability", threshold=0.12, scale=0.06, weight=1.10, lower_is_worse=True)
            _add_component("state_vector_v1.cash_generation", threshold=0.00, scale=0.04, weight=1.15, lower_is_worse=True)
            _add_component("state_vector_v1.interest_coverage", threshold=3.00, scale=1.50, weight=1.30, lower_is_worse=True)
            _add_component("state_vector_v1.net_obligation_burden", threshold=1.50, scale=1.00, weight=1.25, lower_is_worse=False)
            _add_component("state_vector_v1.gross_obligation_burden", threshold=2.50, scale=1.10, weight=1.05, lower_is_worse=False)
            _add_component("state_vector_v1.market_access", threshold=0.70, scale=0.12, weight=1.20, lower_is_worse=True)
            _add_component("state_vector_v1.market_stress", threshold=0.20, scale=0.10, weight=0.85, lower_is_worse=False)
            _add_component("state_vector_v1.credit_spread", threshold=3.00, scale=0.90, weight=0.70, lower_is_worse=False)
            liquidity_weight = 0.35 if liquidity_target_multiplier >= 0.35 else 0.10
            liquidity_threshold = 1.50
            liquidity_scale = 0.75
            liquidity_cap = 5.0
        if prefix_rows:
            liquidity_values = _row_array("state_vector_v1.liquidity_flexibility")
            valid = np.isfinite(liquidity_values)
            if bool(np.any(valid)):
                clipped = np.minimum(liquidity_values, liquidity_cap)
                transformed = _logistic((liquidity_threshold - clipped) / liquidity_scale)
                base[valid] += liquidity_weight * transformed[valid]
                denom[valid] += liquidity_weight
        else:
            liquidity_value = _candidate_value("state_vector_v1.liquidity_flexibility")
            if liquidity_value is not None:
                clipped = min(float(liquidity_value), liquidity_cap)
                transformed = _logistic(np.asarray([(liquidity_threshold - clipped) / liquidity_scale], dtype=float))
                base += liquidity_weight * transformed
                denom += liquidity_weight

        score = np.full_like(base, np.nan, dtype=float)
        valid = denom > 1e-12
        score[valid] = base[valid] / denom[valid]
        return score, denom

    row_stress_score, _ = _stress_score(prefix_rows=True)
    cand_stress_arr, cand_stress_denom = _stress_score(prefix_rows=False)
    cand_stress_score = float(cand_stress_arr[0]) if cand_stress_denom[0] > 1e-12 and np.isfinite(cand_stress_arr[0]) else None
    if cand_stress_score is None:
        financing_pressure_similarity = np.ones(n_rows, dtype=float)
        stress_alignment_similarity = np.ones(n_rows, dtype=float)
    else:
        financing_pressure_similarity = np.where(
            np.isfinite(row_stress_score),
            np.exp(-(2.10 if is_revolver_action else 1.75) * np.abs(row_stress_score - float(cand_stress_score))),
            1.0,
        )
        stress_bucket_threshold = 0.56 if is_revolver_action else 0.58
        cand_bucket = float(cand_stress_score >= stress_bucket_threshold)
        row_bucket = np.where(
            np.isfinite(row_stress_score),
            (row_stress_score >= stress_bucket_threshold).astype(float),
            cand_bucket,
        )
        stress_alignment_similarity = np.where(
            row_bucket == cand_bucket,
            1.0,
            np.where(
                np.isfinite(row_stress_score) & (np.abs(row_stress_score - float(cand_stress_score)) <= (0.14 if is_revolver_action else 0.18)),
                0.72,
                0.35,
            ),
        )

    def _market_regime_similarity() -> np.ndarray:
        components: List[np.ndarray] = []

        rate_target = _candidate_value("state_vector_v1.rates_level")
        if rate_target is not None:
            rate_gap = np.abs(_row_array("state_vector_v1.rates_level") - float(rate_target))
            components.append(np.where(np.isfinite(rate_gap), np.exp(-np.maximum(rate_gap - 0.25, 0.0) / 0.55), np.nan))

        credit_target = _candidate_value("state_vector_v1.credit_spread")
        if credit_target is not None:
            credit_gap = np.abs(_row_array("state_vector_v1.credit_spread") - float(credit_target))
            components.append(np.where(np.isfinite(credit_gap), np.exp(-np.maximum(credit_gap - 0.35, 0.0) / 0.60), np.nan))

        access_target = _candidate_value("state_vector_v1.market_access")
        if access_target is not None:
            access_gap = np.abs(_row_array("state_vector_v1.market_access") - float(access_target))
            components.append(np.where(np.isfinite(access_gap), np.exp(-access_gap / 0.18), np.nan))

        stress_target = _candidate_value("state_vector_v1.market_stress")
        if stress_target is not None:
            stress_gap = np.abs(_row_array("state_vector_v1.market_stress") - float(stress_target))
            components.append(np.where(np.isfinite(stress_gap), np.exp(-stress_gap / 0.12), np.nan))

        if not components:
            return np.ones(n_rows, dtype=float)
        stacked = np.vstack(components)
        valid = np.isfinite(stacked)
        safe = np.where(valid, np.clip(stacked, 1e-9, 1.0), 1.0)
        log_mean = np.divide(
            np.sum(np.where(valid, np.log(safe), 0.0), axis=0),
            np.maximum(np.sum(valid, axis=0), 1),
        )
        similarity = np.exp(log_mean)
        similarity[np.sum(valid, axis=0) == 0] = 1.0
        return similarity

    borrower_quality_similarity = _quality_distance_similarity()
    market_regime_similarity = _market_regime_similarity()
    compatibility_penalty_factor = (
        np.exp(-(2.35 if is_revolver_action else 2.20) * np.maximum((0.74 if is_revolver_action else 0.72) - market_regime_similarity, 0.0))
        * np.exp(-(1.45 if is_revolver_action else 1.35) * np.maximum((0.70 if is_revolver_action else 0.68) - borrower_quality_similarity, 0.0))
        * np.exp(-(1.30 if is_revolver_action else 1.10) * np.maximum((0.72 if is_revolver_action else 0.70) - financing_pressure_similarity, 0.0))
        * np.where(
            stress_alignment_similarity >= 0.999,
            1.0,
            np.where(stress_alignment_similarity >= 0.70, 0.82, 0.52),
        )
    )
    compatibility_penalty_factor = np.clip(compatibility_penalty_factor, 0.10, 1.0)
    return {
        "borrower_quality_similarity": np.asarray(borrower_quality_similarity, dtype=float),
        "financing_pressure_similarity": np.asarray(financing_pressure_similarity, dtype=float),
        "market_regime_similarity": np.asarray(market_regime_similarity, dtype=float),
        "stress_alignment_similarity": np.asarray(stress_alignment_similarity, dtype=float),
        "compatibility_penalty_factor": np.asarray(compatibility_penalty_factor, dtype=float),
    }


def _precedent_distance_profile_version(action_id: str = "", action_subtype: str = "") -> str:
    raw = str(os.environ.get("PRECEDENT_DISTANCE_PROFILE_VERSION", "") or "").strip().lower()
    if raw == _WEIGHTED_DISTANCE_V2_VERSION:
        return _WEIGHTED_DISTANCE_V2_VERSION
    if raw == _WEIGHTED_DISTANCE_V1_VERSION:
        return _WEIGHTED_DISTANCE_V1_VERSION
    if raw:
        return _WEIGHTED_DISTANCE_V1_VERSION
    runtime_payload = _load_precedent_distance_v2_weights()
    runtime_scope = _v2_scope_lookup(runtime_payload, action_id, action_subtype)
    if isinstance(runtime_scope, dict) and bool(runtime_scope.get("default_enabled", False)):
        return _WEIGHTED_DISTANCE_V2_VERSION
    return _WEIGHTED_DISTANCE_V1_VERSION


def _normalize_weight_mapping(weights: Dict[str, float]) -> Dict[str, float]:
    numeric = {
        key: float(value)
        for key, value in weights.items()
        if _to_float(value, None) is not None and float(value) > 0.0
    }
    positives = [value for value in numeric.values() if value > 0.0]
    if positives:
        mean_value = float(np.mean(positives))
        if mean_value > 1e-12:
            numeric = {key: float(value / mean_value) for key, value in numeric.items()}
    return numeric


def _learned_weight_scope(action_id: str, action_subtype: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    payload = _load_precedent_distance_weights()
    scopes = payload.get("scopes") if isinstance(payload, dict) else None
    if not isinstance(scopes, dict) or not scopes:
        return "prior_only", None
    action_text = str(action_id or "").strip().lower()
    family_key = action_text.split(".")[0] if "." in action_text else action_text
    subtype_text = str(action_subtype or "").strip().lower()
    for key in (action_text, subtype_text, family_key, "ALL"):
        value = scopes.get(key)
        if (
            isinstance(value, dict)
            and isinstance(value.get("weights"), dict)
            and bool(value.get("use_in_runtime"))
        ):
            return str(key), value
    return "prior_only", None

def _clean_text_series(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str)
    return s.replace({"nan": "", "None": "", "<NA>": ""})


def _preferred_text_array(
    df: pd.DataFrame,
    preferred_cols: Sequence[str],
    fallback_cols: Sequence[str] = (),
) -> np.ndarray:
    out = pd.Series("", index=df.index, dtype="object")
    for col in tuple(preferred_cols) + tuple(fallback_cols):
        if col not in df.columns:
            continue
        s = _clean_text_series(df[col])
        mask = (out == "") & (s != "")
        out = out.where(~mask, s)
    return out.to_numpy(dtype=object)


def _first_text_series(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    out = pd.Series("", index=df.index, dtype="object")
    for col in columns:
        if col not in df.columns:
            continue
        series = _clean_text_series(df[col])
        mask = (out == "") & (series != "")
        out = out.where(~mask, series)
    return out


def _empty_numeric_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.nan, index=df.index, dtype=float)


def _first_numeric_series(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    out = _empty_numeric_series(df)
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        out = out.where(out.notna(), series)
    return out


def _safe_log10_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=numeric.index, dtype=float)
    mask = numeric > 0
    if bool(mask.any()):
        out.loc[mask] = np.log10(numeric.loc[mask].astype(float))
    return out


def _weighted_average_series(parts: Sequence[Tuple[pd.Series, float]]) -> pd.Series:
    if not parts:
        raise ValueError("parts must be non-empty")
    index = parts[0][0].index
    numer = np.zeros(len(index), dtype=float)
    denom = np.zeros(len(index), dtype=float)
    for series, weight in parts:
        numeric = pd.to_numeric(series, errors="coerce")
        values = numeric.to_numpy(dtype=float)
        ok = np.isfinite(values)
        numer[ok] += float(weight) * values[ok]
        denom[ok] += float(weight)
    out = np.full(len(index), np.nan, dtype=float)
    valid = denom > 0
    out[valid] = numer[valid] / denom[valid]
    return pd.Series(out, index=index, dtype=float)


def _weighted_distance_profile_v1(action_id: str, action_subtype: str) -> Dict[str, Any]:
    action_text = str(action_id or "").strip().lower()
    subtype_text = str(action_subtype or "").strip().lower()
    weights = dict(_STATE_VECTOR_BASE_WEIGHTS)
    critical = set(_STATE_VECTOR_CORE_CRITICAL_FEATURES)
    profile: Dict[str, Any] = {
        "version": _WEIGHTED_DISTANCE_V1_VERSION,
        "weights": weights,
        "critical_features": critical,
        "min_weighted_coverage": 0.60,
        "min_critical_coverage": 0.55,
        "max_size_gap": 1.20,
        "primary_burden_feature": "state_vector_v1.net_obligation_burden",
        "soft_burden_gap": 1.50,
        "distance_scale": 0.45,
        "weight_scope": "prior_only",
    }

    if action_text.startswith("capital_return.dividend") or subtype_text.startswith("dividend"):
        weights["state_vector_v1.net_obligation_burden"] *= 1.30
        weights["state_vector_v1.liquidity_flexibility"] *= 1.35
        weights["state_vector_v1.interest_coverage"] *= 1.25
        weights["state_vector_v1.cash_generation"] *= 1.40
        weights["state_vector_v1.valuation_multiple"] *= 0.90
        weights["state_vector_v1.market_access"] *= 0.85
        critical.update(
            {
                "state_vector_v1.net_obligation_burden",
                "state_vector_v1.liquidity_flexibility",
                "state_vector_v1.interest_coverage",
                "state_vector_v1.cash_generation",
            }
        )
        profile["min_weighted_coverage"] = 0.62
        profile["min_critical_coverage"] = 0.60
        profile["max_size_gap"] = 1.05
        profile["soft_burden_gap"] = 1.25
    elif (
        action_text in {
            "capital_return.open_market_buyback",
            "capital_return.accelerated_share_repurchase",
        }
        or "buyback" in subtype_text
        or "repurchase" in subtype_text
    ):
        weights["state_vector_v1.valuation_multiple"] *= 1.35
        weights["state_vector_v1.cash_generation"] *= 1.25
        weights["state_vector_v1.net_obligation_burden"] *= 1.20
        weights["state_vector_v1.liquidity_flexibility"] *= 1.10
        weights["state_vector_v1.market_access"] *= 0.90
        critical.update(
            {
                "state_vector_v1.net_obligation_burden",
                "state_vector_v1.liquidity_flexibility",
                "state_vector_v1.valuation_multiple",
                "state_vector_v1.cash_generation",
            }
        )
        profile["max_size_gap"] = 1.15
        profile["soft_burden_gap"] = 1.50
    elif action_text.startswith("capital_structure.") or "debt" in action_text or "refinanc" in subtype_text:
        weights["state_vector_v1.gross_obligation_burden"] *= 1.35
        weights["state_vector_v1.liquidity_flexibility"] *= 1.40
        weights["state_vector_v1.interest_coverage"] *= 1.25
        weights["state_vector_v1.market_access"] *= 1.45
        weights["state_vector_v1.credit_spread"] *= 1.35
        weights["state_vector_v1.valuation_multiple"] *= 0.75
        critical.update(
            {
                "state_vector_v1.gross_obligation_burden",
                "state_vector_v1.liquidity_flexibility",
                "state_vector_v1.interest_coverage",
                "state_vector_v1.market_access",
            }
        )
        profile["primary_burden_feature"] = "state_vector_v1.gross_obligation_burden"
        profile["min_weighted_coverage"] = 0.62
        profile["min_critical_coverage"] = 0.60
        profile["max_size_gap"] = 1.30
        profile["soft_burden_gap"] = 1.20

    profile["critical_features"] = tuple(sorted(critical))
    learned_scope, learned = _learned_weight_scope(action_text, subtype_text)
    if learned:
        learned_weights = learned.get("weights")
        if isinstance(learned_weights, dict):
            for feature_name, weight in learned_weights.items():
                if feature_name not in weights:
                    continue
                weight_value = _to_float(weight, None)
                if weight_value is None or weight_value <= 0.0:
                    continue
                weights[feature_name] = float(weight_value)
        profile["weight_scope"] = str(learned_scope)
        profile["learned_holdout_pair_correlation"] = _to_float(learned.get("holdout_pair_correlation"), None)
        profile["learned_prior_holdout_pair_correlation"] = _to_float(
            learned.get("holdout_prior_pair_correlation"), None
        )
        profile["learned_pair_count"] = int(_to_float(learned.get("n_pairs"), 0.0) or 0.0)
    profile["weights"] = weights
    return profile


def _v2_scope_lookup(payload: Dict[str, Any], action_id: str, action_subtype: str) -> Optional[Dict[str, Any]]:
    scopes = payload.get("scopes") if isinstance(payload, dict) else None
    if not isinstance(scopes, dict) or not scopes:
        return None
    action_text = str(action_id or "").strip().lower()
    family_key = action_text.split(".")[0] if "." in action_text else action_text
    subtype_text = str(action_subtype or "").strip().lower()
    for key in (action_text, subtype_text, family_key, "ALL"):
        value = scopes.get(key)
        if isinstance(value, dict) and bool(value.get("use_in_runtime", True)):
            return value
    return None


def _weighted_distance_profile_v2(action_id: str, action_subtype: str) -> Dict[str, Any]:
    action_text = str(action_id or "").strip().lower()
    subtype_text = str(action_subtype or "").strip().lower()
    family_key = action_text.split(".", 1)[0] if "." in action_text else action_text

    group_weights = dict(_STATE_VECTOR_V2_DEFAULT_GROUP_WEIGHTS)
    if action_text.startswith("capital_return.dividend") or subtype_text.startswith("dividend"):
        multipliers = _STATE_VECTOR_V2_DEFAULT_GROUP_MULTIPLIERS.get("capital_return.dividend", {})
    elif (
        action_text in {
            "capital_return.open_market_buyback",
            "capital_return.accelerated_share_repurchase",
        }
        or "buyback" in subtype_text
        or "repurchase" in subtype_text
    ):
        multipliers = _STATE_VECTOR_V2_DEFAULT_GROUP_MULTIPLIERS.get("capital_return.buyback", {})
    else:
        multipliers = _STATE_VECTOR_V2_DEFAULT_GROUP_MULTIPLIERS.get(family_key, {})
    for group_name, multiplier in dict(multipliers or {}).items():
        base_value = float(group_weights.get(group_name, 1.0))
        group_weights[group_name] = base_value * float(multiplier)

    feature_relative_weights = dict(_STATE_VECTOR_V2_DEFAULT_FEATURE_RELATIVE_WEIGHTS)
    feature_transform_mode = _normalize_feature_transform_mode(None)
    feature_transforms: Dict[str, Dict[str, float]] = _initialize_feature_transform_specs(feature_transform_mode)
    interaction_terms: List[Dict[str, Any]] = []
    gates: Dict[str, Any] = dict(_STATE_VECTOR_V2_DEFAULT_GATES)
    penalties: Dict[str, Any] = dict(_STATE_VECTOR_V2_DEFAULT_PENALTIES)
    blend_weights: Dict[str, float] = dict(_STATE_VECTOR_V2_DEFAULT_BLEND_WEIGHTS)
    latent_regime_model: Dict[str, Any] = {}
    latent_regime_penalty_weight = 0.0
    target_regime_mixture: Dict[str, Any] = {}
    second_stage_reranker: Dict[str, Any] = {}
    outcome_aware_reranker: Dict[str, Any] = {}
    max_matches_per_company = 0
    critical = set(_STATE_VECTOR_CORE_CRITICAL_FEATURES)
    primary_burden_feature = "state_vector_v1.net_obligation_burden"

    if action_text.startswith("capital_structure.") or "debt" in action_text or "refinanc" in subtype_text:
        primary_burden_feature = "state_vector_v1.gross_obligation_burden"
        gates["max_size_gap"] = 1.30
        penalties["sector_penalty_weight"] = 0.22
        critical.update({"state_vector_v1.market_access", "state_vector_v1.credit_spread"})
    elif action_text.startswith("capital_return.dividend") or subtype_text.startswith("dividend"):
        gates["max_size_gap"] = 1.05
        gates["soft_burden_gap"] = 1.10
        critical.update({"state_vector_v1.cash_generation"})
    elif "buyback" in action_text or "repurchase" in action_text or "buyback" in subtype_text:
        feature_relative_weights["state_vector_v1.valuation_multiple"] = 1.35
        feature_relative_weights["state_vector_v1.cash_generation"] = 1.20
        gates["max_size_gap"] = 1.15

    runtime_payload = _load_precedent_distance_v2_weights()
    runtime_scope = _v2_scope_lookup(runtime_payload, action_text, subtype_text)
    scope_source = "default_v2"
    if runtime_scope:
        scope_source = str(runtime_scope.get("scope_key") or runtime_scope.get("scope_name") or family_key or "runtime_v2")
        feature_transform_mode = _normalize_feature_transform_mode(runtime_scope.get("feature_transform_mode"))
        feature_transforms = _initialize_feature_transform_specs(feature_transform_mode)
        for key, value in dict(runtime_scope.get("group_weights", {}) or {}).items():
            weight_value = _to_float(value, None)
            if weight_value is not None and weight_value > 0.0:
                group_weights[str(key)] = float(weight_value)
        for key, value in dict(runtime_scope.get("feature_relative_weights", {}) or {}).items():
            weight_value = _to_float(value, None)
            if weight_value is not None and weight_value > 0.0:
                feature_relative_weights[str(key)] = float(weight_value)
        for key, value in dict(runtime_scope.get("feature_transforms", {}) or {}).items():
            normalized_spec = _normalize_matching_transform_spec(value)
            if normalized_spec:
                feature_transforms[str(key)] = normalized_spec
        for item in list(runtime_scope.get("interaction_terms", []) or []):
            if not isinstance(item, dict):
                continue
            features = list(item.get("features") or [])
            if len(features) != 2:
                continue
            left = str(features[0] or "")
            right = str(features[1] or "")
            if left not in _STATE_VECTOR_MATCHING_COLS or right not in _STATE_VECTOR_MATCHING_COLS:
                continue
            weight_value = _to_float(item.get("weight"), None)
            if weight_value is None or weight_value <= 0.0:
                continue
            interaction_terms.append(
                {
                    "features": (left, right),
                    "weight": float(weight_value),
                }
            )
        gates.update({str(key): value for key, value in dict(runtime_scope.get("gates", {}) or {}).items()})
        penalties.update({str(key): value for key, value in dict(runtime_scope.get("penalties", {}) or {}).items()})
        blend_weights.update({str(key): float(value) for key, value in dict(runtime_scope.get("blend_weights", {}) or {}).items() if _to_float(value, None) is not None})
        critical.update({str(x) for x in list(runtime_scope.get("critical_features", []) or []) if str(x)})
        if str(runtime_scope.get("primary_burden_feature") or ""):
            primary_burden_feature = str(runtime_scope.get("primary_burden_feature"))
        latent_weight_value = _to_float(runtime_scope.get("latent_regime_penalty_weight"), None)
        if latent_weight_value is not None and latent_weight_value > 0.0:
            latent_regime_penalty_weight = float(latent_weight_value)
        latent_model_value = runtime_scope.get("latent_regime_model")
        if isinstance(latent_model_value, dict):
            latent_regime_model = dict(latent_model_value)
        target_regime_value = runtime_scope.get("target_regime_mixture")
        if isinstance(target_regime_value, dict) and isinstance(target_regime_value.get("model"), dict):
            target_regime_mixture = {
                "model": dict(target_regime_value.get("model") or {}),
                "regimes": list(target_regime_value.get("regimes") or []),
            }
        reranker_value = runtime_scope.get("second_stage_reranker")
        if isinstance(reranker_value, dict):
            feature_weights: Dict[str, float] = {}
            for key, value in dict(reranker_value.get("feature_weights", {}) or {}).items():
                key_text = str(key)
                numeric_value = _to_float(value, None)
                if key_text not in _SECOND_STAGE_RERANKER_FEATURES or numeric_value is None or numeric_value <= 0.0:
                    continue
                feature_weights[key_text] = float(numeric_value)
            if feature_weights:
                second_stage_reranker = {
                    "feature_weights": feature_weights,
                    "bias": float(_to_float(reranker_value.get("bias"), 0.0) or 0.0),
                    "shortlist_size": int(
                        _to_float(reranker_value.get("shortlist_size"), max(80, 4 * len(_SECOND_STAGE_RERANKER_FEATURES)))
                        or max(80, 4 * len(_SECOND_STAGE_RERANKER_FEATURES))
                    ),
                }
        outcome_reranker_value = runtime_scope.get("outcome_aware_reranker")
        if isinstance(outcome_reranker_value, dict):
            feature_weights = {}
            for key, value in dict(outcome_reranker_value.get("feature_weights", {}) or {}).items():
                key_text = str(key)
                numeric_value = _to_float(value, None)
                if key_text not in _OUTCOME_AWARE_RERANKER_FEATURES or numeric_value is None or numeric_value <= 0.0:
                    continue
                feature_weights[key_text] = float(numeric_value)
            if feature_weights:
                outcome_aware_reranker = {
                    "feature_weights": feature_weights,
                    "bias": float(_to_float(outcome_reranker_value.get("bias"), 0.0) or 0.0),
                    "shortlist_size": int(
                        _to_float(
                            outcome_reranker_value.get("shortlist_size"),
                            max(40, 2 * len(_OUTCOME_AWARE_RERANKER_FEATURES)),
                        )
                        or max(40, 2 * len(_OUTCOME_AWARE_RERANKER_FEATURES))
                    ),
                }
        max_matches_value = _to_float(runtime_scope.get("max_matches_per_company"), None)
        if max_matches_value is not None and max_matches_value > 0.0:
            max_matches_per_company = max(1, int(round(float(max_matches_value))))

    if action_text.startswith("capital_structure.") or "debt" in action_text or "refinanc" in subtype_text:
        penalties["regime_rate_gap_threshold"] = min(
            float(_to_float(penalties.get("regime_rate_gap_threshold"), 0.75) or 0.75),
            0.75,
        )
        penalties["regime_rate_penalty_weight"] = max(
            float(_to_float(penalties.get("regime_rate_penalty_weight"), 0.90) or 0.90),
            0.90,
        )
        penalties["regime_credit_gap_threshold"] = min(
            float(_to_float(penalties.get("regime_credit_gap_threshold"), 0.90) or 0.90),
            0.90,
        )
        penalties["regime_credit_penalty_weight"] = max(
            float(_to_float(penalties.get("regime_credit_penalty_weight"), 1.00) or 1.00),
            1.00,
        )
        blend_weights.update({"state": 0.48, "regime": 0.22, "param": 0.10, "sector": 0.12, "action": 0.08})
        feature_relative_weights["state_vector_v1.liquidity_flexibility"] = (
            float(feature_relative_weights.get("state_vector_v1.liquidity_flexibility", 1.0)) * 0.70
        )
        feature_relative_weights["state_vector_v1.rates_level"] = (
            float(feature_relative_weights.get("state_vector_v1.rates_level", 1.0)) * 1.35
        )
        feature_relative_weights["state_vector_v1.credit_spread"] = (
            float(feature_relative_weights.get("state_vector_v1.credit_spread", 1.0)) * 1.50
        )
        critical.update({"state_vector_v1.market_access", "state_vector_v1.credit_spread", "state_vector_v1.rates_level"})

    normalized_group_weights = _normalize_weight_mapping(group_weights)
    feature_weights: Dict[str, float] = {}
    for feature_name in _STATE_VECTOR_MATCHING_COLS:
        group_name = _STATE_VECTOR_FEATURE_GROUP.get(feature_name, "identity")
        feature_weights[feature_name] = float(
            normalized_group_weights.get(group_name, 1.0) * float(feature_relative_weights.get(feature_name, 1.0))
        )
    feature_weights = _normalize_weight_mapping(feature_weights)

    return {
        "version": _WEIGHTED_DISTANCE_V2_VERSION,
        "weights": feature_weights,
        "feature_relative_weights": dict(feature_relative_weights),
        "feature_transform_mode": feature_transform_mode,
        "feature_transforms": dict(feature_transforms),
        "interaction_terms": list(interaction_terms),
        "latent_regime_model": dict(latent_regime_model),
        "latent_regime_penalty_weight": float(latent_regime_penalty_weight),
        "target_regime_mixture": dict(target_regime_mixture),
        "second_stage_reranker": dict(second_stage_reranker),
        "outcome_aware_reranker": dict(outcome_aware_reranker),
        "max_matches_per_company": int(max_matches_per_company),
        "group_weights": dict(normalized_group_weights),
        "critical_features": tuple(sorted(critical)),
        "min_weighted_coverage": float(_to_float(gates.get("min_weighted_coverage"), 0.75) or 0.75),
        "min_critical_coverage": float(_to_float(gates.get("min_critical_coverage"), 0.80) or 0.80),
        "max_size_gap": float(_to_float(gates.get("max_size_gap"), 1.15) or 1.15),
        "soft_size_gap": float(_to_float(gates.get("soft_size_gap"), 0.35) or 0.35),
        "primary_burden_feature": primary_burden_feature,
        "soft_burden_gap": float(_to_float(gates.get("soft_burden_gap"), 1.25) or 1.25),
        "distance_scale": float(_to_float(penalties.get("distance_scale"), 0.55) or 0.55),
        "missing_penalty_weight": float(_to_float(penalties.get("missing_penalty_weight"), 0.45) or 0.45),
        "critical_missing_penalty_weight": float(_to_float(penalties.get("critical_missing_penalty_weight"), 0.90) or 0.90),
        "size_penalty_weight": float(_to_float(penalties.get("size_penalty_weight"), 1.15) or 1.15),
        "burden_penalty_weight": float(_to_float(penalties.get("burden_penalty_weight"), 0.40) or 0.40),
        "sector_penalty_weight": float(_to_float(penalties.get("sector_penalty_weight"), 0.30) or 0.30),
        "regime_rate_gap_threshold": float(_to_float(penalties.get("regime_rate_gap_threshold"), 1.00) or 1.00),
        "regime_rate_penalty_weight": float(_to_float(penalties.get("regime_rate_penalty_weight"), 0.40) or 0.40),
        "regime_credit_gap_threshold": float(_to_float(penalties.get("regime_credit_gap_threshold"), 1.25) or 1.25),
        "regime_credit_penalty_weight": float(_to_float(penalties.get("regime_credit_penalty_weight"), 0.45) or 0.45),
        "blend_weights": dict(blend_weights),
        "weight_scope": scope_source,
    }


def _weighted_distance_profile(action_id: str, action_subtype: str) -> Dict[str, Any]:
    if _precedent_distance_profile_version(action_id, action_subtype) == _WEIGHTED_DISTANCE_V2_VERSION:
        return _weighted_distance_profile_v2(action_id, action_subtype)
    return _weighted_distance_profile_v1(action_id, action_subtype)


def _normalize_feature_transform_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"identity", "none", "explicit_only"}:
        return "identity"
    return _STATE_VECTOR_V2_DEFAULT_FEATURE_TRANSFORM_MODE


def _normalize_matching_transform_spec(spec: Any) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        return {}
    out: Dict[str, Any] = {}
    kind = str(spec.get("kind") or "").strip().lower()
    if kind and kind not in {"identity", "none"}:
        out["kind"] = kind
    cap = _to_float(spec.get("cap"), None)
    if cap is not None and np.isfinite(cap) and cap > 0.0:
        out["cap"] = float(cap)
    scale = _to_float(spec.get("scale"), None)
    if scale is not None and np.isfinite(scale) and scale > 0.0:
        out["scale"] = float(scale)
    return out


def _initialize_feature_transform_specs(mode: Any) -> Dict[str, Dict[str, float]]:
    normalized_mode = _normalize_feature_transform_mode(mode)
    if normalized_mode == "identity":
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for key, value in _STATE_VECTOR_V2_DEFAULT_FEATURE_TRANSFORMS.items():
        normalized = _normalize_matching_transform_spec(value)
        if normalized:
            out[key] = normalized
    return out


def _transform_matching_values(values: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    arr = np.array(values, dtype=float, copy=True)
    if arr.size == 0 or not isinstance(spec, dict):
        return arr
    kind = str(spec.get("kind") or "identity").strip().lower()
    valid = np.isfinite(arr)
    if not bool(np.any(valid)) or kind in {"", "identity", "none"}:
        return arr

    cap = _to_float(spec.get("cap"), None)
    scale = _to_float(spec.get("scale"), None)
    transformed = arr.copy()
    current = transformed[valid]
    if cap is not None and np.isfinite(cap):
        current = np.clip(current, -float(cap), float(cap))

    if kind == "signed_log1p_cap":
        transformed[valid] = np.sign(current) * np.log1p(np.abs(current))
    elif kind == "log1p_cap":
        transformed[valid] = np.log1p(np.clip(current, 0.0, None))
    elif kind == "signed_asinh":
        denom = float(scale) if scale is not None and np.isfinite(scale) and abs(scale) > 1e-12 else 1.0
        transformed[valid] = np.arcsinh(current / denom)
    elif kind == "clip":
        transformed[valid] = current
    else:
        transformed[valid] = current
    return transformed


def _apply_matching_feature_transforms(
    emb_raw: np.ndarray,
    candidate_vec_raw: np.ndarray,
    embedding_cols: Sequence[str],
    profile: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    transformed_emb = np.array(emb_raw, dtype=float, copy=True)
    transformed_candidate = np.array(candidate_vec_raw, dtype=float, copy=True)
    transform_specs = profile.get("feature_transforms") if isinstance(profile, dict) else None
    if not isinstance(transform_specs, dict) or not transform_specs:
        return transformed_emb, transformed_candidate, tuple()

    applied: List[str] = []
    for idx, feature_name in enumerate(embedding_cols):
        spec = transform_specs.get(feature_name)
        if not isinstance(spec, dict):
            continue
        transformed_emb[:, idx] = _transform_matching_values(transformed_emb[:, idx], spec)
        transformed_candidate[idx] = _transform_matching_values(
            np.array([transformed_candidate[idx]], dtype=float),
            spec,
        )[0]
        applied.append(str(feature_name))
    return transformed_emb, transformed_candidate, tuple(applied)


def _latent_regime_similarity_vector(
    emb_raw: np.ndarray,
    candidate_vec_raw: np.ndarray,
    embedding_cols: Sequence[str],
    model: Dict[str, Any],
) -> np.ndarray:
    feature_names = [str(name) for name in list(model.get("feature_names") or []) if str(name)]
    if not feature_names:
        return np.full(emb_raw.shape[0], np.nan, dtype=float)
    feature_index = {str(col): idx for idx, col in enumerate(embedding_cols)}
    hist_matrix = np.full((emb_raw.shape[0], len(feature_names)), np.nan, dtype=float)
    candidate_matrix = np.full((emb_raw.shape[0], len(feature_names)), np.nan, dtype=float)
    for out_idx, feature_name in enumerate(feature_names):
        raw_idx = feature_index.get(feature_name)
        if raw_idx is None:
            continue
        hist_matrix[:, out_idx] = emb_raw[:, raw_idx]
        candidate_matrix[:, out_idx] = candidate_vec_raw[raw_idx]
    try:
        return latent_regime_similarity(hist_matrix, candidate_matrix, model)
    except Exception:
        return np.full(emb_raw.shape[0], np.nan, dtype=float)


def _target_regime_membership_vector(
    candidate_vec_raw: np.ndarray,
    embedding_cols: Sequence[str],
    model: Dict[str, Any],
) -> np.ndarray:
    feature_names = [str(name) for name in list(model.get("feature_names") or []) if str(name)]
    if not feature_names:
        return np.empty(0, dtype=float)
    feature_index = {str(col): idx for idx, col in enumerate(embedding_cols)}
    target_matrix = np.full((1, len(feature_names)), np.nan, dtype=float)
    for out_idx, feature_name in enumerate(feature_names):
        raw_idx = feature_index.get(feature_name)
        if raw_idx is None:
            continue
        target_matrix[0, out_idx] = candidate_vec_raw[raw_idx]
    try:
        memberships = latent_regime_memberships(target_matrix, model)
    except Exception:
        return np.empty(0, dtype=float)
    if memberships.ndim != 2 or memberships.shape[0] != 1:
        return np.empty(0, dtype=float)
    return np.asarray(memberships[0], dtype=float)


def _weighted_state_similarity_v1(
    *,
    emb_raw: np.ndarray,
    candidate_vec_raw: np.ndarray,
    embedding_cols: Sequence[str],
    action_id: str,
    action_subtype: str,
    feature_weight_multipliers: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    profile = _weighted_distance_profile(action_id, action_subtype)
    if emb_raw.size == 0:
        empty = np.empty(0, dtype=float)
        return {
            "version": str(profile["version"]),
            "profile": profile,
            "state_similarity": empty,
            "weighted_distance": empty,
            "weighted_coverage": empty,
            "critical_coverage": empty,
            "coverage_gate_mask": np.empty(0, dtype=bool),
            "size_gate_mask": np.empty(0, dtype=bool),
            "size_gap": empty,
            "primary_burden_gap": empty,
        }

    emb_norm, candidate_vec = _winsorized_robust_standardize(emb_raw, candidate_vec_raw)

    weights = np.array([float(profile["weights"].get(col, 1.0)) for col in embedding_cols], dtype=float)
    if feature_weight_multipliers:
        weights = np.array(
            [
                float(weights[idx]) * float(_to_float(feature_weight_multipliers.get(col), 1.0) or 1.0)
                for idx, col in enumerate(embedding_cols)
            ],
            dtype=float,
        )
    cand_present = np.isfinite(candidate_vec_raw)
    row_present = np.isfinite(emb_raw)
    pair_present = row_present & cand_present.reshape(1, -1)

    total_weight = float(np.sum(weights[cand_present]))
    overlap_weight = np.sum(pair_present * weights.reshape(1, -1), axis=1)
    if total_weight > 1e-12:
        weighted_coverage = overlap_weight / total_weight
    else:
        weighted_coverage = np.ones(emb_raw.shape[0], dtype=float)

    critical_set = set(profile["critical_features"])
    critical_mask = np.array([(col in critical_set) and cand_present[idx] for idx, col in enumerate(embedding_cols)], dtype=bool)
    critical_total = float(np.sum(weights[critical_mask]))
    critical_overlap = np.sum(pair_present * (weights * critical_mask.astype(float)).reshape(1, -1), axis=1)
    if critical_total > 1e-12:
        critical_coverage = critical_overlap / critical_total
    else:
        critical_coverage = weighted_coverage.copy()

    diff_sq = np.square(emb_norm - candidate_vec.reshape(1, -1))
    weighted_distance = np.divide(
        np.sum(diff_sq * pair_present * weights.reshape(1, -1), axis=1),
        np.maximum(overlap_weight, 1e-12),
        out=np.full(emb_raw.shape[0], np.inf, dtype=float),
        where=overlap_weight > 1e-12,
    )

    size_gap = np.full(emb_raw.shape[0], np.nan, dtype=float)
    if "state_vector_v1.size_log_revenue" in embedding_cols:
        idx = embedding_cols.index("state_vector_v1.size_log_revenue")
        cand_size = candidate_vec_raw[idx]
        row_size = emb_raw[:, idx]
        if np.isfinite(cand_size):
            size_gap = np.abs(row_size - cand_size)

    primary_burden_gap = np.full(emb_raw.shape[0], np.nan, dtype=float)
    primary_feature = str(profile.get("primary_burden_feature") or "")
    if primary_feature in embedding_cols:
        idx = embedding_cols.index(primary_feature)
        cand_primary = candidate_vec_raw[idx]
        row_primary = emb_raw[:, idx]
        if np.isfinite(cand_primary):
            primary_burden_gap = np.abs(row_primary - cand_primary)

    base_similarity = np.exp(-float(profile["distance_scale"]) * np.clip(weighted_distance, 0.0, 25.0))
    coverage_factor = np.clip(0.60 + 0.40 * weighted_coverage, 0.0, 1.0)
    critical_factor = np.clip(0.55 + 0.45 * critical_coverage, 0.0, 1.0)
    size_penalty = np.ones(emb_raw.shape[0], dtype=float)
    size_penalty = np.where(
        np.isfinite(size_gap),
        np.exp(-0.90 * np.maximum(size_gap - 0.35, 0.0)),
        size_penalty,
    )
    burden_penalty = np.ones(emb_raw.shape[0], dtype=float)
    soft_burden_gap = float(profile.get("soft_burden_gap", 1.50) or 1.50)
    burden_penalty = np.where(
        np.isfinite(primary_burden_gap),
        np.exp(-0.35 * np.maximum(primary_burden_gap - soft_burden_gap, 0.0)),
        burden_penalty,
    )
    state_similarity = np.clip(base_similarity * coverage_factor * critical_factor * size_penalty * burden_penalty, 0.0, 1.0)

    coverage_gate_mask = (weighted_coverage >= float(profile["min_weighted_coverage"])) & (
        critical_coverage >= float(profile["min_critical_coverage"])
    )
    size_gate_mask = ~np.isfinite(size_gap) | (size_gap <= float(profile["max_size_gap"]))

    return {
        "version": str(profile["version"]),
        "profile": profile,
        "state_similarity": state_similarity,
        "weighted_distance": weighted_distance,
        "weighted_coverage": weighted_coverage,
        "critical_coverage": critical_coverage,
        "coverage_gate_mask": coverage_gate_mask,
        "size_gate_mask": size_gate_mask,
        "size_gap": size_gap,
        "primary_burden_gap": primary_burden_gap,
    }


def _weighted_state_similarity_v2(
    *,
    emb_raw: np.ndarray,
    candidate_vec_raw: np.ndarray,
    embedding_cols: Sequence[str],
    action_id: str,
    action_subtype: str,
    feature_weight_multipliers: Optional[Dict[str, float]] = None,
    target_action_scale: Optional[float] = None,
) -> Dict[str, Any]:
    profile = _weighted_distance_profile_v2(action_id, action_subtype)
    if emb_raw.size == 0:
        empty = np.empty(0, dtype=float)
        return {
            "version": str(profile["version"]),
            "profile": profile,
            "state_similarity": empty,
            "weighted_distance": empty,
            "weighted_coverage": empty,
            "critical_coverage": empty,
            "coverage_gate_mask": np.empty(0, dtype=bool),
            "size_gate_mask": np.empty(0, dtype=bool),
            "size_gap": empty,
            "primary_burden_gap": empty,
            "rate_gap": empty,
            "credit_gap": empty,
            "missing_penalty_factor": empty,
            "regime_penalty_factor": empty,
            "latent_regime_similarity": empty,
            "latent_regime_penalty_factor": empty,
            "target_regime_membership": empty,
        }

    transformed_emb_raw, transformed_candidate_vec_raw, transformed_features = _apply_matching_feature_transforms(
        emb_raw,
        candidate_vec_raw,
        embedding_cols,
        profile,
    )
    emb_norm, candidate_vec = _winsorized_robust_standardize(transformed_emb_raw, transformed_candidate_vec_raw)

    effective_feature_relative_weights = dict(profile.get("feature_relative_weights") or {})
    effective_interaction_terms = list(profile.get("interaction_terms") or [])
    target_regime_membership = np.empty(0, dtype=float)
    blended_latent_regime_penalty_weight = 0.0
    blended_target_regime_model: Optional[Dict[str, Any]] = None
    target_regime_mixture = profile.get("target_regime_mixture")
    if isinstance(target_regime_mixture, dict) and isinstance(target_regime_mixture.get("model"), dict):
        blended_target_regime_model = dict(target_regime_mixture.get("model") or {})
        target_regime_membership = _target_regime_membership_vector(
            candidate_vec_raw,
            embedding_cols,
            blended_target_regime_model,
        )
        regimes = list(target_regime_mixture.get("regimes") or [])
        if target_regime_membership.size and regimes:
            blended_feature_weights: Dict[str, float] = {}
            blended_interaction_weights: Dict[Tuple[str, str], float] = {}
            for cluster_idx, membership_value in enumerate(target_regime_membership.tolist()):
                if cluster_idx >= len(regimes):
                    break
                membership_float = float(membership_value)
                if membership_float <= 0.0:
                    continue
                regime_payload = dict(regimes[cluster_idx] or {})
                for feature_name, weight_value in dict(regime_payload.get("feature_relative_weights") or {}).items():
                    blended_feature_weights[str(feature_name)] = blended_feature_weights.get(str(feature_name), 0.0) + membership_float * float(weight_value)
                for item in list(regime_payload.get("interaction_terms") or []):
                    if not isinstance(item, dict):
                        continue
                    features = list(item.get("features") or [])
                    if len(features) != 2:
                        continue
                    key = tuple(sorted((str(features[0] or ""), str(features[1] or ""))))
                    if not key[0] or not key[1]:
                        continue
                    blended_interaction_weights[key] = blended_interaction_weights.get(key, 0.0) + membership_float * float(_to_float(item.get("weight"), 0.0) or 0.0)
                blended_latent_regime_penalty_weight += membership_float * float(
                    _to_float(regime_payload.get("latent_regime_penalty_weight"), 0.0) or 0.0
                )
            if blended_feature_weights:
                effective_feature_relative_weights.update(blended_feature_weights)
            if blended_interaction_weights:
                effective_interaction_terms = [
                    {"features": list(key), "weight": float(weight)}
                    for key, weight in blended_interaction_weights.items()
                    if float(weight) > 0.0
                ]

    action_text = str(action_id or "").strip().lower()
    if _is_debt_support_action(action_text):
        target_compact_features = {
            str(col): float(candidate_vec_raw[idx])
            for idx, col in enumerate(embedding_cols)
            if col in _STATE_VECTOR_MATCHING_COLS and np.isfinite(candidate_vec_raw[idx])
        }
        target_debt_profile = _debt_issuance_runtime_archetype_profile(
            target_compact_features,
            action_id_text=action_text,
            action_scale=target_action_scale,
        )
        for feature_name, multiplier in _debt_issuance_target_feature_multipliers(
            target_debt_profile,
            action_id_text=action_text,
        ).items():
            if feature_name not in effective_feature_relative_weights:
                continue
            effective_feature_relative_weights[feature_name] = float(
                effective_feature_relative_weights.get(feature_name, 1.0) * float(multiplier)
            )
        profile["debt_target_archetype_label"] = str(target_debt_profile.get("label") or "")
        profile["debt_target_archetype_scores"] = {
            str(key): float(value)
            for key, value in dict(target_debt_profile.get("scores") or {}).items()
            if _to_float(value, None) is not None
        }

    for feature_name, multiplier in dict(feature_weight_multipliers or {}).items():
        if feature_name not in effective_feature_relative_weights:
            continue
        numeric_multiplier = _to_float(multiplier, None)
        if numeric_multiplier is None or numeric_multiplier <= 0.0:
            continue
        effective_feature_relative_weights[feature_name] = float(
            effective_feature_relative_weights.get(feature_name, 1.0) * float(numeric_multiplier)
        )

    effective_feature_weights_map = _normalize_weight_mapping(
        {
            col: float(profile["group_weights"].get(_STATE_VECTOR_FEATURE_GROUP.get(col, "identity"), 1.0))
            * float(effective_feature_relative_weights.get(col, 1.0))
            for col in embedding_cols
        }
    )
    feature_weights = np.array([float(effective_feature_weights_map.get(col, 1.0)) for col in embedding_cols], dtype=float)
    feature_relative_weights = np.array(
        [float(effective_feature_relative_weights.get(col, 1.0)) for col in embedding_cols],
        dtype=float,
    )
    cand_present = np.isfinite(candidate_vec_raw)
    row_present = np.isfinite(emb_raw)
    pair_present = row_present & cand_present.reshape(1, -1)

    total_weight = float(np.sum(feature_weights[cand_present]))
    overlap_weight = np.sum(pair_present * feature_weights.reshape(1, -1), axis=1)
    if total_weight > 1e-12:
        weighted_coverage = overlap_weight / total_weight
    else:
        weighted_coverage = np.ones(emb_raw.shape[0], dtype=float)

    critical_set = set(profile["critical_features"])
    critical_mask = np.array([(col in critical_set) and cand_present[idx] for idx, col in enumerate(embedding_cols)], dtype=bool)
    critical_total = float(np.sum(feature_weights[critical_mask]))
    critical_overlap = np.sum(pair_present * (feature_weights * critical_mask.astype(float)).reshape(1, -1), axis=1)
    if critical_total > 1e-12:
        critical_coverage = critical_overlap / critical_total
    else:
        critical_coverage = weighted_coverage.copy()

    diff_sq = np.square(emb_norm - candidate_vec.reshape(1, -1))
    abs_diff = np.abs(emb_norm - candidate_vec.reshape(1, -1))
    group_distance_terms: List[np.ndarray] = []
    group_denoms: List[np.ndarray] = []
    for group_name, feature_names in _STATE_VECTOR_GROUPS.items():
        group_weight = float(profile["group_weights"].get(group_name, 1.0))
        if group_weight <= 0.0:
            continue
        mask = np.array([(col in feature_names) for col in embedding_cols], dtype=bool)
        if not bool(np.any(mask)):
            continue
        cand_group_mask = cand_present & mask
        group_total = float(np.sum(feature_relative_weights[cand_group_mask]))
        group_overlap = np.sum(
            pair_present * (feature_relative_weights * mask.astype(float)).reshape(1, -1),
            axis=1,
        )
        group_distance = np.divide(
            np.sum(diff_sq * pair_present * (feature_relative_weights * mask.astype(float)).reshape(1, -1), axis=1),
            np.maximum(group_overlap, 1e-12),
            out=np.full(emb_raw.shape[0], np.nan, dtype=float),
            where=group_overlap > 1e-12,
        )
        group_present = group_overlap > 1e-12
        weighted_group_weight = np.where(group_present, group_weight, 0.0)
        group_distance_terms.append(np.where(np.isfinite(group_distance), group_distance * weighted_group_weight, 0.0))
        group_denoms.append(weighted_group_weight)
        if group_total > 1e-12:
            profile[f"{group_name}_coverage_mean"] = float(np.nanmean(group_overlap[group_present] / group_total)) if bool(np.any(group_present)) else 0.0

    if group_distance_terms and group_denoms:
        distance_numer = np.sum(np.vstack(group_distance_terms), axis=0)
        distance_denom = np.sum(np.vstack(group_denoms), axis=0)
        weighted_distance = np.divide(
            distance_numer,
            np.maximum(distance_denom, 1e-12),
            out=np.full(emb_raw.shape[0], np.inf, dtype=float),
            where=distance_denom > 1e-12,
        )
    else:
        weighted_distance = np.full(emb_raw.shape[0], np.inf, dtype=float)

    interaction_distance = np.zeros(emb_raw.shape[0], dtype=float)
    interaction_denom = np.zeros(emb_raw.shape[0], dtype=float)
    feature_index = {col: idx for idx, col in enumerate(embedding_cols)}
    for item in list(effective_interaction_terms or []):
        features = tuple(item.get("features") or ())
        if len(features) != 2:
            continue
        left_idx = feature_index.get(str(features[0]))
        right_idx = feature_index.get(str(features[1]))
        if left_idx is None or right_idx is None:
            continue
        weight = float(_to_float(item.get("weight"), 0.0) or 0.0)
        if weight <= 0.0:
            continue
        term_present = pair_present[:, left_idx] & pair_present[:, right_idx]
        term_value = abs_diff[:, left_idx] * abs_diff[:, right_idx]
        interaction_distance += np.where(term_present, term_value * weight, 0.0)
        interaction_denom += np.where(term_present, weight, 0.0)
    interaction_distance = np.divide(
        interaction_distance,
        np.maximum(interaction_denom, 1e-12),
        out=np.zeros(emb_raw.shape[0], dtype=float),
        where=interaction_denom > 1e-12,
    )

    size_gap = np.full(emb_raw.shape[0], np.nan, dtype=float)
    if "state_vector_v1.size_log_revenue" in embedding_cols:
        idx = embedding_cols.index("state_vector_v1.size_log_revenue")
        cand_size = candidate_vec_raw[idx]
        row_size = emb_raw[:, idx]
        if np.isfinite(cand_size):
            size_gap = np.abs(row_size - cand_size)

    primary_burden_gap = np.full(emb_raw.shape[0], np.nan, dtype=float)
    primary_feature = str(profile.get("primary_burden_feature") or "")
    if primary_feature in embedding_cols:
        idx = embedding_cols.index(primary_feature)
        cand_primary = candidate_vec_raw[idx]
        row_primary = emb_raw[:, idx]
        if np.isfinite(cand_primary):
            primary_burden_gap = np.abs(row_primary - cand_primary)

    rate_gap = np.full(emb_raw.shape[0], np.nan, dtype=float)
    if "state_vector_v1.rates_level" in embedding_cols:
        idx = embedding_cols.index("state_vector_v1.rates_level")
        cand_rate = candidate_vec_raw[idx]
        row_rate = emb_raw[:, idx]
        if np.isfinite(cand_rate):
            rate_gap = np.abs(row_rate - cand_rate)

    credit_gap = np.full(emb_raw.shape[0], np.nan, dtype=float)
    if "state_vector_v1.credit_spread" in embedding_cols:
        idx = embedding_cols.index("state_vector_v1.credit_spread")
        cand_credit = candidate_vec_raw[idx]
        row_credit = emb_raw[:, idx]
        if np.isfinite(cand_credit):
            credit_gap = np.abs(row_credit - cand_credit)

    base_similarity = np.exp(-float(profile["distance_scale"]) * np.clip(weighted_distance, 0.0, 40.0))
    interaction_penalty_factor = np.where(
        interaction_denom > 1e-12,
        np.exp(-float(profile["distance_scale"]) * np.clip(interaction_distance, 0.0, 40.0)),
        1.0,
    )
    missing_penalty_factor = np.exp(
        -float(profile["missing_penalty_weight"]) * np.maximum(1.0 - weighted_coverage, 0.0)
        -float(profile["critical_missing_penalty_weight"]) * np.maximum(1.0 - critical_coverage, 0.0)
    )
    soft_size_gap = float(profile.get("soft_size_gap", 0.35) or 0.35)
    size_penalty = np.where(
        np.isfinite(size_gap),
        np.exp(-float(profile["size_penalty_weight"]) * np.maximum(size_gap - soft_size_gap, 0.0)),
        1.0,
    )
    soft_burden_gap = float(profile.get("soft_burden_gap", 1.25) or 1.25)
    burden_penalty = np.where(
        np.isfinite(primary_burden_gap),
        np.exp(-float(profile["burden_penalty_weight"]) * np.maximum(primary_burden_gap - soft_burden_gap, 0.0)),
        1.0,
    )
    regime_penalty_factor = np.ones(emb_raw.shape[0], dtype=float)
    rate_threshold = float(profile.get("regime_rate_gap_threshold", 1.0) or 1.0)
    credit_threshold = float(profile.get("regime_credit_gap_threshold", 1.25) or 1.25)
    regime_penalty_factor = np.where(
        np.isfinite(rate_gap),
        regime_penalty_factor
        * np.exp(-float(profile["regime_rate_penalty_weight"]) * np.maximum(rate_gap - rate_threshold, 0.0)),
        regime_penalty_factor,
    )
    regime_penalty_factor = np.where(
        np.isfinite(credit_gap),
        regime_penalty_factor
        * np.exp(-float(profile["regime_credit_penalty_weight"]) * np.maximum(credit_gap - credit_threshold, 0.0)),
        regime_penalty_factor,
    )
    borrower_quality_similarity = np.ones(emb_raw.shape[0], dtype=float)
    financing_pressure_similarity = np.ones(emb_raw.shape[0], dtype=float)
    market_regime_similarity = np.ones(emb_raw.shape[0], dtype=float)
    stress_alignment_similarity = np.ones(emb_raw.shape[0], dtype=float)
    compatibility_penalty_factor = np.ones(emb_raw.shape[0], dtype=float)
    if _is_debt_support_action(action_text):
        debt_compatibility = _debt_issuance_pairwise_compatibility(
            emb_raw=emb_raw,
            candidate_vec_raw=candidate_vec_raw,
            embedding_cols=embedding_cols,
            action_id_text=action_text,
            feature_weight_multipliers=feature_weight_multipliers,
        )
        borrower_quality_similarity = np.asarray(
            debt_compatibility.get("borrower_quality_similarity", borrower_quality_similarity),
            dtype=float,
        )
        financing_pressure_similarity = np.asarray(
            debt_compatibility.get("financing_pressure_similarity", financing_pressure_similarity),
            dtype=float,
        )
        market_regime_similarity = np.asarray(
            debt_compatibility.get("market_regime_similarity", market_regime_similarity),
            dtype=float,
        )
        stress_alignment_similarity = np.asarray(
            debt_compatibility.get("stress_alignment_similarity", stress_alignment_similarity),
            dtype=float,
        )
        compatibility_penalty_factor = np.asarray(
            debt_compatibility.get("compatibility_penalty_factor", compatibility_penalty_factor),
            dtype=float,
        )
        regime_penalty_factor = regime_penalty_factor * compatibility_penalty_factor
    latent_regime_similarity_values = np.full(emb_raw.shape[0], np.nan, dtype=float)
    latent_regime_penalty_factor = np.ones(emb_raw.shape[0], dtype=float)
    latent_model = profile.get("latent_regime_model")
    latent_weight = float(_to_float(profile.get("latent_regime_penalty_weight"), 0.0) or 0.0)
    if blended_latent_regime_penalty_weight > 0.0 and isinstance(blended_target_regime_model, dict):
        latent_model = dict(blended_target_regime_model)
        latent_weight = float(latent_weight) + float(blended_latent_regime_penalty_weight)
    if latent_weight > 0.0 and isinstance(latent_model, dict):
        latent_regime_similarity_values = _latent_regime_similarity_vector(
            emb_raw,
            candidate_vec_raw,
            embedding_cols,
            latent_model,
        )
        latent_regime_penalty_factor = np.where(
            np.isfinite(latent_regime_similarity_values),
            np.exp(-latent_weight * np.maximum(1.0 - latent_regime_similarity_values, 0.0)),
            1.0,
        )

    state_similarity = np.clip(
        base_similarity
        * interaction_penalty_factor
        * missing_penalty_factor
        * size_penalty
        * burden_penalty
        * regime_penalty_factor
        * latent_regime_penalty_factor,
        0.0,
        1.0,
    )
    coverage_gate_mask = (weighted_coverage >= float(profile["min_weighted_coverage"])) & (
        critical_coverage >= float(profile["min_critical_coverage"])
    )
    size_gate_mask = ~np.isfinite(size_gap) | (size_gap <= float(profile["max_size_gap"]))
    profile["candidate_feature_weight_multipliers"] = {
        str(key): float(value)
        for key, value in dict(feature_weight_multipliers or {}).items()
        if _to_float(value, None) is not None and float(value) != 1.0
    }

    return {
        "version": str(profile["version"]),
        "profile": profile,
        "transformed_features": transformed_features,
        "state_similarity": state_similarity,
        "weighted_distance": weighted_distance,
        "weighted_coverage": weighted_coverage,
        "critical_coverage": critical_coverage,
        "coverage_gate_mask": coverage_gate_mask,
        "size_gate_mask": size_gate_mask,
        "size_gap": size_gap,
        "primary_burden_gap": primary_burden_gap,
        "rate_gap": rate_gap,
        "credit_gap": credit_gap,
        "interaction_distance": interaction_distance,
        "interaction_penalty_factor": interaction_penalty_factor,
        "missing_penalty_factor": missing_penalty_factor,
        "regime_penalty_factor": regime_penalty_factor,
        "borrower_quality_similarity": borrower_quality_similarity,
        "financing_pressure_similarity": financing_pressure_similarity,
        "market_regime_similarity": market_regime_similarity,
        "stress_alignment_similarity": stress_alignment_similarity,
        "compatibility_penalty_factor": compatibility_penalty_factor,
        "latent_regime_similarity": latent_regime_similarity_values,
        "latent_regime_penalty_factor": latent_regime_penalty_factor,
        "target_regime_membership": target_regime_membership,
    }


def _weighted_state_similarity(
    *,
    emb_raw: np.ndarray,
    candidate_vec_raw: np.ndarray,
    embedding_cols: Sequence[str],
    action_id: str,
    action_subtype: str,
    feature_weight_multipliers: Optional[Dict[str, float]] = None,
    target_action_scale: Optional[float] = None,
) -> Dict[str, Any]:
    if _precedent_distance_profile_version() == _WEIGHTED_DISTANCE_V2_VERSION:
        return _weighted_state_similarity_v2(
            emb_raw=emb_raw,
            candidate_vec_raw=candidate_vec_raw,
            embedding_cols=embedding_cols,
            action_id=action_id,
            action_subtype=action_subtype,
            feature_weight_multipliers=feature_weight_multipliers,
            target_action_scale=target_action_scale,
        )
    return _weighted_state_similarity_v1(
        emb_raw=emb_raw,
        candidate_vec_raw=candidate_vec_raw,
        embedding_cols=embedding_cols,
        action_id=action_id,
        action_subtype=action_subtype,
        feature_weight_multipliers=feature_weight_multipliers,
    )


def augment_precedent_state_vector_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        for col in _STATE_VECTOR_MATCHING_COLS:
            out[col] = pd.Series(dtype=float)
        return out

    revenue = _first_numeric_series(
        out,
        (
            "scale.revenue_ttm",
            "operating.revenue_ttm_provider_direct",
            "operating.revenue_ttm",
            "base_revenue_ttm",
            "revenue_ttm",
        ),
    )
    direct_revenue = _first_numeric_series(
        out,
        (
            "scale.revenue_ttm",
            "operating.revenue_ttm_provider_direct",
            "operating.revenue_ttm",
            "revenue_ttm",
        ),
    )
    historical_revenue = _first_numeric_series(out, ("base_revenue_ttm",))
    revenue_for_size = revenue.copy()
    historical_revenue_only = direct_revenue.isna() & historical_revenue.notna()
    # Historical action-outcome baselines are stored in source units (generally USD millions),
    # while live snapshots surface dollar values. Normalize the size feature onto a common dollar basis.
    revenue_for_size = revenue_for_size.where(~historical_revenue_only, historical_revenue * 1_000_000.0)
    size_log_revenue = _first_numeric_series(out, ("state_vector_v1.size_log_revenue",))
    size_log_revenue = size_log_revenue.where(size_log_revenue.notna(), _safe_log10_series(revenue_for_size))

    ebitda = _first_numeric_series(
        out,
        (
            "scale.ebitda_ttm",
            "base_ebitda_ttm",
            "operating.ebitda_ltm_provider_direct",
            "operating.operating_earnings_normalized",
            "earnings.ebitda_ttm_provider_direct",
            "ebitda_ttm",
            "ebitda_ltm",
            "ebitda",
        ),
    )
    profitability = _first_numeric_series(out, ("state_vector_v1.profitability",))
    derived_profitability = ebitda / revenue.where(revenue > 0)
    profitability = profitability.where(profitability.notna(), derived_profitability)
    profitability = profitability.where(
        profitability.notna(),
        _first_numeric_series(out, ("base_margin", "ebitda_margin")),
    )

    growth = _first_numeric_series(out, ("state_vector_v1.growth",))
    revenue_lag = _first_numeric_series(
        out,
        (
            "operating.revenue_ttm_lag_1y",
            "base_revenue_ttm_lag_1y",
            "operating.revenue_ttm_prior_year",
            "operating.revenue_ttm_prev_year",
            "revenue_ttm_lag_1y",
        ),
    )
    derived_growth = (revenue / revenue_lag.where(revenue_lag > 0)) - 1.0
    growth = growth.where(growth.notna(), derived_growth)
    growth = growth.where(
        growth.notna(),
        _first_numeric_series(out, ("base_revenue_growth_yoy", "revenue_yoy_last_q", "revenue_yoy", "growth_revenue_yoy")),
    )

    gross_debt = _first_numeric_series(
        out,
        (
            "capital.total_debt",
            "base_total_debt",
            "capital_structure.total_debt_provider_direct",
            "capital_structure.total_debt_reported",
            "capital_structure.total_debt",
            "gross_debt",
        ),
    )
    lease_liabilities = _first_numeric_series(
        out,
        (
            "capital.lease_liabilities",
            "capital_structure.lease_liabilities_sec_exact",
            "lease_liabilities",
        ),
    )
    retirement_liabilities = _first_numeric_series(
        out,
        (
            "capital.combined_retirement_liability",
            "capital_structure.combined_retirement_liability",
            "combined_retirement_liability",
            "capital.net_pension_liability",
            "capital_structure.net_pension_liability",
            "net_pension_liability",
        ),
    )
    gross_obligation_burden = _first_numeric_series(out, ("state_vector_v1.gross_obligation_burden",))
    derived_gross_obligation_burden = (
        gross_debt.fillna(0.0)
        + lease_liabilities.fillna(0.0)
        + retirement_liabilities.fillna(0.0)
    ) / ebitda.where(ebitda > 0)
    gross_obligation_burden = gross_obligation_burden.where(
        gross_obligation_burden.notna(),
        derived_gross_obligation_burden,
    )
    gross_obligation_burden = gross_obligation_burden.where(
        gross_obligation_burden.notna(),
        _first_numeric_series(
            out,
            (
                "capital.gross_leverage_including_retirement",
                "capital_structure.gross_leverage_including_retirement",
                "gross_leverage_including_retirement",
            ),
        ),
    )

    net_debt = _first_numeric_series(
        out,
        (
            "capital.net_debt",
            "capital_structure.net_debt_normalized",
            "capital_structure.net_debt_standardized",
            "capital_structure.net_debt",
            "base_net_debt",
            "net_debt",
        ),
    )
    net_obligation_burden = _first_numeric_series(out, ("state_vector_v1.net_obligation_burden",))
    derived_net_obligation_burden = (
        net_debt.fillna(0.0)
        + retirement_liabilities.fillna(0.0)
    ) / ebitda.where(ebitda > 0)
    net_obligation_burden = net_obligation_burden.where(
        net_obligation_burden.notna(),
        derived_net_obligation_burden,
    )
    net_obligation_burden = net_obligation_burden.where(
        net_obligation_burden.notna(),
        _first_numeric_series(
            out,
            (
                "capital.net_leverage_including_retirement",
                "capital_structure.net_leverage_including_retirement",
                "net_leverage_including_retirement",
                "base_leverage",
                "leverage_net_debt_ebitda",
            ),
        ),
    )

    liquidity_ratio = _first_numeric_series(out, ("state_vector_v1.liquidity_flexibility",))
    cash_liquidity = _first_numeric_series(
        out,
        (
            "base_available_liquidity",
            "base_cash",
            "liquidity.cash_and_short_term_investments_provider_direct",
            "cash_and_short_term_investments",
            "liquidity.cash",
            "cash",
        ),
    )
    marketable_securities = _first_numeric_series(
        out,
        (
            "liquidity.marketable_securities",
            "liquidity.marketable_securities_sec_exact",
            "marketable_securities",
        ),
    )
    undrawn_revolver = _first_numeric_series(
        out,
        (
            "liquidity.revolver_undrawn",
            "revolver_undrawn",
        ),
    )
    available_liquidity = _first_numeric_series(
        out,
        (
            "liquidity.available_liquidity_normalized",
            "available_liquidity_normalized",
            "liquidity.available_for_actions",
            "available_for_actions",
        ),
    )
    near_term_debt = _first_numeric_series(
        out,
        (
            "capital.debt_due_next_24m",
            "debt_due_next_24m",
            "capital.debt_due_0_12m",
            "debt_due_0_12m",
            "base_current_debt",
            "capital.current_debt",
            "capital_structure.current_debt_statement_direct",
            "current_debt",
        ),
    )
    derived_available_liquidity = (
        cash_liquidity.fillna(0.0)
        + marketable_securities.fillna(0.0)
        + undrawn_revolver.fillna(0.0)
    )
    derived_available_liquidity = derived_available_liquidity.where(cash_liquidity.notna(), np.nan)
    available_liquidity = available_liquidity.where(available_liquidity.notna(), derived_available_liquidity)
    derived_liquidity_ratio = available_liquidity / near_term_debt.where(near_term_debt > 0)
    liquidity_ratio = liquidity_ratio.where(liquidity_ratio.notna(), derived_liquidity_ratio)

    interest_coverage = _first_numeric_series(out, ("state_vector_v1.interest_coverage", "capital.interest_coverage", "interest_coverage"))
    interest_expense = _first_numeric_series(
        out,
        (
            "base_interest_expense",
            "capital.interest_expense",
            "capital_structure.interest_expense_statement_direct",
            "interest_expense",
            "interest_expense_ttm",
        ),
    )
    derived_interest_coverage = ebitda / interest_expense.where(interest_expense > 0)
    interest_coverage = interest_coverage.where(interest_coverage.notna(), derived_interest_coverage)

    valuation_multiple = _first_numeric_series(
        out,
        (
            "state_vector_v1.valuation_multiple",
            "market.ev_ebitda",
            "ev_ebitda",
            "base_ev_ebitda",
        ),
    )
    market_cap = _first_numeric_series(
        out,
        (
            "scale.market_cap",
            "market.market_cap_provider_direct",
            "market.market_cap",
            "market_cap",
        ),
    )
    historical_market_cap = _first_numeric_series(out, ("base_market_cap",))
    market_cap = market_cap.where(
        market_cap.notna(),
        _normalize_market_cap_series_to_dollars(historical_market_cap, prefer_source_units=True),
    )
    cash_and_sti = cash_liquidity
    derived_valuation_multiple = (
        market_cap
        + gross_debt
        + lease_liabilities.fillna(0.0)
        - cash_and_sti
    ) / ebitda.where(ebitda > 0)
    derived_valuation_multiple = derived_valuation_multiple.where(
        market_cap.notna() & gross_debt.notna() & cash_and_sti.notna(),
        np.nan,
    )
    valuation_multiple = valuation_multiple.where(valuation_multiple.notna(), derived_valuation_multiple)

    cash_generation = _first_numeric_series(
        out,
        (
            "state_vector_v1.cash_generation",
            "base_fcf_yield",
            "market.fcf_yield",
            "fcf_yield",
            "base_fcf_margin",
            "fcf_margin",
        ),
    )
    free_cash_flow = _first_numeric_series(
        out,
        (
            "cash_flow.free_cash_flow_ttm",
            "operating.free_cash_flow_ttm",
            "free_cash_flow_ttm",
        ),
    )
    derived_cash_generation = free_cash_flow / market_cap.where(market_cap > 0)
    cash_generation = cash_generation.where(cash_generation.notna(), derived_cash_generation)

    market_stress = _first_numeric_series(out, ("state_vector_v1.market_stress",))
    vol_90d = _first_numeric_series(out, ("market.volatility_90d", "volatility_90d", "base_volatility_90d"))
    drawdown_90d = _first_numeric_series(out, ("market.drawdown_90d", "drawdown_90d", "base_drawdown_90d"))
    macro_vix = _first_numeric_series(out, ("macro_vix", "market.vix"))
    derived_market_stress = _weighted_average_series(
        (
            (vol_90d, 0.6),
            (drawdown_90d.abs(), 0.4),
        )
    )
    market_stress = market_stress.where(market_stress.notna(), derived_market_stress)
    market_stress = market_stress.where(
        market_stress.notna(),
        (macro_vix / 80.0).clip(lower=0.0, upper=1.0),
    )

    market_access = _first_numeric_series(out, ("state_vector_v1.market_access",))
    vol_30d = _first_numeric_series(out, ("market.volatility_30d", "volatility_30d", "base_volatility_30d"))
    momentum_60d = _first_numeric_series(out, ("market.momentum_60d", "momentum_60d", "base_momentum_60d"))
    credit_window_proxy = _first_numeric_series(
        out,
        ("market.credit_window_proxy", "credit_window_proxy", "base_credit_window_proxy"),
    )
    equity_window_proxy = _first_numeric_series(
        out,
        ("market.equity_window_proxy", "equity_window_proxy", "base_equity_window_proxy"),
    )
    credit_spread_level = _first_numeric_series(
        out,
        ("market.credit_spread_level", "credit_spread_level", "base_credit_spread_level"),
    )
    derived_equity_window = _weighted_average_series(
        (
            ((1.0 - (vol_30d / 0.8)).clip(lower=0.0, upper=1.0), 1.0),
            (((momentum_60d + 0.2) / 0.4).clip(lower=0.0, upper=1.0), 1.0),
            ((valuation_multiple / 20.0).clip(lower=0.0, upper=1.0), 1.0),
        )
    )
    equity_window_proxy = equity_window_proxy.where(equity_window_proxy.notna(), derived_equity_window)
    spread_access = (1.0 - (credit_spread_level / 0.08)).clip(lower=0.0, upper=1.0)
    derived_credit_window = _weighted_average_series(
        (
            ((1.0 - (credit_spread_level / 0.10)).clip(lower=0.0, upper=1.0), 1.0),
            ((1.0 - (vol_30d / 1.0)).clip(lower=0.0, upper=1.0), 1.0),
        )
    )
    credit_window_proxy = credit_window_proxy.where(credit_window_proxy.notna(), derived_credit_window)
    derived_market_access = _weighted_average_series(
        (
            (credit_window_proxy, 0.4),
            (equity_window_proxy, 0.4),
            (spread_access, 0.2),
        )
    )
    market_access = market_access.where(market_access.notna(), derived_market_access)

    rates_level = _first_numeric_series(
        out,
        (
            "state_vector_v1.rates_level",
            "macro.fed_funds_effective",
            "macro_fed_funds_effective",
            "macro_rate_10y",
            "macro_ust_10y",
            "macro_10y_treasury",
        ),
    )
    credit_spread = _first_numeric_series(
        out,
        (
            "state_vector_v1.credit_spread",
            "macro.hy_oas",
            "macro_hy_oas",
            "macro_credit_spread",
        ),
    )

    out["state_vector_v1.size_log_revenue"] = size_log_revenue
    out["state_vector_v1.profitability"] = profitability
    out["state_vector_v1.growth"] = growth
    out["state_vector_v1.gross_obligation_burden"] = gross_obligation_burden
    out["state_vector_v1.net_obligation_burden"] = net_obligation_burden
    out["state_vector_v1.liquidity_flexibility"] = liquidity_ratio
    out["state_vector_v1.interest_coverage"] = interest_coverage
    out["state_vector_v1.valuation_multiple"] = valuation_multiple
    out["state_vector_v1.cash_generation"] = cash_generation
    out["state_vector_v1.market_stress"] = market_stress
    out["state_vector_v1.market_access"] = market_access
    out["state_vector_v1.rates_level"] = rates_level
    out["state_vector_v1.credit_spread"] = credit_spread
    return out


def _combined_family_key(family: str, subfamily: str) -> str:
    fam = str(family or "").strip()
    sub = str(subfamily or "").strip()
    if fam and sub:
        return f"{fam}.{sub}"
    return fam or sub


def _historical_exact_action_id(action_type: Any, action_subtype: Any) -> str:
    at = _canonical_token(action_type)
    st = _canonical_token(action_subtype)
    if at == "dividend_initiate" and st == "dividend_initiate":
        return "capital_return.dividend_initiate"
    if at == "acquisition" and st == "acquisition_lbo":
        return "mna.go_private_lbo"
    return ""


def _resolved_normalized_subfamily(
    normalized_family: Any,
    normalized_subfamily: Any,
    raw_action_type: Any,
    raw_action_subtype: Any,
) -> str:
    family = str(normalized_family or "").strip()
    subfamily = str(normalized_subfamily or "").strip()
    raw_at = _canonical_token(raw_action_type)
    raw_st = _canonical_token(raw_action_subtype)
    if family == "capital_structure" and subfamily == "refinancing" and raw_at == "loan_refinancing":
        return _normalized_refinancing_action_subtype(raw_action_subtype)
    if family == "mna" and subfamily == "platform_control":
        if raw_at == "acquisition" and raw_st == "acquisition_lbo":
            return "platform_lbo"
        if raw_at == "acquisition" and raw_st == "acquisition_merger":
            return "platform_merger"
    return subfamily


def _preferred_row_action_fields(row: pd.Series) -> Tuple[str, str, str]:
    action_id = str(
        row.get("normalized_action_id")
        or row.get("action_id")
        or _historical_exact_action_id(
            row.get("raw_action_type") or row.get("action_type"),
            row.get("raw_action_subtype") or row.get("action_subtype"),
        )
        or ""
    ).strip()
    action_subfamily = str(
        _resolved_normalized_subfamily(
            row.get("normalized_action_family"),
            row.get("normalized_action_subfamily"),
            row.get("raw_action_type") or row.get("action_type"),
            row.get("raw_action_subtype") or row.get("action_subtype"),
        )
        or row.get("raw_action_subtype")
        or row.get("action_subtype")
        or ""
    ).strip()
    action_family = str(
        row.get("normalized_action_family")
        or row.get("raw_action_type")
        or row.get("action_type")
        or ""
    ).strip()
    return action_id, action_subfamily, action_family


class PrecedentRetrievalIndex:
    """Read-mostly retrieval index to speed repeated precedent lookups."""

    def __init__(self, historical_df: pd.DataFrame) -> None:
        df = historical_df.copy()
        if "action_date" in df.columns:
            df["action_date"] = pd.to_datetime(df["action_date"], utc=True, errors="coerce")
        df = df.reset_index(drop=True)
        df = _enrich_missing_historical_taxonomy(df)
        df = augment_precedent_state_vector_columns(df)
        self.df = df
        self.n_rows = int(len(df))
        compact_cols = [
            col
            for col in _STATE_VECTOR_MATCHING_COLS
            if col in df.columns and int(pd.to_numeric(df.get(col), errors="coerce").notna().sum()) >= 1
        ]
        self.embedding_cols = tuple(compact_cols or _LEGACY_EMBEDDING_COLS)
        self.embedding_values = np.column_stack(
            [pd.to_numeric(df.get(c), errors="coerce").to_numpy(dtype=float) for c in self.embedding_cols]
        ) if self.n_rows else np.empty((0, len(self.embedding_cols)), dtype=float)
        self.raw_action_subtype_arr = _preferred_text_array(df, ("raw_action_subtype", "action_subtype"))
        self.raw_action_type_arr = _preferred_text_array(df, ("raw_action_type", "action_type"))
        self.normalized_action_subfamily_arr = _preferred_text_array(df, ("normalized_action_subfamily",))
        self.normalized_action_family_arr = _preferred_text_array(df, ("normalized_action_family",))
        self.normalized_action_id_arr = _preferred_text_array(df, ("normalized_action_id", "action_id"))
        self.action_subtype_arr = np.array(
            [
                _resolved_normalized_subfamily(
                    self.normalized_action_family_arr[i],
                    self.normalized_action_subfamily_arr[i],
                    self.raw_action_type_arr[i],
                    self.raw_action_subtype_arr[i],
                )
                or str(self.raw_action_subtype_arr[i] or "")
                for i in range(self.n_rows)
            ],
            dtype=object,
        )
        self.action_type_arr = np.array(
            [
                str(self.normalized_action_family_arr[i] or self.raw_action_type_arr[i] or "")
                for i in range(self.n_rows)
            ],
            dtype=object,
        )
        self.action_id_arr = np.array(
            [
                str(
                    self.normalized_action_id_arr[i]
                    or _historical_exact_action_id(self.raw_action_type_arr[i], self.raw_action_subtype_arr[i])
                    or ""
                )
                for i in range(self.n_rows)
            ],
            dtype=object,
        )
        self.preferred_action_key_arr = np.array(
            [
                str(self.action_id_arr[i] or self.action_subtype_arr[i] or self.action_type_arr[i] or "")
                for i in range(self.n_rows)
            ],
            dtype=object,
        )
        self.action_family_arr = np.array(
            [
                _combined_family_key(self.action_type_arr[i], self.action_subtype_arr[i])
                or _historical_action_family(
                    action_type=self.raw_action_type_arr[i],
                    action_subtype=self.raw_action_subtype_arr[i],
                )
                for i in range(self.n_rows)
            ],
            dtype=object,
        )
        self.company_id_arr = df.get("company_id", pd.Series("", index=df.index)).astype(str).to_numpy()
        self.action_date_arr = pd.to_datetime(
            df.get("action_date"),
            utc=True,
            errors="coerce",
        ).to_numpy(dtype="datetime64[ns]")
        self.source_event_id_arr = _preferred_text_array(df, ("source_event_id", "source_id"))
        self.feature_sector_arr = _preferred_text_array(
            df,
            ("taxonomy.sector", "base_sector", "sector", "gics_sector", "sic"),
        )

        action_size = (
            pd.to_numeric(df["action_size"], errors="coerce").to_numpy(dtype=float)
            if "action_size" in df.columns
            else np.full(self.n_rows, np.nan, dtype=float)
        )
        base_market_cap_series = df.get("base_market_cap", pd.Series(np.nan, index=df.index))
        base_mc = _normalize_market_cap_series_to_dollars(
            pd.Series(base_market_cap_series, copy=False),
            prefer_source_units=True,
        ).to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = action_size / base_mc
        self.action_scale_arr = np.where(np.isfinite(scale), np.clip(scale, a_min=0.0, a_max=None), 0.0)
        family_scale_bucket_arr = (
            df.get("family_scale_bucket", pd.Series("", index=df.index)).fillna("").astype(str).to_numpy(dtype=object)
        )
        self.action_family_scale_arr = np.array(
            [
                (
                    _historical_debt_amount_key(
                        family=str(self.action_family_arr[i]),
                        action_size=_to_float(action_size[i], None),
                    )
                    or (
                        f"{self.action_family_arr[i]}.scale_{_canonical_token(family_scale_bucket_arr[i])}"
                        if str(family_scale_bucket_arr[i] or "").strip() and str(self.action_family_arr[i])
                        else _historical_family_scale_key(
                            family=str(self.action_family_arr[i]),
                            action_scale=float(self.action_scale_arr[i]),
                        )
                    )
                )
                for i in range(self.n_rows)
            ],
            dtype=object,
        )

        # Precompute sector tokens and regime flags once.
        sector_sources = [
            df.get(col, pd.Series("", index=df.index)).fillna("").astype(str).to_numpy(dtype=object)
            for col in (
                "taxonomy.sector",
                "base_sector",
                "sector",
                "gics_sector",
                "sector_name",
                "sic",
                "base_sic",
            )
        ]
        self.sector_token_arr = np.array(
            [
                next((str(values[i]).strip().upper() for values in sector_sources if str(values[i]).strip()), "")
                for i in range(self.n_rows)
            ],
            dtype=object,
        )
        subsector_sources = [
            df.get(col, pd.Series("", index=df.index)).fillna("").astype(str).to_numpy(dtype=object)
            for col in (
                "taxonomy.subsector",
                "subsector",
                "industry",
                "base_industry",
            )
        ]
        self.subsector_token_arr = np.array(
            [
                next((str(values[i]).strip().upper() for values in subsector_sources if str(values[i]).strip()), "")
                for i in range(self.n_rows)
            ],
            dtype=object,
        )
        mc_series = _normalize_market_cap_series_to_dollars(
            pd.Series(base_market_cap_series, copy=False),
            prefer_source_units=True,
        )
        self.market_cap_bucket_quantiles = _market_cap_bucket_quantiles(mc_series)
        self.market_cap_bucket_arr = _market_cap_bucket_array(
            mc_series.to_numpy(dtype=float),
            self.market_cap_bucket_quantiles,
        )
        self.thresholds = _regime_thresholds(df)
        self.regime_flags = self._build_regime_flags()

        # Fast action-key lookup maps.
        self.subtype_lookup = self._build_lookup_multi(self.action_subtype_arr, self.raw_action_subtype_arr)
        self.type_lookup = self._build_lookup_multi(self.action_type_arr, self.raw_action_type_arr)
        self.action_id_lookup = self._build_lookup(self.action_id_arr)
        self.family_lookup = self._build_lookup(self.action_family_arr)
        self.family_scale_lookup = self._build_lookup(self.action_family_scale_arr)
        self.company_follow_on_lookup = self._build_company_follow_on_lookup()

    @staticmethod
    def _build_lookup(values: np.ndarray) -> Dict[str, np.ndarray]:
        out: Dict[str, List[int]] = {}
        for i, v in enumerate(values):
            out.setdefault(str(v), []).append(i)
        return {k: np.asarray(v, dtype=np.int64) for k, v in out.items() if k}

    @staticmethod
    def _build_lookup_multi(*values_arrs: np.ndarray) -> Dict[str, np.ndarray]:
        out: Dict[str, List[int]] = {}
        if not values_arrs:
            return out
        n_rows = len(values_arrs[0])
        for arr in values_arrs[1:]:
            if len(arr) != n_rows:
                raise ValueError("lookup arrays must have identical lengths")
        for i in range(n_rows):
            seen: set[str] = set()
            for arr in values_arrs:
                value = str(arr[i] or "")
                if not value or value in seen:
                    continue
                out.setdefault(value, []).append(i)
                seen.add(value)
        return {k: np.asarray(v, dtype=np.int64) for k, v in out.items() if k}

    def _build_regime_flags(self) -> Dict[str, np.ndarray]:
        hy = (
            pd.to_numeric(self.df["macro_hy_oas"], errors="coerce").to_numpy(dtype=float)
            if "macro_hy_oas" in self.df.columns
            else np.full(len(self.df), np.nan, dtype=float)
        )
        vix = (
            pd.to_numeric(self.df["macro_vix"], errors="coerce").to_numpy(dtype=float)
            if "macro_vix" in self.df.columns
            else np.full(len(self.df), np.nan, dtype=float)
        )
        hy_q25 = float(self.thresholds.get("hy_q25", 0.0))
        hy_q75 = float(self.thresholds.get("hy_q75", 0.0))
        vix_q25 = float(self.thresholds.get("vix_q25", 0.0))
        vix_q75 = float(self.thresholds.get("vix_q75", 0.0))
        hy_ok = np.isfinite(hy)
        vix_ok = np.isfinite(vix)
        credit_tight = hy_ok & (hy >= hy_q75)
        credit_loose = hy_ok & (hy <= hy_q25)
        high_vol = vix_ok & (vix >= vix_q75)
        low_vol = vix_ok & (vix <= vix_q25)
        risk_off = credit_tight | high_vol
        risk_on = credit_loose & low_vol
        return {
            "credit_tight": credit_tight,
            "credit_loose": credit_loose,
            "high_vol": high_vol,
            "low_vol": low_vol,
            "risk_off": risk_off,
            "risk_on": risk_on,
        }

    def candidate_indices_for_action(self, action_keys: Sequence[str]) -> np.ndarray:
        idx_parts: List[np.ndarray] = []
        for k in action_keys:
            if k in self.subtype_lookup:
                idx_parts.append(self.subtype_lookup[k])
            if k in self.type_lookup:
                idx_parts.append(self.type_lookup[k])
        if not idx_parts:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(idx_parts))

    def _build_company_follow_on_lookup(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        if self.n_rows == 0:
            return {}
        valid_mask = (~pd.isna(self.action_date_arr)) & (self.company_id_arr != "")
        valid_idx = np.flatnonzero(valid_mask)
        if valid_idx.size == 0:
            return {}

        grouped: Dict[str, List[int]] = {}
        for idx in valid_idx.tolist():
            grouped.setdefault(str(self.company_id_arr[idx]), []).append(int(idx))

        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for company_id, idxs in grouped.items():
            idx_arr = np.asarray(idxs, dtype=np.int64)
            order = np.argsort(self.action_date_arr[idx_arr], kind="mergesort")
            sorted_idx = idx_arr[order]
            out[company_id] = (
                self.action_date_arr[sorted_idx],
                self.preferred_action_key_arr[sorted_idx],
            )
        return out


def build_precedent_retrieval_index(historical_df: pd.DataFrame) -> PrecedentRetrievalIndex:
    return PrecedentRetrievalIndex(historical_df)


def _canonical_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _normalized_refinancing_action_subtype(
    action_subtype: Any,
    action_params: Optional[Dict[str, Any]] = None,
) -> str:
    subtype_text = str(action_subtype or "").strip()
    normalized = str(_refinancing_subfamily(subtype_text or "") or "").strip()
    if normalized and normalized != "refinancing":
        return normalized
    params = dict(action_params or {})
    for key in ("instrument_type", "facility_type", "source_action_subtype", "action_subtype"):
        candidate = str(params.get(key) or "").strip()
        normalized = str(_refinancing_subfamily(candidate) or "").strip()
        if normalized and normalized != "refinancing":
            return normalized
    instrument_type = _canonical_token(params.get("instrument_type"))
    facility_type = _canonical_token(params.get("facility_type"))
    rate_structure = _canonical_token(params.get("rate_structure"))
    fixed_vs_floating = _canonical_token(params.get("fixed_vs_floating"))
    if instrument_type in {"revolver", "line", "credit_facility"} or facility_type in {
        "revolver",
        "line",
        "credit_facility",
        "364_day_facility",
    }:
        return "refinancing_revolver_family"
    if instrument_type in {"loan", "term_loan", "bridge_loan"} or facility_type in {
        "term_loan",
        "term_loan_a",
        "term_loan_b",
        "delay_draw_term_loan",
        "bridge_loan",
    }:
        return "refinancing_term_loan_family"
    if instrument_type in {"bond", "note", "debenture"}:
        return "refinancing_bond_family"
    secured_flag = params.get("secured_flag")
    if fixed_vs_floating == "fixed" or (rate_structure == "fixed" and secured_flag is False):
        return "refinancing_bond_family"
    return str(normalized or "refinancing")


def _effective_action_subtype(
    action_id: Any,
    action_subtype: Any,
    action_params: Optional[Dict[str, Any]] = None,
) -> str:
    action_text = str(action_id or "").strip().lower()
    if action_text == "capital_structure.refinancing":
        return _normalized_refinancing_action_subtype(action_subtype, action_params)
    return str(action_subtype or "").strip()


def _uses_refinancing_family_exact_matching(action_id: Any, action_subtype: Any) -> bool:
    action_text = _canonical_token(action_id)
    subtype_text = _canonical_token(action_subtype)
    return action_text == "capital_structure_refinancing" and subtype_text in {
        "refinancing_term_loan_family",
        "refinancing_revolver_family",
        "refinancing_bond_family",
    }


def _candidate_exact_action_keys(
    action_id: Any,
    action_subtype: Any,
    action_leaf: Any,
    outcomes_subtype_alias: Any,
) -> Tuple[str, ...]:
    if _uses_refinancing_family_exact_matching(action_id, action_subtype):
        keys: List[str] = []
        for key in (action_subtype, outcomes_subtype_alias):
            key_text = str(key or "").strip()
            if not key_text:
                continue
            if _canonical_token(key_text) == "refinancing":
                continue
            if key_text not in keys:
                keys.append(key_text)
        return tuple(keys)
    return tuple(
        k
        for k in sorted({str(action_id or ""), str(action_subtype or ""), str(action_leaf or ""), str(outcomes_subtype_alias or "")})
        if k
    )


def _candidate_action_id_keys(action_id: Any, action_subtype: Any) -> Tuple[str, ...]:
    action_text = str(action_id or "").strip()
    if not action_text:
        return ()
    if _uses_refinancing_family_exact_matching(action_id, action_subtype):
        return ()
    return (action_text,)


def _historical_action_family(*, action_type: Any, action_subtype: Any) -> str:
    at = _canonical_token(action_type)
    st = _canonical_token(action_subtype)
    if at == "loan_refinancing":
        refi_subfamily = _normalized_refinancing_action_subtype(action_subtype)
        if refi_subfamily and refi_subfamily != "refinancing":
            return f"capital_structure.{refi_subfamily}"
        return "capital_structure.refinancing"
    if at == "bond_issuance":
        return "capital_structure.debt_bond"
    if at == "loan_issuance":
        if any(
            token in st
            for token in (
                "revolver",
                "line",
                "facility",
                "letter_of_credit",
                "bridge_loan",
            )
        ):
            return "capital_structure.revolver"
        return "capital_structure.debt_loan"
    if at == "equity_offering_public_proxy":
        return "capital_structure.equity_like"
    if at == "buyback":
        return "capital_return.buyback"
    if at == "acquisition":
        if st == "disclosed_dollar_value_deal":
            return "mna.platform_disclosed"
        if st == "undisclosed_dollar_value_deal":
            return "mna.platform_undisclosed"
        if st == "acquisition_merger":
            return "mna.platform_merger"
        if st == "acquisition_lbo":
            return "mna.platform_lbo"
        if st in {"stake_purchases_deal", "repurchases_deal"}:
            return "mna.tuck_in_incremental"
        if st in {"acquisition_tender", "acquisition_exchange", "acquisition_reverse"}:
            return "mna.acquisition_structured"
        return "mna.acquisition"
    if at == "divestiture":
        return "portfolio.divestiture"
    if at in {"dividend_cut", "dividend_increase", "dividend_special", "dividend_initiate", "dividend_regular"}:
        return f"capital_return.{at}"
    return at


def _acquisition_scale_bucket(value: Optional[float]) -> str:
    if value is None:
        return ""
    try:
        v = float(value)
    except Exception:
        return ""
    if not np.isfinite(v) or v <= 0.0:
        return ""
    if v <= 0.03:
        return "micro"
    if v <= 0.10:
        return "small"
    if v <= 0.25:
        return "medium"
    return "large"


def _divestiture_scale_bucket(value: Optional[float]) -> str:
    if value is None:
        return ""
    try:
        v = float(value)
    except Exception:
        return ""
    if not np.isfinite(v) or v <= 0.0:
        return ""
    if v <= 0.10:
        return "small"
    if v <= 0.30:
        return "medium"
    return "large"


def _equity_scale_bucket(value: Optional[float]) -> str:
    if value is None:
        return ""
    try:
        v = float(value)
    except Exception:
        return ""
    if not np.isfinite(v) or v <= 0.0:
        return ""
    if v < 0.05:
        return "small"
    if v < 0.25:
        return "medium"
    return "large"


def _debt_scale_bucket(value: Optional[float]) -> str:
    if value is None:
        return ""
    try:
        v = float(value)
    except Exception:
        return ""
    if not np.isfinite(v) or v <= 0.0:
        return ""
    if v <= 0.03:
        return "micro"
    if v <= 0.10:
        return "small"
    if v <= 0.25:
        return "medium"
    return "large"


def _debt_amount_bucket(value: Optional[float]) -> str:
    if value is None:
        return ""
    try:
        v = float(value)
    except Exception:
        return ""
    if not np.isfinite(v) or v <= 0.0:
        return ""
    if v <= 1.0e7:
        return "micro"
    if v <= 5.0e7:
        return "small"
    if v <= 2.5e8:
        return "medium"
    return "large"


def _historical_family_scale_key(*, family: str, action_scale: float) -> str:
    fam = str(family or "")
    if fam.startswith("mna.") and fam != "mna.acquisition":
        bucket = _acquisition_scale_bucket(action_scale)
        if not bucket:
            return ""
        return f"{fam}.scale_{bucket}"
    if fam == "capital_structure.equity_issuance":
        bucket = _equity_scale_bucket(action_scale)
        if not bucket:
            return ""
        return f"{fam}.scale_{bucket}"
    if fam in {"capital_structure.debt_bond", "capital_structure.debt_loan", "capital_structure.revolver"}:
        bucket = _debt_scale_bucket(action_scale)
        if not bucket:
            return ""
        return f"{fam}.scale_{bucket}"
    if fam == "portfolio.divestiture":
        bucket = _divestiture_scale_bucket(action_scale)
        if not bucket:
            return ""
        return f"{fam}.scale_{bucket}"
    return ""


def _historical_debt_amount_key(*, family: str, action_size: Optional[float]) -> str:
    fam = str(family or "")
    if fam not in {"capital_structure.debt_bond", "capital_structure.debt_loan", "capital_structure.revolver"}:
        return ""
    bucket = _debt_amount_bucket(action_size)
    if not bucket:
        return ""
    return f"{fam}.amount_{bucket}"


def _candidate_debt_amount(action_params: Dict[str, Any]) -> Optional[float]:
    params = action_params or {}
    for key in (
        "draw_amount_usd",
        "resize_amount_usd",
        "amount_refinanced_usd",
        "amount_usd",
        "size_absolute_usd",
        "amount",
    ):
        value = _to_float(params.get(key), None)
        if value is not None and value > 0:
            return float(value)
    return None


def _candidate_action_family_weights(action_id: str, action_subtype: str) -> Tuple[Tuple[str, float], ...]:
    aid = str(action_id or "")
    leaf = aid.split(".", 1)[1] if "." in aid else str(action_subtype or "")
    leaf = _canonical_token(leaf)
    subtype_text = _canonical_token(action_subtype)
    if aid.startswith("capital_structure."):
        if leaf in {"new_debt_issuance", "refinancing"}:
            if subtype_text == "refinancing_term_loan_family":
                return (
                    ("capital_structure.refinancing_term_loan_family", 0.94),
                    ("capital_structure.debt_loan", 0.86),
                    ("capital_structure.debt_core", 0.78),
                )
            if subtype_text == "refinancing_revolver_family":
                return (
                    ("capital_structure.refinancing_revolver_family", 0.94),
                    ("capital_structure.revolver", 0.86),
                    ("capital_structure.debt_loan", 0.74),
                    ("capital_structure.debt_core", 0.70),
                )
            if subtype_text == "refinancing_bond_family":
                return (
                    ("capital_structure.refinancing_bond_family", 0.94),
                    ("capital_structure.debt_bond", 0.86),
                    ("capital_structure.debt_core", 0.78),
                )
            return (
                ("capital_structure.debt_bond", 0.84),
                ("capital_structure.debt_loan", 0.80),
                ("capital_structure.debt_core", 0.76),
            )
        if leaf == "revolver_draw_or_resize":
            return (
                ("capital_structure.revolver", 0.84),
                ("capital_structure.debt_loan", 0.74),
                ("capital_structure.debt_core", 0.70),
            )
        if leaf in {"tender_offer_debt", "exchange_offer", "liability_management_exercise"}:
            return (
                ("capital_structure.debt_bond", 0.74),
                ("capital_structure.debt_loan", 0.72),
                ("capital_structure.debt_core", 0.68),
            )
        if leaf in {"convertible_issuance", "preferred_issuance", "equity_issuance"}:
            return (("capital_structure.equity_issuance", 0.80),)
    if aid.startswith("capital_return."):
        if leaf in {"open_market_buyback", "accelerated_share_repurchase", "tender_offer_buyback"}:
            return (("capital_return.buyback", 0.80),)
        if leaf in {"dividend_cut", "dividend_increase", "dividend_initiate", "special_dividend"}:
            return ((f"capital_return.{leaf}", 0.86),)
    if aid.startswith("mna."):
        if leaf == "platform_acquisition":
            return (
                ("mna.platform_disclosed", 0.88),
                ("mna.platform_undisclosed", 0.86),
                ("mna.platform_merger", 0.84),
                ("mna.platform_lbo", 0.82),
                ("mna.acquisition", 0.76),
            )
        if leaf == "tuck_in_acquisition":
            return (
                ("mna.tuck_in_incremental", 0.88),
                ("mna.acquisition_structured", 0.76),
                ("mna.acquisition", 0.72),
            )
        if leaf == "transformational_acquisition":
            return (
                ("mna.platform_merger", 0.92),
                ("mna.platform_lbo", 0.90),
                ("mna.platform_disclosed", 0.88),
                ("mna.platform_undisclosed", 0.84),
                ("mna.acquisition", 0.78),
            )
        if leaf == "go_private_lbo":
            return (
                ("mna.platform_lbo", 0.94),
                ("mna.platform_merger", 0.84),
                ("mna.platform_disclosed", 0.80),
                ("mna.acquisition", 0.76),
            )
        return (("mna.acquisition", 0.76),)
    if aid in {"portfolio.divestiture_full", "portfolio.divestiture_partial", "portfolio.asset_sale"}:
        return (("portfolio.divestiture", 0.76),)
    return ()


def _candidate_action_family_scale_weights(
    action_id: str,
    action_subtype: str,
    action_params: Dict[str, Any],
    candidate_features: Dict[str, Any],
) -> Tuple[Tuple[str, float], ...]:
    aid = str(action_id or "")
    leaf = aid.split(".", 1)[1] if "." in aid else str(action_subtype or "")
    leaf = _canonical_token(leaf)
    subtype_text = _canonical_token(action_subtype)
    market_cap = _candidate_market_cap(candidate_features if isinstance(candidate_features, dict) else {})
    scale = _estimate_action_scale(action_params or {}, market_cap)
    debt_bucket = _debt_scale_bucket(scale)
    debt_amount_bucket = _debt_amount_bucket(_candidate_debt_amount(action_params or {}))
    equity_bucket = _equity_scale_bucket(scale)
    if aid.startswith("mna."):
        bucket = _acquisition_scale_bucket(scale)
        if not bucket:
            return ()
        if leaf == "platform_acquisition":
            return (
                (f"mna.platform_disclosed.scale_{bucket}", 0.93),
                (f"mna.platform_undisclosed.scale_{bucket}", 0.91),
                (f"mna.platform_merger.scale_{bucket}", 0.89),
                (f"mna.platform_lbo.scale_{bucket}", 0.87),
            )
        if leaf == "tuck_in_acquisition":
            return (
                (f"mna.tuck_in_incremental.scale_{bucket}", 0.93),
                (f"mna.acquisition_structured.scale_{bucket}", 0.83),
            )
        if leaf == "transformational_acquisition":
            return (
                (f"mna.platform_merger.scale_{bucket}", 0.95),
                (f"mna.platform_lbo.scale_{bucket}", 0.93),
                (f"mna.platform_disclosed.scale_{bucket}", 0.90),
                (f"mna.platform_undisclosed.scale_{bucket}", 0.86),
            )
        if leaf == "go_private_lbo":
            return (
                (f"mna.platform_lbo.scale_{bucket}", 0.96),
                (f"mna.platform_merger.scale_{bucket}", 0.88),
                (f"mna.platform_disclosed.scale_{bucket}", 0.84),
            )
    if aid.startswith("capital_structure.") and debt_bucket:
        instrument_type = _canonical_token((action_params or {}).get("instrument_type"))
        secured_flag = bool((action_params or {}).get("secured_flag"))
        rate_structure = _canonical_token((action_params or {}).get("rate_structure"))
        fixed_vs_floating = _canonical_token((action_params or {}).get("fixed_vs_floating"))
        if leaf == "new_debt_issuance":
            if instrument_type in {"bond", "note", "debenture"}:
                weights = []
                if debt_amount_bucket:
                    weights.append((f"capital_structure.debt_bond.amount_{debt_amount_bucket}", 0.94))
                weights.extend(
                    [
                        (f"capital_structure.debt_bond.scale_{debt_bucket}", 0.92),
                        (f"capital_structure.debt_loan.scale_{debt_bucket}", 0.82),
                    ]
                )
                return tuple(weights)
            if instrument_type in {"loan", "term_loan", "credit_facility"}:
                weights = []
                if debt_amount_bucket:
                    weights.append((f"capital_structure.debt_loan.amount_{debt_amount_bucket}", 0.94))
                weights.extend(
                    [
                        (f"capital_structure.debt_loan.scale_{debt_bucket}", 0.92),
                        (f"capital_structure.debt_bond.scale_{debt_bucket}", 0.82),
                    ]
                )
                return tuple(weights)
            weights = []
            if debt_amount_bucket:
                weights.append((f"capital_structure.debt_bond.amount_{debt_amount_bucket}", 0.90))
            weights.extend(
                [
                    (f"capital_structure.debt_bond.scale_{debt_bucket}", 0.88),
                    (f"capital_structure.debt_loan.scale_{debt_bucket}", 0.84),
                ]
            )
            return tuple(weights)
        if leaf == "refinancing":
            if subtype_text == "refinancing_revolver_family":
                weights = []
                if debt_amount_bucket:
                    weights.append((f"capital_structure.revolver.amount_{debt_amount_bucket}", 0.94))
                weights.extend(
                    [
                        (f"capital_structure.revolver.scale_{debt_bucket}", 0.92),
                        (f"capital_structure.debt_loan.scale_{debt_bucket}", 0.82),
                    ]
                )
                return tuple(weights)
            if subtype_text == "refinancing_term_loan_family":
                weights = []
                if debt_amount_bucket:
                    weights.append((f"capital_structure.debt_loan.amount_{debt_amount_bucket}", 0.94))
                weights.extend(
                    [
                        (f"capital_structure.debt_loan.scale_{debt_bucket}", 0.92),
                        (f"capital_structure.revolver.scale_{debt_bucket}", 0.82),
                    ]
                )
                return tuple(weights)
            if subtype_text == "refinancing_bond_family":
                weights = []
                if debt_amount_bucket:
                    weights.append((f"capital_structure.debt_bond.amount_{debt_amount_bucket}", 0.94))
                weights.extend(
                    [
                        (f"capital_structure.debt_bond.scale_{debt_bucket}", 0.92),
                        (f"capital_structure.debt_loan.scale_{debt_bucket}", 0.82),
                    ]
                )
                return tuple(weights)
            bond_like = (fixed_vs_floating == "fixed") or (rate_structure == "fixed" and not secured_flag)
            if bond_like:
                weights = []
                if debt_amount_bucket:
                    weights.append((f"capital_structure.debt_bond.amount_{debt_amount_bucket}", 0.92))
                weights.extend(
                    [
                        (f"capital_structure.debt_bond.scale_{debt_bucket}", 0.90),
                        (f"capital_structure.debt_loan.scale_{debt_bucket}", 0.84),
                    ]
                )
                return tuple(weights)
            weights = []
            if debt_amount_bucket:
                weights.append((f"capital_structure.debt_loan.amount_{debt_amount_bucket}", 0.92))
            weights.extend(
                [
                    (f"capital_structure.debt_loan.scale_{debt_bucket}", 0.90),
                    (f"capital_structure.debt_bond.scale_{debt_bucket}", 0.82),
                ]
            )
            return tuple(weights)
        if leaf in {"tender_offer_debt", "exchange_offer", "liability_management_exercise"}:
            return (
                (f"capital_structure.debt_bond.scale_{debt_bucket}", 0.86),
                (f"capital_structure.debt_loan.scale_{debt_bucket}", 0.82),
            )
        if leaf == "revolver_draw_or_resize":
            weights = []
            if debt_amount_bucket:
                weights.append((f"capital_structure.revolver.amount_{debt_amount_bucket}", 0.94))
            weights.extend(
                [
                    (f"capital_structure.revolver.scale_{debt_bucket}", 0.92),
                    (f"capital_structure.debt_loan.scale_{debt_bucket}", 0.80),
                ]
            )
            return tuple(weights)
    if aid.startswith("capital_structure.") and leaf in {"convertible_issuance", "preferred_issuance", "equity_issuance"}:
        if not equity_bucket:
            return ()
        return ((f"capital_structure.equity_issuance.scale_{equity_bucket}", 0.84),)
    if aid in {"portfolio.divestiture_full", "portfolio.divestiture_partial", "portfolio.asset_sale"}:
        div_bucket = _divestiture_scale_bucket(scale)
        if not div_bucket:
            return ()
        return ((f"portfolio.divestiture.scale_{div_bucket}", 0.88),)
    return ()


_NARRATIVE_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "over",
    "under",
    "will",
    "have",
    "has",
    "had",
    "was",
    "were",
    "are",
    "is",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "as",
    "an",
    "or",
    "be",
    "it",
    "its",
    "our",
    "their",
}

_NARRATIVE_TEXT_COLS = ("narrative_text", "headline", "text", "title", "description", "summary")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in toks if len(t) >= 3 and t not in _NARRATIVE_STOPWORDS]


def _candidate_narrative_text(
    action_id: str,
    action_subtype: Optional[str],
    action_params: Dict[str, Any],
    candidate_features: Dict[str, Any],
    candidate_regime: Dict[str, Any],
) -> str:
    explicit = _first_str(
        [
            candidate_features.get("narrative_text") if isinstance(candidate_features, dict) else None,
            candidate_features.get("headline") if isinstance(candidate_features, dict) else None,
            candidate_features.get("summary") if isinstance(candidate_features, dict) else None,
        ]
    )
    if explicit:
        return explicit
    parts = [
        str(action_id or ""),
        str(action_subtype or ""),
    ]
    if isinstance(candidate_regime, dict):
        parts.extend(
            [
                str(candidate_regime.get("credit_regime", "")),
                str(candidate_regime.get("risk_regime", "")),
                str(candidate_regime.get("vol_regime", "")),
            ]
        )
    if isinstance(action_params, dict):
        for k in sorted(action_params.keys()):
            v = action_params.get(k)
            if isinstance(v, (dict, list)):
                continue
            parts.append(f"{k} {v}")
    if isinstance(candidate_features, dict):
        for k in ("sector", "gics_sector", "leverage_net_debt_ebitda", "ebitda_margin", "fcf_margin"):
            v = candidate_features.get(k)
            if v is not None:
                parts.append(f"{k} {v}")
    return " ".join([p for p in parts if str(p).strip()])


def _row_narrative_text(row: pd.Series) -> Tuple[str, bool]:
    real_parts: List[str] = []
    for c in _NARRATIVE_TEXT_COLS:
        if c in row.index:
            s = _first_str([row.get(c)])
            if s:
                real_parts.append(s)
    if real_parts:
        return " ".join(real_parts), True

    # Fallback structured narrative (deterministic, low-fidelity).
    parts = [
        _preferred_row_action_fields(row)[0],
        _preferred_row_action_fields(row)[1],
        _preferred_row_action_fields(row)[2],
        f"leverage {row.get('base_leverage')}",
        f"margin {row.get('base_margin')}",
        f"fcf {row.get('base_fcf_margin')}",
        f"hy_oas {row.get('macro_hy_oas')}",
        f"vix {row.get('macro_vix')}",
    ]
    return " ".join([p for p in parts if str(p).strip()]), False


def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    if not tokens:
        return {}
    tf = Counter(tokens)
    vec: Dict[str, float] = {}
    for t, c in tf.items():
        w = float(c) * float(idf.get(t, 1.0))
        if w > 0:
            vec[t] = w
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm <= 1e-12:
        return {}
    return {k: v / norm for k, v in vec.items()}


def _sparse_cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return float(sum(v * b.get(k, 0.0) for k, v in a.items()))


def _narrative_similarity(
    cohort: pd.DataFrame,
    *,
    action_id: str,
    action_subtype: Optional[str],
    action_params: Dict[str, Any],
    candidate_features: Dict[str, Any],
    candidate_regime: Dict[str, Any],
) -> Tuple[np.ndarray, int, float]:
    """Return per-row narrative similarity, number of rows with real text, and top-5 mean similarity."""
    if cohort.empty:
        return np.array([], dtype=float), 0, 0.0

    candidate_text = _candidate_narrative_text(
        action_id=action_id,
        action_subtype=action_subtype,
        action_params=action_params,
        candidate_features=candidate_features,
        candidate_regime=candidate_regime,
    )
    cand_tokens = _tokenize(candidate_text)
    if not cand_tokens:
        return np.full(len(cohort), 0.0, dtype=float), 0, 0.0

    docs: List[List[str]] = []
    real_count = 0
    doc_freq: Counter = Counter()
    for _, row in cohort.iterrows():
        txt, is_real = _row_narrative_text(row)
        if is_real:
            real_count += 1
        toks = _tokenize(txt)
        docs.append(toks)
        doc_freq.update(set(toks))

    n_docs = max(1, len(docs))
    idf = {t: math.log((1.0 + n_docs) / (1.0 + float(df))) + 1.0 for t, df in doc_freq.items()}
    cand_vec = _tfidf_vector(cand_tokens, idf)
    if not cand_vec:
        return np.full(len(cohort), 0.0, dtype=float), real_count, 0.0

    sims = np.array([_sparse_cosine(cand_vec, _tfidf_vector(toks, idf)) for toks in docs], dtype=float)
    top = np.sort(sims)[-5:] if sims.size else np.array([], dtype=float)
    top_mean = float(np.mean(top)) if top.size else 0.0
    return sims, real_count, top_mean


def _minimum_exact_support(min_k: int, top_k: int) -> int:
    min_k = max(1, int(min_k))
    top_k = max(1, int(top_k))
    return max(4, min(min_k, max(6, int(math.ceil(top_k * 0.25)))))


def _support_ratio(count: int, threshold: int) -> float:
    return max(0.0, min(1.0, float(max(0, int(count))) / float(max(1, int(threshold)))))


def _cohort_support_factor(similarity_scores: Sequence[float]) -> float:
    if not similarity_scores:
        return 0.0
    arr = np.clip(np.asarray(similarity_scores, dtype=float), 0.0, 1.0)
    effective = float(np.clip((arr - 0.35) / 0.45, 0.0, 1.0).sum())
    return max(0.0, min(1.0, effective / 8.0))


def _compute_calibration_confidence(
    *,
    retrieval_tier: str,
    exact_match_count: int,
    exact_support_min: int,
    cohort_size: int,
    base_similarity: float,
    top_similarity_mean: float,
    top_similarity_p25: float,
    top_action_match_score: float,
    mismatch_count: int,
    regime_mismatch: bool,
    parameter_mismatch: bool,
    narrative_mismatch: bool,
) -> Dict[str, float]:
    exact_support_ratio = _support_ratio(exact_match_count, exact_support_min)
    cohort_factor = min(1.0, float(max(0, int(cohort_size))) / 20.0)
    support_factor = _cohort_support_factor(
        [
            max(0.0, min(1.0, float(base_similarity))),
            max(0.0, min(1.0, float(top_similarity_mean))),
            max(0.0, min(1.0, float(top_similarity_p25))),
        ]
    )
    similarity_signal = (
        0.50 * max(0.0, min(1.0, float(top_similarity_mean)))
        + 0.20 * max(0.0, min(1.0, float(top_similarity_p25)))
        + 0.20 * max(0.0, min(1.0, float(base_similarity)))
        + 0.10 * max(0.0, min(1.0, float(top_action_match_score)))
    )
    mismatch_penalty = min(
        0.65,
        0.08 * max(0, int(mismatch_count))
        + (0.14 if regime_mismatch else 0.0)
        + (0.14 if parameter_mismatch else 0.0)
        + (0.08 if narrative_mismatch else 0.0),
    )
    confidence_pre_tier_discount = max(
        0.0,
        min(
            1.0,
            similarity_signal * (0.45 + 0.30 * cohort_factor + 0.25 * support_factor) * (1.0 - mismatch_penalty),
        ),
    )
    if retrieval_tier == "exact":
        tier_conf_discount = 0.96 + 0.04 * exact_support_ratio
    elif retrieval_tier == "family":
        tier_conf_discount = 0.86 + 0.08 * exact_support_ratio + 0.06 * max(
            0.0, min(1.0, float(top_action_match_score))
        )
    elif retrieval_tier == "sibling_type":
        tier_conf_discount = 0.80 + 0.12 * exact_support_ratio + 0.08 * max(
            0.0, min(1.0, float(top_action_match_score))
        )
    else:
        tier_conf_discount = 0.62 + 0.12 * max(0.0, min(1.0, float(top_action_match_score))) + 0.10 * max(
            0.0, min(1.0, float(top_similarity_mean))
        )
    calibration_confidence = max(0.0, min(1.0, confidence_pre_tier_discount * tier_conf_discount))
    return {
        "exact_support_ratio": float(exact_support_ratio),
        "cohort_factor": float(cohort_factor),
        "support_factor": float(support_factor),
        "similarity_signal": float(similarity_signal),
        "mismatch_penalty": float(mismatch_penalty),
        "confidence_pre_tier_discount": float(confidence_pre_tier_discount),
        "tier_conf_discount": float(tier_conf_discount),
        "calibration_confidence": float(calibration_confidence),
    }


def _to_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    if isinstance(v, dict):
        v = v.get("value")
        if v is None:
            return default
    try:
        out = float(v)
    except Exception:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _dist(series: pd.Series) -> DistributionStats:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return DistributionStats(
            mean=None,
            median=None,
            p10=None,
            p25=None,
            p75=None,
            p90=None,
            sample_size=0,
        )
    return DistributionStats(
        mean=float(s.mean()),
        median=float(s.median()),
        p10=float(s.quantile(0.10)),
        p25=float(s.quantile(0.25)),
        p75=float(s.quantile(0.75)),
        p90=float(s.quantile(0.90)),
        sample_size=int(len(s)),
    )


def _combine_metric(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index, dtype=float)
    out = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in present})
    return out.mean(axis=1, skipna=True)


def _empty_metric_set() -> MetricDistributionSet:
    e = DistributionStats(None, None, None, None, None, None, 0)
    return MetricDistributionSet(
        valuation_multiple_change=e,
        equity_return_vs_sector=e,
        credit_spread_change=e,
        rating_migration=e,
        leverage_change=e,
        fcf_change=e,
        volatility_change=e,
    )


def _metric_set_from_df(df: pd.DataFrame, horizon: str) -> MetricDistributionSet:
    if df.empty:
        return _empty_metric_set()

    def _series(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)
        return pd.to_numeric(df[col], errors="coerce")

    if horizon == "1m":
        val = _combine_metric(df, ["outcome_pe_6m", "outcome_ev_ebitda_6m"]) / 6.0
        eq = _series("outcome_pe_6m") / 6.0
        credit = _series("credit_spread_change_1m")
        rating = _series("rating_migration_1m")
    elif horizon == "6m":
        val = _combine_metric(df, ["outcome_pe_6m", "outcome_ev_ebitda_6m"])
        eq = _series("outcome_pe_6m")
        credit = _series("credit_spread_change_6m")
        rating = _series("rating_migration_6m")
    elif horizon == "12m":
        val = _combine_metric(df, ["outcome_pe_12m", "outcome_ev_ebitda_12m"])
        eq = _series("outcome_pe_12m")
        credit = _series("credit_spread_change_12m")
        rating = _series("rating_migration_12m")
    else:  # 24m
        val = _combine_metric(df, ["outcome_pe_12m", "outcome_ev_ebitda_12m"]) * 2.0
        eq = _series("outcome_pe_12m") * 2.0
        credit = _series("credit_spread_change_24m")
        rating = _series("rating_migration_24m")

    lev = _series("leverage_delta")
    fcf = _series("fcf_margin_delta")

    # Fallback proxies where direct series are unavailable in the historical table.
    vol = val.abs()
    if credit.dropna().empty:
        credit = pd.Series(np.nan, index=df.index, dtype=float)
    if rating.dropna().empty:
        rating = pd.Series(np.nan, index=df.index, dtype=float)

    return MetricDistributionSet(
        valuation_multiple_change=_dist(val),
        equity_return_vs_sector=_dist(eq),
        credit_spread_change=_dist(credit),
        rating_migration=_dist(rating),
        leverage_change=_dist(lev),
        fcf_change=_dist(fcf),
        volatility_change=_dist(vol),
    )


def _build_outcome_distributions(df: pd.DataFrame) -> OutcomeDistributions:
    return OutcomeDistributions(
        horizon_1m=_metric_set_from_df(df, "1m"),
        horizon_6m=_metric_set_from_df(df, "6m"),
        horizon_12m=_metric_set_from_df(df, "12m"),
        horizon_24m=_metric_set_from_df(df, "24m"),
    )


def _regime_thresholds(full_df: pd.DataFrame) -> Dict[str, float]:
    hy = (
        pd.to_numeric(full_df["macro_hy_oas"], errors="coerce")
        if "macro_hy_oas" in full_df.columns
        else pd.Series(np.nan, index=full_df.index, dtype=float)
    )
    vix = (
        pd.to_numeric(full_df["macro_vix"], errors="coerce")
        if "macro_vix" in full_df.columns
        else pd.Series(np.nan, index=full_df.index, dtype=float)
    )
    return {
        "hy_q25": float(hy.quantile(0.25)) if not hy.dropna().empty else 0.0,
        "hy_q75": float(hy.quantile(0.75)) if not hy.dropna().empty else 0.0,
        "vix_q25": float(vix.quantile(0.25)) if not vix.dropna().empty else 0.0,
        "vix_q75": float(vix.quantile(0.75)) if not vix.dropna().empty else 0.0,
    }


def _label_regime_row(row: pd.Series, t: Dict[str, float]) -> Dict[str, bool]:
    hy = _to_float(row.get("macro_hy_oas"))
    vix = _to_float(row.get("macro_vix"))
    credit_tight = hy is not None and hy >= t["hy_q75"]
    credit_loose = hy is not None and hy <= t["hy_q25"]
    high_vol = vix is not None and vix >= t["vix_q75"]
    low_vol = vix is not None and vix <= t["vix_q25"]
    risk_off = bool(credit_tight or high_vol)
    risk_on = bool(credit_loose and low_vol)
    return {
        "credit_tight": credit_tight,
        "credit_loose": credit_loose,
        "high_vol": high_vol,
        "low_vol": low_vol,
        "risk_off": risk_off,
        "risk_on": risk_on,
    }


def _candidate_regime_flags(regime: Dict[str, Any]) -> Dict[str, bool]:
    credit = str(regime.get("credit_regime", "neutral"))
    risk = str(regime.get("risk_regime", "neutral"))
    vol = str(regime.get("vol_regime", "normal"))
    return {
        "credit_tight": credit == "tight",
        "credit_loose": credit == "loose",
        "high_vol": vol == "high",
        "low_vol": vol == "low",
        "risk_off": risk == "risk_off",
        "risk_on": risk == "risk_on",
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _estimate_action_scale(params: Dict[str, Any], market_cap: Optional[float]) -> float:
    pct_keys = (
        "size_pct_market_cap",
        "target_size_pct_market_cap",
        "target_size_pct_mc",
        "target_size_pct_ev",
        "deal_size_pct_ev",
        "transaction_size_pct_ev",
        "target_ev_pct",
        "percent_divested",
        "percent_sold",
        "stake_pct",
    )
    for key in pct_keys:
        size_pct = _to_float(params.get(key))
        if size_pct is not None:
            return max(0.0, float(size_pct))

    abs_keys = (
        "action_size",
        "estimated_proceeds_usd",
        "draw_amount_usd",
        "resize_amount_usd",
        "size_absolute_usd",
        "amount",
        "amount_usd",
        "amount_refinanced_usd",
        "purchase_price_usd",
        "transaction_value_usd",
        "deal_value_usd",
        "target_enterprise_value_usd",
    )
    if market_cap and market_cap > 0:
        for key in abs_keys:
            size_abs = _to_float(params.get(key))
            if size_abs is not None:
                return max(0.0, float(size_abs) / float(market_cap))

    target_size = params.get("target_size")
    if isinstance(target_size, dict):
        if market_cap and market_cap > 0:
            for key in ("usd", "absolute_usd", "amount_usd", "enterprise_value_usd"):
                size_abs = _to_float(target_size.get(key))
                if size_abs is not None:
                    return max(0.0, float(size_abs) / float(market_cap))
        for key in ("pct_market_cap", "pct_ev", "pct_mc", "size_pct_ev"):
            size_pct = _to_float(target_size.get(key))
            if size_pct is not None:
                return max(0.0, float(size_pct))
    return 0.0


def _safe_action_date(s: pd.Series) -> pd.Timestamp:
    return pd.to_datetime(s.get("action_date"), errors="coerce")


def _feature_key_state(row: pd.Series) -> Dict[str, Any]:
    out = {}
    for c in [
        "base_leverage",
        "base_margin",
        "base_market_cap",
        "base_revenue_ttm",
        "base_roic",
        "base_fcf_margin",
    ]:
        out[c] = _to_float(row.get(c))
    for c in _STATE_VECTOR_MATCHING_COLS:
        out[c] = _to_float(row.get(c))
    out["base_sector"] = _first_str(
        [
            row.get("base_sector"),
            row.get("taxonomy.sector"),
            row.get("sector"),
            row.get("gics_sector"),
            row.get("sic"),
        ]
    )
    out["sector"] = _first_str([row.get("taxonomy.sector"), row.get("sector"), row.get("base_sector"), row.get("gics_sector")])
    out["subsector"] = _first_str([row.get("taxonomy.subsector"), row.get("subsector"), row.get("industry")])
    out["retirement_regime"] = _first_str(
        [
            row.get("state_vector_v1.meta.retirement_regime"),
            row.get("retirement_regime"),
            row.get("capital_structure.retirement_obligation_regime"),
        ]
    )
    return out


def _first_str(values: Sequence[Any]) -> str:
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() not in {"none", "nan"}:
            return s
    return ""


def _sector_token_from_candidate(candidate_features: Dict[str, Any]) -> str:
    return _first_str(
        [
            _extract_metric_value(candidate_features.get("taxonomy.sector")),
            _extract_metric_value(candidate_features.get("state_vector_v1.meta.sector")),
            _extract_metric_value(candidate_features.get("sector")),
            _extract_metric_value(candidate_features.get("gics_sector")),
            _extract_metric_value(candidate_features.get("sector_name")),
            _extract_metric_value(candidate_features.get("sic")),
        ]
    ).upper()


def _subsector_token_from_candidate(candidate_features: Dict[str, Any]) -> str:
    return _first_str(
        [
            _extract_metric_value(candidate_features.get("taxonomy.subsector")),
            _extract_metric_value(candidate_features.get("state_vector_v1.meta.subsector")),
            _extract_metric_value(candidate_features.get("subsector")),
            _extract_metric_value(candidate_features.get("industry")),
        ]
    ).upper()


def _sector_token_from_row(row: pd.Series) -> str:
    return _first_str(
        [
            row.get("taxonomy.sector"),
            row.get("base_sector"),
            row.get("sector"),
            row.get("gics_sector"),
            row.get("sector_name"),
            row.get("sic"),
            row.get("base_sic"),
        ]
    ).upper()


def _subsector_token_from_row(row: pd.Series) -> str:
    return _first_str(
        [
            row.get("taxonomy.subsector"),
            row.get("subsector"),
            row.get("industry"),
            row.get("base_industry"),
        ]
    ).upper()


def _sector_similarity(cand_sector: str, row_sector: str, cand_subsector: str = "", row_subsector: str = "") -> float:
    if cand_subsector and row_subsector and cand_subsector == row_subsector:
        return 1.0
    if not cand_sector or not row_sector:
        return 0.5
    if cand_sector == row_sector:
        return 0.80
    if cand_sector[:2] and row_sector[:2] and cand_sector[:2] == row_sector[:2]:
        return 0.40
    return 0.0


def _market_cap_bucket_quantiles(series: pd.Series) -> Tuple[float, float, float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(s.quantile(0.20)),
        float(s.quantile(0.40)),
        float(s.quantile(0.60)),
        float(s.quantile(0.80)),
    )


def _market_cap_bucket_from_value(value: Optional[float], quantiles: Tuple[float, float, float, float]) -> int:
    if value is None or not np.isfinite(float(value)):
        return -1
    try:
        return int(np.digitize([float(value)], quantiles)[0])
    except Exception:
        return -1


def _market_cap_bucket_array(values: np.ndarray, quantiles: Tuple[float, float, float, float]) -> np.ndarray:
    out = np.full(values.shape[0], -1, dtype=np.int8)
    if values.size == 0:
        return out
    ok = np.isfinite(values)
    if int(np.count_nonzero(ok)) == 0:
        return out
    out[ok] = np.digitize(values[ok], quantiles).astype(np.int8)
    return out


def _winsorized_robust_standardize(
    emb_raw: np.ndarray,
    candidate_vec_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Winsorized robust z-score by feature (median + IQR scale)."""
    if emb_raw.size == 0:
        return np.empty_like(emb_raw), np.zeros_like(candidate_vec_raw, dtype=float)
    emb = np.array(emb_raw, dtype=float, copy=True)
    cand = np.array(candidate_vec_raw, dtype=float, copy=True)
    n_cols = int(emb.shape[1])
    out = np.zeros_like(emb, dtype=float)
    cand_out = np.zeros(n_cols, dtype=float)
    for j in range(n_cols):
        col = emb[:, j]
        ok = np.isfinite(col)
        if int(np.count_nonzero(ok)) == 0:
            continue
        valid = col[ok]
        lo = float(np.nanquantile(valid, 0.05))
        hi = float(np.nanquantile(valid, 0.95))
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo = float(np.nanmin(valid))
            hi = float(np.nanmax(valid))
        if lo > hi:
            lo, hi = hi, lo
        clipped_valid = np.clip(valid, lo, hi)
        med = float(np.nanmedian(clipped_valid))
        q25 = float(np.nanquantile(clipped_valid, 0.25))
        q75 = float(np.nanquantile(clipped_valid, 0.75))
        scale = (q75 - q25) / 1.349
        if (not np.isfinite(scale)) or scale <= 1e-9:
            scale = float(np.nanstd(clipped_valid))
        if (not np.isfinite(scale)) or scale <= 1e-9:
            scale = 1.0
        col_filled = np.where(ok, np.clip(col, lo, hi), med)
        out[:, j] = (col_filled - med) / scale
        cval = cand[j]
        if np.isfinite(cval):
            cand_out[j] = (float(np.clip(cval, lo, hi)) - med) / scale
        else:
            cand_out[j] = 0.0
    return out, cand_out


def _reranker_sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _second_stage_reranker_feature_names() -> Tuple[str, ...]:
    return _SECOND_STAGE_RERANKER_FEATURES


def _second_stage_reranker_feature_matrix(
    *,
    emb_raw: np.ndarray,
    candidate_vec_raw: np.ndarray,
    embedding_cols: Sequence[str],
    action_id: str,
    action_subtype: str,
    feature_weight_multipliers: Optional[Dict[str, float]] = None,
    feature_overrides: Optional[Dict[str, Sequence[float] | np.ndarray]] = None,
    profile_version: Optional[str] = None,
    target_action_scale: Optional[float] = None,
    row_action_scales: Optional[Sequence[float] | np.ndarray] = None,
) -> Dict[str, Any]:
    normalized_profile_version = str(profile_version or "").strip().lower()
    if normalized_profile_version == _WEIGHTED_DISTANCE_V2_VERSION:
        weighted_state = _weighted_state_similarity_v2(
            emb_raw=emb_raw,
            candidate_vec_raw=candidate_vec_raw,
            embedding_cols=embedding_cols,
            action_id=action_id,
            action_subtype=action_subtype,
            feature_weight_multipliers=feature_weight_multipliers,
        )
    else:
        weighted_state = _weighted_state_similarity(
            emb_raw=emb_raw,
            candidate_vec_raw=candidate_vec_raw,
            embedding_cols=embedding_cols,
            action_id=action_id,
            action_subtype=action_subtype,
            feature_weight_multipliers=feature_weight_multipliers,
        )
    profile = dict(weighted_state.get("profile") or {})
    transformed_emb_raw, transformed_candidate_vec_raw, _ = _apply_matching_feature_transforms(
        emb_raw,
        candidate_vec_raw,
        embedding_cols,
        profile,
    )
    emb_norm, candidate_vec = _winsorized_robust_standardize(
        transformed_emb_raw,
        transformed_candidate_vec_raw,
    )
    pair_present = np.isfinite(transformed_emb_raw) & np.isfinite(transformed_candidate_vec_raw.reshape(1, -1))
    abs_diff = np.abs(emb_norm - candidate_vec.reshape(1, -1))
    present_counts = np.sum(pair_present, axis=1).astype(float)
    unweighted_distance = np.full(emb_raw.shape[0], np.nan, dtype=float)
    valid_rows = present_counts > 0.0
    if bool(np.any(valid_rows)):
        unweighted_distance[valid_rows] = (
            np.sum(np.where(pair_present[valid_rows], abs_diff[valid_rows], 0.0), axis=1)
            / present_counts[valid_rows]
        )
    unweighted_similarity = np.where(
        np.isfinite(unweighted_distance),
        1.0 / (1.0 + unweighted_distance),
        0.0,
    )
    size_gap = np.asarray(weighted_state.get("size_gap", np.full(emb_raw.shape[0], np.nan)), dtype=float)
    primary_burden_gap = np.asarray(
        weighted_state.get("primary_burden_gap", np.full(emb_raw.shape[0], np.nan)),
        dtype=float,
    )
    soft_size_gap = float(_to_float(profile.get("soft_size_gap"), 0.35) or 0.35)
    soft_burden_gap = float(_to_float(profile.get("soft_burden_gap"), 1.25) or 1.25)
    size_guardrail_similarity = np.exp(
        -np.maximum(np.where(np.isfinite(size_gap), size_gap, soft_size_gap) - soft_size_gap, 0.0)
    )
    burden_guardrail_similarity = np.exp(
        -np.maximum(
            np.where(np.isfinite(primary_burden_gap), primary_burden_gap, soft_burden_gap) - soft_burden_gap,
            0.0,
        )
    )
    feature_matrix = np.column_stack(
        [
            np.asarray(weighted_state.get("state_similarity", np.zeros(emb_raw.shape[0])), dtype=float),
            unweighted_similarity,
            np.asarray(weighted_state.get("weighted_coverage", np.zeros(emb_raw.shape[0])), dtype=float),
            np.asarray(weighted_state.get("critical_coverage", np.zeros(emb_raw.shape[0])), dtype=float),
            size_guardrail_similarity,
            burden_guardrail_similarity,
            np.asarray(weighted_state.get("regime_similarity", np.zeros(emb_raw.shape[0])), dtype=float),
            np.asarray(weighted_state.get("parameter_similarity", np.zeros(emb_raw.shape[0])), dtype=float),
            np.asarray(weighted_state.get("sector_similarity", np.zeros(emb_raw.shape[0])), dtype=float),
            np.asarray(weighted_state.get("action_match_score", np.zeros(emb_raw.shape[0])), dtype=float),
            np.asarray(weighted_state.get("borrower_quality_similarity", np.ones(emb_raw.shape[0])), dtype=float),
            np.asarray(weighted_state.get("financing_pressure_similarity", np.ones(emb_raw.shape[0])), dtype=float),
            np.asarray(weighted_state.get("market_regime_similarity", np.ones(emb_raw.shape[0])), dtype=float),
            np.asarray(weighted_state.get("stress_alignment_similarity", np.ones(emb_raw.shape[0])), dtype=float),
            np.asarray(weighted_state.get("compatibility_penalty_factor", np.ones(emb_raw.shape[0])), dtype=float),
            np.ones(emb_raw.shape[0], dtype=float),
            np.ones(emb_raw.shape[0], dtype=float),
            np.ones(emb_raw.shape[0], dtype=float),
        ]
    ).astype(float)
    action_id_text = str(action_id or "").strip().lower()
    if _is_debt_support_action(action_id_text):
        debt_features = _debt_issuance_runtime_archetype_features(
            emb_raw=emb_raw,
            candidate_vec_raw=candidate_vec_raw,
            embedding_cols=embedding_cols,
            action_id_text=action_id_text,
            target_action_scale=target_action_scale,
            row_action_scales=row_action_scales,
            borrower_quality_similarity=np.asarray(
                weighted_state.get("borrower_quality_similarity", np.ones(emb_raw.shape[0])),
                dtype=float,
            ),
            financing_pressure_similarity=np.asarray(
                weighted_state.get("financing_pressure_similarity", np.ones(emb_raw.shape[0])),
                dtype=float,
            ),
            market_regime_similarity=np.asarray(
                weighted_state.get("market_regime_similarity", np.ones(emb_raw.shape[0])),
                dtype=float,
            ),
            stress_alignment_similarity=np.asarray(
                weighted_state.get("stress_alignment_similarity", np.ones(emb_raw.shape[0])),
                dtype=float,
            ),
        )
        feature_names = _second_stage_reranker_feature_names()
        debt_feature_map = {
            "debt_archetype_similarity": np.asarray(debt_features.get("archetype_similarity"), dtype=float),
            "debt_style_similarity": np.asarray(debt_features.get("style_similarity"), dtype=float),
            "debt_archetype_gate": np.asarray(debt_features.get("gate"), dtype=float),
        }
        for feature_name, values in debt_feature_map.items():
            feature_idx = feature_names.index(feature_name)
            if values.shape == (emb_raw.shape[0],):
                feature_matrix[:, feature_idx] = values
    feature_idx = {name: idx for idx, name in enumerate(_second_stage_reranker_feature_names())}
    for feature_name, override_values in dict(feature_overrides or {}).items():
        idx = feature_idx.get(str(feature_name))
        if idx is None:
            continue
        override_arr = np.asarray(override_values, dtype=float)
        if override_arr.ndim == 0:
            override_arr = np.full(emb_raw.shape[0], float(override_arr), dtype=float)
        if override_arr.shape != (emb_raw.shape[0],):
            continue
        feature_matrix[:, idx] = override_arr
    return {
        "feature_names": _second_stage_reranker_feature_names(),
        "matrix": feature_matrix,
        "weighted_state": weighted_state,
    }


def _outcome_aware_reranker_feature_names() -> Tuple[str, ...]:
    return _OUTCOME_AWARE_RERANKER_FEATURES


def _favorable_percentile_series(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return pd.Series(np.nan, index=numeric.index, dtype=float)
    ranked = numeric.rank(method="average", pct=True, ascending=not bool(higher_is_better))
    return ranked.astype(float).clip(0.0, 1.0)


def _outcome_aware_reranker_feature_frame(
    cohort: pd.DataFrame,
    *,
    similarity_col: str = "similarity_score",
) -> pd.DataFrame:
    out = pd.DataFrame(index=cohort.index.copy())
    similarity_series = cohort[similarity_col] if similarity_col in cohort.columns else pd.Series(np.nan, index=cohort.index, dtype=float)
    out["current_similarity_score"] = pd.to_numeric(similarity_series, errors="coerce").fillna(0.0).clip(0.0, 1.0)

    metric_coverages: List[pd.Series] = []
    for feature_name, metric_specs in _OUTCOME_AWARE_RERANKER_GROUPS.items():
        score_parts: List[pd.Series] = []
        for metric_name, higher_is_better in metric_specs:
            if metric_name not in cohort.columns:
                continue
            favorable = _favorable_percentile_series(cohort[metric_name], higher_is_better=higher_is_better)
            score_parts.append(favorable.rename(metric_name))
        if score_parts:
            score_frame = pd.concat(score_parts, axis=1)
            out[feature_name] = score_frame.mean(axis=1, skipna=True).fillna(0.5).clip(0.0, 1.0)
            metric_coverages.append(score_frame.notna().mean(axis=1).astype(float))
        else:
            out[feature_name] = 0.5
            metric_coverages.append(pd.Series(0.0, index=cohort.index, dtype=float))

    if metric_coverages:
        coverage_frame = pd.concat(metric_coverages, axis=1)
        out["outcome_support_score"] = coverage_frame.mean(axis=1, skipna=True).fillna(0.0).clip(0.0, 1.0)
    else:
        out["outcome_support_score"] = 0.0

    out = out.reindex(columns=list(_outcome_aware_reranker_feature_names())).astype(float)
    return out


def _apply_company_diversity_cap(
    cohort: pd.DataFrame,
    *,
    company_col: str = "company_id",
    cap: int,
) -> pd.DataFrame:
    if cohort.empty or cap <= 0 or company_col not in cohort.columns:
        return cohort
    company_ids = cohort[company_col].fillna("").astype(str)
    keep_rows: List[bool] = []
    counts: Dict[str, int] = {}
    for company_id in company_ids.tolist():
        seen = int(counts.get(company_id, 0))
        if seen >= int(cap):
            keep_rows.append(False)
            continue
        counts[company_id] = seen + 1
        keep_rows.append(True)
    if all(keep_rows):
        return cohort
    return cohort.loc[np.asarray(keep_rows, dtype=bool)].copy()


def _tail_candidates(
    cohort: pd.DataFrame,
    *,
    column: str,
    metric: str,
    horizon: str,
    min_points: int = 10,
    max_each_side: int = 3,
) -> List[TailEvent]:
    out: List[TailEvent] = []
    if column not in cohort.columns:
        return out
    s = pd.to_numeric(cohort[column], errors="coerce")
    valid = s.dropna()
    if valid.shape[0] < int(min_points):
        return out

    p10 = float(valid.quantile(0.10))
    p90 = float(valid.quantile(0.90))
    lows = cohort[s <= p10].head(max_each_side)
    highs = cohort[s >= p90].head(max_each_side)
    for _, row in lows.iterrows():
        out.append(
            TailEvent(
                precedent_id=f"{row.get('company_id')}::{row.get('action_date')}",
                outcome_metric=metric,
                outcome_value=float(_to_float(row.get(column), 0.0) or 0.0),
                horizon=horizon,
                explanation="Bottom decile historical outcome.",
            )
        )
    for _, row in highs.iterrows():
        out.append(
            TailEvent(
                precedent_id=f"{row.get('company_id')}::{row.get('action_date')}",
                outcome_metric=metric,
                outcome_value=float(_to_float(row.get(column), 0.0) or 0.0),
                horizon=horizon,
                explanation="Top decile historical outcome.",
            )
        )
    return out


def build_precedent_pack_v2(
    *,
    candidate_id: str,
    run_id: str,
    company_id: str,
    action_id: str,
    action_subtype: Optional[str],
    action_params: Dict[str, Any],
    candidate_features: Dict[str, Any],
    candidate_regime: Dict[str, Any],
    historical_df: Optional[pd.DataFrame] = None,
    historical_event_store: Optional[HistoricalEventStore] = None,
    historical_state_store: Optional[HistoricalCompanyStateSnapshotStore] = None,
    historical_outcome_store: Optional[HistoricalOutcomeStore] = None,
    regime_history: Optional[RegimeHistory] = None,
    retrieval_index: Optional[PrecedentRetrievalIndex] = None,
    top_k: int = 30,
    min_k: int = 10,
) -> PrecedentPack:
    t_start = time.perf_counter()
    profile: Dict[str, Any] = {}
    debug_steps = str(os.environ.get("RECO_PRECEDENT_DEBUG", "")).strip().lower() in {"1", "true", "yes", "on"}
    disable_narrative = str(os.environ.get("RECO_DISABLE_PRECEDENT_NARRATIVE", "")).strip().lower() in {"1", "true", "yes", "on"}

    def _debug(stage: str, **extra: Any) -> None:
        if not debug_steps:
            return
        payload = {
            "ok": True,
            "event": "precedent_debug",
            "candidate_id": str(candidate_id),
            "action_id": str(action_id),
            "stage": stage,
            "elapsed_seconds": round(time.perf_counter() - t_start, 6),
        }
        payload.update(extra)
        print(json.dumps(payload), flush=True)

    _debug("start")
    if retrieval_index is None and historical_df is None:
        if historical_event_store is None:
            raise ValueError("historical_df or historical_event_store is required")
        _debug("materialize_historical_frame:start")
        historical_df = materialize_historical_frame(
            historical_event_store=historical_event_store,
            historical_state_store=historical_state_store,
            historical_outcome_store=historical_outcome_store,
            regime_history=regime_history,
        )
        _debug("materialize_historical_frame:done", rows=int(len(historical_df)))

    if retrieval_index is None:
        assert historical_df is not None
        _debug("build_retrieval_index:start", rows=int(len(historical_df)))
        retrieval_index = build_precedent_retrieval_index(historical_df)
        _debug("build_retrieval_index:done", rows=int(retrieval_index.n_rows))
    profile["retrieval_index_seconds"] = round(time.perf_counter() - t_start, 6)

    if retrieval_index.n_rows == 0:
        return PrecedentPack(
            candidate_id=candidate_id,
            run_id=run_id,
            mismatch_diagnostics={"out_of_sample_flag": True, "reason": "no_historical_events"},
            calibration_confidence=0.0,
            profiling={**profile, "total_seconds": round(time.perf_counter() - t_start, 6)},
        )

    df = retrieval_index.df

    # Hierarchical action-conditional retrieval:
    # 1) exact action id / subtype matches
    # 2) domain-specific action family matches
    # 3) same action type family
    # 4) global fallback (only if sparse)
    action_id_text = str(action_id or "")
    action_subtype_text = _effective_action_subtype(
        action_id_text,
        action_subtype,
        action_params if isinstance(action_params, dict) else {},
    )
    action_leaf = action_id_text.split(".")[-1] if action_id_text else ""
    action_domain = action_id_text.split(".")[0] if "." in action_id_text else ""
    action_alias = action_id_to_outcomes_action_type(action_id_text, "")
    outcomes_subtype_alias = action_subtype_to_outcomes_subtype(
        action_id=action_id_text,
        action_type=action_alias,
        action_subtype=action_subtype_text,
    )
    action_family_scale_weights = _candidate_action_family_scale_weights(
        action_id_text,
        action_subtype_text,
        action_params if isinstance(action_params, dict) else {},
        candidate_features if isinstance(candidate_features, dict) else {},
    )
    action_family_weights = _candidate_action_family_weights(action_id_text, action_subtype_text)
    family_scale_keys = tuple(k for k, _ in action_family_scale_weights if k)
    family_keys = tuple(k for k, _ in action_family_weights if k)

    exact_keys = _candidate_exact_action_keys(
        action_id_text,
        action_subtype_text,
        action_leaf,
        outcomes_subtype_alias,
    )
    exact_parts: List[np.ndarray] = []
    for k in exact_keys:
        if k in retrieval_index.action_id_lookup:
            exact_parts.append(retrieval_index.action_id_lookup[k])
        if k in retrieval_index.subtype_lookup:
            exact_parts.append(retrieval_index.subtype_lookup[k])
    exact_idx = np.unique(np.concatenate(exact_parts)) if exact_parts else np.empty(0, dtype=np.int64)

    family_scale_option_parts: List[Tuple[str, np.ndarray]] = []
    for k in family_scale_keys:
        if k in retrieval_index.family_scale_lookup:
            family_scale_option_parts.append((k, retrieval_index.family_scale_lookup[k]))
    family_scale_idx = np.empty(0, dtype=np.int64)
    selected_family_scale_keys: Tuple[str, ...] = ()

    family_option_parts: List[Tuple[str, np.ndarray]] = []
    for k in family_keys:
        if k in retrieval_index.family_lookup:
            family_option_parts.append((k, retrieval_index.family_lookup[k]))
    family_idx = np.empty(0, dtype=np.int64)
    selected_family_keys: Tuple[str, ...] = ()

    type_keys = tuple(sorted({action_alias, action_domain}))
    type_keys = tuple(k for k in type_keys if k)
    type_parts: List[np.ndarray] = []
    for k in type_keys:
        if k in retrieval_index.type_lookup:
            type_parts.append(retrieval_index.type_lookup[k])
    type_idx = np.unique(np.concatenate(type_parts)) if type_parts else np.empty(0, dtype=np.int64)

    exact_support_min = _minimum_exact_support(min_k=int(min_k), top_k=int(top_k))

    retrieval_tier = "exact"
    if exact_idx.shape[0] >= int(exact_support_min):
        candidate_idx = exact_idx
    else:
        cumulative_family_scale_parts: List[np.ndarray] = []
        cumulative_family_scale_keys: List[str] = []
        for family_scale_key, family_scale_part in family_scale_option_parts:
            cumulative_family_scale_parts.append(family_scale_part)
            cumulative_family_scale_keys.append(family_scale_key)
            candidate_family_scale_idx = np.unique(np.concatenate(cumulative_family_scale_parts))
            if candidate_family_scale_idx.shape[0] >= int(exact_support_min):
                family_scale_idx = candidate_family_scale_idx
                selected_family_scale_keys = tuple(cumulative_family_scale_keys)
                break

        if family_scale_idx.shape[0] >= int(exact_support_min):
            candidate_idx = np.unique(np.concatenate([x for x in (exact_idx, family_scale_idx) if x.size]))
            retrieval_tier = "family"
        else:
            cumulative_family_parts: List[np.ndarray] = []
            cumulative_family_keys: List[str] = []
            for family_key, family_part in family_option_parts:
                cumulative_family_parts.append(family_part)
                cumulative_family_keys.append(family_key)
                candidate_family_idx = np.unique(np.concatenate(cumulative_family_parts))
                if candidate_family_idx.shape[0] >= int(exact_support_min):
                    family_idx = candidate_family_idx
                    selected_family_keys = tuple(cumulative_family_keys)
                    break
            if family_idx.shape[0] == 0 and family_option_parts:
                family_idx = np.unique(np.concatenate([part for _, part in family_option_parts]))
                selected_family_keys = tuple(k for k, _ in family_option_parts)

            if family_idx.shape[0] >= int(exact_support_min):
                candidate_idx = np.unique(np.concatenate([x for x in (exact_idx, family_idx) if x.size]))
                retrieval_tier = "family"
            elif family_scale_idx.size or family_idx.size or exact_idx.size or type_idx.size:
                candidate_idx = np.unique(
                    np.concatenate([x for x in (exact_idx, family_scale_idx, family_idx, type_idx) if x.size])
                )
                retrieval_tier = "family" if (family_scale_idx.size or family_idx.size) else "sibling_type"
            else:
                candidate_idx = np.empty(0, dtype=np.int64)
                retrieval_tier = "global"

    if candidate_idx.shape[0] < int(min_k):
        candidate_idx = np.arange(retrieval_index.n_rows, dtype=np.int64)
        retrieval_tier = "global"
    _debug(
        "candidate_pool:done",
        retrieval_tier=str(retrieval_tier),
        exact_match_count=int(exact_idx.shape[0]),
        family_scale_match_count=int(family_scale_idx.shape[0]),
        family_match_count=int(family_idx.shape[0]),
        type_match_count=int(type_idx.shape[0]),
        selected_family_scale_keys=list(selected_family_scale_keys),
        selected_family_keys=list(selected_family_keys),
        candidate_pool_size=int(candidate_idx.shape[0]),
    )
    profile["candidate_pool_seconds"] = round(time.perf_counter() - t_start - sum(float(profile.get(k, 0.0) or 0.0) for k in ("retrieval_index_seconds",)), 6)
    profile["candidate_pool_size"] = int(candidate_idx.shape[0])
    profile["exact_match_count"] = int(exact_idx.shape[0])
    profile["family_scale_match_count"] = int(family_scale_idx.shape[0])
    profile["family_match_count"] = int(family_idx.shape[0])
    profile["type_match_count"] = int(type_idx.shape[0])
    profile["selected_family_scale_keys"] = list(selected_family_scale_keys)
    profile["selected_family_keys"] = list(selected_family_keys)
    profile["retrieval_tier"] = str(retrieval_tier)

    # Coverage reflects exact action-conditional depth, not fallback pool size.
    low_precedent_coverage = int(exact_idx.shape[0]) < int(exact_support_min)

    # Hard pre-filter on sector + market-cap bucket before similarity scoring.
    # If this would make cohort too small, we explicitly relax and continue.
    candidate_feature_values = _flatten_matching_feature_payload(candidate_features if isinstance(candidate_features, dict) else {})
    cand_market_cap_value = _candidate_market_cap(candidate_feature_values)
    if cand_market_cap_value is not None and _to_float(candidate_feature_values.get("market_cap"), None) is None:
        candidate_feature_values["market_cap"] = cand_market_cap_value
    cand_sector_pref = _sector_token_from_candidate(candidate_feature_values)
    cand_subsector_pref = _subsector_token_from_candidate(candidate_feature_values)
    cand_market_cap = _candidate_market_cap(candidate_feature_values)
    target_action_scale = _estimate_action_scale(action_params, cand_market_cap)
    cand_market_cap_bucket = _market_cap_bucket_from_value(
        cand_market_cap,
        retrieval_index.market_cap_bucket_quantiles,
    )
    hard_prefilter_applied = False
    hard_prefilter_relaxed = False
    market_cap_prefilter_applied = False
    if candidate_idx.shape[0] >= int(min_k):
        keep_hard = np.ones(candidate_idx.shape[0], dtype=bool)
        has_hard_dims = False
        if cand_sector_pref:
            row_sectors_hard = retrieval_index.sector_token_arr[candidate_idx]
            row_subsectors_hard = retrieval_index.subsector_token_arr[candidate_idx]
            keep_hard &= np.fromiter(
                (
                    _sector_similarity(cand_sector_pref, str(x), cand_subsector_pref, str(row_subsectors_hard[i])) >= 0.75
                    for i, x in enumerate(row_sectors_hard)
                ),
                dtype=bool,
                count=candidate_idx.shape[0],
            )
            has_hard_dims = True
        if cand_market_cap_bucket >= 0:
            row_buckets_hard = retrieval_index.market_cap_bucket_arr[candidate_idx]
            keep_hard &= (row_buckets_hard == int(cand_market_cap_bucket))
            has_hard_dims = True
            market_cap_prefilter_applied = True
        if has_hard_dims:
            keep_n = int(np.count_nonzero(keep_hard))
            if keep_n >= int(min_k):
                candidate_idx = candidate_idx[keep_hard]
                hard_prefilter_applied = True
            else:
                hard_prefilter_relaxed = True
                market_cap_prefilter_applied = False

    # Optional pre-filters: tighten cohort to similar regimes/sectors when sufficiently deep.
    cand_flags = _candidate_regime_flags(candidate_regime)
    regime_prefilter_applied = False
    if candidate_idx.shape[0] >= max(int(min_k) * 3, 40):
        eq_total_pref = np.zeros(candidate_idx.shape[0], dtype=float)
        for k, flag in cand_flags.items():
            row_flag = retrieval_index.regime_flags.get(k)
            if row_flag is None:
                continue
            eq_total_pref += (row_flag[candidate_idx] == bool(flag)).astype(float)
        regime_similarity_pref = eq_total_pref / max(1, len(cand_flags))
        keep_regime = regime_similarity_pref >= 0.67
        if int(np.count_nonzero(keep_regime)) >= int(min_k):
            candidate_idx = candidate_idx[keep_regime]
            regime_prefilter_applied = True

    sector_prefilter_applied = False
    if cand_sector_pref and candidate_idx.shape[0] >= max(int(min_k) * 3, 40):
        row_sectors_pref = retrieval_index.sector_token_arr[candidate_idx]
        row_subsectors_pref = retrieval_index.subsector_token_arr[candidate_idx]
        keep_sector = np.fromiter(
            (
                _sector_similarity(cand_sector_pref, str(x), cand_subsector_pref, str(row_subsectors_pref[i])) >= 0.75
                for i, x in enumerate(row_sectors_pref)
            ),
            dtype=bool,
            count=candidate_idx.shape[0],
        )
        if int(np.count_nonzero(keep_sector)) >= int(min_k):
            candidate_idx = candidate_idx[keep_sector]
            sector_prefilter_applied = True
    profile["prefilter_seconds"] = round(
        time.perf_counter() - t_start - sum(float(profile.get(k, 0.0) or 0.0) for k in ("retrieval_index_seconds", "candidate_pool_seconds")),
        6,
    )
    profile["candidate_pool_size_after_prefilter"] = int(candidate_idx.shape[0])

    # Embeddings (vectorized) on the compact state vector, with legacy-compatible fallbacks.
    candidate_state_frame = augment_precedent_state_vector_columns(pd.DataFrame([candidate_feature_values]))
    candidate_state_row = candidate_state_frame.iloc[0] if not candidate_state_frame.empty else pd.Series(dtype=object)
    candidate_feature_weight_multipliers = _candidate_state_feature_weight_multipliers(
        candidate_features if isinstance(candidate_features, dict) else {},
        action_id=action_id_text,
        action_subtype=action_subtype_text,
    )
    def _candidate_embedding_value(feature_name: str) -> float:
        row_value = _to_float(candidate_state_row.get(feature_name), None) if feature_name in candidate_state_row.index else None
        if row_value is not None:
            return float(row_value)
        feature_value = _to_float(candidate_feature_values.get(feature_name), None)
        if feature_value is not None:
            return float(feature_value)
        return float("nan")
    candidate_vec_raw = np.array(
        [
            _candidate_embedding_value(col)
            for col in retrieval_index.embedding_cols
        ],
        dtype=float,
    )
    emb_raw = retrieval_index.embedding_values[candidate_idx]
    weighted_state = _weighted_state_similarity(
        emb_raw=emb_raw,
        candidate_vec_raw=candidate_vec_raw,
        embedding_cols=retrieval_index.embedding_cols,
        action_id=action_id_text,
        action_subtype=action_subtype_text,
        feature_weight_multipliers=candidate_feature_weight_multipliers,
        target_action_scale=target_action_scale,
    )
    coverage_gate_mask = np.asarray(weighted_state["coverage_gate_mask"], dtype=bool)
    size_gate_mask = np.asarray(weighted_state["size_gate_mask"], dtype=bool)
    weighted_distance = np.asarray(weighted_state["weighted_distance"], dtype=float)
    weighted_feature_coverage = np.asarray(weighted_state["weighted_coverage"], dtype=float)
    critical_feature_coverage = np.asarray(weighted_state["critical_coverage"], dtype=float)
    size_gap = np.asarray(weighted_state["size_gap"], dtype=float)
    primary_burden_gap = np.asarray(weighted_state["primary_burden_gap"], dtype=float)
    rate_gap = np.asarray(weighted_state.get("rate_gap", np.full(candidate_idx.shape[0], np.nan)), dtype=float)
    credit_gap = np.asarray(weighted_state.get("credit_gap", np.full(candidate_idx.shape[0], np.nan)), dtype=float)
    missing_penalty_factor = np.asarray(
        weighted_state.get("missing_penalty_factor", np.ones(candidate_idx.shape[0])),
        dtype=float,
    )
    regime_penalty_factor = np.asarray(
        weighted_state.get("regime_penalty_factor", np.ones(candidate_idx.shape[0])),
        dtype=float,
    )
    borrower_quality_similarity = np.asarray(
        weighted_state.get("borrower_quality_similarity", np.ones(candidate_idx.shape[0])),
        dtype=float,
    )
    financing_pressure_similarity = np.asarray(
        weighted_state.get("financing_pressure_similarity", np.ones(candidate_idx.shape[0])),
        dtype=float,
    )
    market_regime_similarity = np.asarray(
        weighted_state.get("market_regime_similarity", np.ones(candidate_idx.shape[0])),
        dtype=float,
    )
    stress_alignment_similarity = np.asarray(
        weighted_state.get("stress_alignment_similarity", np.ones(candidate_idx.shape[0])),
        dtype=float,
    )
    compatibility_penalty_factor = np.asarray(
        weighted_state.get("compatibility_penalty_factor", np.ones(candidate_idx.shape[0])),
        dtype=float,
    )
    borrower_quality_similarity = np.asarray(
        weighted_state.get("borrower_quality_similarity", np.ones(candidate_idx.shape[0])),
        dtype=float,
    )
    financing_pressure_similarity = np.asarray(
        weighted_state.get("financing_pressure_similarity", np.ones(candidate_idx.shape[0])),
        dtype=float,
    )
    market_regime_similarity = np.asarray(
        weighted_state.get("market_regime_similarity", np.ones(candidate_idx.shape[0])),
        dtype=float,
    )
    stress_alignment_similarity = np.asarray(
        weighted_state.get("stress_alignment_similarity", np.ones(candidate_idx.shape[0])),
        dtype=float,
    )
    compatibility_penalty_factor = np.asarray(
        weighted_state.get("compatibility_penalty_factor", np.ones(candidate_idx.shape[0])),
        dtype=float,
    )

    weighted_coverage_gate_applied = False
    weighted_coverage_gate_relaxed = False
    size_guardrail_applied = False
    size_guardrail_relaxed = False
    if candidate_idx.shape[0] >= int(min_k) and coverage_gate_mask.shape[0] == candidate_idx.shape[0]:
        keep_n = int(np.count_nonzero(coverage_gate_mask))
        if keep_n >= int(min_k):
            candidate_idx = candidate_idx[coverage_gate_mask]
            emb_raw = emb_raw[coverage_gate_mask]
            weighted_distance = weighted_distance[coverage_gate_mask]
            weighted_feature_coverage = weighted_feature_coverage[coverage_gate_mask]
            critical_feature_coverage = critical_feature_coverage[coverage_gate_mask]
            size_gap = size_gap[coverage_gate_mask]
            primary_burden_gap = primary_burden_gap[coverage_gate_mask]
            rate_gap = rate_gap[coverage_gate_mask]
            credit_gap = credit_gap[coverage_gate_mask]
            missing_penalty_factor = missing_penalty_factor[coverage_gate_mask]
            regime_penalty_factor = regime_penalty_factor[coverage_gate_mask]
            borrower_quality_similarity = borrower_quality_similarity[coverage_gate_mask]
            financing_pressure_similarity = financing_pressure_similarity[coverage_gate_mask]
            market_regime_similarity = market_regime_similarity[coverage_gate_mask]
            stress_alignment_similarity = stress_alignment_similarity[coverage_gate_mask]
            compatibility_penalty_factor = compatibility_penalty_factor[coverage_gate_mask]
            weighted_coverage_gate_applied = True
        else:
            weighted_coverage_gate_relaxed = True
    if candidate_idx.shape[0] >= int(min_k) and size_gate_mask.shape[0] != candidate_idx.shape[0]:
        size_gate_mask = np.asarray(
            _weighted_state_similarity(
                emb_raw=emb_raw,
                candidate_vec_raw=candidate_vec_raw,
                embedding_cols=retrieval_index.embedding_cols,
                action_id=action_id_text,
                action_subtype=action_subtype_text,
                feature_weight_multipliers=candidate_feature_weight_multipliers,
                target_action_scale=target_action_scale,
            )["size_gate_mask"],
            dtype=bool,
        )
    if candidate_idx.shape[0] >= int(min_k) and size_gate_mask.shape[0] == candidate_idx.shape[0]:
        keep_n = int(np.count_nonzero(size_gate_mask))
        if keep_n >= int(min_k):
            candidate_idx = candidate_idx[size_gate_mask]
            emb_raw = emb_raw[size_gate_mask]
            weighted_distance = weighted_distance[size_gate_mask]
            weighted_feature_coverage = weighted_feature_coverage[size_gate_mask]
            critical_feature_coverage = critical_feature_coverage[size_gate_mask]
            size_gap = size_gap[size_gate_mask]
            primary_burden_gap = primary_burden_gap[size_gate_mask]
            rate_gap = rate_gap[size_gate_mask]
            credit_gap = credit_gap[size_gate_mask]
            missing_penalty_factor = missing_penalty_factor[size_gate_mask]
            regime_penalty_factor = regime_penalty_factor[size_gate_mask]
            borrower_quality_similarity = borrower_quality_similarity[size_gate_mask]
            financing_pressure_similarity = financing_pressure_similarity[size_gate_mask]
            market_regime_similarity = market_regime_similarity[size_gate_mask]
            stress_alignment_similarity = stress_alignment_similarity[size_gate_mask]
            compatibility_penalty_factor = compatibility_penalty_factor[size_gate_mask]
            size_guardrail_applied = True
        else:
            size_guardrail_relaxed = True

    weighted_state = _weighted_state_similarity(
        emb_raw=emb_raw,
        candidate_vec_raw=candidate_vec_raw,
        embedding_cols=retrieval_index.embedding_cols,
        action_id=action_id_text,
        action_subtype=action_subtype_text,
        feature_weight_multipliers=candidate_feature_weight_multipliers,
        target_action_scale=target_action_scale,
    )
    state_similarity = np.asarray(weighted_state["state_similarity"], dtype=float)
    weighted_distance = np.asarray(weighted_state["weighted_distance"], dtype=float)
    weighted_feature_coverage = np.asarray(weighted_state["weighted_coverage"], dtype=float)
    critical_feature_coverage = np.asarray(weighted_state["critical_coverage"], dtype=float)
    size_gap = np.asarray(weighted_state["size_gap"], dtype=float)
    primary_burden_gap = np.asarray(weighted_state["primary_burden_gap"], dtype=float)
    rate_gap = np.asarray(weighted_state.get("rate_gap", np.full(candidate_idx.shape[0], np.nan)), dtype=float)
    credit_gap = np.asarray(weighted_state.get("credit_gap", np.full(candidate_idx.shape[0], np.nan)), dtype=float)
    missing_penalty_factor = np.asarray(
        weighted_state.get("missing_penalty_factor", np.ones(candidate_idx.shape[0])),
        dtype=float,
    )
    regime_penalty_factor = np.asarray(
        weighted_state.get("regime_penalty_factor", np.ones(candidate_idx.shape[0])),
        dtype=float,
    )

    # Regime similarity (vectorized over precomputed row flags).
    cand_flags = _candidate_regime_flags(candidate_regime)
    eq_total = np.zeros(candidate_idx.shape[0], dtype=float)
    for k, flag in cand_flags.items():
        row_flag = retrieval_index.regime_flags.get(k)
        if row_flag is None:
            continue
        eq_total += (row_flag[candidate_idx] == bool(flag)).astype(float)
    regime_similarity = eq_total / max(1, len(cand_flags))

    # Parameter scale similarity.
    cand_scale = _estimate_action_scale(action_params, _candidate_market_cap(candidate_feature_values))
    hist_scale = retrieval_index.action_scale_arr[candidate_idx]
    eps = 1e-6
    param_similarity = np.exp(-np.abs(np.log((cand_scale + eps) / (hist_scale + eps))))

    # Action match score (explicitly in retrieval blend).
    subtype_vals = retrieval_index.action_subtype_arr[candidate_idx]
    type_vals = retrieval_index.action_type_arr[candidate_idx]
    action_id_vals = retrieval_index.action_id_arr[candidate_idx]
    family_vals = retrieval_index.action_family_arr[candidate_idx]
    family_scale_vals = retrieval_index.action_family_scale_arr[candidate_idx]
    subtype_key_list = [k for k in (action_subtype_text, action_leaf) if k]
    type_key_list = [k for k in (action_alias, action_domain) if k]
    action_id_key_list = list(_candidate_action_id_keys(action_id_text, action_subtype_text))
    subtype_match = np.isin(subtype_vals, subtype_key_list)
    type_match = np.isin(type_vals, type_key_list)
    action_id_match = np.isin(action_id_vals, action_id_key_list)
    family_scale_match_score = np.zeros(candidate_idx.shape[0], dtype=float)
    for family_scale_key, score in action_family_scale_weights:
        family_scale_mask = np.isin(family_scale_vals, [family_scale_key])
        family_scale_match_score[family_scale_mask] = np.maximum(
            family_scale_match_score[family_scale_mask],
            float(score),
        )
    family_match_score = np.zeros(candidate_idx.shape[0], dtype=float)
    for family_key, score in action_family_weights:
        family_mask = np.isin(family_vals, [family_key])
        family_match_score[family_mask] = np.maximum(family_match_score[family_mask], float(score))
    action_match_score = np.full(candidate_idx.shape[0], 0.45, dtype=float)
    action_match_score[type_match] = np.maximum(action_match_score[type_match], 0.65)
    action_match_score = np.maximum(action_match_score, family_scale_match_score)
    action_match_score = np.maximum(action_match_score, family_match_score)
    action_match_score[subtype_match] = np.maximum(action_match_score[subtype_match], 0.80)
    action_match_score[action_id_match] = np.maximum(action_match_score[action_id_match], 0.92)
    action_match_score = np.clip(action_match_score, 0.0, 1.0)

    # Sector similarity.
    cand_sector = _sector_token_from_candidate(candidate_feature_values)
    cand_subsector = _subsector_token_from_candidate(candidate_feature_values)
    row_sectors = retrieval_index.sector_token_arr[candidate_idx]
    row_subsectors = retrieval_index.subsector_token_arr[candidate_idx]
    sector_similarity = np.fromiter(
        (
            _sector_similarity(cand_sector, str(x), cand_subsector, str(row_subsectors[i]))
            for i, x in enumerate(row_sectors)
        ),
        dtype=float,
        count=candidate_idx.shape[0],
    )
    identity_prefilter_applied = False
    identity_prefilter_mode = ""
    if str(weighted_state.get("version") or "") == _WEIGHTED_DISTANCE_V2_VERSION and cand_sector and candidate_idx.shape[0] >= max(3, int(top_k)):
        known_sector_mask = np.fromiter((bool(str(x or "").strip()) for x in row_sectors), dtype=bool, count=candidate_idx.shape[0])
        same_subsector_mask = sector_similarity >= 0.999
        same_sector_mask = sector_similarity >= 0.75
        identity_keep_mask: Optional[np.ndarray] = None
        min_same_subsector = max(2, min(int(top_k), 4))
        min_same_sector = max(3, min(int(top_k), 6))
        if int(np.count_nonzero(same_subsector_mask)) >= int(min_same_subsector):
            identity_keep_mask = same_subsector_mask | (~known_sector_mask)
            identity_prefilter_mode = "subsector"
        elif int(np.count_nonzero(same_sector_mask)) >= int(min_same_sector):
            identity_keep_mask = same_sector_mask | (~known_sector_mask)
            identity_prefilter_mode = "sector"
        if identity_keep_mask is not None:
            keep_n = int(np.count_nonzero(identity_keep_mask))
            if keep_n >= max(3, min(int(top_k), int(candidate_idx.shape[0]))):
                candidate_idx = candidate_idx[identity_keep_mask]
                emb_raw = emb_raw[identity_keep_mask]
                state_similarity = state_similarity[identity_keep_mask]
                weighted_distance = weighted_distance[identity_keep_mask]
                weighted_feature_coverage = weighted_feature_coverage[identity_keep_mask]
                critical_feature_coverage = critical_feature_coverage[identity_keep_mask]
                size_gap = size_gap[identity_keep_mask]
                primary_burden_gap = primary_burden_gap[identity_keep_mask]
                rate_gap = rate_gap[identity_keep_mask]
                credit_gap = credit_gap[identity_keep_mask]
                missing_penalty_factor = missing_penalty_factor[identity_keep_mask]
                regime_penalty_factor = regime_penalty_factor[identity_keep_mask]
                borrower_quality_similarity = borrower_quality_similarity[identity_keep_mask]
                financing_pressure_similarity = financing_pressure_similarity[identity_keep_mask]
                market_regime_similarity = market_regime_similarity[identity_keep_mask]
                stress_alignment_similarity = stress_alignment_similarity[identity_keep_mask]
                compatibility_penalty_factor = compatibility_penalty_factor[identity_keep_mask]
                regime_similarity = regime_similarity[identity_keep_mask]
                hist_scale = hist_scale[identity_keep_mask]
                param_similarity = param_similarity[identity_keep_mask]
                action_match_score = action_match_score[identity_keep_mask]
                row_sectors = row_sectors[identity_keep_mask]
                row_subsectors = row_subsectors[identity_keep_mask]
                sector_similarity = sector_similarity[identity_keep_mask]
                identity_prefilter_applied = True

    debt_target_archetype_label = ""
    debt_row_archetype_labels = np.array([""] * candidate_idx.shape[0], dtype=object)
    debt_archetype_similarity = np.ones(candidate_idx.shape[0], dtype=float)
    debt_style_similarity = np.ones(candidate_idx.shape[0], dtype=float)
    debt_archetype_gate = np.ones(candidate_idx.shape[0], dtype=float)
    debt_archetype_prefilter_applied = False
    debt_archetype_prefilter_mode = ""
    if _is_debt_support_action(action_id_text) and candidate_idx.size:
        score_keys = tuple(_DEBT_ISSUANCE_ARCHETYPE_LABELS)
        is_revolver_action = action_id_text == "capital_structure.revolver_draw_or_resize"
        target_compact_values = {
            str(col): float(candidate_vec_raw[idx])
            for idx, col in enumerate(retrieval_index.embedding_cols)
            if col in _STATE_VECTOR_MATCHING_COLS and np.isfinite(candidate_vec_raw[idx])
        }
        target_debt_profile = _debt_issuance_runtime_archetype_profile(
            target_compact_values,
            action_id_text=action_id_text,
            action_scale=target_action_scale,
        )
        debt_target_archetype_label = str(target_debt_profile.get("label") or "")
        target_scores = {
            key: float(value)
            for key, value in dict(target_debt_profile.get("scores") or {}).items()
            if key in score_keys and _to_float(value, None) is not None
        }
        feature_index = {str(col): idx for idx, col in enumerate(retrieval_index.embedding_cols)}
        growth_idx = feature_index.get("state_vector_v1.growth")
        valuation_idx = feature_index.get("state_vector_v1.valuation_multiple")
        access_idx = feature_index.get("state_vector_v1.market_access")
        stress_idx = feature_index.get("state_vector_v1.market_stress")
        credit_spread_idx = feature_index.get("state_vector_v1.credit_spread")
        net_burden_idx = feature_index.get("state_vector_v1.net_obligation_burden")
        liquidity_idx = feature_index.get("state_vector_v1.liquidity_flexibility")
        cash_generation_idx = feature_index.get("state_vector_v1.cash_generation")
        target_growth = _to_float(candidate_vec_raw[growth_idx], None) if growth_idx is not None else None
        target_valuation = _to_float(candidate_vec_raw[valuation_idx], None) if valuation_idx is not None else None
        target_access = _to_float(candidate_vec_raw[access_idx], None) if access_idx is not None else None
        target_market_stress = _to_float(candidate_vec_raw[stress_idx], None) if stress_idx is not None else None
        target_credit_spread = (
            _to_float(candidate_vec_raw[credit_spread_idx], None) if credit_spread_idx is not None else None
        )
        target_net_burden = _to_float(candidate_vec_raw[net_burden_idx], None) if net_burden_idx is not None else None
        target_liquidity = _to_float(candidate_vec_raw[liquidity_idx], None) if liquidity_idx is not None else None
        target_cash_generation = (
            _to_float(candidate_vec_raw[cash_generation_idx], None) if cash_generation_idx is not None else None
        )
        cross_label_penalty = {
            "distressed_borrower": {
                "distressed_borrower": 1.0,
                "refinancing_pressure": 0.72,
                "opportunistic_issuer": 0.28,
            },
            "refinancing_pressure": {
                "distressed_borrower": 0.68,
                "refinancing_pressure": 1.0,
                "opportunistic_issuer": 0.35,
            },
            "opportunistic_issuer": {
                "distressed_borrower": 0.22,
                "refinancing_pressure": 0.45,
                "opportunistic_issuer": 1.0,
            },
        }
        row_labels: List[str] = []
        for row_idx in range(candidate_idx.shape[0]):
            row_compact = {
                str(col): float(emb_raw[row_idx, idx])
                for idx, col in enumerate(retrieval_index.embedding_cols)
                if col in _STATE_VECTOR_MATCHING_COLS and np.isfinite(emb_raw[row_idx, idx])
            }
            row_profile = _debt_issuance_runtime_archetype_profile(
                row_compact,
                action_id_text=action_id_text,
                action_scale=_to_float(hist_scale[row_idx], None),
            )
            row_label = str(row_profile.get("label") or "")
            row_labels.append(row_label)
            row_scores = {
                key: float(value)
                for key, value in dict(row_profile.get("scores") or {}).items()
                if key in score_keys and _to_float(value, None) is not None
            }
            shared_keys = [key for key in score_keys if key in target_scores and key in row_scores]
            if shared_keys:
                archetype_distance = float(
                    np.mean([abs(float(target_scores[key]) - float(row_scores[key])) for key in shared_keys])
                )
                debt_archetype_similarity[row_idx] = float(np.exp(-2.60 * archetype_distance))
            label_factor = float(
                (cross_label_penalty.get(debt_target_archetype_label) or {}).get(row_label, 0.35)
            )
            debt_archetype_gate[row_idx] = float(
                label_factor * np.exp(-2.40 * max(0.72 - debt_archetype_similarity[row_idx], 0.0))
            )
            if debt_target_archetype_label == "opportunistic_issuer":
                style_components: List[float] = []
                row_growth = _to_float(emb_raw[row_idx, growth_idx], None) if growth_idx is not None else None
                row_valuation = _to_float(emb_raw[row_idx, valuation_idx], None) if valuation_idx is not None else None
                row_access = _to_float(emb_raw[row_idx, access_idx], None) if access_idx is not None else None
                row_net_burden = _to_float(emb_raw[row_idx, net_burden_idx], None) if net_burden_idx is not None else None
                if is_revolver_action:
                    row_stress = _to_float(emb_raw[row_idx, stress_idx], None) if stress_idx is not None else None
                    row_credit = (
                        _to_float(emb_raw[row_idx, credit_spread_idx], None) if credit_spread_idx is not None else None
                    )
                    row_liquidity = (
                        _to_float(emb_raw[row_idx, liquidity_idx], None) if liquidity_idx is not None else None
                    )
                    if target_liquidity is not None and row_liquidity is not None:
                        style_components.append(float(np.exp(-abs(float(row_liquidity) - float(target_liquidity)) / 0.85)))
                    if target_access is not None and row_access is not None:
                        style_components.append(float(np.exp(-abs(float(row_access) - float(target_access)) / 0.12)))
                    if target_market_stress is not None and row_stress is not None:
                        style_components.append(
                            float(np.exp(-abs(float(row_stress) - float(target_market_stress)) / 0.08))
                        )
                    if target_credit_spread is not None and row_credit is not None:
                        style_components.append(
                            float(np.exp(-abs(float(row_credit) - float(target_credit_spread)) / 0.70))
                        )
                    if target_growth is not None and row_growth is not None:
                        style_components.append(float(np.exp(-abs(float(row_growth) - float(target_growth)) / 0.20)))
                else:
                    if target_growth is not None and row_growth is not None:
                        style_components.append(float(np.exp(-abs(float(row_growth) - float(target_growth)) / 0.16)))
                    if target_valuation is not None and row_valuation is not None:
                        style_components.append(
                            float(
                                np.exp(
                                    -abs(float(row_valuation) - float(target_valuation))
                                    / max(8.0, 0.22 * abs(float(target_valuation)) + 2.0)
                                )
                            )
                        )
                    if target_access is not None and row_access is not None:
                        style_components.append(float(np.exp(-abs(float(row_access) - float(target_access)) / 0.14)))
                if style_components:
                    debt_style_similarity[row_idx] = float(np.exp(np.mean(np.log(np.clip(style_components, 1e-9, 1.0)))))
                else:
                    debt_style_similarity[row_idx] = debt_archetype_similarity[row_idx]
                debt_archetype_gate[row_idx] *= float(
                    np.exp(-2.60 * max(0.70 - debt_style_similarity[row_idx], 0.0))
                    * np.exp(-2.00 * max(0.68 - market_regime_similarity[row_idx], 0.0))
                )
                if is_revolver_action:
                    if target_access is not None and target_access >= 0.78 and row_access is not None:
                        debt_archetype_gate[row_idx] *= float(
                            np.exp(-2.25 * max((float(target_access) - 0.14) - float(row_access), 0.0))
                        )
                    if target_liquidity is not None and row_liquidity is not None:
                        min_liquidity = max(0.90, float(target_liquidity) - 0.85)
                        debt_archetype_gate[row_idx] *= float(
                            np.exp(-1.10 * max(min_liquidity - float(row_liquidity), 0.0))
                        )
                else:
                    if target_access is not None and target_access >= 0.85 and row_access is not None:
                        debt_archetype_gate[row_idx] *= float(
                            np.exp(-2.10 * max((float(target_access) - 0.18) - float(row_access), 0.0))
                        )
                    if target_valuation is not None and target_valuation >= 20.0 and row_valuation is not None:
                        min_valuation = max(8.0, 0.40 * float(target_valuation))
                        debt_archetype_gate[row_idx] *= float(
                            np.exp(-0.16 * max(min_valuation - float(row_valuation), 0.0))
                        )
                    if target_net_burden is not None and target_net_burden <= -2.0 and row_net_burden is not None:
                        debt_archetype_gate[row_idx] *= float(
                            np.exp(-1.35 * max(float(row_net_burden) - 0.75, 0.0))
                        )
            elif debt_target_archetype_label == "distressed_borrower":
                debt_style_similarity[row_idx] = float(borrower_quality_similarity[row_idx])
                row_valuation = _to_float(emb_raw[row_idx, valuation_idx], None) if valuation_idx is not None else None
                row_cash_generation = (
                    _to_float(emb_raw[row_idx, cash_generation_idx], None) if cash_generation_idx is not None else None
                )
                row_liquidity = _to_float(emb_raw[row_idx, liquidity_idx], None) if liquidity_idx is not None else None
                stress_factor = (
                    1.0
                    if float(stress_alignment_similarity[row_idx]) >= 0.999
                    else (0.72 if float(stress_alignment_similarity[row_idx]) >= 0.70 else 0.38)
                )
                debt_archetype_gate[row_idx] *= float(
                    stress_factor
                    * np.exp(-2.30 * max(0.72 - float(borrower_quality_similarity[row_idx]), 0.0))
                    * np.exp(-1.80 * max(0.62 - float(market_regime_similarity[row_idx]), 0.0))
                )
                if is_revolver_action:
                    if target_liquidity is not None and row_liquidity is not None:
                        max_liquidity = max(1.40, float(target_liquidity) + 0.75)
                        debt_archetype_gate[row_idx] *= float(
                            np.exp(-1.45 * max(float(row_liquidity) - max_liquidity, 0.0))
                        )
                    if target_cash_generation is not None and target_cash_generation < 0.0 and row_cash_generation is not None:
                        debt_archetype_gate[row_idx] *= float(
                            np.exp(-3.20 * max(float(row_cash_generation) - 0.03, 0.0))
                        )
                else:
                    if target_valuation is not None and target_valuation <= 8.0 and row_valuation is not None:
                        max_valuation = max(12.0, float(target_valuation) + 6.0)
                        debt_archetype_gate[row_idx] *= float(
                            np.exp(-0.22 * max(float(row_valuation) - max_valuation, 0.0))
                        )
                    if target_cash_generation is not None and target_cash_generation < 0.0 and row_cash_generation is not None:
                        debt_archetype_gate[row_idx] *= float(
                            np.exp(-3.50 * max(float(row_cash_generation) - 0.04, 0.0))
                        )
            else:
                debt_style_similarity[row_idx] = float(financing_pressure_similarity[row_idx])
                row_liquidity = _to_float(emb_raw[row_idx, liquidity_idx], None) if liquidity_idx is not None else None
                debt_archetype_gate[row_idx] *= float(
                    np.exp(-2.50 * max(0.74 - float(financing_pressure_similarity[row_idx]), 0.0))
                    * np.exp(-2.10 * max(0.68 - float(market_regime_similarity[row_idx]), 0.0))
                )
                if target_liquidity is not None and target_liquidity <= 1.50 and row_liquidity is not None:
                    max_liquidity = max(3.0, float(target_liquidity) + 2.5)
                    debt_archetype_gate[row_idx] *= float(
                        np.exp(-0.85 * max(float(row_liquidity) - max_liquidity, 0.0))
                    )
            rate_threshold = 1.50
            credit_threshold = 1.35
            if debt_target_archetype_label == "distressed_borrower":
                rate_threshold = 1.65
                credit_threshold = 1.45
            elif debt_target_archetype_label == "opportunistic_issuer":
                rate_threshold = 1.20
                credit_threshold = 1.15
            if np.isfinite(rate_gap[row_idx]):
                debt_archetype_gate[row_idx] *= float(
                    np.exp(-1.60 * max(float(rate_gap[row_idx]) - rate_threshold, 0.0))
                )
            if np.isfinite(credit_gap[row_idx]):
                debt_archetype_gate[row_idx] *= float(
                    np.exp(-1.90 * max(float(credit_gap[row_idx]) - credit_threshold, 0.0))
                )

        debt_row_archetype_labels = np.asarray(row_labels, dtype=object)
        debt_archetype_gate = np.clip(debt_archetype_gate, 0.05, 1.0)
        same_archetype_mask = debt_row_archetype_labels == debt_target_archetype_label
        if debt_target_archetype_label == "opportunistic_issuer":
            if is_revolver_action:
                strict_mask = (
                    (rate_gap <= 1.35)
                    & (credit_gap <= 1.30)
                    & (market_regime_similarity >= 0.62)
                    & (debt_style_similarity >= 0.66)
                )
                relaxed_mask = (
                    (rate_gap <= 1.90)
                    & (credit_gap <= 1.75)
                    & (market_regime_similarity >= 0.58)
                    & (debt_style_similarity >= 0.58)
                )
                if target_access is not None and access_idx is not None:
                    row_access_arr = np.asarray(emb_raw[:, access_idx], dtype=float)
                    strict_mask = strict_mask & np.where(
                        np.isfinite(row_access_arr),
                        row_access_arr >= max(0.64, float(target_access) - 0.14),
                        False,
                    )
                    relaxed_mask = relaxed_mask & np.where(
                        np.isfinite(row_access_arr),
                        row_access_arr >= max(0.56, float(target_access) - 0.24),
                        False,
                    )
                if target_liquidity is not None and liquidity_idx is not None:
                    row_liquidity_arr = np.asarray(emb_raw[:, liquidity_idx], dtype=float)
                    strict_mask = strict_mask & np.where(
                        np.isfinite(row_liquidity_arr),
                        row_liquidity_arr >= max(0.90, float(target_liquidity) - 0.85),
                        False,
                    )
                    relaxed_mask = relaxed_mask & np.where(
                        np.isfinite(row_liquidity_arr),
                        row_liquidity_arr >= max(0.60, float(target_liquidity) - 1.30),
                        False,
                    )
            else:
                strict_mask = (
                    (rate_gap <= 1.20)
                    & (credit_gap <= 1.15)
                    & (market_regime_similarity >= 0.64)
                    & (debt_style_similarity >= 0.68)
                )
                relaxed_mask = (
                    (rate_gap <= 1.80)
                    & (credit_gap <= 1.60)
                    & (market_regime_similarity >= 0.58)
                    & (debt_style_similarity >= 0.60)
                )
                if target_access is not None and target_access >= 0.85 and access_idx is not None:
                    row_access_arr = np.asarray(emb_raw[:, access_idx], dtype=float)
                    strict_mask = strict_mask & np.where(
                        np.isfinite(row_access_arr),
                        row_access_arr >= max(0.70, float(target_access) - 0.18),
                        False,
                    )
                    relaxed_mask = relaxed_mask & np.where(
                        np.isfinite(row_access_arr),
                        row_access_arr >= max(0.60, float(target_access) - 0.28),
                        False,
                    )
                if target_valuation is not None and target_valuation >= 20.0 and valuation_idx is not None:
                    row_valuation_arr = np.asarray(emb_raw[:, valuation_idx], dtype=float)
                    strict_mask = strict_mask & np.where(
                        np.isfinite(row_valuation_arr),
                        row_valuation_arr >= max(10.0, 0.40 * float(target_valuation)),
                        False,
                    )
                    relaxed_mask = relaxed_mask & np.where(
                        np.isfinite(row_valuation_arr),
                        row_valuation_arr >= max(8.0, 0.28 * float(target_valuation)),
                        False,
                    )
                if target_net_burden is not None and target_net_burden <= -2.0 and net_burden_idx is not None:
                    row_net_burden_arr = np.asarray(emb_raw[:, net_burden_idx], dtype=float)
                    strict_mask = strict_mask & np.where(
                        np.isfinite(row_net_burden_arr),
                        row_net_burden_arr <= 0.75,
                        False,
                    )
                    relaxed_mask = relaxed_mask & np.where(
                        np.isfinite(row_net_burden_arr),
                        row_net_burden_arr <= 2.0,
                        False,
                    )
            preferred_mask = (
                same_archetype_mask
                & (debt_archetype_similarity >= 0.66)
                & (debt_style_similarity >= (0.62 if is_revolver_action else 0.60))
                & (market_regime_similarity >= (0.60 if is_revolver_action else 0.58))
                & strict_mask
            )
            fallback_mask = (
                (debt_archetype_gate >= (0.40 if is_revolver_action else 0.38))
                & (debt_style_similarity >= (0.56 if is_revolver_action else 0.54))
                & (market_regime_similarity >= (0.57 if is_revolver_action else 0.55))
                & relaxed_mask
            )
        elif debt_target_archetype_label == "distressed_borrower":
            if is_revolver_action:
                strict_mask = (
                    (rate_gap <= 1.55)
                    & (credit_gap <= 1.45)
                    & (borrower_quality_similarity >= 0.60)
                    & (stress_alignment_similarity >= 0.72)
                )
                relaxed_mask = (
                    (rate_gap <= 2.10)
                    & (credit_gap <= 1.95)
                    & (borrower_quality_similarity >= 0.56)
                )
                if target_liquidity is not None and liquidity_idx is not None:
                    row_liquidity_arr = np.asarray(emb_raw[:, liquidity_idx], dtype=float)
                    strict_mask = strict_mask & np.where(
                        np.isfinite(row_liquidity_arr),
                        row_liquidity_arr <= max(1.40, float(target_liquidity) + 0.75),
                        False,
                    )
                    relaxed_mask = relaxed_mask & np.where(
                        np.isfinite(row_liquidity_arr),
                        row_liquidity_arr <= max(2.10, float(target_liquidity) + 1.35),
                        False,
                    )
            else:
                strict_mask = (
                    (rate_gap <= 1.65)
                    & (credit_gap <= 1.45)
                    & (borrower_quality_similarity >= 0.58)
                )
                relaxed_mask = (
                    (rate_gap <= 2.35)
                    & (credit_gap <= 2.05)
                    & (borrower_quality_similarity >= 0.54)
                )
                if target_valuation is not None and target_valuation <= 8.0 and valuation_idx is not None:
                    row_valuation_arr = np.asarray(emb_raw[:, valuation_idx], dtype=float)
                    strict_mask = strict_mask & np.where(
                        np.isfinite(row_valuation_arr),
                        row_valuation_arr <= max(12.0, float(target_valuation) + 6.0),
                        False,
                    )
                    relaxed_mask = relaxed_mask & np.where(
                        np.isfinite(row_valuation_arr),
                        row_valuation_arr <= max(18.0, float(target_valuation) + 10.0),
                        False,
                    )
                if target_cash_generation is not None and target_cash_generation < 0.0 and cash_generation_idx is not None:
                    row_cash_generation_arr = np.asarray(emb_raw[:, cash_generation_idx], dtype=float)
                    strict_mask = strict_mask & np.where(
                        np.isfinite(row_cash_generation_arr),
                        row_cash_generation_arr <= 0.04,
                        False,
                    )
                    relaxed_mask = relaxed_mask & np.where(
                        np.isfinite(row_cash_generation_arr),
                        row_cash_generation_arr <= 0.08,
                        False,
                    )
            preferred_mask = (
                same_archetype_mask
                & (borrower_quality_similarity >= (0.64 if is_revolver_action else 0.62))
                & (stress_alignment_similarity >= (0.74 if is_revolver_action else 0.70))
                & (market_regime_similarity >= (0.56 if is_revolver_action else 0.54))
                & strict_mask
            )
            fallback_mask = (
                (debt_archetype_gate >= (0.42 if is_revolver_action else 0.40))
                & (borrower_quality_similarity >= (0.61 if is_revolver_action else 0.60))
                & (market_regime_similarity >= (0.54 if is_revolver_action else 0.52))
                & relaxed_mask
            )
        else:
            strict_mask = (
                (rate_gap <= (1.35 if is_revolver_action else 1.50))
                & (credit_gap <= (1.30 if is_revolver_action else 1.35))
                & (financing_pressure_similarity >= (0.62 if is_revolver_action else 0.60))
                & (market_regime_similarity >= (0.62 if is_revolver_action else 0.60))
            )
            relaxed_mask = (
                (rate_gap <= (1.90 if is_revolver_action else 2.10))
                & (credit_gap <= (1.75 if is_revolver_action else 1.85))
                & (financing_pressure_similarity >= (0.56 if is_revolver_action else 0.54))
                & (market_regime_similarity >= (0.58 if is_revolver_action else 0.56))
            )
            if target_liquidity is not None and target_liquidity <= 1.50 and liquidity_idx is not None:
                row_liquidity_arr = np.asarray(emb_raw[:, liquidity_idx], dtype=float)
                strict_mask = strict_mask & np.where(
                    np.isfinite(row_liquidity_arr),
                    row_liquidity_arr <= max(2.40 if is_revolver_action else 3.0, float(target_liquidity) + (1.80 if is_revolver_action else 2.5)),
                    False,
                )
                relaxed_mask = relaxed_mask & np.where(
                    np.isfinite(row_liquidity_arr),
                    row_liquidity_arr <= max(3.80 if is_revolver_action else 5.0, float(target_liquidity) + (2.80 if is_revolver_action else 4.0)),
                    False,
                )
            preferred_mask = (
                same_archetype_mask
                & (financing_pressure_similarity >= (0.64 if is_revolver_action else 0.62))
                & (market_regime_similarity >= (0.58 if is_revolver_action else 0.56))
                & strict_mask
            )
            fallback_mask = (
                (debt_archetype_gate >= (0.44 if is_revolver_action else 0.42))
                & (financing_pressure_similarity >= (0.60 if is_revolver_action else 0.58))
                & (market_regime_similarity >= (0.56 if is_revolver_action else 0.54))
                & relaxed_mask
            )
        min_prefilter_keep = max(3, min(int(top_k), 6))
        debt_keep_mask: Optional[np.ndarray] = None
        if int(np.count_nonzero(preferred_mask)) >= int(min_prefilter_keep):
            debt_keep_mask = preferred_mask
            debt_archetype_prefilter_mode = "preferred"
        elif int(np.count_nonzero(fallback_mask)) >= int(min_prefilter_keep):
            debt_keep_mask = fallback_mask
            debt_archetype_prefilter_mode = "fallback"
        if debt_keep_mask is not None:
            candidate_idx = candidate_idx[debt_keep_mask]
            state_similarity = state_similarity[debt_keep_mask]
            weighted_distance = weighted_distance[debt_keep_mask]
            weighted_feature_coverage = weighted_feature_coverage[debt_keep_mask]
            critical_feature_coverage = critical_feature_coverage[debt_keep_mask]
            size_gap = size_gap[debt_keep_mask]
            primary_burden_gap = primary_burden_gap[debt_keep_mask]
            rate_gap = rate_gap[debt_keep_mask]
            credit_gap = credit_gap[debt_keep_mask]
            missing_penalty_factor = missing_penalty_factor[debt_keep_mask]
            regime_penalty_factor = regime_penalty_factor[debt_keep_mask]
            borrower_quality_similarity = borrower_quality_similarity[debt_keep_mask]
            financing_pressure_similarity = financing_pressure_similarity[debt_keep_mask]
            market_regime_similarity = market_regime_similarity[debt_keep_mask]
            stress_alignment_similarity = stress_alignment_similarity[debt_keep_mask]
            compatibility_penalty_factor = compatibility_penalty_factor[debt_keep_mask]
            regime_similarity = regime_similarity[debt_keep_mask]
            hist_scale = hist_scale[debt_keep_mask]
            param_similarity = param_similarity[debt_keep_mask]
            action_match_score = action_match_score[debt_keep_mask]
            row_sectors = row_sectors[debt_keep_mask]
            row_subsectors = row_subsectors[debt_keep_mask]
            sector_similarity = sector_similarity[debt_keep_mask]
            debt_row_archetype_labels = debt_row_archetype_labels[debt_keep_mask]
            debt_archetype_similarity = debt_archetype_similarity[debt_keep_mask]
            debt_style_similarity = debt_style_similarity[debt_keep_mask]
            debt_archetype_gate = debt_archetype_gate[debt_keep_mask]
            debt_archetype_prefilter_applied = True

    # Dynamic sector-aware weighting and action-match inclusion.
    blend_weights = dict((weighted_state.get("profile", {}) or {}).get("blend_weights") or {})
    if str(weighted_state.get("version") or "") == _WEIGHTED_DISTANCE_V2_VERSION and blend_weights:
        w_state = float(_to_float(blend_weights.get("state"), 0.58) or 0.58)
        w_regime = float(_to_float(blend_weights.get("regime"), 0.12) or 0.12)
        w_param = float(_to_float(blend_weights.get("param"), 0.12) or 0.12)
        w_sector = float(_to_float(blend_weights.get("sector"), 0.10) or 0.10)
        w_action = float(_to_float(blend_weights.get("action"), 0.08) or 0.08)
        weight_total = max(1e-12, w_state + w_regime + w_param + w_sector + w_action)
        w_state, w_regime, w_param, w_sector, w_action = (
            w_state / weight_total,
            w_regime / weight_total,
            w_param / weight_total,
            w_sector / weight_total,
            w_action / weight_total,
        )
    elif cand_sector:
        w_state, w_regime, w_param, w_sector, w_action = 0.52, 0.16, 0.12, 0.12, 0.08
    else:
        w_state, w_regime, w_param, w_sector, w_action = 0.56, 0.18, 0.14, 0.04, 0.08
    tier_penalty = 1.0 if retrieval_tier == "exact" else (0.96 if retrieval_tier == "sibling_type" else 0.88)
    similarity_score = (
        w_state * state_similarity
        + w_regime * regime_similarity
        + w_param * param_similarity
        + w_sector * sector_similarity
        + w_action * action_match_score
    )
    similarity_score = similarity_score * tier_penalty
    if _is_debt_support_action(action_id_text):
        debt_penalty = np.sqrt(np.clip(compatibility_penalty_factor * debt_archetype_gate, 0.0, 1.0))
        similarity_score = similarity_score * debt_penalty
    if str(weighted_state.get("version") or "") == _WEIGHTED_DISTANCE_V2_VERSION:
        sector_penalty_weight = float(
            _to_float((weighted_state.get("profile", {}) or {}).get("sector_penalty_weight"), 0.30) or 0.30
        )
        sector_penalty_factor = np.exp(-sector_penalty_weight * np.maximum(1.0 - sector_similarity, 0.0))
        similarity_score = similarity_score * sector_penalty_factor
    else:
        sector_penalty_factor = np.ones(candidate_idx.shape[0], dtype=float)
    profile["vector_similarity_seconds"] = round(
        time.perf_counter() - t_start - sum(float(profile.get(k, 0.0) or 0.0) for k in ("retrieval_index_seconds", "candidate_pool_seconds", "prefilter_seconds")),
        6,
    )
    profile["state_distance_version"] = str(weighted_state.get("version") or _WEIGHTED_DISTANCE_V1_VERSION)
    profile["weight_scope"] = str((weighted_state.get("profile", {}) or {}).get("weight_scope") or "prior_only")
    profile["weighted_coverage_gate_applied"] = bool(weighted_coverage_gate_applied)
    profile["weighted_coverage_gate_relaxed"] = bool(weighted_coverage_gate_relaxed)
    profile["size_guardrail_applied"] = bool(size_guardrail_applied)
    profile["size_guardrail_relaxed"] = bool(size_guardrail_relaxed)
    profile["identity_prefilter_applied"] = bool(identity_prefilter_applied)
    profile["identity_prefilter_mode"] = str(identity_prefilter_mode or "")
    profile["debt_target_archetype_label"] = str(debt_target_archetype_label or "")
    profile["debt_archetype_prefilter_applied"] = bool(debt_archetype_prefilter_applied)
    profile["debt_archetype_prefilter_mode"] = str(debt_archetype_prefilter_mode or "")
    profile["second_stage_reranker_applied"] = False
    profile["second_stage_reranker_shortlist_size"] = 0
    profile["second_stage_reranker_features"] = list(
        dict((weighted_state.get("profile", {}) or {}).get("second_stage_reranker", {}).get("feature_weights", {}) or {}).keys()
    )
    profile["outcome_aware_reranker_applied"] = False
    profile["outcome_aware_reranker_shortlist_size"] = 0
    profile["outcome_aware_reranker_features"] = list(
        dict((weighted_state.get("profile", {}) or {}).get("outcome_aware_reranker", {}).get("feature_weights", {}) or {}).keys()
    )
    profile["max_matches_per_company"] = int(
        _to_float((weighted_state.get("profile", {}) or {}).get("max_matches_per_company"), 0.0) or 0.0
    )
    profile["company_diversity_cap_applied"] = False
    profile["company_diversity_cap_relaxed"] = False
    _debug("vector_similarity:done", candidate_pool_size=int(candidate_idx.shape[0]))

    # Deterministic ordering: similarity desc, action_date desc, company_id asc.
    cohort = df.iloc[candidate_idx].copy()
    cohort["state_similarity"] = state_similarity
    cohort["state_distance"] = weighted_distance
    cohort["weighted_feature_coverage"] = weighted_feature_coverage
    cohort["critical_feature_coverage"] = critical_feature_coverage
    cohort["size_gap"] = size_gap
    cohort["primary_burden_gap"] = primary_burden_gap
    cohort["rate_gap"] = rate_gap
    cohort["credit_gap"] = credit_gap
    cohort["missing_penalty_factor"] = missing_penalty_factor
    cohort["regime_penalty_factor"] = regime_penalty_factor
    cohort["borrower_quality_similarity"] = borrower_quality_similarity
    cohort["financing_pressure_similarity"] = financing_pressure_similarity
    cohort["market_regime_similarity"] = market_regime_similarity
    cohort["stress_alignment_similarity"] = stress_alignment_similarity
    cohort["compatibility_penalty_factor"] = compatibility_penalty_factor
    cohort["regime_similarity"] = regime_similarity
    cohort["parameter_similarity"] = param_similarity
    cohort["action_match_score"] = action_match_score
    cohort["sector_similarity"] = sector_similarity
    cohort["sector_penalty_factor"] = sector_penalty_factor
    cohort["debt_archetype_label"] = debt_row_archetype_labels
    cohort["debt_archetype_similarity"] = debt_archetype_similarity
    cohort["debt_style_similarity"] = debt_style_similarity
    cohort["debt_archetype_gate"] = debt_archetype_gate
    cohort["similarity_score"] = similarity_score
    cohort["action_scale_ratio"] = hist_scale
    cohort = cohort.sort_values(["similarity_score", "action_date", "company_id"], ascending=[False, False, True])

    reranker_profile = dict((weighted_state.get("profile", {}) or {}).get("second_stage_reranker") or {})
    second_stage_reranker_applied = False
    second_stage_reranker_shortlist_size = 0
    if reranker_profile and isinstance(reranker_profile.get("feature_weights"), dict) and not cohort.empty:
        reranker_feature_names = _second_stage_reranker_feature_names()
        reranker_weight_vector = np.array(
            [
                float(_to_float((reranker_profile.get("feature_weights") or {}).get(name), 0.0) or 0.0)
                for name in reranker_feature_names
            ],
            dtype=float,
        )
        if bool(np.any(reranker_weight_vector > 0.0)):
            shortlist_n = min(
                int(len(cohort)),
                max(
                    int(top_k),
                    int(_to_float(reranker_profile.get("shortlist_size"), max(80, int(top_k) * 4)) or max(80, int(top_k) * 4)),
                ),
            )
            shortlist_n = max(shortlist_n, min(int(len(cohort)), int(top_k)))
            shortlist = cohort.head(shortlist_n).copy()
            shortlist_emb_raw = np.column_stack(
                [
                    pd.to_numeric(shortlist.get(col), errors="coerce").to_numpy(dtype=float)
                    for col in retrieval_index.embedding_cols
                ]
            )
            reranker_features = _second_stage_reranker_feature_matrix(
                emb_raw=shortlist_emb_raw,
                candidate_vec_raw=candidate_vec_raw,
                embedding_cols=retrieval_index.embedding_cols,
                action_id=action_id_text,
                action_subtype=action_subtype_text,
                feature_weight_multipliers=candidate_feature_weight_multipliers,
                target_action_scale=target_action_scale,
                row_action_scales=pd.to_numeric(shortlist.get("action_scale_ratio"), errors="coerce").to_numpy(dtype=float),
                feature_overrides={
                    "regime_similarity": pd.to_numeric(shortlist.get("regime_similarity"), errors="coerce").to_numpy(dtype=float),
                    "parameter_similarity": pd.to_numeric(
                        shortlist.get("parameter_similarity"),
                        errors="coerce",
                    ).to_numpy(dtype=float),
                    "sector_similarity": pd.to_numeric(shortlist.get("sector_similarity"), errors="coerce").to_numpy(dtype=float),
                    "action_match_score": pd.to_numeric(
                        shortlist.get("action_match_score"),
                        errors="coerce",
                    ).to_numpy(dtype=float),
                    "borrower_quality_similarity": pd.to_numeric(
                        shortlist.get("borrower_quality_similarity"),
                        errors="coerce",
                    ).to_numpy(dtype=float),
                    "financing_pressure_similarity": pd.to_numeric(
                        shortlist.get("financing_pressure_similarity"),
                        errors="coerce",
                    ).to_numpy(dtype=float),
                    "market_regime_similarity": pd.to_numeric(
                        shortlist.get("market_regime_similarity"),
                        errors="coerce",
                    ).to_numpy(dtype=float),
                    "stress_alignment_similarity": pd.to_numeric(
                        shortlist.get("stress_alignment_similarity"),
                        errors="coerce",
                    ).to_numpy(dtype=float),
                    "compatibility_penalty_factor": pd.to_numeric(
                        shortlist.get("compatibility_penalty_factor"),
                        errors="coerce",
                    ).to_numpy(dtype=float),
                    "debt_archetype_similarity": pd.to_numeric(
                        shortlist.get("debt_archetype_similarity"),
                        errors="coerce",
                    ).to_numpy(dtype=float),
                    "debt_style_similarity": pd.to_numeric(
                        shortlist.get("debt_style_similarity"),
                        errors="coerce",
                    ).to_numpy(dtype=float),
                    "debt_archetype_gate": pd.to_numeric(
                        shortlist.get("debt_archetype_gate"),
                        errors="coerce",
                    ).to_numpy(dtype=float),
                },
            )
            reranker_matrix = np.asarray(reranker_features.get("matrix"), dtype=float)
            if (
                reranker_matrix.ndim == 2
                and reranker_matrix.shape[0] == int(len(shortlist))
                and reranker_matrix.shape[1] == int(len(reranker_feature_names))
            ):
                reranker_scores = _reranker_sigmoid(
                    float(_to_float(reranker_profile.get("bias"), 0.0) or 0.0)
                    + reranker_matrix @ reranker_weight_vector
                )
                shortlist["base_similarity_score"] = pd.to_numeric(
                    shortlist.get("similarity_score"),
                    errors="coerce",
                ).to_numpy(dtype=float)
                if _is_debt_support_action(action_id_text):
                    reranker_scores = reranker_scores * pd.to_numeric(
                        shortlist.get("compatibility_penalty_factor"),
                        errors="coerce",
                    ).fillna(1.0).to_numpy(dtype=float)
                    reranker_scores = reranker_scores * pd.to_numeric(
                        shortlist.get("debt_archetype_gate"),
                        errors="coerce",
                    ).fillna(1.0).to_numpy(dtype=float)
                shortlist["second_stage_reranker_score"] = reranker_scores
                shortlist["similarity_score"] = reranker_scores
                shortlist = shortlist.sort_values(["similarity_score", "action_date", "company_id"], ascending=[False, False, True])
                tail = cohort.iloc[shortlist_n:].copy()
                tail["second_stage_reranker_score"] = np.nan
                cohort = pd.concat([shortlist, tail], axis=0)
                second_stage_reranker_applied = True
                second_stage_reranker_shortlist_size = int(shortlist_n)
    profile["second_stage_reranker_applied"] = bool(second_stage_reranker_applied)
    profile["second_stage_reranker_shortlist_size"] = int(second_stage_reranker_shortlist_size)

    outcome_reranker_profile = dict((weighted_state.get("profile", {}) or {}).get("outcome_aware_reranker") or {})
    outcome_aware_reranker_applied = False
    outcome_aware_reranker_shortlist_size = 0
    if outcome_reranker_profile and isinstance(outcome_reranker_profile.get("feature_weights"), dict) and not cohort.empty:
        outcome_feature_names = _outcome_aware_reranker_feature_names()
        outcome_weight_vector = np.array(
            [
                float(_to_float((outcome_reranker_profile.get("feature_weights") or {}).get(name), 0.0) or 0.0)
                for name in outcome_feature_names
            ],
            dtype=float,
        )
        if bool(np.any(outcome_weight_vector > 0.0)):
            shortlist_n = min(
                int(len(cohort)),
                max(
                    int(top_k),
                    int(
                        _to_float(
                            outcome_reranker_profile.get("shortlist_size"),
                            max(40, int(top_k) * 2),
                        )
                        or max(40, int(top_k) * 2)
                    ),
                ),
            )
            shortlist_n = max(shortlist_n, min(int(len(cohort)), int(top_k)))
            feature_frame = _outcome_aware_reranker_feature_frame(cohort)
            shortlist = cohort.head(shortlist_n).copy()
            shortlist_features = feature_frame.loc[shortlist.index, list(outcome_feature_names)].to_numpy(dtype=float)
            if (
                shortlist_features.ndim == 2
                and shortlist_features.shape[0] == int(len(shortlist))
                and shortlist_features.shape[1] == int(len(outcome_feature_names))
            ):
                outcome_scores = _reranker_sigmoid(
                    float(_to_float(outcome_reranker_profile.get("bias"), 0.0) or 0.0)
                    + shortlist_features @ outcome_weight_vector
                )
                for feature_idx, feature_name in enumerate(outcome_feature_names):
                    shortlist[feature_name] = shortlist_features[:, feature_idx]
                shortlist["pre_outcome_aware_similarity_score"] = pd.to_numeric(
                    shortlist.get("similarity_score"),
                    errors="coerce",
                ).to_numpy(dtype=float)
                shortlist["outcome_aware_reranker_score"] = outcome_scores
                shortlist["similarity_score"] = outcome_scores
                shortlist = shortlist.sort_values(["similarity_score", "action_date", "company_id"], ascending=[False, False, True])
                tail = cohort.iloc[shortlist_n:].copy()
                tail["outcome_aware_reranker_score"] = np.nan
                cohort = pd.concat([shortlist, tail], axis=0)
                outcome_aware_reranker_applied = True
                outcome_aware_reranker_shortlist_size = int(shortlist_n)
    profile["outcome_aware_reranker_applied"] = bool(outcome_aware_reranker_applied)
    profile["outcome_aware_reranker_shortlist_size"] = int(outcome_aware_reranker_shortlist_size)

    # Narrative similarity on a bounded pool for cost control.
    narrative_pool_n = max(int(top_k) * 4, int(min_k) * 2, 200)
    narrative_pool = cohort.head(narrative_pool_n).copy()
    t_narr = time.perf_counter()
    _debug("narrative_similarity:start", narrative_pool_size=int(len(narrative_pool)), disabled=bool(disable_narrative))
    if disable_narrative:
        narr_sims = np.zeros(len(narrative_pool), dtype=float)
        narrative_real_count = 0
        narrative_top5_mean = 0.0
    else:
        narr_sims, narrative_real_count, narrative_top5_mean = _narrative_similarity(
            narrative_pool,
            action_id=str(action_id),
            action_subtype=action_subtype,
            action_params=action_params,
            candidate_features=candidate_features if isinstance(candidate_features, dict) else {},
            candidate_regime=candidate_regime if isinstance(candidate_regime, dict) else {},
        )
    cohort["narrative_similarity"] = 0.0
    if len(narrative_pool) == len(narr_sims) and len(narrative_pool) > 0:
        cohort.loc[narrative_pool.index, "narrative_similarity"] = narr_sims
        # Small lexical refinement so real text match improves rank but does not dominate.
        base_scores = pd.to_numeric(cohort.loc[narrative_pool.index, "similarity_score"], errors="coerce").fillna(0.0)
        adjusted = 0.95 * base_scores.to_numpy(dtype=float) + 0.05 * narr_sims
        cohort.loc[narrative_pool.index, "similarity_score"] = adjusted
        cohort = cohort.sort_values(["similarity_score", "action_date", "company_id"], ascending=[False, False, True])
    debt_support_routing_applied = False
    debt_support_lane_counts: Dict[str, int] = {}
    debt_primary_support_count = 0
    debt_same_company_primary_count = 0
    if _is_debt_support_action(action_id_text) and not cohort.empty:
        cohort, routing_profile = _apply_debt_support_routing(
            cohort,
            target_company_id=str(company_id or ""),
            target_label=str(debt_target_archetype_label or ""),
        )
        debt_support_routing_applied = bool(routing_profile.get("applied"))
        debt_support_lane_counts = {
            str(key): int(value)
            for key, value in dict(routing_profile.get("lane_counts") or {}).items()
        }
        debt_primary_support_count = int(_to_float(routing_profile.get("primary_support_count"), 0.0) or 0.0)
        debt_same_company_primary_count = int(
            _to_float(routing_profile.get("same_company_primary_count"), 0.0) or 0.0
        )
    profile["debt_support_routing_applied"] = bool(debt_support_routing_applied)
    profile["debt_support_lane_counts"] = dict(debt_support_lane_counts)
    profile["debt_primary_support_count"] = int(debt_primary_support_count)
    profile["debt_same_company_primary_count"] = int(debt_same_company_primary_count)
    max_matches_per_company = int(
        _to_float((weighted_state.get("profile", {}) or {}).get("max_matches_per_company"), 0.0) or 0.0
    )
    if max_matches_per_company <= 0 and str(action_id_text or "") == "capital_structure.revolver_draw_or_resize":
        max_matches_per_company = 2
    profile["max_matches_per_company"] = int(max_matches_per_company)
    if max_matches_per_company > 0 and not cohort.empty:
        required_n = min(int(len(cohort)), max(int(top_k), int(min_k)))
        diversity_capped = _apply_company_diversity_cap(
            cohort,
            company_col="company_id",
            cap=max_matches_per_company,
        )
        if len(diversity_capped) >= int(required_n):
            cohort = diversity_capped
            profile["company_diversity_cap_applied"] = True
        else:
            profile["company_diversity_cap_relaxed"] = True
    cohort = cohort.head(max(int(top_k), int(min_k)))
    profile["narrative_similarity_seconds"] = round(time.perf_counter() - t_narr, 6)
    profile["narrative_pool_size"] = int(len(narrative_pool))
    _debug(
        "narrative_similarity:done",
        narrative_pool_size=int(len(narrative_pool)),
        narrative_real_count=int(narrative_real_count),
        narrative_seconds=float(profile["narrative_similarity_seconds"]),
    )
    t_pack = time.perf_counter()
    _debug("pack_finalize:start", cohort_size=int(len(cohort)))
    cohort_idx = cohort.index.to_numpy(dtype=np.int64, copy=False)
    cohort_company_ids = retrieval_index.company_id_arr[cohort_idx]
    cohort_action_dates = retrieval_index.action_date_arr[cohort_idx]
    cohort_action_keys = retrieval_index.preferred_action_key_arr[cohort_idx]
    cohort_source_event_ids = retrieval_index.source_event_id_arr[cohort_idx]
    cohort_sector_values = retrieval_index.feature_sector_arr[cohort_idx]
    cohort_regime_flags = {
        regime_label: regime_mask[cohort_idx]
        for regime_label, regime_mask in retrieval_index.regime_flags.items()
    }

    def _cohort_numeric(column: str) -> np.ndarray:
        if column not in cohort.columns:
            return np.full(len(cohort), np.nan, dtype=float)
        return pd.to_numeric(cohort[column], errors="coerce").to_numpy(dtype=float)

    cohort_action_sizes = _cohort_numeric("action_size")
    cohort_state_similarity = _cohort_numeric("state_similarity")
    cohort_regime_similarity = _cohort_numeric("regime_similarity")
    cohort_parameter_similarity = _cohort_numeric("parameter_similarity")
    cohort_action_match_score = _cohort_numeric("action_match_score")
    cohort_sector_similarity = _cohort_numeric("sector_similarity")
    cohort_similarity_score = _cohort_numeric("similarity_score")
    cohort_weighted_feature_coverage = _cohort_numeric("weighted_feature_coverage")
    cohort_critical_feature_coverage = _cohort_numeric("critical_feature_coverage")
    cohort_state_distance = _cohort_numeric("state_distance")
    retrieved: List[PrecedentCase] = []
    sim_scores: List[SimilarityScore] = []
    for idx in range(len(cohort_idx)):
        action_date = cohort_action_dates[idx]
        precedent_id = f"{cohort_company_ids[idx]}::{action_date}::{idx}"
        regime_row = {
            regime_label: bool(regime_mask[idx])
            for regime_label, regime_mask in cohort_regime_flags.items()
        }
        key_state_features = _feature_key_state(cohort.iloc[idx])
        if not key_state_features.get("base_sector"):
            key_state_features["base_sector"] = str(cohort_sector_values[idx] or "").strip()
        retrieved.append(
            PrecedentCase(
                precedent_id=precedent_id,
                company_id=str(cohort_company_ids[idx] or ""),
                decision_time=str(pd.to_datetime(action_date, errors="coerce")),
                action_id=str(cohort_action_keys[idx] or ""),
                parameters={"action_size": float(cohort_action_sizes[idx]) if np.isfinite(cohort_action_sizes[idx]) else None},
                regime=regime_row,
                similarity_score=float(cohort_similarity_score[idx]) if np.isfinite(cohort_similarity_score[idx]) else 0.0,
                key_state_features=key_state_features,
                source_event_id=str(cohort_source_event_ids[idx] or precedent_id),
            )
        )
        sim_scores.append(
            SimilarityScore(
                precedent_id=precedent_id,
                score=float(cohort_similarity_score[idx]) if np.isfinite(cohort_similarity_score[idx]) else 0.0,
                state_similarity=float(cohort_state_similarity[idx]) if np.isfinite(cohort_state_similarity[idx]) else 0.0,
                regime_similarity=float(cohort_regime_similarity[idx]) if np.isfinite(cohort_regime_similarity[idx]) else 0.0,
                parameter_similarity=float(cohort_parameter_similarity[idx]) if np.isfinite(cohort_parameter_similarity[idx]) else 0.0,
                action_match_score=float(cohort_action_match_score[idx]) if np.isfinite(cohort_action_match_score[idx]) else 0.0,
                sector_similarity=float(cohort_sector_similarity[idx]) if np.isfinite(cohort_sector_similarity[idx]) else 0.0,
            )
        )

    out_dists = _build_outcome_distributions(cohort)

    # Legacy distributions list for downstream blend compatibility.
    legacy_distributions = [
        ImpactDistribution(
            metric="outcome_pe_6m",
            horizon_months=6,
            p25=out_dists.horizon_6m.valuation_multiple_change.p25,
            p50=out_dists.horizon_6m.valuation_multiple_change.median,
            p75=out_dists.horizon_6m.valuation_multiple_change.p75,
            n=out_dists.horizon_6m.valuation_multiple_change.sample_size,
        ),
        ImpactDistribution(
            metric="outcome_pe_12m",
            horizon_months=12,
            p25=out_dists.horizon_12m.valuation_multiple_change.p25,
            p50=out_dists.horizon_12m.valuation_multiple_change.median,
            p75=out_dists.horizon_12m.valuation_multiple_change.p75,
            n=out_dists.horizon_12m.valuation_multiple_change.sample_size,
        ),
        ImpactDistribution(
            metric="outcome_ev_ebitda_12m",
            horizon_months=12,
            p25=_dist(pd.to_numeric(cohort.get("outcome_ev_ebitda_12m"), errors="coerce")).p25,
            p50=_dist(pd.to_numeric(cohort.get("outcome_ev_ebitda_12m"), errors="coerce")).median,
            p75=_dist(pd.to_numeric(cohort.get("outcome_ev_ebitda_12m"), errors="coerce")).p75,
            n=_dist(pd.to_numeric(cohort.get("outcome_ev_ebitda_12m"), errors="coerce")).sample_size,
        ),
    ]

    # Tail events across core metrics/horizons.
    tail_events: List[TailEvent] = []
    tail_specs = [
        ("outcome_pe_6m", "equity_return_vs_sector", "6m"),
        ("outcome_pe_12m", "equity_return_vs_sector", "12m"),
        ("outcome_ev_ebitda_6m", "valuation_multiple_change", "6m"),
        ("outcome_ev_ebitda_12m", "valuation_multiple_change", "12m"),
        ("credit_spread_change_1m", "credit_spread_change", "1m"),
        ("credit_spread_change_6m", "credit_spread_change", "6m"),
        ("credit_spread_change_12m", "credit_spread_change", "12m"),
        ("credit_spread_change_24m", "credit_spread_change", "24m"),
        ("rating_migration_1m", "rating_migration", "1m"),
        ("rating_migration_6m", "rating_migration", "6m"),
        ("rating_migration_12m", "rating_migration", "12m"),
        ("rating_migration_24m", "rating_migration", "24m"),
    ]
    for col, metric, horizon in tail_specs:
        tail_events.extend(_tail_candidates(cohort, column=col, metric=metric, horizon=horizon))

    # Regime splits
    regime_splits: List[RegimeDistribution] = []
    for regime_label in ["credit_tight", "credit_loose", "risk_off", "risk_on", "high_vol", "low_vol"]:
        mask_reg = cohort_regime_flags.get(regime_label)
        if mask_reg is None:
            continue
        subset = cohort.iloc[np.flatnonzero(mask_reg)]
        if subset.empty:
            continue
        regime_splits.append(
            RegimeDistribution(
                regime_label=regime_label,
                outcome_distributions=_build_outcome_distributions(subset),
                sample_size=int(len(subset)),
            )
        )

    # Second-order outcomes
    follow_on_rows: Dict[str, List[float]] = {}
    window_days = np.timedelta64(730, "D")
    for company_key, action_date in zip(cohort_company_ids, cohort_action_dates):
        if pd.isna(action_date):
            continue
        follow_on_window = retrieval_index.company_follow_on_lookup.get(str(company_key or ""))
        if follow_on_window is None:
            continue
        company_dates, company_action_keys = follow_on_window
        left = int(np.searchsorted(company_dates, action_date, side="right"))
        right = int(np.searchsorted(company_dates, action_date + window_days, side="right"))
        if right <= left:
            continue
        action_keys_window = company_action_keys[left:right]
        if action_keys_window.size == 0:
            continue
        day_offsets = ((company_dates[left:right] - action_date) / np.timedelta64(1, "D")).astype(float)
        for aid, dt in zip(action_keys_window.tolist(), day_offsets.tolist()):
            action_key = str(aid or "").strip()
            if not action_key:
                continue
            follow_on_rows.setdefault(action_key, []).append(float(dt))

    second_order: List[FollowOnOutcome] = []
    denom = max(1, len(cohort))
    for aid, times in sorted(follow_on_rows.items(), key=lambda x: (-len(x[1]), x[0])):
        second_order.append(
            FollowOnOutcome(
                follow_on_action_id=aid,
                frequency=float(len(times) / denom),
                average_time_to_follow_on=float(np.mean(times)) if times else None,
                median_time_to_follow_on=float(np.median(times)) if times else None,
            )
        )
    second_order = second_order[:10]

    # Mismatch diagnostics
    mismatches: List[FeatureMismatch] = []
    feature_map = {
        feature_name: _to_float(candidate_state_row.get(feature_name))
        for feature_name in retrieval_index.embedding_cols
    }
    for name, cand_v in feature_map.items():
        if cand_v is None or name not in cohort.columns:
            continue
        s = pd.to_numeric(cohort[name], errors="coerce").dropna()
        if s.empty:
            continue
        lo = float(s.quantile(0.05))
        hi = float(s.quantile(0.95))
        if cand_v < lo or cand_v > hi:
            mismatches.append(
                FeatureMismatch(
                    feature_name=name,
                    candidate_value=cand_v,
                    cohort_range=f"[{lo:.4f}, {hi:.4f}]",
                    explanation="Candidate feature lies outside the 5-95% precedent cohort range.",
                )
            )

    avg_regime_similarity = float(cohort["regime_similarity"].mean()) if "regime_similarity" in cohort.columns else 0.0
    regime_mismatch = avg_regime_similarity < 0.55

    scale_series = pd.to_numeric(cohort.get("action_scale_ratio"), errors="coerce").dropna()
    param_mismatch = False
    if not scale_series.empty:
        slo = float(scale_series.quantile(0.10))
        shi = float(scale_series.quantile(0.90))
        param_mismatch = cand_scale < slo or cand_scale > shi

    # Base similarity is used both for confidence and out-of-sample diagnostics.
    base_sim = float(cohort["similarity_score"].mean()) if "similarity_score" in cohort.columns else 0.0
    top_support = cohort.head(min(len(cohort), max(5, min(int(top_k), 10))))
    top_similarity_mean = (
        float(pd.to_numeric(top_support.get("similarity_score"), errors="coerce").fillna(0.0).mean())
        if not top_support.empty
        else 0.0
    )
    top_similarity_p25 = (
        float(pd.to_numeric(top_support.get("similarity_score"), errors="coerce").fillna(0.0).quantile(0.25))
        if not top_support.empty
        else 0.0
    )
    top_action_match_score = (
        float(pd.to_numeric(top_support.get("action_match_score"), errors="coerce").fillna(0.0).mean())
        if not top_support.empty
        else 0.0
    )
    top_weighted_feature_coverage = (
        float(pd.to_numeric(top_support.get("weighted_feature_coverage"), errors="coerce").fillna(0.0).mean())
        if not top_support.empty
        else 0.0
    )
    top_critical_feature_coverage = (
        float(pd.to_numeric(top_support.get("critical_feature_coverage"), errors="coerce").fillna(0.0).mean())
        if not top_support.empty
        else 0.0
    )
    top_state_distance = (
        float(pd.to_numeric(top_support.get("state_distance"), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().mean())
        if not top_support.empty
        else 0.0
    )
    top_rate_gap = (
        float(pd.to_numeric(top_support.get("rate_gap"), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().mean())
        if not top_support.empty
        else 0.0
    )
    top_credit_gap = (
        float(pd.to_numeric(top_support.get("credit_gap"), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().mean())
        if not top_support.empty
        else 0.0
    )
    top_missing_penalty_factor = (
        float(pd.to_numeric(top_support.get("missing_penalty_factor"), errors="coerce").fillna(1.0).mean())
        if not top_support.empty
        else 1.0
    )
    top_regime_penalty_factor = (
        float(pd.to_numeric(top_support.get("regime_penalty_factor"), errors="coerce").fillna(1.0).mean())
        if not top_support.empty
        else 1.0
    )
    top_sector_penalty_factor = (
        float(pd.to_numeric(top_support.get("sector_penalty_factor"), errors="coerce").fillna(1.0).mean())
        if not top_support.empty
        else 1.0
    )
    narrative_mismatch = bool(narrative_real_count >= 5 and narrative_top5_mean < 0.12)
    very_low_similarity = max(base_sim, top_similarity_mean) < 0.42
    has_divestiture_family_scale = any(
        str(k).startswith("portfolio.divestiture.") for k in selected_family_scale_keys
    )
    has_mna_family_scale = any(str(k).startswith("mna.") for k in selected_family_scale_keys)
    has_debt_family_scale = any(
        str(k).startswith("capital_structure.debt_") or str(k).startswith("capital_structure.revolver.")
        for k in selected_family_scale_keys
    )
    has_debt_bond_amount_family_scale = any(
        str(k).startswith("capital_structure.debt_bond.amount_") for k in selected_family_scale_keys
    )
    strong_family_scale_match = (
        retrieval_tier == "family"
        and bool(selected_family_scale_keys)
        and not regime_mismatch
        and (
            (
                not param_mismatch
                and top_action_match_score >= 0.90
                and top_similarity_mean >= 0.65
            )
            or (
                action_id_text == "mna.transformational_acquisition"
                and has_mna_family_scale
                and top_action_match_score >= 0.88
                and top_similarity_mean >= 0.60
            )
            or (
                action_id_text == "mna.go_private_lbo"
                and any(str(k).startswith("mna.platform_lbo.") for k in selected_family_scale_keys)
                and top_action_match_score >= 0.88
                and top_similarity_mean >= 0.60
            )
            or (
                has_divestiture_family_scale
                and top_action_match_score >= 0.85
                and top_similarity_mean >= 0.63
            )
            or (
                has_debt_family_scale
                and top_action_match_score >= 0.82
                and top_similarity_mean >= 0.60
            )
            or (
                action_id_text == "capital_structure.new_debt_issuance"
                and has_debt_bond_amount_family_scale
                and top_action_match_score >= 0.90
                and top_similarity_mean >= 0.58
            )
        )
    )
    strong_exact_match = (
        retrieval_tier == "exact"
        and not regime_mismatch
        and int(exact_idx.shape[0]) >= int(exact_support_min)
        and top_action_match_score >= 0.80
        and top_similarity_mean >= 0.72
    )

    mismatch = MismatchDiagnostics(
        feature_mismatches=mismatches,
        regime_mismatch=regime_mismatch,
        parameter_scale_mismatch=param_mismatch,
        narrative_mismatch=narrative_mismatch,
        # Conservative but not over-triggered:
        # - global retrieval with low exact coverage
        # - regime mismatch
        # - materially outlying feature profile (3+ mismatches), except for
        #   strong family-scale matches where the action/scale neighborhood is
        #   highly aligned and the mismatches are feature-level rather than
        #   regime or parameter-scale failures.
        # - very low similarity even after ranking
        out_of_sample_flag=(
            (low_precedent_coverage and retrieval_tier == "global")
            or regime_mismatch
            or (len(mismatches) >= 3 and not (strong_family_scale_match or strong_exact_match))
            or very_low_similarity
            or (retrieval_tier != "exact" and top_similarity_mean < 0.50 and top_action_match_score < 0.75)
        ),
    )

    # Calibration confidence
    confidence_meta = _compute_calibration_confidence(
        retrieval_tier=retrieval_tier,
        exact_match_count=int(exact_idx.shape[0]),
        exact_support_min=int(exact_support_min),
        cohort_size=int(len(cohort)),
        base_similarity=float(base_sim),
        top_similarity_mean=float(top_similarity_mean),
        top_similarity_p25=float(top_similarity_p25),
        top_action_match_score=float(top_action_match_score),
        mismatch_count=int(len(mismatches)),
        regime_mismatch=bool(regime_mismatch),
        parameter_mismatch=bool(param_mismatch),
        narrative_mismatch=bool(narrative_mismatch),
    )
    calibration_confidence = float(confidence_meta["calibration_confidence"])
    profile["pack_finalize_seconds"] = round(time.perf_counter() - t_pack, 6)
    profile["cohort_size"] = int(len(cohort))
    profile["narrative_real_count"] = int(narrative_real_count)
    profile["strong_family_scale_match"] = bool(strong_family_scale_match)
    profile["strong_exact_match"] = bool(strong_exact_match)
    profile["total_seconds"] = round(time.perf_counter() - t_start, 6)
    _debug("pack_finalize:done", total_seconds=float(profile["total_seconds"]))

    return PrecedentPack(
        candidate_id=candidate_id,
        run_id=run_id,
        retrieved_cohorts=retrieved,
        similarity_scores=sim_scores,
        outcome_distributions=out_dists,
        regime_splits=regime_splits,
        tail_events=tail_events,
        second_order_effects=second_order,
        mismatch_diagnostics={
            **mismatch.to_dict(),
            "low_precedent_coverage": bool(low_precedent_coverage),
            "exact_match_count": int(exact_idx.shape[0]),
            "cohort_size": int(len(cohort)),
            "minimum_cohort_size": int(min_k),
            "minimum_exact_support": int(exact_support_min),
            "retrieval_tier": str(retrieval_tier),
            "exact_support_ratio": float(confidence_meta["exact_support_ratio"]),
            "cohort_factor": float(confidence_meta["cohort_factor"]),
            "cohort_support_factor": float(confidence_meta["support_factor"]),
            "similarity_signal": float(confidence_meta["similarity_signal"]),
            "tier_confidence_discount": float(confidence_meta["tier_conf_discount"]),
            "confidence_pre_tier_discount": float(confidence_meta["confidence_pre_tier_discount"]),
            "hard_prefilter_applied": bool(hard_prefilter_applied),
            "hard_prefilter_relaxed": bool(hard_prefilter_relaxed),
            "regime_prefilter_applied": bool(regime_prefilter_applied),
            "sector_prefilter_applied": bool(sector_prefilter_applied),
            "market_cap_prefilter_applied": bool(market_cap_prefilter_applied),
            "candidate_market_cap_bucket": int(cand_market_cap_bucket),
            "top_similarity_mean": float(top_similarity_mean),
            "top_similarity_p25": float(top_similarity_p25),
            "top_action_match_score": float(top_action_match_score),
            "narrative_top5_similarity": float(narrative_top5_mean),
            "narrative_real_text_rows": int(narrative_real_count),
            "strong_family_scale_match": bool(strong_family_scale_match),
            "strong_exact_match": bool(strong_exact_match),
            "state_distance_version": str(weighted_state.get("version") or _WEIGHTED_DISTANCE_V1_VERSION),
            "weighted_coverage_gate_applied": bool(weighted_coverage_gate_applied),
            "weighted_coverage_gate_relaxed": bool(weighted_coverage_gate_relaxed),
            "size_guardrail_applied": bool(size_guardrail_applied),
            "size_guardrail_relaxed": bool(size_guardrail_relaxed),
            "identity_prefilter_applied": bool(identity_prefilter_applied),
            "identity_prefilter_mode": str(identity_prefilter_mode or ""),
            "top_weighted_feature_coverage": float(top_weighted_feature_coverage),
            "top_critical_feature_coverage": float(top_critical_feature_coverage),
            "top_state_distance": float(top_state_distance),
            "top_rate_gap": float(top_rate_gap),
            "top_credit_gap": float(top_credit_gap),
            "top_missing_penalty_factor": float(top_missing_penalty_factor),
            "top_regime_penalty_factor": float(top_regime_penalty_factor),
            "top_sector_penalty_factor": float(top_sector_penalty_factor),
            "state_weight_scope": str(weighted_state.get("profile", {}).get("weight_scope") or "prior_only"),
            "learned_holdout_pair_correlation": _to_float(
                weighted_state.get("profile", {}).get("learned_holdout_pair_correlation"),
                None,
            ),
            "learned_prior_holdout_pair_correlation": _to_float(
                weighted_state.get("profile", {}).get("learned_prior_holdout_pair_correlation"),
                None,
            ),
        },
        calibration_confidence=calibration_confidence,
        profiling=profile,
        # legacy compatibility payloads
        matches=[asdict(c) for c in retrieved],
        distributions=legacy_distributions,
    )


__all__ = [
    "PrecedentRetrievalIndex",
    "augment_precedent_state_vector_columns",
    "build_precedent_pack_v2",
    "build_precedent_retrieval_index",
]
