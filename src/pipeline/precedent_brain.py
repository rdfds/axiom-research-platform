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


