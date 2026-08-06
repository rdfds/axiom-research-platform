"""
CompanyState components: RegimeClassifier, PeerSetResolver, ProvenanceTracker.
These are lightweight, auditable building blocks used by CompanyStateBuilder.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class InputReference:
    artifact_type: str
    artifact_id: str
    source: Optional[str] = None
    published_at: Optional[str] = None
    ingested_at: Optional[str] = None
    hash: Optional[str] = None


def _zscore(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    s = series.dropna().astype(float)
    if len(s) < 10:
        return None
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0:
        return None
    return float((s.iloc[-1] - mu) / sd)


def _percentile(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    s = series.dropna().astype(float)
    if len(s) < 10:
        return None
    return float((s.rank(pct=True).iloc[-1]) * 100.0)


def _pick_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _pick_time_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_first_col(
        df,
        [
            "observation_time",
            "event_time",
            "trade_date",
            "effective_at",
            "published_at",
            "available_time",
            "ingestion_time",
            "date",
            "as_of_date",
            "timestamp",
        ],
    )


def _pick_value_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_first_col(
        df,
        [
            "value",
            "close",
            "adjusted_close",
            "consensus_value",
            "fact_value",
            "numeric_value",
            "amount",
        ],
    )


def _pick_series_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_first_col(df, ["series_id", "instrument_id", "metric", "field_name"])


class RegimeClassifier:
    def __init__(self, series_map: Optional[Dict[str, str]] = None) -> None:
        self.series_map = series_map or {
            "hy_oas": "BAMLH0A0HYM2",
            "ig_oas": "BAMLC0A0CM",
            "vix": "VIXCLS",
            "sp500": "SP500",
            "rate_2y": "DGS2",
            "rate_10y": "DGS10",
        }

    def classify(self, macro: pd.DataFrame, as_of: pd.Timestamp) -> Dict[str, Any]:
        if macro is None or macro.empty:
            return {
                "credit_regime": "neutral",
                "risk_regime": "neutral",
                "vol_regime": "normal",
                "sector_cycle": "neutral",
                "signals": {},
                "confidence": 0.3,
            }

        time_col = _pick_time_col(macro)
        value_col = _pick_value_col(macro)
        series_col = _pick_series_col(macro)
        if time_col is None or value_col is None or series_col is None:
            return {
                "credit_regime": "neutral",
                "risk_regime": "neutral",
                "vol_regime": "normal",
                "sector_cycle": "neutral",
                "signals": {},
                "confidence": 0.3,
            }

        def series_window(series_id: str) -> pd.Series:
            df = macro[macro[series_col].astype(str) == series_id].copy()
            if df.empty:
                return pd.Series(dtype=float)
            df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
            df = df[df[time_col] <= as_of]
            df = df.sort_values(time_col)
            if df.empty:
                return pd.Series(dtype=float)
            lookback_start = as_of - timedelta(days=365 * 5)
            df = df[df[time_col] >= lookback_start]
            return df[value_col].astype(float)

        hy = series_window(self.series_map["hy_oas"])
        vix = series_window(self.series_map["vix"])
        spx = series_window(self.series_map["sp500"])

        hy_z = _zscore(hy)
        vix_pct = _percentile(vix)
        spx_ret_6m = None
        if not spx.empty and len(spx) > 2:
            idx = max(0, len(spx) - 126)
            if spx.iloc[idx] != 0:
                spx_ret_6m = (spx.iloc[-1] / spx.iloc[idx]) - 1.0

        credit_regime = "neutral"
        if hy_z is not None:
            if hy_z > 1.0:
                credit_regime = "tight"
            elif hy_z < -1.0:
                credit_regime = "loose"

        risk_regime = "neutral"
        if vix_pct is not None:
            if vix_pct > 75:
                risk_regime = "risk_off"
            elif vix_pct < 25:
                risk_regime = "risk_on"

        vol_regime = "normal"
        if vix_pct is not None:
            if vix_pct > 75:
                vol_regime = "high"
            elif vix_pct < 25:
                vol_regime = "low"

        sector_cycle = "neutral"
        if spx_ret_6m is not None:
            if spx_ret_6m > 0.1:
                sector_cycle = "upcycle"
            elif spx_ret_6m < -0.1:
                sector_cycle = "downcycle"

        return {
            "credit_regime": credit_regime,
            "risk_regime": risk_regime,
            "vol_regime": vol_regime,
            "sector_cycle": sector_cycle,
            "signals": {
                "hy_oas_z": hy_z,
                "vix_pctile": vix_pct,
                "spx_6m_ret": spx_ret_6m,
            },
            "confidence": 0.7 if hy_z is not None and vix_pct is not None else 0.4,
        }


class PeerSetResolver:
    def resolve(self, company_id: str, entity_table: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        if entity_table is None or entity_table.empty:
            return {
                "peer_set_id": str(company_id),
                "members": [],
                "method": "unresolved",
                "version": 1,
            }

        # Try to find sector/industry columns
        sector_col = None
        for c in [
            "gics_sector",
            "sector",
            "industry",
            "gics_industry",
            "industry_group",
            "sic",
            "naics",
        ]:
            if c in entity_table.columns:
                sector_col = c
                break

        if sector_col is None or "entity_id" not in entity_table.columns:
            return {
                "peer_set_id": str(company_id),
                "members": [],
                "method": "unresolved",
                "version": 1,
            }

        df = entity_table.copy()
        df["entity_id"] = df["entity_id"].astype(str)
        row = df[df["entity_id"] == str(company_id)]
        if row.empty:
            return {
                "peer_set_id": str(company_id),
                "members": [],
                "method": "unresolved",
                "version": 1,
            }

        sector_val = row.iloc[0].get(sector_col)
        if pd.isna(sector_val):
            return {
                "peer_set_id": str(company_id),
                "members": [],
                "method": "sector_unknown",
                "version": 1,
            }

        peers = df[df[sector_col] == sector_val].copy()

        # Optional size banding if market cap exists
        size_col = None
        for c in ["market_cap", "mkt_cap", "marketcap", "mktcap", "market_capitalization"]:
            if c in peers.columns:
                size_col = c
                break

        method = f"{sector_col}"
        if size_col is not None and not peers[size_col].isna().all():
            peers[size_col] = pd.to_numeric(peers[size_col], errors="coerce")
            # Compute quartile bands within the sector
            qs = peers[size_col].quantile([0.25, 0.5, 0.75]).to_dict()
            target_size = pd.to_numeric(row.iloc[0].get(size_col), errors="coerce")
            if pd.notna(target_size):
                if target_size <= qs.get(0.25, target_size):
                    band = "q1"
                elif target_size <= qs.get(0.5, target_size):
                    band = "q2"
                elif target_size <= qs.get(0.75, target_size):
                    band = "q3"
                else:
                    band = "q4"
                # Filter peers to same size band
                if band == "q1":
                    peers = peers[peers[size_col] <= qs.get(0.25, target_size)]
                elif band == "q2":
                    peers = peers[(peers[size_col] > qs.get(0.25, target_size)) & (peers[size_col] <= qs.get(0.5, target_size))]
                elif band == "q3":
                    peers = peers[(peers[size_col] > qs.get(0.5, target_size)) & (peers[size_col] <= qs.get(0.75, target_size))]
                else:
                    peers = peers[peers[size_col] > qs.get(0.75, target_size)]
                method = f"{sector_col}+size_band_{band}"

        members = peers["entity_id"].astype(str).tolist()
        members = [m for m in members if m != str(company_id)]

        # Cap peers to a reasonable number for stability
        if len(members) > 50:
            members = members[:50]

        return {
            "peer_set_id": f"{sector_col}:{sector_val}",
            "members": members,
            "method": method,
            "version": 1,
        }


class ProvenanceTracker:
    def build_reference(
        self,
        artifact_type: str,
        artifact_id: str,
        source: Optional[str] = None,
        published_at: Optional[str] = None,
        ingested_at: Optional[str] = None,
        hash_value: Optional[str] = None,
    ) -> InputReference:
        return InputReference(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            source=source,
            published_at=published_at,
            ingested_at=ingested_at,
            hash=hash_value,
        )
