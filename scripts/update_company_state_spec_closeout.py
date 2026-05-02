#!/usr/bin/env python
"""
Fast post-processing updater to close remaining spec gaps without re-running
full snapshot computation.

Adds/refreshes:
  - market.ev_ebitda_vs_peer_z
  - market.fcf_yield_percentile_peers
  - operating.guidance_revision_direction
  - operating.cyclicality_macro_beta_proxy
  - operating.revenue_sensitivity_proxy
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        x = float(v)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None


def _feature_value(snapshot: dict, name: str) -> Any:
    return snapshot.get("features", {}).get(name, {}).get("value")


def _feature_record(
    name: str,
    value: Any,
    unit: str,
    as_of_time: str,
    confidence: Optional[float],
    provenance: List[dict],
    missing_reason: Optional[str],
    window: Optional[Dict[str, Any]] = None,
    fallback_used: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "computed_at": _now_iso(),
        "as_of_time": as_of_time,
        "window": window,
        "confidence": confidence,
        "provenance": provenance,
        "missing_reason": missing_reason,
        "fallback_used": fallback_used,
    }


def _reference(
    artifact_type: str,
    artifact_id: str,
    source: Optional[str],
    published_at: Optional[str] = None,
    ingested_at: Optional[str] = None,
) -> dict:
    return {
        "artifact_type": artifact_type,
        "artifact_id": artifact_id,
        "source": source,
        "published_at": published_at,
        "ingested_at": ingested_at,
        "hash": None,
    }


def _load_snapshots(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _write_snapshots(path: Path, snapshots: List[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for s in snapshots:
            f.write(json.dumps(s) + "\n")
    tmp.replace(path)


def _choose_sector_col(df: pd.DataFrame) -> Optional[str]:
    for c in [
        "gics_sector",
        "sector",
        "industry",
        "gics_industry",
        "industry_group",
        "sic",
        "naics",
    ]:
        if c in df.columns:
            return c
    return None


def _build_sector_peers(entity_table_path: Optional[Path]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    if entity_table_path is None or not entity_table_path.exists():
        return {}, {}
    try:
        ent = pd.read_parquet(entity_table_path)
    except Exception:
        return {}, {}
    if "entity_id" not in ent.columns:
        return {}, {}
    col = _choose_sector_col(ent)
    if col is None:
        return {}, {}
    ent = ent[["entity_id", col]].copy()
    ent["entity_id"] = ent["entity_id"].astype(str)
    ent[col] = ent[col].astype(str)
    ent = ent.dropna(subset=[col])
    entity_to_sector: Dict[str, str] = {}
    sector_to_entities: Dict[str, List[str]] = {}
    for _, row in ent.iterrows():
        cid = str(row["entity_id"])
        sec = str(row[col])
        entity_to_sector[cid] = sec
        sector_to_entities.setdefault(sec, []).append(cid)
    return entity_to_sector, sector_to_entities


def _peer_ids_for_snapshot(
    snapshot: dict,
    entity_to_sector: Dict[str, str],
    sector_to_entities: Dict[str, List[str]],
) -> List[str]:
    cid = str(snapshot.get("company_id"))
    members = snapshot.get("peer_set", {}).get("members", []) or []
    members = [str(x) for x in members if x is not None]
    if members:
        return list(dict.fromkeys(members))
    sec = entity_to_sector.get(cid)
    if sec is None:
        return []
    return [x for x in sector_to_entities.get(sec, []) if x != cid]


def _zscore(target: float, values: List[float]) -> Optional[float]:
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) < 3:
        return None
    std = float(arr.std(ddof=0))
    if std == 0 or not np.isfinite(std):
        return None
    return float((target - float(arr.mean())) / std)


def _percentile(target: float, values: List[float]) -> Optional[float]:
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) < 3:
        return None
    return float((arr <= target).sum() / len(arr) * 100.0)


def _load_guidance_scores(facts_path: Optional[Path], asof: str) -> Tuple[Dict[str, float], Dict[str, dict]]:
    if facts_path is None or not facts_path.exists():
        return {}, {}
    con = duckdb.connect()
    if facts_path.is_dir():
        source = f"read_parquet('{facts_path.as_posix()}/year=*/part.parquet', union_by_name=True)"
    else:
        source = f"read_parquet('{facts_path.as_posix()}', union_by_name=True)"

    cutoff = pd.to_datetime(asof).strftime("%Y-%m-%d %H:%M:%S")
    query = f"""
    SELECT
      CAST(entity_id AS VARCHAR) AS entity_id,
      CAST(fact_id AS VARCHAR) AS fact_id,
      CAST(source_type AS VARCHAR) AS source_type,
      CAST(fact_type AS VARCHAR) AS fact_type,
      CAST(context_norm AS VARCHAR) AS context_norm,
      try_cast(published_at AS TIMESTAMP) AS published_at,
      try_cast(ingested_at AS TIMESTAMP) AS ingested_at
    FROM {source}
    WHERE (published_at IS NULL OR try_cast(published_at AS TIMESTAMP) <= TIMESTAMP '{cutoff}')
      AND (ingested_at IS NULL OR try_cast(ingested_at AS TIMESTAMP) <= TIMESTAMP '{cutoff}')
      AND (
        lower(coalesce(fact_type, '')) LIKE '%guidance%'
        OR lower(coalesce(fact_type, '')) LIKE '%revision%'
        OR lower(coalesce(context_norm, '')) LIKE '%guidance%'
      )
    """
    try:
        df = con.execute(query).df()
    except Exception:
        return {}, {}
    if df.empty:
        return {}, {}

    asof_ts = pd.to_datetime(asof, utc=True)
    pos = ["raise", "raised", "increase", "upward", "beat", "above", "improv", "stronger", "reaffirm"]
    neg = ["lower", "lowered", "decrease", "downward", "below", "miss", "cut", "weaker", "reduce"]

    scores: Dict[str, float] = {}
    provenance: Dict[str, dict] = {}
    for entity_id, g in df.groupby("entity_id"):
        num = 0.0
        den = 0.0
        g = g.sort_values("published_at", ascending=False)
        for _, row in g.iterrows():
            txt = f"{row.get('fact_type') or ''} {row.get('context_norm') or ''}".lower()
            s = 0.0
            for k in pos:
                if k in txt:
                    s += 1.0
            for k in neg:
                if k in txt:
                    s -= 1.0
            if s == 0:
                continue
            pub = pd.to_datetime(row.get("published_at"), utc=True, errors="coerce")
            if pd.isna(pub):
                w = 0.75
            else:
                days = max(0.0, float((asof_ts - pub).days))
                w = float(np.exp(-days / 365.0))
            num += s * w
            den += w
        if den > 0:
            val = float(np.clip(num / den, -1.0, 1.0))
            scores[str(entity_id)] = val
            r = g.iloc[0]
            provenance[str(entity_id)] = _reference(
                artifact_type="ExtractedFact",
                artifact_id=str(r.get("fact_id") or f"guidance:{entity_id}"),
                source=str(r.get("source_type") or "facts"),
                published_at=str(r.get("published_at")) if r.get("published_at") is not None else None,
                ingested_at=str(r.get("ingested_at")) if r.get("ingested_at") is not None else None,
            )
    return scores, provenance


def _collect_feature_refs(snapshot: dict, keys: List[str], limit: int = 5) -> List[dict]:
    out: List[dict] = []
    seen = set()
    for k in keys:
        p = snapshot.get("features", {}).get(k, {}).get("provenance", []) or []
        for ref in p:
            aid = ref.get("artifact_id")
            at = ref.get("artifact_type")
            token = (at, aid)
            if token in seen:
                continue
            seen.add(token)
            out.append(ref)
            if len(out) >= limit:
                return out
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Close remaining CompanyState spec gaps without full rebuild.")
    parser.add_argument("--in-path", required=True)
    parser.add_argument("--out-path", required=False, default=None)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--facts-path", default=None, help="Optional facts path for guidance direction extraction.")
    parser.add_argument("--entity-table-path", default=None, help="Optional entity table for sector fallback peers.")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path) if args.out_path else in_path
    facts_path = Path(args.facts_path) if args.facts_path else None
    entity_table_path = Path(args.entity_table_path) if args.entity_table_path else None

    snapshots = _load_snapshots(in_path)
    if not snapshots:
        raise SystemExit("No snapshots found.")

    entity_to_sector, sector_to_entities = _build_sector_peers(entity_table_path)
    guidance_scores, guidance_ref = _load_guidance_scores(facts_path, args.asof)

    ev_map: Dict[str, Optional[float]] = {}
    fcf_yield_map: Dict[str, Optional[float]] = {}
    for s in snapshots:
        cid = str(s.get("company_id"))
        ev_map[cid] = _safe_float(_feature_value(s, "market.ev_ebitda"))
        fcf_yield_map[cid] = _safe_float(_feature_value(s, "market.fcf_yield"))

    updated = 0
    for s in snapshots:
        cid = str(s.get("company_id"))
        as_of_time = s.get("as_of_time", pd.to_datetime(args.asof, utc=True).isoformat())
        features = s.setdefault("features", {})

        # 1) Peer-relative market features.
        peers = _peer_ids_for_snapshot(s, entity_to_sector, sector_to_entities)
        peer_universe = [p for p in peers if p in ev_map] + [cid]
        peer_ev = [ev_map.get(p) for p in peer_universe if ev_map.get(p) is not None]
        own_ev = ev_map.get(cid)
        ev_z = _zscore(float(own_ev), [float(x) for x in peer_ev]) if own_ev is not None else None
        peer_fcf = [fcf_yield_map.get(p) for p in peer_universe if fcf_yield_map.get(p) is not None]
        own_fcf = fcf_yield_map.get(cid)
        fcf_pct = _percentile(float(own_fcf), [float(x) for x in peer_fcf]) if own_fcf is not None else None
        peer_ref = _reference(
            artifact_type="RawDocument",
            artifact_id=str(s.get("peer_set", {}).get("peer_set_id") or f"peer_set:{cid}"),
            source="peer_set",
        )

        features["market.ev_ebitda_vs_peer_z"] = _feature_record(
            name="market.ev_ebitda_vs_peer_z",
            value=ev_z,
            unit="zscore",
            as_of_time=as_of_time,
            window={"type": "asof", "length_days": 0},
            confidence=0.65 if ev_z is not None else None,
            provenance=[peer_ref],
            missing_reason="unavailable" if ev_z is None else None,
            fallback_used=None,
        )
        features["market.fcf_yield_percentile_peers"] = _feature_record(
            name="market.fcf_yield_percentile_peers",
            value=fcf_pct,
            unit="percentile",
            as_of_time=as_of_time,
            window={"type": "asof", "length_days": 0},
            confidence=0.65 if fcf_pct is not None else None,
            provenance=[peer_ref],
            missing_reason="unavailable" if fcf_pct is None else None,
            fallback_used=None,
        )

        # 2) Guidance revision direction from facts.
        g = guidance_scores.get(cid)
        g_ref = guidance_ref.get(cid)
        features["operating.guidance_revision_direction"] = _feature_record(
            name="operating.guidance_revision_direction",
            value=g,
            unit="score_-1_to_1",
            as_of_time=as_of_time,
            window={"type": "lookback", "length_days": 365},
            confidence=(0.55 + 0.35 * abs(float(g))) if g is not None else None,
            provenance=[g_ref] if g_ref is not None else [],
            missing_reason="unavailable" if g is None else None,
            fallback_used="heuristic" if g is not None else None,
        )

        # 3) Cyclicality proxies from existing snapshot features (fast heuristic).
        vol90 = _safe_float(_feature_value(s, "market.volatility_90d"))
        dd90 = _safe_float(_feature_value(s, "market.drawdown_90d"))
        lev = _safe_float(_feature_value(s, "capital_structure.net_leverage"))
        margin_vol = _safe_float(_feature_value(s, "operating.margin_volatility_8q"))
        rev_yoy = _safe_float(_feature_value(s, "operating.revenue_yoy_last_q"))

        cyc_beta = None
        if any(v is not None for v in [vol90, dd90, lev]):
            vol_term = min(1.0, (vol90 or 0.0) / 0.45)
            dd_term = min(1.0, abs(dd90 or 0.0) / 0.5)
            lev_term = min(1.0, max(0.0, (lev or 0.0)) / 6.0)
            cyc_beta = float(np.clip(0.5 * vol_term + 0.3 * dd_term + 0.2 * lev_term, 0.0, 1.0))

        rev_sens = None
        if any(v is not None for v in [margin_vol, rev_yoy]):
            mv_term = min(1.0, abs(margin_vol or 0.0) / 0.20)
            ry_term = min(1.0, abs(rev_yoy or 0.0) / 0.30)
            rev_sens = float(np.clip(0.55 * ry_term + 0.45 * mv_term, 0.0, 1.0))

        cyc_refs = _collect_feature_refs(
            s,
            [
                "market.volatility_90d",
                "market.drawdown_90d",
                "capital_structure.net_leverage",
                "operating.margin_volatility_8q",
                "operating.revenue_yoy_last_q",
            ],
            limit=5,
        )
        features["operating.cyclicality_macro_beta_proxy"] = _feature_record(
            name="operating.cyclicality_macro_beta_proxy",
            value=cyc_beta,
            unit="score_0_1",
            as_of_time=as_of_time,
            window={"type": "lookback", "length_days": 365},
            confidence=0.55 if cyc_beta is not None else None,
            provenance=cyc_refs,
            missing_reason="unavailable" if cyc_beta is None else None,
            fallback_used="heuristic" if cyc_beta is not None else None,
        )
        features["operating.revenue_sensitivity_proxy"] = _feature_record(
            name="operating.revenue_sensitivity_proxy",
            value=rev_sens,
            unit="score_0_1",
            as_of_time=as_of_time,
            window={"type": "lookback", "length_days": 365},
            confidence=0.55 if rev_sens is not None else None,
            provenance=cyc_refs,
            missing_reason="unavailable" if rev_sens is None else None,
            fallback_used="heuristic" if rev_sens is not None else None,
        )

        # Refresh missing-data flags to include new features.
        s.setdefault("provenance", {})
        s["provenance"].setdefault("computation_version", "state_builder_v3")
        s["provenance"]["computation_version"] = "state_builder_v4_closeout"
        s["provenance"]["missing_data_flags"] = {
            k: v.get("missing_reason")
            for k, v in features.items()
            if isinstance(v, dict) and v.get("missing_reason") is not None
        }

        updated += 1

    _write_snapshots(out_path, snapshots)
    print(f"Wrote updated snapshots -> {out_path} rows={updated}")


if __name__ == "__main__":
    main()

