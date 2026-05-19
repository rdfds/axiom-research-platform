#!/usr/bin/env python
"""
Fast delta updater for ownership concentration + issuer rating features.

Use this to enrich an existing CompanyState snapshot JSONL without rerunning
full feature computation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb
import numpy as np


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sql_quote(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def _asof_ts(asof: str) -> str:
    return str(np.datetime64(asof))


def _feature_record(
    name: str,
    value: Any,
    unit: str,
    as_of_time: str,
    confidence: Optional[float],
    provenance: list,
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


def _facts_source_sql(path: Path) -> str:
    if path.is_dir():
        return f"read_parquet('{path.as_posix()}/year=*/part.parquet', union_by_name=True)"
    return f"read_parquet('{path.as_posix()}', union_by_name=True)"


def _load_ownership_map(
    ownership_path: Path,
    facts_path: Optional[Path],
    asof: str,
) -> Dict[str, Dict[str, Any]]:
    if not ownership_path.exists():
        return {}
    con = duckdb.connect()
    asof_ts = _asof_ts(asof)
    base_own = f"""
    WITH own AS (
      SELECT
        CAST(company_id AS VARCHAR) AS company_id,
        try_cast(total_13f_shares AS DOUBLE) AS total_13f_shares,
        try_cast(top5_13f_shares AS DOUBLE) AS top5_13f_shares,
        CAST(source_type AS VARCHAR) AS source_type,
        CAST(artifact_id AS VARCHAR) AS artifact_id,
        try_cast(published_at AS TIMESTAMP) AS published_at,
        try_cast(ingested_at AS TIMESTAMP) AS ingested_at,
        try_cast(effective_at AS TIMESTAMP) AS effective_at,
        try_cast(report_date AS TIMESTAMP) AS report_date,
        try_cast(filing_date AS TIMESTAMP) AS filing_date,
        row_number() OVER (
          PARTITION BY CAST(company_id AS VARCHAR)
          ORDER BY
            coalesce(try_cast(report_date AS TIMESTAMP), try_cast(effective_at AS TIMESTAMP)) DESC NULLS LAST,
            coalesce(try_cast(filing_date AS TIMESTAMP), try_cast(published_at AS TIMESTAMP), try_cast(ingested_at AS TIMESTAMP)) DESC NULLS LAST
        ) AS rn
      FROM read_parquet('{ownership_path.as_posix()}', union_by_name=True)
      WHERE (published_at IS NULL OR try_cast(published_at AS TIMESTAMP) <= TIMESTAMP '{asof_ts}')
        AND (ingested_at IS NULL OR try_cast(ingested_at AS TIMESTAMP) <= TIMESTAMP '{asof_ts}')
        AND (effective_at IS NULL OR try_cast(effective_at AS TIMESTAMP) <= TIMESTAMP '{asof_ts}')
    )
    """
    query = None
    if facts_path is not None and facts_path.exists():
        facts_source = _facts_source_sql(facts_path)
        shares_types = [
            "financial.shares_basic",
            "financial.shares_outstanding",
            "financial.shares_out",
            "shares_outstanding",
            "financial.shares_diluted",
            "financial.diluted_shares_outstanding",
            "diluted_shares_outstanding",
        ]
        shares_in = ", ".join(_sql_quote(x) for x in shares_types)
        query = base_own + f"""
    ,
    shares AS (
      SELECT
        CAST(entity_id AS VARCHAR) AS company_id,
        try_cast(fact_value AS DOUBLE) AS shares_out,
        row_number() OVER (
          PARTITION BY CAST(entity_id AS VARCHAR)
          ORDER BY coalesce(try_cast(published_at AS TIMESTAMP), try_cast(effective_at AS TIMESTAMP), try_cast(ingested_at AS TIMESTAMP)) DESC NULLS LAST
        ) AS rn
      FROM {facts_source}
      WHERE CAST(fact_type AS VARCHAR) IN ({shares_in})
        AND (published_at IS NULL OR try_cast(published_at AS TIMESTAMP) <= TIMESTAMP '{asof_ts}')
        AND (ingested_at IS NULL OR try_cast(ingested_at AS TIMESTAMP) <= TIMESTAMP '{asof_ts}')
    )
    SELECT
      o.company_id,
      o.total_13f_shares,
      o.top5_13f_shares,
      s.shares_out,
      o.source_type,
      o.artifact_id,
      o.published_at,
      o.ingested_at
    FROM own o
    LEFT JOIN shares s
      ON o.company_id = s.company_id AND s.rn = 1
    WHERE o.rn = 1
    """
    else:
        query = base_own + """
    SELECT
      o.company_id,
      o.total_13f_shares,
      o.top5_13f_shares,
      CAST(NULL AS DOUBLE) AS shares_out,
      o.source_type,
      o.artifact_id,
      o.published_at,
      o.ingested_at
    FROM own o
    WHERE o.rn = 1
    """
    df = con.execute(query).df()
    out: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        cid = str(row.get("company_id"))
        total = row.get("total_13f_shares")
        top5 = row.get("top5_13f_shares")
        shares_out = row.get("shares_out")
        top5_pct = None
        inst_pct = None
        if total is not None and not np.isnan(total) and total > 0 and top5 is not None and not np.isnan(top5):
            top5_pct = float(np.clip(float(top5) / float(total), 0.0, 1.0))
        if total is not None and not np.isnan(total) and shares_out is not None and not np.isnan(shares_out) and shares_out > 0:
            inst_pct = float(np.clip(float(total) / float(shares_out), 0.0, 2.0))
        out[cid] = {
            "top5_pct": top5_pct,
            "inst_pct": inst_pct,
            "source_type": row.get("source_type"),
            "artifact_id": row.get("artifact_id"),
            "published_at": row.get("published_at"),
            "ingested_at": row.get("ingested_at"),
        }
    return out


def _load_rating_map(ratings_path: Path, asof: str) -> Dict[str, Dict[str, Any]]:
    if not ratings_path.exists():
        return {}
    con = duckdb.connect()
    asof_ts = _asof_ts(asof)
    query = f"""
    WITH r AS (
      SELECT
        CAST(company_id AS VARCHAR) AS company_id,
        CAST(rating_symbol AS VARCHAR) AS rating_symbol,
        CAST(current_rating_symbol AS VARCHAR) AS current_rating_symbol,
        CAST(outlook AS VARCHAR) AS outlook,
        CAST(creditwatch AS VARCHAR) AS creditwatch,
        CAST(source_type AS VARCHAR) AS source_type,
        CAST(artifact_id AS VARCHAR) AS artifact_id,
        try_cast(rating_date AS TIMESTAMP) AS rating_date,
        try_cast(published_at AS TIMESTAMP) AS published_at,
        try_cast(ingested_at AS TIMESTAMP) AS ingested_at,
        try_cast(effective_at AS TIMESTAMP) AS effective_at,
        row_number() OVER (
          PARTITION BY CAST(company_id AS VARCHAR)
          ORDER BY
            coalesce(try_cast(rating_date AS TIMESTAMP), try_cast(effective_at AS TIMESTAMP), try_cast(published_at AS TIMESTAMP)) DESC NULLS LAST,
            coalesce(try_cast(published_at AS TIMESTAMP), try_cast(ingested_at AS TIMESTAMP)) DESC NULLS LAST
        ) AS rn
      FROM read_parquet('{ratings_path.as_posix()}', union_by_name=True)
      WHERE (published_at IS NULL OR try_cast(published_at AS TIMESTAMP) <= TIMESTAMP '{asof_ts}')
        AND (ingested_at IS NULL OR try_cast(ingested_at AS TIMESTAMP) <= TIMESTAMP '{asof_ts}')
        AND (effective_at IS NULL OR try_cast(effective_at AS TIMESTAMP) <= TIMESTAMP '{asof_ts}')
        AND (rating_date IS NULL OR try_cast(rating_date AS TIMESTAMP) <= TIMESTAMP '{asof_ts}')
    )
    SELECT * FROM r WHERE rn = 1
    """
    df = con.execute(query).df()
    out: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        cid = str(row.get("company_id"))
        out[cid] = {
            "rating_symbol": row.get("rating_symbol"),
            "current_rating_symbol": row.get("current_rating_symbol"),
            "outlook": row.get("outlook"),
            "creditwatch": row.get("creditwatch"),
            "source_type": row.get("source_type"),
            "artifact_id": row.get("artifact_id"),
            "published_at": row.get("published_at"),
            "ingested_at": row.get("ingested_at"),
        }
    return out


def _rating_score(rating: Optional[str]) -> Optional[float]:
    if rating is None:
        return None
    r = str(rating).upper().strip()
    if not r:
        return None
    mapping = {
        "AAA": 1,
        "AA+": 2,
        "AA": 3,
        "AA-": 4,
        "A+": 5,
        "A": 6,
        "A-": 7,
        "BBB+": 8,
        "BBB": 9,
        "BBB-": 10,
        "BB+": 11,
        "BB": 12,
        "BB-": 13,
        "B+": 14,
        "B": 15,
        "B-": 16,
        "CCC+": 17,
        "CCC": 18,
        "CCC-": 19,
        "CC": 20,
        "C": 21,
        "D": 22,
    }
    for key, score in mapping.items():
        if key in r:
            return float(score)
    return None


def _normalize_watch(value: Any) -> Any:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("y", "yes", "true", "watch", "negative", "positive"):
        return True
    if s in ("n", "no", "false", "none", "stable"):
        return False
    return None


def _to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except Exception:
        pass
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast update of ownership/rating features in snapshot JSONL.")
    parser.add_argument("--in-path", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--ownership-path", default="data/inputs_layer/ownership_13f_summary.parquet")
    parser.add_argument("--issuer-ratings-path", default="data/inputs_layer/issuer_rating_history.parquet")
    parser.add_argument("--facts-path", default=None, help="Optional facts path for institutional_pct denominator")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    ownership_path = Path(args.ownership_path)
    issuer_ratings_path = Path(args.issuer_ratings_path)
    facts_path = Path(args.facts_path) if args.facts_path else None

    ownership = _load_ownership_map(ownership_path, facts_path, args.asof)
    ratings = _load_rating_map(issuer_ratings_path, args.asof)
    print(f"[load] ownership companies={len(ownership)} ratings companies={len(ratings)}")

    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    updated = 0
    with in_path.open("r") as fin, tmp_out.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            snap = json.loads(line)
            cid = str(snap.get("company_id"))
            asof_time = snap.get("as_of_time", args.asof)
            feats = snap.setdefault("features", {})

            own = ownership.get(cid)
            if own:
                prov = [
                    {
                        "artifact_type": "RawDocument",
                        "artifact_id": _to_str(own.get("artifact_id")) or f"wrds_13f:{cid}",
                        "source": _to_str(own.get("source_type")) or "wrds_13f",
                        "published_at": _to_str(own.get("published_at")),
                        "ingested_at": _to_str(own.get("ingested_at")),
                        "hash": None,
                    }
                ]
                feats["ownership_governance.top5_holder_pct"] = _feature_record(
                    name="ownership_governance.top5_holder_pct",
                    value=own.get("top5_pct"),
                    unit="ratio",
                    as_of_time=asof_time,
                    confidence=0.65 if own.get("top5_pct") is not None else None,
                    provenance=prov,
                    missing_reason="not_disclosed" if own.get("top5_pct") is None else None,
                    window={"type": "asof", "length_days": 0},
                )
                feats["ownership_governance.institutional_pct"] = _feature_record(
                    name="ownership_governance.institutional_pct",
                    value=own.get("inst_pct"),
                    unit="ratio",
                    as_of_time=asof_time,
                    confidence=0.6 if own.get("inst_pct") is not None else None,
                    provenance=prov,
                    missing_reason="not_disclosed" if own.get("inst_pct") is None else None,
                    window={"type": "asof", "length_days": 0},
                )

            rt = ratings.get(cid)
            if rt:
                rating = rt.get("rating_symbol") or rt.get("current_rating_symbol")
                payload = {
                    "rating": _to_str(rating),
                    "outlook": _to_str(rt.get("outlook")),
                    "watchlist": _normalize_watch(rt.get("creditwatch")),
                    "score": _rating_score(_to_str(rating)),
                }
                prov = [
                    {
                        "artifact_type": "ExtractedFact",
                        "artifact_id": _to_str(rt.get("artifact_id")) or f"issuer_rating:{cid}",
                        "source": _to_str(rt.get("source_type")) or "issuer_ratings",
                        "published_at": _to_str(rt.get("published_at")),
                        "ingested_at": _to_str(rt.get("ingested_at")),
                        "hash": None,
                    }
                ]
                feats["capital_structure.rating_state"] = _feature_record(
                    name="capital_structure.rating_state",
                    value=payload,
                    unit="rating",
                    as_of_time=asof_time,
                    confidence=0.72 if payload.get("rating") is not None else 0.55,
                    provenance=prov,
                    missing_reason="not_disclosed" if payload.get("rating") is None else None,
                )

            prov_root = snap.setdefault("provenance", {})
            inputs_used = prov_root.setdefault("inputs_used", {})
            inputs_used["ownership"] = str(ownership_path)
            inputs_used["issuer_ratings"] = str(issuer_ratings_path)
            updated += 1
            fout.write(json.dumps(snap) + "\n")

    tmp_out.replace(out_path)
    print(f"Wrote updated snapshots -> {out_path} rows={updated}")


if __name__ == "__main__":
    main()
