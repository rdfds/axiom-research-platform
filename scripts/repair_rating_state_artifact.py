#!/usr/bin/env python3
"""Repair rating-state metrics in a materialized company-state artifact."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


REPAIR_METRICS = ["capital_structure.rating_state"]

RATING_SCORE_MAP = {
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

DEFAULT_RATINGS_PATHS = [
    "/tmp/issuer_rating_history.parquet",
    "./data/inputs_layer/issuer_rating_history.parquet",
    "./data/curated/issuer_ratings_ciq.parquet",
    "./data/wrds/ciq/ciq_entity_ratings.csv.gz",
]

CANONICAL_ALIASES = {
    "company_id": {
        "company_id",
        "companyid",
        "cik",
        "issuer_cik",
        "entity_cik",
        "company_cik",
    },
    "rating_symbol": {
        "rating_symbol",
        "current_rating_symbol",
        "rating",
        "current_rating",
        "ratingvalue",
        "rating_value",
        "ratingsymbol",
        "symbol",
    },
    "current_rating_symbol": {
        "current_rating_symbol",
        "rating_symbol",
        "current_rating",
        "rating",
    },
    "outlook": {
        "outlook",
        "rating_outlook",
        "current_outlook",
    },
    "creditwatch": {
        "creditwatch",
        "watchlist",
        "watch",
        "credit_watch",
        "rating_watch",
    },
    "rating_date": {
        "rating_date",
        "ratingdate",
        "effective_at",
        "effective_date",
        "effectivedate",
        "date",
        "announcedate",
        "announcement_date",
        "as_of_date",
        "published_at",
    },
    "published_at": {
        "published_at",
        "publish_date",
        "publisheddate",
        "announcement_date",
        "announcedate",
    },
    "effective_at": {
        "effective_at",
        "effective_date",
        "effectivedate",
        "rating_date",
        "ratingdate",
    },
    "artifact_id": {
        "artifact_id",
        "event_id",
        "id",
        "rating_id",
    },
    "source_type": {
        "source_type",
        "rating_agency",
        "agency",
        "provider",
        "source",
        "source_name",
        "event_subtype",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True, help="Input company-state JSONL artifact")
    parser.add_argument("--ratings-path", help="Optional issuer ratings parquet/csv.gz path")
    parser.add_argument("--out", required=True, help="Output repaired JSONL artifact")
    parser.add_argument("--summary-out", help="Optional summary JSON")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _normalize_company_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return digits.zfill(10)
    return text


def _null_if_na(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _node_support(node: Dict[str, Any] | None) -> str:
    if not node:
        return "unsupported"
    return str(node.get("support_mode") or "unsupported")


def _base_repaired_node(node: Dict[str, Any], *, computed_at: str) -> Dict[str, Any]:
    repaired = copy.deepcopy(node)
    repaired["computed_at"] = computed_at
    repaired["missing_reason"] = None
    repaired["quality_flags"] = repaired.get("quality_flags") or None
    return repaired


def _rating_score(rating: Any) -> Optional[float]:
    rating = _null_if_na(rating)
    if rating is None:
        return None
    normalized = str(rating).upper().strip()
    if not normalized:
        return None
    for label, score in RATING_SCORE_MAP.items():
        if label in normalized:
            return float(score)
    return None


def _coerce_watchlist(value: Any) -> Any:
    value = _null_if_na(value)
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    if lowered in ("y", "yes", "true", "watch", "negative", "positive"):
        return True
    if lowered in ("n", "no", "false", "none", "stable"):
        return False
    return None


def _prefer_fitch_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cols = [col for col in ("source_type", "rating_agency", "agency", "provider", "event_subtype") if col in df.columns]
    if not cols:
        return df
    mask = pd.Series(False, index=df.index)
    for col in cols:
        mask = mask | df[col].astype(str).str.contains("fitch", case=False, na=False)
    preferred = df[mask].copy()
    return preferred if not preferred.empty else df


def _resolve_ratings_path(explicit_path: str | None) -> Path:
    candidates = [explicit_path] if explicit_path else []
    candidates.extend(DEFAULT_RATINGS_PATHS)
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find issuer ratings data. Checked: " + ", ".join(DEFAULT_RATINGS_PATHS)
    )


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized_to_original: Dict[str, str] = {}
    for col in df.columns:
        normalized_to_original.setdefault(_normalize_name(col), col)
    renamed = df.copy()
    for canonical, aliases in CANONICAL_ALIASES.items():
        for alias in aliases:
            source = normalized_to_original.get(_normalize_name(alias))
            if source and canonical not in renamed.columns:
                renamed = renamed.rename(columns={source: canonical})
                break
    return renamed


def load_issuer_ratings(path: Path) -> pd.DataFrame:
    suffixes = path.suffixes
    if suffixes[-2:] == [".csv", ".gz"] or path.suffix == ".csv":
        opener = gzip.open if path.suffixes[-1:] == [".gz"] else open
        with opener(path, "rt", errors="ignore") as handle:
            df = pd.read_csv(handle, low_memory=False)
    else:
        df = pd.read_parquet(path)

    df = _canonicalize_columns(df)
    if "company_id" not in df.columns:
        raise ValueError(f"Ratings file {path} does not expose a company_id-like column after normalization.")

    df["company_id"] = df["company_id"].map(_normalize_company_id)
    df = df[df["company_id"].notna()].copy()

    for col in ("rating_date", "published_at", "effective_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    return df


def _rating_payload_from_row(row: pd.Series) -> Dict[str, Any]:
    rating = _null_if_na(row.get("rating_symbol")) or _null_if_na(row.get("current_rating_symbol"))
    outlook = _null_if_na(row.get("outlook"))
    watchlist = _coerce_watchlist(row.get("creditwatch"))
    return {
        "rating": _null_if_na(rating),
        "outlook": outlook,
        "watchlist": watchlist,
        "score": _rating_score(rating),
    }


def _row_support_mode(row: pd.Series) -> str:
    payload = _rating_payload_from_row(row)
    return "exact" if payload.get("rating") is not None else "proxy_missing_component"


def build_rating_index(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if df.empty:
        return {}
    df = df.copy()
    df["company_id"] = df["company_id"].map(_normalize_company_id)
    df = df[df["company_id"].notna()].copy()
    order_cols = [col for col in ("rating_date", "published_at", "effective_at") if col in df.columns]
    out: Dict[str, Dict[str, Any]] = {}
    for company_id, group in df.groupby("company_id", sort=False):
        group = _prefer_fitch_rows(group.copy())
        if order_cols:
            group = group.sort_values(order_cols, ascending=[False] * len(order_cols))
        row = group.iloc[0]
        out[str(company_id)] = {
            "payload": _rating_payload_from_row(row),
            "support_mode": _row_support_mode(row),
            "source_type": _null_if_na(row.get("source_type")) or "issuer_ratings",
            "artifact_id": _null_if_na(row.get("artifact_id")) or f"issuer_rating:{company_id}:{_null_if_na(row.get('rating_date'))}",
            "published_at": str(_null_if_na(row.get("published_at"))) if _null_if_na(row.get("published_at")) is not None else None,
            "ingested_at": str(_null_if_na(row.get("ingested_at"))) if _null_if_na(row.get("ingested_at")) is not None else None,
            "rating_date": str(_null_if_na(row.get("rating_date"))) if _null_if_na(row.get("rating_date")) is not None else None,
            "source_row": row.to_dict(),
        }
    return out


def repair_rating_state(
    *,
    features: Dict[str, Any],
    company_id: str,
    rating_index: Dict[str, Dict[str, Any]],
    computed_at: str,
) -> bool:
    target = features.get("capital_structure.rating_state")
    if not target or target.get("value") is not None:
        return False

    record = rating_index.get(_normalize_company_id(company_id) or str(company_id))
    if not record:
        return False

    repaired = _base_repaired_node(target, computed_at=computed_at)
    repaired["value"] = record["payload"]
    repaired["support_mode"] = record["support_mode"]
    repaired["confidence"] = 0.72 if record["payload"].get("rating") is not None else 0.55
    repaired["fallback_used"] = None if record["payload"].get("score") is not None else "heuristic"
    repaired["provenance"] = [
        {
            "artifact_type": "ExtractedFact",
            "artifact_id": str(record["artifact_id"]),
            "source": str(record["source_type"]),
            "published_at": record["published_at"],
            "ingested_at": record["ingested_at"],
            "hash": None,
        }
    ]
    repaired["component_breakdown"] = {
        "rating": record["payload"].get("rating"),
        "outlook": record["payload"].get("outlook"),
        "watchlist": record["payload"].get("watchlist"),
        "score": record["payload"].get("score"),
        "source_type": record["source_type"],
        "rating_date": record["rating_date"],
        "selection_rule": "latest_rating_prefer_fitch",
    }
    repaired["quality_flags"] = None
    features["capital_structure.rating_state"] = repaired
    return True


def build_summary(path: Path) -> Dict[str, Dict[str, int]]:
    counters: Dict[str, Counter[str]] = {metric: Counter() for metric in REPAIR_METRICS}
    for row in iter_rows(path):
        features = row.get("features") or {}
        for metric in REPAIR_METRICS:
            node = features.get(metric) or {}
            mode = str(node.get("support_mode") or "unsupported")
            if node.get("value") is None:
                mode = "unsupported"
            counters[metric][mode] += 1
    return {
        metric: {
            "exact": counter["exact"],
            "proxy_missing_component": counter["proxy_missing_component"],
            "unsupported": counter["unsupported"],
        }
        for metric, counter in counters.items()
    }


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)
    ratings_path = _resolve_ratings_path(args.ratings_path)
    ratings_df = load_issuer_ratings(ratings_path)
    rating_index = build_rating_index(ratings_df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    computed_at = _now_iso()

    with out_path.open("w") as out_handle:
        for row in iter_rows(artifact_path):
            repair_rating_state(
                features=row.get("features") or {},
                company_id=str(row.get("company_id")),
                rating_index=rating_index,
                computed_at=computed_at,
            )
            out_handle.write(json.dumps(row) + "\n")

    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(build_summary(out_path), indent=2))

    print(f"Repaired rating-state metrics -> {out_path}")


if __name__ == "__main__":
    main()
