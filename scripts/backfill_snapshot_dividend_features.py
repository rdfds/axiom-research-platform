import argparse
import json
from datetime import timezone
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import pandas as pd


RECURRING_DIVIDEND_EVENT_TYPES = {
    "dividend_regular",
    "dividend_increase",
    "dividend_cut",
    "dividend_initiate",
}
RECURRING_DIVIDEND_SUBTYPES = {
    "regular",
    "dividend_increase",
    "dividend_cut",
    "dividend_initiate",
}


def _now_iso() -> str:
    return pd.Timestamp.now(tz=timezone.utc).isoformat()


def _feature_record(
    *,
    name: str,
    value,
    unit: str,
    as_of_time: str,
    confidence,
    provenance: List[Dict[str, str]],
    missing_reason,
    fallback_used,
) -> Dict[str, object]:
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "computed_at": _now_iso(),
        "as_of_time": as_of_time,
        "window": {"type": "lookback", "length_days": 730},
        "confidence": confidence,
        "provenance": provenance,
        "missing_reason": missing_reason,
        "fallback_used": fallback_used,
    }


def _load_company_ids(snapshot_dir: Path) -> List[str]:
    ids = []
    for path in sorted(snapshot_dir.glob("company_id=*.json")):
        company_id = path.stem.split("company_id=", 1)[-1]
        if company_id:
            ids.append(company_id)
    return ids


def _load_identifier_maps(entity_identifier_path: Path) :
    con = duckdb.connect()
    df = con.execute(
        "SELECT entity_id, identifier_value, identifier_type "
        f"FROM read_parquet('{entity_identifier_path.as_posix()}', union_by_name=True)"
    ).df()
    df = df.dropna(subset=["entity_id", "identifier_value"])
    df["entity_id"] = df["entity_id"].astype(str)
    df["identifier_value"] = df["identifier_value"].astype(str)

    identifier_to_entity: Dict[str, str] = {}
    entity_to_identifiers: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        ent = row["entity_id"]
        ident = row["identifier_value"]
        ident_type = str(row.get("identifier_type", "")).lower() if row.get("identifier_type") is not None else ""
        aliases = {ident}
        if ident_type == "ticker":
            aliases.add(ident.upper())
        if ident_type in ("cusip", "isin", "sedol"):
            aliases.add(ident.upper())
        if ident.isdigit():
            stripped = ident.lstrip("0")
            if stripped:
                aliases.add(stripped)
                for width in (6, 8, 10):
                    aliases.add(stripped.zfill(width))
            for width in (6, 8, 10):
                aliases.add(ident.zfill(width))
        if ident_type == "permno":
            aliases.add(f"permno:{ident}")
            if ident.isdigit():
                stripped = ident.lstrip("0")
                if stripped:
                    aliases.add(f"permno:{stripped}")
        if ident_type == "permco":
            aliases.add(f"permco:{ident}")
            if ident.isdigit():
                stripped = ident.lstrip("0")
                if stripped:
                    aliases.add(f"permco:{stripped}")
        for alias in aliases:
            identifier_to_entity[alias] = ent
            entity_to_identifiers.setdefault(ent, []).append(alias)

    for ent in list(entity_to_identifiers.keys()):
        identifier_to_entity[ent] = ent
        if ent not in entity_to_identifiers[ent]:
            entity_to_identifiers[ent].append(ent)
    return identifier_to_entity, entity_to_identifiers


def _resolve_tickers(
    company_ids: List[str],
    identifier_to_entity: Dict[str, str],
    entity_to_identifiers: Dict[str, List[str]],
) -> pd.DataFrame:
    rows = []
    for company_id in company_ids:
        cid = str(company_id)
        canonical = identifier_to_entity.get(cid, cid)
        aliases = list(entity_to_identifiers.get(canonical, []))
        if cid not in aliases:
            aliases.append(cid)
        if canonical not in aliases:
            aliases.append(canonical)
        seen = set()
        for alias in aliases:
            alias_s = str(alias)
            if alias_s in seen or not alias_s.isalpha():
                continue
            seen.add(alias_s)
            rows.append({"snapshot_company_id": cid, "ticker": alias_s.upper()})
    return pd.DataFrame(rows)


