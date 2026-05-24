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


def _load_identifier_maps(entity_identifier_path: Path) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
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


def _load_dividend_rows_for_tickers(
    outcomes_path: Path,
    company_tickers: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("snapshot_company_tickers", company_tickers)
    cutoff = as_of.isoformat().replace("T", " ")
    query = f"""
    SELECT
        CAST(t.snapshot_company_id AS VARCHAR) AS company_id,
        CAST(o.source_id AS VARCHAR) AS event_id,
        CAST(o.source_dataset AS VARCHAR) AS source_type,
        lower(CAST(o.action_type AS VARCHAR)) AS event_type,
        lower(CAST(o.action_subtype AS VARCHAR)) AS event_subtype,
        try_cast(o.action_date AS TIMESTAMP) AS announced_at,
        try_cast(o.action_date AS TIMESTAMP) AS created_at
    FROM read_parquet('{outcomes_path.as_posix()}', union_by_name=True) o
    INNER JOIN snapshot_company_tickers t
      ON upper(CAST(o.ticker AS VARCHAR)) = CAST(t.ticker AS VARCHAR)
    WHERE try_cast(o.action_date AS TIMESTAMP) <= TIMESTAMP '{cutoff}'
    """
    return con.execute(query).df()


def _build_dividend_lookup(events: pd.DataFrame, as_of: pd.Timestamp) -> Dict[str, Dict[str, object]]:
    lookup: Dict[str, Dict[str, object]] = {}
    if events.empty:
        return lookup

    events = events.copy()
    events["event_type"] = events["event_type"].astype(str).str.lower()
    events["event_subtype"] = events["event_subtype"].astype(str).str.lower()
    events["announced_at"] = pd.to_datetime(events["announced_at"], utc=True, errors="coerce")
    events["created_at"] = pd.to_datetime(events["created_at"], utc=True, errors="coerce")

    for company_id, group in events.groupby("company_id", sort=False):
        company_rows = group.copy()
        recurring = company_rows[
            company_rows["event_type"].isin(RECURRING_DIVIDEND_EVENT_TYPES)
            | company_rows["event_subtype"].isin(RECURRING_DIVIDEND_SUBTYPES)
        ].copy()
        if recurring.empty:
            lookup[company_id] = {
                "flag": False,
                "last_event_type": None,
                "provenance": [],
            }
            continue

        recurring = recurring.sort_values("announced_at", ascending=False, na_position="last")
        latest = recurring.iloc[0]
        latest_ts = latest["announced_at"]
        latest_raw = latest["event_type"] or latest["event_subtype"]
        provenance = []
        for _, row in recurring.head(5).iterrows():
            provenance.append(
                {
                    "artifact_type": "CorporateActionEvent",
                    "artifact_id": row["event_id"] if row["event_id"] not in (None, "", "None") else "unknown",
                    "source": None if row["source_type"] in (None, "", "None") else row["source_type"],
                    "published_at": None if pd.isna(row["announced_at"]) else row["announced_at"].isoformat(),
                    "ingested_at": None if pd.isna(row["created_at"]) else row["created_at"].isoformat(),
                    "hash": None,
                }
            )
        lookup[company_id] = {
            "flag": bool(pd.notna(latest_ts) and latest_ts >= (as_of - pd.Timedelta(days=730))),
            "last_event_type": None if latest_raw in (None, "", "None") else str(latest_raw),
            "provenance": provenance,
        }
    return lookup


def _update_snapshot(path: Path, as_of: pd.Timestamp, dividend_lookup: Dict[str, Dict[str, object]]) -> bool:
    snapshot = json.load(path.open())
    company_id = str(snapshot.get("company_id", ""))
    features = snapshot.setdefault("features", {})
    info = dividend_lookup.get(company_id)

    if info is None:
        flag = None
        last_event_type = None
        provenance = []
    else:
        flag = info["flag"]
        last_event_type = info["last_event_type"]
        provenance = info["provenance"]

    as_of_time = snapshot.get("as_of_time") or as_of.isoformat()
    new_flag = _feature_record(
        name="capital_return.dividend_payer_flag",
        value=flag,
        unit="boolean",
        as_of_time=as_of_time,
        confidence=0.8 if flag is True else (0.6 if flag is False else None),
        provenance=provenance,
        missing_reason="unavailable" if flag is None else None,
        fallback_used="event_lookback_heuristic" if flag is not None else None,
    )
    new_last = _feature_record(
        name="capital_return.last_dividend_event_type",
        value=last_event_type,
        unit="label",
        as_of_time=as_of_time,
        confidence=0.75 if last_event_type is not None else None,
        provenance=provenance,
        missing_reason="unavailable" if last_event_type is None else None,
        fallback_used=None,
    )

    changed = (
        features.get("capital_return.dividend_payer_flag") != new_flag
        or features.get("capital_return.last_dividend_event_type") != new_last
    )
    if not changed:
        return False

    features["capital_return.dividend_payer_flag"] = new_flag
    features["capital_return.last_dividend_event_type"] = new_last
    with path.open("w") as f:
        json.dump(snapshot, f, separators=(",", ":"), ensure_ascii=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill dividend features into keyed company snapshots.")
    parser.add_argument("--snapshot-root", required=True, help="Snapshot root containing keyed/as_of_date=... files")
    parser.add_argument("--asof", required=True, help="As-of date, e.g. 2026-02-28")
    parser.add_argument("--outcomes-path", required=True, help="Path to normalized action outcomes parquet")
    parser.add_argument("--entity-identifier-path", default="data/inputs_layer/entity_identifier.parquet")
    args = parser.parse_args()

    snapshot_root = Path(args.snapshot_root)
    keyed_dir = snapshot_root / "keyed" / f"as_of_date={args.asof}"
    if not keyed_dir.exists():
        raise SystemExit(f"Keyed snapshot directory not found: {keyed_dir}")

    company_ids = _load_company_ids(keyed_dir)
    if not company_ids:
        raise SystemExit(f"No snapshot files found in: {keyed_dir}")

    as_of = pd.to_datetime(args.asof, utc=True)
    identifier_to_entity, entity_to_identifiers = _load_identifier_maps(Path(args.entity_identifier_path))
    company_tickers = _resolve_tickers(company_ids, identifier_to_entity, entity_to_identifiers)
    events = _load_dividend_rows_for_tickers(Path(args.outcomes_path), company_tickers, as_of)
    dividend_lookup = _build_dividend_lookup(events, as_of)

    updated = 0
    for path in sorted(keyed_dir.glob("company_id=*.json")):
        if _update_snapshot(path, as_of, dividend_lookup):
            updated += 1

    true_count = sum(1 for v in dividend_lookup.values() if v["flag"] is True)
    false_count = sum(1 for v in dividend_lookup.values() if v["flag"] is False)
    none_count = len(company_ids) - len(dividend_lookup)
    print(
        json.dumps(
            {
                "ok": True,
                "snapshot_root": snapshot_root.as_posix(),
                "asof": args.asof,
                "company_count": len(company_ids),
                "updated_snapshots": updated,
                "dividend_payer_true": true_count,
                "dividend_payer_false": false_count,
                "dividend_payer_none": none_count,
            }
        )
    )


if __name__ == "__main__":
    main()
