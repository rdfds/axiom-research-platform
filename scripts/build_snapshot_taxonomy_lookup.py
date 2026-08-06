#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def _extract_metric_value(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("value")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact taxonomy lookup from keyed snapshot JSON files.")
    parser.add_argument("--snapshot-root")
    parser.add_argument("--snapshot-catalog-path")
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def _parse_snapshot_taxonomy(path: Path) -> Dict[str, str] | None:
    company_id = path.stem.split("company_id=", 1)[-1].strip()
    if not company_id:
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    features = payload.get("features") if isinstance(payload, dict) else None
    features = features if isinstance(features, dict) else {}
    sector_name = str(_extract_metric_value(features.get("taxonomy.sector")) or "").strip()
    subsector_name = str(_extract_metric_value(features.get("taxonomy.subsector")) or "").strip()
    if not sector_name and not subsector_name:
        return None
    return {
        "company_id": company_id,
        "taxonomy.sector": sector_name,
        "taxonomy.subsector": subsector_name,
    }


def _taxonomy_record_value(record: Any) -> str:
    value = _extract_metric_value(record)
    return str(value or "").strip()


def _catalog_row_sort_key(
    sector_name: str,
    subsector_name: str,
    *,
    support_mode: str,
    confidence: float,
    as_of_time: str,
) -> Tuple[int, int, float, str]:
    support_rank = 1 if str(support_mode or "").strip().lower() == "exact" else 0
    return (
        int(bool(sector_name) and bool(subsector_name)),
        support_rank,
        float(confidence),
        str(as_of_time or ""),
    )


def _parse_snapshot_catalog_taxonomy(path: Path) -> List[Dict[str, str]]:
    rows_by_company: Dict[str, Tuple[Tuple[int, int, float, str], Dict[str, str]]] = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except Exception:
                continue
            company_id = str(payload.get("company_id") or "").strip()
            if not company_id:
                continue
            features = payload.get("features") if isinstance(payload, dict) else None
            features = features if isinstance(features, dict) else {}
            sector_record = features.get("taxonomy.sector")
            subsector_record = features.get("taxonomy.subsector")
            sector_name = _taxonomy_record_value(sector_record)
            subsector_name = _taxonomy_record_value(subsector_record)
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
            ).strip()
            sort_key = _catalog_row_sort_key(
                sector_name,
                subsector_name,
                support_mode=support_mode,
                confidence=confidence,
                as_of_time=str(payload.get("as_of_time") or ""),
            )
            row = {
                "company_id": company_id,
                "taxonomy.sector": sector_name,
                "taxonomy.subsector": subsector_name,
            }
            existing = rows_by_company.get(company_id)
            if existing is None or sort_key > existing[0]:
                rows_by_company[company_id] = (sort_key, row)
    return [row for _, row in rows_by_company.values()]


def main() -> None:
    args = _parse_args()
    snapshot_root = Path(args.snapshot_root) if args.snapshot_root else None
    snapshot_catalog_path = Path(args.snapshot_catalog_path) if args.snapshot_catalog_path else None
    if snapshot_root is None and snapshot_catalog_path is None:
        raise SystemExit("Provide --snapshot-root and/or --snapshot-catalog-path")
    out_path = Path(args.out_path)
    rows: List[Dict[str, str]] = []
    if snapshot_root is not None and snapshot_root.exists():
        snapshot_paths = sorted(snapshot_root.glob("company_id=*.json"))
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            for row in executor.map(_parse_snapshot_taxonomy, snapshot_paths):
                if row:
                    rows.append(row)
    if snapshot_catalog_path is not None and snapshot_catalog_path.exists():
        rows.extend(_parse_snapshot_catalog_taxonomy(snapshot_catalog_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop_duplicates(subset=["company_id"]).to_parquet(out_path, index=False)
    print(
        json.dumps(
            {
                "ok": True,
                "snapshot_root": str(snapshot_root) if snapshot_root is not None else "",
                "snapshot_catalog_path": str(snapshot_catalog_path) if snapshot_catalog_path is not None else "",
                "out_path": str(out_path),
                "row_count": len(rows),
            }
        )
    )


if __name__ == "__main__":
    main()
