#!/usr/bin/env python
"""
Build a tiny SEC-derived financial-facts refresh for a targeted company set.

This avoids the full all-company small-file rebuild when we only need to
validate a handful of companies.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.financial_fact_tags import FINANCIAL_FACT_TAGS
from src.ingestion import compute_version_id
from src.sec_companyfacts_bulk import CompanyFactsBulkSource


def _load_sec_ingest_module():
    path = ROOT / "scripts" / "26_ingest_sec_xbrl.py"
    spec = importlib.util.spec_from_file_location("sec_ingest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build targeted SEC financial-facts refresh.")
    parser.add_argument("--ciks", default=None, help="Comma-separated CIKs/company_ids.")
    parser.add_argument("--ciks-file", default=None, help="Path to newline/comma-delimited CIK/company_id file.")
    parser.add_argument("--years", default="2023,2024")
    parser.add_argument("--companyfacts-zip", default=str(ROOT / "data" / "sec" / "companyfacts.zip"))
    parser.add_argument("--companyfacts-dir", default=str(ROOT / "data" / "sec" / "companyfacts"))
    parser.add_argument("--warehouse-root", default="/tmp/targeted_financial_facts_refresh/warehouse_financials")
    parser.add_argument("--enriched-root", default="/tmp/targeted_financial_facts_refresh/enriched")
    parser.add_argument("--validity-root", default="/tmp/targeted_financial_facts_refresh/validity")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory", default="12GB")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_ciks(args: argparse.Namespace) -> list[str]:
    raw: list[str] = []
    if args.ciks:
        raw.extend([c.strip() for c in str(args.ciks).split(",") if c.strip()])
    if args.ciks_file:
        path = Path(args.ciks_file)
        if not path.exists():
            raise FileNotFoundError(f"--ciks-file not found: {path}")
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                raw.extend([c.strip() for c in line.replace(",", " ").split() if c.strip()])
    ciks = [c.zfill(10) for c in raw if c]
    return list(dict.fromkeys(ciks))


def main() -> None:
    args = _parse_args()
    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    if not years:
        raise ValueError("At least one year is required")
    ciks = _load_ciks(args)
    if not ciks:
        raise ValueError("At least one CIK is required")

    warehouse_root = Path(args.warehouse_root)
    enriched_root = Path(args.enriched_root)
    validity_root = Path(args.validity_root)

    if args.overwrite:
        for root in (warehouse_root, enriched_root, validity_root):
            if root.exists():
                shutil.rmtree(root)

    warehouse_root.mkdir(parents=True, exist_ok=True)
    enriched_root.mkdir(parents=True, exist_ok=True)
    validity_root.mkdir(parents=True, exist_ok=True)

    sec_ingest = _load_sec_ingest_module()
    start_date = pd.Timestamp(f"{min(years)}-01-01")
    end_date = pd.Timestamp(f"{max(years)}-12-31")

    total_records = 0
    companyfacts_zip = Path(args.companyfacts_zip) if args.companyfacts_zip else None
    if companyfacts_zip is not None and not companyfacts_zip.exists():
        companyfacts_zip = None
    companyfacts_dir = Path(args.companyfacts_dir) if args.companyfacts_dir else None

    with CompanyFactsBulkSource(
        companyfacts_dir=companyfacts_dir,
        companyfacts_zip=companyfacts_zip,
        hydrate_cache=False,
        prefer_zip=companyfacts_zip is not None,
    ) as source:
        for cik in ciks:
            payload, raw_hash, origin = source.load_with_metadata(cik)
            if payload is None:
                print(f"[skip] missing companyfacts for {cik}")
                continue
            records, _, _ = sec_ingest.parse_companyfacts(
                payload=payload,
                company_id=cik,
                permno=None,
                permco=None,
                ticker=None,
                cusip=None,
                start_date=start_date,
                end_date=end_date,
                include_8k=False,
                min_available_time=None,
                allowed_tags=FINANCIAL_FACT_TAGS,
            )
            if not records:
                print(f"[skip] no targeted records for {cik} from {origin}")
                continue

            ingest_time = pd.Timestamp.utcnow()
            rows = []
            payload_hash = raw_hash or hashlib.sha256(str(payload).encode("utf-8")).hexdigest()
            for rec in records:
                event_time = pd.to_datetime(rec["event_time"], utc=True)
                available_time = pd.to_datetime(rec["available_time"], utc=True)
                rows.append(
                    {
                        "source_system": "sec_edgar_xbrl",
                        "entity_id": cik,
                        "company_id": cik,
                        "event_time": event_time,
                        "available_time": available_time,
                        "ingestion_time": ingest_time,
                        "version_id": compute_version_id(
                            source_system="sec_edgar_xbrl",
                            entity_id=cik,
                            event_time=event_time.to_pydatetime(),
                            available_time=available_time.to_pydatetime(),
                            raw_payload_hash=payload_hash,
                        ),
                        "fiscal_period_end": pd.to_datetime(rec["fiscal_period_end"]),
                        "fiscal_year": rec.get("fiscal_year"),
                        "fiscal_quarter": rec.get("fiscal_quarter"),
                        "statement_type": rec.get("statement_type"),
                        "line_item": rec.get("line_item"),
                        "value": rec.get("value"),
                        "currency": rec.get("currency"),
                        "units": rec.get("units"),
                    }
                )

            df = pd.DataFrame(rows)
            for year, ydf in df.groupby(df["fiscal_period_end"].dt.year):
                if int(year) not in years:
                    continue
                out_dir = warehouse_root / f"year={int(year)}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"part_{cik}.parquet"
                ydf.to_parquet(out_path, index=False)
            total_records += len(df)
            print(f"[ok] {cik}: {len(df):,} records via {origin}")

    years_arg = ",".join(str(y) for y in years)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_financial_facts_registry.py"),
            "--financials-root",
            str(warehouse_root),
            "--out-root",
            str(enriched_root),
            "--years",
            years_arg,
            "--overwrite",
            "--threads",
            str(args.threads),
            "--memory",
            str(args.memory),
        ],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_fact_validity.py"),
            "--in-path",
            str(enriched_root),
            "--out",
            str(validity_root),
            "--years",
            years_arg,
            "--overwrite",
            "--threads",
            str(args.threads),
            "--memory",
            str(args.memory),
        ],
        check=True,
        cwd=ROOT,
    )

    print(f"[done] targeted records={total_records:,}")
    print(f"[done] warehouse_root={warehouse_root}")
    print(f"[done] enriched_root={enriched_root}")
    print(f"[done] validity_root={validity_root}")


if __name__ == "__main__":
    main()
