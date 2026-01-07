#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a compact fixed-case manifest from one or more historical evaluation reports."
    )
    parser.add_argument(
        "--source-report-json",
        nargs="+",
        required=True,
        help="One or more historical evaluation report JSONs to read cases from.",
    )
    parser.add_argument("--out-json", required=True, help="Destination manifest JSON path.")
    parser.add_argument(
        "--case-count",
        type=int,
        help="Optional maximum number of cases to keep after concatenating and deduping.",
    )
    parser.add_argument("--label", help="Optional short label stored in the manifest metadata.")
    return parser.parse_args()


def _normalize_case(raw_case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    company_id = str(raw_case.get("company_id") or "").strip()
    anchor_action_id = str(raw_case.get("anchor_action_id") or "").strip()
    anchor_action_family = str(raw_case.get("anchor_action_family") or "").strip()
    anchor_action_date = str(raw_case.get("anchor_action_date") or "").strip()
    as_of_time = str(raw_case.get("as_of_time") or "").strip()
    if not company_id or not anchor_action_id or not anchor_action_family or not anchor_action_date or not as_of_time:
        return None
    return {
        "company_id": company_id,
        "source_company_id": str(raw_case.get("source_company_id") or company_id).strip(),
        "ticker": str(raw_case.get("ticker") or "").strip(),
        "mapping_method": str(raw_case.get("mapping_method") or "").strip(),
        "anchor_action_id": anchor_action_id,
        "anchor_action_family": anchor_action_family,
        "anchor_action_date": anchor_action_date,
        "as_of_time": as_of_time,
    }


