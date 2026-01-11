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


def _case_key(case: Dict[str, Any]) -> str:
    return "|".join(
        [
            str(case.get("company_id") or ""),
            str(case.get("as_of_time") or ""),
            str(case.get("anchor_action_id") or ""),
        ]
    )


def main() -> None:
    args = _parse_args()
    selected: List[Dict[str, Any]] = []
    seen = set()
    max_cases = max(1, int(args.case_count)) if args.case_count else None

    for report_path_str in args.source_report_json:
        report_path = Path(report_path_str)
        payload = json.loads(report_path.read_text())
        for raw_case in payload.get("cases", []) or []:
            case = _normalize_case(raw_case)
            if case is None:
                continue
            case_key = _case_key(case)
            if case_key in seen:
                continue
            seen.add(case_key)
            selected.append(case)
            if max_cases is not None and len(selected) >= max_cases:
                break
        if max_cases is not None and len(selected) >= max_cases:
            break

    manifest = {
        "manifest_generated_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label or "",
        "source_reports": [str(Path(path)) for path in args.source_report_json],
        "case_count": len(selected),
        "cases": selected,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
