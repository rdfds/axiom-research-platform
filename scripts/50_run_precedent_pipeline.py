#!/usr/bin/env python
"""
Run precedent-based matching for a single company + action.

Example:
  python -u scripts/50_run_precedent_pipeline.py \
    --company-id 001690 \
    --as-of 2024-12-31 \
    --action-id capital_return.open_market_buyback \
    --param size_pct_market_cap=0.2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.run import run_precedent


def _default_precedent_outcomes_path() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    return str(repo_root / "data" / "curated" / "action_outcomes_with_credit_ratings.normalized_full.parquet")


def _parse_param_values(items: list[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid --param format: {raw}. Expected key=value.")
        key, val = raw.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            raise ValueError(f"Invalid --param key: {raw}")
        if val.lower() in {"true", "false"}:
            out[key] = val.lower() == "true"
            continue
        try:
            if "." in val:
                out[key] = float(val)
            else:
                out[key] = int(val)
            continue
        except Exception:
            pass
        out[key] = val
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--company-id", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--action-id", default=None, help="Canonical action id, e.g. capital_return.open_market_buyback")
    parser.add_argument("--action-type", default=None, help="Action type root or legacy alias (e.g. buyback).")
    parser.add_argument("--action-subtype", default=None, help="Action subtype without root.")
    parser.add_argument("--param", action="append", default=[], help="Action parameter key=value (repeatable).")
    parser.add_argument("--size", type=float, default=None, help="Legacy size alias; mapped to size_pct_market_cap if not provided.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--outcomes-path", default=_default_precedent_outcomes_path())
    parser.add_argument(
        "--state-snapshot-root",
        default=None,
        help="Optional CompanyState snapshot root that contains keyed/as_of_date=... files (fastest path).",
    )
    parser.add_argument(
        "--state-snapshot-path",
        default=None,
        help="Optional CompanyState JSONL path to use as baseline (fast path, avoids warehouse scans).",
    )
    parser.add_argument("--out", default=None, help="Optional output JSON path")
    parser.add_argument("--quiet", action="store_true", help="Do not print full JSON payload to stdout.")
    args = parser.parse_args()

    action_params = _parse_param_values(args.param)
    if args.size is not None and "size_pct_market_cap" not in action_params:
        action_params["size_pct_market_cap"] = args.size

    pack = run_precedent(
        company_id=args.company_id,
        as_of_date=args.as_of,
        action_id=args.action_id,
        action_type=args.action_type,
        action_subtype=args.action_subtype,
        action_params=action_params,
        config_path=args.config,
        outcomes_path=args.outcomes_path,
        state_snapshot_root=args.state_snapshot_root,
        state_snapshot_path=args.state_snapshot_path,
    )

    payload = pack.to_dict()
    if not args.quiet:
        print(json.dumps(payload, indent=2, default=str))
    else:
        legacy_dists = payload.get("legacy_distributions", [])
        if not isinstance(legacy_dists, list):
            legacy_dists = []
        dist_count = len(legacy_dists)
        if dist_count == 0 and isinstance(payload.get("distributions"), list):
            dist_count = len(payload.get("distributions", []))
        print(
            json.dumps(
                {
                    "matches": len(payload.get("matches", [])),
                    "distributions": dist_count,
                }
            )
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
