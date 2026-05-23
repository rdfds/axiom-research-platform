#!/usr/bin/env python
"""
Debug a single signal computation from the command line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signals import SignalEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gvkey", required=True)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--signal", default="valuation_dislocation")
    args = parser.parse_args()

    engine = SignalEngine()
    if args.signal == "valuation_dislocation":
        result = engine.compute_valuation_dislocation(args.gvkey, args.date)
    elif args.signal == "size_factor":
        result = engine.compute_size_factor(args.gvkey, args.date)
    elif args.signal == "balance_sheet_optionality":
        result = engine.compute_balance_sheet_optionality(args.gvkey, args.date)
    elif args.signal == "growth_momentum":
        result = engine.compute_growth_momentum(args.gvkey, args.date)
    elif args.signal == "margin_trend":
        result = engine.compute_margin_trend(args.gvkey, args.date)
    elif args.signal == "refinancing_pressure":
        result = engine.compute_refinancing_pressure(args.gvkey, args.date)
    elif args.signal == "asset_intensity":
        result = engine.compute_asset_intensity(args.gvkey, args.date)
    else:
        raise SystemExit(f"Unknown signal: {args.signal}")

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
