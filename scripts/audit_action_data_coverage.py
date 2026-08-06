from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.action_data_support import build_action_support_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit canonical action-data coverage for recommender-relevant action IDs.")
    parser.add_argument(
        "--outcomes-path",
        default="./data/curated/action_outcomes_with_credit_ratings.normalized_full.parquet",
        help="Path to normalized outcomes parquet.",
    )
    parser.add_argument("--out-json", default="", help="Optional path to write JSON report.")
    args = parser.parse_args()

    report = build_action_support_report(outcomes_path=Path(args.outcomes_path))
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
