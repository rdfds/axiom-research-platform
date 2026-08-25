#!/usr/bin/env python3
"""Build the deterministic public benchmark artifact from its committed report."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = Path(
    "examples/hd_market_expectations/"
    "forward_gap_placebo_walk_forward_operating_ex_energy.sample.md"
)
DEFAULT_OUTPUT_PATH = Path("results/public_benchmark.json")


def _required(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not parse {label} from {SOURCE_PATH}")
    return match.group(1)


def _parse_placebo_rows(text: str) -> list[dict[str, float]]:
    section = _required(
        r"## Placebo Check\n\n([\s\S]*?)(?=\n## )",
        text,
        "placebo table",
    )
    rows: list[dict[str, float]] = []
    for line in section.splitlines():
        if not line.startswith("|") or "---" in line or "Lambda" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 5:
            continue
        values = [float(cell) for cell in cells]
        rows.append(
            {
                "lambda": values[0],
                "actual_mean_mae_improvement": values[1],
                "placebo_mean_mae_improvement": values[2],
                "actual_minus_placebo": values[3],
                "actual_beats_placebo_rate": values[4],
            }
        )
    if not rows:
        raise ValueError(f"Could not parse placebo rows from {SOURCE_PATH}")
    return rows


def build_payload(source_path: Path | None = None) -> dict[str, Any]:
    relative_source = source_path or SOURCE_PATH
    absolute_source = REPOSITORY_ROOT / relative_source
    source_bytes = absolute_source.read_bytes()
    text = source_bytes.decode("utf-8")

    train_ends = _required(
        r"Walk-forward splits: train ends `([^`]+)`",
        text,
        "walk-forward cutoffs",
    ).split(", ")
    placebo_rows = _parse_placebo_rows(text)
    best_lambda = float(_required(r"Best lambda: `([^`]+)`", text, "best lambda"))
    selected = next(row for row in placebo_rows if row["lambda"] == best_lambda)

    return {
        "schema_version": 1,
        "benchmark": "forward-gap lambda policy",
        "source": {
            "path": relative_source.as_posix(),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "evaluation": {
            "as_of": _required(r"As of: `([^`]+)`", text, "as-of date"),
            "mode": _required(r"Validation mode: \*\*([^*]+)\*\*", text, "mode"),
            "train_end_dates": train_ends,
            "test_window_years": int(
                _required(r"test window `([0-9]+)` years", text, "test window")
            ),
            "placebo_runs": int(
                _required(r"Placebo runs: \*\*([0-9]+)\*\*", text, "placebo runs")
            ),
            "candidates_attempted": int(
                _required(
                    r"Candidates attempted: \*\*([0-9]+)\*\*",
                    text,
                    "candidate count",
                )
            ),
            "successful_driver_horizon_evaluations": int(
                _required(
                    r"Successful driver/horizon evaluations: \*\*([0-9]+)\*\*",
                    text,
                    "evaluation count",
                )
            ),
        },
        "selected_policy": {
            "lambda": best_lambda,
            "mean_mae_improvement": float(
                _required(
                    r"Mean MAE improvement: `([^`]+)`",
                    text,
                    "mean MAE improvement",
                )
            ),
        },
        "placebo_comparison": selected,
        "all_placebo_comparisons": placebo_rows,
    }


def _serialized(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Repository-relative path for the JSON artifact.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the committed artifact differs from the source report.",
    )
    args = parser.parse_args()

    output_path = REPOSITORY_ROOT / args.output
    expected = _serialized(build_payload())
    if args.check:
        if not output_path.exists() or output_path.read_text(encoding="utf-8") != expected:
            print(f"Benchmark artifact is stale: {args.output}")
            return 1
        print(f"Benchmark artifact is reproducible: {args.output}")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(expected, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
