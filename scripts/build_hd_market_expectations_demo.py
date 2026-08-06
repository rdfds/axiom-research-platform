#!/usr/bin/env python3
"""Build the public Home Depot market-expectations demo from sample data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


SAMPLE_TO_RUNTIME = {
    "valuation_driver_data.sample.json": "valuation_driver_data.json",
    "expectation_driver_history.sample.json": "expectation_driver_history.json",
    "expectation_evidence_cohort.sample.json": "expectation_evidence_cohort.json",
    "forward_gap_placebo_walk_forward_operating_ex_energy.sample.md": "forward_gap_placebo_walk_forward_operating_ex_energy.md",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def copy_sample_inputs(sample_dir: Path, build_dir: Path) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    for sample_name, runtime_name in SAMPLE_TO_RUNTIME.items():
        src = sample_dir / sample_name
        if not src.exists():
            raise FileNotFoundError(f"Missing sample input: {src}")
        shutil.copy2(src, build_dir / runtime_name)


def build_demo(sample_dir: Path, build_dir: Path) -> Path:
    root = repo_root()
    copy_sample_inputs(sample_dir, build_dir)

    env = os.environ.copy()
    env["AXIOM_MNA_INSIGHTS_DIR"] = str(build_dir)

    subprocess.run(
        [sys.executable, str(root / "scripts" / "build_valuation_action_bridge.py")],
        cwd=root,
        env=env,
        check=True,
    )
    return build_dir / "valuation_action_bridge.html"


def main() -> int:
    root = repo_root()
    default_sample_dir = root / "examples" / "hd_market_expectations"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=default_sample_dir,
        help="Directory containing the *.sample.* inputs.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=default_sample_dir / "build",
        help="Output directory for runtime inputs and generated HTML.",
    )
    args = parser.parse_args()

    html = build_demo(args.sample_dir.resolve(), args.build_dir.resolve())
    print(f"Built HD market-expectations demo: {html}#HD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
