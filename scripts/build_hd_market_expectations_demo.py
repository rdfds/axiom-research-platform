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


def build_demo(sample_dir: Path, build_dir: Path) :
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


