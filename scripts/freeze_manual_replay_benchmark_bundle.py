#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK_CONFIG = ROOT / "configs" / "historical_eval_manifests" / "2026-03-17" / "manual_replay_benchmark_lock.json"
DEFAULT_BUNDLE_ROOT = ROOT / "out" / "manual_replay_bundle_20260405"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a manual replay benchmark bundle into a durable workspace path.")
    parser.add_argument("--lock-config", default=str(DEFAULT_LOCK_CONFIG))
    parser.add_argument("--bundle-root", default=str(DEFAULT_BUNDLE_ROOT))
    parser.add_argument("--include-feedback-model", action="store_true", help="Also freeze the v7 feedback HGB model artifacts.")
    return parser.parse_args()


