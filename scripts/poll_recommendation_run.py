#!/usr/bin/env python
"""Poll a RecommendationRun JSON file until completion."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Poll RecommendationRun status from run store")
    p.add_argument("--run-id", required=True)
    p.add_argument("--runs-root", required=True)
    p.add_argument("--interval-seconds", type=float, default=5.0)
    p.add_argument("--max-waits", type=int, default=0, help="0 means wait indefinitely")
    return p.parse_args()


