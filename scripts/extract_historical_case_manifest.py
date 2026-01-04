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


