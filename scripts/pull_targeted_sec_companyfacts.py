#!/usr/bin/env python3
"""Download targeted SEC companyfacts JSON files to a non-iCloud local path.

This is meant to avoid Desktop/iCloud file-provider issues by pulling only the
CIKs we care about into a stable local folder such as `/Users/.../code/...`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company-ids-file", required=True, help="File with one CIK/entity_id per line")
    parser.add_argument("--out-root", required=True, help="Output folder for downloaded companyfacts JSON")
    parser.add_argument(
        "--user-agent",
        default="AxiomResearch/1.0 (research contact: rvariankaval)",
        help="SEC fair-access User-Agent string",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.15, help="Delay between SEC requests")
    parser.add_argument("--overwrite", action="store_true", help="Re-download files even if they already exist")
    return parser.parse_args()


