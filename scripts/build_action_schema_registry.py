#!/usr/bin/env python
"""
Build and persist ActionSchemaRegistry artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.action_ontology import build_default_action_schema_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ActionSchemaRegistry JSON artifact.")
    parser.add_argument("--version", default="v1.0", help="Registry semantic version.")
    parser.add_argument(
        "--out",
        default="data/action_ontology/action_schema_registry_v1.json",
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip schema and integrity validation before writing.",
    )
    return parser.parse_args()


