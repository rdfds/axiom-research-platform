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


def main() -> None:
    args = parse_args()
    registry = build_default_action_schema_registry(version=args.version)

    if not args.skip_validate:
        schema_errors = registry.validate_schema()
        integrity_errors = registry.validate_registry_integrity()
        all_errors = schema_errors + integrity_errors
        if all_errors:
            print("[error] registry validation failed:")
            for e in all_errors:
                print(f"  - {e}")
            sys.exit(1)

    out_path = registry.write_json(args.out)
    print(f"Wrote ActionSchemaRegistry -> {out_path} actions={len(registry.actions)} version={registry.version}")

    # Quick parse check.
    loaded = json.loads(Path(out_path).read_text())
    print(f"[check] loaded actions={len(loaded.get('actions', []))}")


if __name__ == "__main__":
    main()

