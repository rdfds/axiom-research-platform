#!/usr/bin/env python
"""
Build an empty PrivateOverlayRegistry (placeholder).

This is a scaffold for private inputs like covenants, projections,
and board constraints. It creates a schema‑compliant empty parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="data/inputs_layer/private_overlay_registry.parquet",
        help="Output PrivateOverlayRegistry parquet.",
    )
    args = parser.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Minimal schema‑compliant empty frame
    df = pd.DataFrame(
        columns=[
            "overlay_id",
            "entity_id",
            "overlay_type",
            "overlay_version",
            "author_id",
            "created_at",
            "expires_at",
            "payload",
            "source_id",
            "source_type",
            "published_at",
            "effective_at",
            "ingested_at",
            "confidence_score",
            "raw_pointer",
        ]
    )
    df.to_parquet(out_path, index=False)
    print(f"Saved PrivateOverlayRegistry -> {out_path} (0 rows)")


if __name__ == "__main__":
    main()
