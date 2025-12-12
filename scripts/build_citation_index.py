#!/usr/bin/env python
"""
Build citation index from ExtractedFactRegistry (enriched) + doc chunks.

Uses chunk text to create offset spans (0..len(text)) and excerpt text.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts-root", default="data/inputs_layer/extracted_fact_registry_enriched")
    parser.add_argument("--chunks-root", default="data/warehouse/warehouse_doc_chunks")
    parser.add_argument("--out", default="data/inputs_layer/citation_index")
    parser.add_argument("--years", default=None, help="Comma-separated years (default: all).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--excerpt_len", type=int, default=300)
    args = parser.parse_args()

    facts_root = ROOT / args.facts_root
    chunks_root = ROOT / args.chunks_root
    out_path = ROOT / args.out
    out_path.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.overwrite:
        # allow incremental year builds
        pass

    if args.years:
        years = [y.strip() for y in args.years.split(",") if y.strip()]
    else:
        years = sorted({p.name.split('=')[1] for p in facts_root.glob('year=*')})

    citations = []
    for y in years:
        facts_path = facts_root / f"year={y}"
        chunks_path = chunks_root / f"year={y}"
        if not facts_path.exists() or not chunks_path.exists():
            continue

        year_out_dir = out_path / f"year={y}"
        year_out_dir.mkdir(parents=True, exist_ok=True)
        year_out_file = year_out_dir / "part.parquet"
        if year_out_file.exists() and not args.overwrite:
            continue

        facts = pd.read_parquet(facts_path)
        if facts.empty or "citation_span" not in facts.columns:
            continue
        facts = facts[facts["citation_span"].notna()].copy()
        if facts.empty:
            continue
        needed_chunks = set(facts["citation_span"].astype("string"))

        # Read chunks safely (skip corrupt parquet parts)
        chunk_files = sorted(chunks_path.glob("*.parquet"))
        chunks_parts = []
        for f in chunk_files:
            try:
                part = pd.read_parquet(
                    f,
                    columns=["chunk_id", "document_id", "text", "source_system", "ingestion_time"],
                )
                if not part.empty:
                    part = part[part["chunk_id"].astype("string").isin(needed_chunks)]
                    if not part.empty:
                        chunks_parts.append(part)
            except Exception:
                continue
        if not chunks_parts:
            continue
        chunks = pd.concat(chunks_parts, ignore_index=True)

        # join on chunk_id
        merged = facts.merge(
            chunks[["chunk_id", "document_id", "text", "source_system", "ingestion_time"]],
            left_on="citation_span",
            right_on="chunk_id",
            how="left",
        )

        # resolve document_id (prefer fact's document_id if present)
        if "document_id_x" in merged.columns:
            merged["document_id_resolved"] = merged["document_id_x"].combine_first(merged.get("document_id_y"))
        else:
            merged["document_id_resolved"] = merged.get("document_id")

        merged["start_char_offset"] = 0
        merged["end_char_offset"] = merged["text"].fillna("").str.len()
        merged["excerpt_text"] = merged["text"].fillna("").str.slice(0, args.excerpt_len)

        out_df = pd.DataFrame(
            {
                "citation_id": merged["fact_id"].astype("string") + "::" + merged["chunk_id"].astype("string"),
                "fact_id": merged["fact_id"].astype("string"),
                "document_id": merged["document_id_resolved"].astype("string"),
                "chunk_id": merged["chunk_id"].astype("string"),
                "start_char_offset": merged["start_char_offset"],
                "end_char_offset": merged["end_char_offset"],
                "excerpt_text": merged["excerpt_text"],
                "source_id": merged.get("source_system"),
                "source_type": merged.get("source_system"),
                "ingested_at": pd.to_datetime(merged.get("ingestion_time"), errors="coerce", utc=True),
            }
        )
        out_df.to_parquet(year_out_file, index=False)
        print(f"Wrote CitationIndex year={y} -> {year_out_file} ({len(out_df)} rows)")

    print(f"Saved CitationIndex -> {out_path}")


if __name__ == "__main__":
    main()
