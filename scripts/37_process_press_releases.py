#!/usr/bin/env python
"""
Chunk press releases (B4) + extract text signals into canonical tables.

Reads:
  data/warehouse/warehouse_press_releases/year=*/part_*.parquet

Writes:
  data/warehouse/warehouse_documents/year=*/part_*.parquet
  data/warehouse/warehouse_doc_chunks/year=*/part_*.parquet
  data/warehouse/warehouse_text_signals/year=*/part_*.parquet

Env:
  PR_START_YEAR=2000
  PR_END_YEAR=YYYY
  PR_LIMIT_DOCS=0 (0 = no limit)
  PR_FLUSH_EVERY=200 (docs per flush)
  PR_CHUNK_TOKENS=400
  PR_CHUNK_MIN=300
  PR_CHUNK_MAX=500
  PR_RESUME=1
  PR_START_FILE_INDEX=0
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import compute_raw_payload_hash, compute_version_id


DATA_DIR = Path(__file__).parent.parent / "data"
WAREHOUSE_DIR = DATA_DIR / "warehouse"
SEC_DIR = DATA_DIR / "sec"

PR_START_YEAR = int(os.getenv("PR_START_YEAR", "2000"))
PR_END_YEAR = int(os.getenv("PR_END_YEAR", datetime.utcnow().year))
PR_LIMIT_DOCS = int(os.getenv("PR_LIMIT_DOCS", "0"))
PR_FLUSH_EVERY = int(os.getenv("PR_FLUSH_EVERY", "200"))
PR_CHUNK_TOKENS = int(os.getenv("PR_CHUNK_TOKENS", "400"))
PR_CHUNK_MIN = int(os.getenv("PR_CHUNK_MIN", "300"))
PR_CHUNK_MAX = int(os.getenv("PR_CHUNK_MAX", "500"))
PR_RESUME = os.getenv("PR_RESUME", "1") == "1"
PR_START_FILE_INDEX = int(os.getenv("PR_START_FILE_INDEX", "0"))
PR_REQUIRE_TEXT = os.getenv("PR_REQUIRE_TEXT", "0") == "1"


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def ensure_list(value: Optional[object]) -> List[str]:
    if value is None:
        return []

    try:
        import numpy as np
    except Exception:
        np = None  # type: ignore

    try:
        from pandas.api.types import is_scalar
    except Exception:
        def is_scalar(_):  # type: ignore
            return isinstance(_, (str, int, float, bool))

    if np is not None and isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        items: List[object] = list(value)
    elif isinstance(value, pd.Series):
        if value.empty:
            return []
        items = value.tolist()
    else:
        if is_scalar(value):
            try:
                if pd.isna(value):
                    return []
            except Exception:
                pass
        if isinstance(value, list):
            items = value
        elif isinstance(value, (tuple, set)):
            items = list(value)
        else:
            items = [value]

    flat: List[str] = []
    for item in items:
        if item is None:
            continue
        if np is not None and isinstance(item, np.ndarray):
            if item.size == 0:
                continue
            for sub in item.tolist():
                if sub is None:
                    continue
                flat.append(str(sub))
            continue
        if isinstance(item, pd.Series):
            if item.empty:
                continue
            for sub in item.tolist():
                if sub is None:
                    continue
                flat.append(str(sub))
            continue
        if is_scalar(item):
            try:
                if pd.isna(item):
                    continue
            except Exception:
                pass
        if isinstance(item, list):
            for sub in item:
                if sub is None or (isinstance(sub, float) and pd.isna(sub)):
                    continue
                flat.append(str(sub))
        elif isinstance(item, (tuple, set)):
            flat.extend([str(sub) for sub in item])
        else:
            flat.append(str(item))
    return flat


def iter_press_release_files() -> List[Path]:
    base = WAREHOUSE_DIR / "warehouse_press_releases"
    if not base.exists():
        raise FileNotFoundError(f"Missing press releases: {base}")
    files = sorted(base.glob("year=*/part_*.parquet"))
    filtered: List[Path] = []
    for path in files:
        try:
            year = int(path.parent.name.split("=")[1])
        except Exception:
            continue
        if year < PR_START_YEAR or year > PR_END_YEAR:
            continue
        filtered.append(path)
    return filtered


def load_checkpoint(path: Path) -> set:
    processed: set = set()
    try:
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    processed.add(line)
    except Exception as exc:
        log(f"Checkpoint read failed ({path}): {exc}. Using {len(processed):,} entries.")
    return processed


def chunk_text(text: str, target: int, min_tokens: int, max_tokens: int) -> List[str]:
    tokens = text.split()
    if not tokens:
        return []
    if len(tokens) <= max_tokens:
        return [" ".join(tokens)]
    chunks = []
    idx = 0
    while idx < len(tokens):
        end = min(idx + max_tokens, len(tokens))
        chunk = tokens[idx:end]
        if len(chunk) < min_tokens and chunks:
            chunks[-1] = f"{chunks[-1]} {' '.join(chunk)}"
            break
        chunks.append(" ".join(chunk))
        idx = end
    return chunks


def _count_hits(text: str, patterns: Iterable[str]) -> int:
    count = 0
    for pattern in patterns:
        if pattern in text:
            count += 1
    return count


def extract_signals(
    chunks: List[Tuple[str, str]]
) -> List[Dict[str, object]]:
    """
    chunks: list of (chunk_id, chunk_text)
    Returns signal dicts with supporting_chunk_ids.
    """
    signals: List[Dict[str, object]] = []
    if not chunks:
        return signals

    # Precompute lowercase text per chunk
    chunk_texts = [(cid, text.lower()) for cid, text in chunks]
    all_text = " ".join(text for _, text in chunk_texts)
    token_count = len(all_text.split())

    def add_signal(name: str, value: float, confidence: float, supporting: List[str], extra_flags: Optional[List[str]] = None) -> None:
        if not supporting:
            return
        signals.append(
            {
                "signal_name": name,
                "value": float(max(0, min(100, value))),
                "confidence": float(max(0, min(1, confidence))),
                "supporting_chunk_ids": supporting,
                "quality_flags": extra_flags or [],
            }
        )

    # Keyword sets
    risk_seek = [
        "aggressive",
        "bold",
        "accelerate expansion",
        "rapid growth",
        "opportunistic acquisition",
        "invest heavily",
        "growth initiatives",
        "expansion plan",
    ]
    risk_cautious = [
        "conservative",
        "cautious",
        "risk mitigation",
        "prudently",
        "defensive",
        "cost discipline",
        "headwinds",
    ]
    cap_return = [
        "share repurchase",
        "buyback",
        "return capital",
        "dividend",
        "capital return",
        "repurchase program",
    ]
    cap_reinvest = [
        "reinvest",
        "capital expenditure",
        "capex",
        "growth investment",
        "deleveraging",
        "pay down debt",
    ]
    growth_terms = [
        "growth",
        "expand",
        "accelerate",
        "scale",
        "investment",
        "new market",
        "increase capacity",
    ]
    defense_terms = [
        "cost reduction",
        "cost cutting",
        "restructuring",
        "defensive",
        "protect margins",
        "headwinds",
        "slowdown",
    ]
    hedging_terms = [
        "may",
        "might",
        "could",
        "uncertain",
        "subject to",
        "potentially",
        "expect",
        "anticipate",
    ]
    pressure_terms = [
        "competitive pressure",
        "regulatory",
        "activist",
        "litigation",
        "headwinds",
        "supply chain disruption",
    ]
    guidance_up = [
        "raise guidance",
        "increased guidance",
        "raise outlook",
        "increased outlook",
        "upwardly revise",
    ]
    guidance_down = [
        "lower guidance",
        "reduced guidance",
        "lowered outlook",
        "downwardly revise",
    ]
    guidance_reaffirm = [
        "reaffirm guidance",
        "maintain guidance",
        "reiterate guidance",
        "unchanged guidance",
    ]
    restructuring_terms = [
        "restructuring",
        "layoff",
        "workforce reduction",
        "reorganization",
        "cost savings program",
    ]
    mna_terms = [
        "strategic alternatives",
        "acquisition",
        "merger",
        "m&a",
        "divestiture",
        "sale of",
        "exploring a sale",
        "buyout",
    ]
    liquidity_terms = [
        "liquidity",
        "covenant",
        "going concern",
        "cash runway",
        "financing",
        "credit facility",
    ]
    pricing_terms = [
        "pricing power",
        "price increase",
        "price increases",
        "pricing actions",
        "promotional",
        "discounting",
    ]

    def hits_for_terms(terms: List[str]) -> Tuple[int, List[str]]:
        hits = 0
        supporting = []
        for cid, text in chunk_texts:
            cnt = _count_hits(text, terms)
            if cnt:
                hits += cnt
                supporting.append(cid)
        return hits, supporting

    # Management risk posture
    seek_hits, seek_chunks = hits_for_terms(risk_seek)
    cautious_hits, cautious_chunks = hits_for_terms(risk_cautious)
    if seek_hits + cautious_hits > 0:
        score = 50 + (seek_hits - cautious_hits) * 10
        confidence = min(1.0, 0.2 + 0.15 * (seek_hits + cautious_hits))
        add_signal(
            "management_risk_posture",
            score,
            confidence,
            sorted(set(seek_chunks + cautious_chunks)),
        )

    # Capital allocation intent
    ret_hits, ret_chunks = hits_for_terms(cap_return)
    reinvest_hits, reinvest_chunks = hits_for_terms(cap_reinvest)
    if ret_hits + reinvest_hits > 0:
        score = 50 + (ret_hits - reinvest_hits) * 10
        confidence = min(1.0, 0.2 + 0.15 * (ret_hits + reinvest_hits))
        add_signal(
            "capital_allocation_intent",
            score,
            confidence,
            sorted(set(ret_chunks + reinvest_chunks)),
        )

    # Growth vs defense
    growth_hits, growth_chunks = hits_for_terms(growth_terms)
    def_hits, def_chunks = hits_for_terms(defense_terms)
    if growth_hits + def_hits > 0:
        score = 50 + (growth_hits - def_hits) * 10
        confidence = min(1.0, 0.2 + 0.15 * (growth_hits + def_hits))
        add_signal(
            "growth_vs_defense",
            score,
            confidence,
            sorted(set(growth_chunks + def_chunks)),
        )

    # Uncertainty / hedging intensity
    hedge_hits, hedge_chunks = hits_for_terms(hedging_terms)
    if hedge_hits > 0 and token_count > 0:
        ratio = hedge_hits / max(1, token_count)
        score = min(100, ratio * 5000)
        confidence = min(1.0, 0.2 + ratio * 5)
        add_signal(
            "uncertainty_hedging_intensity",
            score,
            confidence,
            hedge_chunks,
        )

    # Strategic pressure
    pressure_hits, pressure_chunks = hits_for_terms(pressure_terms)
    if pressure_hits > 0:
        score = min(100, pressure_hits * 20)
        confidence = min(1.0, 0.3 + 0.1 * pressure_hits)
        add_signal(
            "strategic_pressure_indicator",
            score,
            confidence,
            pressure_chunks,
        )

    # Guidance change
    up_hits, up_chunks = hits_for_terms(guidance_up)
    down_hits, down_chunks = hits_for_terms(guidance_down)
    re_hits, re_chunks = hits_for_terms(guidance_reaffirm)
    guidance_support = sorted(set(up_chunks + down_chunks + re_chunks))
    if up_hits + down_hits + re_hits > 0:
        flags: List[str] = []
        if up_hits and down_hits:
            score = 50
            flags.append("source_conflict")
        elif up_hits:
            score = 100
        elif down_hits:
            score = 0
        else:
            score = 50
        confidence = min(1.0, 0.4 + 0.2 * (up_hits + down_hits + re_hits))
        add_signal("guidance_change", score, confidence, guidance_support, flags)

    # Restructuring
    restruct_hits, restruct_chunks = hits_for_terms(restructuring_terms)
    if restruct_hits > 0:
        add_signal("restructuring_signal", 100, 0.8, restruct_chunks)

    # M&A intent
    mna_hits, mna_chunks = hits_for_terms(mna_terms)
    if mna_hits > 0:
        score = min(100, mna_hits * 25)
        confidence = min(1.0, 0.3 + 0.1 * mna_hits)
        add_signal("mna_intent_signal", score, confidence, mna_chunks)

    # Liquidity concern
    liq_hits, liq_chunks = hits_for_terms(liquidity_terms)
    if liq_hits > 0:
        score = min(100, liq_hits * 25)
        confidence = min(1.0, 0.3 + 0.1 * liq_hits)
        add_signal("liquidity_concern_signal", score, confidence, liq_chunks)

    # Pricing power
    price_hits, price_chunks = hits_for_terms(pricing_terms)
    if price_hits > 0:
        score = min(100, price_hits * 20)
        confidence = min(1.0, 0.3 + 0.1 * price_hits)
        add_signal("pricing_power_signal", score, confidence, price_chunks)

    return signals


def write_partitioned(table: str, records: List[Dict[str, object]]) -> int:
    if not records:
        return 0
    df = pd.DataFrame(records)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["available_time"] = pd.to_datetime(df["available_time"], errors="coerce")
    df["ingestion_time"] = pd.to_datetime(df["ingestion_time"], errors="coerce")
    for col in ["quality_flags", "upstream_version_ids", "supporting_chunk_ids"]:
        if col in df.columns:
            df[col] = df[col].apply(ensure_list)
    df["year"] = df["event_time"].dt.year.astype("Int64")

    out_dir = WAREHOUSE_DIR / table
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = 0
    for year, ydf in df.groupby("year"):
        if pd.isna(year):
            continue
        year_dir = out_dir / f"year={int(year)}"
        year_dir.mkdir(parents=True, exist_ok=True)
        part_path = year_dir / f"part_{int(datetime.utcnow().timestamp())}_{os.getpid()}.parquet"
        ydf.drop(columns=["year"]).to_parquet(part_path, index=False)
        rows += len(ydf)
    return rows


def main() -> None:
    files = iter_press_release_files()
    if PR_START_FILE_INDEX:
        files = files[PR_START_FILE_INDEX:]
    if not files:
        log("No press release files found.")
        return

    checkpoint_path = SEC_DIR / "press_release_chunk_checkpoint.txt"
    processed = set()
    if PR_RESUME and checkpoint_path.exists():
        processed = load_checkpoint(checkpoint_path)

    total_docs = 0
    total_chunks = 0
    total_signals = 0
    start_ts = time.perf_counter()

    doc_buffer: List[Dict[str, object]] = []
    chunk_buffer: List[Dict[str, object]] = []
    signal_buffer: List[Dict[str, object]] = []

    ingestion_time = datetime.utcnow()

    for fidx, path in enumerate(files, start=1):
        log(f"Processing file {fidx}/{len(files)}: {path.name}")
        try:
            if path.stat().st_size == 0:
                log(f"Skipping empty parquet: {path}")
                continue
            df = pd.read_parquet(path)
        except Exception as exc:
            log(f"Skipping unreadable parquet {path}: {exc}")
            continue
        if df.empty:
            continue

        for _, row in df.iterrows():
            source_system = row.get("source_system") or "sec_8k_press_release"
            document_id = row.get("document_id") or row.get("accession")
            if not document_id or (isinstance(document_id, float) and pd.isna(document_id)):
                fallback_entity = row.get("cik") or row.get("entity_id")
                fallback_date = pd.to_datetime(row.get("event_time"), errors="coerce")
                fallback_date_str = fallback_date.strftime("%Y%m%d") if pd.notna(fallback_date) else "unknown"
                document_id = f"pr-{source_system}-{fallback_entity}-{fallback_date_str}"

            if PR_RESUME and document_id in processed:
                continue

            text = row.get("text")
            if isinstance(text, float) and pd.isna(text):
                text = None
            headline = row.get("headline")
            event_time = pd.to_datetime(row.get("event_time"), errors="coerce")
            available_time = pd.to_datetime(row.get("available_time"), errors="coerce")
            if pd.isna(event_time) or pd.isna(available_time):
                continue

            base_flags = ensure_list(row.get("quality_flags"))
            if not text:
                if "missing_data" not in base_flags:
                    base_flags.append("missing_data")
                if "partial_coverage" not in base_flags:
                    base_flags.append("partial_coverage")
                if PR_REQUIRE_TEXT:
                    continue

            doc_payload = {
                "document_id": document_id,
                "document_type": "press_release",
                "title": headline,
                "release_date": row.get("release_date"),
                "text": text,
            }
            doc_raw_hash = compute_raw_payload_hash(doc_payload)
            doc_version_id = compute_version_id(
                source_system=source_system,
                entity_id=str(row.get("entity_id")),
                event_time=event_time.to_pydatetime(),
                available_time=available_time.to_pydatetime(),
                raw_payload_hash=doc_raw_hash,
            )

            doc_record = {
                "source_system": source_system,
                "entity_id": str(row.get("entity_id")),
                "company_id": str(row.get("company_id")),
                "security_id": None,
                "event_time": event_time,
                "available_time": available_time,
                "ingestion_time": ingestion_time,
                "version_id": doc_version_id,
                "raw_payload_hash": doc_raw_hash,
                "upstream_version_ids": ensure_list(row.get("version_id")),
                "quality_flags": base_flags,
                "document_id": document_id,
                "document_type": "press_release",
                "title": headline,
                "publisher": None,
                "analyst": None,
                "rating": None,
                "price_target": None,
                "call_date": None,
                "publish_date": None,
                "presentation_date": None,
                "release_date": row.get("release_date"),
                "source_url": None,
            }

            doc_buffer.append(doc_record)
            total_docs += 1

            chunk_ids: List[Tuple[str, str]] = []
            chunk_versions: Dict[str, str] = {}
            if text:
                chunks = chunk_text(text, PR_CHUNK_TOKENS, PR_CHUNK_MIN, PR_CHUNK_MAX)
                for idx, chunk in enumerate(chunks):
                    chunk_id = f"{document_id}::chunk{idx:04d}"
                    token_count = len(chunk.split())
                    chunk_payload = {
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "chunk_index": idx,
                        "text": chunk,
                    }
                    chunk_raw_hash = compute_raw_payload_hash(chunk_payload)
                    chunk_version_id = compute_version_id(
                        source_system=source_system,
                        entity_id=str(row.get("entity_id")),
                        event_time=event_time.to_pydatetime(),
                        available_time=available_time.to_pydatetime(),
                        raw_payload_hash=chunk_raw_hash,
                    )
                    chunk_versions[chunk_id] = chunk_version_id
                    chunk_ids.append((chunk_id, chunk))
                    chunk_buffer.append(
                        {
                            "source_system": source_system,
                            "entity_id": str(row.get("entity_id")),
                            "company_id": str(row.get("company_id")),
                            "security_id": None,
                            "event_time": event_time,
                            "available_time": available_time,
                            "ingestion_time": ingestion_time,
                            "version_id": chunk_version_id,
                            "raw_payload_hash": chunk_raw_hash,
                            "upstream_version_ids": [doc_version_id],
                            "quality_flags": base_flags,
                            "chunk_id": chunk_id,
                            "document_id": document_id,
                            "chunk_index": idx,
                            "slide_number": None,
                            "text": chunk,
                            "speaker": None,
                            "speaker_role": None,
                            "section_type": "press_release",
                            "token_count": token_count,
                        }
                    )
                total_chunks += len(chunks)

            if chunk_ids:
                signal_defs = extract_signals(chunk_ids)
                for sig in signal_defs:
                    supporting = sig.get("supporting_chunk_ids", [])
                    supporting_versions = [chunk_versions[cid] for cid in supporting if cid in chunk_versions]
                    signal_payload = {
                        "signal_name": sig["signal_name"],
                        "value": sig["value"],
                        "confidence": sig["confidence"],
                        "supporting_chunk_ids": supporting,
                    }
                    sig_raw_hash = compute_raw_payload_hash(signal_payload)
                    sig_version_id = compute_version_id(
                        source_system=source_system,
                        entity_id=str(row.get("entity_id")),
                        event_time=event_time.to_pydatetime(),
                        available_time=available_time.to_pydatetime(),
                        raw_payload_hash=sig_raw_hash,
                    )
                    signal_buffer.append(
                        {
                            "source_system": source_system,
                            "entity_id": str(row.get("entity_id")),
                            "company_id": str(row.get("company_id")),
                            "security_id": None,
                            "event_time": event_time,
                            "available_time": available_time,
                            "ingestion_time": ingestion_time,
                            "version_id": sig_version_id,
                            "raw_payload_hash": sig_raw_hash,
                            "upstream_version_ids": [doc_version_id] + supporting_versions,
                            "quality_flags": base_flags + ensure_list(sig.get("quality_flags")),
                            "signal_name": sig["signal_name"],
                            "value": sig["value"],
                            "confidence": sig["confidence"],
                            "supporting_chunk_ids": supporting,
                        }
                    )
                total_signals += len(signal_defs)

            if PR_RESUME:
                with checkpoint_path.open("a") as f:
                    f.write(f"{document_id}\n")

            if PR_LIMIT_DOCS and total_docs >= PR_LIMIT_DOCS:
                break

            if PR_FLUSH_EVERY and (total_docs % PR_FLUSH_EVERY == 0):
                write_partitioned("warehouse_documents", doc_buffer)
                write_partitioned("warehouse_doc_chunks", chunk_buffer)
                write_partitioned("warehouse_text_signals", signal_buffer)
                doc_buffer.clear()
                chunk_buffer.clear()
                signal_buffer.clear()
                elapsed = time.perf_counter() - start_ts
                log(
                    f"Progress: {total_docs:,} docs | {total_chunks:,} chunks | {total_signals:,} signals | elapsed {elapsed/60:.1f}m"
                )

        if PR_LIMIT_DOCS and total_docs >= PR_LIMIT_DOCS:
            break

    # Flush remaining
    write_partitioned("warehouse_documents", doc_buffer)
    write_partitioned("warehouse_doc_chunks", chunk_buffer)
    write_partitioned("warehouse_text_signals", signal_buffer)

    elapsed = time.perf_counter() - start_ts
    log(
        f"Done. {total_docs:,} docs | {total_chunks:,} chunks | {total_signals:,} signals | elapsed {elapsed/60:.1f}m"
    )


if __name__ == "__main__":
    main()
