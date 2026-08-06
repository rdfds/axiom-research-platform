"""
Text Processing Utilities (MVP)
===============================
Chunking + heuristic signal extraction for unstructured sources.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


DATA_DIR = Path(__file__).parent.parent / "data"
WAREHOUSE_DIR = DATA_DIR / "warehouse"


def ensure_list(value: Optional[object]) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    items: List[object]
    if isinstance(value, list):
        items = value
    elif isinstance(value, (tuple, set)):
        items = list(value)
    else:
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                items = list(value)
            else:
                items = [value]
        except Exception:
            items = [value]
    flat: List[str] = []
    for item in items:
        if item is None or (isinstance(item, float) and pd.isna(item)):
            continue
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
            try:
                import numpy as np

                if isinstance(item, np.ndarray):
                    flat.extend([str(sub) for sub in list(item)])
                else:
                    flat.append(str(item))
            except Exception:
                flat.append(str(item))
    return flat


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

    seek_hits, seek_chunks = hits_for_terms(risk_seek)
    cautious_hits, cautious_chunks = hits_for_terms(risk_cautious)
    if seek_hits + cautious_hits > 0:
        score = 50 + (seek_hits - cautious_hits) * 10
        confidence = min(1.0, 0.2 + 0.15 * (seek_hits + cautious_hits))
        add_signal("management_risk_posture", score, confidence, sorted(set(seek_chunks + cautious_chunks)))

    ret_hits, ret_chunks = hits_for_terms(cap_return)
    reinvest_hits, reinvest_chunks = hits_for_terms(cap_reinvest)
    if ret_hits + reinvest_hits > 0:
        score = 50 + (ret_hits - reinvest_hits) * 10
        confidence = min(1.0, 0.2 + 0.15 * (ret_hits + reinvest_hits))
        add_signal("capital_allocation_intent", score, confidence, sorted(set(ret_chunks + reinvest_chunks)))

    growth_hits, growth_chunks = hits_for_terms(growth_terms)
    def_hits, def_chunks = hits_for_terms(defense_terms)
    if growth_hits + def_hits > 0:
        score = 50 + (growth_hits - def_hits) * 10
        confidence = min(1.0, 0.2 + 0.15 * (growth_hits + def_hits))
        add_signal("growth_vs_defense", score, confidence, sorted(set(growth_chunks + def_chunks)))

    hedge_hits, hedge_chunks = hits_for_terms(hedging_terms)
    if hedge_hits > 0 and token_count > 0:
        ratio = hedge_hits / max(1, token_count)
        score = min(100, ratio * 5000)
        confidence = min(1.0, 0.2 + ratio * 5)
        add_signal("uncertainty_hedging_intensity", score, confidence, hedge_chunks)

    pressure_hits, pressure_chunks = hits_for_terms(pressure_terms)
    if pressure_hits > 0:
        score = min(100, pressure_hits * 20)
        confidence = min(1.0, 0.3 + 0.1 * pressure_hits)
        add_signal("strategic_pressure_indicator", score, confidence, pressure_chunks)

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

    restruct_hits, restruct_chunks = hits_for_terms(restructuring_terms)
    if restruct_hits > 0:
        add_signal("restructuring_signal", 100, 0.8, restruct_chunks)

    mna_hits, mna_chunks = hits_for_terms(mna_terms)
    if mna_hits > 0:
        score = min(100, mna_hits * 25)
        confidence = min(1.0, 0.3 + 0.1 * mna_hits)
        add_signal("mna_intent_signal", score, confidence, mna_chunks)

    liq_hits, liq_chunks = hits_for_terms(liquidity_terms)
    if liq_hits > 0:
        score = min(100, liq_hits * 25)
        confidence = min(1.0, 0.3 + 0.1 * liq_hits)
        add_signal("liquidity_concern_signal", score, confidence, liq_chunks)

    price_hits, price_chunks = hits_for_terms(pricing_terms)
    if price_hits > 0:
        score = min(100, price_hits * 20)
        confidence = min(1.0, 0.3 + 0.1 * price_hits)
        add_signal("pricing_power_signal", score, confidence, price_chunks)

    return signals


def write_partitioned(table: str, records: List[Dict[str, object]], base_dir: Path = WAREHOUSE_DIR) -> int:
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

    out_dir = base_dir / table
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
