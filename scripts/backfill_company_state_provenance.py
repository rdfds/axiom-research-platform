#!/usr/bin/env python
"""
Backfill provenance coverage + transform lineage for existing CompanyState JSONL snapshots
without rebuilding features.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_ref(
    artifact_type: str,
    artifact_id: str,
    source: str | None,
    as_of_time: str | None,
) -> Dict[str, Any]:
    return {
        "artifact_type": artifact_type,
        "artifact_id": artifact_id,
        "source": source,
        "published_at": as_of_time,
        "ingested_at": as_of_time,
        "hash": None,
    }


def _build_input_refs(snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    prov = snapshot.get("provenance", {}) if isinstance(snapshot.get("provenance"), dict) else {}
    inputs = prov.get("inputs_used", {}) if isinstance(prov.get("inputs_used"), dict) else {}
    as_of = snapshot['as_of_time']
    return {
        "facts": _default_ref("ExtractedFact", "facts:scan", str(inputs.get("facts")) if inputs.get("facts") is not None else None, as_of),
        "timeseries": _default_ref("RawTimeseries", "timeseries:scan", str(inputs.get("timeseries")) if inputs.get("timeseries") is not None else None, as_of),
        "macro": _default_ref("RawTimeseries", "macro:scan", str(inputs.get("macro")) if inputs.get("macro") is not None else None, as_of),
        "events": _default_ref("Event", "events:scan", str(inputs.get("events")) if inputs.get("events") is not None else None, as_of),
        "ownership": _default_ref("RawDocument", "ownership:scan", str(inputs.get("ownership")) if inputs.get("ownership") is not None else None, as_of),
        "issuer_ratings": _default_ref("ExtractedFact", "issuer_ratings:scan", str(inputs.get("issuer_ratings")) if inputs.get("issuer_ratings") is not None else None, as_of),
        "entity": _default_ref("RawDocument", "entity:scan", str(inputs.get("entity")) if inputs.get("entity") is not None else None, as_of),
    }


def _fallback_refs(name: str, inputs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    if name.startswith("liquidity."):
        return [inputs["facts"]]
    if name.startswith("capital_structure."):
        out = [inputs["facts"]]
        if "rating" in name:
            out.extend([inputs["issuer_ratings"], inputs["events"]])
        if "debt_due_" in name or "maturity" in name or "refi" in name:
            out.append(inputs["events"])
        return out
    if name.startswith("market."):
        out = [inputs["timeseries"], inputs["facts"]]
        if "window_proxy" in name:
            out.append(inputs["macro"])
        return out
    if name.startswith("operating."):
        out = [inputs["facts"]]
        if "cyclicality" in name:
            out.append(inputs["macro"])
        return out
    if name.startswith("ownership_governance."):
        return [inputs["ownership"], inputs["events"], inputs["facts"]]
    if name.startswith("strategic."):
        return [inputs["facts"], inputs["events"]]
    if name.startswith("peer_context."):
        return [inputs["entity"], inputs["events"], inputs["facts"]]
    return [inputs["facts"]]


