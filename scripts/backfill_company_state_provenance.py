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
    as_of = snapshot.get("as_of_time")
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


def _transform_steps(name: str, fallback_used: Any, missing_reason: Any) -> List[str]:
    steps: List[str] = []
    if name.startswith("liquidity."):
        steps.extend(["extract_latest_facts", "compute_liquidity_metrics"])
    elif name.startswith("capital_structure."):
        steps.extend(["extract_latest_facts", "compute_capital_structure_metrics"])
    elif name.startswith("market."):
        steps.extend(["extract_market_timeseries", "compute_market_metrics"])
    elif name.startswith("operating."):
        steps.extend(["extract_operating_facts", "compute_operating_metrics"])
    elif name.startswith("ownership_governance."):
        steps.extend(["extract_ownership_signals", "compute_ownership_governance_metrics"])
    elif name.startswith("strategic."):
        steps.extend(["extract_strategic_signals", "compute_strategic_metrics"])
    elif name.startswith("peer_context."):
        steps.extend(["resolve_peer_set", "compute_peer_relative_metrics"])
    else:
        steps.append("compute_feature")
    if fallback_used not in (None, "", "none"):
        steps.append(f"fallback:{fallback_used}")
    if missing_reason not in (None, ""):
        steps.append(f"missing:{missing_reason}")
    return steps


def backfill_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    feats = snapshot.get("features", {})
    if not isinstance(feats, dict):
        return snapshot

    inputs = _build_input_refs(snapshot)
    feature_lineage: Dict[str, Dict[str, Any]] = {}

    for name, feat in feats.items():
        if not isinstance(feat, dict):
            continue
        prov = feat.get("provenance")
        if not isinstance(prov, list) or len(prov) == 0:
            feat["provenance"] = _fallback_refs(name, inputs)
        feature_lineage[name] = {
            "inputs": feat.get("provenance", []),
            "transforms": _transform_steps(name, feat.get("fallback_used"), feat.get("missing_reason")),
            "computation_version": "state_builder_v5",
        }

    prov_root = snapshot.get("provenance")
    if not isinstance(prov_root, dict):
        prov_root = {}
        snapshot["provenance"] = prov_root
    prov_root["feature_lineage"] = feature_lineage
    prov_root["computation_version"] = "state_builder_v5_provenance_backfill"
    prov_root["provenance_backfilled_at"] = _now_iso()
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill provenance coverage and feature lineage in snapshot JSONL.")
    parser.add_argument("--in-path", required=True)
    parser.add_argument("--out-path", required=True)
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")

    rows = 0
    with in_path.open("r") as fin, tmp.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            snap = json.loads(line)
            snap = backfill_snapshot(snap)
            fout.write(json.dumps(snap) + "\n")
            rows += 1
    tmp.replace(out_path)
    print(f"Wrote provenance-backfilled snapshots -> {out_path} rows={rows}")


if __name__ == "__main__":
    main()

