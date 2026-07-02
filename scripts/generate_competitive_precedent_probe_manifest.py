#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _score_actions(candidate_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    action_scores: Dict[str, float] = {}
    for row in candidate_rows:
        action_id = str(row['action_id'] or "").strip()
        if not action_id:
            continue
        confidence = row.get("precedent_confidence")
        if confidence is None:
            continue
        score = float(confidence)
        if action_id not in action_scores or score > action_scores[action_id]:
            action_scores[action_id] = score
    return action_scores


def _action_family(action_id: str) -> str:
    return action_id.split(".", 1)[0] if "." in action_id else action_id


def _analyze_case(
    *,
    case: Dict[str, Any],
    precedent_index: Dict[str, Any],
) -> Dict[str, Any]:
    action_scores = _score_actions(list(precedent_index.get("candidate_rows", []) or []))
    sorted_actions = sorted(action_scores.items(), key=lambda item: (-item[1], item[0]))
    action_rank_lookup = {action_id: index + 1 for index, (action_id, _) in enumerate(sorted_actions)}

    anchor_action_id = str(case["anchor_action_id"])
    anchor_family = str(case["anchor_action_family"])
    family_scores: Dict[str, float] = {}
    for action_id, score in sorted_actions:
        family = _action_family(action_id)
        if family not in family_scores:
            family_scores[family] = score

    anchor_score = action_scores.get(anchor_action_id)
    anchor_rank = action_rank_lookup.get(anchor_action_id)
    family_present = anchor_family in family_scores
    best_other_action_score = max(
        (score for action_id, score in sorted_actions if action_id != anchor_action_id),
        default=None,
    )
    anchor_margin = (
        None
        if anchor_score is None or best_other_action_score is None
        else float(anchor_score) - float(best_other_action_score)
    )

    return {
        "company_id": case["company_id"],
        "source_company_id": case.get("source_company_id", case["company_id"]),
        "ticker": case.get("ticker", ""),
        "mapping_method": case.get("mapping_method"),
        "anchor_action_id": anchor_action_id,
        "anchor_action_family": anchor_family,
        "anchor_action_date": case.get("anchor_action_date"),
        "as_of_time": case.get("as_of_time"),
        "distinct_action_count": len(sorted_actions),
        "distinct_action_ids": [action_id for action_id, _ in sorted_actions],
        "anchor_action_present": anchor_action_id in action_scores,
        "anchor_family_present": family_present,
        "anchor_action_rank": anchor_rank,
        "anchor_action_score": anchor_score,
        "anchor_action_margin": anchor_margin,
        "top_action_id": sorted_actions[0][0] if sorted_actions else None,
        "top_action_score": sorted_actions[0][1] if sorted_actions else None,
    }


