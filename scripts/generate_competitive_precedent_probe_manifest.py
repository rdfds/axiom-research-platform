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


