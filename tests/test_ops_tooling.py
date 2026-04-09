from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_causal_rescue_plan as rescue  # noqa: E402
import gate_recommendation_canary as gate  # noqa: E402
import run_manual_replay_benchmark as replay  # noqa: E402
import run_recommendation_prod as prod  # noqa: E402
import train_causal_impact_model as train  # noqa: E402
import train_causal_rescue_model as rescue_train  # noqa: E402


def test_build_causal_rescue_plan_adds_low_row_blocklist_entries():
    audit = {
        "actions": [
            {"action_id": "a.high", "rows": 200, "causal_rate": 0.0, "strict_pass_rate": 0.0},
            {"action_id": "a.low", "rows": 60, "causal_rate": 0.0, "strict_pass_rate": 0.0},
            {"action_id": "a.healthy", "rows": 300, "causal_rate": 1.0, "strict_pass_rate": 1.0},
        ]
    }
    out = rescue.build_causal_rescue_plan(
        audit=audit,
        strict_pass_threshold=0.5,
        min_action_rows=100,
        low_row_blocklist_threshold=50,
    )
    assert [row["action_id"] for row in out["rescue_actions"]] == ["a.high"]
    assert [row["action_id"] for row in out["low_row_blocklist_actions"]] == ["a.low"]
    assert out["suggested_blocklist"] == ["a.high", "a.low"]


