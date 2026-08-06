from __future__ import annotations

import pandas as pd

from src.action_data_support import build_action_support_report, resolve_action_support


def test_build_action_support_report_distinguishes_exact_family_only_and_unsupported(tmp_path):
    outcomes_path = tmp_path / "outcomes.parquet"
    frame = pd.DataFrame(
        [
            {
                "normalized_action_family": "capital_return",
                "normalized_action_id": "capital_return.open_market_buyback",
            },
            {
                "normalized_action_family": "capital_structure",
                "normalized_action_id": None,
            },
        ]
    )
    frame.to_parquet(outcomes_path, index=False)

    report = build_action_support_report(
        outcomes_path=outcomes_path,
        relevant_action_ids=[
            "capital_return.open_market_buyback",
            "capital_structure.refinancing",
            "governance.board_refresh",
        ],
    )

    actions = {item["action_id"]: item for item in report["relevant_actions"]}
    assert actions["capital_return.open_market_buyback"]["support_mode"] == "exact_supported"
    assert actions["capital_return.open_market_buyback"]["exact_support_status"] == "thin"
    assert actions["capital_structure.refinancing"]["support_mode"] == "family_only"
    assert actions["governance.board_refresh"]["support_mode"] == "unsupported"


def test_resolve_action_support_falls_back_to_family_only_when_exact_action_missing():
    support_report = {
        "family_counts": {"capital_structure": 10},
        "relevant_actions": [],
    }

    resolved = resolve_action_support(
        action_id="capital_structure.exchange_offer",
        action_family="capital_structure",
        support_report=support_report,
    )

    assert resolved["support_mode"] == "family_only"
    assert resolved["family_count"] == 10
    assert resolved["exact_support_status"] == "missing"
