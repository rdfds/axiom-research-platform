from src.backtest_costs import resolve_transaction_cost_model
from src.backtest_protocol import BacktestProtocol
from src.backtest_scorecard import (
    build_portfolio_strategy_scorecard,
    render_portfolio_strategy_scorecard_markdown,
)


def _report_fixture():
    return {
        "reference_metrics": {
            "mean_alignment_score": 0.80,
            "anchor_primary_exact_rate": 0.25,
            "anchor_primary_family_rate": 0.80,
            "unsupported_case_count": 0,
        },
        "cases": [
            {
                "company_id": "A",
                "source_company_id": "A",
                "top_action_ids": ["capital_return.open_market_buyback"],
                "recommended_action_support": [{"support_mode": "exact_supported"}],
                "historical_alignment": {"score": 1.0, "reason": "anchor_primary_exact"},
            },
            {
                "company_id": "B",
                "source_company_id": "B",
                "anchor_action_family": "capital_structure",
                "top_action_ids": ["capital_structure.refinancing"],
                "recommended_action_support": [{"support_mode": "family_supported"}],
                "historical_alignment": {"score": 0.85, "reason": "future_family_support_adjusted"},
            },
            {
                "company_id": "C",
                "source_company_id": "C",
                "unsupported_reason": "insufficient_snapshot_coverage",
            },
            {
                "company_id": "D",
                "source_company_id": "D",
                "error": "boom",
            },
        ],
    }


