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


def test_build_portfolio_strategy_scorecard_adds_cost_adjusted_net_metrics():
    protocol = BacktestProtocol(
        key="test",
        label="Test",
        min_case_count=2,
    )
    model = resolve_transaction_cost_model("manual_replay_event_equal_weight_v1")

    scorecard = build_portfolio_strategy_scorecard(
        _report_fixture(),
        protocol=protocol,
        cost_model=model,
    )

    proxy = scorecard["portfolio_proxy"]
    assert scorecard["case_counts"]["scored_cases"] == 2
    assert scorecard["case_counts"]["unsupported_cases"] == 1
    assert scorecard["case_counts"]["error_cases"] == 1
    assert proxy["gross_mean_alignment_score"] > proxy["net_mean_alignment_score"]
    assert scorecard["coverage"]["recommended_family_counts"] == {
        "capital_return": 1,
        "capital_structure": 1,
    }
    assert scorecard["benchmark_comparison"]["delta_unsupported_case_count"] == 1


def test_render_portfolio_strategy_scorecard_markdown_includes_flags_section():
    protocol = BacktestProtocol(
        key="test",
        label="Test",
        min_case_count=10,
    )
    model = resolve_transaction_cost_model("manual_replay_event_equal_weight_v1")
    scorecard = build_portfolio_strategy_scorecard(
        _report_fixture(),
        protocol=protocol,
        cost_model=model,
    )

    markdown = render_portfolio_strategy_scorecard_markdown(scorecard)

    assert "# Canonical Backtest Scorecard" in markdown
    assert "`insufficient_scored_cases`" in markdown
