from __future__ import annotations

from src.planner_types import (
    Plan,
    PlanBranch,
    PlanExplanation,
    PlanRisk,
    PlanScoreBreakdown,
    PlanStep,
    PlanTimeline,
    PlanTrigger,
)


def test_plan_to_dict_preserves_explanation_and_scores():
    plan = Plan(
        plan_id="plan-1",
        run_id="run-1",
        steps=[
            PlanStep(
                step_id="step-1",
                action_id="capital_return.open_market_buyback",
                parameters={"size_pct_market_cap": 0.05},
                earliest_start="2026-04-01",
                expected_duration={"minimum_days": 1, "median_days": 30},
                prerequisites=["capital_structure.refinancing"],
                probability_of_success=0.82,
                impact_contribution={"value_creation": 0.12, "risk_reduction": 0.03},
                explanation=PlanExplanation(
                    problem_statement="Excess capital is underutilized.",
                    why_this_action="Repurchases improve per-share capital efficiency.",
                    why_now="Debt maturities are manageable after refinancing.",
                    key_supporting_facts=["Liquidity is above the operating floor."],
                    main_tradeoffs=["Some flexibility is sacrificed."],
                    why_not_alternatives=["Transformational M&A is lower-confidence here."],
                ),
            )
        ],
        timeline=PlanTimeline(
            start_time="2026-04-01",
            step_schedule=[{"step_id": "step-1", "start": "2026-04-01", "end": "2026-05-01"}],
        ),
        triggers=[
            PlanTrigger(
                trigger_type="valuation_discount",
                condition="valuation_discount > 0.15",
                evaluation_frequency="weekly",
                trigger_probability=0.7,
                explanation="Only accelerate execution when the stock is still clearly discounted.",
            )
        ],
        branches=[
            PlanBranch(
                branch_condition="credit_spreads_widen > 100bps",
                branch_plan_steps=["portfolio.asset_sale"],
                branch_probability=0.2,
                explanation="Preserve balance-sheet flexibility if financing conditions worsen.",
            )
        ],
        score_breakdown=PlanScoreBreakdown(
            expected_utility=0.63,
            feasibility_chain=0.82,
            robustness_score=0.58,
            tail_risk_penalty=0.08,
            complexity_penalty=0.04,
            time_discount_factor=0.95,
            total_score=0.51,
            components={"value_creation": 0.4, "risk_reduction": 0.11},
        ),
        risks=PlanRisk(
            main_failure_modes=["Macro shock compresses valuation before repurchases complete."],
            regime_sensitivity=["Less attractive in tight-credit, high-volatility regimes."],
            execution_risks=["Authorization pace may lag market windows."],
        ),
        summary_explanation="Return capital after stabilizing the balance sheet; branch to asset sale if conditions deteriorate.",
    )

    payload = plan.to_dict()

    assert payload["steps"][0]["explanation"]["why_this_action"] == "Repurchases improve per-share capital efficiency."
    assert payload["triggers"][0]["trigger_type"] == "valuation_discount"
    assert payload["branches"][0]["branch_probability"] == 0.2
    assert payload["score_breakdown"]["total_score"] == 0.51
    assert payload["risks"]["execution_risks"] == ["Authorization pace may lag market windows."]
