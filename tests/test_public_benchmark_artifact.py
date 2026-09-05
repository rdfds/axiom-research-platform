import json
from pathlib import Path

import pytest

from scripts.build_public_benchmark import REPOSITORY_ROOT, build_payload


def test_public_benchmark_preserves_evaluation_contract() -> None:
    benchmark = build_payload()

    assert benchmark["evaluation"]["successful_driver_horizon_evaluations"] == 912
    assert len(benchmark["evaluation"]["train_end_dates"]) == 4
    assert benchmark["evaluation"]["placebo_runs"] == 3
    assert benchmark["selected_policy"]["lambda"] == 0.5
    assert benchmark["placebo_comparison"]["actual_minus_placebo"] == pytest.approx(
        0.007885185456210397
    )
    assert benchmark["placebo_comparison"]["actual_beats_placebo_rate"] == pytest.approx(
        0.6348684210526315
    )


