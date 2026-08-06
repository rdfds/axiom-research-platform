from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.pipeline.precedent_distance_learning import (
    learn_precedent_distance_weights,
    learn_scope_weights,
)


def _synthetic_scope_df(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    valuation = rng.normal(12.0, 2.0, size=n)
    leverage = rng.normal(1.5, 0.4, size=n)
    market_stress = rng.normal(0.25, 0.08, size=n)
    df = pd.DataFrame(
        {
            "normalized_action_family": ["capital_return"] * n,
            "state_vector_v1.size_log_revenue": rng.normal(10.5, 0.2, size=n),
            "state_vector_v1.profitability": rng.normal(0.18, 0.03, size=n),
            "state_vector_v1.growth": rng.normal(0.04, 0.02, size=n),
            "state_vector_v1.gross_obligation_burden": leverage + rng.normal(0.1, 0.05, size=n),
            "state_vector_v1.net_obligation_burden": leverage,
            "state_vector_v1.liquidity_flexibility": rng.normal(1.8, 0.4, size=n),
            "state_vector_v1.interest_coverage": rng.normal(8.0, 1.5, size=n),
            "state_vector_v1.valuation_multiple": valuation,
            "state_vector_v1.cash_generation": rng.normal(0.04, 0.01, size=n),
            "state_vector_v1.market_stress": market_stress,
            "state_vector_v1.market_access": rng.normal(0.78, 0.08, size=n),
            "state_vector_v1.rates_level": rng.normal(4.0, 0.2, size=n),
            "state_vector_v1.credit_spread": rng.normal(3.0, 0.25, size=n),
        }
    )
    outcome_signal = 0.65 * valuation - 0.35 * leverage + rng.normal(0.0, 0.3, size=n)
    df["outcome_pe_12m"] = outcome_signal
    df["outcome_pe_6m"] = 0.7 * outcome_signal + rng.normal(0.0, 0.2, size=n)
    df["outcome_ev_ebitda_12m"] = 0.55 * valuation + rng.normal(0.0, 0.3, size=n)
    df["outcome_ev_ebitda_6m"] = 0.45 * valuation + rng.normal(0.0, 0.3, size=n)
    df["leverage_delta"] = -0.25 * leverage + rng.normal(0.0, 0.2, size=n)
    df["fcf_margin_delta"] = 0.15 * valuation + rng.normal(0.0, 0.2, size=n)
    return df


def test_learn_scope_weights_returns_nonempty_result():
    df = _synthetic_scope_df()
    learned = learn_scope_weights(
        df,
        scope_key="capital_return",
        scope_col="normalized_action_family",
        max_pairs=3000,
        min_rows=100,
        min_outcome_non_null=100,
        ridge_lambda=10.0,
        seed=7,
    )
    assert learned is not None
    assert learned["n_rows"] == len(df)
    assert learned["n_pairs"] > 0
    weights = learned["weights"]
    assert weights["state_vector_v1.valuation_multiple"] > weights["state_vector_v1.market_stress"]


def test_learn_precedent_distance_weights_writes_family_scope():
    df = _synthetic_scope_df()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "synthetic.parquet"
        df.to_parquet(path, index=False)
        payload = learn_precedent_distance_weights(
            path,
            max_pairs=3000,
            min_rows=100,
            min_outcome_non_null=100,
            ridge_lambda=10.0,
            seed=7,
        )
    assert payload["version"] == "precedent_distance_weights_v1"
    assert "capital_return" in payload["scopes"]
    weights = payload["scopes"]["capital_return"]["weights"]
    assert weights["state_vector_v1.valuation_multiple"] > weights["state_vector_v1.market_stress"]
    json.dumps(payload)
