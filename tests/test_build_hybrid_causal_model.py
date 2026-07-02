from scripts.build_hybrid_causal_model import _pick_source


def test_pick_source_prefers_challenger_when_champion_disabled() -> None:
    champion = {"enabled": False, "oos_r2": 0.30}
    challenger = {"enabled": True, "oos_r2": 0.12}
    picked = _pick_source(
        champion_model=champion,
        challenger_model=challenger,
        challenger_min_oos_r2=0.08,
        replace_min_delta_oos_r2=0.0,
    )
    assert picked == "challenger"


