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


def test_pick_source_keeps_champion_when_challenger_under_floor() -> None:
    champion = {"enabled": True, "oos_r2": 0.11}
    challenger = {"enabled": True, "oos_r2": 0.05}
    picked = _pick_source(
        champion_model=champion,
        challenger_model=challenger,
        challenger_min_oos_r2=0.08,
        replace_min_delta_oos_r2=0.0,
    )
    assert picked == "champion"


def test_pick_source_keeps_champion_when_challenger_not_better_by_delta() -> None:
    champion = {"enabled": True, "oos_r2": 0.15}
    challenger = {"enabled": True, "oos_r2": 0.16}
    picked = _pick_source(
        champion_model=champion,
        challenger_model=challenger,
        challenger_min_oos_r2=0.08,
        replace_min_delta_oos_r2=0.02,
    )
    assert picked == "champion"


def test_pick_source_replaces_when_champion_missing() -> None:
    challenger = {"enabled": True, "oos_r2": 0.20}
    picked = _pick_source(
        champion_model=None,
        challenger_model=challenger,
        challenger_min_oos_r2=0.08,
        replace_min_delta_oos_r2=0.0,
    )
    assert picked == "challenger"
