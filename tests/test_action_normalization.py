from src.action_normalization import augment_action_outcomes_df, normalize_action_record


def test_normalize_platform_acquisition_family_scale():
    row = {
        "action_type": "acquisition",
        "action_subtype": "Disclosed Dollar Value Deal",
        "action_size": 2_000_000_000.0,
        "base_market_cap": 10_000_000_000.0,
    }
    out = normalize_action_record(row)
    assert out["normalized_action_family"] == "mna"
    assert out["normalized_action_subfamily"] == "platform_disclosed"
    assert out["normalized_action_id"] == "mna.platform_acquisition"
    assert out["normalization_level"] == "family_scale"
    assert out["family_scale_bucket"] == "medium"


