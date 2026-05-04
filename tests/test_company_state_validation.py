from src.company_state_validation import (
    check_invariants,
    validate_peer_percentiles,
    validate_peer_zscores,
    validate_peer_bands,
)


def _snapshot(**feature_values):
    feats = {}
    for k, v in feature_values.items():
        feats[k] = {"value": v, "as_of_time": "2026-02-28T00:00:00Z"}
    return {
        "company_id": "001",
        "as_of_time": "2026-02-28T00:00:00Z",
        "features": feats,
    }


