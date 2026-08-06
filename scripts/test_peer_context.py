import argparse
import json
from pathlib import Path

from src.company_state_builder import CompanyStateBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick test for peer_context features.")
    parser.add_argument("--company-id", required=True)
    parser.add_argument("--asof", required=True)
    args = parser.parse_args()

    builder = CompanyStateBuilder()
    snap = builder.build(args.company_id, args.asof)
    feats = snap.features
    keys = [
        "peer_context.peer_actions_rate",
        "peer_context.action_rate_percentile",
        "peer_context.action_rate_z",
        "peer_context.action_rate_band",
        "peer_context.valuation_percentile",
        "peer_context.leverage_percentile",
        "peer_context.margin_percentile",
        "peer_context.valuation_z",
        "peer_context.leverage_z",
        "peer_context.margin_z",
        "peer_context.valuation_band",
        "peer_context.leverage_band",
        "peer_context.margin_band",
    ]
    for k in keys:
        print(k, feats.get(k))


if __name__ == "__main__":
    main()
