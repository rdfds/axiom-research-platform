# Final Status

As of March 25, 2026, the project has two stable reference scoreboards.

## Standard Benchmark

This remains the main trusted benchmark path.

- capital return: `/tmp/fixed_manifest_capreturn_holdout_buyback_recap_v36.json`
- capital structure: `/tmp/fixed_manifest_capstructure_holdout_buyback_recap_v36.json`

Topline metrics:

- capital return `mean_alignment_score = 0.831481`
- capital return `anchor_primary_exact_rate = 0.62963`
- capital return `anchor_primary_family_rate = 0.814815`
- capital structure `mean_alignment_score = 0.965517`
- capital structure `anchor_primary_exact_rate = 0.931034`
- capital structure `anchor_primary_family_rate = 0.965517`

## Manual Replay

This is the replay hardening / debugging path.

- capital return: `/tmp/fixed_manifest_capreturn_holdout_manual_v42_noplanfix.json`
- capital structure: `/tmp/fixed_manifest_capstructure_holdout_manual_v43_recaprank.json`

Topline metrics:

- capital return `mean_alignment_score = 0.82963`
- capital return `anchor_primary_exact_rate = 0.259259`
- capital return `anchor_primary_family_rate = 0.814815`
- capital return `unsupported_case_count = 0`
- capital structure `mean_alignment_score = 0.764`
- capital structure `anchor_primary_exact_rate = 0.44`
- capital structure `anchor_primary_family_rate = 0.56`
- capital structure `unsupported_case_count = 4`

