#!/usr/bin/env python3
"""Build the committed Home Depot market-expectations sample as static HTML.

The original Axiom workspace renders a richer interactive surface from local
artifacts. This public builder keeps the same evidence shape but uses only the
small committed sample so anyone can reproduce the view without private data.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path


def _load(build_dir: Path, name: str) -> dict:
    return json.loads((build_dir / name).read_text())


def build_page(build_dir: Path) -> Path:
    payload = _load(build_dir, "valuation_driver_data.json")
    company = payload["companies"][0]
    market = company["market_expectations"]
    forward = company["forward_expectations"]
    driver_model = company["driver_weight_model"]
    assumptions = market.get("assumptions", [])
    diagnostics = forward.get("diagnostics", [])
    rows = []
    for item in assumptions:
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['label'])}</td>"
            f"<td>{item.get('weight', 0):.1%}</td>"
            f"<td>{item.get('current_percentile', 0):.0%}</td>"
            f"<td>{item.get('required_percentile_clipped', 0):.0%}</td>"
            f"<td>{html.escape(str(item.get('feasibility', 'unknown')))}</td>"
            "</tr>"
        )
    diagnostic_rows = []
    for item in diagnostics:
        validation = item.get("validation", {})
        diagnostic_rows.append(
            "<tr>"
            f"<td>{html.escape(item['label'])}</td>"
            f"<td>{html.escape(item.get('best_horizon', 'n/a'))}</td>"
            f"<td>{html.escape(item.get('validation_grade', 'n/a'))}</td>"
            f"<td>{validation.get('mae_improvement_vs_baseline', 0):.4f}</td>"
            f"<td>{validation.get('directional_hit_rate', 0):.1%}</td>"
            "</tr>"
        )
    serialized = json.dumps(payload, separators=(",", ":"))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title> Axiom — Market expectations for {html.escape(company['name'])}</title>
<style>
body{{font:15px/1.5 system-ui,-apple-system,sans-serif;background:#0b1020;color:#edf2ff;margin:0}}
main{{max-width:1100px;margin:0 auto;padding:32px 20px 64px}} h1,h2{{letter-spacing:-.02em}}
.lede{{color:#b9c5e2;max-width:780px}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:24px 0}}
.card{{background:#151d34;border:1px solid #2b3a61;border-radius:12px;padding:18px}} .metric{{font-size:28px;font-weight:700}}
table{{width:100%;border-collapse:collapse;background:#151d34;border-radius:12px;overflow:hidden;margin:12px 0 28px}}
th,td{{padding:11px 12px;border-bottom:1px solid #2b3a61;text-align:left}} th{{color:#9fb0d4;font-size:12px;text-transform:uppercase}}
.note{{border-left:3px solid #69d2a3;padding-left:14px;color:#c8d6f5}} code{{color:#9fe3c5}}
</style></head><body><main id="HD">
<h1>Axiom: market-implied valuation gap</h1>
<p class="lede">A reproducible public sample showing how Axiom separates a valuation premium into validated forward driver expectations and residual value outside the measured driver surface.</p>
<div class="grid">
<div class="card"><div>Company</div><div class="metric">{html.escape(company['name'])}</div><div>{html.escape(company['ticker'])} · as of {html.escape(company['data_timing']['as_of'])}</div></div>
<div class="card"><div>Observed multiple</div><div class="metric">{market['actual_multiple']:.1f}x</div><div>fair translated multiple {driver_model['native_translation']['translated_fair_multiple']:.1f}x</div></div>
<div class="card"><div>Forward grade</div><div class="metric">{html.escape(forward['forward_grade'])}</div><div>{len(diagnostics)} driver validations in sample</div></div>
<div class="card"><div>Model fit</div><div class="metric">R² {driver_model['r2']:.3f}</div><div>rank IC {driver_model['rank_ic']:.3f}</div></div>
</div>
<h2>Driver assumptions underwriting the gap</h2>
<table><thead><tr><th>Driver</th><th>Weight</th><th>Current percentile</th><th>Required percentile</th><th>Feasibility</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Forward validation</h2>
<p class="note">Axiom only presents a market-implied driver as decision evidence when the gap blend improves out-of-sample validation. A residual is labeled explicitly rather than forced into a driver.</p>
<table><thead><tr><th>Driver</th><th>Horizon</th><th>Grade</th><th>MAE improvement</th><th>Directional hit rate</th></tr></thead><tbody>{''.join(diagnostic_rows)}</tbody></table>
<h2>Reproducibility</h2><p>Generated from committed sample inputs by <code>python scripts/build_hd_market_expectations_demo.py</code>. No private data, credentials, or external service calls are needed.</p>
<script>const DATA={serialized};</script>
</main></body></html>"""
    output = build_dir / "valuation_action_bridge.html"
    output.write_text(page)
    return output


def main() -> int:
    build_dir = Path(os.environ.get("AXIOM_MNA_INSIGHTS_DIR", "examples/hd_market_expectations/build"))
    build_dir.mkdir(parents=True, exist_ok=True)
    output = build_page(build_dir)
    print(f"Built {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
