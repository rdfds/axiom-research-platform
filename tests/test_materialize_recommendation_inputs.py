from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path


def test_materialize_recommendation_inputs_script(tmp_path: Path):
    snapshot_root = tmp_path / "snapshots"
    keyed_dir = snapshot_root / "keyed" / "as_of_date=2026-02-28"
    keyed_dir.mkdir(parents=True, exist_ok=True)
    (keyed_dir / "company_id=0000320193.json").write_text('{"company_id":"0000320193"}')

    entity_graph = tmp_path / "entity_graph.parquet"
    entity_identifier = tmp_path / "entity_identifier.parquet"
    outcomes = tmp_path / "outcomes.parquet"
    model = tmp_path / "model.json"
    blocklist = tmp_path / "blocklist.txt"
    for path in [entity_graph, entity_identifier, outcomes, model, blocklist]:
        path.write_text(path.name)

    dest_root = tmp_path / "local_bundle"
    manifest = tmp_path / "manifest.json"

    argv = [
        "materialize_recommendation_inputs.py",
        "--snapshot-root",
        str(snapshot_root),
        "--as-of",
        "2026-02-28",
        "--companies",
        "0000320193",
        "--entity-graph-path",
        str(entity_graph),
        "--entity-identifier-path",
        str(entity_identifier),
        "--outcomes-path",
        str(outcomes),
        "--causal-model-path",
        str(model),
        "--causal-action-blocklist-path",
        str(blocklist),
        "--dest-root",
        str(dest_root),
        "--out-manifest",
        str(manifest),
    ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        runpy.run_path("./scripts/materialize_recommendation_inputs.py", run_name="__main__")
    finally:
        sys.argv = old_argv

    payload = json.loads(manifest.read_text())
    assert Path(payload["snapshot_root"]).exists()
    assert Path(payload["entity_graph_path"]).exists()
    assert Path(payload["outcomes_path"]).exists()
    assert (Path(payload["snapshot_root"]) / "keyed" / "as_of_date=2026-02-28" / "company_id=0000320193.json").exists()
