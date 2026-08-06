from __future__ import annotations

from pathlib import Path

from src.company_state_store import SnapshotStore


def _snap(company_id: str, as_of: str) -> dict:
    return {
        "snapshot_id": f"snap-{company_id}",
        "company_id": company_id,
        "as_of_time": f"{as_of}T00:00:00Z",
        "features": {},
        "regime": {},
        "constraint_set": {"hard": [], "soft": []},
        "peer_set": {"peer_set_id": "p", "members": [], "method": "test", "version": 1},
        "provenance": {},
    }


def test_write_and_load_keyed_snapshot(tmp_path: Path):
    store = SnapshotStore(tmp_path / "snapshots", temp_dir=tmp_path / "tmp")
    as_of = "2026-02-28"
    store.write_keyed_json([_snap("ABC", as_of), _snap("XYZ", as_of)], as_of, expected_count=2)

    abc = store.load_keyed_snapshot("ABC", as_of)
    xyz = store.load_keyed_snapshot("XYZ", as_of)
    assert abc is not None and abc["company_id"] == "ABC"
    assert xyz is not None and xyz["company_id"] == "XYZ"


def test_write_jsonl_row_count_guard(tmp_path: Path):
    store = SnapshotStore(tmp_path / "snapshots", temp_dir=tmp_path / "tmp")
    as_of = "2026-02-28"
    try:
        store.write_jsonl([_snap("ABC", as_of)], as_of, expected_count=2)
    except RuntimeError as e:
        assert "row-count mismatch" in str(e)
        return
    raise AssertionError("Expected RuntimeError for row-count mismatch")

