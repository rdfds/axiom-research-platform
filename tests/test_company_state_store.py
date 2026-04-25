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


