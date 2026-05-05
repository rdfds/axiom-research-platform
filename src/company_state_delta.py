"""Delta updater for CompanyState snapshots."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Dict

import pandas as pd

from .company_state_builder import CompanyStateBuilder


def update_snapshot(snapshot: Dict, builder: CompanyStateBuilder, as_of: str, mode: str = "both") -> Dict:
    asof_dt = pd.to_datetime(as_of, utc=True)
    cid = snapshot.get("company_id")
    if mode in ("market", "both"):
        facts = builder._load_facts(cid, asof_dt)
        ts = builder._load_timeseries(cid, asof_dt)
        market = builder._compute_market(ts, facts, asof_dt)
        feature_updates: Dict[str, object] = {}
        for k, v in market.items():
            if is_dataclass(v):
                feature_updates[k] = asdict(v)
            else:
                feature_updates[k] = v
        snapshot["features"].update(feature_updates)
    if mode in ("regime", "both"):
        macro = builder._load_macro(asof_dt)
        snapshot["regime"] = builder._compute_regime(macro, asof_dt)
    snapshot["as_of_time"] = asof_dt.isoformat()
    return snapshot
