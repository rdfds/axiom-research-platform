import argparse
from pathlib import Path
from typing import List

import pandas as pd

from src.company_state_builder import CompanyStateBuilder
from src.company_state_store import SnapshotStore


def parse_dates(start: str, end: str) -> List[str]:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    dates = pd.date_range(start_dt, end_dt, freq="D")
    return [d.strftime("%Y-%m-%d") for d in dates]


