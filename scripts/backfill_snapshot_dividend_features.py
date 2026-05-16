import argparse
import json
from datetime import timezone
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import pandas as pd


RECURRING_DIVIDEND_EVENT_TYPES = {
    "dividend_regular",
    "dividend_increase",
    "dividend_cut",
    "dividend_initiate",
}
RECURRING_DIVIDEND_SUBTYPES = {
    "regular",
    "dividend_increase",
    "dividend_cut",
    "dividend_initiate",
}


def _now_iso() -> str:
    return pd.Timestamp.now(tz=timezone.utc).isoformat()


