import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from src.company_state_validation import (
    check_invariants,
    validate_peer_percentiles,
    validate_peer_zscores,
    validate_peer_bands,
)


def load_snapshots(path: Path) -> List[dict]:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    out: List[dict] = []
    with path.open("r") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


