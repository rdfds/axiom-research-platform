import argparse
import json
from pathlib import Path
from typing import List

from src.company_state_builder import CompanyStateBuilder
from src.company_state_delta import update_snapshot


def load_snapshots(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


