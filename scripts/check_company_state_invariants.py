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


def main() -> None:
    parser = argparse.ArgumentParser(description="Check CompanyState invariant rules.")
    parser.add_argument("--path", required=True, help="Snapshot file (jsonl or parquet)")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--max-errors", type=int, default=50)
    args = parser.parse_args()

    path = Path(args.path)
    snapshots = load_snapshots(path)
    if args.sample:
        snapshots = snapshots[: args.sample]

    total = 0
    bad = 0
    for snap in snapshots:
        total += 1
        errs = (
            check_invariants(snap)
            + validate_peer_percentiles(snap)
            + validate_peer_zscores(snap)
            + validate_peer_bands(snap)
        )
        if errs:
            bad += 1
            if bad <= args.max_errors:
                print(f"snapshot {snap.get('company_id')}: {errs}")

    print(f"Checked {total} snapshots. Violations: {bad}")


if __name__ == "__main__":
    main()
