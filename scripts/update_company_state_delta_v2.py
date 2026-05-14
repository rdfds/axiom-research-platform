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


def main() -> None:
    parser = argparse.ArgumentParser(description="Update market/regime fields in snapshots.")
    parser.add_argument("--in-path", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--mode", choices=["market", "regime", "both"], default="both")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    builder = CompanyStateBuilder()
    snapshots = load_snapshots(in_path)

    with out_path.open("w") as f:
        for snap in snapshots:
            snap = update_snapshot(snap, builder, args.asof, args.mode)
            f.write(json.dumps(snap) + "\n")

    print(f"Wrote updated snapshots -> {out_path}")


if __name__ == "__main__":
    main()
