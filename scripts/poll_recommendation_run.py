#!/usr/bin/env python
"""Poll a RecommendationRun JSON file until completion."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Poll RecommendationRun status from run store")
    p.add_argument("--run-id", required=True)
    p.add_argument("--runs-root", required=True)
    p.add_argument("--interval-seconds", type=float, default=5.0)
    p.add_argument("--max-waits", type=int, default=0, help="0 means wait indefinitely")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_path = Path(args.runs_root) / "runs" / f"run_id={args.run_id}.json"
    waits = 0

    while True:
        if run_path.exists():
            obj = json.loads(run_path.read_text())
            metadata = obj.get("metadata", {}) or {}
            progress = metadata.get("progress") or {}
            audit_log = obj.get("audit_log") or []
            last_event = audit_log[-1].get("event_type", "") if audit_log else ""
            line = {
                "ok": True,
                "run_id": args.run_id,
                "status": obj.get("status"),
                "progress_completed": progress.get("completed", 0),
                "progress_total": progress.get("total", 0),
                "last_event": last_event,
            }
            print(json.dumps(line), flush=True)
            if str(obj.get("status")) in {"completed", "failed"}:
                return
        else:
            print(json.dumps({"ok": True, "run_id": args.run_id, "status": "missing"}), flush=True)

        waits += 1
        if args.max_waits > 0 and waits >= args.max_waits:
            return
        time.sleep(max(0.1, float(args.interval_seconds)))


if __name__ == "__main__":
    main()
