#!/usr/bin/env python3
"""Download targeted SEC companyfacts JSON files to a non-iCloud local path.

This is meant to avoid Desktop/iCloud file-provider issues by pulling only the
CIKs we care about into a stable local folder such as `/Users/.../code/...`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company-ids-file", required=True, help="File with one CIK/entity_id per line")
    parser.add_argument("--out-root", required=True, help="Output folder for downloaded companyfacts JSON")
    parser.add_argument(
        "--user-agent",
        default="AxiomResearch/1.0 (research contact: rvariankaval)",
        help="SEC fair-access User-Agent string",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.15, help="Delay between SEC requests")
    parser.add_argument("--overwrite", action="store_true", help="Re-download files even if they already exist")
    return parser.parse_args()


def iter_ciks(path: Path) -> Iterable[str]:
    for line in path.read_text().splitlines():
        cik = line.strip()
        if not cik:
            continue
        yield cik.zfill(10)


def fetch_json(url: str, user_agent: str) -> dict:
    request = Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "identity"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    success = 0
    skipped = 0
    failed = 0

    for cik in iter_ciks(Path(args.company_ids_file)):
        out_path = out_root / f"CIK{cik}.json"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        url = SEC_COMPANYFACTS_URL.format(cik=cik)
        try:
            payload = fetch_json(url, args.user_agent)
            out_path.write_text(json.dumps(payload))
            success += 1
            print(f"downloaded {cik} -> {out_path}")
        except HTTPError as exc:
            failed += 1
            print(f"http_error {cik} status={exc.code}")
        except URLError as exc:
            failed += 1
            print(f"url_error {cik} reason={exc.reason}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"error {cik} {exc!r}")

        time.sleep(args.sleep_seconds)

    print(f"done success={success} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
