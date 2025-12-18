import argparse
import time
from pathlib import Path
from typing import List, Optional

import duckdb


def _is_readable(path: Path) -> bool:
    try:
        st = path.stat()
        if hasattr(st, "st_blocks") and st.st_blocks == 0:
            try:
                with open(path, "rb") as f:
                    f.read(4)
                return True
            except Exception:
                return False
        with open(path, "rb") as f:
            f.read(4)
        return True
    except Exception:
        return False


def parse_years(arg: Optional[str]) -> List[int]:
    if not arg:
        return []
    out = []
    for part in arg.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build as-of fact registry (filtered) by year.")
    ap.add_argument("--asof", required=True, help="As-of timestamp (e.g., 2026-02-28)")
    ap.add_argument("--in-path", default="data/inputs_layer/extracted_fact_registry_validity")
    ap.add_argument("--out", default=None, help="Output root (default: data/inputs_layer/facts_asof_YYYY)")
    ap.add_argument("--years", default=None, help="Comma-separated list of years")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--memory", default="4GB")
    ap.add_argument("--temp-dir", default=None)
    ap.add_argument("--max-temp", default=None)
    args = ap.parse_args()

    asof = args.asof
    in_root = Path(args.in_path)
    if args.out:
        out_root = Path(args.out)
    else:
        out_root = Path(f"data/inputs_layer/facts_asof_{asof[:4]}")
    out_root.mkdir(parents=True, exist_ok=True)

    years = parse_years(args.years)
    if not years:
        years = sorted([int(p.name.split("=")[1]) for p in in_root.glob("year=*")])

    con = duckdb.connect()
    con.execute(f"SET threads={int(args.threads)}")
    if args.memory:
        con.execute(f"SET memory_limit='{args.memory}'")
    if args.temp_dir:
        con.execute(f"SET temp_directory='{args.temp_dir}'")
    if args.max_temp:
        con.execute(f"PRAGMA max_temp_directory_size='{args.max_temp}'")

    for y in years:
        in_file = in_root / f"year={y}" / "part.parquet"
        if not in_file.exists():
            print(f"[skip] missing year={y}")
            continue
        if not _is_readable(in_file):
            print(f"[skip] unreadable year={y} {in_file}")
            continue
        out_dir = out_root / f"year={y}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "part.parquet"
        if out_file.exists():
            if args.overwrite:
                out_file.unlink()
            elif args.resume:
                print(f"[skip] exists year={y}")
                continue
        t0 = time.time()
        print(f"[year {y}] filtering...", flush=True)
        con.execute(
            f"""
            COPY (
              SELECT *
              FROM read_parquet('{in_file.as_posix()}', union_by_name=True)
              WHERE (published_at IS NULL OR published_at <= TIMESTAMPTZ '{asof}')
                AND (ingested_at IS NULL OR ingested_at <= TIMESTAMPTZ '{asof}')
                AND ((valid_from IS NULL OR valid_from <= TIMESTAMPTZ '{asof}')
                     AND (valid_to IS NULL OR valid_to > TIMESTAMPTZ '{asof}'))
            ) TO '{out_file.as_posix()}' (FORMAT 'parquet');
            """
        )
        print(f"[year {y}] wrote {out_file} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
