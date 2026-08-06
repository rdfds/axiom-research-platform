import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CompanyState checks suite.")
    parser.add_argument("--snapshot", required=True, help="Snapshot jsonl/parquet path")
    parser.add_argument("--sample", type=int, default=200)
    args = parser.parse_args()

    commands = [
        ["python", "-u", "scripts/check_company_state_invariants.py", "--path", args.snapshot, "--sample", str(args.sample)],
        ["python", "-m", "py_compile", "src/company_state_builder.py"],
        ["python", "-m", "py_compile", "src/company_state_validation.py"],
    ]

    for cmd in commands:
        print("[run]", " ".join(cmd))
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print("[fail]", " ".join(cmd))
            sys.exit(res.returncode)

    print("All checks passed.")


if __name__ == "__main__":
    main()
