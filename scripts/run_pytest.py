#!/usr/bin/env python3
"""Stable local pytest entrypoint for the axiom workspace.

The desktop environment injects third-party pytest plugins globally, which can
make raw `pytest` runs flaky or hang during collection. This wrapper keeps test
runs project-local and repeatable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    )
    cmd = [sys.executable, "-m", "pytest", *sys.argv[1:]]
    return subprocess.call(cmd, cwd=repo_root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
