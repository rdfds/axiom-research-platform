"""Project-local Python startup tweaks.

Keep pytest isolated from globally installed third-party plugins so local test
behavior is stable inside the shared desktop environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _running_pytest(argv: list[str]) -> bool:
    head = [part.lower() for part in argv[:2]]
    argv0 = Path(argv[0]).name.lower() if argv else ""
    return bool(
        argv0 in {"pytest", "py.test"}
        or "pytest" in argv0
        or head == ["-m", "pytest"]
        or (head and head[0] == "pytest")
    )


if "PYTEST_DISABLE_PLUGIN_AUTOLOAD" not in os.environ and _running_pytest(sys.argv):
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
