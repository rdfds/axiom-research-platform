#!/usr/bin/env python
"""
Minimal on-demand HTTP API for precedent inference.

Endpoints:
  GET  /health
  POST /run_precedent
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple

# Allow running as `python scripts/run_precedent_api.py` from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.run import run_precedent


def _default_precedent_outcomes_path() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    return str(repo_root / "data" / "curated" / "action_outcomes_with_credit_ratings.normalized_full.parquet")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve run_precedent over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--config", default=None)
    parser.add_argument("--outcomes-path", default=_default_precedent_outcomes_path())
    parser.add_argument("--state-snapshot-root", default=None)
    parser.add_argument("--state-snapshot-path", default=None)
    return parser.parse_args()


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, default=str).encode("utf-8")


def _read_json_body(handler: BaseHTTPRequestHandler) -> Tuple[Dict[str, Any], str | None]:
    length_raw = handler.headers.get("Content-Length", "0")
    try:
        length = int(length_raw)
    except Exception:
        return {}, "invalid_content_length"
    if length <= 0:
        return {}, "empty_body"
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode("utf-8")), None
    except Exception:
        return {}, "invalid_json"


def build_handler(defaults: argparse.Namespace):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: Dict[str, Any]) -> None:
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            # Keep server output concise.
            return

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"ok": True})
                return
            self._send(404, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:
            if self.path != "/run_precedent":
                self._send(404, {"ok": False, "error": "not_found"})
                return
            body, err = _read_json_body(self)
            if err:
                self._send(400, {"ok": False, "error": err})
                return

            company_id = body.get("company_id")
            as_of = body.get("as_of")
            action_id = body.get("action_id")
            action_type = body.get("action_type")
            action_subtype = body.get("action_subtype")
            action_params = body.get("action_params", {})

            if not company_id or not as_of:
                self._send(400, {"ok": False, "error": "missing_company_id_or_as_of"})
                return
            if not action_id and not action_type:
                self._send(400, {"ok": False, "error": "missing_action_id_or_action_type"})
                return

            try:
                pack = run_precedent(
                    company_id=str(company_id),
                    as_of_date=str(as_of),
                    action_id=str(action_id) if action_id is not None else None,
                    action_type=str(action_type) if action_type is not None else None,
                    action_subtype=str(action_subtype) if action_subtype is not None else None,
                    action_params=action_params if isinstance(action_params, dict) else {},
                    config_path=defaults.config,
                    outcomes_path=defaults.outcomes_path,
                    state_snapshot_root=defaults.state_snapshot_root,
                    state_snapshot_path=defaults.state_snapshot_path,
                )
                payload = pack.to_dict()
                self._send(200, {"ok": True, "result": payload})
            except Exception as exc:
                self._send(400, {"ok": False, "error": str(exc)})

    return Handler


def main() -> None:
    args = _parse_args()
    handler = build_handler(args)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        json.dumps(
            {
                "ok": True,
                "message": "run_precedent_api_started",
                "host": args.host,
                "port": args.port,
            }
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
