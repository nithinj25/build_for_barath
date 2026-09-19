"""Run the system locally: python scripts/serve_local.py [--port 8000] [--demo]

Serves the analyst UI from ui/ and routes /api/* into handlers.api.handler as
API Gateway HTTP API (v2) events — the exact code Lambda runs, so what works
here is what deploys. State goes to local JSON-lines files instead of DynamoDB.

    --demo   mark synthetic ground-truth links in shortlists (DEMO_GROUND_TRUTH=1)
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "ui"


def make_handler(api):
    class Handler(BaseHTTPRequestHandler):
        def _api(self, method: str) -> None:
            url = urlsplit(self.path)
            length = int(self.headers.get("content-length") or 0)
            event = {"rawPath": url.path[len("/api"):] or "/",
                     "queryStringParameters": dict(parse_qsl(url.query)) or None,
                     "requestContext": {"http": {"method": method}},
                     "headers": {k.lower(): v for k, v in self.headers.items()},
                     "body": self.rfile.read(length).decode("utf-8") if length else None}
            result = api(event)
            body = result.get("body", "").encode("utf-8")
            self.send_response(result["statusCode"])
            for k, v in result.get("headers", {}).items():
                self.send_header(k, v)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _static(self) -> None:
            rel = urlsplit(self.path).path.lstrip("/") or "index.html"
            target = (UI / rel).resolve()
            if UI.resolve() not in target.parents or not target.is_file():
                target = UI / "index.html"
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("content-type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._api("GET") if self.path.startswith("/api/") else self._static()

        def do_POST(self):
            self._api("POST") if self.path.startswith("/api/") else self.send_error(405)

        def do_OPTIONS(self):
            self._api("OPTIONS")

        def log_message(self, fmt, *args):
            if self.path.startswith("/api/"):
                sys.stderr.write(f"{self.command} {self.path} {args[1] if len(args) > 1 else ''}\n")

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--bundle", default=str(ROOT / "data/serve"))
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("BUNDLE_URI", args.bundle)
    os.environ.setdefault("STORE", f"local:{Path(args.bundle) / 'state'}")
    if args.demo:
        os.environ["DEMO_GROUND_TRUTH"] = "1"
    sys.path.insert(0, str(ROOT))
    from handlers.api import handler

    handler({"rawPath": "/meta", "requestContext": {"http": {"method": "GET"}}})     # warm the bundle
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(handler))
    print(f"serving on http://127.0.0.1:{args.port}  (bundle {os.environ['BUNDLE_URI']}"
          f"{', demo ground truth ON' if args.demo else ''})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
