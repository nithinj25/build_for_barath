"""HTTP API and the analyst UI. On AWS it sits behind a Lambda Function URL
(payload v2 — the same shape as API Gateway HTTP API); scripts/serve_local.py
feeds it real requests locally. The same code both places.

    GET  /                                       the analyst UI (ui/index.html)
    GET  /health
    GET  /meta                                   corpus context, demo cases
    GET  /cases/{id}                             canonical record, no PII
    GET  /cases/{id}/links?scope=same|all&limit  ranked shortlist
    POST /links/{id_a}__{id_b}/feedback          {"verdict": "confirmed"|"rejected", "note": "..."}

Every shortlist request is audited: who, which case, which scope, when.
Responses never carry a probability (CLAUDE.md rule 4). The UI calls /api/...;
that prefix is stripped here so one URL serves both page and API.

Environment:
    BUNDLE_URI          directory or s3://bucket/prefix (default data/serve)
    STORE               local:<dir> or dynamodb:<feedback table>,<audit table>
    DEMO_GROUND_TRUTH   "1" adds synthetic true-link marks for demos (default off)
    ALLOWED_ORIGIN      CORS origin (default *)
"""
from __future__ import annotations

import base64
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from handlers.store import load_bundle, open_store
from linkage.serve import Shortlister

CASE = r"([0-9a-f]{16})"
VERDICTS = ("confirmed", "rejected")
UI_PAGE = Path(__file__).resolve().parent.parent / "ui" / "index.html"
_APP: dict = {}


def _app() -> dict:
    if not _APP:
        bundle = load_bundle(os.environ.get("BUNDLE_URI", "data/serve"))
        demo = os.environ.get("DEMO_GROUND_TRUTH") == "1" and bundle["truth_groups"] is not None
        _APP["shortlister"] = Shortlister(bundle["weights"], bundle["case_ids"], bundle["crime_types"],
                                          bundle["codes"], bundle["meta"], bundle["cases"],
                                          bundle["truth_groups"] if demo else None)
        _APP["cases"] = bundle["cases"]
        _APP["meta"] = {**bundle["meta"], "ground_truth_marks": demo}
        _APP["store"] = open_store(os.environ.get("STORE", "local:data/serve/state"))
    return _APP


def _response(status: int, body) -> dict:
    return {"statusCode": status,
            "headers": {"content-type": "application/json",
                        "access-control-allow-origin": os.environ.get("ALLOWED_ORIGIN", "*"),
                        "access-control-allow-headers": "content-type,authorization",
                        "access-control-allow-methods": "GET,POST,OPTIONS"},
            "body": json.dumps(body) if body is not None else ""}


def _actor(event: dict) -> str:
    claims = (((event.get("requestContext") or {}).get("authorizer") or {}).get("jwt") or {}).get("claims") or {}
    return claims.get("email") or claims.get("sub") or "anonymous"


def _body(event: dict) -> dict:
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode("utf-8")
    return json.loads(raw)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def handler(event: dict, context=None) -> dict:
    method = ((event.get("requestContext") or {}).get("http") or {}).get("method", "GET")
    path = event.get("rawPath", "/").rstrip("/") or "/"
    if path == "/api" or path.startswith("/api/"):
        path = path[len("/api"):] or "/"
    query = event.get("queryStringParameters") or {}
    if method == "OPTIONS":
        return _response(204, None)
    if method == "GET" and path in ("/", "/index.html"):
        if not UI_PAGE.exists():
            return _response(404, {"error": "UI not packaged"})
        return {"statusCode": 200, "headers": {"content-type": "text/html; charset=utf-8",
                                               "cache-control": "no-cache"},
                "body": UI_PAGE.read_text(encoding="utf-8")}
    if path == "/health":
        return _response(200, {"status": "ok"})
    try:
        app = _app()
        if method == "GET" and path == "/meta":
            return _response(200, app["meta"])

        if method == "GET" and (m := re.fullmatch(f"/cases/{CASE}", path)):
            case = app["cases"].get(m.group(1))
            return _response(200, case) if case else _response(404, {"error": "unknown case"})

        if method == "GET" and (m := re.fullmatch(f"/cases/{CASE}/links", path)):
            scope = query.get("scope", "same")
            limit = int(query.get("limit", "10"))
            if not 1 <= limit <= 50:
                return _response(400, {"error": "limit must be 1-50"})
            result = app["shortlister"].shortlist(m.group(1), scope, limit)
            app["store"].record_audit({"actor_id": _actor(event), "timestamp": _now(), "action": "shortlist",
                                       "case_id": m.group(1), "scope": scope})
            return _response(200, result)

        if method == "POST" and (m := re.fullmatch(f"/links/{CASE}__{CASE}/feedback", path)):
            body = _body(event)
            verdict = body.get("verdict")
            if verdict not in VERDICTS:
                return _response(400, {"error": f"verdict must be one of {list(VERDICTS)}"})
            if m.group(1) not in app["cases"] or m.group(2) not in app["cases"]:
                return _response(404, {"error": "unknown case"})
            item = {"pair_id": f"{m.group(1)}__{m.group(2)}", "actor_id": _actor(event), "timestamp": _now(),
                    "verdict": verdict, "note": str(body.get("note", ""))[:500]}
            app["store"].record_feedback(item)
            return _response(201, item)

        return _response(404, {"error": "not found"})
    except KeyError:
        return _response(404, {"error": "unknown case"})
    except (ValueError, json.JSONDecodeError) as exc:
        return _response(400, {"error": str(exc)})
