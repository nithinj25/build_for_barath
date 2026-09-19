"""HTTP API and the analyst UI. On AWS it sits behind a Lambda Function URL
(payload v2 — the same shape as API Gateway HTTP API); scripts/serve_local.py
feeds it real requests locally. The same code both places.

    GET  /                                       the analyst UI (ui/index.html)
    GET  /health
    GET  /meta                                   corpus context, demo cases
    GET  /leads?lane&state&district&crime_type&tier&cross_state&limit&offset
                                                 proactive leads inbox, rarest first; lane is
                                                 same_district | same_state | cross_state
    GET  /search?q=&limit                        FIR number, police station, district or case id
    GET  /cases/{id}                             canonical record, no PII
    GET  /cases/{id}/links?scope=same|all&limit  ranked shortlist
    GET  /pairs/{id_a}__{id_b}                   side-by-side comparison with reasons
    POST /links/{id_a}__{id_b}/feedback          {"verdict": "confirmed"|"rejected"|"investigate", "note": "..."}

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
VERDICTS = ("confirmed", "rejected", "investigate")
SEARCH_FIELDS = ("fir_no", "police_station", "district", "case_id")
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
        _APP["leads"] = bundle["leads"] if demo else [
            {k: v for k, v in lead.items() if k != "ground_truth_link"} for lead in bundle["leads"]]
        _APP["truth"] = bundle["truth_groups"] if demo else None
        meta = {**bundle["meta"], "ground_truth_marks": demo}
        if not demo:
            meta.pop("lead_quality", None)
        _APP["meta"] = meta
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

        if method == "GET" and path == "/leads":
            limit, offset = int(query.get("limit", "25")), int(query.get("offset", "0"))
            if not 1 <= limit <= 100 or offset < 0:
                return _response(400, {"error": "limit must be 1-100, offset >= 0"})
            state, district = query.get("state"), query.get("district")
            rows = [l for l in app["leads"]
                    if (not state or state in (l["state_a"], l["state_b"]))
                    and (not district or district in (l["district_a"], l["district_b"]))
                    and (not query.get("crime_type") or l["crime_type"] == query["crime_type"])
                    and (not query.get("tier") or l["tier"] == query["tier"])
                    and (query.get("cross_state") != "1" or l["cross_state"])]
            lanes: dict = {}
            for l in rows:
                lanes[l["lane"]] = lanes.get(l["lane"], 0) + 1
            if query.get("lane"):
                rows = [l for l in rows if l["lane"] == query["lane"]]
            page = rows[offset:offset + limit]
            if app["truth"] is not None:
                truth = app["truth"]
                page = [{**l, "ground_truth_link": truth.get(l["a"]) is not None
                         and truth.get(l["a"]) == truth.get(l["b"])} for l in page]
            return _response(200, {"total": len(rows), "offset": offset, "lanes": lanes, "items": page})

        if method == "GET" and path == "/search":
            text = (query.get("q") or "").strip().lower()
            if len(text) < 2:
                return _response(400, {"error": "search needs at least 2 characters"})
            limit = min(int(query.get("limit", "20")), 50)
            hits = []
            for c in app["cases"].values():
                if any(text in str(c.get(k) or "").lower() for k in SEARCH_FIELDS):
                    hits.append({k: c.get(k) for k in ("case_id", "fir_no", "crime_type", "state_code",
                                                      "district", "police_station", "occurred_from")})
                    if len(hits) >= limit:
                        break
            return _response(200, {"items": hits})

        if method == "GET" and (m := re.fullmatch(f"/pairs/{CASE}__{CASE}", path)):
            result = app["shortlister"].pair(m.group(1), m.group(2))
            app["store"].record_audit({"actor_id": _actor(event), "timestamp": _now(), "action": "compare",
                                       "case_id": m.group(1), "other_case_id": m.group(2)})
            return _response(200, result)

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
