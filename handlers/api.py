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
    GET  /cases/{id}/links?scope=same|all&limit&rank=blind|nearby
                                                 ranked shortlist; "nearby" adds place (officer's choice)
    GET  /pairs/{id_a}__{id_b}?rank=blind|nearby side-by-side comparison with reasons
    GET  /series?state&district&crime_type&limit&offset   possible series, multi-district first
    GET  /series/{id}                            one series: its FIRs in date order and the links joining them
    GET  /checks?state&district&limit&offset     FIRs whose MO looks like another crime type
    GET  /form                                   fields and allowed values per crime type, stations per district
    GET  /form/sample?crime_type                 a held-out test FIR to re-enter as if new (demo), with its text
    POST /read                                   {"text": "...", "crime_type"?} fields read from FIR text, and where
    POST /match                                  {"record": {...}, "rank": "blind"|"nearby", "demo_source": id?}
                                                 shortlists for a newly entered FIR, not stored in the corpus
    GET  /links/{id_a}__{id_b}/feedback          decisions already recorded on this pair
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
from linkage import schema, textread
from linkage.serve import Shortlister

CASE = r"([0-9a-f]{16})"
SERIES = r"([0-9a-f]{10})"
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
                                          bundle["truth_groups"] if demo else None, hub_r=bundle["hub_r"])
        _APP["cases"] = bundle["cases"]
        _APP["leads"] = bundle["leads"] if demo else [
            {k: v for k, v in lead.items() if k != "ground_truth_link"} for lead in bundle["leads"]]
        _APP["truth"] = bundle["truth_groups"] if demo else None
        _APP["series"] = bundle["series"]
        _APP["series_by_id"] = {s["id"]: s for s in bundle["series"]}
        _APP["series_of_case"] = {m["case_id"]: s["id"] for s in bundle["series"] for m in s["members"]}
        _APP["checks"] = bundle["checks"]
        stations: dict = {}
        for c in bundle["cases"].values():
            stations.setdefault(c["district"], set()).add(c.get("police_station"))
        _APP["stations"] = {d: sorted(s for s in v if s) for d, v in stations.items()}
        _APP["sample_turn"] = 0
        _APP["phrases"] = bundle["phrases"]
        _APP["check_of_case"] = {c["case_id"]: c for c in bundle["checks"]}
        meta = {**bundle["meta"], "ground_truth_marks": demo}
        if not demo:
            for key in ("lead_quality", "series_quality", "check_quality"):
                meta.pop(key, None)
        _APP["meta"] = meta
        _APP["store"] = open_store(os.environ.get("STORE", "local:data/serve/state"))
    return _APP


def _response(status: int, body) -> dict:
    return {"statusCode": status,
            "headers": {"content-type": "application/json",
                        "access-control-allow-origin": os.environ.get("ALLOWED_ORIGIN", "*"),
                        "access-control-allow-headers": "content-type,authorization,x-officer",
                        "access-control-allow-methods": "GET,POST,OPTIONS"},
            "body": json.dumps(body) if body is not None else ""}


def _actor(event: dict) -> str:
    """Who is asking. A verified login (JWT claims) when the stack has one; until
    then the name an officer types into the page, marked "demo:" because
    nothing verifies it."""
    claims = (((event.get("requestContext") or {}).get("authorizer") or {}).get("jwt") or {}).get("claims") or {}
    if claims.get("email") or claims.get("sub"):
        return claims.get("email") or claims.get("sub")
    name = re.sub(r"[^\w .-]", "", str((event.get("headers") or {}).get("x-officer", "")))[:40].strip()
    return f"demo:{name}" if name else "anonymous"


def _page(rows: list, query: dict, default: int = 20) -> tuple[list, int] | None:
    limit, offset = int(query.get("limit", str(default))), int(query.get("offset", "0"))
    if not 1 <= limit <= 100 or offset < 0:
        return None
    return rows[offset:offset + limit], offset


def _where(rows: list, query: dict, states, districts) -> list:
    state, district = query.get("state"), query.get("district")
    return [r for r in rows if (not state or state in states(r)) and (not district or district in districts(r))
            and (not query.get("crime_type") or r.get("crime_type", r.get("recorded")) == query["crime_type"])]


def _body(event: dict) -> dict:
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode("utf-8")
    return json.loads(raw)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _form(app: dict) -> dict:
    pools = app["shortlister"].scorer.pools
    return {"crime_types": {ct: [{"field": f, "values": pools[ct]["vocab"][f], "tag": f in schema.TAG_FIELDS}
                                 for f in pools[ct]["fields"]] for ct in schema.CRIME_TYPES if ct in pools},
            "places": app["meta"]["places"], "stations": app["stations"]}


def _new_record(app: dict, raw: dict) -> dict:
    """A newly entered FIR, checked against the vocabulary; anything unknown is refused."""
    pools = app["shortlister"].scorer.pools
    crime_type = raw.get("crime_type")
    if crime_type not in schema.CRIME_TYPES or crime_type not in pools:
        raise ValueError("choose a crime type")
    state, district = raw.get("state_code"), raw.get("district")
    if district not in app["meta"]["places"].get(state, []):
        raise ValueError("choose a known state and district")
    try:
        occurred = datetime.fromisoformat(str(raw.get("occurred_from"))).isoformat()
    except ValueError:
        raise ValueError("enter the date of the offence") from None
    record = {"crime_type": crime_type, "state_code": state, "district": district, "occurred_from": occurred,
              "police_station": re.sub(r"[^\w/-]", "", str(raw.get("police_station") or ""))[:24] or None,
              "fir_no": re.sub(r"[^\w/ -]", "", str(raw.get("fir_no") or ""))[:40] or "New FIR"}
    fields = raw.get("fields") or {}
    for f in pools[crime_type]["fields"]:
        v, vocab = fields.get(f), pools[crime_type]["vocab"][f]
        if v in (None, "", []):
            record[f] = None                                 # not recorded: costs and earns nothing
        elif f in schema.TAG_FIELDS:
            values = v if isinstance(v, list) else [v]
            if values == ["none"]:
                record[f] = ""                               # recorded as none (e.g. no tools): a known empty set
                continue
            if not all(x in vocab for x in values):
                raise ValueError(f"unknown value for {f}")
            record[f] = ";".join(sorted(values))
        elif v in vocab:
            record[f] = v
        else:
            raise ValueError(f"unknown value for {f}")
    return record


def _sample(app: dict, crime_type: str | None) -> dict:
    """A held-out test FIR (from the demo set: it has a true partner) as a new entry."""
    demo = [d for d in app["meta"]["demo_cases"] if not crime_type or d["crime_type"] == crime_type]
    if not demo:
        raise KeyError(crime_type)
    pick = demo[app["sample_turn"] % len(demo)]
    app["sample_turn"] += 1
    case = app["cases"][pick["case_id"]]
    pools = app["shortlister"].scorer.pools
    fields = {}
    for f in pools[case["crime_type"]]["fields"]:
        v = case.get(f)
        known = v is not None and v not in schema.TOKENS
        if f in schema.TAG_FIELDS and known:
            fields[f] = [x for x in v.split(";") if x] or ["none"]
        else:
            fields[f] = v if known else None
    text = " ".join(x for x in (case.get("narrative_text"), case.get("mo_description")) if x)
    return {"source_case_id": case["case_id"], "source_fir_no": case["fir_no"], "text": text,
            "record": {"crime_type": case["crime_type"], "state_code": case["state_code"], "district": case["district"],
                       "police_station": case.get("police_station"), "occurred_from": (case.get("occurred_from") or "")[:10],
                       "fields": fields}}


def _annotate(app: dict, case: dict) -> dict:
    """A case record plus what the bundle found about it: a crime-type check, a series."""
    out = dict(case)
    if case["case_id"] in app["check_of_case"]:
        out["type_check"] = app["check_of_case"][case["case_id"]]
    if case["case_id"] in app["series_of_case"]:
        series = app["series_by_id"][app["series_of_case"][case["case_id"]]]
        out["series_id"], out["series_size"] = series["id"], series["size"]
    return out


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
            rows = [l for l in _where(app["leads"], query, lambda l: (l["state_a"], l["state_b"]),
                                      lambda l: (l["district_a"], l["district_b"]))
                    if (not query.get("tier") or l["tier"] == query["tier"])
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

        if method == "GET" and path == "/series":
            rows = _where(app["series"], query, lambda s: s["states"], lambda s: s["districts"])
            paged = _page(rows, query)
            if paged is None:
                return _response(400, {"error": "limit must be 1-100, offset >= 0"})
            items = [{k: v for k, v in s.items() if k != "links"} for s in paged[0]]
            if app["truth"] is not None:
                truth = app["truth"]
                items = [{**s, "ground_truth_offenders": len({truth.get(x["case_id"], x["case_id"]) for x in s["members"]})}
                         for s in items]
            return _response(200, {"total": len(rows), "offset": paged[1], "items": items})

        if method == "GET" and (m := re.fullmatch(f"/series/{SERIES}", path)):
            series = app["series_by_id"].get(m.group(1))
            if series is None:
                return _response(404, {"error": "unknown series"})
            if app["truth"] is not None:
                truth = app["truth"]
                same = lambda a, b: truth.get(a) is not None and truth.get(a) == truth.get(b)
                series = {**series, "links": [{**l, "ground_truth_link": same(l["a"], l["b"])} for l in series["links"]],
                          "ground_truth_offenders": len({truth.get(x["case_id"], x["case_id"]) for x in series["members"]})}
            app["store"].record_audit({"actor_id": _actor(event), "timestamp": _now(), "action": "series",
                                       "series_id": m.group(1)})
            return _response(200, series)

        if method == "GET" and path == "/form":
            return _response(200, _form(app))

        if method == "GET" and path == "/form/sample":
            return _response(200, _sample(app, query.get("crime_type")))

        if method == "POST" and path == "/read":
            if app["phrases"] is None:
                return _response(404, {"error": "text reading not packaged"})
            body = _body(event)
            text = str(body.get("text") or "")[:8000]
            crime_type = body.get("crime_type") if body.get("crime_type") in schema.CRIME_TYPES else None
            pool_fields = {ct: app["shortlister"].scorer.pools[ct]["fields"] for ct in schema.CRIME_TYPES
                           if ct in app["shortlister"].scorer.pools}
            return _response(200, textread.read_fir(text, app["phrases"], app["meta"]["places"], pool_fields, crime_type))

        if method == "POST" and path == "/match":
            body = _body(event)
            record = _new_record(app, body.get("record") or {})
            source = body.get("demo_source")
            if source is not None and source not in app["cases"]:
                return _response(404, {"error": "unknown case"})
            result = app["shortlister"].match_record(record, body.get("rank", "blind"), 10, exclude=source)
            app["store"].record_audit({"actor_id": _actor(event), "timestamp": _now(), "action": "match_new",
                                       "crime_type": record["crime_type"], "district": record["district"],
                                       "demo_source": source or ""})
            return _response(200, {"record": record, **result})

        if method == "GET" and path == "/checks":
            rows = _where(app["checks"], query, lambda c: (c["state_code"],), lambda c: (c["district"],))
            paged = _page(rows, query)
            if paged is None:
                return _response(400, {"error": "limit must be 1-100, offset >= 0"})
            return _response(200, {"total": len(rows), "offset": paged[1], "items": paged[0]})

        if method == "GET" and (m := re.fullmatch(f"/links/{CASE}__{CASE}/feedback", path)):
            items = app["store"].feedback_for(f"{m.group(1)}__{m.group(2)}")
            items += app["store"].feedback_for(f"{m.group(2)}__{m.group(1)}")
            return _response(200, {"items": sorted(items, key=lambda f: f["timestamp"], reverse=True)})

        if method == "GET" and (m := re.fullmatch(f"/pairs/{CASE}__{CASE}", path)):
            result = app["shortlister"].pair(m.group(1), m.group(2), query.get("rank", "blind"))
            result["case_a"] = _annotate(app, result["case_a"])
            result["case_b"] = _annotate(app, result["case_b"])
            app["store"].record_audit({"actor_id": _actor(event), "timestamp": _now(), "action": "compare",
                                       "case_id": m.group(1), "other_case_id": m.group(2)})
            return _response(200, result)

        if method == "GET" and (m := re.fullmatch(f"/cases/{CASE}", path)):
            case = app["cases"].get(m.group(1))
            return _response(200, _annotate(app, case)) if case else _response(404, {"error": "unknown case"})

        if method == "GET" and (m := re.fullmatch(f"/cases/{CASE}/links", path)):
            scope = query.get("scope", "same")
            limit = int(query.get("limit", "10"))
            if not 1 <= limit <= 50:
                return _response(400, {"error": "limit must be 1-50"})
            result = app["shortlister"].shortlist(m.group(1), scope, limit, query.get("rank", "blind"))
            app["store"].record_audit({"actor_id": _actor(event), "timestamp": _now(), "action": "shortlist",
                                       "case_id": m.group(1), "scope": scope, "rank": result["rank"]})
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
