"""API contract, against the real serve bundle: pytest -q tests/test_api.py

Exercises handlers.api.handler with API Gateway v2 events — the same code
Lambda runs. Skipped when no bundle has been built (python -m linkage.bundle).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "data/serve"
pytestmark = pytest.mark.skipif(not (BUNDLE / "meta.json").exists(), reason="no serve bundle built")
TIERS = {"strong", "possible", "weak"}


@pytest.fixture(scope="module")
def call(tmp_path_factory):
    os.environ["BUNDLE_URI"] = str(BUNDLE)
    os.environ["STORE"] = f"local:{tmp_path_factory.mktemp('state')}"
    os.environ.pop("DEMO_GROUND_TRUTH", None)
    from handlers import api
    api._APP.clear()

    def _call(method, path, query=None, body=None):
        r = api.handler({"rawPath": path, "requestContext": {"http": {"method": method}},
                         "queryStringParameters": query, "body": json.dumps(body) if body else None})
        return r["statusCode"], (json.loads(r["body"]) if r["body"] else None)
    return _call


@pytest.fixture(scope="module")
def case_id(call):
    return call("GET", "/meta")[1]["demo_cases"][0]["case_id"]


def test_health_and_meta(call):
    assert call("GET", "/health") == (200, {"status": "ok"})
    status, meta = call("GET", "/meta")
    assert status == 200 and meta["synthetic"] is True and meta["demo_cases"]
    assert meta["ground_truth_marks"] is False                   # off unless explicitly enabled
    assert "lead_quality" not in meta                            # measured against truth: demo only
    assert meta["places"] and meta["crime_types"]


def test_shortlist_is_ranked_explained_and_never_a_probability(call, case_id):
    status, body = call("GET", f"/cases/{case_id}/links", {"scope": "all", "limit": "10"})
    assert status == 200 and set(body["lists"]) == {"same_type", "cross_type"}
    for lst in body["lists"].values():
        items = lst["items"]
        assert [i["rank"] for i in items] == list(range(1, len(items) + 1))
        assert all(a["bits"] >= b["bits"] for a, b in zip(items, items[1:]))
        assert all(a["one_in"] >= b["one_in"] for a, b in zip(items, items[1:]))   # rarity follows evidence
        assert case_id not in {i["case_id"] for i in items}      # never matched to itself
        for i in items:
            assert i["tier"] in TIERS
            assert sum(r["bits"] for r in i["reasons"]) == pytest.approx(i["bits"], abs=0.05)  # reasons add up
            assert "ground_truth_link" not in i
    assert "probab" not in json.dumps(body).lower()


def test_reasons_lead_with_timing_and_say_how_common_a_shared_habit_is(call, case_id):
    _, body = call("GET", f"/cases/{case_id}/links", {"scope": "same", "limit": "5"})
    for item in body["lists"]["same_type"]["items"]:
        kinds = [r["kind"] for r in item["reasons"]]
        assert kinds[0] == "timing"
        assert kinds == sorted(kinds, key=["timing", "shared", "different", "not_recorded"].index)
        for r in item["reasons"]:
            if r["kind"] == "shared" and r.get("share") is not None:
                assert 0 < r["share"] <= 1


def test_same_scope_returns_only_same_type(call, case_id):
    _, body = call("GET", f"/cases/{case_id}/links", {"scope": "same"})
    assert set(body["lists"]) == {"same_type"}
    assert {i["crime_type"] for i in body["lists"]["same_type"]["items"]} == {body["case"]["crime_type"]}


def test_pair_comparison_matches_the_shortlist(call, case_id):
    _, body = call("GET", f"/cases/{case_id}/links", {"scope": "same", "limit": "1"})
    top = body["lists"]["same_type"]["items"][0]
    status, pair = call("GET", f"/pairs/{case_id}__{top['case_id']}")
    assert status == 200 and pair["same_type"] is True
    assert pair["bits"] == pytest.approx(top["bits"], abs=0.01) and pair["tier"] == top["tier"]
    assert pair["case_a"]["case_id"] == case_id and "narrative_text" in pair["case_a"]
    assert call("GET", f"/pairs/{case_id}__{case_id}")[0] == 400


def test_leads_inbox_filters_and_pages(call):
    status, body = call("GET", "/leads", {"limit": "10"})
    assert status == 200 and body["total"] > 0 and len(body["items"]) == 10
    assert all("ground_truth_link" not in l for l in body["items"])
    _, cross = call("GET", "/leads", {"cross_state": "1", "limit": "100"})
    assert all(l["cross_state"] for l in cross["items"])
    ct = body["items"][0]["crime_type"]
    _, typed = call("GET", "/leads", {"crime_type": ct, "limit": "100"})
    assert typed["items"] and all(l["crime_type"] == ct for l in typed["items"])
    assert sum(body["lanes"].values()) == body["total"]         # every lead sits in exactly one lane
    for lane, count in body["lanes"].items():
        _, only = call("GET", "/leads", {"lane": lane, "limit": "100"})
        assert only["total"] == count and all(l["lane"] == lane for l in only["items"])
        assert all(l["cross_state"] == (lane == "cross_state") for l in only["items"])
    _, page2 = call("GET", "/leads", {"limit": "10", "offset": "10"})
    assert {l["a"] + l["b"] for l in page2["items"]}.isdisjoint({l["a"] + l["b"] for l in body["items"]})


def test_search_finds_a_case_by_fir_number(call, case_id):
    _, case = call("GET", f"/cases/{case_id}")
    _, hits = call("GET", "/search", {"q": case["fir_no"]})
    assert case_id in {h["case_id"] for h in hits["items"]}
    assert call("GET", "/search", {"q": "x"})[0] == 400


def test_feedback_is_validated_and_recorded(call, case_id):
    _, body = call("GET", f"/cases/{case_id}/links", {"scope": "same", "limit": "1"})
    other = body["lists"]["same_type"]["items"][0]["case_id"]
    for verdict in ("confirmed", "rejected", "investigate"):
        status, item = call("POST", f"/links/{case_id}__{other}/feedback", body={"verdict": verdict, "note": "x"})
        assert status == 201 and item["verdict"] == verdict
    assert call("POST", f"/links/{case_id}__{other}/feedback", body={"verdict": "maybe"})[0] == 400


@pytest.mark.parametrize("method, path, query, expected", [
    ("GET", "/cases/0000000000000000", None, 404),
    ("GET", "/cases/not-a-case-id", None, 404),
    ("GET", "/cases/{case}/links", {"scope": "everything"}, 400),
    ("GET", "/cases/{case}/links", {"limit": "500"}, 400),
    ("GET", "/leads", {"limit": "500"}, 400),
    ("GET", "/pairs/{case}__0000000000000000", None, 404),
    ("GET", "/nowhere", None, 404),
])
def test_bad_requests_fail_cleanly(call, case_id, method, path, query, expected):
    assert call(method, path.replace("{case}", case_id), query)[0] == expected
