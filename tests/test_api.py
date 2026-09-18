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


def test_shortlist_is_ranked_explained_and_never_a_probability(call, case_id):
    status, body = call("GET", f"/cases/{case_id}/links", {"scope": "all", "limit": "10"})
    assert status == 200 and set(body["lists"]) == {"same_type", "cross_type"}
    for lst in body["lists"].values():
        items = lst["items"]
        assert [i["rank"] for i in items] == list(range(1, len(items) + 1))
        assert all(a["bits"] >= b["bits"] for a, b in zip(items, items[1:]))
        assert case_id not in {i["case_id"] for i in items}      # never matched to itself
        for i in items:                                          # contributions add up to the shown bits
            assert sum(c["bits"] for c in i["contributions"]) == pytest.approx(i["bits"], abs=0.05)
            assert "ground_truth_link" not in i
    assert "probab" not in json.dumps(body).lower()


def test_same_scope_returns_only_same_type(call, case_id):
    _, body = call("GET", f"/cases/{case_id}/links", {"scope": "same"})
    assert set(body["lists"]) == {"same_type"}
    assert {i["crime_type"] for i in body["lists"]["same_type"]["items"]} == {body["crime_type"]}


def test_feedback_is_validated_and_recorded(call, case_id):
    _, body = call("GET", f"/cases/{case_id}/links", {"scope": "same", "limit": "1"})
    other = body["lists"]["same_type"]["items"][0]["case_id"]
    status, item = call("POST", f"/links/{case_id}__{other}/feedback", body={"verdict": "rejected", "note": "x"})
    assert status == 201 and item["verdict"] == "rejected"
    assert call("POST", f"/links/{case_id}__{other}/feedback", body={"verdict": "maybe"})[0] == 400


@pytest.mark.parametrize("method, path, query, expected", [
    ("GET", "/cases/0000000000000000", None, 404),
    ("GET", "/cases/not-a-case-id", None, 404),
    ("GET", "/cases/{case}/links", {"scope": "everything"}, 400),
    ("GET", "/cases/{case}/links", {"limit": "500"}, 400),
    ("GET", "/nowhere", None, 404),
])
def test_bad_requests_fail_cleanly(call, case_id, method, path, query, expected):
    assert call(method, path.replace("{case}", case_id), query)[0] == expected
