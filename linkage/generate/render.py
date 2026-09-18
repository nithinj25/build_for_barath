"""Rendering: recorded canonical values → state-native feeds and narratives.

Narratives are written from RECORDED values, with two deliberate mismatches:
a field blank in the structured record is sometimes mentioned anyway (from the
true value — extraction adds information), and a filled field is sometimes
never mentioned (extraction must return null). Filler sentences carry detail
that is not in any scored field, so narrative text is not a pure re-encoding
of the MO columns. It is still templated English; see manifest limitations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from linkage import schema
from linkage.generate import rng as streams
from linkage.schema import ABSENT, MISSING, UNKNOWABLE

OPENINGS = {
    "MH": [
        "Complainant reports that a {crime} took place within {station} police station limits, {district}, between {frm} and {to}.",
        "As per the complaint, unknown persons committed {crime} in {district} ({station}) sometime between {frm} and {to}.",
        "On receipt of information, it was learnt that a {crime} occurred in {district} between {frm} and {to}.",
    ],
    "MP": [
        "Complainant appeared at {station} and stated that a {crime} happened between {frm} and {to} in {district}.",
        "Report received at {station}, {district}: {crime} between {frm} and {to}.",
        "The informant reported a {crime} in the jurisdiction of {station} between {frm} and {to}.",
    ],
    "KA": [
        "Brief facts: the complainant states that on or between {frm} and {to} a {crime} was committed in {district} limits.",
        "The complainant lodged a complaint at {station} regarding {crime} between {frm} and {to}.",
        "It is alleged that between {frm} and {to} unknown accused committed {crime} within {district}.",
    ],
    "TG": [
        "Complaint received at {station} ({district}) regarding {crime} between {frm} and {to}.",
        "The complainant reported {crime} that occurred between {frm} and {to} in {district}.",
        "Case registered at {station} on a complaint of {crime} committed between {frm} and {to}.",
    ],
}

FILLER = [
    "The complainant noticed the incident on returning and informed the police.",
    "Neighbours stated they did not hear anything unusual.",
    "The dog squad was summoned to the spot.",
    "A spot panchnama was drawn up in the presence of witnesses.",
    "Fingerprint experts visited the scene.",
    "The total value of the stolen property is estimated at Rs. {amount}.",
    "The complainant does not suspect anyone.",
    "It had been raining heavily that evening.",
    "Street lights in the area were reported not working.",
    "Nearby shops were closed at the time.",
    "The complainant has submitted bills for some of the items.",
    "Enquiries with nearby residents are in progress.",
    "A relative of the complainant was the first to notice.",
    "The beat constable had passed the area about an hour earlier.",
]

N_FIELDS = len(schema.ALL_MO_FIELDS)
WIDTH = 3 * N_FIELDS + 4            # per field: omit, mention, order; opening, 2 filler, amount


def _join(clauses: list[str]) -> str:
    return clauses[0] if len(clauses) == 1 else ", ".join(clauses[:-1]) + " and " + clauses[-1]


def _clauses(f: str, value: str, vocab: dict) -> list[str]:
    labels = vocab["fields"][f]
    if f in schema.TAG_FIELDS:
        tags = value.split(";") if value else []
        return [labels[t]["en"] for t in tags] or [labels["none_observed"]["en"]]
    return [labels[value]["en"]]


def _mo_clauses(case: dict, state_rec: dict, vocab: dict, u: np.ndarray, for_narrative: bool):
    ordered = []
    for i, f in enumerate(schema.ALL_MO_FIELDS):
        rec_v, true_v = case[f"rec_{f}"], case[f]
        u_omit, u_mention, u_order = u[3 * i: 3 * i + 3]
        if rec_v is None or rec_v == UNKNOWABLE:
            continue
        if rec_v in (MISSING, ABSENT):
            if not (for_narrative and u_mention < state_rec["narrative"]["mentions_unrecorded"]):
                continue
            value = true_v
        else:
            if for_narrative and u_omit < state_rec["narrative"]["omits_recorded"]:
                continue
            value = rec_v
        ordered.append((u_order, _clauses(f, value, vocab)))
    return [c for _, cs in sorted(ordered, key=lambda t: t[0]) for c in cs]


def _narrative(case: dict, code: str, st: dict, vocab: dict, u: np.ndarray, frm: str, to: str) -> str:
    opening = OPENINGS[code][min(int(u[-4] * 3), 2)].format(
        crime=vocab["crime_type"][case["rec_crime_type"]]["en"],
        station=case["police_station"], district=case["district"], frm=frm, to=to)
    parts = [opening]
    clauses = _mo_clauses(case, st["recording"], vocab, u, for_narrative=True)
    if clauses:
        parts.append(f"It is stated that the accused {_join(clauses)}.")
    i, j = int(u[-3] * len(FILLER)), int(u[-2] * len(FILLER))
    amount = f"{int(5000 + u[-1] * 495000):,}"
    for k in dict.fromkeys((min(i, len(FILLER) - 1), min(j, len(FILLER) - 1))):
        parts.append(FILLER[k].format(amount=amount))
    return " ".join(parts)


def _cell(f: str, value, code: str, vocab: dict, delim: str) -> str:
    if value is None or value == MISSING:
        return ""
    labels = vocab["fields"][f]
    if f in schema.TAG_FIELDS:
        tags = value.split(";") if value else []
        return delim.join(labels[t][code] for t in tags) if tags else labels["none_observed"][code]
    return labels[value][code]


def feeds(cases: pd.DataFrame, cfg: dict, vocab: dict, seed: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """State feeds (ingested cases only) and each case's position in its feed."""
    U = streams.stream(seed, streams.RENDER).random((int(cases["case_uid"].max()) + 1, WIDTH))
    out, positions = {}, []
    for code, st in cfg["states"]["states"].items():
        feed = st["feed"]
        sub = cases[(cases["state_code"] == code) & cases["ingested"]].sort_values(["registered_at", "case_id"])
        rows = []
        for row_idx, case in enumerate(sub.to_dict("records")):
            u = U[case["case_uid"]]
            occ_fmt = feed["date_only_format"] if case["rec_time_band"] == MISSING else feed["date_format"]
            frm, to = case["occurred_from"].strftime(occ_fmt), case["occurred_to"].strftime(occ_fmt)
            row = {}
            for canonical, native in feed["columns"].items():
                if canonical in ("fir_no", "district", "police_station"):
                    row[native] = case[canonical]
                elif canonical == "crime_type":
                    row[native] = vocab["crime_type"][case["rec_crime_type"]][code]
                elif canonical == "occurred_from":
                    row[native] = frm
                elif canonical == "occurred_to":
                    row[native] = to
                elif canonical == "registered_at":
                    row[native] = case["registered_at"].strftime(feed["date_format"])
                elif canonical == "narrative":
                    row[native] = _narrative(case, code, st, vocab, u, frm, to)
                elif canonical == schema.MO_DESCRIPTION:
                    clauses = _mo_clauses(case, st["recording"], vocab, u, for_narrative=False)
                    row[native] = ("Accused " + "; ".join(clauses) + ".") if clauses else ""
                else:
                    row[native] = _cell(canonical, case[f"rec_{canonical}"], code, vocab,
                                        feed["multi_value_delimiter"])
            rows.append(row)
            positions.append({"case_uid": case["case_uid"], "feed_row": row_idx, "feed_rows": len(sub)})
        out[code] = pd.DataFrame(rows, columns=list(feed["columns"].values()))
    return out, pd.DataFrame(positions)
