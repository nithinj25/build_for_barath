"""Recording: what each state actually wrote down about a true crime.

Three independent per-state mechanisms — confusion along the confusability
graph, per-field dropout, structural absence — plus crime-type confusion at
genuine boundaries. Output stays in canonical vocabulary (render.py makes it
state-native). Uniforms are indexed by case_uid, so truth held fixed and the
corruption seed varied changes only recording noise.
"""
from __future__ import annotations

import pandas as pd

from linkage import schema
from linkage.generate import rng as streams
from linkage.generate.sample import Model
from linkage.schema import ABSENT, MISSING, TOKENS, UNKNOWABLE  # noqa: F401  (re-exported)


def _adjacency(edges: list) -> dict[str, list[str]]:
    adj: dict[str, list[str]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    return adj


def _layout(model: Model) -> tuple[dict, int]:
    """Categorical: [confuse, which, drop]. Tags: [confuse, which] per value slot + [drop]."""
    slots, pos = {}, 2                       # 0, 1: crime-type confuse / which
    for f in schema.ALL_MO_FIELDS:
        width = 3
        if f in schema.TAG_FIELDS:
            s = model.slots[f]
            width = 2 * (s.stop - s.start) + 1
        slots[f] = slice(pos, pos + width)
        pos += width
    return slots, pos


def _pick(options: list[str], u: float) -> str:
    return options[min(int(u * len(options)), len(options) - 1)]


def record(truth: pd.DataFrame, cfg: dict, seed: int, model: Model) -> pd.DataFrame:
    """One row per case_uid: rec_crime_type, ingested, rec_<field>."""
    states = cfg["states"]["states"]
    oos = set(cfg["states"]["out_of_scope_crime_types"])
    graph = {f: _adjacency(e) for f, e in cfg["states"]["confusability"].items()}
    layout, width = _layout(model)
    U = streams.stream(seed, streams.CORRUPTION).random((int(truth["case_uid"].max()) + 1, width))

    rows = []
    for r in truth.to_dict("records"):
        rec = states[r["state_code"]]["recording"]
        u = U[r["case_uid"]]
        true_ct = r["crime_type"]

        rec_ct = true_ct
        ct_nbs = graph["crime_type"].get(true_ct, [])
        if ct_nbs and u[0] < rec["crime_type_confusion"]:
            rec_ct = _pick(ct_nbs, u[1])
        out = {"case_uid": r["case_uid"], "rec_crime_type": rec_ct,
               "ingested": rec_ct not in oos}

        applicable = set(schema.mo_fields(true_ct))
        absent = set(rec["structurally_absent"] or [])
        drop_rates = rec["dropout"]["fields"] or {}
        for f in schema.ALL_MO_FIELDS:
            key = f"rec_{f}"
            if f not in applicable:
                out[key] = None
                continue
            if f in absent:
                out[key] = ABSENT
                continue
            uf = u[layout[f]]
            dropped = uf[-1] < drop_rates.get(f, rec["dropout"]["default"])

            if f == "time_band":
                out[key] = MISSING if dropped else schema.recorded_time_band(r["occurred_from"], r["occurred_to"])
                continue
            if dropped:
                out[key] = MISSING
                continue

            fm = model.fields[true_ct, f]
            valid = set(fm.values)
            adj = graph[f]
            if fm.tag:
                present = set(r[f].split(";")) if r[f] else set()
                recorded = set()
                for j, tag in enumerate(fm.values):
                    if tag not in present:
                        continue
                    options = [x for x in adj.get(tag, []) if x in valid and x not in present]
                    if options and uf[2 * j] < rec["value_confusion"]:
                        recorded.add(_pick(options, uf[2 * j + 1]))
                    else:
                        recorded.add(tag)
                out[key] = ";".join(v for v in fm.values if v in recorded)
            else:
                v = r[f]
                options = [x for x in adj.get(v, []) if x in valid]
                if options and uf[0] < rec["value_confusion"]:
                    v = _pick(options, uf[1])
                out[key] = v
        rows.append(out)
    return pd.DataFrame(rows)
