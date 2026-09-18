"""Manifest content: realised marginals, drift, link counts and priors, limitations."""
from __future__ import annotations

from math import comb, log2

import pandas as pd

from linkage import schema
from linkage.generate import sample

LIMITATIONS = [
    "SYNTHETIC. Evaluating on this corpus measures whether a model can invert this generator, not real-world performance.",
    "Every marginal and loading is an assumption (provenance tag 'assumption'); loadings were drafted by Claude at the project owner's request and need owner review.",
    "Narratives are templated English built from recorded values plus filler sentences. Cross-lingual retrieval cannot be demonstrated, and retrieval on templated text will look easier than on real FIRs.",
    "Background one-offs are generated at style s = 0 exactly (brief decision). Rare, heavily-loaded values are therefore over-represented among serial crimes specifically: a generator artifact a scorer can exploit, and one that moves with serial_case_fraction.",
    "Every case has a ground-truth offender, so non-links here are clean. Real unlabelled pairs are not confirmed non-links (positive-unlabelled); do not carry accuracy-style metrics over.",
    "Validator 4's check that state is not predictable after normalisation was not run: no normaliser exists yet.",
    "Within-crime implications are a small hand-written set (exit follows approach, vehicle class, vehicle_used; gas cutting implies gas cutter). Other real dependencies are absent.",
]


def _freq(frame: pd.DataFrame, f: str, v: str) -> float:
    if len(frame) == 0:
        return 0.0
    if f in schema.TAG_FIELDS:
        return round(float(frame[f].fillna("").str.split(";").apply(lambda xs: v in xs).mean()), 4)
    return round(float((frame[f] == v).mean()), 4)


def realised_marginals(truth: pd.DataFrame, cfg: dict) -> tuple[dict, list]:
    m = cfg["marginals"]
    serial = truth[truth["is_serial"]]
    out = {"crime_type": {}}
    drifts = []
    for ct, p in m["crime_types"]["share"].items():
        realised = round(float((truth["crime_type"] == ct).mean()), 4)
        out["crime_type"][ct] = {"config": p, "realised": realised,
                                 "serial_only": round(float((serial["crime_type"] == ct).mean()), 4)}
        drifts.append({"value": f"crime_type.{ct}", "config": p, "realised": realised,
                       "relative_drift": round(realised / p - 1, 3)})
    for ct in schema.CRIME_TYPES:
        all_ct, ser_ct = truth[truth["crime_type"] == ct], serial[serial["crime_type"] == ct]
        out[ct] = {}
        for f in schema.mo_fields(ct):
            spec = m["by_crime_type"][ct][f]
            dist = spec["rate"] if f in schema.TAG_FIELDS else spec["p"]
            out[ct][f] = {}
            for v, p in dist.items():
                realised = _freq(all_ct, f, v)
                out[ct][f][v] = {"config": p, "realised": realised, "serial_only": _freq(ser_ct, f, v)}
                if 0.02 <= p < 1 and (ct, f) not in sample.IMPLIED:
                    drifts.append({"value": f"{ct}.{f}.{v}", "config": p, "realised": realised,
                                   "relative_drift": round(realised / p - 1, 3)})
    drifts.sort(key=lambda d: -abs(d["relative_drift"]))
    return out, drifts[:15]


def core_agreement(truth: pd.DataFrame, sample_pairs: int = 20000, seed: int = 0) -> dict:
    """Do mo_core habits transfer across crime types?

    Agreement on categorical mo_core fields in TRUE values (no recording
    noise), for same-offender pairs against random pairs, split by whether the
    pair shares a crime type. Cross-type lift near zero means core habits are
    not person-level — the defect cross_type_sharing fixes — and no scorer can
    link across types on such a corpus.
    """
    import numpy as np

    fields = [f for f in schema.MO_CORE if f not in schema.TAG_FIELDS
              and not any((ct, f) in sample.IMPLIED for ct in schema.CRIME_TYPES)]
    t = truth[truth["ingested"]].reset_index(drop=True)
    types = t["crime_type"].to_numpy()
    rng = np.random.default_rng(seed)

    same, cross = [], []
    for _, idx in t[t["is_serial"]].groupby("offender_id").groups.items():
        idx = list(idx)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                (same if types[idx[a]] == types[idx[b]] else cross).append((idx[a], idx[b]))
    r = rng.integers(0, len(t), (4 * sample_pairs, 2))
    r = r[r[:, 0] != r[:, 1]]

    def rates(pairs):
        pairs = np.asarray(pairs)[:sample_pairs]
        if not len(pairs):
            return {}
        return {f: round(float((t[f].to_numpy()[pairs[:, 0]] == t[f].to_numpy()[pairs[:, 1]]).mean()), 3)
                for f in fields}

    out = {"fields": fields, "note": core_agreement.__doc__.strip().splitlines()[0]}
    for label, pairs in (("same_offender_same_type", same),
                         ("random_same_type", r[types[r[:, 0]] == types[r[:, 1]]]),
                         ("same_offender_cross_type", cross),
                         ("random_cross_type", r[types[r[:, 0]] != types[r[:, 1]]])):
        out[label] = rates(pairs)
    for scope in ("same_type", "cross_type"):
        a, b = out[f"same_offender_{scope}"], out[f"random_{scope}"]
        out[f"lift_{scope}"] = {f: round(a[f] - b[f], 3) for f in fields} if a and b else {}
        out[f"mean_lift_{scope}"] = round(sum(out[f"lift_{scope}"].values()) / len(fields), 3) if a and b else None
    return out


def _bits(true_pairs: int, total_pairs: int):
    return round(log2(true_pairs / (total_pairs - true_pairs)), 3) if 0 < true_pairs < total_pairs else None


def pair_stats(cases: pd.DataFrame) -> dict:
    """Ground-truth link counts and priors over the cases the pipeline ingests."""
    ing = cases[cases["ingested"]]
    ser = ing[ing["is_serial"]]
    n_type = ing["crime_type"].value_counts()
    by_off = ser.groupby("offender_id").size()
    by_off_type = ser.groupby(["offender_id", "crime_type"]).size()
    by_off_state = ser.groupby(["offender_id", "state_code"]).size()

    true_all = int(sum(comb(int(n), 2) for n in by_off))
    same = {}
    for ct in schema.CRIME_TYPES:
        sizes = by_off_type.xs(ct, level=1) if ct in by_off_type.index.get_level_values(1) else []
        t = int(sum(comb(int(n), 2) for n in sizes))
        same[ct] = {"cases": int(n_type.get(ct, 0)), "true_pairs": t,
                    "prior_bits": _bits(t, comb(int(n_type.get(ct, 0)), 2))}
    same_true = sum(v["true_pairs"] for v in same.values())
    all_pairs = comb(len(ing), 2)
    cross_pairs = all_pairs - sum(comb(int(n), 2) for n in n_type)
    same_state_true = int(sum(comb(int(n), 2) for n in by_off_state))
    return {
        "ingested_cases": int(len(ing)),
        "dropped_out_of_scope": int((~cases["ingested"]).sum()),
        "true_pairs_all": true_all,
        "prior_bits_all_property": _bits(true_all, all_pairs),
        "same_type": same,
        "cross_type": {"true_pairs": true_all - same_true,
                       "prior_bits": _bits(true_all - same_true, cross_pairs)},
        "cross_state_true_pairs": true_all - same_state_true,
    }
