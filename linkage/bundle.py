"""Build the serve bundle the API loads: python -m linkage.bundle --out data/serve

    --normalised   canonical records (linkage.normalise output)
    --weights      trained scorer (model/weights.json)
    --extractions  optional: LLM-extracted MO for free-text states (enrich.extract
                   output); fills those fields with provenance "llm_extracted"
    --truth        optional: synthetic ground truth. Used to pick demo cases and,
                   with --with-ground-truth, to let demo mode mark true links.

Writes weights.json, index.json (case order and crime types), codes.npz
(per-pool encoded fields), cases.json.gz (display records) and meta.json
(per-pool context: typical true-link evidence and break-even, from results/).
Local build step only; the API reads the output.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from linkage import checks, dataset, features, rank, schema
from linkage.score import Scorer

DISPLAY = ("case_id", "state_code", "district", "police_station", "fir_no", "crime_type",
           "occurred_from", "occurred_to", "registered_at", "narrative_text", "mo_description",
           "needs_extraction")


def apply_extractions(df: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, int]:
    rows = {json.loads(l)["case_id"]: json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l}
    df = df.copy()
    provenance = df["field_provenance"].map(lambda p: json.loads(p) if isinstance(p, str) else dict(p or {}))
    applied = 0
    for i in np.flatnonzero(df["needs_extraction"].astype(bool).to_numpy()):
        r = rows.get(df["case_id"].iat[i])
        if r is None:
            continue
        prov = provenance.iat[i]
        for f, value in r["fields"].items():
            df.at[df.index[i], f] = schema.MISSING if value is None else value
            prov[f] = None if value is None else "llm_extracted"
        df.at[df.index[i], "needs_extraction"] = False
        applied += 1
    df["field_provenance"] = provenance.map(json.dumps)
    return df, applied


RANDOM_PAIRS = 200_000
LEADS_PER_CASE, LEADS_KEPT, LEAD_MIN_ONE_IN = 3, 5000, 1000
# Series edges: chosen on held-out pair precision (FINDINGS §12), not tuned per demo.
SERIES_NEAR_ONE_IN, SERIES_FAR_ONE_IN, SERIES_MIN_SIZE = 10_000, 200_000, 3
TYPE_CHECK_MIN_BITS = 2.0


def random_pair_rarity(scorer: Scorer, codes: dict, types: np.ndarray, seed: int,
                       ranker: rank.Ranker | None = None, mode: str = "blind") -> dict:
    """Scores of random UNRELATED-by-construction pairs per pool: the reference
    for "1 in N unrelated pairs look this alike". Scored exactly as the ranking
    scores (evidence + distinctiveness, + place in "nearby" mode). Stored as a
    coarse body (percentiles 0–99) plus the exact top 1%, where leads live."""
    rng = np.random.default_rng(seed)
    out = {}
    for pool, spec in scorer.pools.items():
        if mode == "nearby" and pool == features.CROSS:
            continue
        rows = np.arange(len(types)) if pool == features.CROSS else np.flatnonzero(types == pool)
        pairs = rows[rng.integers(0, len(rows), size=(RANDOM_PAIRS * 2, 2))]
        keep = pairs[:, 0] != pairs[:, 1]
        if pool == features.CROSS:
            keep &= types[pairs[:, 0]] != types[pairs[:, 1]]
        pairs = pairs[keep][:RANDOM_PAIRS]
        bits = scorer.field_bits(pool, {k: v[pairs[:, 0]] for k, v in codes[pool].items()},
                                 {k: v[pairs[:, 1]] for k, v in codes[pool].items()})
        ev = scorer.evidence(pool, bits)
        if ranker is not None:
            terms = ranker.pair_terms(pool, pairs[:, 0], pairs[:, 1], mode)
            ev = ev + terms["distinctiveness"] + terms["place"]
        ev = np.sort(ev)
        out[pool] = {"n": int(len(ev)), "body": np.round(np.percentile(ev, np.arange(100)), 4).tolist(),
                     "tail": np.round(ev[int(len(ev) * 0.99):], 4).tolist()}
    return out


LANES = ("same_district", "same_state", "cross_state")


def lane_of(ca: dict, cb: dict) -> str:
    """Where the two FIRs sit relative to each other. Never part of the score
    (a location-blind score is what lets cross-state links surface at all); it
    organises the inbox, because how often a lead is real differs sharply by lane."""
    if ca["state_code"] != cb["state_code"]:
        return "cross_state"
    return "same_district" if ca["district"] == cb["district"] else "same_state"


def top_matches(short, per_case: int = LEADS_PER_CASE) -> dict:
    """Every case against its whole same-type pool (no blocking). Returns
    {(a, b): (bits, times_seen)}; seen == 2 means mutual — each case is in the
    other's top matches."""
    scorer, types, codes = short.scorer, short.types, short.codes
    best: dict[tuple[int, int], float] = {}
    seen: dict[tuple[int, int], int] = {}
    for pool in np.unique(types):
        rows = np.flatnonzero(types == pool)
        cand_codes = {k: v[rows] for k, v in codes[pool].items()}
        for q in rows:
            ev = scorer.evidence(pool, scorer.field_bits(pool, {k: v[q] for k, v in codes[pool].items()}, cand_codes))
            ev = ev + short.ranker.terms(pool, q, rows, "blind")["distinctiveness"]
            ev[rows == q] = -np.inf
            for j in np.argpartition(-ev, per_case)[:per_case]:
                key = (min(q, rows[j]), max(q, rows[j]))
                best[key] = float(ev[j])
                seen[key] = seen.get(key, 0) + 1
    return {k: (best[k], seen[k]) for k in best}


def _shared_reasons(short, pool: str, a: int, b: int, n: int = 3) -> list[dict]:
    codes = short.codes[pool]
    row = short.scorer.field_bits(pool, {k: v[a] for k, v in codes.items()}, {k: v[b] for k, v in codes.items()})
    ca, cb = short.cases.get(short.case_ids[a], {}), short.cases.get(short.case_ids[b], {})
    shared = [r for r in short._reasons(pool, ca, cb, row) if r["kind"] == "shared"][:n]
    return [{"field": r["field"], "value": r.get("shared_value"), "share": r.get("share")} for r in shared]


def compute_leads(short, matches: dict, doubtful: set[str] = frozenset(), keep: int = LEADS_KEPT) -> list[dict]:
    """Candidate leads from every case's top matches, ranked by rarity so pools
    of different sizes mix fairly. Leads touching a FIR whose crime type looks
    wrong go last (flagged, not hidden): their rare-in-this-pool values make
    misfiled FIRs look alike (FINDINGS §13)."""
    types = short.types
    scored = []
    for (a, b), (bits, seen) in matches.items():
        pool = str(types[a])
        n, rarer = short.one_in(pool, bits)
        if n >= LEAD_MIN_ONE_IN:
            doubt = short.case_ids[a] in doubtful or short.case_ids[b] in doubtful
            scored.append((n, bits, a, b, pool, rarer, seen >= 2, doubt))
    scored.sort(key=lambda t: (t[7], -t[6], -t[0], -t[1]))    # sound type first, mutual, then rarest
    leads = []
    for n, bits, a, b, pool, rarer, mutual, doubt in scored[:keep]:
        ca, cb = short.cases[short.case_ids[a]], short.cases[short.case_ids[b]]
        tier_id, label = short.tier(n)
        leads.append({
            "a": short.case_ids[a], "b": short.case_ids[b], "pool": pool, "crime_type": pool,
            "bits": round(bits, 2), "one_in": n, "rarer_than_measured": rarer, "tier": tier_id, "tier_label": label,
            "cross_state": ca["state_code"] != cb["state_code"], "lane": lane_of(ca, cb), "mutual": mutual,
            "type_doubt": doubt,
            "same_station": ca.get("police_station") == cb.get("police_station"),
            "state_a": ca["state_code"], "district_a": ca["district"], "date_a": ca.get("occurred_from"),
            "state_b": cb["state_code"], "district_b": cb["district"], "date_b": cb.get("occurred_from"),
            "top_reasons": _shared_reasons(short, pool, a, b)})
    return leads


def compute_series(short, matches: dict, doubtful: set[str] = frozenset()) -> list[dict]:
    """Possible series: connected groups of FIRs joined by mutual strong matches.
    Transitive chaining turns weak edges into bogus mega-clusters (CLAUDE.md),
    so an edge must be mutual AND, inside one district, rarer than
    SERIES_NEAR_ONE_IN; across districts of one state, rarer than
    SERIES_FAR_ONE_IN. Cross-state edges are left out: in testing they were
    almost never real and chained unrelated cases into clusters of thousands.
    FIRs whose crime type looks wrong (`doubtful`, from type_checks) never join:
    two misfiled FIRs share values that are rare only in the wrong pool."""
    types, cases, ids = short.types, short.cases, short.case_ids
    parent: dict[int, int] = {}

    def root(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = []
    for (a, b), (bits, seen) in matches.items():
        ca, cb = cases[ids[a]], cases[ids[b]]
        if seen < 2 or ca["state_code"] != cb["state_code"] or ids[a] in doubtful or ids[b] in doubtful:
            continue
        pool = str(types[a])
        n, rarer = short.one_in(pool, bits)
        if n >= (SERIES_NEAR_ONE_IN if ca["district"] == cb["district"] else SERIES_FAR_ONE_IN):
            edges.append((a, b, n, rarer))
            parent[root(a)] = root(b)
    groups: dict[int, list[int]] = {}
    for x in parent:
        groups.setdefault(root(x), []).append(x)
    out = []
    for members in groups.values():
        if len(members) < SERIES_MIN_SIZE:
            continue
        member_set = set(members)
        pool = str(types[members[0]])
        members.sort(key=lambda i: (cases[ids[i]].get("occurred_from") or "", ids[i]))
        links = []
        for a, b, n, rarer in edges:
            if a in member_set:
                tier_id, label = short.tier(n)
                links.append({"a": ids[a], "b": ids[b], "one_in": n, "rarer_than_measured": rarer,
                              "tier": tier_id, "tier_label": label,
                              "top_reasons": _shared_reasons(short, pool, a, b)})
        rows = [cases[ids[i]] for i in members]
        dates = [r.get("occurred_from") for r in rows if r.get("occurred_from")]
        key = hashlib.sha1("|".join(sorted(ids[i] for i in members)).encode()).hexdigest()[:10]
        district_of = {ids[i]: cases[ids[i]]["district"] for i in members}
        near = sum(district_of[l["a"]] == district_of[l["b"]] for l in links) / len(links)
        out.append({
            "id": key, "crime_type": pool, "size": len(members), "near_share": round(near, 2),
            "members": [{k: r.get(k) for k in ("case_id", "fir_no", "police_station", "district", "state_code",
                                                "occurred_from")} for r in rows],
            "districts": sorted({r["district"] for r in rows}), "states": sorted({r["state_code"] for r in rows}),
            "first": min(dates) if dates else None, "last": max(dates) if dates else None,
            "weakest_one_in": min(l["one_in"] for l in links),
            "common_habits": _common_habits(short, pool, members),
            "links": links})
    # Larger groups and groups held together by same-district links were purer in testing (FINDINGS §12).
    out.sort(key=lambda r: (-r["size"], -r["near_share"], -r["weakest_one_in"]))
    return out


def _common_habits(short, pool: str, members: list[int]) -> list[dict]:
    """Values every FIR in the group records identically, rarest first."""
    spec = short.scorer.pools[pool]
    rows = [short.cases[short.case_ids[i]] for i in members]
    out = []
    for f in spec["fields"]:
        values = {r.get(f) for r in rows}
        if len(values) != 1:
            continue
        v = values.pop()
        if v is None or v in schema.TOKENS or f in schema.TAG_FIELDS or v not in spec["vocab"][f]:
            continue
        out.append({"field": f, "value": v, "share": round(float(spec["u"][f][spec["vocab"][f].index(v)]), 3)})
    return sorted(out, key=lambda h: h["share"])[:5]


def type_checks(short) -> list[dict]:
    """FIRs whose MO looks like another crime type in the same family — e.g. a
    'house burglary' with shutter entry into a shop closed for the holidays
    (linkage.checks). Uses no labels."""
    out = []
    for i, cid in enumerate(short.case_ids):
        c = short.cases[cid]
        flag = checks.type_check(short.scorer.pools, c, str(short.types[i]), TYPE_CHECK_MIN_BITS)
        if flag:
            out.append({"case_id": cid, **flag, "state_code": c["state_code"], "district": c["district"],
                        "fir_no": c.get("fir_no"), "police_station": c.get("police_station"),
                        "occurred_from": c.get("occurred_from")})
    return sorted(out, key=lambda r: -r["bits"])


def _held_out_share(df: pd.DataFrame, seed: int) -> float:
    """Share of all true same-type pairs that belong to held-out (test) offenders."""
    test = dataset.split_of(df["offender_id"], seed) == "test"
    everyone = np.ones(len(df), dtype=bool)
    pools = [p for p in schema.CRIME_TYPES]
    held = sum(len(dataset.positive_pairs(df, p, test)) for p in pools)
    total = sum(len(dataset.positive_pairs(df, p, everyone)) for p in pools)
    return held / total


def _precision(pairs: list[tuple[str, str]], offender: dict, split: dict, held_share: float) -> dict:
    """How many of these pairs are one offender, estimated without trusting the
    offenders the scorer was trained on.

    Restricting to pairs where BOTH cases are held-out offenders is biased: a
    true pair (one offender) survives that filter with probability ~0.25, a false
    pair (two offenders) with ~0.0625, so it inflates the odds about 4x. Instead
    count true pairs whose offender is held out, scale by the held-out share of
    all true pairs, and divide by every pair."""
    n = len(pairs)
    same = [(a, b) for a, b in pairs if offender[a] == offender[b]]
    held = sum(split[offender[a]] == "test" for a, _ in same)
    return {"n": n, "true_all_offenders": len(same), "true_held_out": held,
            "rate_all_offenders": round(len(same) / n, 4) if n else None,
            "rate": round(held / held_share / n, 4) if n else None}


def lead_quality(leads: list[dict], truth: Path, seed: int, normalised: Path) -> dict:
    """Share of leads that are true links, per lane (see _precision for why the
    estimate scales held-out hits rather than filtering to held-out pairs)."""
    df = dataset.load(normalised, truth, oracle_extraction=False)
    offender = dict(zip(df["case_id"], df["offender_id"]))
    split = dict(zip(df["offender_id"], dataset.split_of(df["offender_id"], seed)))
    held_share = _held_out_share(df, seed)
    groups = {"all": leads, "cross_state": [l for l in leads if l["cross_state"]],
              **{lane: [l for l in leads if l["lane"] == lane] for lane in LANES}}
    out = {name: _precision([(l["a"], l["b"]) for l in rows], offender, split, held_share) for name, rows in groups.items()}
    # chance: how often a random same-type pair is a true link, for the "x times better than chance" line
    true_pairs = sum(len(dataset.positive_pairs(df, p, np.ones(len(df), bool))) for p in schema.CRIME_TYPES)
    all_pairs = sum(dataset.pool_pair_count(df, p) for p in schema.CRIME_TYPES)
    out["chance_rate"] = true_pairs / all_pairs
    out["held_out_share"] = round(held_share, 4)
    return out


def demo_cases(normalised: Path, truth: Path, seed: int, n: int = 12) -> list[dict]:
    """A random sample of held-out cases that have a true same-type partner —
    chosen without looking at how well they score, so the demo is not cherry-picked."""
    df = dataset.load(normalised, truth, oracle_extraction=False)
    test = dataset.split_of(df["offender_id"], seed) == "test"
    serial = df[df["is_serial"] & test]
    counts = serial.groupby(["offender_id", "crime_type"])["case_id"].transform("count")
    eligible = serial[counts >= 2]
    pick = eligible.sample(n=min(n, len(eligible)), random_state=seed)
    return [{"case_id": c, "crime_type": t, "state_code": s}
            for c, t, s in zip(pick["case_id"], pick["crime_type"], pick["state_code"])]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m linkage.bundle", description=__doc__.splitlines()[0])
    ap.add_argument("--normalised", type=Path, default=Path("data/final_sharing1_normalised.parquet"))
    ap.add_argument("--weights", type=Path, default=Path("model/weights.json"))
    ap.add_argument("--extractions", type=Path, default=None)
    ap.add_argument("--truth", type=Path, default=Path("data/final_sharing1/truth.parquet"))
    ap.add_argument("--eval", type=Path, default=Path("results/ranked/eval.json"))
    ap.add_argument("--evidence", type=Path, default=Path("results/with_time/evidence_distribution.json"))
    ap.add_argument("--with-ground-truth", action="store_true", help="enable demo-mode true-link marks")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    weights = json.loads(args.weights.read_text(encoding="utf-8"))
    scorer = Scorer(weights)
    df = pd.read_parquet(args.normalised)
    df = df[df["crime_type"].notna()].reset_index(drop=True)
    applied = 0
    if args.extractions:
        df, applied = apply_extractions(df, args.extractions)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.weights, out / "weights.json")
    (out / "index.json").write_text(json.dumps({"case_ids": df["case_id"].tolist(),
                                                "crime_types": df["crime_type"].tolist()}), encoding="utf-8")
    arrays = {}
    days = dataset.day_numbers(df["occurred_from"]).tolist()
    for pool, spec in scorer.pools.items():
        encoded = scorer.encode(pool, {**{f: df[f].tolist() for f in spec["fields"]}, features.DAYS: days})
        for f, codes in encoded.items():
            arrays[f"{pool}|{f}"] = codes.astype(np.int16)
    np.savez_compressed(out / "codes.npz", **arrays)

    cases = {}
    for r in df.to_dict("records"):
        rec = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in r.items()
               if k in DISPLAY or k in schema.ALL_MO_FIELDS}
        rec["needs_extraction"] = bool(r["needs_extraction"])
        prov = r["field_provenance"]
        rec["field_provenance"] = json.loads(prov) if isinstance(prov, str) else prov
        cases[r["case_id"]] = rec
    with gzip.open(out / "cases.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(cases, fh)

    evaluation = json.loads(args.eval.read_text(encoding="utf-8"))
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    pools_meta = {}
    for pool in scorer.pools:
        prior = (evaluation["pools"].get(pool) or {}).get("prior_bits")
        dist = evidence["pools"].get(pool, {})
        pools_meta[pool] = {
            "breakeven_bits": round(-prior, 2) if prior is not None else None,
            "typical_true_link_bits": {"median": dist.get("median"), "p90": dist.get("p90")},
        }
    types = df["crime_type"].to_numpy()
    codes_by_pool: dict = {}
    for key, codes in arrays.items():
        pool, field = key.split("|", 1)
        codes_by_pool.setdefault(pool, {})[field] = codes.astype(np.int64)
    hub_r = rank.hub_scores(scorer, codes_by_pool, types) if weights.get("rank") else None
    if hub_r is not None:
        np.savez_compressed(out / "extras.npz", hub_r=hub_r.astype(np.float32))
    places = [df[c].to_numpy() for c in ("state_code", "district", "police_station")]
    ranker = rank.Ranker(weights.get("rank"), hub_r, types, *places)
    for mode, key in (("blind", "random_pairs"), ("nearby", "random_pairs_nearby")):
        if mode == "nearby" and not weights.get("rank"):
            continue
        for pool, r in random_pair_rarity(scorer, codes_by_pool, types, weights["seed"], ranker, mode).items():
            pools_meta[pool][key] = r

    places: dict = {}
    for s, d in zip(df["state_code"], df["district"]):
        places.setdefault(s, set()).add(d)
    # the default ranking ("blind": evidence + distinctiveness) when the weights carry one
    default = "blind" if evaluation.get("same_type", {}).get("blind") else "fs_lr"
    summary = {k: evaluation[k][default if k == "same_type" else "fs_lr"] for k in ("same_type", "cross_type") if evaluation.get(k)}
    if evaluation.get("same_type", {}).get("nearby"):
        summary["nearby"] = evaluation["same_type"]["nearby"]
        summary["evidence_only"] = evaluation["same_type"]["fs_lr"]
        summary["cross_state_partner_in_top10"] = evaluation.get("same_type_cross_state_partner_in_top10")
    meta = {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "synthetic": True,
            "cases": len(df), "weights_seed": weights["seed"], "extracted_cases": applied,
            "extraction_source": str(args.extractions) if args.extractions else None,
            "pools": pools_meta,
            "places": {s: sorted(d) for s, d in sorted(places.items())},
            "crime_types": sorted(set(types)),
            "model_summary": summary,
            "demo_cases": demo_cases(args.normalised, args.truth, weights["seed"]) if args.truth.exists() else [],
            "demo_cases_note": "random held-out cases with a true same-type partner; not selected by score",
            "ground_truth_available": bool(args.with_ground_truth)}

    from linkage.serve import Shortlister
    short = Shortlister(weights, df["case_id"].tolist(), types.tolist(), codes_by_pool, meta, cases, hub_r=hub_r)
    matches = top_matches(short)
    checks = type_checks(short)
    doubtful = {c["case_id"] for c in checks}
    leads = compute_leads(short, matches, doubtful)
    series = compute_series(short, matches, doubtful)
    for name, rows in (("leads", leads), ("series", series), ("checks", checks)):
        with gzip.open(out / f"{name}.json.gz", "wt", encoding="utf-8") as fh:
            json.dump(rows, fh)
    meta["leads"] = {"count": len(leads), "per_case": LEADS_PER_CASE, "min_one_in": LEAD_MIN_ONE_IN}
    meta["series"] = {"count": len(series), "near_one_in": SERIES_NEAR_ONE_IN, "far_one_in": SERIES_FAR_ONE_IN,
                      "multi_district": sum(len(s["districts"]) > 1 for s in series)}
    meta["checks"] = {"count": len(checks), "min_bits": TYPE_CHECK_MIN_BITS}
    if args.truth.exists():
        meta["lead_quality"] = lead_quality(leads, args.truth, weights["seed"], args.normalised)
        meta["series_quality"] = series_quality(series, args.truth, weights["seed"], args.normalised)
        meta["check_quality"] = type_check_quality(checks, args.truth)
        print("series", meta["series"], meta["series_quality"])
        print("checks", meta["check_quality"])
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.with_ground_truth:
        truth = pd.read_parquet(args.truth, columns=["case_id", "offender_id", "is_serial"])
        groups = {c: o for c, o, s in zip(truth["case_id"], truth["offender_id"], truth["is_serial"]) if s}
        (out / "truth_groups.json").write_text(json.dumps(groups), encoding="utf-8")

    size = sum(p.stat().st_size for p in out.iterdir()) / 1e6
    print(f"bundle → {out}: {len(df)} cases, {len(arrays)} code arrays, {applied} extracted, {size:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
