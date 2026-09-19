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
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from linkage import dataset, features, schema
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


def random_pair_rarity(scorer: Scorer, codes: dict, types: np.ndarray, seed: int) -> dict:
    """Evidence of random UNRELATED-by-construction pairs per pool: the reference
    for "1 in N unrelated pairs look this alike". Stored as a coarse body
    (percentiles 0–99) plus the exact top 1%, where leads live."""
    rng = np.random.default_rng(seed)
    out = {}
    for pool, spec in scorer.pools.items():
        rows = np.arange(len(types)) if pool == features.CROSS else np.flatnonzero(types == pool)
        pairs = rows[rng.integers(0, len(rows), size=(RANDOM_PAIRS * 2, 2))]
        keep = pairs[:, 0] != pairs[:, 1]
        if pool == features.CROSS:
            keep &= types[pairs[:, 0]] != types[pairs[:, 1]]
        pairs = pairs[keep][:RANDOM_PAIRS]
        bits = scorer.field_bits(pool, {k: v[pairs[:, 0]] for k, v in codes[pool].items()},
                                 {k: v[pairs[:, 1]] for k, v in codes[pool].items()})
        ev = np.sort(scorer.evidence(pool, bits))
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


def compute_leads(short, per_case: int = LEADS_PER_CASE, keep: int = LEADS_KEPT) -> list[dict]:
    """Every case against its whole same-type pool; its top matches become
    candidate leads. Ranked by rarity so pools of different sizes mix fairly."""
    scorer, types, codes = short.scorer, short.types, short.codes
    best: dict[tuple[int, int], float] = {}
    seen: dict[tuple[int, int], int] = {}          # 2 = mutual: each case is in the other's top matches
    for pool in np.unique(types):
        rows = np.flatnonzero(types == pool)
        cand_codes = {k: v[rows] for k, v in codes[pool].items()}
        for q in rows:
            ev = scorer.evidence(pool, scorer.field_bits(pool, {k: v[q] for k, v in codes[pool].items()}, cand_codes))
            ev[rows == q] = -np.inf
            for j in np.argpartition(-ev, per_case)[:per_case]:
                key = (min(q, rows[j]), max(q, rows[j]))
                best[key] = float(ev[j])
                seen[key] = seen.get(key, 0) + 1
    scored = []
    for (a, b), bits in best.items():
        pool = str(types[a])
        n, rarer = short.one_in(pool, bits)
        if n >= LEAD_MIN_ONE_IN:
            scored.append((n, bits, a, b, pool, rarer, seen[(a, b)] >= 2))
    scored.sort(key=lambda t: (-t[6], -t[0], -t[1]))          # mutual first, then rarest
    leads = []
    for n, bits, a, b, pool, rarer, mutual in scored[:keep]:
        ca, cb = short.cases[short.case_ids[a]], short.cases[short.case_ids[b]]
        row = scorer.field_bits(pool, {k: v[a] for k, v in codes[pool].items()},
                                {k: v[b] for k, v in codes[pool].items()})
        shared = [r for r in short._reasons(pool, a, b, row) if r["kind"] == "shared"][:3]
        tier_id, label = short.tier(n)
        leads.append({
            "a": short.case_ids[a], "b": short.case_ids[b], "pool": pool, "crime_type": pool,
            "bits": round(bits, 2), "one_in": n, "rarer_than_measured": rarer, "tier": tier_id, "tier_label": label,
            "cross_state": ca["state_code"] != cb["state_code"], "lane": lane_of(ca, cb), "mutual": mutual,
            "same_station": ca.get("police_station") == cb.get("police_station"),
            "state_a": ca["state_code"], "district_a": ca["district"], "date_a": ca.get("occurred_from"),
            "state_b": cb["state_code"], "district_b": cb["district"], "date_b": cb.get("occurred_from"),
            "top_reasons": [{"field": r["field"], "value": r.get("shared_value"), "share": r.get("share")}
                            for r in shared]})
    return leads


def lead_quality(leads: list[dict], truth: Path, seed: int, normalised: Path) -> dict:
    """Share of leads that are true links, per tier — held-out offenders only, so
    offenders the scorer was trained on cannot flatter the number."""
    t = pd.read_parquet(truth, columns=["case_id", "offender_id"])
    offender = dict(zip(t["case_id"], t["offender_id"]))
    split = dict(zip(t["offender_id"], dataset.split_of(t["offender_id"], seed)))
    held_out = [l for l in leads if split[offender[l["a"]]] == "test" and split[offender[l["b"]]] == "test"]
    groups = {"all": held_out,
              "mutual": [l for l in held_out if l["mutual"]],
              "one_sided": [l for l in held_out if not l["mutual"]],
              "cross_state": [l for l in held_out if l["cross_state"]],
              "mutual_cross_state": [l for l in held_out if l["mutual"] and l["cross_state"]],
              **{lane: [l for l in held_out if l["lane"] == lane] for lane in LANES}}
    out = {}
    for name, rows in groups.items():
        hits = sum(offender[l["a"]] == offender[l["b"]] for l in rows)
        out[name] = {"leads": len(rows), "true_links": hits,
                     "true_link_rate": round(hits / len(rows), 3) if rows else None}
    # chance: how often a random same-type pair is a true link, for the "x times better than chance" line
    df = dataset.load(normalised, truth, oracle_extraction=False)
    true_pairs = sum(len(dataset.positive_pairs(df, p, np.ones(len(df), bool))) for p in schema.CRIME_TYPES)
    all_pairs = sum(dataset.pool_pair_count(df, p) for p in schema.CRIME_TYPES)
    out["chance_rate"] = true_pairs / all_pairs
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
    ap.add_argument("--eval", type=Path, default=Path("results/with_time/eval.json"))
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
    rarity = random_pair_rarity(scorer, codes_by_pool, types, weights["seed"])
    for pool, r in rarity.items():
        pools_meta[pool]["random_pairs"] = r

    places: dict = {}
    for s, d in zip(df["state_code"], df["district"]):
        places.setdefault(s, set()).add(d)
    summary = {k: evaluation[k]["fs_lr"] for k in ("same_type", "cross_type") if evaluation.get(k)}
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
    short = Shortlister(weights, df["case_id"].tolist(), types.tolist(), codes_by_pool, meta, cases)
    leads = compute_leads(short)
    with gzip.open(out / "leads.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(leads, fh)
    meta["leads"] = {"count": len(leads), "per_case": LEADS_PER_CASE, "min_one_in": LEAD_MIN_ONE_IN}
    if args.truth.exists():
        meta["lead_quality"] = lead_quality(leads, args.truth, weights["seed"], args.normalised)
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
