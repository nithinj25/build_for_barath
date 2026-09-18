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
    ap.add_argument("--eval", type=Path, default=Path("results/post_core_fix/eval.json"))
    ap.add_argument("--evidence", type=Path, default=Path("results/post_core_fix/evidence_distribution.json"))
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
    for pool, spec in scorer.pools.items():
        encoded = scorer.encode(pool, {f: df[f].tolist() for f in spec["fields"]})
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
    meta = {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "synthetic": True,
            "cases": len(df), "weights_seed": weights["seed"], "extracted_cases": applied,
            "extraction_source": str(args.extractions) if args.extractions else None,
            "pools": pools_meta,
            "demo_cases": demo_cases(args.normalised, args.truth, weights["seed"]) if args.truth.exists() else [],
            "demo_cases_note": "random held-out cases with a true same-type partner; not selected by score",
            "ground_truth_available": bool(args.with_ground_truth)}
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
