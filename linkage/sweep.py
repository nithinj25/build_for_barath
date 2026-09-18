"""Two-dimensional sweep: repeat_rate × cross_type_sharing.

    python -m linkage.sweep --out results/sweep [--n-cases 45000]

For each grid point: generate truth, apply recording, take the pipeline's view
of it, train on TRAIN offenders and evaluate on TEST offenders. One JSON per
point (resumable), plus summary.json.

Neither axis has a defensible default. repeat_rate says how consistent an
offender is within a crime type; cross_type_sharing says how much of their
mo_core habit carries across types. No published figure fixes either, so the
result is the curve, never a single point.

The pipeline view is built from the recorded canonical values rather than by
re-rendering feeds and parsing them back: linkage.normalise is verified to
reproduce those values exactly for structured states (44,533 of 44,533), so
this skips ~30 s per point without changing what the scorer sees. Free-text
states keep their MO fields unset, exactly as the pipeline has them until
extraction runs.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from linkage import config as config_mod
from linkage import dataset, evaluate, features, schema, train
from linkage.generate import corrupt, report, sample, validate
from linkage.generate import rng as streams
from linkage.score import Scorer

REPEAT_RATES = [0.2, 0.35, 0.5, 0.65, 0.8, 0.9]
SHARING = [0.0, 0.25, 0.5, 0.75, 1.0]


def pipeline_view(cases: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """What the scorer sees: recorded canonical values for ingested cases."""
    free_text = {code for code, st in cfg["states"]["states"].items()
                 if st["feed"]["layout"] == "free_text_mo"}
    ing = cases[cases["ingested"]].reset_index(drop=True)
    view = pd.DataFrame({
        "case_id": ing["case_id"], "offender_id": ing["offender_id"], "is_serial": ing["is_serial"],
        "state_code": ing["state_code"], "crime_type": ing["rec_crime_type"],
        "needs_extraction": ing["state_code"].isin(free_text),
    })
    for f in schema.ALL_MO_FIELDS:
        col = ing[f"rec_{f}"]
        view[f] = col.where(~view["needs_extraction"] | (f in schema.DERIVED_AT_NORMALISATION), None)
    return view


def run_point(cfg: dict, repeat_rate: float, sharing: float, seed: int, max_queries: int) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["corpus"]["repeat_rate"]["value"] = repeat_rate
    cfg["corpus"]["cross_type_sharing"] = sharing
    started = time.time()

    truth, offenders = sample.generate(cfg, seed)
    cases = truth.merge(corrupt.record(truth, cfg, seed, sample.build_model(cfg)), on="case_uid")
    view = pipeline_view(cases, cfg)

    weights = train.train(view, seed, oracle_extraction=False)
    scorer = Scorer(weights)
    test = dataset.split_of(view["offender_id"], seed) == "test"
    rng = np.random.default_rng(seed + 1)
    pools = {p: evaluate.evaluate_pool(view, scorer, p, test, rng, max_queries) for p in features.pools()}

    vrng = streams.stream(seed, 99)
    v2 = validate.truth_mi(truth, vrng)
    v3 = validate.recorded_mi(cases, v2, vrng)
    same = [p for p in features.pools() if p != features.CROSS]
    return {
        "repeat_rate": repeat_rate, "cross_type_sharing": sharing, "seed": seed,
        "n_cases": cfg["corpus"]["n_cases"], "alpha": config_mod.alpha(repeat_rate, 0.10),
        "seconds": round(time.time() - started, 1),
        "core_habit_transfer": {k: v for k, v in report.core_agreement(cases).items() if k.startswith("mean_lift")},
        "pairs": report.pair_stats(cases),
        "validators": {"2_truth_mi": {k: v2[k] for k in ("passed", "excess_bits")},
                       "3_recorded_mi": {k: v3[k] for k in ("passed", "excess_bits", "excess_kept_vs_truth")}},
        "same_type": evaluate._pooled(pools, same), "cross_type": pools.get(features.CROSS),
        "pools": pools,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m linkage.sweep", description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("results/sweep"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-cases", type=int, default=45000)
    ap.add_argument("--max-queries", type=int, default=600, help="per pool per point")
    ap.add_argument("--repeat-rates", type=float, nargs="*", default=REPEAT_RATES)
    ap.add_argument("--sharing", type=float, nargs="*", default=SHARING)
    ap.add_argument("--force", action="store_true", help="recompute points that already have a file")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = config_mod.load()
    readiness = config_mod.check(cfg)
    if readiness.errors or readiness.unfilled:
        print("config not ready — run `python -m linkage.config -v`")
        return 1
    cfg["corpus"]["n_cases"] = args.n_cases
    args.out.mkdir(parents=True, exist_ok=True)

    points = [(rr, s) for rr in args.repeat_rates for s in args.sharing]
    print(f"{len(points)} points, {args.n_cases} cases each, seed {args.seed}")
    for i, (rr, sharing) in enumerate(points, 1):
        path = args.out / f"rr{rr:g}_share{sharing:g}.json"
        if path.exists() and not args.force:
            print(f"[{i}/{len(points)}] {path.name} exists, skipped")
            continue
        result = run_point(cfg, rr, sharing, args.seed, args.max_queries)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        st, ct = result["same_type"]["fs_lr"], result["cross_type"]["fs_lr"]
        print(f"[{i}/{len(points)}] repeat_rate {rr:<4} sharing {sharing:<4} "
              f"same-type hit@10 {st['hit_at_10']:.3f}  cross-type hit@10 {ct['hit_at_10']:.3f}  "
              f"core lift cross {result['core_habit_transfer']['mean_lift_cross_type']:+.3f}  "
              f"({result['seconds']:.0f}s)")

    summary = []
    for path in sorted(args.out.glob("rr*_share*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        summary.append({
            "repeat_rate": r["repeat_rate"], "cross_type_sharing": r["cross_type_sharing"],
            "alpha": round(r["alpha"], 3),
            "core_lift_same_type": r["core_habit_transfer"]["mean_lift_same_type"],
            "core_lift_cross_type": r["core_habit_transfer"]["mean_lift_cross_type"],
            "same_type": r["same_type"]["fs_lr"], "cross_type": r["cross_type"]["fs_lr"],
            "same_type_mean_true_pair_bits": round(float(np.mean(
                [p["mean_true_pair_evidence_bits"] for k, p in r["pools"].items()
                 if p and k != features.CROSS])), 3),
            "cross_type_mean_true_pair_bits": r["cross_type"]["mean_true_pair_evidence_bits"],
            "cross_type_prior_bits": r["cross_type"]["prior_bits"],
            "validators": r["validators"],
        })
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out / 'summary.json'} ({len(summary)} points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
