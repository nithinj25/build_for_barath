"""Tune the ranking on CALIB offenders and add it to the weights. Local only.

    python -m linkage.tune --weights model/weights.json --out model/weights.json

1. place bits: log2 P(where | same offender) / P(where | random pair), TRAIN
   offenders, same-type pools, bins from linkage.rank.GEO_BINS
2. distinctiveness: r per case (label-free), then hub_beta chosen by pooled
   same-type PR-AUC on CALIB queries
3. geo_beta for "nearby first", chosen the same way with hub_beta fixed
TEST offenders are untouched; linkage.evaluate reports them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from linkage import dataset, evaluate, features, rank
from linkage.score import Scorer

HUB_GRID = (0.0, 0.15, 0.25, 0.35, 0.5)
GEO_GRID = (0.25, 0.5, 0.75, 1.0)
CALIB_QUERIES = 700


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m linkage.tune", description=__doc__.splitlines()[0])
    ap.add_argument("--normalised", type=Path, default=Path("data/final_sharing1_normalised.parquet"))
    ap.add_argument("--truth", type=Path, default=Path("data/final_sharing1/truth.parquet"))
    ap.add_argument("--weights", type=Path, default=Path("model/weights.json"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    weights = json.loads(args.weights.read_text(encoding="utf-8"))
    weights.pop("rank", None)
    scorer = Scorer(weights)
    seed = weights["seed"]
    df = dataset.load(args.normalised, args.truth, weights["oracle_extraction"])
    split = dataset.split_of(df["offender_id"], seed)
    types = df["crime_type"].to_numpy()
    st, di, ps = (df[c].to_numpy() for c in ("state_code", "district", "police_station"))
    same_type = [p for p in features.pools() if p != features.CROSS and p in scorer.pools]

    rng = np.random.default_rng(seed + 3)
    same_bins, random_bins = [], []
    for pool in same_type:
        pos = dataset.positive_pairs(df, pool, split == "train")
        neg = dataset.negative_pairs(df, pool, 20 * len(pos), rng)
        same_bins.append(rank.geo_bins(st[pos[:, 0]], di[pos[:, 0]], ps[pos[:, 0]], st[pos[:, 1]], di[pos[:, 1]], ps[pos[:, 1]]))
        random_bins.append(rank.geo_bins(st[neg[:, 0]], di[neg[:, 0]], ps[neg[:, 0]], st[neg[:, 1]], di[neg[:, 1]], ps[neg[:, 1]]))
    geo_bits = rank.fit_geo_bits(np.concatenate(same_bins), np.concatenate(random_bins))
    print("place bits:", dict(zip(rank.GEO_BINS, geo_bits.round(2))))

    codes = {p: scorer.encode(p, {**{f: df[f].tolist() for f in scorer.pools[p]["fields"]},
                                  features.DAYS: df[features.DAYS].tolist()}) for p in scorer.pools}
    r = rank.hub_scores(scorer, codes, types)
    calib = split == "calib"

    def pr_auc(hub_beta, geo_beta, model):
        ranker = rank.Ranker({"hub_beta": hub_beta, "geo_beta": geo_beta, "geo_bits": geo_bits.tolist()}, r, types, st, di, ps)
        rng_q = np.random.default_rng(seed + 2)
        res = {p: evaluate.evaluate_pool(df, scorer, p, calib, rng_q, CALIB_QUERIES, ranker) for p in same_type}
        return evaluate._pooled(res, same_type)[model]["pr_auc"]

    hub_scores = {b: pr_auc(b, 0.0, "blind") for b in HUB_GRID}
    hub_beta = max(hub_scores, key=hub_scores.get)
    print("calib PR-AUC by hub_beta:", hub_scores, "->", hub_beta)
    geo_scores = {b: pr_auc(hub_beta, b, "nearby") for b in GEO_GRID}
    geo_beta = max(geo_scores, key=geo_scores.get)
    print("calib PR-AUC by geo_beta:", geo_scores, "->", geo_beta)

    weights["rank"] = {"hub_beta": hub_beta, "hub_k": rank.HUB_K, "geo_beta": geo_beta,
                       "geo_bins": list(rank.GEO_BINS), "geo_bits": np.round(geo_bits, 4).tolist(),
                       "tuned_on": "calib offenders, pooled same-type PR-AUC",
                       "calib_pr_auc": {"hub": hub_scores, "geo": geo_scores}}
    args.out.write_text(json.dumps(weights), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
