"""Evaluate the scorer on held-out TEST offenders.

    python -m linkage.evaluate --normalised data/final_normalised.parquet \\
        --truth data/final/truth.parquet --weights data/weights.json --out data/eval.json

Each query is a test offender's case that has at least one true partner in
its pool. It is scored against EVERY other case in the pool — no blocking,
so "rank r of N" is literal — and ranked with random tie-breaking.

Reported per pool, then same-type and cross-type separately (never blended):
  hit@10       share of queries with a true partner in the top 10
  recall@10    true partners in the top 10 / all true partners
  precision@10 true partners in the top 10 / 10 (capped by how few partners exist)
  PR-AUC       mean average precision of each query's full ranking — honest
               under the real imbalance, unlike AUC on sampled pairs
  first rank   median rank of the best-ranked true partner
Baselines: random ranking, agreement_count (shared fields), fs (raw
Fellegi-Sunter bits). The evidence model is fs_lr. With a tuned ranking
(weights "rank", from linkage.tune) two more rows: blind = fs_lr +
distinctiveness (the default ranking) and nearby = blind + place (an officer's
"nearby first"). Each model also reports how many queries with a cross-state
partner found one in the top 10 — the cost of using place.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb, log2
from pathlib import Path

import numpy as np

from linkage import dataset, features, rank
from linkage.score import Scorer

TOP = 10
MODELS = ("agreement_count", "fs", "fs_lr")


def _query_metrics(scores: np.ndarray, partners: np.ndarray, tie: np.ndarray) -> dict:
    order = np.lexsort((tie, -scores))
    ranks = np.flatnonzero(partners[order]) + 1                 # 1-based ranks of true partners
    hits = int((ranks <= TOP).sum())
    return {"hit": float(hits > 0), "recall": hits / len(ranks), "precision": hits / TOP,
            "ap": float(np.mean(np.arange(1, len(ranks) + 1) / ranks)), "first_rank": int(ranks[0])}


def evaluate_pool(df, scorer: Scorer, pool: str, test_mask: np.ndarray, rng, max_queries: int,
                  ranker: rank.Ranker | None = None) -> dict | None:
    if pool not in scorer.pools:
        return None
    spec = scorer.pools[pool]
    fields = spec["fields"]
    columns = {f: df[f].tolist() for f in fields}
    if features.DAYS in df:
        columns[features.DAYS] = df[features.DAYS].tolist()
    codes = scorer.encode(pool, columns)
    rows = dataset.pool_rows(df, pool)
    types, offenders = df["crime_type"].to_numpy(), df["offender_id"].to_numpy()
    serial = df["is_serial"].to_numpy()

    def candidates(q):
        c = rows[rows != q]
        return c[types[c] != types[q]] if pool == features.CROSS else c

    queries = [q for q in rows[serial[rows] & test_mask[rows]] if (offenders[candidates(q)] == offenders[q]).any()]
    if not queries:
        return None
    sampled = len(queries) > max_queries
    if sampled:
        queries = rng.choice(queries, size=max_queries, replace=False)

    models = MODELS + (("blind", "nearby") if ranker is not None and pool != features.CROSS else ())
    per_model = {m: [] for m in models}
    states = df["state_code"].to_numpy()
    cross_found, cross_queries = {m: 0 for m in models}, 0
    random_precision, n_candidates, true_evidence = [], [], []
    for q in queries:
        cand = candidates(q)
        partners = offenders[cand] == offenders[q]
        a = {k: v[q] for k, v in codes.items()}
        b = {k: v[cand] for k, v in codes.items()}
        fb = scorer.field_bits(pool, a, b)
        scores = {"agreement_count": features.agreement_count(fields, spec["vocab"], a, b),
                  "fs": fb.sum(axis=1), "fs_lr": scorer.evidence(pool, fb)}
        if "blind" in models:
            near = ranker.terms(pool, q, cand, "nearby")
            scores["blind"] = scores["fs_lr"] + near["distinctiveness"]
            scores["nearby"] = scores["blind"] + near["place"]
        tie = rng.random(len(cand))
        far = partners & (states[cand] != states[q])
        cross_queries += bool(far.any())
        for m in models:
            per_model[m].append(_query_metrics(scores[m], partners, tie))
            if far.any():
                cross_found[m] += bool(far[np.lexsort((tie, -scores[m]))[:TOP]].any())
        random_precision.append(partners.sum() / len(cand))       # expected P@10 of a random ranking
        n_candidates.append(len(cand))
        true_evidence.extend(scores["fs_lr"][partners].tolist())

    def summary(ms):
        return {"hit_at_10": round(float(np.mean([x["hit"] for x in ms])), 4),
                "recall_at_10": round(float(np.mean([x["recall"] for x in ms])), 4),
                "precision_at_10": round(float(np.mean([x["precision"] for x in ms])), 4),
                "pr_auc": round(float(np.mean([x["ap"] for x in ms])), 4),
                "median_first_rank": int(np.median([x["first_rank"] for x in ms]))}

    all_off = dataset.positive_pairs(df, pool, np.ones(len(df), dtype=bool))
    total = dataset.pool_pair_count(df, pool)
    return {"queries": len(queries), "queries_sampled": bool(sampled),
            "mean_candidates": int(np.mean(n_candidates)),
            "prior_bits": round(log2(len(all_off) / (total - len(all_off))), 3) if len(all_off) else None,
            "mean_true_pair_evidence_bits": round(float(np.mean(true_evidence)), 3),
            "random": {"precision_at_10": round(float(np.mean(random_precision)), 6)},
            "cross_state_partner_in_top10": {m: [cross_found[m], cross_queries] for m in models},
            **{m: summary(per_model[m]) for m in models}}


def _pooled(results: dict, pools: list[str]) -> dict | None:
    parts = [results[p] for p in pools if results.get(p)]
    if not parts:
        return None
    n = sum(p["queries"] for p in parts)
    out = {"queries": n}
    for m in [m for m in (*MODELS, "blind", "nearby") if all(m in p for p in parts)]:
        out[m] = {k: round(sum(p[m][k] * p["queries"] for p in parts) / n, 4)
                  for k in ("hit_at_10", "recall_at_10", "precision_at_10", "pr_auc")}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m linkage.evaluate", description=__doc__.splitlines()[0])
    ap.add_argument("--normalised", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-queries", type=int, default=1500, help="per pool; sampled with the seed if exceeded")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    weights = json.loads(args.weights.read_text(encoding="utf-8"))
    scorer = Scorer(weights)
    df = dataset.load(args.normalised, args.truth, weights["oracle_extraction"])
    test = dataset.split_of(df["offender_id"], weights["seed"]) == "test"
    rng = np.random.default_rng(weights["seed"] + 1)

    ranker = None
    if weights.get("rank"):
        types = df["crime_type"].to_numpy()
        codes = {p: scorer.encode(p, {**{f: df[f].tolist() for f in scorer.pools[p]["fields"]},
                                      features.DAYS: df[features.DAYS].tolist()}) for p in scorer.pools}
        ranker = rank.Ranker(weights["rank"], rank.hub_scores(scorer, codes, types), types,
                             df["state_code"].to_numpy(), df["district"].to_numpy(), df["police_station"].to_numpy())
    results = {pool: evaluate_pool(df, scorer, pool, test, rng, args.max_queries, ranker) for pool in features.pools()}
    same_type = [p for p in features.pools() if p != features.CROSS]
    cross_found = {}
    for pool in same_type:
        for m, (found, total) in (results.get(pool) or {}).get("cross_state_partner_in_top10", {}).items():
            f0, t0 = cross_found.get(m, (0, 0))
            cross_found[m] = (f0 + found, t0 + total)
    report = {"weights": str(args.weights), "seed": weights["seed"],
              "oracle_extraction": weights["oracle_extraction"], "split": "test offenders (25%)",
              "same_type": _pooled(results, same_type), "cross_type": results.get(features.CROSS),
              "same_type_cross_state_partner_in_top10": {m: f"{a}/{b}" for m, (a, b) in cross_found.items()},
              "rank": weights.get("rank"), "pools": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    flag = " [ORACLE extraction — upper bound]" if weights["oracle_extraction"] else ""
    print(f"test offenders only{flag}\n")
    print(f"{'pool':<22}{'queries':>8}{'cands':>7}{'prior':>8}  {'model':<16}{'hit@10':>7}{'rec@10':>8}{'P@10':>7}{'PR-AUC':>8}{'rank1':>7}")
    for pool, r in results.items():
        if not r:
            continue
        for i, m in enumerate(m for m in (*MODELS, "blind", "nearby") if m in r):
            s = r[m]
            head = f"{pool:<22}{r['queries']:>8}{r['mean_candidates']:>7}{r['prior_bits']:>8}" if i == 0 else " " * 45
            print(f"{head}  {m:<16}{s['hit_at_10']:>7.3f}{s['recall_at_10']:>8.3f}{s['precision_at_10']:>7.3f}"
                  f"{s['pr_auc']:>8.3f}{s['median_first_rank']:>7}")
    for label, r in (("SAME-TYPE (pooled)", report["same_type"]), ("CROSS-TYPE", report["cross_type"])):
        for m in [m for m in ("fs_lr", "blind", "nearby") if r and m in r]:
            s = r[m]
            print(f"\n{label}: {m} hit@10 {s['hit_at_10']:.3f}  recall@10 {s['recall_at_10']:.3f}  "
                  f"P@10 {s['precision_at_10']:.3f}  PR-AUC {s['pr_auc']:.3f}  ({r['queries']} queries)")
    print("\nsame-type queries whose cross-state partner is in the top 10:", report["same_type_cross_state_partner_in_top10"])
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
