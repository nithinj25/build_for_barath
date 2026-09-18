"""Retrieval stage and the full two-stage pipeline, on TEST offenders.

    python -m linkage.evaluate_retrieval --normalised data/final_sharing1_normalised.parquet \\
        --truth data/final_sharing1/truth.parquet --vectors data/final_sharing1_vectors \\
        --weights model/weights.json --out results/retrieval.json

Reported per pair class (never blended), for the same queries as
linkage.evaluate:
  recall@50        true partners the embedding retrieval keeps / all partners
  hit@50           queries with at least one partner retrieved
  pipeline hit@10  retrieval top-50 re-ranked by the scorer — what the
                   deployed system returns; partners retrieval missed count
                   as misses
  full-pool hit@10 the scorer ranking the whole pool with no retrieval, for
                   comparison: the gap is what retrieval costs

Caveat: narratives here are templated from the same recorded MO the scorer
uses, so embeddings partly re-encode the scored fields and retrieval looks
easier than it will on real FIR text.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from linkage import dataset, features
from linkage.retrieval import top_k
from linkage.score import Scorer

K, TOP = 50, 10


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m linkage.evaluate_retrieval", description=__doc__.splitlines()[0])
    ap.add_argument("--normalised", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--vectors", type=Path, required=True, help="directory from enrich.embed")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-queries", type=int, default=1500)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    weights = json.loads(args.weights.read_text(encoding="utf-8"))
    scorer = Scorer(weights)
    df = dataset.load(args.normalised, args.truth, weights["oracle_extraction"])
    ids = json.loads((args.vectors / "case_ids.json").read_text(encoding="utf-8"))
    vectors = np.load(args.vectors / "vectors.npy")
    position = {cid: i for i, cid in enumerate(ids)}
    vectors = vectors[[position[c] for c in df["case_id"]]]          # align rows with df

    test = dataset.split_of(df["offender_id"], weights["seed"]) == "test"
    types, offenders = df["crime_type"].to_numpy(), df["offender_id"].to_numpy()
    states = df["state_code"].to_numpy()
    serial = df["is_serial"].to_numpy()
    rng = np.random.default_rng(weights["seed"] + 1)
    report = {"k": K, "embedding": json.loads((args.vectors / "meta.json").read_text(encoding="utf-8"))["model"],
              "split": "test offenders (25%)", "classes": {}}

    for label, scope in (("same_type", "same"), ("cross_type", "other")):
        partners_of = {}
        for q in np.flatnonzero(serial & test):
            same_off = (offenders == offenders[q]) & (np.arange(len(df)) != q)
            mask = same_off & ((types == types[q]) if scope == "same" else (types != types[q]))
            if mask.any():
                partners_of[q] = np.flatnonzero(mask)
        queries = np.array(sorted(partners_of))
        if len(queries) > args.max_queries:
            queries = np.sort(rng.choice(queries, size=args.max_queries, replace=False))
        idx, _ = top_k(queries, vectors, K, groups=types, scope=scope)

        recall, hit50, pipe_hit, pipe_recall, random_recall = [], [], [], [], []
        found_by_state = {"same_state": [0, 0], "cross_state": [0, 0]}      # [found, total]
        for row, q in enumerate(queries):
            partners = set(partners_of[q].tolist())
            retrieved = idx[row]
            found = [c for c in retrieved if c in partners]
            recall.append(len(found) / len(partners))
            for c in partners:                   # the product is cross-jurisdiction: split by state
                key = "same_state" if states[c] == states[q] else "cross_state"
                found_by_state[key][1] += 1
                found_by_state[key][0] += int(c in found)
            hit50.append(float(bool(found)))
            pool = features.CROSS if scope == "other" else types[q]
            fields = scorer.pools[pool]["fields"]
            codes = scorer.encode(pool, {f: df[f].iloc[np.r_[q, retrieved]].tolist() for f in fields})
            bits = scorer.field_bits(pool, {f: codes[f][0] for f in fields}, {f: codes[f][1:] for f in fields})
            order = retrieved[np.argsort(-scorer.evidence(pool, bits), kind="stable")]
            top = [c for c in order[:TOP] if c in partners]
            pipe_hit.append(float(bool(top)))
            pipe_recall.append(len(top) / len(partners))
            pool_size = int(((types == types[q]) if scope == "same" else (types != types[q])).sum()) - (scope == "same")
            random_recall.append(min(K / pool_size, 1.0))

        full = json.loads(Path("results/post_core_fix/eval.json").read_text(encoding="utf-8"))
        full_hit = (full["same_type"] if scope == "same" else full["cross_type"])["fs_lr"]["hit_at_10"]
        report["classes"][label] = {
            "queries": int(len(queries)),
            "recall_at_50": round(float(np.mean(recall)), 4), "hit_at_50": round(float(np.mean(hit50)), 4),
            "random_recall_at_50": round(float(np.mean(random_recall)), 5),
            "pipeline_hit_at_10": round(float(np.mean(pipe_hit)), 4),
            "pipeline_recall_at_10": round(float(np.mean(pipe_recall)), 4),
            "full_pool_scorer_hit_at_10": full_hit,
            "recall_at_50_same_state_partners": round(found_by_state["same_state"][0] / max(found_by_state["same_state"][1], 1), 4),
            "recall_at_50_cross_state_partners": round(found_by_state["cross_state"][0] / max(found_by_state["cross_state"][1], 1), 4),
            "partners_same_state": found_by_state["same_state"][1],
            "partners_cross_state": found_by_state["cross_state"][1],
        }
        r = report["classes"][label]
        print(f"{label:<11} {r['queries']} queries | recall@50 {r['recall_at_50']:.3f} "
              f"(random {r['random_recall_at_50']:.4f}) hit@50 {r['hit_at_50']:.3f} | "
              f"pipeline hit@10 {r['pipeline_hit_at_10']:.3f} vs full-pool scorer {full_hit:.3f}")
        print(f"            recall@50 same-state partners {r['recall_at_50_same_state_partners']:.3f} "
              f"(n={r['partners_same_state']}) | cross-state partners {r['recall_at_50_cross_state_partners']:.3f} "
              f"(n={r['partners_cross_state']})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
