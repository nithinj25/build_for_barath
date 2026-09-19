"""Train the scorer locally and export weights.json. Never runs on AWS.

    python -m linkage.train --normalised data/final_normalised.parquet \\
        --truth data/final/truth.parquet --out weights.json [--oracle-extraction]

Per pool (each crime type, plus CROSS_TYPE):
  1. u  — value frequencies over every case in the pool (unlabelled)
  2. alpha per field — from same-offender pairs of TRAIN offenders
  3. logistic regression on per-field FS bits — corrects double counting of
     correlated fields; negatives are random different-offender pairs
  4. isotonic calibration on CALIB offenders, importance-weighted back to the
     pool's real link rate (internal thresholds only; never rendered)
TEST offenders (25%) are untouched here; linkage.evaluate uses them.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from linkage import dataset, features

NEG_PER_POS = 20
MIN_POSITIVES = 50


def train(df, seed: int, oracle_extraction: bool) -> dict:
    split = dataset.split_of(df["offender_id"], seed)
    rng = np.random.default_rng(seed)
    out = {"format": 1, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "seed": seed, "split": {"train": 0.60, "calib": 0.15, "test": 0.25, "by": "offender"},
           "oracle_extraction": oracle_extraction, "pools": {}}

    for pool in features.pools():
        fields = features.pool_fields(pool)
        rows = dataset.pool_rows(df, pool)
        pos = dataset.positive_pairs(df, pool, split == "train")
        cal_pos = dataset.positive_pairs(df, pool, split == "calib")
        if len(pos) < MIN_POSITIVES or len(cal_pos) < 10:
            print(f"  {pool}: skipped ({len(pos)} train / {len(cal_pos)} calib positives)")
            continue

        vocab = {f: dataset.vocab_for(df, f) for f in fields}
        codes = {f: features.encode(df[f].tolist(), f, vocab[f]) for f in fields}
        codes[features.DAYS] = df[features.DAYS].to_numpy()
        u = {f: features.fit_u(codes[f][rows], f, vocab[f]) for f in fields}
        alpha = {f: features.fit_alpha(codes[f][pos[:, 0]], codes[f][pos[:, 1]], u[f], f, vocab[f]) for f in fields}
        tables = {f: features.weight_table(u[f], alpha[f], f) for f in fields}

        neg = dataset.negative_pairs(df, pool, NEG_PER_POS * len(pos), rng)
        days = codes[features.DAYS]
        gap_bits = features.fit_gap_bits(features.gap_bins(days[pos[:, 0]], days[pos[:, 1]]),
                                         features.gap_bins(days[neg[:, 0]], days[neg[:, 1]]))

        def bits(pairs):
            return features.field_bits(tables, fields, {k: v[pairs[:, 0]] for k, v in codes.items()},
                                       {k: v[pairs[:, 1]] for k, v in codes.items()}, gap_bits)

        X = np.vstack([bits(pos), bits(neg)])
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        lr = LogisticRegression(max_iter=2000).fit(X, y)

        cal_neg = dataset.negative_pairs(df, pool, NEG_PER_POS * len(cal_pos), rng)
        Xc = np.vstack([bits(cal_pos), bits(cal_neg)])
        yc = np.r_[np.ones(len(cal_pos)), np.zeros(len(cal_neg))]
        est_true = len(cal_pos) / out["split"]["calib"]            # links among ALL offenders, estimated
        est_false = dataset.pool_pair_count(df, pool) - est_true
        weight = np.r_[np.full(len(cal_pos), est_true / len(cal_pos)), np.full(len(cal_neg), est_false / len(cal_neg))]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(lr.decision_function(Xc), yc, sample_weight=weight)

        out["pools"][pool] = {
            "fields": list(fields),
            "vocab": vocab,
            "u": {f: np.round(u[f], 6).tolist() for f in fields},
            "alpha": {f: round(alpha[f], 6) for f in fields},
            "time": {"edges": list(features.GAP_EDGES), "bits": np.round(gap_bits, 6).tolist()},
            "lr": {"coef": np.round(lr.coef_[0], 6).tolist(), "intercept": round(float(lr.intercept_[0]), 6)},
            "isotonic": {"x": np.round(iso.X_thresholds_, 6).tolist(), "y": np.round(iso.y_thresholds_, 8).tolist()},
            "training": {"cases": int(len(rows)), "train_positives": int(len(pos)),
                         "calib_positives": int(len(cal_pos)), "negatives_per_positive": NEG_PER_POS},
        }
        coef = dict(zip([*fields, "days_apart"], lr.coef_[0] / np.log(2)))
        top = sorted(coef.items(), key=lambda kv: -abs(kv[1] - 1))[:3]
        print(f"  {pool}: {len(rows)} cases, {len(pos)} train positives; "
              f"coef/ln2 furthest from 1: {', '.join(f'{k} {v:.2f}' for k, v in top)}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m linkage.train", description=__doc__.splitlines()[0])
    ap.add_argument("--normalised", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("weights.json"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--oracle-extraction", action="store_true",
                    help="fill free-text states' MO fields from truth (upper bound; not a pipeline result)")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    df = dataset.load(args.normalised, args.truth, args.oracle_extraction)
    print(f"training on {len(df)} cases (seed {args.seed}{', ORACLE extraction' if args.oracle_extraction else ''})")
    weights = train(df, args.seed, args.oracle_extraction)
    args.out.write_text(json.dumps(weights), encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB, {len(weights['pools'])} pools)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
