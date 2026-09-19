"""Evidence carried by TRUE links, per pool: python scripts/evidence_distribution.py \\
       --weights model/weights.json --out results/with_time/evidence_distribution.json

Held-out offenders only. The bundle uses the median and 90th percentile as
the "typical true link" context shown next to every lead, and the spec's
+5.85 example is placed against this distribution.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from linkage import dataset, features
from linkage.score import Scorer

STRONG_EXAMPLE = 5.845


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--normalised", type=Path, default=Path("data/final_sharing1_normalised.parquet"))
    ap.add_argument("--truth", type=Path, default=Path("data/final_sharing1/truth.parquet"))
    ap.add_argument("--weights", type=Path, default=Path("model/weights.json"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    weights = json.loads(args.weights.read_text(encoding="utf-8"))
    scorer = Scorer(weights)
    df = dataset.load(args.normalised, args.truth, weights["oracle_extraction"])
    test = dataset.split_of(df["offender_id"], weights["seed"]) == "test"
    out = {"source": f"{args.weights} on {args.normalised}, held-out offenders", "pools": {}}
    for pool in features.pools():
        fields = scorer.pools[pool]["fields"]
        codes = scorer.encode(pool, {**{f: df[f].tolist() for f in fields},
                                     features.DAYS: df[features.DAYS].tolist()})
        pos = dataset.positive_pairs(df, pool, test)
        bits = scorer.field_bits(pool, {k: v[pos[:, 0]] for k, v in codes.items()},
                                 {k: v[pos[:, 1]] for k, v in codes.items()})
        e = scorer.evidence(pool, bits)
        q = lambda p: round(float(np.percentile(e, p)), 2)
        out["pools"][pool] = {"true_pairs": int(len(pos)), "mean": round(float(e.mean()), 2), "median": q(50),
                              "p90": q(90), "p99": q(99), "max": round(float(e.max()), 2),
                              "share_at_or_above_strong_example": round(float((e >= STRONG_EXAMPLE).mean()), 4)}
        s = out["pools"][pool]
        print(f"{pool:<22} median {s['median']:+.2f}  p90 {s['p90']:+.2f}  max {s['max']:+.2f}  "
              f">= +5.85: {s['share_at_or_above_strong_example']:.1%}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
