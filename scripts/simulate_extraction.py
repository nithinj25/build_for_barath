"""Is running the LLM over every free-text case worth it? python scripts/simulate_extraction.py

Three versions of the free-text state's MO fields, same everything else:
  blank       what the pipeline has now (fields unset until extraction)
  simulated   true recorded values corrupted at the error rates the
              llama3.2 pilot MEASURED (data/extractions/pilot2_llama3.2.score.json):
              per-field accuracy where a value was recorded; where nothing was
              recorded, decline / recover the true value / invent a wrong one
              at the measured overall rates
  perfect     the recorded values themselves (upper bound)
Each is trained and evaluated exactly as linkage.train / linkage.evaluate do.

A SIMULATION of the extraction's effect, not a run of it — it answers whether
a multi-hour run is worth doing before paying for it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from linkage import dataset, evaluate, features, schema, train
from linkage.score import Scorer

NORMALISED = Path("data/final_sharing1_normalised.parquet")
TRUTH = Path("data/final_sharing1/truth.parquet")
PILOT = Path("data/extractions/pilot2_llama3.2.score.json")
FIELDS = tuple(f for f in schema.ALL_MO_FIELDS if f not in schema.DERIVED_AT_NORMALISATION)


def corrupt(df: pd.DataFrame, rec: pd.DataFrame, pilot: dict, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    rates = pilot["overall"]
    n_unrec = max(rates["unrecorded"], 1)
    p_decline, p_recover = rates["declined"] / n_unrec, rates["recovered"] / n_unrec
    pending = np.flatnonzero(out["needs_extraction"].to_numpy())
    for f in FIELDS:
        stats = pilot["per_field"].get(f)
        acc = stats["accuracy"] if stats and stats["recorded"] >= 10 else rates["accuracy"]
        vocab = dataset.vocab_for(df, f)          # structured states carry every canonical value
        col = out[f].to_numpy(dtype=object)
        for i in pending:
            recorded, true_value = rec[f"rec_{f}"].iat[i], rec[f].iat[i]
            if recorded is None:
                continue
            u = rng.random()
            if recorded not in schema.TOKENS:
                col[i] = recorded if u < acc else _wrong(recorded, vocab, f, rng)
            elif u < p_decline:
                col[i] = schema.MISSING
            elif u < p_decline + p_recover:
                col[i] = true_value
            else:
                col[i] = _wrong(true_value, vocab, f, rng)
        out[f] = col
    return out


def _wrong(value, vocab, field, rng):
    if field in schema.TAG_FIELDS:                           # flip one tag in or out
        tags = set((value or "").split(";")) - {""}
        tags ^= {vocab[rng.integers(len(vocab))]}
        return ";".join(v for v in vocab if v in tags)
    others = [v for v in vocab if v != value]
    return others[rng.integers(len(others))] if others else value


def run(df: pd.DataFrame, label: str) -> dict:
    weights = train.train(df, 7, oracle_extraction=False)
    scorer = Scorer(weights)
    test = dataset.split_of(df["offender_id"], 7) == "test"
    rng = np.random.default_rng(8)
    pools = {p: evaluate.evaluate_pool(df, scorer, p, test, rng, 1500) for p in features.pools()}
    same = evaluate._pooled(pools, [p for p in features.pools() if p != features.CROSS])
    result = {"same_type": same["fs_lr"], "cross_type": pools[features.CROSS]["fs_lr"]}
    print(f"{label:<10} same-type hit@10 {result['same_type']['hit_at_10']:.3f}  "
          f"PR-AUC {result['same_type']['pr_auc']:.3f} | cross-type hit@10 {result['cross_type']['hit_at_10']:.3f}")
    return result


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    blank = dataset.load(NORMALISED, TRUTH, oracle_extraction=False)
    rec = blank[["case_id"]].merge(
        pd.read_parquet(TRUTH, columns=["case_id", *FIELDS, *[f"rec_{f}" for f in FIELDS]]), on="case_id")
    perfect = dataset.load(NORMALISED, TRUTH, oracle_extraction=True)
    simulated = corrupt(blank, rec, pilot, np.random.default_rng(7))

    report = {"note": __doc__.strip().splitlines()[0], "pilot_rates": pilot["overall"], "conditions": {}}
    for label, frame in (("blank", blank), ("simulated", simulated), ("perfect", perfect)):
        report["conditions"][label] = run(frame, label)
    Path("results/extraction_simulation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
