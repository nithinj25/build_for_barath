"""Pre-deployment checks: python scripts/preflight.py

1. State predictability after normalisation — validator 4's deferred half.
   Can a classifier tell which state a case came from, using only the
   normalised MO values? Some signal is by design (states differ in dropout
   and one has no occupancy column), so the test compares against the same
   classifier on the TRUE values under the same missingness mask: normalisation
   must not add state signal beyond what recording differences already carry.
2. Inference is pure numpy — imports linkage.score in a subprocess with
   pandas, sklearn and scipy blocked, scores a pair and prints the bits. This
   is what Lambda runs (CLAUDE.md rules 2 and 3).
3. Model artifact — copies the trained weights to model/weights.json, the
   path handlers load, and reports its size.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from linkage import dataset, schema

NORMALISED = Path("data/final_sharing1_normalised.parquet")
TRUTH = Path("data/final_sharing1/truth.parquet")
WEIGHTS = Path("results/post_core_fix/weights.json")
ARTIFACT = Path("model/weights.json")


def state_predictability(sample: int = 20000, seed: int = 7) -> dict:
    df = dataset.load(NORMALISED, TRUTH, oracle_extraction=False)
    truth = pd.read_parquet(TRUTH, columns=["case_id", *[f"rec_{f}" for f in schema.ALL_MO_FIELDS]])
    joined = df.merge(truth, on="case_id").sample(n=min(sample, len(df)), random_state=seed)
    fields = [f for f in schema.ALL_MO_FIELDS if f != "time_band"]

    pending = joined["needs_extraction"].to_numpy(bool)

    def encode(frame, prefix=""):
        cols = {}
        for f in fields:
            series = frame[f"{prefix}{f}"].astype("string")
            if prefix:            # baseline: blank the same fields the pipeline has not extracted yet,
                series = series.mask(pending)     # else it identifies the free-text state for free
            codes, _ = pd.factorize(series.fillna("__NONE__"))
            cols[f] = codes
        return pd.DataFrame(cols).to_numpy(float)

    y = joined["state_code"].to_numpy()
    baseline = encode(joined, "rec_")        # true recorded values, same missingness
    observed = encode(joined)                # what the pipeline sees after normalisation
    model = HistGradientBoostingClassifier(max_iter=60, random_state=seed)
    acc_obs = float(cross_val_score(model, observed, y, cv=3, scoring="accuracy").mean())
    acc_base = float(cross_val_score(model, baseline, y, cv=3, scoring="accuracy").mean())
    majority = float(pd.Series(y).value_counts(normalize=True).max())
    return {"normalised_accuracy": round(acc_obs, 4), "recorded_values_accuracy": round(acc_base, 4),
            "majority_class": round(majority, 4), "added_by_normalisation": round(acc_obs - acc_base, 4),
            "passed": acc_obs - acc_base < 0.02,
            "criterion": "normalisation adds < 0.02 accuracy over the same values pre-rendering; "
                         "state is predictable at all only because states differ in what they record"}


def numpy_only_inference() -> dict:
    code = textwrap.dedent("""
        import sys
        for blocked in ("pandas", "sklearn", "scipy", "yaml"):
            sys.modules[blocked] = None          # any import of these now fails
        import json
        import numpy as np
        from linkage.score import Scorer
        weights = json.load(open("results/post_core_fix/weights.json", encoding="utf-8"))
        scorer = Scorer(weights)
        pool = "BURGLARY_RESIDENTIAL"
        fields = scorer.pools[pool]["fields"]
        a = {"time_band": ["night"], "group_size_est": ["2-3"], "tools": ["cutter"],
             "counter_forensic": ["cctv_disabled"], "target_selection": ["scouted"],
             "property_taken": ["gold;cash"], "approach_mode": ["two_wheeler"],
             "exit_mode": ["two_wheeler"], "entry_point": ["roof"], "entry_method": ["lock_broken"],
             "premise": ["independent_house"], "occupancy": ["temporarily_away"],
             "search_pattern": ["selective"]}
        ca = scorer.encode(pool, a)
        codes_a = {f: ca[f][0] for f in fields}
        bits = scorer.field_bits(pool, codes_a, ca)
        print(json.dumps({"loaded": sorted(m for m in ("pandas", "sklearn", "scipy") if sys.modules[m] is None),
                          "evidence_bits": round(float(scorer.evidence(pool, bits)[0]), 3),
                          "driving": scorer.explain(pool, bits[0])}))
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**__import__("os").environ, "PYTHONPATH": "."})
    if out.returncode != 0:
        return {"passed": False, "error": out.stderr.strip().splitlines()[-1] if out.stderr else "failed"}
    result = json.loads(out.stdout)
    return {"passed": True, "blocked_modules": result["loaded"],
            "self_pair_evidence_bits": result["evidence_bits"], "driving_fields": result["driving"],
            "criterion": "linkage.score imports and scores with pandas/sklearn/scipy unavailable"}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    checks = {"state_predictability": state_predictability(), "numpy_only_inference": numpy_only_inference()}

    ARTIFACT.parent.mkdir(exist_ok=True)
    shutil.copy(WEIGHTS, ARTIFACT)
    checks["model_artifact"] = {"path": str(ARTIFACT), "kb": round(ARTIFACT.stat().st_size / 1024, 1),
                                "passed": ARTIFACT.stat().st_size < 5_000_000,
                                "criterion": "weights ship as JSON, well inside Lambda's limits"}

    Path("results/preflight.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    for name, res in checks.items():
        print(f"{'PASS' if res.get('passed') else 'FAIL'}  {name}")
        for k, v in res.items():
            if k not in ("passed", "criterion"):
                print(f"        {k}: {v}")
    return 0 if all(c.get("passed") for c in checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
