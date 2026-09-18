"""How good is the extraction? python -m enrich.score_extraction data/extractions/llama3.2.jsonl

Compared with the recorded canonical values the free text was rendered from
(truth.rec_*), per field:
  accuracy        a value was recorded → did the model return it
  declined        nothing was recorded → did the model correctly return null
  when it answered anyway on an unrecorded field, was the answer
    recovered     the TRUE value (the narrative mentioned it — legitimate)
    invented      not the true value (a hallucination)
The spec's rule: extraction must return null rather than invent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from linkage import schema

EXTRACTED = tuple(f for f in schema.ALL_MO_FIELDS if f not in schema.DERIVED_AT_NORMALISATION)


def score(extractions: Path, truth_path: Path) -> dict:
    rows = [json.loads(line) for line in extractions.read_text(encoding="utf-8").splitlines() if line]
    truth = pd.read_parquet(truth_path).set_index("case_id")
    tally = {f: {"recorded": 0, "correct": 0, "unrecorded": 0, "declined": 0, "recovered": 0, "invented": 0}
             for f in EXTRACTED}
    seconds = []
    for r in rows:
        seconds.append(r["seconds"])
        t = truth.loc[r["case_id"]]
        for f, got in r["fields"].items():
            recorded, true_value = t[f"rec_{f}"], t[f]
            if recorded is None:
                continue                                   # not applicable to this crime type
            s = tally[f]
            if recorded in schema.TOKENS:                  # nothing recorded for this field
                s["unrecorded"] += 1
                if got is None:
                    s["declined"] += 1
                elif got == true_value:
                    s["recovered"] += 1
                else:
                    s["invented"] += 1
            else:
                s["recorded"] += 1
                s["correct"] += int(got == recorded)
    per_field = {f: {"accuracy": round(s["correct"] / s["recorded"], 3) if s["recorded"] else None,
                     "declined_when_unrecorded": round(s["declined"] / s["unrecorded"], 3) if s["unrecorded"] else None,
                     "invented_when_unrecorded": round(s["invented"] / s["unrecorded"], 3) if s["unrecorded"] else None,
                     **s} for f, s in tally.items() if s["recorded"] or s["unrecorded"]}
    total = {k: sum(s[k] for s in tally.values()) for k in ("recorded", "correct", "unrecorded", "declined",
                                                             "recovered", "invented")}
    return {"cases": len(rows), "model": rows[0]["model"] if rows else None,
            "mean_seconds_per_case": round(sum(seconds) / max(len(seconds), 1), 2),
            "overall": {"accuracy": round(total["correct"] / max(total["recorded"], 1), 4),
                        "declined_when_unrecorded": round(total["declined"] / max(total["unrecorded"], 1), 4),
                        "recovered_when_unrecorded": round(total["recovered"] / max(total["unrecorded"], 1), 4),
                        "invented_when_unrecorded": round(total["invented"] / max(total["unrecorded"], 1), 4),
                        **total},
            "per_field": per_field}


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    sys.stdout.reconfigure(encoding="utf-8")
    path = Path(argv[0])
    truth = Path(argv[1]) if len(argv) > 1 else Path("data/final_sharing1/truth.parquet")
    report = score(path, truth)
    o = report["overall"]
    print(f"{report['model']}: {report['cases']} cases, {report['mean_seconds_per_case']} s/case | "
          f"accuracy {o['accuracy']:.3f} on {o['recorded']} recorded values | when unrecorded: declined "
          f"{o['declined_when_unrecorded']:.3f}, recovered {o['recovered_when_unrecorded']:.3f}, "
          f"invented {o['invented_when_unrecorded']:.3f} (n={o['unrecorded']})")
    worst = sorted(((f, s["accuracy"]) for f, s in report["per_field"].items() if s["accuracy"] is not None),
                   key=lambda kv: kv[1])[:4]
    print("  weakest fields:", ", ".join(f"{f} {a:.2f}" for f, a in worst))
    out = path.with_suffix(".score.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
