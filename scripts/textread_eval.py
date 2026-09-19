"""How well linkage.textread reads FIR text: python scripts/textread_eval.py \\
       --out results/textread/eval.json

For every FIR, read narrative_text + mo_description WITHOUT telling the reader
the crime type, then compare with (a) the recorded structured fields and (b)
the generator's true values. Per field: how often a value was read, and how
often a read value was right. Synthetic narratives are built from the same
phrases the reader starts from, so this is an upper bound for real FIR text.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from linkage import schema, textread  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--normalised", type=Path, default=Path("data/final_sharing1_normalised.parquet"))
    ap.add_argument("--truth", type=Path, default=Path("data/final_sharing1/truth.parquet"))
    ap.add_argument("--vocab", type=Path, default=Path("config/vocab.yaml"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    table = textread.phrase_table(yaml.safe_load(args.vocab.read_text(encoding="utf-8")))
    df = pd.read_parquet(args.normalised)
    df = df[df["crime_type"].notna()]
    truth = pd.read_parquet(args.truth).set_index("case_id")
    places = {s: sorted(set(g["district"])) for s, g in df.groupby("state_code")}
    pool_fields = {ct: [*schema.MO_CORE, *schema.MO_EXT[schema.FAMILY[ct]]] for ct in schema.CRIME_TYPES}

    stats = defaultdict(lambda: defaultdict(int))
    meta = defaultdict(int)
    for r in df.to_dict("records"):
        text = " ".join(x for x in (r.get("narrative_text"), r.get("mo_description")) if isinstance(x, str))
        got = textread.read_fir(text, table, places, pool_fields)
        t = truth.loc[r["case_id"]]
        meta["firs"] += 1
        meta["crime_type_read"] += got.get("crime_type") is not None
        meta["crime_type_right"] += got.get("crime_type") == r["crime_type"]
        meta["district_right"] += got.get("district") == r["district"]
        meta["station_right"] += got.get("police_station") == r["police_station"]
        meta["date_right"] += got.get("occurred_from") == str(r["occurred_from"])[:10]
        if got.get("crime_type") != r["crime_type"]:
            continue
        for f in pool_fields[r["crime_type"]]:
            read = got["fields"].get(f)
            if isinstance(read, list):                                  # tags compare as sets
                read = frozenset() if read == ["none"] else frozenset(read)
            as_set = (lambda v: frozenset(x for x in v.split(";") if x)) if f in schema.TAG_FIELDS else (lambda v: v)
            rec = r.get(f)
            true = t.get(f)
            rec = as_set(rec) if isinstance(rec, str) and rec not in schema.TOKENS else None
            true = as_set(true) if isinstance(true, str) else None
            s = stats[f]
            s["applicable"] += 1
            s["recorded"] += rec is not None
            s["read"] += read is not None
            if read is not None and rec is not None:
                s["read_and_recorded"] += 1
                s["agrees_with_record"] += read == rec
            if read is not None and true is not None:
                s["read_with_truth"] += 1
                s["agrees_with_truth"] += read == true
    fields = {}
    for f, s in stats.items():
        fields[f] = {"read_rate": round(s["read"] / s["applicable"], 3),
                     "right_vs_record": round(s["agrees_with_record"] / s["read_and_recorded"], 3) if s["read_and_recorded"] else None,
                     "right_vs_truth": round(s["agrees_with_truth"] / s["read_with_truth"], 3) if s["read_with_truth"] else None}
    total_read = sum(s["read_with_truth"] for s in stats.values())
    report = {"firs": meta["firs"],
              "crime_type": {"read": round(meta["crime_type_read"] / meta["firs"], 3), "right": round(meta["crime_type_right"] / meta["firs"], 3)},
              "district_right": round(meta["district_right"] / meta["firs"], 3),
              "police_station_right": round(meta["station_right"] / meta["firs"], 3),
              "date_right": round(meta["date_right"] / meta["firs"], 3),
              "mo_values_read_right_vs_truth": round(sum(s["agrees_with_truth"] for s in stats.values()) / total_read, 4),
              "fields": dict(sorted(fields.items()))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "fields"}, indent=1))
    for f, v in sorted(fields.items(), key=lambda kv: kv[1]["right_vs_truth"] or 0):
        print(f"  {f:<18} read {v['read_rate']:.2f}   right vs record {v['right_vs_record']}   right vs truth {v['right_vs_truth']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
