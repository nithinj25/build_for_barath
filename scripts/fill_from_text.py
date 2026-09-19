"""Fill MO fields the structured record leaves blank from the FIR's own text:
python scripts/fill_from_text.py --out data/final_sharing1_textfilled.parquet

Only blank cells (None, __MISSING__, __ABSENT__) are filled; a recorded value
is never overwritten, and __UNKNOWABLE__ stays unknowable. Each filled cell's
field_provenance becomes "text_read" so an analyst can tell it from a recorded
value (CLAUDE.md conventions). Uses linkage.textread with the crime type the
FIR was recorded under. Labels are never read.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from linkage import schema, textread  # noqa: E402

BLANK = {schema.MISSING, schema.ABSENT}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--normalised", type=Path, default=Path("data/final_sharing1_normalised.parquet"))
    ap.add_argument("--vocab", type=Path, default=Path("config/vocab.yaml"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    table = textread.phrase_table(yaml.safe_load(args.vocab.read_text(encoding="utf-8")))
    df = pd.read_parquet(args.normalised)
    places = {s: sorted(set(g["district"])) for s, g in df[df["crime_type"].notna()].groupby("state_code")}
    pool_fields = {ct: [*schema.MO_CORE, *schema.MO_EXT[schema.FAMILY[ct]]] for ct in schema.CRIME_TYPES}

    filled, by_state, cleared = Counter(), Counter(), 0
    rows = df.to_dict("records")
    for r in rows:
        ct = r.get("crime_type")
        if ct not in pool_fields:
            continue
        text = " ".join(x for x in (r.get("narrative_text"), r.get("mo_description")) if isinstance(x, str))
        read = textread.read_fir(text, table, places, pool_fields, crime_type=ct)["fields"]
        prov = r["field_provenance"]
        prov = json.loads(prov) if isinstance(prov, str) else dict(prov or {})
        any_filled = False
        for f in pool_fields[ct]:
            v = r.get(f)
            if not (v is None or v in BLANK) or f not in read:
                continue
            value = read[f]
            if isinstance(value, list):
                value = "" if value == ["none"] else ";".join(x for x in value if x != "none")
            r[f] = value
            prov[f] = "text_read"
            filled[f] += 1
            by_state[r["state_code"]] += 1
            any_filled = True
        if any_filled and r.get("needs_extraction"):
            r["needs_extraction"] = False
            cleared += 1
        r["field_provenance"] = json.dumps(prov)
    out = pd.DataFrame(rows, columns=df.columns)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"filled {sum(filled.values()):,} blank MO cells; by state {dict(by_state)}; "
          f"{cleared:,} free-text FIRs no longer pending")
    print("by field:", dict(filled.most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
