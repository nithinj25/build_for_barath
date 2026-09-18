"""Normalisation: state-native feed rows → canonical case records.

    python -m linkage.normalise --feeds data/final/feeds --out data/final_normalised.parquet \\
        --check data/final/truth.parquet

All state knowledge lives in adapters/{STATE}.yaml. There is no per-state
code, so onboarding a state is a YAML diff. Row-level functions are pure over
dicts; only `main` touches disk. Never imports linkage.generate or config/.

Each record keeps blank (MISSING), no-such-column (ABSENT) and
window-too-wide (UNKNOWABLE) distinct, and carries field_provenance:
"source" for values read or derived from the feed, None where there is no
value. Free-text states set needs_extraction; their MO fields stay None until
the enrich step fills them as "llm_extracted".
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from linkage import schema
from linkage.schema import ABSENT, MISSING

ADAPTER_DIR = Path(__file__).resolve().parent.parent / "adapters"
META = ("fir_no", "district", "police_station")


def _key(label: str) -> str:
    return " ".join(str(label).split()).casefold()


def compile_adapter(raw: dict) -> dict:
    """Add lookup indexes: canonical → native column, and labels matched
    case- and whitespace-insensitively."""
    return {
        **raw,
        "column_for": {canon: native for native, canon in raw["columns"].items()},
        "crime_type_index": {_key(k): v for k, v in raw["crime_type"].items()},
        "value_index": {f: {_key(k): v for k, v in labels.items()}
                        for f, labels in (raw.get("values") or {}).items()},
    }


def _parse_time(text: str, adapter: dict) -> tuple[datetime | None, bool]:
    """(parsed, time_known). A date-only value parses with time_known False."""
    text = text.strip()
    if not text:
        return None, False
    try:
        return datetime.strptime(text, adapter["datetime_format"]), True
    except ValueError:
        pass
    try:
        return datetime.strptime(text, adapter["date_only_format"]), False
    except ValueError:
        return None, False


def normalise_row(row: dict, adapter: dict) -> dict:
    cols = adapter["column_for"]

    def cell(canon: str) -> str:
        return (row.get(cols[canon]) or "").strip() if canon in cols else ""

    issues: list[str] = []
    provenance: dict[str, str | None] = {}
    rec: dict = {"state_code": adapter["state_code"], **{k: cell(k) or None for k in META}}

    label = cell("crime_type")
    crime_type = adapter["crime_type_index"].get(_key(label)) if label else None
    if crime_type is None:
        issues.append(f"crime_type: unmapped label {label!r}")
    rec["crime_type"] = crime_type

    times = {}
    for name in ("occurred_from", "occurred_to", "registered_at"):
        parsed, known = _parse_time(cell(name), adapter)
        if parsed is None:
            issues.append(f"{name}: unparseable {cell(name)!r}")
        times[name] = (parsed, known)
        rec[name] = parsed
    registered = times["registered_at"][0]
    rec["case_id"] = schema.case_id(adapter["state_code"], rec["fir_no"], registered.year) \
        if registered and rec["fir_no"] else None

    (occ_from, from_known), (occ_to, to_known) = times["occurred_from"], times["occurred_to"]
    if occ_from and occ_to and from_known and to_known:
        rec["time_band"] = schema.recorded_time_band(occ_from, occ_to)
        provenance["time_band"] = "source"
    else:
        rec["time_band"] = MISSING
        provenance["time_band"] = None

    applicable = set(schema.mo_fields(crime_type)) if crime_type else set()
    free_text = adapter["layout"] == "free_text_mo"
    delimiter = adapter.get("multi_value_delimiter")
    for f in schema.ALL_MO_FIELDS:
        if f == "time_band":
            continue
        if f not in applicable:
            rec[f] = None
            continue
        provenance[f] = None
        if free_text:
            rec[f] = None
            continue
        if f not in cols:
            rec[f] = ABSENT
            continue
        text = cell(f)
        if not text:
            rec[f] = MISSING
            continue
        index = adapter["value_index"].get(f, {})
        if f in schema.TAG_FIELDS:
            parts = [p.strip() for p in text.split(delimiter) if p.strip()]
            mapped = [index.get(_key(p)) for p in parts]
            for p, m in zip(parts, mapped):
                if m is None:
                    issues.append(f"{f}: unmapped label {p!r}")
            tags = {m for m in mapped if m and m != "none_observed"}
            rec[f] = ";".join(v for v in schema.MO_CORE_VOCAB[f] if v in tags)
        else:
            value = index.get(_key(text))
            if value is None:
                issues.append(f"{f}: unmapped label {text!r}")
                rec[f] = MISSING
                continue
            rec[f] = value
        provenance[f] = "source"

    rec["narrative_text"] = cell("narrative")
    rec["narrative_lang"] = adapter["narrative_language"]
    rec["mo_description"] = cell(schema.MO_DESCRIPTION) if free_text else None
    rec["needs_extraction"] = free_text
    rec["field_provenance"] = provenance
    rec["issues"] = issues
    return rec


def feed_drift(columns: list[str], adapter: dict) -> dict:
    """Columns the adapter expects but the feed lacks, and the reverse."""
    expected = set(adapter["columns"])
    return {"missing": sorted(expected - set(columns)), "unexpected": sorted(set(columns) - expected)}


# --- check against the generator's answer key ----------------------------------

def _same(a, b) -> bool:
    if (a is None or (isinstance(a, float) and pd.isna(a))) and (b is None or (isinstance(b, float) and pd.isna(b))):
        return True
    if isinstance(a, pd.Timestamp) or isinstance(b, pd.Timestamp):
        return pd.Timestamp(a) == pd.Timestamp(b) if not (pd.isna(a) or pd.isna(b)) else pd.isna(a) and pd.isna(b)
    return a == b


def check_against_truth(norm: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Compare normalised records with the recorded canonical values the
    generator wrote (truth.rec_*). This tests normaliser + adapters, not
    extraction: free-text MO fields are counted as pending, not compared."""
    ingested = truth[truth["ingested"]]
    merged = norm.merge(ingested, on="case_id", how="outer", suffixes=("", "_truth"), indicator=True)
    report = {"normalised_rows": int(len(norm)), "truth_ingested_rows": int(len(ingested)),
              "unmatched_normalised": int((merged["_merge"] == "left_only").sum()),
              "unmatched_truth": int((merged["_merge"] == "right_only").sum()), "fields": {}}
    both = merged[merged["_merge"] == "both"]

    def compare(name, left, right, rows):
        bad = [i for i in rows.index if not _same(rows.at[i, left], rows.at[i, right])]
        report["fields"][name] = {"compared": int(len(rows)), "mismatches": len(bad),
                                  "examples": [{"case_id": rows.at[i, "case_id"], "normalised": str(rows.at[i, left]),
                                                "truth": str(rows.at[i, right])} for i in bad[:3]]}

    compare("crime_type", "crime_type", "rec_crime_type", both)
    compare("time_band", "time_band", "rec_time_band", both)
    for name in ("fir_no", "district", "police_station", "registered_at"):
        compare(name, name, f"{name}_truth", both)
    timed = both[both["time_band"] != MISSING]
    for name in ("occurred_from", "occurred_to"):
        compare(name, name, f"{name}_truth", timed)
    structured = both[~both["needs_extraction"].astype(bool)]
    for f in schema.ALL_MO_FIELDS:
        if f != "time_band":
            compare(f, f, f"rec_{f}", structured)
    report["mo_fields_pending_extraction_rows"] = int(both["needs_extraction"].astype(bool).sum())
    report["passed"] = (report["unmatched_normalised"] == 0 and report["unmatched_truth"] == 0
                        and all(v["mismatches"] == 0 for v in report["fields"].values()))
    return report


# --- CLI ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m linkage.normalise", description=__doc__.splitlines()[0])
    ap.add_argument("--feeds", type=Path, required=True, help="directory of {STATE}.csv feeds")
    ap.add_argument("--out", type=Path, required=True, help="output parquet")
    ap.add_argument("--adapters", type=Path, default=ADAPTER_DIR)
    ap.add_argument("--check", type=Path, help="truth.parquet to compare against (synthetic corpora only)")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    records, failed = [], False
    for path in sorted(args.adapters.glob("*.yaml")):
        adapter = compile_adapter(yaml.safe_load(path.read_text(encoding="utf-8")))
        feed = args.feeds / f"{adapter['state_code']}.csv"
        if not feed.exists():
            print(f"  {adapter['state_code']}: no feed at {feed}, skipped")
            continue
        frame = pd.read_csv(feed, dtype=str, keep_default_na=False, encoding="utf-8")
        drift = feed_drift(list(frame.columns), adapter)
        if drift["missing"]:
            print(f"  {adapter['state_code']}: feed lacks adapter columns {drift['missing']} — fix the adapter")
            failed = True
            continue
        rows = [normalise_row(r, adapter) for r in frame.to_dict("records")]
        with_issues = sum(1 for r in rows if r["issues"])
        print(f"  {adapter['state_code']}: {len(rows)} rows, {with_issues} with issues"
              + (f", unexpected columns {drift['unexpected']}" if drift["unexpected"] else "")
              + (", MO fields pending extraction" if adapter["layout"] == "free_text_mo" else ""))
        records.extend(rows)

    norm = pd.DataFrame(records)
    out = norm.copy()
    out["field_provenance"] = out["field_provenance"].map(json.dumps)
    out["issues"] = out["issues"].map(json.dumps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"wrote {args.out} ({len(norm)} records)")

    if args.check:
        report = check_against_truth(norm, pd.read_parquet(args.check))
        print(f"\ncheck against {args.check}: {'PASS' if report['passed'] else 'FAIL'}")
        print(f"  unmatched: normalised {report['unmatched_normalised']}, truth {report['unmatched_truth']}; "
              f"rows with MO pending extraction {report['mo_fields_pending_extraction_rows']}")
        for name, res in report["fields"].items():
            if res["mismatches"]:
                print(f"  {name}: {res['mismatches']} of {res['compared']} differ, e.g. {res['examples']}")
        failed |= not report["passed"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
