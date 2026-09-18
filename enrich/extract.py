"""MO extraction from free text with a local LLM (Ollama).

    python -m enrich.extract --normalised data/final_sharing1_normalised.parquet \\
        --model llama3.2 --out data/extractions/llama3.2.jsonl [--limit 500]

For states that publish MO as prose (TG here), the scored fields are blank
until extraction fills them. Each case's text goes to the model with a JSON
schema in which every applicable field is an enum of canonical values or
null, so the model can pick a value or decline but cannot invent a label.
The instruction is explicit: a field not stated in the text is null.

The canonical vocabulary is read from the trained weights — the pipeline's
own copy — never from the generator's config. Output is appended as JSON
lines and the run resumes where it stopped, so a long run can be interrupted.
time_band is not extracted: it is derived from the occurrence timestamps.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from linkage import schema

OLLAMA = "http://localhost:11434/api/chat"
SYSTEM = (
    "You extract modus operandi fields from an Indian police First Information Report. "
    "Use only what the text states. If a field is not stated, return null for it — never guess, "
    "never infer from what is typical. For tools and property_taken return a list; return an empty "
    "list only when the text says none were used or nothing was taken, and null when it says nothing."
)
EXTRACTED = tuple(f for f in schema.ALL_MO_FIELDS if f not in schema.DERIVED_AT_NORMALISATION)

# What each canonical value means, in plain words. Without this a small model
# cannot map "wore gloves" onto the enum `gloves` and declines instead (pilot:
# counter_forensic 0% on 41 cases). Written as a schema description would be
# for a real deployment — NOT copied from the generator's phrase bank, which
# would make extraction a lookup that inverts the templates.
FIELD_GUIDE = {
    "group_size_est": "how many offenders — 1: one person; 2-3: two or three; 4+: four or more",
    "tools": "tools used or left behind — crowbar: iron bar or rod for prying; cutter: bolt, wire or "
             "lock cutter, plier; screwdriver; gas_cutter: gas or welding torch",
    "counter_forensic": "steps taken to avoid identification — none: no such steps; gloves; face_covered: "
                        "mask, helmet or cloth over the face; cctv_disabled: camera damaged, turned or disconnected",
    "target_selection": "how the target was chosen — opportunistic: by chance; scouted: watched or surveyed "
                        "beforehand, recce; insider_info: tip-off or inside knowledge",
    "property_taken": "what was taken — gold: ornaments, jewellery, chains; cash; electronics: phone, laptop, TV; "
                      "documents; vehicle",
    "approach_mode": "how the offenders arrived — on_foot; two_wheeler: motorcycle, scooter, bike; "
                     "four_wheeler: car, jeep, van, truck",
    "exit_mode": "how the offenders left — on_foot; two_wheeler: motorcycle, scooter, bike; "
                 "four_wheeler: car, jeep, van, truck",
    "entry_point": "where they got in — door; window; balcony; roof: roof or terrace; wall_breach: a hole made "
                   "in the wall; shutter: rolling shutter",
    "entry_method": "how they got in — lock_broken; forced_open: door or window forced or kicked; grill_cut; "
                    "no_force: entered without force; duplicate_key",
    "premise": "the premises — independent_house: bungalow; apartment: flat; row_house; mixed_use: "
               "shop-cum-residence; shop; office; warehouse: godown, storage",
    "occupancy": "who was there — occupied: people inside; temporarily_away: occupants out for hours; "
                 "vacant_extended: locked for days, out of station; closed_overnight: business closed for "
                 "the night; closed_extended: closed for a holiday",
    "search_pattern": "how they searched — selective: took only chosen items; ransacked: searched everything; "
                      "targeted: went straight to where valuables were kept",
    "vehicle_class": "the stolen vehicle — motorcycle; scooter; car; commercial: goods vehicle, truck, tempo",
    "ignition_method": "how the vehicle was started or moved — duplicate_key: master or duplicate key; "
                       "direct_wiring: wires joined or bypassed; handle_lock_broken; key_left: key left in the "
                       "vehicle; towed_or_lifted",
    "location_type": "where the vehicle was taken from — residential_parking: society or home parking; street: "
                     "roadside; market; public_parking: pay-and-park lot; transit_hub: railway station, bus stand",
    "vehicle_used": "what the snatchers rode — motorcycle; scooter; none: on foot",
    "victim_activity": "what the victim was doing — walking; morning_walk; shopping; riding_pillion: sitting "
                       "behind on a two-wheeler; waiting_for_transport: at a bus stop",
    "escape_direction": "where they escaped — main_road; lanes: by-lanes or cross roads; highway",
    "machine_type": "the ATM — onsite_branch: at a bank branch; offsite_kiosk: standalone kiosk; white_label: "
                    "non-bank ATM",
    "attack_method": "how the ATM was attacked — skimmer: card-skimming device; cash_trap: device trapping "
                     "notes; gas_cutting; physical_forcing: machine forced or broken open; explosive",
    "alarm_defeated": "the alarm — defeated: disabled; not_defeated: left intact or working",
}


def vocabulary(weights_path: Path) -> dict[str, list[str]]:
    """Canonical values per field, as the trained scorer knows them."""
    pools = json.loads(weights_path.read_text(encoding="utf-8"))["pools"]
    vocab: dict[str, list[str]] = {}
    for spec in pools.values():
        for f, values in spec["vocab"].items():
            vocab.setdefault(f, [])
            vocab[f] += [v for v in values if v not in vocab[f]]
    return vocab


def response_schema(crime_type: str, vocab: dict[str, list[str]]) -> dict:
    props = {}
    for f in schema.mo_fields(crime_type):
        if f not in EXTRACTED:
            continue
        if f in schema.TAG_FIELDS:
            props[f] = {"type": ["array", "null"], "items": {"type": "string", "enum": vocab[f]}}
        else:
            props[f] = {"type": ["string", "null"], "enum": [*vocab[f], None]}
    return {"type": "object", "properties": props, "required": list(props)}


def extract_one(case: dict, model: str, vocab: dict[str, list[str]], timeout: float = 120) -> dict:
    text = "\n".join(t for t in (case.get("mo_description"), case.get("narrative_text")) if t)
    fields = [f for f in schema.mo_fields(case["crime_type"]) if f in EXTRACTED]
    guide = "\n".join(f"- {f}: {FIELD_GUIDE[f]}" for f in fields)
    payload = {
        "model": model, "stream": False,
        "format": response_schema(case["crime_type"], vocab),
        "options": {"temperature": 0, "seed": 7},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"Crime type: {case['crime_type']}\n\nFields and what their "
                                                 f"values mean:\n{guide}\n\nFIR text:\n{text}"}],
    }
    started = time.perf_counter()
    reply = requests.post(OLLAMA, json=payload, timeout=timeout)
    reply.raise_for_status()
    fields = json.loads(reply.json()["message"]["content"])
    out = {}
    for f, value in fields.items():
        if f in schema.TAG_FIELDS and isinstance(value, list):
            out[f] = ";".join(v for v in schema.MO_CORE_VOCAB[f] if v in value)
        else:
            out[f] = value
    return {"case_id": case["case_id"], "model": model, "fields": out,
            "seconds": round(time.perf_counter() - started, 2)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m enrich.extract", description=__doc__.splitlines()[0])
    ap.add_argument("--normalised", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=Path("model/weights.json"))
    ap.add_argument("--model", default="llama3.2")
    ap.add_argument("--out", type=Path, required=True, help="JSON lines, appended; resumes")
    ap.add_argument("--limit", type=int, default=None, help="first N pending cases (a fixed, seeded sample)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    norm = pd.read_parquet(args.normalised)
    pending = norm[norm["needs_extraction"].astype(bool) & norm["crime_type"].notna()]
    pending = pending.sample(frac=1.0, random_state=7)            # fixed order: --limit is a fair sample
    if args.limit:
        pending = pending.head(args.limit)
    done = set()
    if args.out.exists():
        done = {json.loads(line)["case_id"] for line in args.out.read_text(encoding="utf-8").splitlines() if line}
    todo = [c for c in pending.to_dict("records") if c["case_id"] not in done]
    print(f"{len(pending)} cases selected, {len(done)} already extracted, {len(todo)} to go ({args.model})")

    vocab = vocabulary(args.weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started, n, failures = time.time(), 0, 0
    with args.out.open("a", encoding="utf-8") as sink, ThreadPoolExecutor(args.workers) as pool:
        for result in pool.map(lambda c: _safe(c, args.model, vocab), todo):
            if result is None:
                failures += 1
                continue
            sink.write(json.dumps(result) + "\n")
            n += 1
            if n % 100 == 0:
                sink.flush()
                rate = n / (time.time() - started)
                print(f"  {n}/{len(todo)}  {rate:.1f} cases/s  eta {(len(todo) - n) / rate / 60:.0f} min")
    print(f"extracted {n} cases in {time.time() - started:.0f}s, {failures} failed → {args.out}")
    return 0


def _safe(case, model, vocab):
    try:
        return extract_one(case, model, vocab)
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"  {case['case_id']}: {type(exc).__name__}: {exc}")
        return None


if __name__ == "__main__":
    sys.exit(main())
