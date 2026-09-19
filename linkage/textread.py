"""Read an FIR's text into the fields the matcher uses. Pure Python; the API
Lambda runs this, so it is a phrase dictionary, not a language model.

Phrases come from two places:
  1. each state's own wording (config/vocab.yaml `en` clauses), copied into
     the serve bundle by linkage.bundle as phrases.json;
  2. a short list of everyday variants below ("iron rod", "masked"), so the
     reader is not only an inverse of our own generator.
Every candidate phrase is matched longest-first and each stretch of text is
claimed once: "fled on foot" is an escape, and its "on foot" cannot also be
read as "used no vehicle"; "targeted a shop-cum-residence" is not "a shop".
Only the fields that exist for the FIR's crime type are read.

The officer confirms or corrects every value; the reader returns where in the
text each value came from so the page can show it. English only.
FINDINGS §15 has its accuracy on the synthetic narratives.
"""
from __future__ import annotations

import re
from datetime import datetime

from linkage import schema

# Everyday variants, per field and canonical value. Conservative on purpose:
# a wrong fill costs the officer more than a blank one.
VARIANTS = {
    "time_band": {"night": ["at night", "night hours", "midnight"], "early_morning": ["early morning", "early hours", "at dawn"],
                  "day": ["during the day", "daytime", "broad daylight", "in the afternoon"]},
    "group_size_est": {"1": ["acted alone", "a lone", "single person"], "2-3": ["two or three", "two persons", "three persons", "two men", "three men"],
                       "4+": ["four or more", "gang of", "group of four", "group of five"]},
    "tools": {"crowbar": ["crowbar", "iron rod"], "cutter": ["cutter"], "screwdriver": ["screwdriver"], "gas_cutter": ["gas cutter", "welding torch"],
              "none": ["no sign of tools", "no tools"]},
    "counter_forensic": {"none": ["no attempt to avoid identification", "faces were visible"], "gloves": ["gloves"],
                         "face_covered": ["faces covered", "face covered", "covered their faces", "masked", "wore masks", "wearing masks"],
                         "cctv_disabled": ["disabled the cctv", "cctv was disabled", "cctv disabled", "cut the cctv", "damaged the cctv"]},
    "target_selection": {"opportunistic": ["by chance", "opportunistic"], "scouted": ["watched the place beforehand", "recce", "scouted"],
                         "insider_info": ["inside information", "insider"]},
    "property_taken": {"gold": ["gold"], "cash": ["cash"], "electronics": ["electronic", "laptop", "mobile phone", "television"],
                       "documents": ["documents"], "vehicle": ["took a vehicle"], "none": ["took nothing", "nothing was stolen"]},
    "approach_mode": {"on_foot": ["came on foot", "arrived on foot", "walked in"],
                      "two_wheeler": ["came on a two-wheeler", "came on a motorcycle", "came on a bike", "arrived on a two-wheeler", "arrived on a motorcycle"],
                      "four_wheeler": ["came in a four-wheeler", "came in a car", "arrived in a car", "arrived in a four-wheeler"]},
    "exit_mode": {"on_foot": ["fled on foot", "escaped on foot", "ran away"],
                  "two_wheeler": ["fled on a two-wheeler", "fled on a motorcycle", "escaped on a motorcycle", "sped away on a bike"],
                  "four_wheeler": ["fled in a four-wheeler", "fled in a car", "escaped in a car"]},
    "entry_point": {"door": ["through the door", "main door", "front door"], "window": ["through a window", "through the window"],
                    "balcony": ["balcony"], "roof": ["through the roof", "via the roof"], "wall_breach": ["hole in the wall", "breached the wall"],
                    "shutter": ["shutter"]},
    "entry_method": {"lock_broken": ["broke the lock", "lock was broken", "broken lock", "breaking the lock"],
                     "forced_open": ["forced the door open", "forced open", "prised open"], "grill_cut": ["cut the grill", "grill was cut"],
                     "no_force": ["without force", "no forced entry"], "duplicate_key": ["duplicate key"]},
    "premise": {"independent_house": ["independent house", "bungalow"], "apartment": ["apartment"], "row_house": ["row house"],
                "mixed_use": ["shop-cum-residence"], "shop": ["targeted a shop", "the shop"], "office": ["office"], "warehouse": ["godown", "warehouse"]},
    "occupancy": {"occupied": ["occupants were inside", "family was asleep", "residents were at home"],
                  "temporarily_away": ["occupants were out", "family was away"], "vacant_extended": ["locked for days", "locked for several days"],
                  "closed_overnight": ["closed for the night"], "closed_extended": ["closed for a holiday", "closed for holidays", "closed for the festival"]},
    "search_pattern": {"selective": ["only selected valuables"], "ransacked": ["ransacked"], "targeted": ["knew where the valuables were"]},
    "vehicle_class": {"motorcycle": ["stole a motorcycle", "motorcycle was stolen", "bike was stolen"], "scooter": ["stole a scooter", "scooter was stolen"],
                      "car": ["stole a car", "car was stolen"], "commercial": ["goods vehicle", "tempo", "truck"]},
    "ignition_method": {"duplicate_key": ["duplicate key"], "direct_wiring": ["joining the wires", "direct wiring", "hot-wired", "hotwired"],
                        "handle_lock_broken": ["handle lock"], "key_left": ["key left in", "keys left in"], "towed_or_lifted": ["towed", "lifted"]},
    "location_type": {"residential_parking": ["residential parking", "society parking", "parked outside the house"], "street": ["roadside"],
                      "market": ["market"], "public_parking": ["public parking"], "transit_hub": ["station parking", "railway station", "bus stand"]},
    "vehicle_used": {"motorcycle": ["rode a motorcycle", "on a motorcycle", "on a bike"], "scooter": ["rode a scooter", "on a scooter"],
                     "none": ["used no vehicle", "on foot"]},
    "victim_activity": {"walking": ["who was walking"], "morning_walk": ["morning walk"], "shopping": ["shopping"], "riding_pillion": ["pillion"],
                        "waiting_for_transport": ["waiting for transport", "waiting for a bus", "bus stop"]},
    "escape_direction": {"main_road": ["main road"], "lanes": ["through the lanes", "by-lanes", "narrow lanes"], "highway": ["highway"]},
    "machine_type": {"onsite_branch": ["atm at a bank branch", "branch atm"], "offsite_kiosk": ["offsite atm", "off-site atm", "atm kiosk"],
                     "white_label": ["white-label atm", "white label atm"]},
    "attack_method": {"skimmer": ["skimming device", "skimmer"], "cash_trap": ["cash trap"], "gas_cutting": ["with gas", "gas cutting"],
                      "physical_forcing": ["forced the machine open", "prised open the machine"], "explosive": ["explosive"]},
    "alarm_defeated": {"defeated": ["disabled the alarm", "alarm was disabled"], "not_defeated": ["left the alarm untouched", "alarm went off", "alarm rang"]},
}
CRIME_VARIANTS = {"BURGLARY_RESIDENTIAL": ["house burglary", "house breaking", "house-breaking"],
                  "BURGLARY_COMMERCIAL": ["commercial premises", "shop breaking", "shop burglary"],
                  "VEHICLE_THEFT": ["vehicle theft", "motor vehicle theft", "two-wheeler theft", "car theft"],
                  "SNATCHING": ["chain snatching", "snatched"], "ATM_TAMPERING": ["atm tampering", "atm theft", "atm fraud"]}
STATION = re.compile(r"\b([A-Z]{2})-([A-Z]{3})-(\d{2})\b")
MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
# 08/06/2023 07:11 · 20-04-2024 04:40 AM · 31-Aug-2024 0412 hrs · 2024-04-20
WHEN = re.compile(r"\b(\d{1,2})[/-](\d{1,2}|[A-Za-z]{3})[/-](\d{4})(?:[ ,T]+(\d{1,2}):?(\d{2})(?:\s*([AaPp][Mm]|hrs))?)?"
                  r"|\b(\d{4})-(\d{2})-(\d{2})")


def phrase_table(vocab: dict) -> dict:
    """{'fields': {field: {value: [phrase...]}}, 'crime_types': {type: [phrase...]}} from
    config/vocab.yaml plus VARIANTS; built once by linkage.bundle."""
    fields = {}
    for f, values in vocab["fields"].items():
        out = {}
        for value, labels in values.items():
            value = "none" if value == "none_observed" else value
            out.setdefault(value, []).append(labels["en"].lower())
        for value, extra in VARIANTS.get(f, {}).items():
            out.setdefault(value, []).extend(x.lower() for x in extra)
        fields[f] = {v: sorted(set(p)) for v, p in out.items()}
    crimes = {}
    for ct, labels in vocab["crime_type"].items():
        crimes[ct] = sorted({str(x).lower().replace("_", " ") for x in labels.values()} | set(CRIME_VARIANTS.get(ct, [])))
    return {"fields": fields, "crime_types": crimes}


def _claims(text: str, candidates: list[tuple[str, object]]) -> list[tuple[int, int, str, object]]:
    """Longest-first, non-overlapping matches of (phrase, payload) on word boundaries."""
    taken, found = [], []
    for phrase, payload in sorted(candidates, key=lambda c: -len(c[0])):
        for m in re.finditer(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text):
            a, b = m.span()
            if all(b <= x or a >= y for x, y in taken):
                taken.append((a, b))
                found.append((a, b, phrase, payload))
    return sorted(found)


def _when(text: str) -> tuple[datetime | None, datetime | None, bool]:
    stamps = []
    for d, mth, y, hh, mm, ampm, iy, im, iday in WHEN.findall(text):
        try:
            if iy:
                stamps.append((datetime(int(iy), int(im), int(iday)), False))
                continue
            month = MONTHS[mth[:3].lower()] if mth.isalpha() else int(mth)
            hour = int(hh) if hh else 0
            if ampm and ampm.lower() in ("am", "pm"):
                hour = hour % 12 + (12 if ampm.lower() == "pm" else 0)
            stamps.append((datetime(int(y), month, int(d), hour, int(mm) if mm else 0), bool(hh)))
        except (ValueError, KeyError):
            continue
    if not stamps:
        return None, None, False
    frm, timed = stamps[0]
    to = stamps[1][0] if len(stamps) > 1 else frm
    return frm, to, timed and (len(stamps) < 2 or stamps[1][1])


def read_fir(text: str, table: dict, places: dict, pool_fields: dict, crime_type: str | None = None) -> dict:
    """Fields read from the text, and where each came from.

    places: {state: [district, ...]}; pool_fields: {crime type: [field, ...]}.
    crime_type: if the officer already chose one, read that type's fields."""
    raw = text or ""
    low = raw.lower()
    found: list[dict] = []
    out: dict = {"fields": {}, "found": found}

    if crime_type is None:
        hits = _claims(low, [(p, ct) for ct, ps in table["crime_types"].items() for p in ps])
        if hits:
            a, b, phrase, ct = hits[0]
            crime_type = ct
            found.append({"field": "crime_type", "value": ct, "start": a, "end": b})
    if crime_type:
        out["crime_type"] = crime_type

    m = STATION.search(raw)
    if m and m.group(1) in places:
        out["state_code"], out["police_station"] = m.group(1), m.group(0)
        found.append({"field": "police_station", "value": m.group(0), "start": m.start(), "end": m.end()})
    districts = [(d.lower(), (s, d)) for s, ds in places.items() for d in ds]
    for a, b, phrase, (s, d) in _claims(low, districts)[:1]:
        if out.get("state_code") in (None, s):
            out["state_code"], out["district"] = s, d
            found.append({"field": "district", "value": d, "start": a, "end": b})

    frm, to, timed = _when(raw)
    if frm:
        out["occurred_from"] = frm.date().isoformat()
        if timed and "time_band" in pool_fields.get(crime_type, []):
            band = schema.recorded_time_band(frm, to)
            if band != schema.UNKNOWABLE:
                out["fields"]["time_band"] = band
                found.append({"field": "time_band", "value": band, "start": None, "end": None})

    fields = pool_fields.get(crime_type) or sorted({f for fs in pool_fields.values() for f in fs})
    cands = [(p, (f, v)) for f in fields if f in table["fields"] for v, ps in table["fields"][f].items() for p in ps]
    for a, b, phrase, (f, v) in _claims(low, cands):
        if f in schema.TAG_FIELDS:
            values = out["fields"].setdefault(f, [])
            if v == "none":
                if not values:
                    values.append("none")
            else:
                if "none" in values:
                    values.remove("none")
                if v not in values:
                    values.append(v)
        elif f in out["fields"]:
            continue                      # first mention wins; a second, different value is left for the officer
        else:
            out["fields"][f] = v
        found.append({"field": f, "value": v, "start": a, "end": b})
    return out
