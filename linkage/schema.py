"""Frozen case schema (TECHNICAL_SPEC.md §2, §3.1).

Field names and kinds live here; probabilities live in config/. Value
vocabularies for mo_ext are defined by config/marginals.yaml.

Shared by the generator and the pipeline. The pipeline (normalise, score)
imports this module and never linkage.generate or config/.
"""
import hashlib
from datetime import datetime

CRIME_TYPES = (
    "BURGLARY_RESIDENTIAL",
    "BURGLARY_COMMERCIAL",
    "VEHICLE_THEFT",
    "SNATCHING",
    "ATM_TAMPERING",
)

# mo_ext fields are shared within a family, so both burglary types score
# against each other on the full burglary field set.
FAMILY = {
    "BURGLARY_RESIDENTIAL": "burglary",
    "BURGLARY_COMMERCIAL": "burglary",
    "VEHICLE_THEFT": "vehicle",
    "SNATCHING": "snatching",
    "ATM_TAMPERING": "atm",
}

MO_CORE = (
    "time_band",
    "group_size_est",
    "tools",
    "counter_forensic",
    "target_selection",
    "property_taken",
    "approach_mode",
    "exit_mode",
)

MO_EXT = {
    "burglary": ("entry_point", "entry_method", "premise", "occupancy", "search_pattern"),
    "vehicle": ("vehicle_class", "ignition_method", "location_type"),
    "snatching": ("vehicle_used", "victim_activity", "escape_direction"),
    "atm": ("machine_type", "attack_method", "alarm_defeated"),
}

# Multi-label fields: one independent rate per tag. Everything else is
# categorical: one value per crime, probabilities sum to 1.
TAG_FIELDS = frozenset({"tools", "property_taken"})

# True-behaviour vocabulary for mo_core, as spec §3.1 minus the values that
# only exist on the recording side: `unknown` (nobody saw) and
# `none_observed` (empty tool set as written in a feed). A crime always has
# a true group size; whether it got recorded is corruption's job.
MO_CORE_VOCAB = {
    "time_band": ("night", "early_morning", "day"),
    "group_size_est": ("1", "2-3", "4+"),
    "tools": ("crowbar", "cutter", "screwdriver", "gas_cutter"),
    "counter_forensic": ("none", "gloves", "face_covered", "cctv_disabled"),
    "target_selection": ("opportunistic", "scouted", "insider_info"),
    "property_taken": ("gold", "cash", "electronics", "documents", "vehicle"),
    "approach_mode": ("on_foot", "two_wheeler", "four_wheeler"),
    "exit_mode": ("on_foot", "two_wheeler", "four_wheeler"),
}

# Sign convention: positive = the named pole.
#   stealth  + stealth   / − force
#   planned  + planned   / − opportunistic
#   group    + group     / − solo
STYLE_AXES = ("stealth", "planned", "group")

# Case metadata a feed carries besides MO fields. `case_id` is not here: it
# is computed at normalisation as hash(state_code + fir_no + year).
FEED_META = (
    "fir_no",
    "district",
    "police_station",
    "occurred_from",
    "occurred_to",
    "registered_at",
    "crime_type",
    "narrative",
)

# Never a feed column: feeds carry the occurrence window and the band is
# derived from it. Date-only timestamps make it missing; a window too wide
# to place the crime in one band makes it unknowable. The two stay distinct.
DERIVED_AT_NORMALISATION = ("time_band",)

# A free-text feed replaces every structured MO column with this one.
MO_DESCRIPTION = "mo_description"


# Canonical record markers. A value that is not a real field value is one of:
MISSING = "__MISSING__"          # the source has the column; this cell is blank
UNKNOWABLE = "__UNKNOWABLE__"    # time_band only: window too wide to place in one band
ABSENT = "__ABSENT__"            # the source has no such column at all
TOKENS = frozenset({MISSING, UNKNOWABLE, ABSENT})

# What each time band means, [start, end) on the 24h clock. corpus.yaml must
# agree (linkage.config checks), so generator and pipeline derive bands alike.
TIME_BAND_HOURS = {"night": (22, 4), "early_morning": (4, 7), "day": (7, 22)}
UNKNOWABLE_ABOVE_HOURS = 12


def mo_fields(crime_type: str) -> tuple[str, ...]:
    return MO_CORE + MO_EXT[FAMILY[crime_type]]


ALL_MO_FIELDS = MO_CORE + tuple(f for fields in MO_EXT.values() for f in fields)


def band_hours(band: str) -> list[int]:
    a, b = TIME_BAND_HOURS[band]
    return list(range(a, b)) if a < b else [*range(a, 24), *range(0, b)]


BAND_OF_HOUR = {h: band for band in TIME_BAND_HOURS for h in band_hours(band)}


def recorded_time_band(occurred_from: datetime, occurred_to: datetime) -> str:
    """Band of a recorded window with known times: the midpoint's band, or
    UNKNOWABLE when the window is too wide to place in one band."""
    if (occurred_to - occurred_from).total_seconds() / 3600 > UNKNOWABLE_ABOVE_HOURS:
        return UNKNOWABLE
    return BAND_OF_HOUR[(occurred_from + (occurred_to - occurred_from) / 2).hour]


def case_id(state_code: str, fir_no: str, year: int) -> str:
    return hashlib.sha1(f"{state_code}{fir_no}{year}".encode()).hexdigest()[:16]
