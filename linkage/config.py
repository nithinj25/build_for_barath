"""Load and check config/ before anything samples from it.

    python -m linkage.config        errors, unfilled counts, derived values
    python -m linkage.config -v     also list every unfilled entry

Checks are pure functions over the loaded dicts; only `load` touches disk.
Exit status 1 on any error. Unfilled entries (null loadings, null
provenance) are reported, not errors — the generator is what refuses them.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import yaml

from linkage import schema

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
FILES = ("marginals", "loadings", "states", "corpus", "vocab")
PROVENANCE_TAGS = ("published", "derived", "assumption")
FEED_LAYOUTS = ("structured", "free_text_mo")
TOL = 1e-6


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    unfilled: dict[str, list[str]] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)

    def error(self, where: str, msg: str) -> None:
        self.errors.append(f"{where}: {msg}")

    def count(self, group: str, where: str, filled: bool) -> None:
        self.totals[group] = self.totals.get(group, 0) + 1
        if not filled:
            self.unfilled.setdefault(group, []).append(where)


def load(config_dir: Path = CONFIG_DIR) -> dict:
    return {
        name: yaml.safe_load((config_dir / f"{name}.yaml").read_text(encoding="utf-8"))
        for name in FILES
    }


# --- helpers ---------------------------------------------------------------

def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _rate(r: Report, where: str, x) -> bool:
    if not _is_num(x) or not 0 <= x <= 1:
        r.error(where, f"expected a probability in [0, 1], got {x!r}")
        return False
    return True


def _positive(r: Report, where: str, x) -> bool:
    if not _is_num(x) or x <= 0:
        r.error(where, f"expected a positive number, got {x!r}")
        return False
    return True


def _same_keys(r: Report, where: str, actual, expected) -> bool:
    if not isinstance(actual, dict):
        r.error(where, f"expected a mapping, got {type(actual).__name__}")
        return False
    missing = [k for k in expected if k not in actual]
    extra = [k for k in actual if k not in expected]
    if missing:
        r.error(where, f"missing {missing}")
    if extra:
        r.error(where, f"unexpected {extra}")
    return not missing and not extra


def _lognormal(r: Report, where: str, spec) -> None:
    if _same_keys(r, where, spec, ("median", "sigma")):
        _positive(r, f"{where}.median", spec["median"])
        _positive(r, f"{where}.sigma", spec["sigma"])


def field_values(marginals: dict, fieldname: str, crime_types=schema.CRIME_TYPES) -> list[str]:
    """Union of a field's values across crime types, first-seen order."""
    seen: list[str] = []
    for ct in crime_types:
        spec = marginals["by_crime_type"].get(ct, {}).get(fieldname) or {}
        for v in spec.get("p") or spec.get("rate") or {}:
            if v not in seen:
                seen.append(v)
    return seen


def alpha(repeat_rate: float, reference_frequency: float) -> float:
    """Dirichlet concentration giving target m at the reference frequency.

    Inverts m = (alpha * q + 1) / (alpha + 1). Small alpha = high repetition.
    """
    return (1 - repeat_rate) / (repeat_rate - reference_frequency)


# --- marginals ---------------------------------------------------------------

def _provenance(r: Report, where: str, prov) -> None:
    if not _same_keys(r, f"{where}.provenance", prov, ("tag", "detail")):
        return
    tag, detail = prov["tag"], prov["detail"]
    if tag is not None and tag not in PROVENANCE_TAGS:
        r.error(f"{where}.provenance.tag", f"{tag!r} not one of {PROVENANCE_TAGS}")
    if tag is not None and not detail:
        r.error(f"{where}.provenance.detail", "a tag needs a detail")
    r.count("provenance", where, filled=tag is not None)


def check_marginals(m: dict, r: Report) -> None:
    if not _same_keys(r, "marginals", m, ("crime_types", "by_crime_type")):
        return

    ct_block = m["crime_types"]
    if _same_keys(r, "marginals.crime_types", ct_block, ("provenance", "share")):
        _provenance(r, "crime_types", ct_block["provenance"])
        shares = ct_block["share"]
        if _same_keys(r, "marginals.crime_types.share", shares, schema.CRIME_TYPES):
            if all(_rate(r, f"marginals.crime_types.share.{k}", v) for k, v in shares.items()):
                if abs(sum(shares.values()) - 1) > TOL:
                    r.error("marginals.crime_types.share", f"sums to {sum(shares.values()):.6f}, not 1")

    if not _same_keys(r, "marginals.by_crime_type", m["by_crime_type"], schema.CRIME_TYPES):
        return
    for ct, fields in m["by_crime_type"].items():
        if not _same_keys(r, f"marginals.{ct}", fields, schema.mo_fields(ct)):
            continue
        for fname, spec in fields.items():
            where = f"marginals.{ct}.{fname}"
            kind = "rate" if fname in schema.TAG_FIELDS else "p"
            if not _same_keys(r, where, spec, ("provenance", kind)):
                continue
            _provenance(r, f"{ct}.{fname}", spec["provenance"])
            dist = spec[kind]
            if not isinstance(dist, dict) or not dist:
                r.error(f"{where}.{kind}", "expected a non-empty mapping")
                continue
            if any(not isinstance(v, str) for v in dist):
                r.error(f"{where}.{kind}", "value names must be strings (quote numbers like \"1\")")
            if fname in schema.MO_CORE_VOCAB:
                _same_keys(r, f"{where}.{kind}", dist, schema.MO_CORE_VOCAB[fname])
            if not all(_rate(r, f"{where}.{kind}.{v}", x) for v, x in dist.items()):
                continue
            if kind == "p" and abs(sum(dist.values()) - 1) > TOL:
                r.error(f"{where}.p", f"sums to {sum(dist.values()):.6f}, not 1")


# --- loadings ----------------------------------------------------------------

def _loading_entries(r: Report, where: str, entries, expected_values) -> None:
    if not _same_keys(r, where, entries, expected_values):
        return
    for value, entry in entries.items():
        at = f"{where}.{value}"
        if not isinstance(entry, dict):
            r.error(at, "expected {stealth, planned, group[, why]}")
            continue
        missing = [a for a in schema.STYLE_AXES if a not in entry]
        extra = [k for k in entry if k not in (*schema.STYLE_AXES, "why")]
        if missing or extra:
            r.error(at, f"axes missing {missing}, unexpected keys {extra}")
            continue
        weights = [entry[a] for a in schema.STYLE_AXES]
        bad = [w for w in weights if w is not None and not _is_num(w)]
        if bad:
            r.error(at, f"loadings must be numbers or null, got {bad}")
            continue
        filled = all(w is not None for w in weights)
        r.count("loadings", at, filled=filled)
        if filled and any(w != 0 for w in weights):
            r.count("why on non-zero loadings", at, filled=bool(entry.get("why")))


def check_loadings(lo: dict, m: dict, r: Report) -> None:
    if not isinstance(lo, dict):
        r.error("loadings", "expected a mapping")
        return
    allowed = ("crime_type", "mo_core", "mo_ext")
    extra = [k for k in lo if k not in allowed]
    missing = [k for k in ("mo_core", "mo_ext") if k not in lo]
    if extra or missing:
        r.error("loadings", f"sections missing {missing}, unexpected {extra}")
        return

    if "crime_type" in lo:
        _loading_entries(r, "loadings.crime_type", lo["crime_type"], schema.CRIME_TYPES)

    if _same_keys(r, "loadings.mo_core", lo["mo_core"], schema.MO_CORE):
        for fname in schema.MO_CORE:
            _loading_entries(r, f"loadings.mo_core.{fname}", lo["mo_core"][fname],
                             field_values(m, fname))

    if _same_keys(r, "loadings.mo_ext", lo["mo_ext"], tuple(schema.MO_EXT)):
        for fam, fields in schema.MO_EXT.items():
            if not _same_keys(r, f"loadings.mo_ext.{fam}", lo["mo_ext"][fam], fields):
                continue
            cts = [ct for ct in schema.CRIME_TYPES if schema.FAMILY[ct] == fam]
            for fname in fields:
                _loading_entries(r, f"loadings.mo_ext.{fam}.{fname}", lo["mo_ext"][fam][fname],
                                 field_values(m, fname, cts))


# --- states ------------------------------------------------------------------

def _check_edges(r: Report, s: dict, m: dict) -> None:
    graph = s["confusability"]
    expected = ("crime_type", *schema.ALL_MO_FIELDS)
    if not _same_keys(r, "states.confusability", graph, expected):
        return
    for fname, edges in graph.items():
        where = f"states.confusability.{fname}"
        valid = (set(schema.CRIME_TYPES) | set(s["out_of_scope_crime_types"])
                 if fname == "crime_type" else set(field_values(m, fname)))
        if not isinstance(edges, list):
            r.error(where, "expected a list of [a, b] edges")
            continue
        for edge in edges:
            if not (isinstance(edge, list) and len(edge) == 2 and edge[0] != edge[1]):
                r.error(where, f"edge {edge!r} must be two distinct values")
                continue
            unknown = [v for v in edge if v not in valid]
            if unknown:
                r.error(where, f"edge {edge} uses unknown values {unknown}")


def _check_feed(r: Report, code: str, feed, absent: list[str]) -> None:
    where = f"states.{code}.feed"
    if not _same_keys(r, where, feed, ("layout", "date_format", "date_only_format",
                                       "multi_value_delimiter", "columns")):
        return
    layout = feed["layout"]
    if layout not in FEED_LAYOUTS:
        r.error(f"{where}.layout", f"{layout!r} not one of {FEED_LAYOUTS}")
        return

    sample = datetime(2024, 3, 9, 14, 30)
    try:
        if datetime.strptime(sample.strftime(feed["date_format"]), feed["date_format"]) != sample:
            r.error(f"{where}.date_format", "does not round-trip date and time to the minute")
    except (TypeError, ValueError) as exc:
        r.error(f"{where}.date_format", str(exc))

    try:
        day = sample.date()
        if datetime.strptime(day.strftime(feed["date_only_format"]), feed["date_only_format"]).date() != day:
            r.error(f"{where}.date_only_format", "does not round-trip the date")
    except (TypeError, ValueError) as exc:
        r.error(f"{where}.date_only_format", str(exc))

    delim = feed["multi_value_delimiter"]
    if layout == "structured" and not (isinstance(delim, str) and delim):
        r.error(f"{where}.multi_value_delimiter", "a structured feed needs one")
    if layout == "free_text_mo" and delim is not None:
        r.error(f"{where}.multi_value_delimiter", "free-text feeds have no multi-value columns")

    if layout == "structured":
        expected = (*schema.FEED_META, *(
            f for f in schema.ALL_MO_FIELDS
            if f not in schema.DERIVED_AT_NORMALISATION and f not in absent))
    else:
        expected = (*schema.FEED_META, schema.MO_DESCRIPTION)
    if _same_keys(r, f"{where}.columns", feed["columns"], expected):
        native = list(feed["columns"].values())
        if any(not isinstance(n, str) or not n for n in native):
            r.error(f"{where}.columns", "native column names must be non-empty strings")
        elif len(set(native)) != len(native):
            r.error(f"{where}.columns", "native column names must be unique")


def check_states(s: dict, m: dict, r: Report) -> None:
    if not _same_keys(r, "states", s, ("out_of_scope_crime_types", "confusability", "states")):
        return
    oos = s["out_of_scope_crime_types"]
    if not isinstance(oos, list) or any(t in schema.CRIME_TYPES for t in oos):
        r.error("states.out_of_scope_crime_types", "must be a list of labels outside the property family")
        return
    _check_edges(r, s, m)

    states = s["states"]
    if not isinstance(states, dict) or not states:
        r.error("states.states", "expected a non-empty mapping")
        return
    keys = ("name", "narrative_language", "case_share", "neighbours", "districts",
            "stations_per_district", "recording", "feed")
    for code, st in states.items():
        where = f"states.{code}"
        if not _same_keys(r, where, st, keys):
            continue
        _rate(r, f"{where}.case_share", st["case_share"])
        for nb in st["neighbours"]:
            if nb not in states or nb == code:
                r.error(f"{where}.neighbours", f"{nb!r} is not another configured state")
            elif code not in states[nb].get("neighbours", []):
                r.error(f"{where}.neighbours", f"{nb} does not list {code} back")
        if not (isinstance(st["districts"], list) and st["districts"]):
            r.error(f"{where}.districts", "expected a non-empty list")
        if not (isinstance(st["stations_per_district"], int) and st["stations_per_district"] > 0):
            r.error(f"{where}.stations_per_district", "expected a positive integer")

        rec = st["recording"]
        rec_keys = ("crime_type_confusion", "value_confusion", "dropout", "structurally_absent", "narrative")
        absent: list[str] = []
        if _same_keys(r, f"{where}.recording", rec, rec_keys):
            _rate(r, f"{where}.recording.crime_type_confusion", rec["crime_type_confusion"])
            _rate(r, f"{where}.recording.value_confusion", rec["value_confusion"])
            if _same_keys(r, f"{where}.recording.dropout", rec["dropout"], ("default", "fields")):
                _rate(r, f"{where}.recording.dropout.default", rec["dropout"]["default"])
                for fname, x in (rec["dropout"]["fields"] or {}).items():
                    if fname not in schema.ALL_MO_FIELDS:
                        r.error(f"{where}.recording.dropout.fields", f"unknown field {fname!r}")
                    _rate(r, f"{where}.recording.dropout.fields.{fname}", x)
            absent = rec["structurally_absent"] or []
            for fname in absent:
                if fname not in schema.ALL_MO_FIELDS or fname in schema.DERIVED_AT_NORMALISATION:
                    r.error(f"{where}.recording.structurally_absent", f"{fname!r} is not a feed MO field")
            if _same_keys(r, f"{where}.recording.narrative", rec["narrative"],
                          ("mentions_unrecorded", "omits_recorded")):
                for k, x in rec["narrative"].items():
                    _rate(r, f"{where}.recording.narrative.{k}", x)

        _check_feed(r, code, st["feed"], absent)

    shares = [st.get("case_share") for st in states.values() if isinstance(st, dict)]
    if all(_is_num(x) for x in shares) and abs(sum(shares) - 1) > TOL:
        r.error("states", f"case_share sums to {sum(shares):.6f}, not 1")


# --- corpus ------------------------------------------------------------------

def check_corpus(c: dict, m: dict, r: Report) -> None:
    keys = ("n_cases", "date_range", "geography", "exit_same_as_approach",
            "serial_case_fraction", "repeat_rate", "style", "tau", "cross_type_sharing",
            "series_length", "gap_days", "relocation", "occurrence_window_hours",
            "registration_delay_hours", "time_band")
    if not _same_keys(r, "corpus", c, keys):
        return

    if not (isinstance(c["n_cases"], int) and c["n_cases"] > 0):
        r.error("corpus.n_cases", "expected a positive integer")
    if _same_keys(r, "corpus.date_range", c["date_range"], ("start", "end")):
        try:
            start, end = (date.fromisoformat(str(c["date_range"][k])) for k in ("start", "end"))
            if start >= end:
                r.error("corpus.date_range", "start must precede end")
        except ValueError as exc:
            r.error("corpus.date_range", str(exc))

    if _same_keys(r, "corpus.geography", c["geography"], ("same_district_rate",)):
        _rate(r, "corpus.geography.same_district_rate", c["geography"]["same_district_rate"])
    _rate(r, "corpus.exit_same_as_approach", c["exit_same_as_approach"])
    if c["cross_type_sharing"] is not None:      # null = must be passed on the command line
        _rate(r, "corpus.cross_type_sharing", c["cross_type_sharing"])

    frac = c["serial_case_fraction"]
    if not (_is_num(frac) and 0 < frac < 1):
        r.error("corpus.serial_case_fraction", "expected a number strictly between 0 and 1")

    rr = c["repeat_rate"]
    if _same_keys(r, "corpus.repeat_rate", rr, ("value", "reference_frequency")):
        q, v = rr["reference_frequency"], rr["value"]
        if _rate(r, "corpus.repeat_rate.value", v) and _rate(r, "corpus.repeat_rate.reference_frequency", q):
            if not 0 < q < v:
                r.error("corpus.repeat_rate",
                        f"need 0 < reference_frequency < value (got {q} and {v}); alpha has no solution otherwise")

    if _same_keys(r, "corpus.style", c["style"], ("axes", "sd")):
        if tuple(c["style"]["axes"]) != schema.STYLE_AXES:
            r.error("corpus.style.axes", f"must be {list(schema.STYLE_AXES)} in that order")
        _positive(r, "corpus.style.sd", c["style"]["sd"])
    _positive(r, "corpus.tau", c["tau"])

    sl = c["series_length"]
    if _same_keys(r, "corpus.series_length", sl, ("minimum", "extra_mean", "extra_dispersion")):
        if not (isinstance(sl["minimum"], int) and sl["minimum"] >= 2):
            r.error("corpus.series_length.minimum", "a series needs at least 2 crimes")
        _positive(r, "corpus.series_length.extra_mean", sl["extra_mean"])
        _positive(r, "corpus.series_length.extra_dispersion", sl["extra_dispersion"])

    gaps = c["gap_days"]
    if not (isinstance(gaps, list) and gaps):
        r.error("corpus.gap_days", "expected a non-empty list of {weight, mean}")
    else:
        ok = True
        for i, comp in enumerate(gaps):
            if _same_keys(r, f"corpus.gap_days[{i}]", comp, ("weight", "mean")):
                ok &= _rate(r, f"corpus.gap_days[{i}].weight", comp["weight"])
                ok &= _positive(r, f"corpus.gap_days[{i}].mean", comp["mean"])
            else:
                ok = False
        if ok and abs(sum(comp["weight"] for comp in gaps) - 1) > TOL:
            r.error("corpus.gap_days", "weights must sum to 1")

    rel = c["relocation"]
    if _same_keys(r, "corpus.relocation", rel, ("probability", "dormancy_days", "destination")):
        _rate(r, "corpus.relocation.probability", rel["probability"])
        d = rel["dormancy_days"]
        if not (isinstance(d, list) and len(d) == 2 and all(_is_num(x) for x in d) and 0 < d[0] <= d[1]):
            r.error("corpus.relocation.dormancy_days", "expected [low, high] with 0 < low <= high")
        if rel["destination"] != "neighbour":
            r.error("corpus.relocation.destination", "only 'neighbour' is supported")

    win = c["occurrence_window_hours"]
    if _same_keys(r, "corpus.occurrence_window_hours", win, ("by_occupancy", "by_crime_type")):
        if _same_keys(r, "corpus.occurrence_window_hours.by_occupancy", win["by_occupancy"],
                      field_values(m, "occupancy")):
            for k, spec in win["by_occupancy"].items():
                _lognormal(r, f"corpus.occurrence_window_hours.by_occupancy.{k}", spec)
        non_burglary = [ct for ct in schema.CRIME_TYPES if schema.FAMILY[ct] != "burglary"]
        if _same_keys(r, "corpus.occurrence_window_hours.by_crime_type", win["by_crime_type"], non_burglary):
            for k, spec in win["by_crime_type"].items():
                _lognormal(r, f"corpus.occurrence_window_hours.by_crime_type.{k}", spec)
    _lognormal(r, "corpus.registration_delay_hours", c["registration_delay_hours"])

    tb = c["time_band"]
    if isinstance(tb, dict):
        hours = {k: tuple(v) for k, v in (tb.get("hours") or {}).items() if isinstance(v, list)}
        if hours != schema.TIME_BAND_HOURS or tb.get("unknowable_above_hours") != schema.UNKNOWABLE_ABOVE_HOURS:
            r.error("corpus.time_band", "must match schema.TIME_BAND_HOURS and UNKNOWABLE_ABOVE_HOURS "
                                        "(the pipeline derives bands from the schema)")
    if _same_keys(r, "corpus.time_band", tb, ("hours", "unknowable_above_hours")):
        _positive(r, "corpus.time_band.unknowable_above_hours", tb["unknowable_above_hours"])
        bands = tb["hours"]
        if _same_keys(r, "corpus.time_band.hours", bands, schema.MO_CORE_VOCAB["time_band"]):
            covered: list[int] = []
            for band, span in bands.items():
                if not (isinstance(span, list) and len(span) == 2
                        and all(isinstance(h, int) and 0 <= h <= 23 for h in span) and span[0] != span[1]):
                    r.error(f"corpus.time_band.hours.{band}", "expected [start, end) hours 0–23, start != end")
                    break
                a, b = span
                covered += list(range(a, b)) if a < b else [*range(a, 24), *range(0, b)]
            else:
                if sorted(covered) != list(range(24)):
                    r.error("corpus.time_band.hours", "bands must cover every hour exactly once")


# --- vocab -------------------------------------------------------------------

def check_vocab(v: dict, s: dict, m: dict, r: Report) -> None:
    if not _same_keys(r, "vocab", v, ("crime_type", "fields")):
        return
    states = s.get("states") or {}
    structured = {code for code, st in states.items()
                  if isinstance(st, dict) and (st.get("feed") or {}).get("layout") == "structured"}

    def labels(where, entry, need):
        if not isinstance(entry, dict):
            r.error(where, "expected {state: label, en: clause}")
            return
        for key in need:
            if not (isinstance(entry.get(key), str) and entry[key].strip()):
                r.error(where, f"missing label for {key}")

    if _same_keys(r, "vocab.crime_type", v["crime_type"], schema.CRIME_TYPES):
        for ct, entry in v["crime_type"].items():
            labels(f"vocab.crime_type.{ct}", entry, (*states, "en"))

    fields = v["fields"]
    if not _same_keys(r, "vocab.fields", fields, schema.ALL_MO_FIELDS):
        return
    for fname in schema.ALL_MO_FIELDS:
        expected = field_values(m, fname)
        if fname in schema.TAG_FIELDS:
            expected = [*expected, "none_observed"]
        if not _same_keys(r, f"vocab.fields.{fname}", fields[fname], expected):
            continue
        if fname in schema.DERIVED_AT_NORMALISATION:
            need_states = []
        else:
            need_states = [code for code in structured
                           if fname not in ((states[code].get("recording") or {}).get("structurally_absent") or [])]
        for value, entry in fields[fname].items():
            labels(f"vocab.fields.{fname}.{value}", entry, (*need_states, "en"))
        for code in need_states:
            native = [e.get(code) for e in fields[fname].values() if isinstance(e, dict)]
            if len(set(native)) != len(native):
                r.error(f"vocab.fields.{fname}", f"{code} labels are not one-to-one")


# --- entry point -------------------------------------------------------------

def check(cfg: dict) -> Report:
    r = Report()
    missing = [name for name in FILES if not isinstance(cfg.get(name), dict)]
    if missing:
        r.error("config", f"empty or unreadable: {missing}")
        return r
    check_marginals(cfg["marginals"], r)
    check_loadings(cfg["loadings"], cfg["marginals"], r)
    check_states(cfg["states"], cfg["marginals"], r)
    check_corpus(cfg["corpus"], cfg["marginals"], r)
    check_vocab(cfg["vocab"], cfg["states"], cfg["marginals"], r)
    return r


def derived(cfg: dict) -> list[str]:
    c = cfg["corpus"]
    rr = c["repeat_rate"]
    q = rr["reference_frequency"]
    sl = c["series_length"]
    mean_len = sl["minimum"] + sl["extra_mean"]
    serial_cases = c["n_cases"] * c["serial_case_fraction"]
    return [
        f"alpha            {alpha(rr['value'], q):.3f}  (repeat_rate {rr['value']} at q_ref {q}; "
        f"sweep 0.2 → {alpha(0.2, q):.3f}, 0.9 → {alpha(0.9, q):.3f})",
        f"series length    mean {mean_len:g}, ≈{serial_cases / mean_len:.0f} serial offenders, "
        f"≈{c['n_cases'] - serial_cases:.0f} background one-offs",
        f"cross_type_sharing {c['cross_type_sharing'] if c['cross_type_sharing'] is not None else 'null — pass --cross-type-sharing'}",
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m linkage.config", description=__doc__.splitlines()[0])
    ap.add_argument("-v", "--verbose", action="store_true", help="list every unfilled entry")
    ap.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = load(args.config_dir)
    r = check(cfg)
    print(f"config  {args.config_dir}\n")

    print(f"errors ({len(r.errors)})")
    for e in r.errors:
        print(f"  {e}")

    print("\nunfilled")
    for group, total in r.totals.items():
        todo = r.unfilled.get(group, [])
        print(f"  {group:<26} {len(todo):>3} of {total}")
        if args.verbose:
            for item in todo:
                print(f"      {item}")

    if not r.errors:
        print("\nderived")
        for line in derived(cfg):
            print(f"  {line}")

    ready = not r.errors and not r.unfilled
    print(f"\n{'READY' if ready else 'NOT READY'} — generator "
          f"{'can run' if ready else 'refuses to run until errors and unfilled are both empty'}")
    return 1 if r.errors else 0


if __name__ == "__main__":
    sys.exit(main())
