"""Truth: offenders, their style, and what each crime actually was.

    p (config) → tilt by style s → q → θ ~ Dirichlet(alpha·q) → draw x

Draws are laid out so sweeps stay coupled. Structure (style, series length,
gaps, relocation, geography, clock uniforms) has its own stream and never
depends on repeat_rate, tau or marginals. Each crime's value draws read
fixed slots of a fixed-size uniform block, so changing a swept parameter
moves a value only where the swept distribution actually moved.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from scipy.special import betaincinv, erf, gammaincinv

from linkage import schema
from linkage.config import alpha as solve_alpha
from linkage.generate import rng as streams

AXES = schema.STYLE_AXES
TWO_WHEELERS = {"motorcycle", "scooter"}

# (crime_type, field) pairs rewritten by within-crime implications. Their
# background marginals legitimately differ from config (validator 1 skips them).
IMPLIED = frozenset({
    *((ct, "exit_mode") for ct in schema.CRIME_TYPES),
    ("SNATCHING", "approach_mode"),
    ("ATM_TAMPERING", "tools"),
})


@dataclass(frozen=True)
class FieldModel:
    name: str
    values: tuple[str, ...]
    p: np.ndarray          # categorical: sums to 1; tags: independent rates
    loadings: np.ndarray   # (n_values, n_axes)
    tag: bool

    def tilt(self, s: np.ndarray, tau: float) -> np.ndarray:
        """q for one style (s: (axes,) → (V,)) or many (s: (n, axes) → (n, V))."""
        shift = tau * (np.asarray(s) @ self.loadings.T)
        if self.tag:
            inner = (self.p > 0) & (self.p < 1)          # 0 and 1 are structural
            logit = np.zeros_like(self.p)
            logit[inner] = np.log(self.p[inner]) - np.log1p(-self.p[inner])
            return np.where(inner, 1.0 / (1.0 + np.exp(-(logit + shift))), self.p)
        with np.errstate(divide="ignore"):
            logw = np.log(self.p) + shift
        logw = logw - np.where(np.isfinite(logw), logw, -np.inf).max(axis=-1, keepdims=True)
        w = np.exp(logw)
        return w / w.sum(axis=-1, keepdims=True)

    def theta(self, q: np.ndarray, a: float, rng: np.random.Generator) -> np.ndarray:
        th = q.copy()
        if self.tag:
            inner = (q > 0) & (q < 1)
            if inner.any():
                draw = rng.beta(a * q[inner], a * (1 - q[inner]))
                bad = ~np.isfinite(draw)
                draw[bad] = q[inner][bad]
                th[inner] = draw
            return th
        support = q > 0
        draw = rng.dirichlet(a * q[support])
        if np.all(np.isfinite(draw)) and draw.sum() > 0:
            th[support] = draw / draw.sum()
        return th

    def draw(self, weights: np.ndarray, u):
        """Categorical: one uniform. Tags: one uniform per value."""
        if self.tag:
            return tuple(v for v, w, ui in zip(self.values, weights, u) if ui < w)
        cum = np.cumsum(weights)
        idx = int(np.searchsorted(cum, u * cum[-1], side="right"))
        return self.values[min(idx, len(self.values) - 1)]


@dataclass(frozen=True)
class Model:
    crime_type: FieldModel
    fields: dict            # (crime_type, field) -> FieldModel
    core_shared: dict       # mo_core field -> pooled model for person-level habits
    slots: dict             # field -> slice of a crime's uniform block
    block: int


def build_model(cfg: dict) -> Model:
    m, lo = cfg["marginals"], cfg["loadings"]

    def matrix(entries, values):
        if entries is None:
            return np.zeros((len(values), len(AXES)))
        return np.array([[float(entries[v][ax]) for ax in AXES] for v in values])

    cts = schema.CRIME_TYPES
    crime_type = FieldModel("crime_type", cts,
                            np.array([m["crime_types"]["share"][ct] for ct in cts], float),
                            matrix(lo.get("crime_type"), cts), False)
    fields = {}
    for ct in cts:
        fam = schema.FAMILY[ct]
        for f in schema.mo_fields(ct):
            dist = m["by_crime_type"][ct][f]["rate" if f in schema.TAG_FIELDS else "p"]
            values = tuple(dist)
            entries = lo["mo_core"][f] if f in schema.MO_CORE else lo["mo_ext"][fam][f]
            fields[ct, f] = FieldModel(f, values, np.array([dist[v] for v in values], float),
                                       matrix(entries, values), f in schema.TAG_FIELDS)

    # Pooled mo_core marginals: the centre of an offender's person-level habit.
    shares = np.array([m["crime_types"]["share"][ct] for ct in cts])
    core_shared = {}
    for f in schema.MO_CORE:
        first = fields[cts[0], f]
        if any(fields[ct, f].values != first.values for ct in cts):
            raise ValueError(f"{f}: mo_core value order differs between crime types")
        pooled = shares @ np.array([fields[ct, f].p for ct in cts])
        core_shared[f] = FieldModel(f, first.values, pooled if first.tag else pooled / pooled.sum(),
                                    first.loadings, first.tag)

    slots, pos = {"crime_type": slice(0, 1)}, 1
    for f in schema.ALL_MO_FIELDS:
        width = 1
        if f in schema.TAG_FIELDS:
            width = max(len(fm.values) for (_, name), fm in fields.items() if name == f)
        slots[f] = slice(pos, pos + width)
        pos += width
    slots["exit_copy"] = slice(pos, pos + 1)
    return Model(crime_type, fields, core_shared, slots, pos + 1)


def theta_coupled(q: np.ndarray, a: float, z: np.ndarray, tag: bool) -> np.ndarray:
    """θ ~ Dirichlet(a·q) (or Beta per tag) built from standard normals by
    inverse CDF, so that correlating the normals shares habit across crime
    types without touching anything within a type.

    mo_core fields (time band, group size, concealment, target selection,
    approach, tools, property, exit) are properties of the offender, so they
    should carry across the crime types one offender commits; mo_ext stays
    type-specific. Here the offender's habit is a set of quantiles: the same
    z, read against each type's own centre q, favours the same values while
    each type keeps its own base rates — a night worker is a night worker
    everywhere, and a rarely-nocturnal crime type still rarely happens at
    night.

    Because Φ(z) is uniform whatever z's correlation, each type's θ keeps
    EXACTLY the law it had, so cross_type_sharing cannot change within-type
    repetition. Two earlier attempts got this wrong and were discarded:
    blending the drawn thetas flattened them (same-type hit@10 fell a third at
    mid rho), and displacing the centre then re-solving alpha was infeasible —
    a sharpened centre already agrees more often than the target, so alpha ran
    to its ceiling (1.25 → ~4250) and offenders lost their spread. See
    FINDINGS.md.
    """
    u = np.clip(0.5 * (1.0 + erf(np.asarray(z) / np.sqrt(2.0))), 1e-12, 1 - 1e-12)
    if tag:
        out = q.copy()
        inner = (q > 0) & (q < 1)
        if inner.any():
            out[inner] = betaincinv(a * q[inner], a * (1 - q[inner]), u[inner])
        return out
    support = q > 0
    g = np.zeros_like(q)
    g[support] = gammaincinv(a * q[support], u[support])
    total = g.sum()
    if not np.isfinite(total) or total <= 0:      # every gamma underflowed
        out = np.zeros_like(q)
        out[int(np.argmax(q))] = 1.0
        return out
    return g / total


def draw_values(model: Model, ct_weights, weights_for, u: np.ndarray, exit_same: float):
    ct = model.crime_type.draw(ct_weights, u[0])
    vals = {}
    for f in schema.mo_fields(ct):
        fm = model.fields[ct, f]
        su = u[model.slots[f]]
        vals[f] = fm.draw(weights_for(ct, f), su if fm.tag else su[0])
    _implications(ct, vals, u[model.slots["exit_copy"]][0], exit_same)
    return ct, vals


def _implications(ct: str, vals: dict, u_exit: float, exit_same: float) -> None:
    if ct == "VEHICLE_THEFT":
        vals["exit_mode"] = "two_wheeler" if vals["vehicle_class"] in TWO_WHEELERS else "four_wheeler"
    elif ct == "SNATCHING":
        mode = "on_foot" if vals["vehicle_used"] == "none" else "two_wheeler"
        vals["approach_mode"] = vals["exit_mode"] = mode
    elif u_exit < exit_same:
        vals["exit_mode"] = vals["approach_mode"]
    if ct == "ATM_TAMPERING" and vals["attack_method"] == "gas_cutting" and "gas_cutter" not in vals["tools"]:
        order = schema.MO_CORE_VOCAB["tools"]
        vals["tools"] = tuple(sorted((*vals["tools"], "gas_cutter"), key=order.index))


def _hours(span: list[int]) -> list[int]:
    a, b = span
    return list(range(a, b)) if a < b else [*range(a, 24), *range(0, b)]


def _minute(t: datetime) -> datetime:
    return t.replace(second=0, microsecond=0)


class World:
    """Geography and clocks: states, districts, stations, windows."""

    def __init__(self, cfg: dict):
        c, s = cfg["corpus"], cfg["states"]["states"]
        self.start = date.fromisoformat(str(c["date_range"]["start"]))
        self.days = (date.fromisoformat(str(c["date_range"]["end"])) - self.start).days
        self.states = s
        self.codes = list(s)
        self.share_cum = np.cumsum([s[k]["case_share"] for k in self.codes])
        self.band_hours = {b: _hours(span) for b, span in c["time_band"]["hours"].items()}
        self.windows = c["occurrence_window_hours"]
        self.delay = c["registration_delay_hours"]
        self.same_district = c["geography"]["same_district_rate"]

    def pick_state(self, u: float) -> str:
        i = int(np.searchsorted(self.share_cum, u * self.share_cum[-1], side="right"))
        return self.codes[min(i, len(self.codes) - 1)]

    def district(self, code: str, u: float) -> str:
        d = self.states[code]["districts"]
        return d[min(int(u * len(d)), len(d) - 1)]

    def station(self, code: str, district: str, u: float) -> str:
        n = self.states[code]["stations_per_district"]
        return f"{code}-{district[:3].upper()}-{min(int(u * n), n - 1) + 1:02d}"

    def clock(self, day_offset: float, ct: str, vals: dict, u_tod: float, u_pos: float,
              z_width: float, z_delay: float):
        hours = self.band_hours[vals["time_band"]]
        x = u_tod * len(hours)
        h = hours[min(int(x), len(hours) - 1)]
        at = datetime.combine(self.start + timedelta(days=int(day_offset)), datetime.min.time())
        at += timedelta(hours=h, minutes=int((x - int(x)) * 60))
        spec = (self.windows["by_occupancy"][vals["occupancy"]] if schema.FAMILY[ct] == "burglary"
                else self.windows["by_crime_type"][ct])
        width = spec["median"] * float(np.exp(spec["sigma"] * z_width))
        occurred_from = _minute(at - timedelta(hours=u_pos * width))
        occurred_to = max(_minute(occurred_from + timedelta(hours=width)), at)
        delay = self.delay["median"] * float(np.exp(self.delay["sigma"] * z_delay))
        registered = _minute(occurred_to + timedelta(hours=delay))
        return at, occurred_from, occurred_to, registered


def _case(offender_id, serial, k, s, code, district, station, ct, vals, clock) -> dict:
    at, occurred_from, occurred_to, registered = clock
    row = {
        "offender_id": offender_id, "is_serial": serial, "series_index": k,
        **{f"style_{ax}": float(x) for ax, x in zip(AXES, s)},
        "state_code": code, "district": district, "police_station": station,
        "crime_type": ct, "true_at": at, "occurred_from": occurred_from,
        "occurred_to": occurred_to, "registered_at": registered,
    }
    for f in schema.ALL_MO_FIELDS:
        v = vals.get(f)
        row[f] = ";".join(v) if isinstance(v, tuple) else v
    return row


def _offender(i: int, seed: int, cfg: dict, model: Model, world: World, a: float, tau: float):
    c = cfg["corpus"]
    rs = streams.stream(seed, streams.STRUCTURE, i)
    s = rs.normal(0.0, c["style"]["sd"], len(AXES))
    sl = c["series_length"]
    r_, mean = sl["extra_dispersion"], sl["extra_mean"]
    drawn = sl["minimum"] + int(rs.negative_binomial(r_, r_ / (r_ + mean)))
    comps = c["gap_days"]
    wcum = np.cumsum([g["weight"] for g in comps])
    which = np.minimum(np.searchsorted(wcum, rs.random(drawn - 1) * wcum[-1], side="right"), len(comps) - 1)
    gaps = rs.exponential(np.array([g["mean"] for g in comps], float)[which])
    rel = c["relocation"]
    wants_move = rs.random() < rel["probability"]
    move_at = int(rs.integers(1, drawn))
    dormancy = rs.uniform(*rel["dormancy_days"])
    u_home, u_dest, u_start = rs.random(3)
    per = rs.random((drawn, 5))              # district, stay, station, time of day, window position
    z = rs.standard_normal((drawn, 2))       # window width, registration delay

    home = world.pick_state(u_home)
    nbs = world.states[home]["neighbours"]
    dest = nbs[min(int(u_dest * len(nbs)), len(nbs) - 1)] if (wants_move and nbs) else None
    if dest:
        gaps[move_at - 1] += dormancy
    offsets = np.concatenate([[0.0], np.cumsum(gaps)])
    span = offsets[-1]
    if span <= world.days:
        offsets += u_start * (world.days - span)
    elif (offsets <= world.days).sum() < 2:
        offsets *= world.days / span
    n = int((offsets <= world.days).sum())
    moved = dest is not None and move_at < n

    rho = c["cross_type_sharing"]
    habit = {}
    if rho:                       # own stream, so the rho = 0 theta draws keep their order
        rh = streams.stream(seed, streams.SHARED, i)
        habit = {f: rh.standard_normal(len(fm.values)) for f, fm in model.core_shared.items()}

    rt = streams.stream(seed, streams.THETA, i)
    theta_ct = model.crime_type.theta(model.crime_type.tilt(s, tau), a, rt)
    theta = {}
    spread = np.sqrt(1 - rho ** 2) if rho else 0.0
    for key, fm in model.fields.items():
        q = fm.tilt(s, tau)
        if rho and fm.name in schema.MO_CORE:
            z_core = rho * habit[fm.name] + spread * rt.standard_normal(len(q))
            theta[key] = theta_coupled(q, a, z_core, fm.tag)
        else:
            theta[key] = fm.theta(q, a, rt)
    blocks = streams.stream(seed, streams.CRIMES, i).random((drawn, model.block))

    offender_id = f"S{i:05d}"
    base: dict[str, str] = {}
    crimes = []
    for k in range(n):
        code = dest if (moved and k >= move_at) else home
        base.setdefault(code, world.district(code, per[k, 0]))
        district = base[code] if per[k, 1] < world.same_district else world.district(code, per[k, 0])
        ct, vals = draw_values(model, theta_ct, lambda ct_, f: theta[ct_, f], blocks[k], c["exit_same_as_approach"])
        clock = world.clock(offsets[k], ct, vals, per[k, 3], per[k, 4], z[k, 0], z[k, 1])
        crimes.append(_case(offender_id, True, k, s, code, district,
                            world.station(code, district, per[k, 2]), ct, vals, clock))

    offender = {
        "offender_id": offender_id,
        **{f"style_{ax}": float(x) for ax, x in zip(AXES, s)},
        "repeat_rate": c["repeat_rate"]["value"], "alpha": a,
        "cross_type_sharing": c["cross_type_sharing"],
        "series_length_drawn": drawn, "series_length": n,
        "home_state": home,
        "destination_state": dest if moved else None,
        "relocation_index": move_at if moved else None,
    }
    return offender, crimes


def _background(seed: int, n: int, cfg: dict, model: Model, world: World) -> list[dict]:
    rb = streams.stream(seed, streams.BACKGROUND)
    u_state, day = rb.random(n), rb.random(n)
    per = rb.random((n, 5))
    z = rb.standard_normal((n, 2))
    blocks = rb.random((n, model.block))
    zero = np.zeros(len(AXES))
    exit_same = cfg["corpus"]["exit_same_as_approach"]
    rows = []
    for j in range(n):
        code = world.pick_state(u_state[j])
        district = world.district(code, per[j, 0])
        # s = 0 exactly: q = p, and a single crime's marginal under θ ~ Dir(alpha·p) is p.
        ct, vals = draw_values(model, model.crime_type.p, lambda ct_, f: model.fields[ct_, f].p,
                               blocks[j], exit_same)
        clock = world.clock(day[j] * world.days, ct, vals, per[j, 3], per[j, 4], z[j, 0], z[j, 1])
        rows.append(_case(f"B{j:06d}", False, 0, zero, code, district,
                          world.station(code, district, per[j, 2]), ct, vals, clock))
    return rows


def _assign_fir(truth: pd.DataFrame) -> pd.DataFrame:
    """FIR numbers run per station per year in registration order, so they carry
    time (legitimately) but nothing about who committed the crime."""
    truth = truth.sort_values(["registered_at", "case_uid"]).reset_index(drop=True)
    year = truth["registered_at"].dt.year
    serial = truth.groupby([truth["police_station"], year]).cumcount() + 1
    truth["fir_no"] = [f"{ps}/{k:04d}/{y}" for ps, k, y in zip(truth["police_station"], serial, year)]
    truth["case_id"] = [schema.case_id(st, fir, y)
                        for st, fir, y in zip(truth["state_code"], truth["fir_no"], year)]
    return truth.sort_values("case_uid").reset_index(drop=True)


def generate(cfg: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Truth table (one row per case, indexed by case_uid) and offenders table."""
    c = cfg["corpus"]
    model, world = build_model(cfg), World(cfg)
    rr = c["repeat_rate"]
    a = solve_alpha(rr["value"], rr["reference_frequency"])
    target = round(c["n_cases"] * c["serial_case_fraction"])

    offenders, rows = [], []
    i = 0
    while len(rows) < target:
        off, crimes = _offender(i, seed, cfg, model, world, a, c["tau"])
        offenders.append(off)
        rows.extend(crimes)
        i += 1
    rows.extend(_background(seed, max(c["n_cases"] - len(rows), 0), cfg, model, world))

    truth = pd.DataFrame(rows)
    truth.insert(0, "case_uid", np.arange(len(truth)))
    return _assign_fir(truth), pd.DataFrame(offenders)
