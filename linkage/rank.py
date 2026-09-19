"""Ranking on top of MO + time evidence. Pure numpy; the API Lambda runs this.

Two adjustments, each an explicit reason an analyst can read:

distinctiveness (hub correction)
    Some FIRs have such generic MO that they sit near the top of many lists.
    r(case) = mean of the case's top-10 evidence in its pool (label-free).
    A pair's score moves by -beta * ((r_a + r_b) / 2 - mean r): a match
    between two unusual FIRs counts for more, a match with a "looks like
    everything" FIR for less. Default ranking. FINDINGS §14.

place (only when an officer chooses "nearby first")
    log2 P(where | same offender) / P(where | random pair) for same police
    station / same district / same state / another state. Location stays out
    of the default ranking: it is what lets cross-state links surface at all,
    and it buries them when used (FINDINGS §14).
"""
from __future__ import annotations

import numpy as np

HUB_K = 10
GEO_BINS = ("same_station", "same_district", "same_state", "other_state")
MODES = ("blind", "nearby")


def geo_bins(state_a, district_a, station_a, state_b, district_b, station_b) -> np.ndarray:
    """Index into GEO_BINS for each pair; arguments broadcast."""
    sa, sb = np.asarray(state_a), np.asarray(state_b)
    out = np.full(np.broadcast(sa, sb).shape, 3)
    out[sa == sb] = 2
    out[np.asarray(district_a) == np.asarray(district_b)] = 1
    out[np.asarray(station_a) == np.asarray(station_b)] = 0
    return out


def fit_geo_bits(same_bins: np.ndarray, random_bins: np.ndarray) -> np.ndarray:
    same = np.bincount(same_bins, minlength=len(GEO_BINS)) + 0.5
    rand = np.bincount(random_bins, minlength=len(GEO_BINS)) + 0.5
    return np.log2((same / same.sum()) / (rand / rand.sum()))


def hub_scores(scorer, codes: dict, types: np.ndarray, k: int = HUB_K) -> np.ndarray:
    """r per case: mean of its top-k evidence against its own same-type pool."""
    r = np.zeros(len(types))
    for pool in np.unique(types):
        pool = str(pool)
        if pool not in scorer.pools:
            continue
        rows = np.flatnonzero(types == pool)
        cand = {name: v[rows] for name, v in codes[pool].items()}
        for q in rows:
            ev = scorer.evidence(pool, scorer.field_bits(pool, {name: v[q] for name, v in codes[pool].items()}, cand))
            ev[rows == q] = -np.inf
            r[q] = -np.mean(np.partition(-ev, k)[:k])
    return r


class Ranker:
    """Adds the distinctiveness and place terms to raw evidence."""

    def __init__(self, spec: dict | None, hub_r: np.ndarray | None, types: np.ndarray,
                 states=None, districts=None, stations=None):
        spec = spec or {}
        self.hub_beta = float(spec.get("hub_beta", 0.0)) if hub_r is not None else 0.0
        self.geo_beta = float(spec.get("geo_beta", 0.0))
        self.geo_bits = np.asarray(spec.get("geo_bits", [0.0] * len(GEO_BINS)), float)
        self.r = hub_r
        self.mean_r = {str(p): float(hub_r[types == p].mean()) for p in np.unique(types)} if hub_r is not None else {}
        self.places = (np.asarray(states), np.asarray(districts), np.asarray(stations)) if states is not None else None

    def terms(self, pool: str, q: int, cand: np.ndarray, mode: str = "blind") -> dict:
        """{'distinctiveness': array, 'place': array} in bits, zeros where not used."""
        r_q = self.r[q] if self.r is not None else 0.0
        where = tuple(p[q] for p in self.places) if self.places is not None else None
        return self.terms_for(pool, r_q, where, cand, mode)

    def terms_for(self, pool: str, r_q: float, where_q: tuple | None, cand: np.ndarray, mode: str = "blind") -> dict:
        """The same terms for a query given by value — a newly entered FIR that is
        not in the corpus: its r and its (state, district, police station)."""
        n = np.shape(cand)
        hub = np.zeros(n)
        if self.hub_beta and pool in self.mean_r:
            hub = -self.hub_beta * ((r_q + self.r[cand]) / 2 - self.mean_r[pool])
        place = np.zeros(n)
        if mode == "nearby" and self.places is not None and self.geo_beta and where_q is not None:
            st, di, ps = self.places
            place = self.geo_beta * self.geo_bits[geo_bins(*where_q, st[cand], di[cand], ps[cand])]
        return {"distinctiveness": hub, "place": place}

    def pair_terms(self, pool: str, a: np.ndarray, b: np.ndarray, mode: str = "blind") -> dict:
        """Vectorised over pair lists (for random-pair rarity)."""
        n = np.shape(a)
        hub = np.zeros(n)
        if self.hub_beta and pool in self.mean_r:
            hub = -self.hub_beta * ((self.r[a] + self.r[b]) / 2 - self.mean_r[pool])
        place = np.zeros(n)
        if mode == "nearby" and self.places is not None and self.geo_beta:
            st, di, ps = self.places
            place = self.geo_beta * self.geo_bits[geo_bins(st[a], di[a], ps[a], st[b], di[b], ps[b])]
        return {"distinctiveness": hub, "place": place}
