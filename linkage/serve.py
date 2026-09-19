"""Shortlists, pair explanations and lead strength. Pure numpy + stdlib; the API
Lambda runs this.

Strength is stated the way an analyst can use it: how rare is this much
similarity between UNRELATED cases? The bundle carries a large sample of
scores from random unrelated pairs per pool (and per ranking mode), so "1 in
25,000 unrelated pairs look this alike" is measured, not modelled. Tiers
follow from it. There is never a probability (CLAUDE.md rule 4): a lead is a
reason to look, not a finding.

A pair's score is its MO + time evidence plus the ranking terms from
linkage.rank — distinctiveness always, place only in "nearby" mode — and
every term is returned as a reason, so the reasons add up to the score.

Same-type and cross-type are separate pools with separate models; rarity is
measured within each pool, which is also what makes leads from different
pools comparable in one inbox.
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np

from linkage import checks, features, rank, schema
from linkage.score import Scorer

SAME, CROSS = "same_type", "cross_type"
# (minimum "1 in N", tier id, label). Measured on unrelated pairs within the pool.
TIERS = ((10_000, "strong", "Strong lead"), (1_000, "possible", "Worth checking"), (0, "weak", "Weak lead"))
REASON_ORDER = {"timing": 0, "place": 1, "shared": 2, "different": 3, "distinctiveness": 4, "not_recorded": 5}


class Shortlister:
    def __init__(self, weights: dict, case_ids: list[str], crime_types: list[str], codes: dict,
                 meta: dict, cases: dict | None = None, truth_groups: dict | None = None,
                 hub_r: np.ndarray | None = None):
        self.scorer = Scorer(weights)
        self.case_ids = list(case_ids)
        self.index = {cid: i for i, cid in enumerate(self.case_ids)}
        self.types = np.asarray(crime_types)
        self.codes = codes                  # pool -> field -> int array over all cases
        self.meta = meta
        self.cases = cases or {}
        self.truth_groups = truth_groups    # synthetic ground truth, demo mode only
        place = [[self.cases.get(c, {}).get(k) for c in self.case_ids] for k in ("state_code", "district", "police_station")]
        self.ranker = rank.Ranker(weights.get("rank"), hub_r, self.types, *place)
        self.rarity = {}
        for pool, ctx in meta["pools"].items():
            for mode, key in (("blind", "random_pairs"), ("nearby", "random_pairs_nearby")):
                r = ctx.get(key)
                if r:
                    self.rarity[(pool, mode)] = (np.asarray(r["body"], float), np.asarray(r["tail"], float), int(r["n"]))

    # --- strength -------------------------------------------------------------------

    def one_in(self, pool: str, score: float, mode: str = "blind") -> tuple[int, bool]:
        """(N, rarer_than_measured): about 1 in N unrelated pairs score at least this."""
        key = (pool, mode) if (pool, mode) in self.rarity else (pool, "blind")
        if key not in self.rarity:
            return 1, False
        body, tail, n = self.rarity[key]
        if score >= tail[0]:
            count = len(tail) - int(np.searchsorted(tail, score, side="left"))
            return (n, True) if count == 0 else (max(1, round(n / count)), False)
        below = float(np.interp(score, body, np.linspace(0, 0.99, len(body))))
        return max(1, round(1 / max(1 - below, 1e-9))), False

    @staticmethod
    def tier(one_in: int) -> tuple[str, str]:
        for threshold, tier_id, label in TIERS:
            if one_in >= threshold:
                return tier_id, label
        return TIERS[-1][1], TIERS[-1][2]

    # --- scoring ------------------------------------------------------------------------

    def score(self, pool: str, q: int, cand: np.ndarray, mode: str = "blind"):
        """(field bits, ranking terms, total score) of q against each candidate."""
        codes = self.codes[pool]
        bits = self.scorer.field_bits(pool, {k: v[q] for k, v in codes.items()}, {k: v[cand] for k, v in codes.items()})
        terms = self.ranker.terms(pool, q, cand, mode if pool != features.CROSS else "blind")
        total = self.scorer.evidence(pool, bits) + terms["distinctiveness"] + terms["place"]
        return bits, terms, total

    # --- explanations ------------------------------------------------------------------

    def _pool_for(self, a: int, b: int) -> str:
        return str(self.types[a]) if self.types[a] == self.types[b] else features.CROSS

    def _reasons(self, pool: str, case_a: dict, case_b: dict, bits_row: np.ndarray, terms: dict | None = None) -> list[dict]:
        """Every field's contribution for one pair, as reasons that add up to the score.
        Takes the two FIR records, so a newly entered FIR is explained the same way."""
        spec = self.scorer.pools[pool]
        contrib = self.scorer.contributions(pool, bits_row)
        out = []
        names = self.scorer.evidence_names(pool)
        if len(names) > len(spec["fields"]):
            days = _days_apart(case_a.get("occurred_from"), case_b.get("occurred_from"))
            out.append({"field": "days_apart", "kind": "timing", "days": days,
                        "bits": round(float(contrib[len(spec["fields"])]), 3)})
        for i, f in enumerate(spec["fields"]):
            va, vb = case_a.get(f), case_b.get(f)
            known = all(v is not None and v not in schema.TOKENS for v in (va, vb))
            item = {"field": f, "value": va, "other_value": vb, "bits": round(float(contrib[i]), 3)}
            if not known:
                item["kind"] = "not_recorded"
            elif f in schema.TAG_FIELDS:
                shared = sorted(set(va.split(";")) & set(vb.split(";")) - {""})
                item["kind"] = "shared" if shared or (va == vb) else "different"
                vocab, u = spec["vocab"][f], spec["u"][f]
                if shared:
                    rarest = min(shared, key=lambda t: u[vocab.index(t)] if t in vocab else 1.0)
                    item["shared_value"] = rarest
                    item["share"] = round(float(u[vocab.index(rarest)]), 3) if rarest in vocab else None
            else:
                item["kind"] = "shared" if va == vb else "different"
                vocab, u = spec["vocab"][f], spec["u"][f]
                if va == vb and va in vocab:
                    item["shared_value"] = va
                    item["share"] = round(float(u[vocab.index(va)]), 3)
            out.append(item)
        if terms is not None:
            if abs(terms["distinctiveness"]) >= 0.005:
                out.append({"field": "distinctiveness", "kind": "distinctiveness",
                            "bits": round(float(terms["distinctiveness"]), 3)})
            if abs(terms["place"]) >= 0.005:
                bin_ = rank.geo_bins(case_a.get("state_code"), case_a.get("district"), case_a.get("police_station"),
                                     case_b.get("state_code"), case_b.get("district"), case_b.get("police_station"))
                out.append({"field": "place", "kind": "place", "where": rank.GEO_BINS[int(bin_)],
                            "bits": round(float(terms["place"]), 3)})
        return sorted(out, key=lambda r: (REASON_ORDER[r["kind"]], -r["bits"] if r["kind"] == "shared" else r["bits"]))

    def _summary(self, idx: int) -> dict:
        c = self.cases.get(self.case_ids[idx], {})
        return {k: c.get(k) for k in ("case_id", "fir_no", "crime_type", "state_code", "district",
                                     "police_station", "registered_at", "occurred_from")}

    def _strength(self, pool: str, score: float, mode: str = "blind") -> dict:
        n, rarer = self.one_in(pool, score, mode)
        tier_id, label = self.tier(n)
        return {"bits": round(float(score), 2), "one_in": n, "rarer_than_measured": rarer,
                "tier": tier_id, "tier_label": label}

    def _truth(self, a: int, b: int) -> dict:
        if self.truth_groups is None:
            return {}
        ga, gb = self.truth_groups.get(self.case_ids[a]), self.truth_groups.get(self.case_ids[b])
        return {"ground_truth_link": bool(ga is not None and ga == gb)}

    # --- API surface -------------------------------------------------------------------------

    def shortlist(self, case_id: str, scope: str = "same", limit: int = 10, mode: str = "blind") -> dict:
        if case_id not in self.index:
            raise KeyError(case_id)
        if scope not in ("same", "all"):
            raise ValueError("scope must be 'same' or 'all'")
        if mode not in rank.MODES:
            raise ValueError(f"rank must be one of {list(rank.MODES)}")
        q = self.index[case_id]
        crime_type = str(self.types[q])
        lists = {SAME: self._rank(q, crime_type, self.types == crime_type, limit, mode)}
        if scope == "all":
            lists[CROSS] = self._rank(q, features.CROSS, self.types != crime_type, limit, "blind")
        return {"case": self._summary(q), "scope": scope, "rank": mode, "lists": lists}

    def _rank(self, q: int, pool: str, candidate_mask: np.ndarray, limit: int, mode: str) -> dict:
        cand = np.flatnonzero(candidate_mask)
        cand = cand[cand != q]
        bits, terms, total = self.score(pool, q, cand, mode)
        take = min(limit, len(cand))
        top = np.argpartition(-total, take - 1)[:take] if take < len(cand) else np.arange(len(cand))
        top = top[np.lexsort((cand[top], -total[top]))]
        items = []
        for rank_, j in enumerate(top, 1):
            c = int(cand[j])
            reasons = self._reasons(pool, self.cases.get(self.case_ids[q], {}), self.cases.get(self.case_ids[c], {}),
                                    bits[j], {k: v[j] for k, v in terms.items()})
            items.append({"rank": rank_, **self._summary(c), **self._strength(pool, total[j], mode if pool != features.CROSS else "blind"),
                          "cross_state": self.cases.get(self.case_ids[c], {}).get("state_code")
                          != self.cases.get(self.case_ids[q], {}).get("state_code"),
                          "reasons": reasons, **self._truth(q, c)})
        context = self.meta["pools"].get(pool, {})
        return {"pool": pool, "pool_size": int(len(cand)), "items": items,
                "typical_true_link_bits": context.get("typical_true_link_bits"),
                "breakeven_bits": context.get("breakeven_bits")}

    def pair(self, case_a: str, case_b: str, mode: str = "blind") -> dict:
        """Everything the side-by-side comparison needs for one pair."""
        if case_a not in self.index or case_b not in self.index:
            raise KeyError(case_a if case_a not in self.index else case_b)
        if mode not in rank.MODES:
            raise ValueError(f"rank must be one of {list(rank.MODES)}")
        a, b = self.index[case_a], self.index[case_b]
        if a == b:
            raise ValueError("a case cannot be compared with itself")
        pool = self._pool_for(a, b)
        mode = mode if pool != features.CROSS else "blind"
        bits, terms, total = self.score(pool, a, np.array([b]), mode)
        ca, cb = self.cases.get(case_a, {}), self.cases.get(case_b, {})
        return {"pool": pool, "same_type": pool != features.CROSS, "rank": mode,
                "case_a": ca, "case_b": cb, **self._strength(pool, float(total[0]), mode),
                "cross_state": ca.get("state_code") != cb.get("state_code"),
                "days_apart": _days_apart(ca.get("occurred_from"), cb.get("occurred_from")),
                "reasons": self._reasons(pool, ca, cb, bits[0], {k: v[0] for k, v in terms.items()}),
                "breakeven_bits": self.meta["pools"].get(pool, {}).get("breakeven_bits"),
                **self._truth(a, b)}


    # --- a FIR that is not in the corpus ----------------------------------------------

    def match_record(self, record: dict, mode: str = "blind", limit: int = 10, exclude: str | None = None) -> dict:
        """Shortlists for a newly entered FIR, scored exactly as a stored one:
        against every FIR of its type (and, core habits only, of other types).
        `exclude` hides one stored case — the demo re-enters a test FIR as new."""
        if mode not in rank.MODES:
            raise ValueError(f"rank must be one of {list(rank.MODES)}")
        crime_type = record.get("crime_type")
        if crime_type not in self.scorer.pools:
            raise ValueError("unknown crime type")
        days = _day_number(record.get("occurred_from"))
        where = (record.get("state_code"), record.get("district"), record.get("police_station"))
        keep = np.ones(len(self.case_ids), dtype=bool)
        if exclude in self.index:
            keep[self.index[exclude]] = False
        lists, r_new = {}, 0.0
        for key, pool, mask, m in ((SAME, crime_type, self.types == crime_type, mode),
                                   (CROSS, features.CROSS, self.types != crime_type, "blind")):
            if pool not in self.scorer.pools:
                continue
            spec = self.scorer.pools[pool]
            q = {k: v[0] for k, v in self.scorer.encode(pool, {**{f: [record.get(f)] for f in spec["fields"]},
                                                               features.DAYS: [days]}).items()}
            cand = np.flatnonzero(mask & keep)
            bits = self.scorer.field_bits(pool, q, {k: v[cand] for k, v in self.codes[pool].items()})
            evidence = self.scorer.evidence(pool, bits)
            if key == SAME:          # the new FIR's own typical top-10 evidence, as for stored FIRs
                k = min(rank.HUB_K, len(evidence) - 1)
                r_new = float(-np.mean(np.partition(-evidence, k)[:k]))
            terms = self.ranker.terms_for(pool, r_new, where, cand, m)
            total = evidence + terms["distinctiveness"] + terms["place"]
            take = min(limit, len(cand))
            top = np.argpartition(-total, take - 1)[:take]
            top = top[np.lexsort((cand[top], -total[top]))]
            items = []
            for rank_, j in enumerate(top, 1):
                c = int(cand[j])
                case_c = self.cases.get(self.case_ids[c], {})
                items.append({"rank": rank_, **self._summary(c), **self._strength(pool, total[j], m),
                              "cross_state": case_c.get("state_code") != record.get("state_code"),
                              "reasons": self._reasons(pool, record, case_c, bits[j], {n: v[j] for n, v in terms.items()}),
                              **({"ground_truth_link": self.truth_groups.get(self.case_ids[c]) is not None and
                                  self.truth_groups.get(self.case_ids[c]) == self.truth_groups.get(exclude)}
                                 if self.truth_groups is not None and exclude else {})})
            lists[key] = {"pool": pool, "pool_size": int(len(cand)), "items": items}
        return {"rank": mode, "type_check": checks.type_check(self.scorer.pools, record, crime_type), "lists": lists}


def _day_number(x: str | None) -> int:
    """Days since 1970-01-01, as linkage.dataset.day_numbers; -1 if unknown."""
    try:
        return (datetime.fromisoformat(str(x)).date() - date(1970, 1, 1)).days
    except (TypeError, ValueError):
        return -1


def _days_apart(x: str | None, y: str | None) -> int | None:
    try:
        return abs((datetime.fromisoformat(x) - datetime.fromisoformat(y)).days)
    except (TypeError, ValueError):
        return None
