"""Shortlists, pair explanations and lead strength. Pure numpy + stdlib; the API
Lambda runs this.

Strength is stated the way an analyst can use it: how rare is this much
similarity between UNRELATED cases? The bundle carries a large sample of
evidence scores from random unrelated pairs per pool, so "1 in 25,000
unrelated pairs look this alike" is measured, not modelled. Tiers follow from
it. There is never a probability (CLAUDE.md rule 4): a lead is a reason to
look, not a finding.

Same-type and cross-type are separate pools with separate models; rarity is
measured within each pool, which is also what makes leads from different
pools comparable in one inbox.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from linkage import features, schema
from linkage.score import Scorer

SAME, CROSS = "same_type", "cross_type"
# (minimum "1 in N", tier id, label). Measured on unrelated pairs within the pool.
TIERS = ((10_000, "strong", "Strong lead"), (1_000, "possible", "Worth checking"), (0, "weak", "Weak lead"))


class Shortlister:
    def __init__(self, weights: dict, case_ids: list[str], crime_types: list[str], codes: dict,
                 meta: dict, cases: dict | None = None, truth_groups: dict | None = None):
        self.scorer = Scorer(weights)
        self.case_ids = list(case_ids)
        self.index = {cid: i for i, cid in enumerate(self.case_ids)}
        self.types = np.asarray(crime_types)
        self.codes = codes                  # pool -> field -> int array over all cases
        self.meta = meta
        self.cases = cases or {}
        self.truth_groups = truth_groups    # synthetic ground truth, demo mode only
        self.rarity = {}
        for pool, ctx in meta["pools"].items():
            r = ctx.get("random_pairs")
            if r:
                self.rarity[pool] = (np.asarray(r["body"], float), np.asarray(r["tail"], float), int(r["n"]))

    # --- strength -------------------------------------------------------------------

    def one_in(self, pool: str, bits: float) -> tuple[int, bool]:
        """(N, rarer_than_measured): about 1 in N unrelated pairs score at least this."""
        if pool not in self.rarity:
            return 1, False
        body, tail, n = self.rarity[pool]
        if bits >= tail[0]:
            count = len(tail) - int(np.searchsorted(tail, bits, side="left"))
            return (n, True) if count == 0 else (max(1, round(n / count)), False)
        below = float(np.interp(bits, body, np.linspace(0, 0.99, len(body))))
        return max(1, round(1 / max(1 - below, 1e-9))), False

    @staticmethod
    def tier(one_in: int) -> tuple[str, str]:
        for threshold, tier_id, label in TIERS:
            if one_in >= threshold:
                return tier_id, label
        return TIERS[-1][1], TIERS[-1][2]

    # --- explanations ------------------------------------------------------------------

    def _pool_for(self, a: int, b: int) -> str:
        return str(self.types[a]) if self.types[a] == self.types[b] else features.CROSS

    def _reasons(self, pool: str, a: int, b: int, bits_row: np.ndarray) -> list[dict]:
        spec = self.scorer.pools[pool]
        contrib = self.scorer.contributions(pool, bits_row)
        case_a = self.cases.get(self.case_ids[a], {})
        case_b = self.cases.get(self.case_ids[b], {})
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
        order = {"timing": 0, "shared": 1, "different": 2, "not_recorded": 3}
        return sorted(out, key=lambda r: (order[r["kind"]], -r["bits"] if r["kind"] == "shared" else r["bits"]))

    def _summary(self, idx: int) -> dict:
        c = self.cases.get(self.case_ids[idx], {})
        return {k: c.get(k) for k in ("case_id", "fir_no", "crime_type", "state_code", "district",
                                     "police_station", "registered_at", "occurred_from")}

    def _strength(self, pool: str, bits: float) -> dict:
        n, rarer = self.one_in(pool, bits)
        tier_id, label = self.tier(n)
        return {"bits": round(float(bits), 2), "one_in": n, "rarer_than_measured": rarer,
                "tier": tier_id, "tier_label": label}

    def _truth(self, a: int, b: int) -> dict:
        if self.truth_groups is None:
            return {}
        ga, gb = self.truth_groups.get(self.case_ids[a]), self.truth_groups.get(self.case_ids[b])
        return {"ground_truth_link": bool(ga is not None and ga == gb)}

    # --- API surface -------------------------------------------------------------------------

    def shortlist(self, case_id: str, scope: str = "same", limit: int = 10) -> dict:
        if case_id not in self.index:
            raise KeyError(case_id)
        if scope not in ("same", "all"):
            raise ValueError("scope must be 'same' or 'all'")
        q = self.index[case_id]
        crime_type = str(self.types[q])
        lists = {SAME: self._rank(q, crime_type, self.types == crime_type, limit)}
        if scope == "all":
            lists[CROSS] = self._rank(q, features.CROSS, self.types != crime_type, limit)
        return {"case": self._summary(q), "scope": scope, "lists": lists}

    def _rank(self, q: int, pool: str, candidate_mask: np.ndarray, limit: int) -> dict:
        cand = np.flatnonzero(candidate_mask)
        cand = cand[cand != q]
        codes = self.codes[pool]
        bits = self.scorer.field_bits(pool, {k: v[q] for k, v in codes.items()}, {k: v[cand] for k, v in codes.items()})
        evidence = self.scorer.evidence(pool, bits)
        take = min(limit, len(cand))
        top = np.argpartition(-evidence, take - 1)[:take] if take < len(cand) else np.arange(len(cand))
        top = top[np.lexsort((cand[top], -evidence[top]))]
        items = []
        for rank, j in enumerate(top, 1):
            c = int(cand[j])
            reasons = self._reasons(pool, q, c, bits[j])
            items.append({"rank": rank, **self._summary(c), **self._strength(pool, evidence[j]),
                          "cross_state": self.cases.get(self.case_ids[c], {}).get("state_code")
                          != self.cases.get(self.case_ids[q], {}).get("state_code"),
                          "reasons": reasons, **self._truth(q, c)})
        context = self.meta["pools"].get(pool, {})
        return {"pool": pool, "pool_size": int(len(cand)), "items": items,
                "typical_true_link_bits": context.get("typical_true_link_bits"),
                "breakeven_bits": context.get("breakeven_bits")}

    def pair(self, case_a: str, case_b: str) -> dict:
        """Everything the side-by-side comparison needs for one pair."""
        if case_a not in self.index or case_b not in self.index:
            raise KeyError(case_a if case_a not in self.index else case_b)
        a, b = self.index[case_a], self.index[case_b]
        if a == b:
            raise ValueError("a case cannot be compared with itself")
        pool = self._pool_for(a, b)
        codes = self.codes[pool]
        bits = self.scorer.field_bits(pool, {k: v[a] for k, v in codes.items()}, {k: v[b] for k, v in codes.items()})
        evidence = float(self.scorer.evidence(pool, bits))
        ca, cb = self.cases.get(case_a, {}), self.cases.get(case_b, {})
        return {"pool": pool, "same_type": pool != features.CROSS,
                "case_a": ca, "case_b": cb, **self._strength(pool, evidence),
                "cross_state": ca.get("state_code") != cb.get("state_code"),
                "days_apart": _days_apart(ca.get("occurred_from"), cb.get("occurred_from")),
                "reasons": self._reasons(pool, a, b, bits),
                "breakeven_bits": self.meta["pools"].get(pool, {}).get("breakeven_bits"),
                **self._truth(a, b)}


def _days_apart(x: str | None, y: str | None) -> int | None:
    try:
        return abs((datetime.fromisoformat(x) - datetime.fromisoformat(y)).days)
    except (TypeError, ValueError):
        return None
