"""Shortlists on request. Pure numpy + stdlib; this is what the API Lambda runs.

Scoring one case against its whole pool takes 3–10 ms (measured), so the API
ranks on request instead of storing millions of precomputed links. A
shortlist is:

    rank r of N · +b bits · driving fields

— never a probability (CLAUDE.md rule 4). Same-type and cross-type are
separate lists with separate context, never one blended ranking: their bits
come from different models against different priors.
"""
from __future__ import annotations

import numpy as np

from linkage import features
from linkage.score import Scorer

SAME, CROSS = "same_type", "cross_type"


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
        return {"case_id": case_id, "crime_type": crime_type, "scope": scope, "lists": lists}

    def _rank(self, q: int, pool: str, candidate_mask: np.ndarray, limit: int) -> dict:
        spec = self.scorer.pools[pool]
        fields = spec["fields"]
        cand = np.flatnonzero(candidate_mask)
        cand = cand[cand != q]
        codes = self.codes[pool]
        bits = self.scorer.field_bits(pool, {f: codes[f][q] for f in fields}, {f: codes[f][cand] for f in fields})
        evidence = self.scorer.evidence(pool, bits)
        take = min(limit, len(cand))
        top = np.argpartition(-evidence, take - 1)[:take] if take < len(cand) else np.arange(len(cand))
        top = top[np.lexsort((cand[top], -evidence[top]))]       # best first, ties by corpus order
        query_case = self.cases.get(self.case_ids[q], {})
        items = []
        for rank, j in enumerate(top, 1):
            other = self.case_ids[cand[j]]
            contrib = self.scorer.contributions(pool, bits[j])
            order = np.argsort(-np.abs(contrib))
            other_case = self.cases.get(other, {})
            item = {
                "case_id": other, "rank": rank, "bits": round(float(evidence[j]), 2),
                "crime_type": other_case.get("crime_type"), "state_code": other_case.get("state_code"),
                "district": other_case.get("district"), "registered_at": other_case.get("registered_at"),
                "contributions": [
                    {"field": fields[i], "bits": round(float(contrib[i]), 3),
                     "value": query_case.get(fields[i]), "other_value": other_case.get(fields[i])}
                    for i in order if abs(contrib[i]) >= 0.005],
            }
            if self.truth_groups is not None:
                qg, og = self.truth_groups.get(self.case_ids[q]), self.truth_groups.get(other)
                item["ground_truth_link"] = bool(qg is not None and qg == og)
            items.append(item)
        context = self.meta["pools"].get(pool, {})
        return {"pool": pool, "pool_size": int(len(cand)), "items": items,
                "typical_true_link_bits": context.get("typical_true_link_bits"),
                "breakeven_bits": context.get("breakeven_bits")}
