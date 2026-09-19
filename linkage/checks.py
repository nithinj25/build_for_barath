"""Crime-type check for one FIR: does its MO read like another type in the same
family? Pure Python + the exported weights; the bundle runs it over the corpus
and the API runs it on a newly entered FIR. FINDINGS §13.
"""
from __future__ import annotations

import math

from linkage import schema

MIN_BITS = 2.0


def type_check(pools: dict, record: dict, own: str, min_bits: float = MIN_BITS) -> dict | None:
    """Sum over the fields both pools score of log2(frequency of the recorded value
    under the other type / under the recorded type). None unless some other type
    in the family beats the recorded one by min_bits. Uses no labels."""
    best = None
    for other in schema.CRIME_TYPES:
        if other == own or schema.FAMILY[other] != schema.FAMILY.get(own) or own not in pools or other not in pools:
            continue
        po, pt = pools[own], pools[other]
        terms = []
        for f in po["fields"]:
            if f not in pt["fields"] or f in schema.TAG_FIELDS:
                continue
            v = record.get(f)
            if v is None or v in schema.TOKENS or v not in po["vocab"][f] or v not in pt["vocab"][f]:
                continue
            uo = max(po["u"][f][po["vocab"][f].index(v)], 1e-4)
            ut = max(pt["u"][f][pt["vocab"][f].index(v)], 1e-4)
            terms.append((math.log2(ut / uo), f, v, uo, ut))
        bits = sum(t[0] for t in terms)
        if bits >= min_bits and (best is None or bits > best["bits"]):
            terms.sort(reverse=True)
            best = {"recorded": own, "likely": other, "bits": round(bits, 2),
                    "reasons": [{"field": f, "value": v, "share_recorded": round(uo, 3), "share_likely": round(ut, 3)}
                                for b, f, v, uo, ut in terms[:3] if b > 0]}
    return best
