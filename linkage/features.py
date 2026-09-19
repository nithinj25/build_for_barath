"""Fellegi-Sunter evidence per MO field, in bits. Pure numpy; ships to Lambda.

Pools: a same-type pool is one crime type and scores that type's full field
set. The CROSS pool pairs cases of different types and scores mo_core only,
with pooled frequencies (TECHNICAL_SPEC.md §4.4–4.5).

    u_v   frequency of value v in the pool: how often an unrelated case agrees
    m_v   (alpha·u_v + 1) / (alpha + 1): how often the same offender repeats v
          — the same expression as (1 − u)/(alpha + 1) + u, used for tags too
    alpha one concentration per field per pool, fitted to the agreement rate
          of labelled same-offender pairs. Rare values then carry more weight
          than common ones automatically; rarity weighting is the signal.

Agreement weighs log2(m/u). Disagreement weighs log2((1−m)/(1−u)), which
under this m reduces to log2(alpha/(alpha+1)) for every value — so the score
is symmetric in A and B without choosing whose value to use (the table still
averages both sides, which matters only where m is clipped). Anything unknown
on either side — missing, absent, unknowable, pending extraction — gives 0.
"""
from __future__ import annotations

import numpy as np

from linkage import schema

CROSS = "CROSS_TYPE"
ALPHA_BOUNDS = (1e-3, 1e4)

# Time between offences — evidence from outside the MO (spec §11). Serial
# offending is bursty, so a short gap raises the odds of one offender and a
# gap of years lowers them. Weights per bin are learned from training pairs;
# an unknown date contributes 0. Codes are day numbers (-1 = unknown) under
# this key, alongside the MO codes.
DAYS = "__days"
GAP_EDGES = (0, 30, 90, 180, 365, 730)            # bin i covers [edge i, edge i+1) days


def pools() -> tuple[str, ...]:
    return (*schema.CRIME_TYPES, CROSS)


def pool_fields(pool: str) -> tuple[str, ...]:
    return schema.MO_CORE if pool == CROSS else schema.mo_fields(pool)


def is_tag(field: str) -> bool:
    return field in schema.TAG_FIELDS


def unknown_code(field: str, vocab: list[str]) -> int:
    return 2 ** len(vocab) if is_tag(field) else len(vocab)


def encode(values: list, field: str, vocab: list[str]) -> np.ndarray:
    """Categorical → index into vocab. Tags → bitmask over vocab ("" is the
    known empty set, mask 0). Unknown → unknown_code."""
    unknown = unknown_code(field, vocab)
    out = np.full(len(values), unknown, dtype=np.int64)
    if is_tag(field):
        bit = {v: 1 << i for i, v in enumerate(vocab)}
        for i, v in enumerate(values):
            if isinstance(v, str) and v not in schema.TOKENS:
                out[i] = sum(bit[t] for t in v.split(";") if t in bit)
        return out
    index = {v: i for i, v in enumerate(vocab)}
    for i, v in enumerate(values):
        if isinstance(v, str):
            out[i] = index.get(v, unknown)
    return out


def _bits(masks: np.ndarray, k: int) -> np.ndarray:
    return (masks[:, None] >> np.arange(k)) & 1


def fit_u(codes: np.ndarray, field: str, vocab: list[str]) -> np.ndarray:
    """Smoothed value (or per-tag) frequency among known values in the pool."""
    k = len(vocab)
    if is_tag(field):
        known = codes[codes < 2 ** k]
        return (_bits(known, k).sum(0) + 0.5) / (len(known) + 1.0)
    known = codes[codes < k]
    return (np.bincount(known, minlength=k) + 0.5) / (len(known) + 0.5 * k)


def fit_alpha(a: np.ndarray, b: np.ndarray, u: np.ndarray, field: str, vocab: list[str]) -> float:
    """Closed-form alpha matching the observed agreement of same-offender pairs
    (codes a[i], b[i]), weighting each value by how often it occurs in them."""
    lo, hi = ALPHA_BOUNDS
    k = len(vocab)
    if is_tag(field):
        known = (a < 2 ** k) & (b < 2 ** k)
        A, B = _bits(a[known], k), _bits(b[known], k)
        n_t = A.sum(0) + B.sum(0)                 # tag present on one side, both directions
        k_t = 2 * (A & B).sum(0)                  # ... and repeated on the other
        n = n_t.sum()
        if n == 0:
            return hi
        rate = k_t.sum() / n
        chance = (n_t * u).sum() / n              # R(alpha) = chance + spread / (alpha + 1)
        spread = (n_t * (1 - u)).sum() / n
        return float(np.clip(spread / (rate - chance) - 1, lo, hi)) if rate > chance else hi
    known = (a < k) & (b < k)
    if not known.any():
        return hi
    a, b = a[known], b[known]
    rate = float((a == b).mean())
    f = np.bincount(np.r_[a, b], minlength=k) / (2 * len(a))
    chance = float((f * u).sum())                 # A(alpha) = (alpha·chance + 1) / (alpha + 1)
    return float(np.clip((1 - rate) / (rate - chance), lo, hi)) if rate > chance else hi


def m_of(u: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip((1 - u) / (alpha + 1) + u, u, 1 - 1e-6)


def weight_table(u: np.ndarray, alpha: float, field: str) -> np.ndarray:
    """Bits for every (code_a, code_b), unknown row and column included (zeros)."""
    m = m_of(u, alpha)
    agree = np.log2(m / u)
    disagree = np.log2((1 - m) / (1 - u))
    k = len(u)
    if is_tag(field):
        bits = _bits(np.arange(2 ** k), k)
        both = bits[:, None, :] & bits[None, :, :]
        one_side = bits[:, None, :] ^ bits[None, :, :]
        table = both @ agree + one_side @ disagree
    else:
        table = 0.5 * (disagree[:, None] + disagree[None, :])
        np.fill_diagonal(table, agree)
    return np.pad(table, ((0, 1), (0, 1)))


def gap_bins(a_days, b_days) -> np.ndarray:
    """Bin index of |days apart|, or -1 where either date is unknown."""
    a, b = np.asarray(a_days), np.asarray(b_days)
    bins = np.searchsorted(GAP_EDGES, np.abs(a - b), side="right") - 1
    return np.where((a < 0) | (b < 0), -1, bins)


def fit_gap_bits(same_bins: np.ndarray, unrelated_bins: np.ndarray) -> np.ndarray:
    """log2 P(bin | same offender) / P(bin | unrelated), smoothed."""
    k = len(GAP_EDGES)
    same = np.bincount(same_bins[same_bins >= 0], minlength=k) + 0.5
    unrel = np.bincount(unrelated_bins[unrelated_bins >= 0], minlength=k) + 0.5
    return np.log2((same / same.sum()) / (unrel / unrel.sum()))


def field_bits(tables: dict, fields: tuple[str, ...], a_codes: dict, b_codes: dict,
               gap_bits: np.ndarray | None = None) -> np.ndarray:
    """(n_pairs, n_fields [+1 for time]) evidence. Codes broadcast: a scalar query
    against many. With gap_bits, a final column holds the time evidence (0 when
    either date is unknown or no day codes were given)."""
    cols = [tables[f][a_codes[f], b_codes[f]] for f in fields]
    if gap_bits is not None:
        if DAYS in a_codes and DAYS in b_codes:
            bins = gap_bins(a_codes[DAYS], b_codes[DAYS])
            cols.append(np.where(bins >= 0, np.asarray(gap_bits)[np.clip(bins, 0, None)], 0.0))
        else:
            cols.append(np.zeros(np.broadcast(cols[0]).shape))
    return np.stack(cols, axis=-1)


def agreement_count(fields: tuple[str, ...], vocab: dict, a_codes: dict, b_codes: dict) -> np.ndarray:
    """Naive baseline: number of fields both sides know and share (any common tag)."""
    total = 0
    for f in fields:
        unknown = unknown_code(f, vocab[f])
        a, b = a_codes[f], b_codes[f]
        known = (a != unknown) & (b != unknown)
        same = (a & b) != 0 if is_tag(f) else a == b
        total = total + (known & same)
    return np.asarray(total, dtype=float)
