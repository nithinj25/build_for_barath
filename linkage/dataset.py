"""Local training / evaluation data: normalised records joined to ground truth.

Reads parquet; local only, never shipped to Lambda. Splits are by OFFENDER,
never by case (CLAUDE.md rule 6): one offender's crimes all land in the same
split, deterministically from a seed.
"""
from __future__ import annotations

import hashlib
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from linkage import features, schema

SPLIT_EDGES = (("train", 0.60), ("calib", 0.75))      # remainder: test (25%)
MO_NON_DERIVED = tuple(f for f in schema.ALL_MO_FIELDS if f not in schema.DERIVED_AT_NORMALISATION)


def load(normalised_path, truth_path, oracle_extraction: bool = False) -> pd.DataFrame:
    """Normalised records plus offender labels.

    oracle_extraction fills free-text states' pending MO fields with the values
    their text was rendered from — an upper bound on what LLM extraction could
    recover, for use until the enrich step exists. Never report it as a result
    of the pipeline.
    """
    norm = pd.read_parquet(normalised_path)
    cols = ["case_id", "offender_id", "is_serial"] + ([f"rec_{f}" for f in MO_NON_DERIVED] if oracle_extraction else [])
    df = norm.merge(pd.read_parquet(truth_path, columns=cols), on="case_id", how="left", validate="one_to_one")
    if df["offender_id"].isna().any():
        raise ValueError(f"{int(df['offender_id'].isna().sum())} normalised records have no ground truth")
    df["needs_extraction"] = df["needs_extraction"].astype(bool)
    if oracle_extraction:
        pending = df["needs_extraction"]
        for f in MO_NON_DERIVED:
            df.loc[pending, f] = df.loc[pending, f"rec_{f}"]
        df = df.drop(columns=[f"rec_{f}" for f in MO_NON_DERIVED])
    return df[df["crime_type"].notna()].reset_index(drop=True)


def split_of(offender_ids: pd.Series, seed: int) -> np.ndarray:
    h = offender_ids.map(lambda o: int(hashlib.sha1(f"{seed}:{o}".encode()).hexdigest()[:8], 16) / 16 ** 8)
    return np.select([h < SPLIT_EDGES[0][1], h < SPLIT_EDGES[1][1]],
                     [SPLIT_EDGES[0][0], SPLIT_EDGES[1][0]], "test")


def pool_rows(df: pd.DataFrame, pool: str) -> np.ndarray:
    return np.arange(len(df)) if pool == features.CROSS else np.flatnonzero(df["crime_type"].to_numpy() == pool)


def pool_pair_count(df: pd.DataFrame, pool: str) -> int:
    counts = df["crime_type"].value_counts()
    if pool == features.CROSS:
        return comb(len(df), 2) - sum(comb(int(n), 2) for n in counts)
    return comb(int(counts.get(pool, 0)), 2)


def positive_pairs(df: pd.DataFrame, pool: str, offender_mask: np.ndarray) -> np.ndarray:
    """(n, 2) row indices of same-offender pairs in the pool class."""
    types = df["crime_type"].to_numpy()
    serial = df.index[df["is_serial"].to_numpy() & offender_mask]
    pairs = []
    for _, idx in df.loc[serial].groupby("offender_id").groups.items():
        for i, j in combinations(idx, 2):
            if pool == features.CROSS:
                if types[i] != types[j]:
                    pairs.append((i, j))
            elif types[i] == pool and types[j] == pool:
                pairs.append((i, j))
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def negative_pairs(df: pd.DataFrame, pool: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, 2) random pairs in the pool class from different offenders."""
    rows = pool_rows(df, pool)
    types, offenders = df["crime_type"].to_numpy(), df["offender_id"].to_numpy()
    out = np.empty((0, 2), dtype=np.int64)
    while len(out) < n:
        cand = rows[rng.integers(0, len(rows), size=(2 * n, 2))]
        ok = (cand[:, 0] != cand[:, 1]) & (offenders[cand[:, 0]] != offenders[cand[:, 1]])
        if pool == features.CROSS:
            ok &= types[cand[:, 0]] != types[cand[:, 1]]
        out = np.vstack([out, cand[ok]])
    return out[:n]


def vocab_for(df: pd.DataFrame, field: str) -> list[str]:
    if field in schema.MO_CORE_VOCAB:
        return list(schema.MO_CORE_VOCAB[field])
    values = df[field].dropna()
    return sorted(v for v in values.unique() if v not in schema.TOKENS)
