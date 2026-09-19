"""Inference from weights.json. Pure numpy; this is what ships to Lambda.

    evidence_bits = Σ_f coef_f · bits_f / ln 2

The logistic-regression coefficients correct Fellegi-Sunter for correlated
fields; dividing by ln 2 keeps the result in bits, so a field whose FS weight
needed no correction (coef = ln 2) contributes exactly its FS bits. Evidence
is additive: the per-field contributions sum to the total the UI shows.

link_probability applies isotonic calibration for internal thresholds
(clustering) only. Never render it (CLAUDE.md rule 4).
"""
from __future__ import annotations

import numpy as np

from linkage import features

LN2 = float(np.log(2))


class Scorer:
    def __init__(self, weights: dict):
        self.meta = {k: v for k, v in weights.items() if k != "pools"}
        self.pools = {}
        for pool, spec in weights["pools"].items():
            fields = tuple(spec["fields"])
            self.pools[pool] = {
                "fields": fields,
                "vocab": spec["vocab"],
                "u": {f: np.asarray(spec["u"][f], float) for f in fields},   # how common each value is
                "tables": {f: features.weight_table(np.asarray(spec["u"][f]), spec["alpha"][f], f)
                           for f in fields},
                "gap_bits": np.asarray(spec["time"]["bits"], float) if spec.get("time") else None,
                "coef": np.asarray(spec["lr"]["coef"], float),
                "intercept": float(spec["lr"]["intercept"]),
                "iso_x": np.asarray(spec["isotonic"]["x"], float),
                "iso_y": np.asarray(spec["isotonic"]["y"], float),
            }

    def encode(self, pool: str, columns: dict) -> dict:
        """columns: field → list of canonical values (one per case); optionally
        features.DAYS → day numbers (-1 unknown) for the time evidence."""
        p = self.pools[pool]
        codes = {f: features.encode(columns[f], f, p["vocab"][f]) for f in p["fields"]}
        if features.DAYS in columns:
            codes[features.DAYS] = np.asarray(columns[features.DAYS], dtype=np.int64)
        return codes

    def field_bits(self, pool: str, a_codes: dict, b_codes: dict) -> np.ndarray:
        p = self.pools[pool]
        return features.field_bits(p["tables"], p["fields"], a_codes, b_codes, p["gap_bits"])

    def evidence_names(self, pool: str) -> list[str]:
        """Column names of field_bits output: MO fields, then days_apart if timed."""
        p = self.pools[pool]
        return [*p["fields"], *(["days_apart"] if p["gap_bits"] is not None else [])]

    def contributions(self, pool: str, field_bits: np.ndarray) -> np.ndarray:
        return field_bits * self.pools[pool]["coef"] / LN2

    def evidence(self, pool: str, field_bits: np.ndarray) -> np.ndarray:
        return self.contributions(pool, field_bits).sum(axis=-1)

    def link_probability(self, pool: str, field_bits: np.ndarray) -> np.ndarray:
        p = self.pools[pool]
        logit = p["intercept"] + field_bits @ p["coef"]
        return np.interp(logit, p["iso_x"], p["iso_y"])

    def explain(self, pool: str, field_bits_row: np.ndarray, top: int = 3) -> list[tuple[str, float]]:
        """Driving fields for one pair, strongest first: [(field, bits), ...]."""
        contrib = self.contributions(pool, field_bits_row)
        names = self.evidence_names(pool)
        order = np.argsort(-np.abs(contrib))[:top]
        return [(names[i], round(float(contrib[i]), 3)) for i in order if contrib[i] != 0]
