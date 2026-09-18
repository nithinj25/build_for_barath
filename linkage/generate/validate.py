"""The six validators from TASK_DATA_GENERATION.md. Each returns a dict with
`passed`, the numbers behind it, and `criterion` — the pass rule, fixed before
any corpus was generated, so nobody tunes the rule to the result.
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from linkage import schema
from linkage.generate.corrupt import _adjacency
from linkage.schema import ABSENT, MISSING, TOKENS
from linkage.generate.sample import IMPLIED

Z_MAX = 4.0


def _z(observed: float, expected: float, n: int) -> float:
    if expected in (0.0, 1.0):
        return 0.0 if observed == expected else float("inf")
    return abs(observed - expected) / np.sqrt(expected * (1 - expected) / n)


# 1 -------------------------------------------------------------------------

def background_marginals(truth: pd.DataFrame, cfg: dict) -> dict:
    bg = truth[~truth["is_serial"]]
    m = cfg["marginals"]
    worst, failures, checks = 0.0, [], 0
    share = bg["crime_type"].value_counts(normalize=True)
    for ct in schema.CRIME_TYPES:
        z = _z(share.get(ct, 0.0), m["crime_types"]["share"][ct], len(bg))
        checks += 1
        worst = max(worst, z)
        if z > Z_MAX:
            failures.append(f"crime_type share {ct}: z={z:.1f}")
    for ct in schema.CRIME_TYPES:
        sub = bg[bg["crime_type"] == ct]
        for f in schema.mo_fields(ct):
            if (ct, f) in IMPLIED or len(sub) == 0:
                continue
            spec = m["by_crime_type"][ct][f]
            if f in schema.TAG_FIELDS:
                tags = sub[f].fillna("").str.split(";")
                observed = {v: tags.apply(lambda xs, v=v: v in xs).mean() for v in spec["rate"]}
                expected = spec["rate"]
            else:
                freq = sub[f].value_counts(normalize=True)
                observed = {v: freq.get(v, 0.0) for v in spec["p"]}
                expected = spec["p"]
            for v, p in expected.items():
                z = _z(observed[v], p, len(sub))
                checks += 1
                if np.isfinite(z):
                    worst = max(worst, z)
                if z > Z_MAX:
                    failures.append(f"{ct}.{f}.{v}: config {p}, background {observed[v]:.4f}, z={z:.1f}")
    return {"passed": not failures, "checks": checks, "worst_abs_z": round(worst, 2),
            "failures": failures, "skipped_implied_fields": sorted(f"{a}.{b}" for a, b in IMPLIED),
            "criterion": f"every background value within |z| <= {Z_MAX} of marginals.yaml"}


# 2, 3 --------------------------------------------------------------------------

def _mi_bits(a: np.ndarray, b: np.ndarray, na: int, nb: int) -> float:
    joint = np.bincount(a * nb + b, minlength=na * nb).reshape(na, nb) / len(a)
    outer = np.outer(joint.sum(1), joint.sum(0))
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / outer[nz])).sum())


def mi_statistic(frame: pd.DataFrame, prefix: str, rng: np.random.Generator, n_perm: int) -> dict:
    """Summed pairwise MI across categorical MO fields, within each true crime
    type, against a column-permutation null. Implied fields are excluded: they
    carry deterministic within-crime structure, not style correlation."""
    observed, null = 0.0, np.zeros(n_perm)
    for ct in schema.CRIME_TYPES:
        sub = frame[frame["crime_type"] == ct]
        if len(sub) < 100:
            continue
        fields = [f for f in schema.mo_fields(ct)
                  if f not in schema.TAG_FIELDS and (ct, f) not in IMPLIED]
        cols = {}
        for f in fields:
            s = sub[f"{prefix}{f}"]
            cols[f] = s.where(~s.isin(TOKENS))
        for f, g in combinations(fields, 2):
            mask = cols[f].notna() & cols[g].notna()
            if mask.sum() < 50:
                continue
            a, ua = pd.factorize(cols[f][mask])
            b, ub = pd.factorize(cols[g][mask])
            observed += _mi_bits(a, b, len(ua), len(ub))
            for k in range(n_perm):
                null[k] += _mi_bits(a, rng.permutation(b), len(ua), len(ub))
    return {"observed_bits": observed, "null_mean_bits": float(null.mean()),
            "null_max_bits": float(null.max()), "excess_bits": observed - float(null.mean())}


def _one_per_offender(frame: pd.DataFrame) -> pd.DataFrame:
    """First crime of each type per offender. A series shares one θ, so its
    crimes are not independent rows: with concentrated θ, chance
    co-occurrence inside a series looks like cross-field correlation, and a
    row-level permutation null cannot see that. One row per offender restores
    independence (verified: at τ = 0 the pooled version reported ~1.5 bits of
    'excess' MI that cannot exist)."""
    return frame.sort_values(["offender_id", "series_index"]).groupby(["offender_id", "crime_type"]).head(1)


def truth_mi(truth: pd.DataFrame, rng: np.random.Generator, n_perm: int = 50) -> dict:
    stat = mi_statistic(_one_per_offender(truth[truth["is_serial"]]), "", rng, n_perm)
    return {"passed": stat["observed_bits"] > stat["null_max_bits"],
            **{k: round(v, 4) for k, v in stat.items()},
            "criterion": f"summed cross-field MI in serial truth (one crime per offender per type) "
                         f"exceeds all {n_perm} permutation nulls"}


def recorded_mi(cases: pd.DataFrame, truth_result: dict, rng: np.random.Generator, n_perm: int = 50) -> dict:
    sub = _one_per_offender(cases[cases["is_serial"] & cases["ingested"]])
    stat = mi_statistic(sub, "rec_", rng, n_perm)
    kept = stat["excess_bits"] / truth_result["excess_bits"] if truth_result["excess_bits"] > 0 else 0.0
    return {"passed": stat["observed_bits"] > stat["null_max_bits"] and kept >= 0.5,
            **{k: round(v, 4) for k, v in stat.items()},
            "excess_kept_vs_truth": round(kept, 3),
            "criterion": f"recorded MI exceeds all {n_perm} permutation nulls and keeps >= 50% of truth's excess"}


# 4 ------------------------------------------------------------------------------

def cheat_detector(cases: pd.DataFrame, positions: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Can metadata that should carry no offender signal predict same-offender
    pairs beyond what elapsed time legitimately explains?

    Row position and FIR serial are both time-ordered, and crimes in a series
    really are close in time, so on their own they predict links. The test is
    the AUC they ADD on top of the time gap."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ing = cases[cases["ingested"]].merge(positions, on="case_uid").reset_index(drop=True)
    ing["row_frac"] = ing["feed_row"] / ing["feed_rows"].clip(lower=1)
    ing["hash_frac"] = ing["case_id"].apply(lambda h: int(h, 16) / 16 ** 16)
    ing["fir_serial"] = ing["fir_no"].str.split("/").str[1].astype(int)
    ing["reg_days"] = (ing["registered_at"] - ing["registered_at"].min()).dt.total_seconds() / 86400

    pos = []
    for _, idx in ing[ing["is_serial"]].groupby("offender_id").groups.items():
        pos.extend(combinations(list(idx), 2))
    pos = np.array(pos[:20000])
    if len(pos) < 100:
        return {"passed": False, "reason": "too few same-offender pairs to test"}
    neg = rng.integers(0, len(ing), size=(len(pos) * 2, 2))
    off = ing["offender_id"].to_numpy()
    neg = neg[(neg[:, 0] != neg[:, 1]) & (off[neg[:, 0]] != off[neg[:, 1]])][: len(pos)]
    pairs = np.vstack([pos, neg])
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]

    def gap(col):
        v = ing[col].to_numpy(float)
        return np.abs(v[pairs[:, 0]] - v[pairs[:, 1]])

    time = np.log1p(gap("reg_days"))[:, None]
    same_station = (ing["police_station"].to_numpy()[pairs[:, 0]] == ing["police_station"].to_numpy()[pairs[:, 1]])
    meta = np.c_[gap("row_frac"), gap("hash_frac"), np.where(same_station, np.log1p(gap("fir_serial")), -1)]

    def auc(X):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        return float(cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean())

    auc_time, auc_both, auc_meta = auc(time), auc(np.c_[time, meta]), auc(meta)
    added = auc_both - auc_time
    return {"passed": added < 0.01, "auc_time_only": round(auc_time, 4),
            "auc_time_plus_metadata": round(auc_both, 4), "auc_metadata_only": round(auc_meta, 4),
            "metadata_added_auc": round(added, 4), "pairs": int(len(pairs)),
            "not_run": "state predictability after normalisation: no normaliser exists yet",
            "criterion": "row position, case_id hash and FIR serial add < 0.01 AUC over the time gap alone"}


# 5 ----------------------------------------------------------------------------------

def series_tail(offenders: pd.DataFrame, cfg: dict) -> dict:
    extra = offenders["series_length_drawn"] - cfg["corpus"]["series_length"]["minimum"]
    dispersion = float(extra.var() / extra.mean()) if extra.mean() > 0 else 0.0
    pairs = offenders["series_length"].apply(lambda n: comb(int(n), 2)).sort_values(ascending=False)
    top = pairs.iloc[: max(len(pairs) // 10, 1)].sum() / max(pairs.sum(), 1)
    return {"passed": dispersion > 1.5, "dispersion_index": round(dispersion, 2),
            "mean_length": round(float(offenders["series_length_drawn"].mean()), 2),
            "max_length": int(offenders["series_length_drawn"].max()),
            "top10pct_offenders_share_of_true_pairs": round(float(top), 3),
            "truncated_by_date_range": int((offenders["series_length"] < offenders["series_length_drawn"]).sum()),
            "criterion": "var/mean of (length - minimum) > 1.5 (Poisson gives 1)"}


# 6 ------------------------------------------------------------------------------------

def recording_rates(cases: pd.DataFrame, cfg: dict) -> dict:
    failures, checks = [], 0
    graph = {f: _adjacency(e) for f, e in cfg["states"]["confusability"].items()}
    per_state = {}
    for code, st in cfg["states"]["states"].items():
        rec = st["recording"]
        sub = cases[cases["state_code"] == code]
        absent = set(rec["structurally_absent"] or [])
        summary = {}

        changed = sub["rec_crime_type"] != sub["crime_type"]
        eligible = sub["crime_type"].map(lambda ct: bool(graph["crime_type"].get(ct)))
        n = int(eligible.sum())
        rate = float(changed[eligible].mean())
        z = _z(rate, rec["crime_type_confusion"], n)
        checks += 1
        summary["crime_type_confusion"] = {"config": rec["crime_type_confusion"], "realised": round(rate, 4)}
        if z > Z_MAX:
            failures.append(f"{code} crime_type_confusion {rate:.4f} vs {rec['crime_type_confusion']}, z={z:.1f}")

        flips, eligible_n = 0, 0
        for f in schema.ALL_MO_FIELDS:
            col = sub[f"rec_{f}"]
            applicable = col.notna()
            if f in absent:
                if not (col[applicable] == ABSENT).all():
                    failures.append(f"{code}.{f}: structurally absent but has values")
                continue
            if (col == ABSENT).any():
                failures.append(f"{code}.{f}: ABSENT where the column exists")
            expected = (rec["dropout"]["fields"] or {}).get(f, rec["dropout"]["default"])
            n = int(applicable.sum())
            if n >= 50:
                rate = float((col[applicable] == MISSING).mean())
                z = _z(rate, expected, n)
                checks += 1
                if z > Z_MAX:
                    failures.append(f"{code}.{f} dropout {rate:.4f} vs {expected}, z={z:.1f}")
            if f in schema.TAG_FIELDS or f in schema.DERIVED_AT_NORMALISATION:
                continue
            kept = applicable & (col != MISSING)
            has_nb = sub.loc[kept, f].map(lambda v, f=f: bool(graph[f].get(v)))
            idx = has_nb[has_nb].index
            eligible_n += len(idx)
            flips += int((sub.loc[idx, f"rec_{f}"] != sub.loc[idx, f]).sum())
        if eligible_n:
            rate = flips / eligible_n
            summary["value_confusion"] = {"config": rec["value_confusion"], "realised_upper_bound": round(rate, 4)}
            # A flip can only land on a neighbour valid for the crime type, so
            # realised <= config. Fail only if it exceeds config.
            z = (rate - rec["value_confusion"]) / np.sqrt(rec["value_confusion"] * (1 - rec["value_confusion"]) / eligible_n)
            checks += 1
            if z > Z_MAX:
                failures.append(f"{code} value_confusion {rate:.4f} above config {rec['value_confusion']}")
        per_state[code] = summary
    return {"passed": not failures, "checks": checks, "failures": failures, "per_state": per_state,
            "criterion": f"dropout and crime-type confusion within |z| <= {Z_MAX} of states.yaml; "
                         "value confusion not above config; structural absence exact"}
