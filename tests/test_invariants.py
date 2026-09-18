"""Invariants that must survive deployment work: pytest -q

Each test pins a property something else in the project relies on, and each
one has failed at least once during the build.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

from linkage import config as config_mod
from linkage import features, normalise, schema
from linkage.generate import sample
from linkage.score import Scorer

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "results/post_core_fix/weights.json"
ALPHA = 7.05                      # the spec's worked example


# --- schema -------------------------------------------------------------------

def test_time_bands_cover_the_clock_exactly_once():
    hours = [h for band in schema.TIME_BAND_HOURS for h in schema.band_hours(band)]
    assert sorted(hours) == list(range(24))


def test_wide_window_is_unknowable_not_a_band():
    narrow = schema.recorded_time_band(datetime(2024, 3, 9, 23, 0), datetime(2024, 3, 10, 1, 0))
    wide = schema.recorded_time_band(datetime(2024, 3, 9, 8, 0), datetime(2024, 3, 12, 8, 0))
    assert narrow == "night" and wide == schema.UNKNOWABLE


# --- Fellegi-Sunter weights ---------------------------------------------------

@pytest.mark.parametrize("field, u, index, expected", [
    ("entry_point", [0.55, 0.25, 0.10, 0.06, 0.04], 3, 1.559),        # roof
    ("counter_forensic", [0.80, 0.10, 0.07, 0.03], 3, 2.327),         # cctv_disabled
    ("tools", [0.30, 0.16, 0.35, 0.02], 1, 0.724),                    # cutter (tag; vocab index 1)
])
def test_agreement_weights_match_the_spec_worked_example(field, u, index, expected):
    table = features.weight_table(np.array(u), ALPHA, field)
    code = 1 << index if features.is_tag(field) else index
    assert table[code, code] == pytest.approx(expected, abs=5e-4)


def test_scoring_is_symmetric_and_unknowns_are_free():
    table = features.weight_table(np.array([0.5, 0.3, 0.2]), ALPHA, "premise")
    assert np.allclose(table, table.T)
    assert not table[-1].any() and not table[:, -1].any()      # the unknown row/column


def test_fit_alpha_recovers_a_known_concentration():
    rng = np.random.default_rng(0)
    u = np.array([0.5, 0.3, 0.15, 0.05])
    theta = rng.dirichlet(2.0 * u, size=20000)
    draws = np.array([[rng.choice(4, p=t) for _ in range(2)] for t in theta])
    assert features.fit_alpha(draws[:, 0], draws[:, 1], u, "x", list("abcd")) == pytest.approx(2.0, abs=0.2)


# --- the coupled habit construction -------------------------------------------

def test_sharing_does_not_change_within_type_repetition():
    """cross_type_sharing must move only what transfers between crime types."""
    rng = np.random.default_rng(0)
    q, alpha, n = np.array([0.52, 0.18, 0.30]), 1.25, 4000
    rates = {}
    for rho in (0.0, 1.0):
        spread = np.sqrt(1 - rho ** 2)
        first, second = [], []
        for _ in range(n):
            habit = rng.standard_normal(len(q))
            theta = sample.theta_coupled(q, alpha, rho * habit + spread * rng.standard_normal(len(q)), False)
            first.append(rng.choice(3, p=theta))
            second.append(rng.choice(3, p=theta))
        rates[rho] = float((np.array(first) == np.array(second)).mean())
    assert rates[0.0] == pytest.approx(rates[1.0], abs=0.02)


def test_coupled_theta_keeps_the_dirichlet_mean():
    rng = np.random.default_rng(1)
    q = np.array([0.52, 0.18, 0.30])
    draws = np.array([sample.theta_coupled(q, 1.25, rng.standard_normal(3), False) for _ in range(4000)])
    assert np.allclose(draws.mean(axis=0), q, atol=0.03)


# --- config -------------------------------------------------------------------

def test_shipped_config_is_valid_and_sharing_has_no_default():
    cfg = config_mod.load()
    report = config_mod.check(cfg)
    assert not report.errors and not report.unfilled
    assert cfg["corpus"]["cross_type_sharing"] is None


# --- normalisation ------------------------------------------------------------

def test_adapter_labels_are_one_to_one_and_parse_back():
    adapter = normalise.compile_adapter(
        yaml.safe_load((ROOT / "adapters/MH.yaml").read_text(encoding="utf-8")))
    labels = adapter["values"]["entry_method"]
    assert len(set(labels.values())) == len(labels)
    row = {adapter["column_for"]["crime_type"]: "HOUSE BREAKING",
           adapter["column_for"]["entry_method"]: "  lock broken  ",     # case and space insensitive
           adapter["column_for"]["fir_no"]: "MH-PUN-01/0001/2023",
           adapter["column_for"]["registered_at"]: "02/01/2023 10:00",
           adapter["column_for"]["occurred_from"]: "01/01/2023 22:00",
           adapter["column_for"]["occurred_to"]: "02/01/2023 02:00"}
    record = normalise.normalise_row(row, adapter)
    assert record["crime_type"] == "BURGLARY_RESIDENTIAL"
    assert record["entry_method"] == "lock_broken"
    assert record["time_band"] == "night"
    assert record["field_provenance"]["entry_method"] == "source"
    assert record["issues"] == []


def test_unmapped_label_is_reported_not_guessed():
    adapter = normalise.compile_adapter(
        yaml.safe_load((ROOT / "adapters/MH.yaml").read_text(encoding="utf-8")))
    row = {adapter["column_for"]["crime_type"]: "HOUSE BREAKING",
           adapter["column_for"]["entry_method"]: "TELEPORTED IN",
           adapter["column_for"]["registered_at"]: "02/01/2023 10:00"}
    record = normalise.normalise_row(row, adapter)
    assert record["entry_method"] == schema.MISSING
    assert any("TELEPORTED IN" in issue for issue in record["issues"])


def test_missing_absent_and_unknowable_stay_distinct():
    ka = normalise.compile_adapter(
        yaml.safe_load((ROOT / "adapters/KA.yaml").read_text(encoding="utf-8")))
    row = {ka["column_for"]["crime_type"]: "Burglary - Dwelling",
           ka["column_for"]["entry_method"]: "",                       # blank cell
           ka["column_for"]["registered_at"]: "02-Jan-2023 1000 hrs",
           ka["column_for"]["occurred_from"]: "01-Jan-2023 0800 hrs",
           ka["column_for"]["occurred_to"]: "04-Jan-2023 0800 hrs"}    # 72h window
    record = normalise.normalise_row(row, ka)
    assert record["entry_method"] == schema.MISSING
    assert record["occupancy"] == schema.ABSENT          # KA has no occupancy column
    assert record["time_band"] == schema.UNKNOWABLE
    assert record["vehicle_class"] is None               # not applicable to a burglary


# --- scoring ------------------------------------------------------------------

@pytest.mark.skipif(not WEIGHTS.exists(), reason="trained weights not present")
def test_contributions_sum_to_the_evidence_shown():
    scorer = Scorer(json.loads(WEIGHTS.read_text(encoding="utf-8")))
    pool = "BURGLARY_RESIDENTIAL"
    fields = scorer.pools[pool]["fields"]
    rng = np.random.default_rng(0)
    codes = {f: rng.integers(0, len(scorer.pools[pool]["vocab"][f]), size=50) for f in fields}
    bits = scorer.field_bits(pool, {f: codes[f][0] for f in fields}, codes)
    assert np.allclose(scorer.contributions(pool, bits).sum(axis=-1), scorer.evidence(pool, bits))
