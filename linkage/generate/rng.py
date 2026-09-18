"""Named random streams derived from one seed.

A single `default_rng` threaded through everything desyncs on any config
change: Dirichlet and Beta draws consume a variable amount of randomness, so
changing repeat_rate would reshuffle every later draw — series lengths, gaps,
relocation — and the sweeps would compare different offenders. Instead each
component, and each serial offender, reads its own stream derived from the
one seed via SeedSequence spawn keys.
"""
import numpy as np

STRUCTURE = 0    # per offender: style, series length, gaps, relocation, geography, clocks
THETA = 1        # per offender: Dirichlet / Beta draws (depend on alpha and tau)
CRIMES = 2       # per offender: fixed-size uniform block per crime for value draws
BACKGROUND = 3   # one stream for all background one-offs
CORRUPTION = 4   # recording noise; takes its own seed so truth can be held fixed
RENDER = 5       # narrative choices
SHARED = 6       # per offender: person-level mo_core habit (cross_type_sharing)


def stream(seed: int, kind: int, index: int = 0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(kind, index)))
