# Findings

Source for the writeup. Every number here is measured and names the file it
came from. Corpus: 45,000 synthetic cases, seed 7, τ = 0.75, evaluated on
held-out TEST offenders (25%, split by offender).

---

## 1. Defect: mo_core habits were drawn per crime type

`mo_core` exists because those fields are **person-level** — time band, group
size, concealment, target selection, approach and exit mode. A burglar who
works nights works nights when he snatches chains too; that is a property of
the person, not of the offence category. `mo_ext` holds the type-specific
fields. That split is the whole reason cross-type scoring is possible at all.

The generator drew core fields independently for each crime type, which
contradicts that rationale. Measured in the **true** values, before any
recording noise:

| agreement on core fields | same type | cross type |
|---|---|---|
| same offender | 0.71 | 0.41 |
| random pairs | 0.40 | 0.36 |
| **lift** | **+0.24** | **+0.05** |

A serial offender's crimes of different types agreed about as often as random
pairs did, so at this corpus's repeat_rate (0.5) cross-type linkage was
impossible **by construction**, not by measurement — no scorer could have
found those links (hit@10 0.004, evidence −0.01 bits).

One qualification the sweep adds (§7): at extreme within-type consistency
(repeat_rate 0.9) cross-type signal appears even with zero habit transfer,
because style tilts every crime type's centre and style is shared. The defect
is real and worth fixing, but "no transfer means no cross-type linkage" holds
only away from that extreme.

**Fix** (2026-09-18): an offender's core habit is a set of quantiles, drawn
once and read against each crime type's own centre, so the same values are
favoured everywhere while each type keeps its own base rates — a night worker
is a night worker everywhere, and a rarely-nocturnal crime type still rarely
happens at night. `cross_type_sharing` is the correlation between an
offender's types. Measured on the two committed corpora:

| | core lift same-type | core lift cross-type |
|---|---|---|
| sharing 0 (`data/final/`) | +0.240 | +0.050 |
| sharing 1 (`data/final_sharing1/`) | +0.239 | +0.214 |

Cross-type transfer appears; same-type is untouched, which is the property the
sweep needs. Both corpora pass all six validators.

### Two parameterisations were discarded before this one

Both failed the same test — sharing must not change within-type behaviour —
and both were caught by measurement, not by inspection:

1. **Blend the drawn habits**, `(1−ρ)·θ_type + ρ·θ_person`. Mixing two
   concentrated distributions gives a flatter one, so mid-range sharing cut
   within-type repetition: same-type hit@10 went 0.411 → 0.258 → 0.457 across
   sharing at repeat_rate 0.8, a dip of a third with no meaning.
   (`results/discarded/sweep_mixture_summary.json`)
2. **Displace the type's centre, then re-solve α** to hold the within-type
   agreement rate. Infeasible: a sharpened centre already agrees more often
   than the target, so α ran to its ceiling (1.25 → ~4250), which removed the
   offender-level spread entirely and still left same-type rising with
   sharing (0.134 → 0.181 → 0.239).

The accepted construction builds θ from coupled uniforms (inverse CDF), so
each type's θ keeps **exactly** its original law whatever the correlation. A
harness over 40,000 simulated offenders confirms it: within-type agreement
0.665 / 0.664 / 0.665 / 0.666 / 0.663 across sharing 0 → 1, while cross-type
agreement rises 0.293 → 0.490 and both random baselines stay flat.

This is a defect report, not a parameter added until the feature worked:
- At `cross_type_sharing = 0` the generator reproduces the previous corpus
  **byte for byte** (all four feeds), so the old results remain valid as the
  zero endpoint — tag `pre-core-fix`.
- The parameter has **no default**. `corpus.yaml` holds `null` and the
  generator refuses to run without an explicit value, because no published
  figure says how much habit transfers between offence types.
- The result is reported as a curve over sharing, never as one value.

## 2. Evidence per link is small; +5.85 bits is a strong case, not a typical one

The spec's §4.3 worked example totals +5.845 bits. It was constructed to
illustrate rarity weighting and had quietly become the headline number.
Measured on test offenders (`results/pre_core_fix/evidence_distribution.json`):

| pool | median | p90 | p99 | max | share ≥ +5.85 |
|---|---|---|---|---|---|
| residential burglary | +0.33 | +4.91 | +10.35 | +11.74 | 7.7% |
| commercial burglary | +1.49 | +6.53 | +9.03 | +11.51 | 11.2% |
| vehicle theft | +0.66 | +4.16 | +7.12 | +10.30 | 3.6% |
| snatching | +0.45 | +2.99 | +4.68 | +6.79 | 0.5% |
| ATM tampering | +0.00 | +3.65 | +6.83 | +9.35 | 1.9% |
| cross-type (pre-fix) | +0.00 | +0.50 | +1.59 | +2.93 | 0% |

Most linked pairs carry about a bit of evidence; a few carry six. The
system's job is to find the rare high-evidence pairs in a haystack, which is
what a ranker does. Quote +5.85 only as a strong match, with the typical
range beside it. Priors are −12 to −17 bits, so no amount of this clears the
base rate: **ranker, not classifier**.

## 3. Extraction quality is not the bottleneck — information is

One state publishes MO as free text, so those fields need LLM extraction.
Substituting the true values (a perfect-extraction upper bound) moves
same-type hit@10 from **0.136 to 0.158**, a 2.2-point gap
(`results/pre_core_fix/eval.json` vs `eval_oracle.json`).

So extraction prompts and extra extraction passes buy almost nothing. The
ceiling is how few bits an FIR's MO carries, not how well it is parsed.

## 4. What the fix bought, and what cross-type success means

Same corpus size, seed and τ; only sharing differs
(`results/pre_core_fix/eval.json`, `results/post_core_fix/eval.json`):

| | same-type hit@10 | cross-type hit@10 | cross-type mean true-pair evidence |
|---|---|---|---|
| sharing 0 | 0.136 | 0.004 (≈5× random) | −0.01 bits |
| sharing 1 | 0.128 | 0.013 (≈17× random) | +0.18 bits |

Cross-type roughly triples; same-type does not move (0.136 → 0.128 is within
query-sampling noise), which is the independence the construction guarantees.
In absolute terms cross-type stays small: 1.3% of queries put a true partner
in the top 10. Its evidence on the sharing-1 corpus is median +0.00, p90
+1.73, max +8.90 bits, with 0.5% of true pairs at or above the spec's +5.85
example (`results/post_core_fix/evidence_distribution.json`).


Cross-type scores 8 fields instead of 13 against a prior ~2 bits worse, so it
will stay much weaker than same-type linking even after the fix. The
defensible claim:

> Keyword search on IPC sections cannot surface these pairs at all — not
> weakly, at all — because the offences file under different sections. This
> surfaces them as candidates for an analyst.

Going from no capability to a weak one is the interesting claim. "We link
reliably across crime types" is not supportable and should not be said.

## 5. Same-type scoring, measured

Full-pool ranking, no blocking, so rank is out of every case in the pool
(`results/pre_core_fix/eval.json`):

| | hit@10 | recall@10 | P@10 | PR-AUC | vs random P@10 |
|---|---|---|---|---|---|
| same-type (pooled) | 0.136 | 0.051 | 0.018 | 0.031 | 22–96× |
| cross-type (pre-fix) | 0.004 | 0.003 | 0.000 | 0.002 | ~5× |

A true partner typically ranks in the top ~2% of the pool (median first
partner at rank 289 of 12,564 for residential burglary). Baselines in the
same file: counting shared fields is clearly worse than Fellegi-Sunter
weighting, and the logistic correction adds little beyond FS — it mostly
down-weights fields the generator makes redundant (`exit_mode`, which copies
`approach_mode`) and zeroes `property_taken` for cross-type pairs, where what
is taken is dictated by crime type rather than the offender.

## 6. Validator 3 is a ratio of two noisy estimates

Validator 3 requires the recorded table to keep ≥ 50% of truth's excess
cross-field mutual information. Both committed 45k corpora pass it (sharing 1
keeps 0.609), but the 10k dev corpus fails for lack of power, and the two
discarded parameterisations failed it at 0.486 — close enough to the
threshold that it flips on noise. The threshold has not been adjusted. Sweep
points record their own validator results and `summary.csv` carries the
pass/fail column, so any failing point is reported as failing.

## 7. The sweep: what consistency and habit transfer actually buy

30 points, repeat_rate 0.2–0.9 × cross_type_sharing 0–1, one 45k corpus each
(`results/sweep/summary.csv`, figure `results/sweep/sweep.png`). hit@10 on
held-out test offenders, full-pool ranking.

| repeat_rate | same-type, sharing 0 | same-type, sharing 1 | cross-type, sharing 0 | cross-type, sharing 1 |
|---|---|---|---|---|
| 0.2 | 0.020 | 0.023 | 0.000 | 0.003 |
| 0.35 | 0.041 | 0.055 | 0.002 | 0.008 |
| 0.5 | 0.134 | 0.128 | 0.002 | 0.013 |
| 0.65 | 0.247 | 0.268 | 0.003 | 0.033 |
| 0.8 | 0.411 | 0.390 | 0.043 | 0.082 |
| 0.9 | 0.502 | 0.502 | 0.125 | 0.138 |

Three things to say from this, and only these:

1. **Same-type linkage is governed by within-type consistency alone.** Across
   the whole grid, sharing leaves it unchanged — the construction's guarantee,
   visible in the data.
2. **Cross-type linkage needs habit transfer — until consistency gets
   extreme.** At repeat_rate 0.5–0.65 sharing takes it from ~0.002 to
   0.013–0.033, roughly a tenfold difference. At 0.9 it is ~0.13 whatever the
   sharing, because style already tilts every crime type's centre and style is
   shared by construction. Do not claim sharing is the only route.
3. **Evidence never approaches the base rate.** The strongest point on the
   grid averages 5.42 bits per true pair against a ~14.1 bit break-even. A
   ranker throughout; never a classifier.

The x-axis reader-facing version is panel C: cross-type hit@10 against the
**measured** cross-type agreement lift, because "sharing 0.5" means nothing to
an audience while "offenders repeat 21% more often than chance across types"
does. Sharing 1 produces a measured lift of 0.085 (repeat_rate 0.2) to 0.32
(repeat_rate 0.9).

### How much of this is noise

Grid points are single-seed. Three seeds (7, 11, 23) at four points
(`results/sweep_seeds/`) give the floor:

| point | same-type hit@10 | cross-type hit@10 |
|---|---|---|
| rr 0.65, sharing 0 | 0.247 / 0.269 / 0.250 | 0.003 / 0.012 / 0.003 |
| rr 0.65, sharing 1 | 0.268 / 0.247 / 0.270 | 0.033 / 0.023 / 0.018 |
| rr 0.9, sharing 0 | 0.502 / 0.501 / 0.508 | 0.125 / 0.099 / 0.098 |
| rr 0.9, sharing 1 | 0.502 / 0.488 / 0.510 | 0.138 / 0.116 / 0.114 |

Read as: differences below **±0.02 same-type** and **±0.015 cross-type** are
seed noise, so

- sharing's effect on same-type (0.255 vs 0.262 at repeat_rate 0.65) is **not
  distinguishable from zero** — as the construction requires;
- sharing's effect on cross-type at repeat_rate 0.65 (0.006 vs 0.025, ranges
  0.003–0.012 vs 0.018–0.033) **is real**, with no overlap;
- sharing's effect on cross-type at repeat_rate 0.9 (0.107 vs 0.123, ranges
  overlapping) **is not** — at extreme consistency, sharing adds nothing
  measurable.

## 8. Embedding retrieval matches on place, not behaviour — so it is not the gate

The spec's pipeline retrieved each case's top-50 by narrative-embedding
cosine and only then scored MO. Measured with `intfloat/multilingual-e5-base`
on the sharing-1 corpus, test offenders (`results/retrieval.json`,
`results/retrieval_no_place.json`):

| retrieval recall@50 | same-state partners | cross-state partners |
|---|---|---|
| full narratives | 0.127 | **0.000** |
| place/date opening sentence removed | 0.021 | 0.013 |

Every narrative opens with district, police station and dates, and serial
offenders stay in one district 70% of the time. The embeddings therefore find
partners by **place**: they retrieved **none** of the 1,094 cross-state
same-type partners — the pairs this product exists to find. With the place
text removed, retrieval is barely above random (0.020 against 0.0057), and
the two-stage pipeline reaches hit@10 **0.045** where the MO scorer on the
whole pool reaches **0.128**. The gate discards most true partners before the
scorer ever sees them.

The gate existed for cost, and the cost is not there at this scale: scoring
every case against its entire pool takes **~9 minutes on one core** for all
44,533 cases (≈2 min same-type, ≈7 min cross-type) — about 12 s per chunk
across the spec's existing Step Functions Map.

**Decision:** LinkBatch scores the full pool, blocked only by crime-type pool,
which is not a scored field, so "never block on a feature you also score"
still holds. Embeddings are kept out of the ranking path. At national scale
(millions of cases) a gate returns, and it must be one that does not encode
place: a time window, or crime type.

Two cautions this also raises:
- Cheat-detector validator 4 checked metadata columns only; place leaked
  through **narrative text**. Any free-text channel needs the same scrutiny.
- The templated narratives here are easier than real FIRs, so even the
  place-stripped number is optimistic.

## 9. MO extraction with a local model — in progress

Bedrock is unavailable, so extraction for the free-text state runs locally
through Ollama with a JSON schema that restricts every field to canonical
values or null (`enrich/extract.py`, scored by `enrich/score_extraction.py`
against the values the text was rendered from).

First pilot, llama3.2 (3B), 41 cases, prompt without value descriptions:
**55% accuracy** on recorded values, and on fields where nothing was recorded
it **invented a value 23% of the time**. The failure was specific: with no
explanation of what the enum values mean, the model could not map "wore
gloves" to `gloves` and declined (counter_forensic 0%). The prompt now
carries a hand-written field guide; the rerun and the full 8,295-case pass
are pending. Invented values matter more than blanks here — a wrong value
scores as evidence, a blank scores zero — so the rerun must report the
invention rate, not just accuracy. Also expect optimism: the templated text
and a guide written by the same author make this easier than real FIRs.

---

## Where things are

| | |
|---|---|
| `pre-core-fix` tag | corpus + results with core habits drawn per type (sharing = 0) |
| `results/pre_core_fix/` | eval, oracle eval, weights, evidence distribution |
| `results/sweep/` | repeat_rate × cross_type_sharing grid, one file per point |
| `data/final/` | the 45k corpus (sharing = 0; see manifest `derived`) |
| `config/tau_selection.json` | how τ = 0.75 was chosen, including two discarded sweeps |
