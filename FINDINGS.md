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
gloves" to `gloves` and declined (counter_forensic 0%). With a hand-written field guide in the prompt (60 cases,
`results/extraction_pilot_llama3.2.json`):

| llama3.2 | no guide | with guide |
|---|---|---|
| accuracy on recorded values | 55% | **81%** |
| unrecorded field: correctly left null | 56% | 18% |
| unrecorded field: recovered the true value from the narrative | 21% | 48% |
| unrecorded field: **invented a wrong value** | 23% | **34%** |

The guide made the model answer more readily — much better where a value
was recorded, but it now fills a third of unrecorded fields wrongly, and a
wrong value scores as evidence where a blank scores zero.

Does it help linkage anyway? Injecting errors into the free-text state's true
values at exactly those measured rates, then retraining and evaluating
(`scripts/simulate_extraction.py`, `results/extraction_simulation.json`):

| free-text state's MO fields | same-type hit@10 | PR-AUC |
|---|---|---|
| blank (pipeline today) | 0.128 | 0.041 |
| extracted at llama3.2's measured error rates | 0.141 | 0.041 |
| perfect extraction | 0.156 | 0.046 |

It helps despite the inventions, recovering about half of an already small
gain, with PR-AUC unmoved — consistent with §3. This is a simulation of the
extraction's effect, not a full run; the full 8,295-case pass (~4 h locally)
is deferred as low value. Expect optimism throughout: the templated text and
a guide written by the same author make extraction easier than real FIRs.

---

## 10. Time between offences doubles same-type linkage — on the generator's own assumption

The scorer now carries one more piece of evidence: days between the two
offences, binned at 30 / 90 / 180 / 365 / 730 days, weighted
log2 P(gap | same offender) / P(gap | unrelated) on training offenders, then
passed through the same logistic correction as the MO fields (its
coefficient lands near 1, so the learned weight is used almost as-is).

Learned weights (bits), same for every pool within ±0.5:

| gap | < 30 d | 30–90 | 90–180 | 180–365 | 365–730 | > 2 y |
|---|---|---|---|---|---|---|
| house burglary | +2.97 | +1.71 | +0.51 | +0.08 | −0.61 | −3.67 |
| vehicle theft | +2.97 | +1.85 | +0.93 | +0.06 | −1.04 | −4.19 |

Held-out offenders, full pool, sharing-1 corpus
(`results/post_core_fix/eval.json` → `results/with_time/eval.json`):

| | hit@10 | recall@10 | P@10 | PR-AUC |
|---|---|---|---|---|
| same-type, MO only | 0.128 | 0.062 | 0.016 | 0.041 |
| **same-type, MO + time** | **0.235** | **0.120** | **0.030** | **0.083** |
| cross-type, MO only | 0.013 | 0.006 | 0.001 | 0.003 |
| **cross-type, MO + time** | **0.029** | **0.015** | **0.003** | **0.008** |

Median evidence on a true same-type link rose from 0.0–1.5 bits to
+0.8–2.9 bits depending on pool, and cross-type from 0.0 to +1.2
(`results/{post_core_fix,with_time}/evidence_distribution.json`).

**Caveat, and it is a large one.** The gain comes straight from the
generator's gap model (`config/corpus.yaml` `gap_days`: 70% of gaps mean 20
days). The scorer learned back an assumption we wrote. Offending *is* bursty
in the literature, so the direction is defensible, but the size of this gain
must be re-measured on real FIRs before it is quoted as a property of the
method. Say "time roughly doubles linkage on synthetic data", never "doubles
linkage".

**It works against cross-state links.** 35% of serial offenders relocate to a
neighbouring state after 90–365 dormant days, so the gap that marks a
relocation is exactly the gap the time weight penalises. On held-out true
pairs of the same crime type:

| true pairs | n | median evidence | median time part | time is negative |
|---|---|---|---|---|
| same state | 1,683 | +2.15 | +1.84 | 9% |
| different states | 547 | +0.03 | −0.59 | 55% |

A relocation-aware gap model (a separate gap curve when the two FIRs are in
different states) is the principled fix; it was not built, because it
conditions the score on location — see §11.

## 11. Leads inbox: strength as rarity, organised by where the FIRs are

**Strength is stated as rarity, not probability.** For each pool the bundle
scores 200,000 random unrelated pairs and keeps their percentiles plus the
exact top 1%. A lead's strength is "about 1 in N unrelated pairs look this
alike"; tiers are strong ≥ 1 in 10,000, worth checking ≥ 1 in 1,000, weak
otherwise. Beyond the sample the UI says "rarer than 1 in 200,000" instead
of extrapolating.

**The inbox.** Every case is scored against its whole same-type pool; its
top 3 matches become candidates, rarer than 1 in 1,000 are kept, mutual
matches (each is in the other's top 3) first, then rarest. 5,000 are kept
(`data/serve/meta.json` `lead_quality`):

| lane | leads | estimated real links | vs random pair |
|---|---|---|---|
| same district | 433 | **20.9%** (about 1 in 5) | ~5,400× |
| other district, same state | 1,485 | 3.8% (about 1 in 27) | ~1,000× |
| **other state** | **3,082** | **~0** (3 real among all offenders) | — |
| all | 5,000 | 2.9% | ~760× |

Leads touching a FIR the crime-type check flags (§13) are listed last in each
lane. They were almost all false: in the same-district lane 4 of 36 such
leads were real vs 120 of 399 others (all offenders), and demoting them took
the true links among the lane's first 20 from 10 to 19.

**Correction.** An earlier version of this section reported 22 of 36
same-district leads (61%) and 11% overall. Those numbers counted only leads
whose *both* FIRs came from held-out offenders. That filter is biased: a true
pair (one offender) survives it with probability ≈ 0.23, a false pair (two
offenders) with ≈ 0.23² ≈ 0.05, so it inflates the odds about 4×. The
estimate above counts true leads whose offender is held out, divides by the
held-out share of all true pairs (0.232), and divides by every lead. It is
unbiased for the rate an unseen offender would get. Counting all offenders
(including the 75% the scorer was trained on) gives 28.9% same district —
the gap is the training optimism the estimate removes. `evaluate.py`'s
hit@10 was not affected: it ranks a held-out query against the whole pool.

Chance: a random same-type pair is a true link 1 in 25,942.

**Location stays out of the score and organises the inbox instead.** The
score is location-blind on purpose, so that a match across a state line can
surface at all. The cost shows in the table: 73.5% of same-type pairs cross a
state line and most true links do not, so among high-scoring pairs the
cross-state ones are overwhelmingly coincidence (3 real out of 3,064 in the
whole inbox). Mixing them into one list put noise at the top. The inbox now
has three lanes, defaults to same district, and states each lane's measured
record next to it; the cross-state lane is labelled low confidence rather
than hidden. The case shortlist stays one location-blind ranking with the
same warning.

Two things we did not do, and why. Adding a same-state evidence term would
lift precision sharply but bury exactly the cross-state links the project
exists for. Filtering cross-state leads out would hide a lane that no keyword
search can reach. Both trade the pitch for a number.

**Where the top of the house-burglary lane goes wrong.** Several of the
strongest false leads share shutter entry, "closed for holidays" and shop
premises — commercial MO values in the residential pool. Pairs of
misclassified FIRs look alike because they are misclassified, not because
one offender did both. On real data this is the IPC 380/454/457 confusion;
flagging "MO atypical for the recorded crime type" as its own finding would
be more useful than a link.

## 12. Possible series: chaining only the links that hold up

A series is a connected group of FIRs joined by mutual top-3 matches. Plain
connected components fail the way CLAUDE.md warns: with every strong mutual
edge, cross-state edges chain unrelated cases into components of 3,000–7,000
FIRs. Tested on held-out pair precision (scratch sweep, not committed):

| edges allowed | series | largest | pairs that are one offender* |
|---|---|---|---|
| all mutual ≥ 1 in 1,000, any geography | 1,376 | 7,007 | 0.00 |
| same state only, ≥ 1 in 10,000 | 1,126 | 13 | 0.16 |
| same district ≥ 1 in 10,000 | 101 | 6 | 0.72 |
| same district ≥ 1 in 10,000 + other district ≥ 1 in 200,000 | 206 | 7 | 0.53 |

\*both-held-out filter, so inflated the same way as §11's first draft; the
ranking between rows is what was used.

Shipped: the last row, minus any FIR the crime-type check (§13) flags — two
misfiled FIRs share values that are rare only because they sit in the wrong
pool. Result (`meta.json` `series_quality`, estimator from §11): **182 series,
102 spanning districts; about 1 in 5 FIR pairs inside a series are the same
offender** (20.4%); 25 series are wholly one offender. Larger series and
series held together by same-district links were purer, so the list is
ordered by size, then share of same-district links. The UI calls a series "a
set of links to check one by one, not a confirmed gang" and shows every link
with its own strength.

## 13. Crime-type check: misfiled FIRs, found without labels

The generator records some FIRs under the wrong type — 948 of 44,533, all
house ↔ commercial burglary (`states.yaml` confusability). For each FIR in a
family with more than one type, sum over the fields both pools score of
log2(frequency of the recorded value under the other type ÷ under the recorded
type). Flag at ≥ 2 bits. No labels are used, only value frequencies.

| threshold | flagged | truly misfiled | recall |
|---|---|---|---|
| 2 bits (shipped) | 714 | **97.1%** | **73.1%** |
| 4 bits | 575 | 100% | 61% |
| 8 bits | 181 | 100% | 19% |

Caveat: the generator's misfiling moves a FIR between types whose MO differs
sharply (shop vs house premises), so this mostly shows the check can invert
our own confusion model. Real misfiling (IPC 380 vs 454 vs 457) is subtler.
It matters for linkage because the pools are by recorded type: a misfiled
commercial burglary is compared only with house burglaries, so its real
partners never appear in its shortlist, and its rare-in-this-pool values
manufacture false strong leads. The case view and the comparison page show
the flag.

---

## Where things are

| | |
|---|---|
| `pre-core-fix` tag | corpus + results with core habits drawn per type (sharing = 0) |
| `results/pre_core_fix/` | eval, oracle eval, weights, evidence distribution |
| `results/with_time/` | weights, eval and evidence distribution with the time evidence (current model) |
| `results/sweep/` | repeat_rate × cross_type_sharing grid, one file per point |
| `data/final/` | the 45k corpus (sharing = 0; see manifest `derived`) |
| `config/tau_selection.json` | how τ = 0.75 was chosen, including two discarded sweeps |
