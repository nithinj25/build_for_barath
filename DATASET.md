# Synthetic corpus

**Synthetic.** Every number measured on this corpus measures whether a model
can invert this generator — not real-world performance.

## Regenerate

Two corpora are committed: the **endpoints** of the `cross_type_sharing`
sweep. That parameter says how much of an offender's mo_core habit carries
across crime types, nothing published fixes its value, so the finding is the
curve between these two — never one of them on its own
(`results/sweep/`, `FINDINGS.md`).

| Corpus | `cross_type_sharing` | Use it for |
|---|---|---|
| `data/final/` | 0 — habits per crime type | the zero endpoint; cross-type linkage is impossible on it by construction (tag `pre-core-fix`) |
| `data/final_sharing1/` | 1 — habits fully person-level | the upper endpoint; **use this one for demos and pipeline work** |

```bash
python -m linkage.config                                          # must print READY
python -m linkage.generate --seed 7 --n-cases 45000 --cross-type-sharing 1 --out data/final_sharing1/
python -m linkage.generate --seed 7 --cross-type-sharing 1 --out data/dev/   # 10k dev corpus
python -m linkage.generate.tau --n-cases 45000                    # re-derive τ
python -m linkage.sweep --out results/sweep                       # repeat_rate × sharing grid
```

`--cross-type-sharing` is required and has no default. Both committed corpora
are usable without running anything; the rest of `data/` is gitignored. Seed +
`config/` reproduce any corpus exactly.

## Artifacts

| File | What | Enters the pipeline? |
|---|---|---|
| `feeds/{MH,MP,KA,TG}.csv` | Recorded, corrupted, state-native feeds | **Yes — ingest these** |
| `truth.parquet` | One row per case: `offender_id`, style, true values and dates, and the recorded canonical values (`rec_*`, `ingested`) | Never. Labels and normaliser answer key |
| `offenders.parquet` | Style, repeat_rate, alpha, series length, home/destination state, relocation index | Never |
| `manifest.json` | Seeds, full config, config vs realised marginals, link counts and priors, validator results, limitations | — |

Join feeds to truth on `case_id = sha1(state_code + fir_no + year)[:16]`.

`rec_*` tokens: `__MISSING__` blank cell · `__UNKNOWABLE__` occurrence window
too wide to know the time band · `__ABSENT__` state has no such column.

## Final corpus (seed 7, 45,000 cases)

- 1,809 serial offenders (628 relocate across a state border), 7,204 serial cases
- Feed rows: MH 15,620 · MP 10,779 · KA 9,839 · TG 8,295
- 467 cases recorded as ROBBERY and dropped at ingestion; 44,533 ingested
- 15,810 true same-offender pairs: 5,907 cross-type, 2,768 cross-state
- Prior, all property crime: −15.94 bits. Same-type: −15.08 (residential
  burglary) to −12.16 (ATM). Cross-type: −16.93.
- Top 10% of offenders hold 49.5% of true pairs — report per-offender metrics
  alongside per-pair ones.
Both corpora share those counts: sharing changes what offenders do, not how
many cases, offenders or links there are.

**Validators.** Both 45k corpora pass all six. The 10k dev corpus fails
validator 3 (recorded cross-field mutual information not significant at that
size); use a 45k corpus for evaluation.

**Linkage measured on each** (held-out test offenders, full-pool ranking):

| | same-type hit@10 | cross-type hit@10 | cross-type mean true-pair evidence |
|---|---|---|---|
| sharing 0 | 0.136 | 0.004 (≈5× random) | −0.01 bits |
| sharing 1 | 0.128 | 0.013 (≈17× random) | +0.18 bits |

Cross-type stays weak either way: it scores 8 fields against a prior ~2 bits
worse. The claim that holds is that keyword search on IPC sections cannot
surface these pairs at all, and this surfaces them as candidates.

## Limitations

- Every marginal is an **assumption** (see provenance tags); the loadings
  were drafted by Claude at the owner's request and need owner review.
- τ = 0.75 was chosen by the rule in `config/tau_selection.json`. Expected
  drift in serial crimes reaches +229% for rare heavily-loaded values
  (ATM explosive); corpus-level realised drift is in the manifest.
- Narratives are templated English. Cross-lingual retrieval is **not**
  demonstrated, and retrieval on templated text will look easier than on
  real FIRs.
- Background one-offs have style exactly 0 (brief decision), so rare,
  heavily-loaded values are over-represented among serial crimes: an artifact
  a scorer can exploit. The ATM same-type prior (−12.16) partly reflects it.
- Crime type is drawn through the style chain (proposal adopted), so style
  selects offenders into crime types and amplifies within-type drift.
- **This corpus has `cross_type_sharing = 0`** (see manifest `derived`):
  offenders' mo_core habits were drawn per crime type, so person-level habits
  do not transfer across types and cross-type linkage is impossible on it by
  construction. That was a defect, fixed 2026-09-18; this corpus is kept as
  the zero endpoint of the sweep (tag `pre-core-fix`). See `FINDINGS.md` §1.
- **Evidence per link is small.** Median true pair +0.0 to +1.5 bits; the
  spec's +5.85 example is a strong match (top 0.5–11%), not a typical one.
- Non-links here are clean. Real unlabelled pairs are not confirmed
  non-links (positive-unlabelled); don't carry accuracy-style metrics over.
- Not checked: state predictability after normalisation (the normaliser now
  exists; the check does not).
