# Cross-jurisdiction crime linkage

Links property-crime cases across state borders by behavioural similarity.
Surfaces a ranked shortlist for a human analyst. Never identifies a person.

Full spec: `TECHNICAL_SPEC.md`. Read it before changing the scorer or the schema.
Generator brief: `TASK_DATA_GENERATION.md`.

## Hard rules

These are easy to violate by accident and each one breaks something real.

1. **Never block on a feature you also score.** Filtering by a feature and
   scoring by that feature are the same operation — verified numerically:
   pre-filtering to roof-entry cases improves the prior by exactly the roof
   agreement weight. If someone suggests "filter to X first to speed it up"
   and X is a scored field, the answer is no.
   **LinkBatch scores the full pool**, blocked only by crime-type pool (not a
   scored field). Narrative embeddings are NOT a retrieval gate: measured,
   they match on place text and retrieved 0 of 1,094 cross-state partners
   (FINDINGS §8). Full-pool scoring of 44.5k cases takes ~9 min on one core.
   Any future gate must not encode place — a time window or crime type.

2. **`linkage/` must not import boto3.** Core logic is pure functions over
   plain dicts and arrays. Handlers in `handlers/` do the AWS glue. If a
   scoring function needs to know where its data came from, it's wrong.

3. **Never ship scikit-learn to Lambda.** Lambda's unzipped package limit is
   250 MB and sklearn+scipy+numpy gets close. Train locally, export
   coefficients to JSON, do inference in pure numpy:
   `1 / (1 + exp(-(w @ x + b)))`.

4. **Never render a probability in the UI.** A strong match posteriors at
   ~0.27% and that reads as broken. Render rank, bits, and the driving
   features: `rank 2 of 10,482 · +5.85 bits · roof entry, disabled CCTV`.
   **+5.85 is a STRONG match, not a typical one** — it is the spec's
   constructed example and sits in the top 5–21% of same-type true pairs
   (MO + time) depending on crime type. Measured medians are +0.8 to +2.9
   bits, p90 +4.8 to +7.5 (`results/with_time/evidence_distribution.json`).
   Never quote +5.85 as what a link scores; quote it as a strong case beside
   the typical range. The UI states strength as rarity ("about 1 in N
   unrelated pairs look this alike"), measured on random unrelated pairs.

5. **No protected attributes as features.** No caste, religion, community,
   or any proxy for them. Not in the schema, not in the generator, not in
   the model.

6. **Split evaluation by offender, never by case.** Splitting by case leaks
   the same offender across train and test and the numbers come out
   beautiful and meaningless.

7. **Report same-type and cross-type pairs separately.** Never blend them.
   Cross-type scores on `mo_core` only and has a worse prior.

8. **PII never enters the scoring path.** `linkage/` and the link batch job
   have no access to the pii table. That's enforced by IAM, not convention —
   don't add a code path that assumes otherwise.

## Layout

```
linkage/        pure functions, no AWS imports
  schema.py     field names, kinds, mo_core vocabulary (spec §3.1)
  config.py     loads and checks config/*.yaml
  normalise.py  feed rows → canonical records via adapters/ (no per-state code)
  features.py   pair featurisation, Fellegi-Sunter weights (numpy only)
  score.py      inference from exported coefficients (numpy only)
  dataset.py    local: normalised records + labels, offender split, pairs
  generate/     synthetic corpus with ground-truth offender IDs
    sample.py     truth: style tilt → θ → values, series, geography, clocks
    corrupt.py    per-state recording noise (canonical vocab, tokens)
    render.py     state-native feeds + narratives
    validate.py   the six validators
    report.py     manifest content: realised marginals, priors, limitations
    tau.py        τ selection sweep → config/tau_selection.json
  train.py      local only, writes weights.json
  evaluate.py   test offenders, full-pool ranking: hit/recall/precision@10, PR-AUC per pair_class
config/         generator inputs: marginals, loadings, states, corpus, vocab
adapters/       one YAML per state feed; the pipeline's only state knowledge
  serve.py      shortlists on request (pure numpy; what the API Lambda runs)
  bundle.py     local: package corpus + weights into the serve bundle
  retrieval.py  chunked cosine top-k (kept for diagnostics; NOT a gate)
enrich/         local models: embeddings (sentence-transformers), MO extraction (Ollama)
handlers/       Lambda entry points (api.py, store.py); boto3 lives here only
ui/             analyst view (index.html), served locally by scripts/serve_local.py
infra/          SAM template: free tier — Lambda + Function URL + DynamoDB (stack linkage-demo, ap-south-1)
TECHNICAL_SPEC.md           full technical spec
TASK_DATA_GENERATION.md     generator brief
DATASET.md      dataset card: artifacts, key numbers, limitations
FINDINGS.md     measured results + their provenance; read before quoting numbers
DEMO.md         3-minute demo script with the FIRs to click
scripts/        local tooling (plot_sweep.py)
results/        committed results: pre_core_fix/, post_core_fix/, sweep/
data/           gitignored except the two committed sweep-endpoint corpora
```

## Key constants

- Region: `ap-south-1` (Bedrock model access must be enabled per model)
- Models: `anthropic.claude-haiku-4-5`, `cohere.embed-multilingual-v3` —
  **unverified short names**. Replace with exactly what
  `aws bedrock list-foundation-models --region ap-south-1` and
  `aws bedrock list-inference-profiles --region ap-south-1` return; never guess
  the versioned form.
- `repeat_rate` is the target `m` at `reference_frequency` (config, 0.10).
  Solved internally: `alpha = (1 - repeat_rate) / (repeat_rate - q_ref)`.
  Small alpha = high repetition. Never expose raw alpha or a "consistency"
  knob — the direction reads backwards.
- Agreement weight: `log2(m / u)`; disagreement: `log2((1-m) / (1-u))`
- `m_v = (alpha * p_v + 1) / (alpha + 1)` for categorical fields
- `m_tag = (1 - p) / (alpha + 1) + p` for multi-label tags
- Dev corpus: 10,000 cases. Prior at that size is −14.35 bits.

## Commands

```bash
python -m linkage.config                           # check config/, list unfilled entries
python -m linkage.generate --seed 7 --cross-type-sharing 1 --out data/dev/   # corpus (sharing is required)
python -m linkage.sweep --out results/sweep         # repeat_rate × cross_type_sharing grid
python scripts/plot_sweep.py results/sweep          # headline figure + summary.csv
python -m linkage.normalise --feeds data/final/feeds --out data/final_normalised.parquet --check data/final/truth.parquet
python -m linkage.train --normalised data/final_sharing1_normalised.parquet --truth data/final_sharing1/truth.parquet --out data/weights.json
python -m linkage.evaluate --normalised data/final_sharing1_normalised.parquet --truth data/final_sharing1/truth.parquet --weights data/weights.json --out data/eval.json
python -m linkage.bundle --out data/serve --with-ground-truth   # serve bundle for the API
python scripts/serve_local.py --demo               # UI + API locally on :8000, same handler as Lambda
python scripts/package_lambda.py                   # stage build/app + Linux numpy layer (no Docker)
sam build && sam deploy --stack-name linkage-demo --resolve-s3 --capabilities CAPABILITY_IAM   # from infra/
```

## Conventions

- Vector similarity is chunked: 1,000 query rows at a time against the full
  corpus, keep top 50, discard the block. Never materialise the full matrix.
- `u` values come from an Athena GROUP BY over the corpus, computed per
  crime type. Cross-type pairs use pooled property-crime frequencies.
- `m` is estimated separately for cross-type pairs — same-type consistency
  overstates cross-type agreement.
- Clustering uses a deliberately high edge threshold. Transitive chaining
  creates bogus mega-clusters otherwise.
- Every MO field carries `field_provenance`: `source` or `llm_extracted`.
  An analyst must be able to tell a recorded value from an inferred one.

## Honesty requirements

These go in the README and the demo, not just the code.

- The corpus is synthetic. Evaluating on data from our own generator
  measures whether the model can invert our generator, not real-world
  performance. Say so.
- Don't tune `repeat_rate` until the numbers look good. Present the sweep
  from 0.2 to 0.9 as the finding.
- `cross_type_sharing` has no default and `corpus.yaml` leaves it null: no
  published figure says how much MO habit transfers between offence types.
  Report the repeat_rate × sharing curve, never a single value.
- Cross-type is "keyword search cannot surface these pairs at all; this
  surfaces them as candidates" — not "we link reliably across crime types".
  It scores 8 fields against a prior ~2 bits worse and stays weak.
- Extraction quality is not the bottleneck: perfect extraction of the
  free-text state moves same-type hit@10 by 2.2 points. Don't spend time on
  extraction prompts expecting linkage gains.
- Measured results and their provenance live in `FINDINGS.md`. Read it before
  quoting any number in a writeup or demo.
- Unlabelled pairs are not confirmed non-links. This is positive-unlabelled
  learning; don't report accuracy treating unlabelled as negative.
