# Cross-Jurisdiction Crime Linkage — Technical Spec

Property-crime case linkage across state boundaries. Surfaces probably-linked
cases for a human analyst. Never names a person.

---

## 1. Problem

The data plumbing already exists. CCTNS runs in ~95% of police stations and ICJS
links police, courts, prisons and forensics. Cri-MAC allows inter-state sharing,
but it is **human-initiated and keyword-based** — somebody must already suspect a
link and go searching.

Nobody proactively computes "this FIR in Nashik resembles that one in Indore from
ten months ago." An offender who relocates and doesn't trigger a manual alert is
invisible by default.

**The pipes exist. The inference doesn't.**

---

## 2. Scope

Property crime family only:

| Type | Code |
|---|---|
| Residential burglary | `BURGLARY_RESIDENTIAL` |
| Commercial break-in | `BURGLARY_COMMERCIAL` |
| Vehicle theft | `VEHICLE_THEFT` |
| Chain snatching | `SNATCHING` |
| ATM tampering | `ATM_TAMPERING` |

**Explicitly excluded:** homicide, sexual offences, POCSO. Rationale in §11.

**Headline capability:** cross-type linking. A gang doing house burglaries in one
state and chain snatchings in another files under different IPC sections, so
keyword search can never connect them — not weakly, at all. This surfaces them
as *candidates*.

State the claim that way. Cross-type scores on 8 fields instead of 13 against a
prior ~2 bits worse, so it will stay far weaker than same-type linking and will
not produce confident links. Going from no capability to a weak one is the
claim that holds; "we link reliably across crime types" is not.

---

## 3. Data model

### 3.1 Canonical case record

```
case_id                 hash(state_code + fir_no + year)
state_code, district
occurred_from           FIRs give a window, not a point
occurred_to
registered_at
crime_type              controlled vocab, see §2

mo_core:                # scoreable across ALL property crime types
  time_band             night | early_morning | day
  group_size_est        1 | 2-3 | 4+ | unknown
  tools[]               crowbar | cutter | screwdriver | gas_cutter | none_observed
  counter_forensic      none | gloves | face_covered | cctv_disabled
  target_selection      opportunistic | scouted | insider_info
  property_taken[]      gold | cash | electronics | documents | vehicle
  approach_mode         on_foot | two_wheeler | four_wheeler | unknown
  exit_mode             on_foot | two_wheeler | four_wheeler | unknown

mo_ext:                 # type-specific, scoreable only within same type
  burglary:   entry_point, entry_method, premise, occupancy, search_pattern
  vehicle:    vehicle_class, ignition_method, location_type
  snatching:  vehicle_used, victim_activity, escape_direction
  atm:        machine_type, attack_method, alarm_defeated

narrative_text          original language, verbatim
narrative_lang
embedding               1024-dim float32
field_provenance        per field: "source" | "llm_extracted" | null
pii_ref                 pointer only — never inline
```

Two fields are load-bearing:

- `field_provenance` — an analyst must distinguish an MO value recorded in the
  original FIR from one an LLM inferred from narrative text.
- `pii_ref` — makes the access boundary structural, not a promise. See §7.

### 3.2 DynamoDB

```
cases      PK case_id           SK "META"
           GSI1  PK crime_type#year_quarter    SK case_id
pii        PK case_id           -- separate table, separate CMK, separate role
links      PK case_id_a         SK bits_desc#case_id_b
           attrs: bits, pair_class, contributions[], retrieval_rank
audit      PK actor_id          SK iso_timestamp
feedback   PK pair_id           SK actor#timestamp
```

`links` sorts on score, so "top 10 for this case" is one range query, no compute.

---

## 4. The scoring model

### 4.1 What is computed

Posterior odds that two cases share an offender, in bits:

```
posterior(bits) = prior(bits) + Σ evidence weights(bits)
```

Per field, two quantities:

- `u` — how often two *unrelated* cases agree on this value (= its frequency)
- `m` — how often two cases by the *same* offender agree on it

```
agreement    weight = log2( m / u )
disagreement weight = log2( (1-m) / (1-u) )
```

### 4.2 Deriving m

Offender preference `θ ~ Dirichlet(α·p)`. Then:

```
m_v = (α·p_v + 1) / (α + 1)
```

Limits check: `α→∞` gives `m_v = p_v` (offender = population). `α→0` gives
`m_v = 1` (deterministic). Both correct.

For multi-label tags with offender rate `~ Beta(αp, α(1-p))`:

```
m_tag = (1-p)/(α+1) + p
```

### 4.3 Worked example — a STRONG match, not a typical one (burglary, α = 7.05)

This pair was constructed to show the mechanism. **It is not representative**,
and the total below must never be quoted as what a link scores. Measured on
held-out offenders (45k corpus, repeat_rate 0.5, `results/`):

| | median true pair | p90 | this example |
|---|---|---|---|
| residential burglary | +0.33 bits | +4.91 | +5.85 (top 7.7%) |
| commercial burglary | +1.49 | +6.53 | top 11.2% |
| vehicle theft | +0.66 | +4.16 | top 3.6% |
| snatching | +0.45 | +2.99 | top 0.5% |
| ATM tampering | +0.00 | +3.65 | top 1.9% |

Most linked pairs carry about a bit; a few carry six. Finding those few in a
haystack is exactly a ranker's job — and why §11's base-rate limit stands.


| field | value | u | m | bits |
|---|---|---|---|---|
| entry_point | roof | 0.06 | 0.177 | **+1.559** |
| entry_method | lock_broken | 0.45 | 0.518 | +0.204 |
| premise | independent_house | 0.42 | 0.492 | +0.228 |
| occupancy | *disagrees* | 0.58 | 0.632 | −0.191 |
| time_band | night | 0.52 | 0.580 | +0.157 |
| search_pattern | selective | 0.28 | 0.369 | +0.400 |
| counter_forensic | cctv_disabled | 0.03 | 0.150 | **+2.327** |
| group_size | 2-3 | 0.48 | 0.545 | +0.182 |
| tool | cutter | 0.16 | 0.264 | +0.724 |
| property | gold | 0.61 | 0.658 | +0.110 |
| property | cash | 0.54 | 0.597 | +0.145 |
| | | | | **+5.845** |

Two rare agreements carry 3.9 of the 5.8 bits. Rarity weighting **is** the
signal — which is also why the typical pair, agreeing only on common values,
lands near a bit.

### 4.4 Prior — derived from pool composition

Burglary only:
```
true pairs = 400 offenders × C(4,2)  = 2,400
all pairs  = C(10,000, 2)            = 49,995,000
prior      = log2(2400 / 49,992,600) = −14.35 bits
```

Widened to all property crime (~40k cases, ~1600 offenders):
```
true pairs = 1600 × C(4,2)           = 9,600
all pairs  = C(40,000, 2)            = 799,980,000
prior      = log2(9600 / 799,970,400) = −16.35 bits
```

**Widening costs ~2 bits of prior.** Cross-type pairs also score on `mo_core`
only — 8 fields instead of 13. Less evidence against a worse prior.

Consequence for UX: default the search to same-type. Offer "widen to all property
crime" as an explicit analyst action, so the prior is conditioned on their choice
rather than flooding them by default.

### 4.5 Estimating u and m per pair class

`u` must be computed **within the relevant pool** — "night" has a different
frequency for burglary than for vehicle theft. Athena query groups by
`crime_type`; cross-type pairs use pooled property-crime frequencies.

`m` must be estimated **separately for cross-type pairs**. An offender may work
at night for burglary but daytime for snatching, so same-type `m` overestimates
cross-type agreement. Fit from labelled cross-type pairs only.

Report the two pair classes separately in all metrics. Never blend them.

### 4.6 Critical design rule

**Never block on a feature you also score.** Filtering by roof entry and scoring
roof entry are the same operation — verified numerically: pre-filtering to
roof-entry cases improves the prior by exactly +1.559 bits, which is exactly the
roof agreement weight.

Therefore: **scoring uses structured MO only**, and nothing blocks on a scored
field.

*Revised 2026-09-18 after measurement.* The original design gated scoring
behind a narrative-embedding top-50. Measured, that gate matched on place
text (every narrative opens with district, station and dates) and retrieved
**0 of 1,094 cross-state partners**; with the place text removed it was
barely above random, and the two-stage pipeline scored hit@10 0.045 against
the MO scorer's 0.128 on the full pool. Full-pool scoring of all 44,533 cases
takes ~9 minutes on one core, so the gate is unnecessary at this scale.
LinkBatch now scores each case against its entire crime-type pool (crime type
defines the pool; it is not scored). Any gate needed at national scale must
not encode place. See FINDINGS.md §8.

### 4.7 Model choice

Fellegi-Sunter weights, then logistic regression for residual correction, then
isotonic calibration on held-out offenders.

No neural scorer. With ~9,600 positive pairs and ~13 features that's ~700 events
per parameter — LR is comfortably supported, and a deeper model would destroy the
per-feature bit contributions the product depends on.

Clustering: connected components over edges above a **deliberately high**
threshold. Transitive chaining (A–B linked, B–C linked, A–C unrelated) otherwise
produces bogus mega-clusters. Report mean intra-cluster score alongside membership.

---

## 5. Models used

| Purpose | Model | Notes |
|---|---|---|
| MO extraction | `anthropic.claude-haiku-4-5` | Strict JSON, per-field confidence, explicit null |
| Embedding | `cohere.embed-multilingual-v3` | 1024-dim, 100+ languages, ap-south-1 |
| Pair scoring | FS weights + LogisticRegression | ~30 floats, shipped as JSON |
| Calibration | Isotonic, held-out by offender | Raw LR overconfident at this imbalance |
| Explanation prose | `anthropic.claude-haiku-4-5` | Writes *from* computed weights, never decides |

Cohere multilingual means a Marathi FIR and a Kannada FIR embed into the same
space with **no translation step**. Cross-lingual linking is a demo highlight.

Alternative embedding: `amazon.titan-embed-text-v2:0`, also ap-south-1,
configurable to 256 dims — useful for the memory math in §6.

**Deliberately not used:** SageMaker (model is 30 coefficients, trains in 3s),
OpenSearch Serverless (OCU floor ≈ $350/mo for a 50k-row index), SQS (Step
Functions Map already handles retries). State this in the submission.

---

## 6. Pipeline

```
EventBridge (S3 arrival | nightly)
  └─ Step Functions
       ├─ Normalise        Map over state files, adapter YAML per state
       ├─ Enrich           Map, MaxConcurrency 5 (Bedrock TPM limits)
       ├─ BuildIndex       single .npy to S3
       ├─ ComputeFreq      Athena → u-values per crime_type
       ├─ LinkBatch        Map over 1k-case chunks
       └─ PublishMetrics   CloudWatch
```

**Chunking is mandatory.** A full 50k × 50k similarity matrix is 10 GB. Process
1,000 query rows at a time against all 50k, keep top 50, discard the block:
~200 MB resident, 50 chunks, finishes inside a 15-minute Lambda.

Vector memory: 50k × 1024 × 4 B = 205 MB. At Titan's 256-dim setting, 51 MB.
Either fits a 1 GB Lambda.

Adapters are **config, not code** — adding a state is a YAML diff, not a deploy.

---

## 7. AWS services and IAM

| Service | Role in system |
|---|---|
| S3 | Raw feeds, adapter YAMLs, curated parquet, vectors, model coefficients |
| Lambda | Normalise, enrich, batch link, API handlers |
| DynamoDB | 5 tables per §3.2 |
| Bedrock | Haiku 4.5, Cohere embeddings |
| Step Functions | Pipeline orchestration, Map states, retries |
| EventBridge | S3 trigger + nightly schedule |
| API Gateway (HTTP API) | REST surface, Cognito JWT authorizer |
| Cognito | Officer identity → Cedar principal + audit actor |
| Verified Permissions | Managed Cedar, policies visible in console |
| KMS | Separate CMK on `pii` table only |
| Athena + Glue | SQL over parquet → u-frequencies |
| Amplify Hosting | Frontend |
| CloudWatch | Logs + linkage metrics dashboard |
| SAM | One-command deploy |

### IAM boundaries

| Role | Can reach | Cannot reach |
|---|---|---|
| `IngestRole` | S3 raw/curated, `cases` | `pii`, Bedrock |
| `EnrichRole` | Bedrock, `cases`, S3 vectors | `pii` |
| `LinkRole` | `cases`, vectors, `links` | `pii`, Bedrock |
| `ApiRole` | `cases`, `links`, `audit`, `feedback`, AVP | `pii` directly |
| `PiiRole` | `pii`, KMS decrypt | everything else |

`PiiRole` is assumable **only** after Verified Permissions returns ALLOW on a
request carrying a stated reason. The claim "the linkage engine cannot see
personal data" is an IAM boundary plus a separate KMS key — it holds even if the
Lambda code has a bug.

---

## 8. API

```
GET  /cases/{id}                          canonical record, no PII
GET  /cases/{id}/links?scope=same|all     ranked, with bit contributions
GET  /clusters/{id}                       members + cohesion
POST /cases/{id}/pii-request              AVP-evaluated, reason required
POST /links/{pair_id}/feedback            confirmed | rejected
```

`scope` implements §4.4 — same-type by default, cross-type on request.

`feedback` is 20 minutes of work and gives you both future training labels and a
demonstrable human-in-the-loop.

---

## 9. Evaluation

Split **by offender**, never by case, or the same offender's crimes leak across
train and test.

| Metric | Why |
|---|---|
| recall@50 | Retrieval stage — did the true partner survive blocking? |
| precision@10 | Scoring stage — what does the analyst's shortlist look like? |
| PR-AUC | Honest under extreme imbalance; ROC-AUC flatters here |
| Reported per pair_class | same-type and cross-type never blended |

**Headline plot:** linkage AUC vs `repeat_rate`, with the −14.35 bit
line drawn on it. Shows exactly how behaviourally consistent an offender must be
before the system can find them with geography removed. That's a finding, not a
demo.

---

## 10. UI

Never render a probability. A strong match posteriors at ~0.27% and a UI showing
that looks broken to anyone who hasn't followed the derivation.

Render instead:

```
rank 2 of 10,482   ·   +5.85 bits          ← a strong match (top ~8%)
driven by: roof entry, disabled CCTV, cutter marks

rank 7 of 10,482   ·   +1.10 bits          ← what most true links look like
driven by: night, 2-3 persons
```

Design the panel for the second one. A shortlist of median-strength links is
the normal case; show the bit total next to the pool's typical range so an
analyst can tell a rare-value match from an ordinary one.

Contribution bar chart: supporting evidence right, opposing left, marker at the
break-even bit line so the analyst sees how far short of certainty the evidence
falls.

---

## 11. Known limits — state these, don't hide them

1. **Positive-unlabelled problem.** Unlabelled pairs aren't confirmed non-links;
   they may be links by an uncaught offender. Negatives are contaminated.
2. **Selection bias.** Training on solved cases learns the MO of offenders who
   got caught.
3. **Signal ceiling.** FIRs are written for legal sufficiency, not behavioural
   detail. ViCLAS-style systems use 100+ analyst-coded fields; an FIR carries a
   fraction of that.
4. **Base rate.** MO evidence alone cannot clear a −14 to −16 bit prior. The
   system is a ranker, not a classifier. Extra bits must come from *outside* the
   MO distribution — recovered property serials, pawnshop records, vehicle
   sightings — not from more MO columns. Measured: true pairs average +0.4 to
   +2.1 bits, median +0.0 to +1.5 (§4.3).
5. **Extraction quality is not the bottleneck; information is.** Replacing LLM
   extraction with the true values for the free-text state moves same-type
   hit@10 from 0.136 to 0.158 — 2.2 points. Effort spent on extraction prompts
   or extra passes buys almost nothing; the ceiling is in how few bits an FIR's
   MO carries. (`results/pre_core_fix/eval_oracle.json`)
6. **Protected attributes excluded by design.** No caste, religion or community
   fields. Say so explicitly.
7. **Serious crime out of scope.** The method extends there, and international
   systems like ViCLAS operate there, but it needs analyst-coded input, tighter
   access control, and a legal basis for sensitive inter-state sharing. Scoped to
   property crime where the data is honest and a false link is recoverable.

---

## 12. Build order

| When | What |
|---|---|
| First | Enable Bedrock model access in ap-south-1 (per-model, not instant) |
| First | `sam deploy` an empty stack end-to-end before it holds anything |
| Day 1 | Adapters + normalisation, multi-type schema frozen |
| Day 2 | Link batch job + evaluation. Get recall@50 on paper before touching UI |
| Day 3 | UI + contribution chart |
| Day 4 AM | Verified Permissions, audit, video |

Cut order if behind: clustering → cross-type → feedback endpoint → Cedar.
Never cut: normalisation demo, scorer, contribution panel, consistency sweep.
