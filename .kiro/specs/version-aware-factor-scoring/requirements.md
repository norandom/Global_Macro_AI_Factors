# Requirements Document

## Introduction

This spec makes **AI macro factors** — continuous, relative, structured exposures/characterizations —
the unit of work for Track A, and measures contamination in a **version-aware** way so candidate
prompts can be compared and refined by their measured memorization. It is a follow-up to
`track-a-macro-steering` and builds additively on that steering engine and the released contamination
library it depends on.

The MVP delivers exactly two factor archetypes — a **regime-as-loadings** characterization and a
**BL-tilt-as-exposure** reframe of the agent's views — plus a **version-aware** contamination scorer
that scores each prompt version's own factor prompt and reasoning (not a version-independent,
input-only directional prompt), and an **honesty-adjusted exposure** that down-weights a factor by its
measured contamination. Success is non-predictive throughout: factors are characterizations, never
forecasts.

## Boundary Context

- **In scope**: a version-aware contamination scorer that distinguishes prompt versions; the two MVP
  factor archetypes (regime-as-loadings, BL-tilt-as-exposure); the honesty-adjusted exposure
  (raw loading discounted by measured contamination); a prompt-refinement playbook that compares
  versions by version-aware contamination + factor stability + head-to-head; an append-only research
  log.
- **Out of scope**: any predictive-return / alpha objective; the deferred (future) factor archetypes —
  cross-asset macro-exposure **β matrix**, **thematic-intensity** factors, **cross-sectional
  regime-alignment** ranking, and **macro-dispersion / causal-contradiction tail** factors; changing
  the Baseline or Track B contracts; modifying the contamination library or its directional scoring
  façade; new external data sources beyond the existing FRED macro panel and the FMP calibration corpus.
- **Adjacent expectations**:
  - Depends on the released contamination library's public membership-inference primitives and a
    valid scoring credential at run time; the spec does not own credential provisioning.
  - Reuses the existing point-in-time slicing, rolling z-scores, anonymized assets, the Black-Litterman
    view-to-input conversion, and the head-to-head evaluation without changing them.
  - Live dependencies (NIM scoring, FMP corpus, OpenRouter agent) are available; the project's price
    database is not provisioned in this environment, so notebooks substitute a public price source
    (documented), and price-dependent artifacts are not committed.
- **Locked decisions (2026-06-26, from the project description)** — recorded as constraints; the
  acceptance criteria stay tool-neutral:
  - Contamination is measured via the released library's **public MIA primitives**
    (`NvidiaLM` logprobs → `compute_mia_features` → `MCSCalibrator.predict_proba`, with
    `build_baseline` / `train_mcs` for calibration), **bypassing the directional `MemoryGuardedScorer`
    façade** (no `direction + confidence` parse required).
  - Reuse the validated calibrator: `meta/llama-4-maverick-17b-128e-instruct` @ cutoff `2024-08-01`
    (holdout_auc ≈ 0.9, `is_weak=False`).
  - Named macro axes for the regime-loadings factor: inflation pressure, growth/cycle, credit/liquidity
    stress, policy stance, risk appetite.
  - Honesty-adjusted exposure = `raw_loading × (1 − p_memorized)`.
  - Build additively on `macro_framework/steering.py` via new symbols/modules; new playbooks continue
    the existing numbering (`13_…`, `14_…`).

## Requirements

### Requirement 1: Version-aware contamination scoring of factor prompts

**Objective:** As a researcher, I want each prompt version's contamination measured on that version's
own factor prompt and reasoning, so that prompt versions are distinguishable by measured memorization.

#### Acceptance Criteria

1. When a prompt version produces a factor output for a rebalance date, the scoring layer shall compute
   a `p_memorized` in `[0, 1]` from that version's own prompt content and the model's emitted factor
   reasoning.
2. When two distinct prompt versions are scored over the same point-in-time macro state, the scoring
   layer shall be able to produce different `p_memorized` values for the two versions.
3. The scoring layer shall measure contamination without requiring the model's response to be a
   parseable buy/sell direction with confidence.
4. While scoring a rebalance, the scoring layer shall use only information available as-of that
   rebalance date.
5. If the scoring credential is missing or rejected, the scoring layer shall surface a clear
   configuration error rather than returning an unscored or silently invalid result.
6. While the contamination calibrator is weak or uncalibrated, the scoring layer shall surface that
   status rather than reporting a contamination value as if validated.

### Requirement 2: Regime-as-loadings macro factor

**Objective:** As a researcher, I want the agent to characterize the macro state as continuous loadings
on named macro axes, so that the regime is represented as a factor vector rather than a directional bet.

#### Acceptance Criteria

1. For each rebalance date, the factor layer shall produce a regime-loadings vector with one continuous
   loading per named macro axis, each bounded in `[-1, +1]`.
2. The regime-loadings output shall not contain a buy/sell direction or an expected return.
3. When computing the loadings for a rebalance date, the factor layer shall use only macro observations
   dated before that date.
4. The factor layer shall emit the per-rebalance regime-loadings vector as a consumable artifact keyed
   by rebalance date.
5. The factor layer shall not define a forecasting target.

### Requirement 3: BL-tilt-as-exposure factor

**Objective:** As a Track A maintainer, I want each view expressed as a dimensionless exposure tilt
rather than a return forecast, so that Black-Litterman is driven by macro exposures, not predictions.

#### Acceptance Criteria

1. The factor layer shall express each per-asset view as a dimensionless exposure tilt rather than an
   expected return.
2. When converting exposure tilts to Black-Litterman inputs, the steering layer shall derive each
   view's magnitude as the tilt scaled by a conviction term that is dimensionless and not
   return-bearing, rather than as a forecast return.
3. The exposure-tilt conversion shall reuse the existing Black-Litterman view-to-input conversion
   without modifying it.
4. The exposure-tilt factor shall not optimize a predictive-return objective.

### Requirement 4: Honesty-adjusted exposure

**Objective:** As a researcher, I want each factor down-weighted by its measured contamination, so that
recall-tainted reasoning is discounted while genuine inference is retained.

#### Acceptance Criteria

1. When two factors have equal raw loadings, the factor layer shall produce a lower-or-equal
   honesty-adjusted exposure for the factor with the higher measured `p_memorized`.
2. Where the measured `p_memorized` is zero, the honesty-adjusted exposure shall equal the raw loading.
3. If the contamination score is unavailable or the calibrator is weak, the factor layer shall leave the
   raw exposure unadjusted rather than applying an unvalidated discount.
4. The honesty adjustment shall affect only exposure magnitude and shall not introduce a return
   objective.

### Requirement 5: Prompt refinement by version-aware contamination

**Objective:** As a researcher, I want to compare prompt versions by measured contamination, factor
stability, and head-to-head metrics, so that I can adopt prompts that reason with lower contamination at
no risk cost.

#### Acceptance Criteria

1. The refinement playbook shall evaluate multiple prompt versions over the same point-in-time stream
   and report each version's `p_memorized` distribution.
2. For each prompt version, the playbook shall report a factor-stability metric defined as the
   variability of that version's factor loadings across the point-in-time stream.
3. When comparing prompt versions, the playbook shall report the head-to-head evaluation deltas.
4. Where a refined prompt is adopted, its measured contamination shall be no greater than and its
   head-to-head metrics no worse than the incumbent prompt's.
5. The playbook shall preserve prior prompt versions rather than overwriting them.

### Requirement 6: Additive, append-only, non-predictive delivery

**Objective:** As a maintainer, I want the work strictly additive and non-predictive, so existing
results stay reproducible and success is defined without forecasting.

#### Acceptance Criteria

1. The work shall not modify the released contamination library, the existing modules, notebooks
   `01–12`, or existing data artifacts.
2. New playbooks and modules shall be added under new filenames following the existing
   numbering/naming conventions.
3. The research log shall be append-only: new dated entries are added and earlier entries are left
   unchanged.
4. The variant's success shall be defined as factor stability with no-greater measured contamination
   and non-degraded head-to-head metrics — not improved forecast accuracy.
5. The scoring and factor layers shall measure factor contamination without depending on the
   contamination library's directional-scoring façade (the architectural counterpart of the
   directional-parse-free requirement 1.3).
