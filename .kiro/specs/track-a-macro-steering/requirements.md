# track-a-macro-steering — Requirements

## Introduction

This spec wires the released `recall_guard` library (pinned `@v0.1.0`) into **Track A**
(`macro_framework/llm_agent.py`, `notebooks/09_track_a_llm_agent.ipynb`) and adds two
consecutive analysis playbooks (`11_…`, `12_…`) that reuse the existing macro data. The aim
is **not predictive factors / alpha**. It is to (a) characterise the macro panel
**statistically** and use that to **steer** the agent, and (b) **refine the agent's prompts**
to pursue the same — both under **point-in-time (PIT)** inference, with a **measured**
memorization/contamination score (`p_memorized`) attached to every LLM decision.

Track A already avoids lookahead *qualitatively* (anonymized assets, rolling z-scores,
`walk_forward.py` slicing strictly before each rebalance date). recall_guard adds the
*quantitative* half. Because the agent's DSPy → OpenRouter (Claude) path does not expose
per-token logprobs, the score is produced by a **separate logprob-bearing inference path on
the same PIT prompt**; the agent's own call is unchanged. The contamination score and the
macro statistics steer the agent's Black-Litterman view confidence; success is judged by the
existing head-to-head evaluation plus the contamination metric, never by forecast accuracy.

## Boundary Context

- **In scope**: declaring recall_guard as a pinned dependency; a PIT scoring layer that
  attaches `p_memorized` to each Track A decision via a separate inference path; a statistical
  characterization of the macro panel (playbook `11_…`) used to steer the agent; macro- and
  contamination-driven shaping of Black-Litterman view confidence (a new Track A variant); a
  prompt-refinement playbook (`12_…`) that compares prompt versions by measured contamination
  under PIT; integration into the existing head-to-head evaluation; an append-only research
  log.
- **Out of scope**: any predictive-return / alpha objective; changing the Baseline, Track B,
  or the head-to-head evaluation contracts; new external data sources beyond the existing FRED
  macro panel + parquets; modifying the `recall_guard` library; exposing logprobs from
  DSPy/OpenRouter (the agent path stays as-is).
- **Adjacent expectations**:
  - Live scoring depends on the recall_guard NIM endpoint and a valid NIM credential at run
    time; the spec does not own credential provisioning.
  - Reuses `macro_framework/{walk_forward,macro,anonymize,evaluation}.py` and
    `notebooks/06,09,10` behaviour without changing them.
  - The agent continues to run on its existing DSPy/OpenRouter path for decisions.
- **Locked decisions (2026-06-25, from the project description)** — recorded as constraints;
  the acceptance criteria stay tool-neutral:
  - Dependency: `recall_guard @ v0.1.0` (git), via `recall_guard.MemoryGuardedScorer`.
  - Scoring runs as a **parallel NIM call on the same PIT prompt** (no DSPy logprobs hack).
  - New work is **additive** (playbooks `11_`/`12_`, new modules/artifacts); existing
    notebooks `01–10`, code, and `data/` are untouched; the research log is append-only.
  - Objective is statistically-grounded, low-contamination steering — **not** alpha.

## Requirements

### Requirement 1: recall_guard dependency and PIT memorization scoring

**Objective:** As a Track A maintainer, I want every agent decision scored for measured
memorization on the same point-in-time prompt, so that contamination is observable rather than
assumed.

#### Acceptance Criteria

1. The project shall declare `recall_guard` as a version-controlled dependency pinned at
   `v0.1.0`.
2. When Track A produces views for a rebalance date, the scoring layer shall compute a
   `p_memorized` in `[0, 1]` for that decision from the same anonymized, z-scored prompt
   content used for the decision.
3. The scoring layer shall obtain its logprob-bearing inference from a separate inference path,
   leaving the agent's existing decision call unchanged.
4. While scoring a rebalance, the scoring layer shall use only information available as-of that
   rebalance date, consistent with the existing walk-forward discipline.
5. If the scoring credential is missing or rejected, the scoring layer shall surface a clear
   configuration error rather than returning an unscored or silently invalid result.
6. Where scoring is unavailable or disabled, the original Track A view outputs and prompt shall
   be unaffected (scoring is additive).
7. While the scoring model's calibrator is weak or uncalibrated, the scoring layer shall
   surface that status rather than steer on an unvalidated score.

### Requirement 2: Point-in-time statistical characterization of the macro panel

**Objective:** As a researcher, I want a point-in-time statistical characterization of the
macro panel, so that steering is grounded in the macro data's structure rather than free
association.

#### Acceptance Criteria

1. The analysis playbook (`11_…`) shall produce, for each rebalance date, a regime label and a
   per-series point-in-time z-score summary of the macro panel.
2. When the analysis computes any statistic attributed to a decision date, it shall exclude
   observations dated after that date.
3. The analysis shall reuse the existing macro panel and existing macro artifacts without
   regenerating them.
4. The analysis shall emit per-rebalance steering signals (regime labels and/or per-view
   confidence adjustments) consumable by Track A.
5. The analysis shall not define a forecasting target.

### Requirement 3: Macro- and contamination-steered Track A variant

**Objective:** As a Track A maintainer, I want the macro statistics and the measured
`p_memorized` to shape view confidence, so that views inconsistent with the macro state or
showing high contamination are down-weighted.

#### Acceptance Criteria

1. When converting agent views to Black-Litterman inputs, the steering layer shall adjust each
   view's confidence as a function of the measured `p_memorized` (confidence falls as
   `p_memorized` rises).
2. Where a view's measured contamination exceeds the configured threshold, the steering layer
   shall down-weight or exclude that view.
3. The steered variant shall produce its own portfolio targets and decision log, leaving the
   original Track A targets and decision log unchanged.
4. The steering layer's only effect shall be confidence/inclusion shaping grounded in the
   macro statistics and the contamination score.
5. The steering layer shall not optimize a predictive-return objective.

### Requirement 4: PIT prompt refinement compared by measured contamination

**Objective:** As a researcher, I want to compare candidate agent prompts by their measured
memorization under PIT, so that I can select prompts that reason from the macro statistics with
lower contamination.

#### Acceptance Criteria

1. A new playbook (`12_…`) shall evaluate multiple agent prompt versions over the same
   point-in-time prompt stream.
2. For each prompt version, the playbook shall report the `p_memorized` distribution and the
   view-stability metrics.
3. When comparing prompt versions, the playbook shall report the head-to-head evaluation deltas
   so a refinement is accepted only when it does not degrade the existing evaluation.
4. The playbook shall preserve prior prompt versions (versioned, additive) rather than
   overwriting them.
5. Where a refined prompt is adopted, its head-to-head metrics shall be no worse than the
   current prompt's.

### Requirement 5: Evaluation with a non-predictive success definition

**Objective:** As a stakeholder, I want the steered/refined variant measured by the existing
head-to-head framework plus the contamination metric, so success is defined without a forecast
objective.

#### Acceptance Criteria

1. The new variant shall be evaluable by the existing head-to-head metrics (return, volatility,
   Sharpe/Sortino/Calmar, max drawdown, crisis analytics, turnover, view stability) alongside
   Baseline / Track A / Track B.
2. The evaluation shall additionally report the measured `p_memorized` distribution for the
   variant.
3. The variant's success shall be defined as lower-or-equal measured contamination with
   non-degraded head-to-head metrics — not as improved forecast accuracy.

### Requirement 6: Additive, append-only, data-reusing delivery

**Objective:** As a maintainer, I want the work to be strictly additive, so existing results
stay reproducible.

#### Acceptance Criteria

1. The work shall not modify existing notebooks `01–10`, existing modules, or existing `data/`
   artifacts.
2. New playbooks shall follow the existing numbering convention as `11_…` and `12_…`.
3. The research log shall be append-only: new dated entries are added and earlier entries are
   left unchanged.
4. New portfolio-target, decision-log, score, and macro steering-signal artifacts shall be
   written under new filenames rather than overwriting existing artifacts.
