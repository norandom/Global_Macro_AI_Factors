# Requirements Document

## Project Description (Input)

**Follow-up to the completed `track-a-macro-steering` spec** (same repo, `Global_Macro_AI_Factors`).
Build additively on its steering engine (`macro_framework/steering.py`) and the released
`recall_guard@v0.1.0` dependency.

### Who has the problem
The Track A researcher/maintainer who wants to produce **AI macro factors** (continuous, relative,
structured exposures/characterizations) and **refine the prompts that generate them by measured
memorization** — i.e., select prompts that reason from the macro statistics with lower contamination.

### Current situation (from track-a-macro-steering findings, 2026-06-26)
- The contamination score is **version-invariant**: the directional scoring prompt
  (`render_directional`) encodes only the macro **input** (state-only), so `p_memorized` is identical
  across prompt versions — the prompt-refinement playbook (nb12) could not compare prompts on the
  contamination axis and fell back to view-stability + head-to-head.
- Scoring is locked into a **directional-predictor frame**: the `MemoryGuardedScorer` façade requires a
  parseable `direction + confidence` response, which both forces a forecast-shaped task and, for
  `meta/llama-4-maverick`, fails to parse on ~80% of macro prompts (thin measurement coverage).
- Together these block the project's intent (b): "refine the agent's prompts to pursue
  statistically-grounded, low-contamination steering."

### What should change
1. **Macro-factor prompt taxonomy (not directional predictors).** Define prompts whose output is a
   structured, continuous, relative **factor representation** rather than a buy/sell direction —
   candidates: regime-as-loadings on named macro axes; cross-asset macro-exposure (β) matrices;
   thematic-intensity factors; cross-sectional regime-alignment scores; and the minimal Track A reframe
   of `MacroView.expected_excess_annualized` as a dimensionless **exposure tilt** feeding the BL `Q`
   via tilt × conviction. The LLM never returns a direction.
2. **Version-aware contamination scoring.** A new scorer that measures `p_memorized` on the
   **version-specific factor prompt + the model's actual factor reasoning/output**, using recall_guard's
   **public MIA primitives** (`NvidiaLM` logprobs → `compute_mia_features` → `MCSCalibrator.predict_proba`,
   with `build_baseline` / `train_mcs` for calibration) — **bypassing the directional parse gate** in the
   `MemoryGuardedScorer` façade. This makes `p_memorized` differentiate prompt versions and reflect
   factor-reasoning honesty. Reuse/extend the calibrator already validated for
   `meta/llama-4-maverick-17b-128e-instruct` @ cutoff `2024-08-01` (holdout_auc ≈ 0.9, `is_weak=False`).
3. **Prompt-refinement loop on measured contamination.** Compare prompt versions by version-aware
   contamination **and** factor stability **and** non-degraded head-to-head; the accept-gate selects the
   lower-contamination prompt at equal-or-better risk metrics.

### Constraints (must hold)
- **Additive only.** New module(s)/notebook(s) under new filenames; do NOT modify the released
  `recall_guard` library, the existing `macro_framework` modules (incl. `llm_agent.py` /
  `steering.py` / `evaluation.py`), notebooks `01–12`, or existing `data/` artifacts. Extend
  `steering.py`'s capabilities via new symbols/modules, not edits.
- **Non-predictive.** Factors are exposures/characterizations; success = factor stability +
  lower-or-equal measured contamination + non-degraded head-to-head (return, vol, Sharpe/Sortino/Calmar,
  max drawdown, crisis analytics, turnover) — never improved forecast accuracy. No directional signal is
  used as a return factor.
- **Point-in-time** throughout (reuse `walk_forward` `< rebalance_date` slicing + rolling z-scores;
  anonymized assets).
- **Append-only research log** (new dated entries only).
- **Reuse `recall_guard@v0.1.0` PUBLIC primitives only** (`NvidiaLM`, `compute_mia_features`,
  `MiaFeatures`, `ControlBaseline`, `build_baseline`, `standardise`, `MCSCalibrator`, `train_mcs`,
  `LOGPROB_FLOOR`) — no edits to that library.
- **Environment** (carry forward): NIM inference works (rotated key), FMP (ultra plan, stable API),
  OpenRouter (agent). The Postgres price DB is NOT provisioned here — notebooks substitute yfinance
  (documented); price-dependent artifacts are gitignored.

### Out of scope
- Any predictive-return / alpha objective.
- Changing the Baseline or Track B contracts.
- Modifying `recall_guard` or the directional `MemoryGuardedScorer` façade itself (this spec uses the
  lower-level public primitives instead).
- New external data sources beyond the existing FRED macro panel + the FMP calibration corpus.

## Requirements
<!-- Will be generated in /kiro-spec-requirements phase -->
