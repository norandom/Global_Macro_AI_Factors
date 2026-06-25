# Requirements Document

## Project Description (Input)

Wire the released `recall_guard` library (github.com/norandom/memguard_alpha, pinned
`@v0.1.0`) into **Track A** of this project (`macro_framework/llm_agent.py`,
`notebooks/09_track_a_llm_agent.ipynb`) and add two **consecutive new playbooks**
(notebooks `11_…` and `12_…`) that **reuse the existing macro data**. The work is
explicitly **not** about predictive factors / alpha. It is about (a) doing **statistical
analysis on the macro data to steer** the agent, then (b) **refining the agent's prompts**
to pursue the same — with **point-in-time (PIT) inference** and a **measured** memorization
score on every LLM call.

### Why

- Track A is a zero-temperature DSPy agent (OpenRouter → Claude) that emits Black-Litterman
  view triples from an **anonymized, rolling-z-scored** macro state (`cpi_yoy_z`, `t10y2y_z`,
  `hy_oas_z`) and four pseudo-assets (`Asset_A..D`). It already practices recall-avoidance
  **qualitatively** (no dates, no years, no real tickers; `walk_forward.py` slices strictly
  `< rebalance_date`).
- `recall_guard` adds the **quantitative** half: a per-prompt `p_memorized` contamination
  score derived from per-token logprobs. DSPy/OpenRouter hides logprobs, so the score is
  produced by a **parallel NIM call on the same PIT prompt** (the agent stays on Claude).
- Framing to preserve: near-coin-flip directional accuracy is the correct result; the product
  is the honesty/contamination signal and statistically-grounded steering, not forecasting.

### Phase (a) — statistical analysis to steer

Characterise the macro panel (`data/macro_panel_monthly.parquet`) statistically — regime
structure, z-score distributions/thresholds, cross-series and macro→asset relationships,
stability of the rolling-z signal — **point-in-time** (as-of each decision date, no
full-sample lookahead). Use that characterisation to **steer** Track A: shape the macro-state
inputs and/or post-process the agent's view confidence so views are grounded in the macro
statistics rather than in (possibly memorised) free association. New playbook: `11_…`.

### Phase (b) — refine prompts to pursue the same, under PIT

Refine the agent's prompt/instructions so the agent itself reasons from the phase-(a)
statistical framing, while keeping PIT discipline. Each candidate prompt is scored with
`recall_guard.MemoryGuardedScorer` (parallel NIM on the identical PIT prompt) to compare
`p_memorized` across prompt versions — pick prompts that lower measured contamination
without degrading the existing head-to-head evaluation. New playbook: `12_…`.

### Integration surface (from the codebase scout — see research.md)

- `MemoryGuardedScorer.calibrate(api_key, model, is_memorized, oos_control, …)` →
  `.score(prompt)` / `.score_many(...)` → `GuardedScore(signal, p_memorized,
  memguard_confidence, …)`; `ConfigurationError` on missing/rejected NIM key.
- Attachment point: after Track A's DSPy call, run the scorer on the same anonymized,
  z-scored, PIT prompt; attach `p_memorized` per view and steer/gate BL confidence (e.g. the
  memguard discount `confidence * (1 - p_memorized)`) before `views_to_bl`.

### Hard constraints

- **Append-only / non-destructive.** Do not overwrite existing notebooks (`01–10`), code, or
  `data/` artifacts. New work is additive (`11_…`, `12_…`, new modules); the **research log
  grows in one direction** (`research.md` is append-only).
- **Reuse the data.** Build on `data/macro_panel_monthly.parquet` and the existing parquets
  (`ssr_scores`, `baseline_*`, `track_a_*`, `track_b_*`); do not regenerate them as part of
  this work.
- **PIT inference.** Macro state and every scored prompt must be as-of the decision date
  (reuse the `walk_forward.py` `< rebalance_date` discipline + rolling z-scores). No
  full-sample leakage.
- **Dependency.** Add `recall_guard` as a `uv` git dependency pinned `@v0.1.0`; do not modify
  `recall_guard` (it is a released dependency). Python `>=3.12,<3.13` (recall_guard supports
  3.12).
- **No DSPy logprobs monkey-patch.** Logprobs come from the parallel NIM call, not by hacking
  DSPy/OpenRouter.

### Non-goals

- No predictive-alpha objective; do not tune toward forecast accuracy.
- No change to the Baseline, Track B, or the head-to-head evaluation contracts; new work plugs
  into the existing `evaluation.py` metrics.
- No new data sources beyond what the repo already has (FRED macro panel + existing parquets).

language: en

## Requirements
<!-- Will be generated in /kiro-spec-requirements phase -->
