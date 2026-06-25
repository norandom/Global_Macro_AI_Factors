# Research & Design Log — track-a-macro-steering

> Append-only. The research log grows in one direction: add new dated entries at the bottom;
> never rewrite earlier entries. Each entry is a baseline snapshot or a decision/finding.

## Summary
- **Feature**: `track-a-macro-steering`
- **Discovery Scope**: Extension of an existing system (Global_Macro_AI_Factors) + new analysis
  playbooks; integrates the external `recall_guard` library.
- **Key Findings (2026-06-25)**:
  - PIT discipline already exists structurally (`walk_forward.py` slices `< rebalance_date`;
    rolling 60-month z-scores, not full-sample).
  - Track A's LLM path (DSPy → OpenRouter Claude) does not expose logprobs; the memorization
    score must come from a parallel NIM call on the same prompt.
  - The macro signal is sparse/coarse: ~72 unique monthly macro states over 2019–2024.

---

## 2026-06-25 — Baseline capture (existing state)

_Source: codebase scout of `/home/mc/projects/Global_Macro_AI_Factors` (commit on `main`)._

### Track A — `macro_framework/llm_agent.py`, `notebooks/09_track_a_llm_agent.ipynb`
- `LlmMacroAgent.views_for_state(macro_state: dict, asset_snapshot: list) -> (list[MacroView], reasoning_text)`.
  - `macro_state = {cpi_yoy_z, t10y2y_z, hy_oas_z}` (rolling 5y z-scores, monthly, rounded ~2dp).
  - `asset_snapshot = [{id: "Asset_A", category, trailing_12m_return, trailing_vol_ann}, …]` — no ticker/date.
  - `MacroView(asset_long, asset_short|None, expected_excess_annualized, confidence∈~0.4–0.6, rationale)`.
- `views_to_bl(views, real_symbols) -> (P, Q)`; `Q = (expected_excess * confidence) / 252`.
- LLM: DSPy 3.1.3 → OpenRouter `claude-sonnet-4.6`, temperature 0, max_tokens 1024, diskcache keyed
  by `(prompt_version, macro_state, asset_snapshot)`. Prompt text = `AGENT_INSTRUCTIONS` + the
  `MacroViewSignature` docstring. **Logprobs NOT exposed.**
- Notebook 09: 72 monthly rebalances (2019–2024); base = HRP-CVaR (BIL pinned 25%); posterior = BL-MV;
  final blend 0.7·HRP + 0.3·BL; simulate via vectorbt; persists targets/equity/`track_a_agent_log.json`.

### Macro data + stats — `macro_framework/macro.py`, `notebooks/06_macro_zscores.ipynb`
- `build_macro_panel(series_map, freq="ME", zscore_window=60)`; `rolling_zscore(s, window=60, min_periods≈30)`.
- Series: CPIAUCSL→cpi_yoy, T10Y2Y→t10y2y, BAMLH0A0HYM2→hy_oas (+ `_z`). ~196 months (2010–2026).
- `data/macro_panel_monthly.parquet` holds raw + z columns. Existing stats: |z| overlay vs drawdown (nb06);
  Track B rule-based regimes (nb08). **No existing macro→agent steering / causal regression.**

### PIT / walk-forward — `macro_framework/walk_forward.py`
- `build_walk_forward_targets(prices, rebalance_dates, weight_fns, macro_panel, lookback_days=756, min_history=60)`.
- At each `rb`: `prices[index < rb].tail(lookback)`, `macro_panel[index < rb]` → `StrategyContext`. Lookahead
  is structurally impossible (slicing, not discipline). Monthly rebalance = first trading day.

### Anonymization — `macro_framework/anonymize.py`
- `AssetMap` fixed: SWDA.L↔Asset_A (world_equity), XLK↔Asset_B (tech), IAU↔Asset_C (gold), BIL↔Asset_D (cash).
  `pseudo_assets()` → `{id, category}` only. Qualitative recall-avoidance.

### Evaluation — `macro_framework/evaluation.py`, `notebooks/10_head_to_head_evaluation.ipynb`
- Metrics across Baseline / Track A / Track B: total/annualized return, vol, Sharpe/Sortino/Calmar, max DD,
  crisis (2022) analytics, anticipation lead time (BIL+IAU ≥ 40%), turnover, Track-A `view_stability()`.

### Conventions
- Python 3.12 (`<3.13`); deps include dspy==3.1.3, openai, riskfolio-lib, vectorbt, diskcache.
- Notebooks numbered 01–10 → new ones are `11_…`, `12_…`. `.llm_cache/` persists Claude responses.
- Regenerable: macro panel, baseline/track_a/track_b targets+equity (track_a needs cached LLM responses).
  Precious: `track_a_agent_log.json` (rationales).

### recall_guard integration surface (the new dependency)
- `MemoryGuardedScorer.calibrate(api_key, model, is_memorized: Sequence[str], oos_control: Sequence[str],
  reference_model=None, …)` → `.score(prompt)` / `.score_many(...)` → `GuardedScore(signal, p_memorized,
  memguard_confidence, features, fail_reason)`; `ConfigurationError` on missing/rejected NIM key.
- Recommended: **parallel NIM scorer** — after the DSPy call, score the identical anonymized, z-scored,
  PIT prompt on a NIM model; attach `p_memorized` per view; steer/gate BL confidence
  (`confidence * (1 - p_memorized)`) before `views_to_bl`. recall_guard's NIM endpoint is live.

## Open decisions / Research Needed (carry to requirements/design)
1. **IS/OOS calibration corpus.** recall_guard splits prompts by the NIM model's training cutoff
   (IS = before, OOS = after). For anonymized macro prompts, the split key is the macro state's as-of
   date. Candidate sources: mine the 2019–2024 agent log split by the chosen NIM model's cutoff; or
   synthesize macro-state prompts spanning pre/post cutoff. Risk: small/homogeneous corpus → unstable AUC.
2. **NIM model for scoring.** Which model (e.g. `meta/llama-3.1-8b-instruct`, `openai/gpt-oss-20b`)? Its
   cutoff defines the IS/OOS split and the cost/latency (~72 calls, then cached).
3. **Steering objective (a).** Confirm: steer toward statistically-grounded, low-`p_memorized` views, NOT
   alpha. Define the concrete steering levers (input shaping vs confidence post-processing) and a
   non-predictive success metric (e.g. lower `p_memorized` + unchanged/improved head-to-head, stable views).
4. **Prompt-refinement loop (b).** How to compare prompt versions: `p_memorized` distribution + view
   stability + head-to-head deltas, under PIT. Avoid overfitting prompts to the eval window.
5. **Z-score quantization.** Rounding z to 2dp can flip views at threshold edges; consider binning
   (e.g. 0.5σ) for prompt/cache stability — design decision.
6. **Confidence-adjustment function.** memguard linear `conf*(1-p_mem)` vs a gate vs sigmoid — pick and
   justify in design.
