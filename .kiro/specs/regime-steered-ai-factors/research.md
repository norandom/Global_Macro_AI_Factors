# Gap Analysis — regime-steered-ai-factors

_Generated during `/kiro-validate-gap`. Brownfield analysis of the existing `macro_framework` + notebooks/scripts against the 8 requirements. Information over decisions: options and effort/risk, not final choices._

## 1. Current-state investigation

The repository already contains most of the primitives this feature needs; the work is largely **glue + one genuinely new orchestration layer (the /loop)**, not new science.

**Metric primitives (reusable as-is).**
- `macro_framework/ssr.py` — `compute_ssr` (ssr.py:65), `newey_west_var` (ssr.py:41, Bartlett HAC long-run variance), `andrews_bandwidth` (ssr.py:25), `SSRResult` (ssr.py:54). Pure numpy/pandas, `TRADING_DAYS=252`. Covers the SSR gate **and** supplies a HAC estimator.
- `scripts/build_tear_sheet.py` — hand-rolled OLS `_ols` (build_tear_sheet.py:80, `np.linalg.lstsq`, returns `(coef, r2, resid)`) and a **4-ETF basket regression already implemented** in `main` (build_tear_sheet.py:139-157): `r2_basket_4etf`, `residual_vol_ann_basket`, plus CAPM `beta_spy/alpha_ann_vs_spy/r2_capm/idio_vol_ann_capm`. `BASKET=["SWDA.L","XLK","IAU","BIL"]` (build_tear_sheet.py:44). `_active_value` strips warm-up stub (build_tear_sheet.py:62). Convention `ANNUAL=365`.
- `workbook/factor_workbook/rederive.py` — `equity_metrics` (rederive.py:243, → total/ann return, vol, sharpe, sortino, **calmar**, max_dd), `wilson_ci` (rederive.py:116), `guarded_tilt` (rederive.py:140).
- `macro_framework/evaluation.py` — `head_to_head_report` (evaluation.py:89), `crisis_analytics`, `turnover_stats`. `macro_framework/returns.py` — `daily_returns` (returns.py:6). `macro_framework/backtest.py` — vectorbt `buy_and_hold`/`single_asset_buy_and_hold`/`summary`.

**No-recall gate (already exists — this is the biggest reuse win).**
- `FactorScorer` (factor_scoring.py:588) produces `p_memorized`; `.calibrate` (factor_scoring.py:615), `.score`/`.score_many`, `.is_weak`.
- `run_pit_vs_nonpit_contrast` (factor_scoring.py:1567) → `ContrastResult` (factor_scoring.py:1473) → **`.contamination_premium()` (factor_scoring.py:1502)** = the PIT-vs-non-PIT memorization premium the "no-recall gate" is defined against.
- Offline certification: `certification_stats` (factor_scoring.py:2026, sklearn bootstrap CI + permutation p, deterministic by seed), `certification_verdict` (factor_scoring.py:2195, `no_recall_p=0.1`, `min_parse_rate=0.9`).

**PIT walk-forward (enforced by construction).**
- `build_walk_forward_targets` (walk_forward.py:36), `monthly_rebalance_dates` (walk_forward.py:21), `lookback_days=756`. **PIT is structural**: every context slice is strictly-before the rebalance date — `prices.loc[prices.index < rb].tail(lookback_days)`, returns `< rb`, macro panel `< rb` (walk_forward.py:58-64). Any regime/overlay reading `ctx["returns"]` is PIT-clean automatically.

**AI-view pipeline + mutation levers (all located).**
- Path: `render_regime_loadings_prompt` (factor_scoring.py:91) → `parse_loadings` → `loadings_to_tilt_views` (factor_scoring.py:1093, dots against hand-set `REGIME_ASSET_EXPOSURE` factor_scoring.py:1050) → `recall_guarded_adjust` (factor_scoring.py:1219, `tilt*(1-p_mem)`) → `views_to_bl` (llm_agent.py:190, `Q = tilt*conviction/252` at llm_agent.py:214) → blend in `combine` (extend_stream_2026.py:627-638, `w=(1-TILT)*w_hrp+TILT*w_bl`, `TILT=0.30`).
- **Levers** (each a one-mutation target): blend `TILT` (extend_stream_2026.py:64) and cash pin `{"BIL":0.25}` (extend_stream_2026.py:630); BL `tau/delta/obj` (allocation.py:60-69); Q scaling `/252`+`confidence` (llm_agent.py:214); conviction `_conviction_from_loadings` (factor_scoring.py:1663); prompt text (factor_scoring.py:146-173) and axes `MACRO_AXES` (factor_scoring.py:54); exposure table `REGIME_ASSET_EXPOSURE` (factor_scoring.py:1050); macro-series `_RAW_TO_Z` (factor_scoring.py:481); posterior-vs-single-view = switch `factor_rebalance`/`recall_guarded_adjust` → `steer_rebalance`/`steer_views` (steering.py:662, gate at :718).

**Ablation rungs (already produced as separate lines).**
- PIT vs non-PIT built hold-all-else-equal via `render_regime_loadings_prompt(identifying=True)` (factor_scoring.py:178-198); non-PIT diagnostic line `run_variant_line("factor_nonpit_ext2026", ...)` (extend_stream_2026.py:687), persisted "NEVER deployable" (extend_stream_2026.py:728). Replay-vs-live split: `CUTOFF=2024-06-01`, `_ReplayScorer` (extend_stream_2026.py:363) → **zero NIM calls on 2019-2024**, live only on 2025+.

**Reproducibility scaffolding (follow, don't invent).**
- `run_header.json` (calibrate_factor_scorer.py:101; ext variant extend_stream_2026.py:997 with `premium_summary`/`split_table`/`luck_vs_skill_ssr`/`immutability`); decision logs `factor_decision_log_*.json` (contract.py:159); additive `_v1/_v2/_ext2026` filenames (never overwrite, extend_stream_2026.py:995); `data/csv_mirrors/` (US + `_de` locale); schema contract `contract.AssetSpec`/`SchemaError`. Determinism: seeded RNG (`default_rng(42)`), no-RNG corpus sampling.

**Dependencies.** `statsmodels` is importable (transitive via `riskfolio-lib`/`arch`) but **not a declared direct dep**, and no existing eval code uses it (OLS is hand-rolled, certification uses sklearn). `riskfolio-lib>=7.2.1`, `scipy`, `scikit-learn`, `vectorbt`, `quantstats`, `recall-guard` are present. Artifacts land in `data/` (machine, gitignored, released as `data-vN`), `data/tear_sheet/` (Excel), `reports/` (human MD/CSV).

**Regime code: absent here, portable from facdrone.** No VIX/correlation/vol-scaling anywhere in this repo. `mc_regime.py` is a Monte-Carlo *allocator* (macro-z classification → full weight vector), not a scalar de-risk multiplier — treat as unrelated prior art. facdrone `src/facdrone/library/regime_overlays.py` has `correlation_scale` (regime_overlays.py:64, continuous `[0.20,1.0]` dampener), `avg_pairwise_correlation` (:50), `ewma_correlation_matrix` (:34) — pure numpy/pandas, ~50 LOC, no new deps; and the Sparse Jump Model `jump_regime.py` (`fit_jump_model` :192, `classify_latest` :241) which is heavier and needs an external walk-forward refit wrapper.

## 2. Requirement → asset map (gaps tagged)

| Req | Existing asset to reuse | Gap | Tag |
|---|---|---|---|
| **R1** basket-residual metric | `build_tear_sheet._ols` + basket block; `ssr.newey_west_var` for HAC | Expose as a callable harness; add **HAC t-stat on α_own** (basket OLS today has no HAC t); residual-floor guard; disclose annualization basis | **Missing (glue)** + **Constraint** (repo hand-rolls OLS) |
| **R2** gates + PASS/FAIL verdict | `compute_ssr`; `ContrastResult.contamination_premium`; `equity_metrics` (Calmar/DD) | A single verdict object combining skill-t>2, SSR≥1.96, recall≈0, no-Calmar/tail-regression | **Missing (orchestration)** |
| **R3** PIT / no-recall | `walk_forward` strictly-before slicing (walk_forward.py:58-64) | None structural; overlay/regime **must** read `ctx["returns"]` and fit walk-forward; add explicit look-ahead rejection+log | **Constraint** (mostly satisfied) |
| **R4** ablation ladder | `factor_pit` / `factor_nonpit_diagnostic` / baseline lines already run | Score all rungs on the **new basket-residual metric** side-by-side on one OOS window | **Missing (wiring)** |
| **R5** /loop driver | walk-forward + all levers located; `autoresearch` /loop skill exists generically | Project-specific mutation registry, verify=harness, keep/revert on gates, ledger, loop-until-dry | **Missing (main new build)** |
| **R6** regime de-risk overlay | HRP cash-pin `hrp_cvar_weights_with_fixed` (allocation.py:41); facdrone `correlation_scale` | Port 3 funcs (~50 LOC); compute dynamic `bil_pin` in `combine` (~3 lines) — **no `allocation.py` change** | **Missing (low-risk port)** |
| **R7** regime-conditioned AI view | `render_regime_loadings_prompt`, loadings pipeline | New mutation feeding a **PIT** regime label into the prompt/loadings; gated by R2 | **Missing** |
| **R8** reproducibility | `run_header.json`, decision logs, additive filenames, `contract.AssetSpec`, seeded determinism | Follow conventions; add the /loop ledger as a new additive artifact | **Constraint** (satisfied by adherence) |

## 3. Implementation approach options

**Option A — extend in place.** Fold the harness into `build_tear_sheet.py`, add levers/overlay inline in `extend_stream_2026.py`. ✅ fewest files. ❌ `factor_scoring.py` (2650 LOC) and `extend_stream_2026.py` (~1000 LOC) are already large; the /loop does not belong inside a one-shot script.

**Option B — all-new modules.** New `skill_metric.py`, `regime_overlay.py`, `scripts/factor_loop.py`. ✅ clean seams, isolated tests. ❌ risks duplicating the basket-OLS/annualization conventions already in `build_tear_sheet.py`/`rederive.py` (the exact divergence R8.2 forbids).

**Option C — hybrid (recommended).** New **small** modules for the two genuinely-new units — `macro_framework/skill_metric.py` (basket-residual appraisal + HAC t + `GateVerdict`, reusing `newey_west_var`/`compute_ssr`/`equity_metrics`) and `macro_framework/regime_overlay.py` (ported `correlation_scale` family) — **reusing** the existing walk-forward, replay driver, contrast, and the `combine` closure as the integration seam. The `/loop` is a new thin script `scripts/factor_loop.py` that parameterizes the existing walk-forward with a mutation config and calls the harness for the verify step, writing a `factor_loop_ledger_*.json` under the existing artifact conventions. ✅ new code only where new responsibility exists; ✅ no divergent re-implementation; ✅ overlay needs zero `allocation.py` change. ❌ requires disciplined interface design at the `combine`/verify seams.

## 4. Effort & risk

| Component | Effort | Risk | Justification |
|---|---|---|---|
| R1–R2 skill harness + verdict | **S–M** | **Low** | Glue existing SSR/HAC/OLS/Calmar; only new logic is HAC-t on the basket α |
| R4 ablation ladder scoring | **S** | **Low** | Rungs already run; wire through the harness |
| R6 regime de-risk overlay (correlation) | **S** | **Low** | ~50 LOC port + ~3 lines in `combine`; PIT-clean via `ctx["returns"]`; no `allocation.py` change |
| R5 /loop driver + ledger | **M–L** | **Medium** | New orchestration; **live-NIM cost/latency per iteration** on the OOS window is the real risk |
| R7 regime-conditioned AI view | **M** | **Med–High** | Return-seeking regime tilt — facdrone measured this **net-harmful**; PIT-prompt plumbing + likely-negative result |
| R8 reproducibility integration | **S** | **Low** | Follow `run_header`/decision-log/contract conventions |
| R6 VIX variant (optional) | **S** | **Medium** | Needs `VIXCLS` ingested into `fred_series` (absent today) |
| R6 SJM variant (optional) | **M** | **Medium** | Port `jump_regime.py` **plus** write the walk-forward refit wrapper (not in facdrone) |

## 5. Recommendations for design + Research-Needed items

**Preferred approach:** Option C. Build R1/R2 (harness + verdict) and R6 (correlation de-risk overlay) first — both are Low-risk and immediately give the ablation ladder (R4) and the control leg. Then R5 (/loop) wrapping them. Defer R7 to a gated loop mutation, expecting a likely-negative result (documented facdrone precedent). Optional VIX/SJM only if the correlation dampener underperforms.

**Research Needed (carry into design):**
1. **HAC regression t-stat — statsmodels vs hand-rolled.** `statsmodels` is importable but undeclared and unused by repo eval code (which hand-rolls OLS). Decide: (a) declare `statsmodels` as a direct dep and use its HAC (what nb05's attribution cell already does), or (b) extend `_ols` with a HAC coefficient covariance built from `newey_west_var`. (b) keeps the zero-new-dep convention but is more code.
2. **Metric ceiling is low by construction — set an honest bar.** A long-only reweight of the *same* four ETFs has basket R²≈0.98 and a ≈1.2% residual (measured this session): allocation-timing alpha is *structurally small*. Confirm the metric regresses the dynamic strategy on **static** (constant-beta) basket exposure so timing shows up, and decide whether the acceptance bar should be **relative improvement + luck-excluded CI** rather than the absolute `t>2`.
3. **Absolute gate thresholds may be unreachable.** Current strategies sit at **SSR≈0.14** (far below 1.96) on ~10y; the post-cutoff OOS window is only ~1.5y (~15 monthly rebalances) — thin for a HAC t or SSR. Design should reconsider whether R2's gates are absolute (`SSR≥1.96`, `t>2`) or **relative + significance-of-improvement**, and whether a longer/rolling or block-bootstrap evaluation is required for statistical power. _This is the single most important feasibility risk._
4. **Loop cost control.** Replay covers only pre-2025; each mutation on the OOS window needs live NIM scoring unless structured to reuse cache. Partition mutations into **cache-reusing** (blend/τ/overlay/exposure-table — no re-scoring) vs **re-scoring** (prompt/axes/factor-set) and sequence the cheap ones first.
5. **OOS window definition.** Fix the held-out window and confirm it is disjoint from any calibration/cutoff window (`CUTOFF=2024-06-01`); decide walk-forward-CV vs single post-cutoff split.
6. **VIX ingestion** (`VIXCLS` → `fred_series`) only if a VIX overlay is wanted; the correlation overlay off `ctx["returns"]` needs nothing new.

---

# Design Synthesis (from /kiro-spec-design)

**Feature classification:** Extension (integration-focused). Discovery reused the gap-analysis above; no external web research required — all integration seams are internal and already mapped.

**Generalizations found.**
- The own-basket regression buried in `build_tear_sheet.py:main` (plain `np.linalg.lstsq`, 365-basis, no HAC) is generalized into a reusable, HAC-corrected `macro_framework/skill_metric.basket_residual`. `build_tear_sheet.py` can later delegate to it (dedupe), but that refactor is deferred (Non-Goal) to avoid disturbing published tear-sheet outputs.
- The four ablation rungs already exist as separate persisted lines; the ladder is a *scoring* wrapper, not new sim code.

**Build-vs-adopt decisions.**
- **Adopt `statsmodels` (new direct dep)** for the HAC regression t-stat on own-basket α, rather than hand-rolling a HAC coefficient covariance from `newey_west_var`. Rationale: the nb05 attribution cell already uses `OLS(...).fit(cov_type="HAC")`; a hand-rolled HAC coefficient covariance is error-prone; statsmodels is already installed transitively. Cost: one declared dependency. The SSR gate still uses the repo's own `ssr` module (no change).
- **Port (not depend on) the correlation overlay** from facdrone (`correlation_scale` family, ~50 LOC, numpy/pandas-only) into `macro_framework/regime_overlay.py`. Cross-repo import is not an allowed dependency; copying the pure functions is.

**Simplifications.**
- The de-risk overlay requires **no `allocation.py` change** — the pin fraction `fixed_weights["BIL"]` is already the de-risk knob; the overlay computes a dynamic pin at the `combine` seam and passes it to the existing `hrp_cvar_weights_with_fixed`.
- v1 ships the **correlation** overlay only; VIX (needs `VIXCLS` ingestion) and SJM (needs a walk-forward refit wrapper) are deferred Non-Goals.

**Gate-feasibility mitigation (design-level, requirements unchanged).**
- The gap analysis flagged that the literal R2 gates (`SSR≥1.96`, `t>2`) may be structurally unreachable (basket residual ≈1.2%; SSR≈0.14 today; ~1.5y OOS). The design keeps those as the **default `absolute` thresholds** (satisfying R2 verbatim) and adds a configurable `GateConfig.mode="relative_improvement"` plus effect-size/CI reporting as the escape hatch. This is an implementation configuration, not a requirements change; the default verdict still encodes the requirement. Flagged as Open Question 1–2 for the tasks phase to set the loop's default mode.

**Boundary decisions recorded.**
- New **library** modules (`skill_metric`, `regime_overlay`) import only library-tier + external; they must not import pipeline-tier or scripts. Scripts (`factor_loop`, `ablation_ladder`) may import everything. Dependency direction Library → Pipeline → Scripts is enforced.
- The `combine`-closure overlay hook defaults to `None` so published `*_ext2026` results stay byte-identical unless the overlay is explicitly enabled.
