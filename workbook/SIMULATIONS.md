# Simulation register — every run, its window, and why

One entry per simulation in the project. Engine for all dynamic runs:
**vectorbt** over daily target weights (Riskfolio-Lib HRP-CVaR base + fixed 25%
BIL sleeve, 70/30 blend with a Black-Litterman tilt where a tilt exists);
3-year (756-trading-day) lookback for estimation; monthly rebalances unless
stated. Metrics on vectorbt's 365-day calendar convention. Prices: Postgres DB
originally, yfinance substitution in later runs (documented per entry).

## Universe (fixed by nb02/nb03)

Top Sharpe-Stability-Ratio ETF per category over 2016–2026: SWDA.L (world
equity), XLK (tech), IAU (gold), BIL (1–3M T-bills). **This selection is
in-sample** — the central caveat carried by every run below.

## The three time frames and their reasons

| Frame | Used by | Reason |
|---|---|---|
| **2016-01 → 2026-01** ("the decade") | nb04 static, nb05 annual | The 10-year showcase window over which the universe was SSR-selected — deliberately in-sample, kept as the "problem" exhibit (S0). |
| **2019-01 → 2024-12** ("the stream", 72 rebalances) | nb07, nb08, nb09, nb11, nb12, nb13, nb14 | Fixed in nb09 when the LLM track was designed and inherited by every comparable line ("same rebalance stream" requirement for fair head-to-head). Substantive constraints at design time: (a) the FMP news corpus used for the original memorization calibration is thin before ~2019; (b) the window lies inside the scoring models' training cutoffs (maverick 2024-08, gpt-oss-20b 2024-06) — which is exactly where memorization is POSSIBLE and therefore measurable. Not a data limitation (panel and prices run to 2026). |
| **2014-01 → 2024-12** (walk-forward frame) | artifact storage for the stream lines + `static_bh_2014_2024` | Storage frame anchored 2014 so the 3-year lookback preceding the first 2019 rebalance is inside the frame; lines are flat ("stub") before their first rebalance. Tear-sheet metrics use the ACTIVE span only. |

## The runs

### 1. nb04 — Static buy-and-hold (S0, "the problem")
- **What**: 25% equal weight at inception, 4 trades total, no rebalancing. Benchmark SPY B&H.
- **Window**: 2016-01-31 → 2026-01-31 — the full SSR-selection decade, *deliberately* in-sample: this run exists to show how good hindsight selection looks.
- **Engine**: vectorbt `buy_and_hold`; re-implemented as plain share arithmetic in `scripts/build_static_bh.py` (matches to ~3 decimals; residual = yfinance-vs-DB prices).
- **Artifacts**: `static_bh_equity_2016_2026*`, `static_bh_stats.json`; headline: Sharpe 1.44, maxDD −19.6%, IR vs SPY 0.05, SSR 0.147 (luck-compatible).

### 2. nb05 — Annual rebalance HRP-CVaR + BL
- **What**: first dynamic engine test: annual rebalances, HRP-CVaR with the pinned 25% BIL sleeve, BL tilt toward a fixed prior view.
- **Window**: 2016-01-31 → 2026-01-31 (same decade as nb04 for direct comparison); 3y lookback pulls prices from 2013.
- **Role**: methodology bridge between static (nb04) and the monthly walk-forwards; not part of the head-to-head table.

### 3. nb07 — Baseline: HRP + momentum
- **What**: the no-LLM reference: monthly HRP with a 252-day momentum overlay on a broader ETF menu.
- **Window**: 2019-01-01 → 2024-12-31 (the stream). First line to use it after nb09 fixed it.
- **Artifacts**: `baseline_targets/equity_2019_2024`. Note its 4-ETF basket R² is only 0.41 — it trades a wider universe than the other lines.

### 4. nb08 — Track B: Monte-Carlo / Nash regime allocation
- **What**: regime probabilities from Monte-Carlo simulation, Nash-equilibrium blend of regime portfolios. No LLM.
- **Window**: the stream (2019–2024).
- **Artifacts**: `track_b_*`; regime probabilities in `track_b_regime_probs.parquet`.

### 5. nb09 — Track A: LLM directional agent
- **What**: monthly macro views from an LLM (DSPy → OpenRouter), converted via `views_to_bl` (Q = view·confidence/252) into the BL tilt on the HRP base.
- **Window origin**: **this notebook fixed SIM_START/SIM_END = 2019-01-01/2024-12-31** — 72 monthly rebalances. All later comparable lines inherit it for stream-identical head-to-heads.
- **Artifacts**: `track_a_*`, agent log with per-date views.

### 6. nb11 — Track A memory-guarded (steering)
- **What**: nb09's agent + recall_guard's `MemoryGuardedScorer`: per-date `p_memorized` gates/discounts the LLM views (directional façade: hard gate).
- **Window**: the stream (comparability with nb09 is the experiment).
- **Calibration**: news-corpus based (FMP), model maverick @ cutoff 2024-08-01.

### 7. nb12 — Prompt refinement (PIT)
- **What**: two prompt versions of the directional agent over the stream; accept-gate on contamination + performance.
- **Window**: the stream. Artifacts `prompt_refinement_*`.

### 8. nb13 — Factor variant, recall-guarded (S2–S4 deployable line)
- **What**: regime-as-loadings factors (gpt-oss-20b on anonymized z-scored numbers), number-native contamination scoring, guard = tilt·(1−p_memorized) (discount-only), through the unchanged HRP+BL blend.
- **Window**: the stream — mandated by the spec ("same rebalance stream") so the factor line joins the existing head-to-head. Additional property: the stream lies inside gpt-oss-20b's training window (cutoff 2024-06-01), i.e. recall is possible there — the regime the guard exists for.
- **Scoring model choice**: gpt-oss-20b selected DESPITE maximal recall (screen AUC 0.926) to demonstrate guarding; certified-no-recall set was empty.
- **Artifacts**: `factor_*_v1`, naive directional eval (S2), calibrator dir.

### 9. nb14 — Prompt v2 + PIT-vs-non-PIT contrast (S5)
- **What**: v2 prompt over the same stream (REJECTED by the gate: contamination 0.271 > 0.236); non-PIT diagnostic line (identifying prompts: dates+tickers+raw levels) vs the PIT line; premium +0.528 (d=1.93) in memory, differential SSR 0.002 in returns.
- **Window**: the stream — R7.6 requires all-else-equal, so both variants share it exactly.
- **Artifacts**: `factor_*_v2`, `factor_nonpit_diagnostic_*`, `factor_contrast_*`, `factor_luck_vs_skill_v1`.

### 10. `scripts/build_static_bh.py` — S0 data pack (data-v2)
- **What**: nb04 persisted as data on two windows: the decade (2016–2026) and the walk-forward-aligned frame (2014–2024, incl. drifting weights).
- **Why two windows**: the decade is the didactic exhibit; the aligned frame joins the stream lines date-for-date in Excel/workbook comparisons.

### 11. Tear-sheet pack (`scripts/build_tear_sheet.py`, data-v2)
- **What**: uniform re-computation across all 9 equity lines — active-span, one convention — plus CAPM-vs-SPY and 4-ETF-basket risk decomposition, IR-vs-SPY series.
- **Why active spans**: the 2014-anchored storage frames contain flat stubs that would dilute CAGR/vol/Sharpe; the tear sheet trims to first movement and discloses the window per row (this is why its Sharpes differ from the full-frame figures embedded in `factor_contrast_summary_v1.json`).

### 12. PLANNED — Extension to 2026-06: the post-cutoff natural experiment (→ data-v3)
- **What**: extend the stream by ~18 rebalances (2025-01 → 2026-06) for the factor PIT line, the non-PIT diagnostic, the naive eval, and the cheap comparison tracks; re-cut contrast/luck-vs-skill with an explicit **in-training vs post-cutoff split**.
- **Why**: 2019–2024 sits inside gpt-oss-20b's training window (recall possible — the guard's regime). Post-2024-06 data is unseeable by construction. Falsifiable prediction: the PIT-vs-non-PIT `p_memorized` premium (+0.53 in-training) **collapses toward 0 post-cutoff**, while return behavior stays similar. If observed, it validates that the contamination signal measures memorization specifically.
- **Status**: amendment task in the version-aware-factor-scoring spec; results will ship as `data-v3` (data-v2 stays immutable).

## Standing caveats (apply to every run)

1. Universe selection is in-sample (SSR over 2016–2026) — quantified in S0 (IR 0.05, SSR 0.147).
2. No fees/slippage/taxes; monthly rebalancing at close prices.
3. SWDA.L is priced in GBp on the LSE; series are mixed-currency in local terms (consistent across all lines, so comparisons are fair; absolute levels are idealized).
4. yfinance auto-adjusted closes substitute the original price DB in later runs; cross-checked deltas are ~1e-3 on decade-level stats.
5. All lines share the same 4-ETF universe except the baseline (broader menu) — visible in the basket-R² column of `risk_decomposition*.csv`.
