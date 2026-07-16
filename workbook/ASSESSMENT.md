# Step-by-step assessment in Excel (Power Query, German locale)

A didactic walkthrough of the whole storyboard — S0 (the static problem) through
S5 (luck vs skill) — using one reusable Excel sheet fed from the versioned
[data-v2 release](https://github.com/norandom/Global_Macro_AI_Factors/releases/tag/data-v2).
Every file referenced here is a `_de` variant: **semicolon-separated, comma
decimals — loads correctly in German Excel with zero transform steps.**

URL prefix for everything (prepend to every filename below):

```
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/
```

---

## Part 1 — One-time setup: the reusable Power Query

You need **two sheets**, because the data comes in two shapes:

- **Sheet "TS"** (time series): columns `Date; value; daily_return; drawdown` —
  all equity lines share this shape, so charts and formulas survive a data swap.
- **Sheet "REF"** (reference tables): `tear_sheet_de.csv`, `risk_decomposition_de.csv`
  and the per-step tables (each has its own columns — load side by side, don't swap).

**Create the reusable TS query once:**

1. *Daten → Daten abrufen → Aus dem Web* → paste `static_bh_equity_2016_2026_de.csv`
   (with the prefix) → **Laden in… → Tabelle** on sheet "TS".
2. Name the query `TS` (*Abfragen und Verbindungen* → Rechtsklick → Umbenennen).
3. Build your charts and formulas against that table (`TS[value]`, `TS[daily_return]`, …).

**To iterate to the next dataset — the whole trick:**

1. *Daten → Abfragen und Verbindungen* → Doppelklick auf `TS` (opens Power Query)
2. Click the ⚙ next to the **Quelle** step → replace only the filename in the URL
3. *Schließen & laden* — every chart and formula on the sheet updates.

Sanity checks after every load: equity files start at `value = 10000`;
`daily_return`/`drawdown` are small fractions (−0,008 = −0,8 %). If you see
9-digit integers, you loaded a non-`_de` file — switch to the `_de` variant.

**Charts to build once on "TS"** (they survive every swap):

- *Equity*: Linie, `Date` × `value`
- *Drawdown*: Fläche, `Date` × `drawdown` (place under the equity chart — crisis
  episodes become self-evident)

---

## Part 2 — One-time reference: definitions, equations, PM feedback points

Conventions used throughout: daily simple returns; annualization on the repo's
**365-day calendar basis** (vectorbt convention; the SSR block uses 252 —
disclosed where it appears). Excel formulas given in German / English.

### 2.1 Return & risk

| Metric | Equation | Excel | PM feedback point |
|---|---|---|---|
| Total return | $V_T/V_0 - 1$ | last/first `value` −1 | "Over what window — and who chose the window?" |
| CAGR | $(1+TR)^{365/n}-1$ | — (in `tear_sheet_de.csv`) | "Geometric, not average-of-years? Fees, slippage, taxes?" |
| Ann. volatility | $\sigma_d\sqrt{365}$ | `=STABW.S(r)*WURZEL(365)` / `STDEV.S…SQRT` | "Vol assumes symmetric, thin tails — check kurtosis before trusting it." |
| Sharpe | $\bar r_d / \sigma_d \cdot \sqrt{365}$ | `=MITTELWERT(r)/STABW.S(r)*WURZEL(365)` | "In-sample or out? Basis (252/365)? A Sharpe without a window and basis is a rumor." |
| Sortino | $\bar r_d / \sigma_{down} \cdot \sqrt{365}$ | needs downside-only STDEV | "Only penalizes downside — flatters asymmetric books; compare WITH Sharpe, not instead." |
| Max drawdown | $\min_t (V_t/\max_{s\le t}V_s - 1)$ | `=MIN(drawdown)` | "Depth, duration, AND recovery date. Would the investor have stayed in the seat?" |
| Calmar | CAGR / \|maxDD\| | — | "One crisis dominates it; unstable across windows." |
| VaR 95 (daily) | 5% quantile of $r_d$ | `=QUANTIL.INKL(r;0,05)` | "A boundary, not an expectation — pair with CVaR." |
| CVaR 95 (daily) | mean of returns ≤ VaR | `=MITTELWERTWENN(r;"<="&VaR)` | "The tail's average — the number risk committees actually fear." |
| Skew / excess kurtosis | 3rd/4th standardized moments | `=SCHIEFE(r)` / `=KURT(r)` | "Kurtosis ≫ 0 ⇒ vol understates tail risk ⇒ every Gaussian metric above is flattered." |
| Hit rate | share of up days | `=ZÄHLENWENN(r;">0")/ANZAHL(r)` | "Barely above 50% is normal even for good strategies — magnitude matters, not frequency." |

### 2.2 Benchmark-relative (needs an aligned benchmark series)

| Metric | Equation | Excel (aligned columns) | PM feedback point |
|---|---|---|---|
| **Correlation to S&P 500** | $\rho(r_p, r_{SPY})$ | `=KORREL(r_p;r_spy)` / `CORREL` | "First diversification question: does this move WITH the market? ρ≈0.85 = an equity book in disguise; ρ≈0.5 = genuine diversifier." |
| Beta | $\mathrm{cov}(r_p,r_m)/\mathrm{var}(r_m)$ | `=STEIGUNG(r_p;r_spy)` / `SLOPE` | "The market throttle. ρ tells you HOW RELIABLY it co-moves, β tells you HOW MUCH." |
| Active return | $\bar{(r_p - r_{spy})}\cdot 365$ | `=MITTELWERT(active)*365` | "Did it actually beat the index, before risk adjustment?" |
| Tracking error | $\sigma(r_p-r_{spy})\sqrt{365}$ | `=STABW.S(active)*WURZEL(365)` | "TE² ≈ (β−1)²σ_m² + σ_idio² — how much TE is just the beta gap?" |
| Information Ratio | active return / TE | `=MITTELWERT(active)/STABW.S(active)*WURZEL(365)` | "≥0.5 decent, ≥1.0 excellent, ~0 = no information. THE active-management verdict." |

### 2.3 Factor decomposition (Paleologo)

Model: $r_p = \alpha + \beta' f + \epsilon$; variance additivity
$\sigma^2_{total} = \beta'\Omega_f\beta + \sigma^2_\epsilon$.

| Metric | Equation | Excel | PM feedback point |
|---|---|---|---|
| R² (systematic share) | explained variance share | `=BESTIMMTHEITSMASS(r_p;r_spy)` / `RSQ` | "The share of risk RENTED from factors. And: which factors? One-factor R² hides the rest." |
| Idio vol σ_ε | residual std, annualized | `=STFEHLERYX(r_p;r_spy)*WURZEL(365)` / `STEYX` | "Only meaningful RELATIVE to the factor model. CAPM-'idio' in a multi-asset book = other factors (gold, rates)." |
| α (regression) | intercept ×365 | `=ACHSENABSCHNITT(r_p;r_spy)*365` / `INTERCEPT` | "One model's alpha is another model's beta. Re-run with the full factor set before calling it skill." |
| Appraisal ratio | $\alpha/\sigma_\epsilon$ | combine the two above | "Grinold/Kahn's IR. Spectacular under a bad model, honest under a good one — the model IS the claim." |

### 2.4 Stability (the honesty instrument — not computable in Excel)

| Metric | What it is | PM feedback point |
|---|---|---|
| **SSR** (Sharpe Stability Ratio) | t-statistic of the rolling 1-year Sharpe against 0, with Newey-West HAC standard errors (Andrews bandwidth) — precomputed in `tear_sheet_de.csv` | "Sharpe says how good; SSR says whether it was RELIABLY good. Needs ≥1.96. Autocorrelation of rolling windows kills naive t-tests — HAC is non-negotiable." |
| Wilson 95% CI | binomial interval for an accuracy | "Does the interval contain 0.5? Then the hit rate is coin-flip-compatible." |
| Paired Cohen's d | mean(Δ)/std(Δ) on paired observations | "Effect size for paired designs — big d with tiny P&L difference means the effect is real but not monetizable." |

---

## Part 3 — The iteration (swap the TS query / load one REF table per step)

Load once into "REF" and keep: `tear_sheet_de.csv` (one row per line — most step
numbers are just row-lookups here) and `risk_decomposition_de.csv`.

### S0 — The problem: a hindsight-selected static portfolio

**TS query →** `static_bh_equity_2016_2026_de.csv` · **also load:**
`active_returns_static_vs_spy_2016_2026_de.csv` (aligned static/SPY/active
returns — the live-regression playground), optional `spy_bh_equity_2016_2026_de.csv`
for the benchmark line, `monthly_returns_static_bh_2016_2026_de.csv` for the heatmap.

Walk through, in seduction order:

1. Equity chart: 10.000 → ~42.000 (+320%; SPY ends ≈42.200 — *same destination!*).
2. Tear-sheet row `static_bh_2016_2026`: Sharpe **1,44** (SPY: 0,96), maxDD **−19,6 %**
   (SPY −34 %), CAGR ~23 %. Kurtosis **11,8** — fat tails, remember 2.1.
3. Drawdown chart: COVID-2020 and 2022 dents visible — crisis stats:
   COVID −19,4 % vs SPY −33,7 %; 2022 −19,2 % vs −24,5 %.
4. **Correlation with the S&P 500** (the diversification question): on the aligned file
   `=KORREL(static_return;spy_return)` → **0,838**. β `=STEIGUNG(...)` → **0,605**.
   Read: it co-moves with the market 84% reliably at 60% amplitude — a damped
   equity book, not an uncorrelated machine. (All lines' `corr_spy` sit in
   `risk_decomposition_de.csv`.)
5. Information Ratio on `active_return`: **0,048** → no information over SPY.
6. The alpha mirage: `=ACHSENABSCHNITT(...)*365` → **+9,2 %** "alpha", appraisal 1,09 —
   then `risk_decomposition_de.csv` row: `r2_basket_4etf` **0,985**, basket alpha
   **−1,8 %** → the "alpha" was gold/cash factor return; true idio ≈ 1,9 %.
7. The verdict: `ssr` = **0,147** ≪ 1,96 — *luck-compatible* (column `ssr_verdict`).

**PM summary S0**: gorgeous curve, real crisis cushioning, ρ=0.84 to the market,
IR≈0, fake CAPM alpha, zero certifiable skill — and the selection peeked at the
answers (assets chosen by SSR over the simulated decade).

### S1 — Model selection: does the AI remember the data?

**REF load →** `norecall_screen_results_de.csv` (one row per candidate LLM).
Optional drill-down: `norecall_evidence_openai_gpt-oss-20b_de.csv` (raw per-prompt
evidence: prompt, reply, memorization features, per-row inclusion).

Read per row: `controlled_auc` with CI and `controlled_perm_p` (recall detector),
`positive_control_*` (does the detector work at all), `parse_rate`, `verdict`.

Expected: **every screenable model "recalls"** — gpt-oss-20b AUC **0,926**
(p=0,002), phi-4-mini 0,695, maverick 0,637, gpt-oss-120b 0,633; llama-3.3-70b
unscreenable ("not exonerated"). The certified-no-recall set is **empty**;
gpt-oss-20b was selected *despite* maximal recall, to run **recall-guarded**.

**PM point**: model due diligence = data due diligence. "Has the model seen the
test set?" is the AI version of "was the backtest in-sample?" — same S0 disease,
now measured instead of assumed.

### S2 — No predictive alpha: the coin-flip check

**REF load →** `naive_directional_eval_openai_gpt-oss-20b_de.csv` (72 rows:
date, prompt, reply, predicted vs realized direction, `correct`).

Compute live: accuracy `=MITTELWERT(correct)` → **0,389**; Wilson 95% CI
[0,285; 0,504] — **contains 0,5**.

**PM point**: the expected, CORRECT result. The model was never supposed to
predict; anyone selling >55% directional accuracy on monthly macro should meet
your S0 checklist first. Everything that follows is exposure engineering, not
forecasting.

### S3 — Factor development: exposures, memorization, the guard

**REF load →** `factor_views_v1_de.csv` (per rebalance × asset: `raw_tilt`,
`p_memorized`, `guarded_tilt`). Also: `factor_scores_v1_de.csv` +
`factor_scores_v2_de.csv` (per-date memorization), `factor_loadings_v1_de.csv`.

Compute live:
- Guard formula check (one helper column): `guarded = raw_tilt*(1-p_memorized)`
  → matches `guarded_tilt` row for row.
- Memorization distribution: `=MITTELWERT(p_memorized)` → v1 **0,236**, v2 **0,271**;
  p90 via `=QUANTIL.INKL(...;0,9)`.

Narrative: prompt v2 was **rejected** by the accept-gate — every performance
check passed but contamination rose (0,271 > 0,236). The pipeline turns memory
into a measured, priced quantity.

**PM point**: "show me the discipline that REJECTS an improvement" — a gate that
only ever accepts is marketing. Here is a documented rejection on contamination.

### S4 — Two portfolio lines: guarded vs recall-enabled

**TS query →** `factor_equity_v1_de.csv` (PIT recall-guarded, deployable), then swap to
`factor_nonpit_diagnostic_equity_v1_de.csv` (recall-enabled, **diagnostic only**).
Per-date guard detail: `factor_decision_log_v1_de.csv` (p_memorized / steered / parse_ok).

Read from `tear_sheet_de.csv` rows (active-span, consistent basis):
factor_pit_v1 Sharpe **1,63**, nonpit **1,62**; `risk_decomposition_de.csv`:
**corr_spy 0,57 / β 0,23** — the factor book is only ⅓ market risk (vs the
static line's ρ 0,84 / β 0,61): the AI-factor tilt genuinely de-correlates from
the S&P, mostly via gold/rates exposure.

The near-coincidence of the two equity lines is the point: the model that
*maximally remembers* the decade gains ~nothing in P&L from being allowed to peek.

**PM point**: never deploy the diagnostic line — it exists to measure lookahead,
the way a placebo arm exists to measure a drug.

### S5 — Luck vs skill: closing the loop

**REF load →** `factor_contrast_v1_de.csv` (72 paired dates: `pit_p`, `nonpit_p`,
`delta`) and `factor_luck_vs_skill_v1_de.csv` (the SSR table with verdict text).

Compute live: contamination premium `=MITTELWERT(delta)` → **+0,528**
(memorization: PIT 0,236 vs non-PIT 0,764 — the model *demonstrably recognizes*
dated, named data). Paired Cohen's d: `=MITTELWERT(delta)/STABW.S(delta)` → **1,93**.

Read from the SSR table: PIT **0,124**, non-PIT **0,130**, and the
**differential 0,002** — the recall premium in *returns* is statistically
indistinguishable from zero. Huge in memory, nil in P&L, and the honest
instrument (same SSR that judged S0) says: luck-compatible, never attainable skill.

**PM summary S5 / thesis close**: S0 showed hindsight making a dumb portfolio
look brilliant. S1–S5 built the machinery to *measure* hindsight inside an AI
pipeline — and priced it at zero. The seduction and its audit, end to end, in
one spreadsheet.

---

## Full URLs (copy-paste ready)

**Reference (load once)**

```
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/tear_sheet_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/risk_decomposition_de.csv
```

**S0 — static line**

```
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/static_bh_equity_2016_2026_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/static_bh_equity_2014_2024_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/static_bh_targets_2014_2024_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/spy_bh_equity_2016_2026_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/spy_bh_equity_2014_2024_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/active_returns_static_vs_spy_2016_2026_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/active_returns_static_vs_spy_2014_2024_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/monthly_returns_static_bh_2016_2026_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/static_bh_stats.json
```

**S1 — model screen**

```
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/norecall_screen_results_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/norecall_evidence_openai_gpt-oss-20b_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/norecall_evidence_openai_gpt-oss-120b_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/norecall_evidence_microsoft_phi-4-mini-instruct_de.csv
```

**S2 — coin-flip eval**

```
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/naive_directional_eval_openai_gpt-oss-20b_de.csv
```

**S3 — factor development**

```
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/factor_views_v1_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/factor_scores_v1_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/factor_scores_v2_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/factor_loadings_v1_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/factor_loadings_v2_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/prompt_version_gate_v1.json
```

**S4 — two lines**

```
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/factor_equity_v1_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/factor_nonpit_diagnostic_equity_v1_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/factor_targets_v1_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/factor_nonpit_diagnostic_targets_v1_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/factor_decision_log_v1_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/factor_nonpit_diagnostic_decision_log_v1_de.csv
```

**S5 — luck vs skill**

```
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/factor_contrast_v1_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/factor_luck_vs_skill_v1_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/factor_contrast_summary_v1.json
```

**Inputs**

```
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v2/macro_panel_monthly_de.csv
```
