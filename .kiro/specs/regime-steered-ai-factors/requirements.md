# Requirements Document

## Introduction

This feature turns a diagnosis into a disciplined research programme. The tear-sheet and Paleologo attribution built this session show that the current HRP-CVaR + Black-Litterman book earns its results from **low market beta plus static factor premia (gold/cash), not timing skill** — the own-4-ETF basket regression explains ≈ 98% of return with a ≈ 1.2% idiosyncratic residual, the Sharpe is luck-compatible (SSR far below 1.96), the information ratio is ≈ 0, and the AI/BL view is contractionary and swamped by the optimizer. The goal is to derive a *measurably better* AI macro factor for the monthly rebalance, judged by a single skill metric (the own-basket appraisal ratio) under strict point-in-time / no-recall discipline, and to formalize the search as an automated iterate-verify-keep loop. A non-LLM regime de-risk overlay is included as a control leg and falsification bar, because the sibling `facdrone` project measured regime-based *return-seeking* tilts as net-harmful and prone to look-ahead — so regime signals are used here to de-risk, never to pick winners.

> **Context note:** No steering documents exist under `.kiro/steering/` at generation time. These requirements are derived from the seeded project description and the in-session evidence; running `/kiro-steering` later may surface product/tech constraints worth reconciling.

## Project Description (Input)

See the seeded description retained below the requirements ("Appendix: Original Input") for the full diagnosis, metric definition, five-step process, and non-negotiables.

## Boundary Context

- **In scope**: the skill-metric evaluation harness; the ablation ladder; the automated iterate-verify-keep (`/loop`) protocol; the non-LLM regime de-risk overlay (control) and a gated regime-conditioned AI view; evaluation on the existing 4-ETF universe (SWDA.L, XLK, IAU, BIL) and monthly-rebalance lineage.
- **Out of scope**: changing the recall-guard internals or its point-in-time guarantees; expanding the asset universe beyond the current four ETFs; onboarding new data vendors; live/production trading execution; selecting or training a different LLM (the existing scoring pipeline is reused); universe re-selection (still-in-sample SSR selection is a known, separate issue).
- **Adjacent expectations**: relies on the existing walk-forward machinery (nb13 lineage and the 2026 stream-extension), the recall guard's point-in-time guarantees, the macro panel and ETF price source, and the existing LLM factor-scoring pipeline. This feature consumes those and does not own them.

## Requirements

### Requirement 1: Basket-residual skill metric

**Objective:** As a quant researcher, I want a single metric that isolates timing skill from owned factor premia, so that "a better AI factor" is judged on genuine allocation skill rather than premia the book already earns.

#### Acceptance Criteria
1. When a strategy daily return series and its own-factor (4-ETF) return series are provided, the Evaluation Harness shall compute the own-basket appraisal ratio as annualized regression alpha divided by annualized residual (idiosyncratic) volatility.
2. When computing the own-basket appraisal ratio, the Evaluation Harness shall report the alpha's t-statistic using an autocorrelation- and heteroskedasticity-consistent (Newey-West HAC) standard error.
3. The Evaluation Harness shall report, alongside the skill metric, the own-basket regression R² and residual volatility so that the share of return explained by static factor exposure is visible.
4. Where a market benchmark series is provided, the Evaluation Harness shall additionally report the single-factor market alpha, beta, and R² so that single-factor and own-basket attributions can be compared.
5. If the regression residual volatility falls below a defined numerical floor, then the Evaluation Harness shall report the appraisal ratio as undefined rather than emit a divide-by-near-zero value.

### Requirement 2: Acceptance gates and keep/discard verdict

**Objective:** As a quant researcher, I want a mechanical pass/fail verdict from fixed gates, so that keep/discard decisions are reproducible and not eyeballed.

#### Acceptance Criteria
1. When a candidate configuration is evaluated, the Evaluation Harness shall emit a PASS verdict only if every acceptance gate passes, and a FAIL verdict otherwise.
2. The Evaluation Harness shall treat the skill gate as passed only when the own-basket alpha HAC t-statistic exceeds 2.
3. The Evaluation Harness shall treat the stability gate as passed only when the Sharpe Stability Ratio is at least 1.96.
4. The Evaluation Harness shall treat the no-recall gate as passed only when the point-in-time versus non-point-in-time memorization premium is not statistically distinguishable from zero on the post-cutoff window.
5. The Evaluation Harness shall treat the risk-shape gate as passed only when out-of-sample Calmar is not below the baseline Calmar and out-of-sample maximum drawdown is not worse than baseline beyond a stated tolerance.
6. If any gate fails, then the Evaluation Harness shall report which gate failed and the value that missed its threshold.

### Requirement 3: Point-in-time / no-recall discipline

**Objective:** As a PM/operator, I want every evaluation to be strictly point-in-time, so that reported skill cannot be an artifact of look-ahead or memorization.

#### Acceptance Criteria
1. The Research System shall evaluate the objective metric only on an out-of-sample window that is disjoint from any window used to tune or fit the candidate.
2. While fitting any data-driven component, including a regime detector, the Research System shall use only information available at or before each decision date (walk-forward / expanding fit).
3. If a candidate or a component would require information dated after its decision date, then the Research System shall reject that candidate and record the look-ahead reason.
4. When reporting any headline result, the Research System shall disclose the evaluation window and confirm that it is out-of-sample.
5. The Research System shall not select or rank candidates using in-sample Sharpe or in-sample return.

### Requirement 4: Ablation ladder

**Objective:** As a quant researcher, I want each pipeline layer scored on the skill metric, so that the AI view's marginal contribution is isolated from the optimizer and the blend.

#### Acceptance Criteria
1. When the ablation ladder is run, the Ablation Runner shall evaluate at least these rungs: HRP-only; HRP+BL with a fixed view; HRP+BL with the point-in-time AI view; and HRP+BL with the non-point-in-time diagnostic AI view.
2. The Ablation Runner shall report the own-basket skill metric for every rung on the same out-of-sample window.
3. When two adjacent rungs differ only by the AI view, the Ablation Runner shall report the change in the skill metric as the AI view's marginal contribution.
4. The Ablation Runner shall label the non-point-in-time rung as diagnostic-only and shall exclude it from any deployable recommendation.

### Requirement 5: Autoresearch iterate-verify-keep loop

**Objective:** As a quant researcher, I want an automated one-mutation-per-iteration loop with an audit trail, so that improvements accumulate under the gates without manual bookkeeping.

#### Acceptance Criteria
1. When a loop iteration begins, the Research Loop shall apply exactly one mutation to the current best configuration.
2. When a mutation has been applied, the Research Loop shall evaluate it on the out-of-sample point-in-time window using the acceptance gates.
3. When an evaluated mutation both improves the own-basket skill metric and passes every gate, the Research Loop shall adopt it as the new best configuration.
4. If an evaluated mutation fails any gate or does not improve the skill metric, then the Research Loop shall revert to the prior best configuration.
5. When no mutation has been adopted for a configured number of consecutive iterations, the Research Loop shall stop (loop-until-dry).
6. The Research Loop shall record, for every iteration, the mutation applied, the metric values, the gate outcomes, and the keep/revert decision in an auditable ledger.

### Requirement 6: Regime de-risk overlay as non-LLM control

**Objective:** As a PM/operator, I want a non-LLM regime de-risk control leg, so that the AI factor must out-earn a simple risk-off overlay to justify itself and the unsolved tail is directly addressed.

#### Acceptance Criteria
1. Where the regime de-risk overlay is enabled, the Regime Overlay shall reduce only the risky-sleeve weight in stress regimes and shall allocate the reduction to the cash sleeve.
2. The Regime Overlay shall never raise the risky-sleeve weight above its no-overlay level (de-risk only; no leverage and no winner-picking).
3. While a stress regime is detected, the Regime Overlay shall reduce risky exposure by the configured regime scale.
4. When a regime detector is used, the Regime Overlay shall fit it walk-forward so that the regime label uses no post-decision information.
5. When the AI factor is evaluated, the Research System shall also evaluate the non-LLM regime de-risk overlay on the same skill and risk-shape metrics as a control.
6. The Research System shall treat the AI factor as justified only where it out-performs the non-LLM control on the own-basket skill metric.

### Requirement 7: Regime-conditioned AI view (gated lever)

**Objective:** As a quant researcher, I want to test a regime-conditioned AI view as one gated mutation, so that its value is measured rather than assumed (the sibling project measured such tilts as net-harmful).

#### Acceptance Criteria
1. Where the regime-conditioned AI view is enabled, the Research System shall supply only point-in-time regime information to the view construction.
2. When a regime-conditioned AI view is evaluated, the Research System shall subject it to the same acceptance gates as any other mutation.
3. If the regime-conditioned AI view does not out-earn both the non-LLM control and the unconditioned AI view on the skill metric, then the Research System shall not recommend it for deployment.
4. The Research System shall bound the regime-conditioned view's influence within the existing HRP/BL blend so that it cannot become an unconstrained directional bet.

### Requirement 8: Reproducibility, conventions, and auditable artifacts

**Objective:** As a PM/operator, I want convention-consistent, reproducible, auditable outputs, so that results compare across specs and can be replayed.

#### Acceptance Criteria
1. The Research System shall compute risk/return, tail, skill, and attribution metrics under the repository's published conventions and shall disclose the annualization basis used (calendar/√252 versus simulation/365-day).
2. The Research System shall reuse the repository's existing metric, allocation, and walk-forward implementations and shall not introduce a divergent re-implementation of a quantity the repository already computes.
3. When an evaluation run completes, the Research System shall write its metrics and per-iteration ledger to durable, versioned artifacts under the repository's results outputs.
4. When re-run with the same inputs and configuration, the Research System shall reproduce the same metric values, replaying persisted LLM scores where live inference is not repeated.
5. The Research System shall not overwrite previously published result artifacts (outputs are additive, with versioned filenames).

## Appendix: Original Input

**Feature:** regime-steered-ai-factors

**Goal.** Derive a measurably better AI macro factor to steer the HRP-CVaR + Black-Litterman monthly rebalance (nb05/nb13/nb14 lineage), under strict point-in-time / no-recall discipline (`recall_guard`), and formalize the research as a metric-driven autoresearch `/loop`.

**Diagnosis (2016–2026, ~10y, √252, HAC).** HRP's edge is low market beta (0.27) + static factor premia (gold/cash), not timing skill — the own-4-ETF basket regression gives R² ≈ 0.98 with idiosyncratic residual ≈ 1.2%, i.e. allocation-timing alpha ≈ 0. The single-factor "alpha vs SPY" (8.7%/yr, HAC t ≈ 4.2) is factor premia invisible to a market-only model, not skill. SSR is far below 1.96 (luck-compatible), IR ≈ 0, and fat tails survive CVaR (excess kurtosis 7.2, drawdowns −12% to −14%). The current AI/BL view (5%/yr world > cash) is contractionary (equilibrium already implies ~18%/yr) and swamped by the MV utility optimizer; PIT ≈ non-PIT and the memorization premium collapses post-cutoff.

**Objective metric.** The own-basket appraisal ratio — regress strategy returns on the 4-ETF own factors, take annualized alpha_own / σε_own with a Newey-West HAC t-stat. Gates (all out-of-sample / walk-forward PIT): t(alpha_own) HAC > 2; SSR ≥ 1.96; recall/memorization premium ≈ 0 post-cutoff; no OOS Calmar/tail regression.

**Five-step process.** (1) Basket-residual evaluation harness. (2) Ablation ladder HRP-only → HRP+BL(fixed) → HRP+BL(AI PIT) → HRP+BL(AI non-PIT diagnostic). (3) The autoresearch `/loop`: one mutation per iteration, walk-forward PIT verify, keep iff skill improves and gates hold else revert, loop-until-dry. (4) Regime de-risk overlay as a non-LLM control leg and falsification bar (VIX tiers / EWMA correlation-crisis / Sparse Jump Model; de-risk only). (5) Regime-conditioned AI view as a gated lever, expected net-neutral/harmful per facdrone's measured evidence (SJM return-seeking NET-HARMFUL: 2024 +2.89% vs +28.84% composite; +1.6% SJM-on vs +7.5% SJM-off after a look-ahead bug fix; kept only scoped to forex; see `facdrone research/lookahead_sjm_incident.md`).

**Non-negotiables.** Strict PIT / no-recall (a regime detector refit on full history is a look-ahead vector as real as an LLM recalling a ticker — walk-forward-fit only); metric evaluated out-of-sample only; reuse existing framework rather than reinventing.
