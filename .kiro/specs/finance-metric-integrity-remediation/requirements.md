# Requirements Document

## Introduction

Researchers and report users currently rely on notebooks, reports, and shared financial calculations affected by 15 confirmed defects in performance metrics, Sharpe Stability Ratio (SSR) inference, factor attribution, calendar handling, Markowitz analysis, cash returns, crisis analytics, and dated replay integrity. This project shall correct those defects as one coordinated remediation, preserve intentional published conventions where they remain valid, and regenerate affected outputs from corrected producers with regression evidence.

## Boundary Context

- **In scope**: The confirmed defects in the extended-stream publisher, factor-loop verification, shared evaluation, skill-metric and SSR behavior, the reviewed tear-sheet and SJM notebooks, the 10-year and maximum-window trio notebooks, and their directly generated CSV, Parquet, JSON, Markdown, TeX, and PNG artifacts.
- **Out of scope**: LLM provider or model changes, factor-prompt redesign beyond preserving dated identities, new strategy-selection or allocation features, discretionary strategy retuning, repository-wide market-data cleanup, and blanket conversion of intentionally retained historical 365-basis release fields.
- **Adjacent expectations**: Existing deterministic strategy configurations and the deliberate distinction between reader-facing financial metrics and explicitly labeled legacy release-parity metrics shall remain intact unless corrected financial inputs require rerunning the existing process. Artifact-only edits without correction of the owning calculation are not acceptable.

## Requirements

### Requirement 1: Consistent Performance Measurement

**Objective:** As a report user, I want every performance table to apply an explicit and internally consistent measurement convention, so that reported return and risk statistics are financially comparable.

#### Acceptance Criteria

1. When a report presents reader-facing metrics from business-daily returns, the reporting system shall calculate compound growth from elapsed time and annualize volatility-based statistics using the established trading-day convention.
2. Where a legacy release-parity metric uses a different annualization basis, the reporting system shall identify that basis explicitly and shall not present it as the reader-facing convention.
3. When multiple statistics appear in one report row, the reporting system shall derive them from the same portfolio definition, return stream, and measurement window unless the differing basis is explicitly identified in that row.
4. When a differential portfolio is reported, the reporting system shall derive total return, risk statistics, and SSR statistics from one consistent differential-return definition.
5. If an endpoint wealth difference is retained alongside statistics derived from a daily return spread, the reporting system shall label them as distinct measures rather than presenting them as statistics of one portfolio.
6. The reporting system shall preserve all intentional annualization alternatives as separately named and testable outputs rather than silently replacing or mixing them.

### Requirement 2: Financially Valid SSR Inference

**Objective:** As a researcher, I want SSR estimates and verdicts to use valid excess returns and validated inference settings, so that stability conclusions are economically and statistically meaningful.

#### Acceptance Criteria

1. When SSR is calculated for a portfolio, the SSR system shall calculate rolling Sharpe inputs from portfolio returns in excess of the designated total-return cash benchmark over matching sessions.
2. When an SSR result is evaluated against a configured significance level, the SSR system shall use that same significance level throughout inference and gate evaluation.
3. If the configured significance level is not strictly between zero and one, the SSR system shall reject the inference request with a clear validation error.
4. If the bootstrap count is not a positive integer, the SSR system shall reject the inference request with a clear validation error.
5. If the rolling window is invalid for the supplied series or is less than two observations, the SSR system shall reject the inference request with a clear validation error.
6. If the Sharpe benchmark is non-finite, the SSR system shall reject the inference request with a clear validation error.
7. When a non-default valid significance level is configured, the gate-verification workflow shall complete without an inference-configuration mismatch.
8. The SSR system shall retain deterministic inference metadata sufficient to reproduce the reported verdict under the same inputs and settings.

### Requirement 3: Calendar-Consistent Attribution

**Objective:** As a report user, I want factor and market attribution to compare returns over identical economic intervals, so that alpha, beta, residual risk, and appraisal metrics are not biased by calendar mismatches.

#### Acceptance Criteria

1. When factor or benchmark prices are aligned to strategy sessions, the attribution system shall align price levels before calculating returns.
2. When a strategy return spans one or more sessions absent from the strategy calendar, the attribution system shall compound the corresponding factor or benchmark movement over the same interval.
3. If a requested benchmark date label is absent, the attribution system shall reject the calculation rather than silently shorten the sample.
4. If a requested benchmark price value is missing or non-finite, the attribution system shall reject the calculation rather than silently remove affected observations.
5. When attribution starts at the first reported strategy return, the attribution system shall include the preceding benchmark price anchor required to retain that first return.
6. When a report row combines performance and attribution statistics, the reporting system shall use one common end date and observation set for both groups of statistics.
7. If attribution coverage cannot reach the performance end date, the reporting system shall either fail the report or identify the attribution start date, end date, and observation count as a separate measurement window.
8. When an intercept is labeled CAPM alpha or Jensen alpha, the attribution system shall regress portfolio excess returns on market excess returns using the same cash benchmark.
9. Where a regression uses raw portfolio and market returns, the reporting system shall label the intercept as a raw-return market-model intercept rather than CAPM alpha or Jensen alpha.

### Requirement 4: Correct Cash-Sleeve and Crisis Returns

**Objective:** As a strategy evaluator, I want cash allocations and crisis windows to include all economically earned returns, so that portfolio comparisons and stress results are not understated or shifted at their boundaries.

#### Acceptance Criteria

1. When a strategy allocates less than full weight to risky assets, the portfolio system shall apply the designated cash benchmark's total return to the residual allocation.
2. If a cash-benchmark return is unavailable for a required interval, the portfolio system shall reject or explicitly flag the calculation rather than substitute zero without evidence that the economic return was zero.
3. When an overlay and its control are compared, the portfolio system shall apply the same cash-return convention to both portfolios.
4. When a crisis window begins on a trading session after an earlier portfolio observation, the crisis-analysis system shall include the return from the preceding observation into the first crisis session.
5. When crisis metrics are exported, the reporting system shall reproduce the same boundary-inclusive crisis return shown by the shared crisis calculation.

### Requirement 5: Coherent Markowitz Inputs

**Objective:** As an allocation researcher, I want every Markowitz opportunity set expressed in one currency and one coherent time basis, so that means, covariance, frontiers, and implied weights describe an investable portfolio.

#### Acceptance Criteria

1. When assets quoted in different currencies are included in one opportunity set, the Markowitz analysis shall convert their returns to one disclosed base currency before calculating portfolio statistics.
2. When assets trade on different exchange calendars, the Markowitz analysis shall ensure each return observation represents a consistent measurement interval across the opportunity set.
3. If exchange-specific sessions are removed or combined, the Markowitz analysis shall use an annualization factor consistent with the effective observation frequency.
4. The Markowitz analysis shall disclose the base currency, analysis start date, analysis end date, and effective observation count used for each displayed frontier.
5. When a figure claims one full analysis window, the strategy points, benchmark, opportunity set, and frontier shall share the stated start and end boundaries.
6. If complete market data do not cover the claimed analysis window, the reporting system shall stop generation or label the shorter window explicitly.

### Requirement 6: Dated Replay Integrity

**Objective:** As a researcher, I want each rebalance date to retain its own generated evidence and scores, so that a replay reproduces the dated simulation rather than a prompt-deduplicated approximation.

#### Acceptance Criteria

1. When identical prompt text occurs on multiple rebalance dates, the replay system shall preserve a distinct response, score set, and derived portfolio record for each date.
2. When dated records are persisted and replayed, the replay system shall associate each response and score with its original rebalance date rather than with prompt text alone.
3. If a dated response or score is missing, duplicated inconsistently, or associated with another date, the replay system shall fail validation before publishing portfolio artifacts.
4. When a replay completes, the replay system shall demonstrate equality between the dated source records and the dated values consumed by the simulation.

### Requirement 7: Producer-to-Artifact Consistency

**Objective:** As a report user, I want every regenerated artifact to reflect corrected calculations and disclosed provenance, so that notebooks, tables, and figures do not disagree for the same portfolio and window.

#### Acceptance Criteria

1. When a shared financial calculation changes, the artifact-generation workflow shall regenerate every directly affected user-visible output from the corrected producer.
2. When the same portfolio and window appear in multiple outputs, the artifact-generation workflow shall produce matching values within the documented numeric tolerance.
3. If an output retains a deliberate legacy convention, the artifact-generation workflow shall identify that convention in the output or its accompanying provenance.
4. When generation completes, the artifact-generation workflow shall record the input window, observation count, annualization basis, cash benchmark, base currency where applicable, and deterministic inference settings needed to interpret the output.
5. If any affected artifact cannot be regenerated successfully, the artifact-generation workflow shall report the failure and shall not represent the remediation as complete.
6. The artifact-generation workflow shall not correct a published value solely by editing the artifact when its owning producer remains incorrect.

### Requirement 8: Regression and Financial-Parity Validation

**Objective:** As a project maintainer, I want executable checks for each corrected invariant, so that future notebook or framework changes cannot silently reintroduce the reviewed defects.

#### Acceptance Criteria

1. When the remediation is validated, the validation suite shall include at least one deterministic regression check for each of the 15 confirmed defect classes.
2. When a defect belongs to a shared calculation, the validation suite shall verify the invariant at the shared calculation boundary and at one affected report or artifact boundary.
3. When calendar alignment is tested, the validation suite shall include exchange-holiday gaps, a required first-return anchor, absent date labels, and present-but-missing price values.
4. When SSR behavior is tested, the validation suite shall include excess-return construction, non-default valid significance levels, invalid inference parameters, and deterministic verdict reproduction.
5. When Markowitz behavior is tested, the validation suite shall include mixed-currency inputs, cross-exchange calendars, and consistency between effective observation frequency and annualization.
6. When report parity is tested, the validation suite shall detect mixed measurement windows, mixed portfolio definitions, undisclosed annualization bases, and stale generated values.
7. When dated replay behavior is tested, the validation suite shall include identical prompts occurring on different rebalance dates and shall verify that their dated records remain distinct.
8. While any required regression or artifact-parity check is failing, the remediation project shall remain incomplete.
