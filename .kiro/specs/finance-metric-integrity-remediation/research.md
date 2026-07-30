# Gap Analysis: Finance Metric Integrity Remediation

## Analysis Context

- **Specification:** `finance-metric-integrity-remediation`
- **Requirements status:** Generated and approved
- **Codebase type:** Brownfield research and report-generation repository
- **Analysis basis:** Current working tree, including existing uncommitted changes and generated artifacts
- **Steering context:** No `.kiro/steering/` files are present; this analysis relies on the approved requirements, repository conventions, tests, artifacts, and the confirmed ultra-review findings

## Executive Summary

The repository already contains most of the numerical primitives needed for the remediation, but their use is inconsistent across shared modules, release scripts, and notebooks. The central implementation gap is not a missing financial library; it is the absence of strict shared contracts for return construction, measurement metadata, dated replay identity, and artifact parity.

The strongest reusable foundations are:

- `macro_framework.evaluation.metric_block` for reader-facing elapsed-time CAGR and 252-day risk statistics, with explicitly named legacy alternatives;
- `macro_framework.skill_metric.factor_returns_on`, `basket_residual`, and `market_attribution` for attribution, although the current return-alignment helper still needs finite-value and anchor handling;
- `macro_framework.ssr.ssr_inference` for deterministic moving-block-bootstrap inference;
- existing deterministic tests for annualization parity, calendar-gap compounding, gate semantics, and ledger replay.

The principal gaps are:

1. dated factor evidence is stored by date but consumed through prompt-keyed maps, so legitimate duplicate prompts overwrite earlier dates;
2. reader-facing reports mix elapsed-time, vectorbt, row-count, 252-day, and 365-day conventions without reliable labels;
3. portfolio SSR is calculated from gross rather than cash-excess returns, and invalid inference settings are not rejected;
4. attribution still permits missing values, shortened samples, missing first-return anchors, and raw-return regressions labeled as CAPM alpha;
5. the SJM cash sleeve uses price-only BIL and substitutes missing cash returns with zero;
6. crisis windows omit the return entering the first crisis session;
7. Markowitz inputs mix GBP and USD returns and combine cross-exchange calendars with a fixed 252 scaling;
8. generated artifacts lack a complete provenance and parity contract, allowing stale or internally inconsistent output families.

A hybrid implementation is the best fit: strengthen existing shared primitives, add only small explicit adapters/contracts where semantics differ, keep notebooks thin, and validate producer-to-artifact parity before publication. The estimated scope is **Large (1–2 weeks)** with **medium-high risk**, driven by artifact regeneration, deterministic replay correction, total-return data coverage, workbook parity, and cross-notebook propagation rather than by algorithmic novelty.

---

## 1. Current Architecture and Reusable Assets

### 1.1 User-visible production flows

| Surface | Current responsibility | Relevant outputs |
|---|---|---|
| `scripts/extend_stream_2026.py` | Extends dated factor streams, replays scores/loadings, publishes headline and release outputs | factor loadings/scores/targets/equities, decision logs, luck-vs-skill, tear sheets, risk decomposition, monthly returns, run header, CSV mirrors |
| `scripts/factor_loop.py` | Evaluates candidate factor variants and writes immutable decision ledgers | `factor_loop_ledger_<run_id>.json`, optional CSV mirror |
| `notebooks/15_2_tear_sheet_gallery.ipynb` | Main reader-facing tear-sheet gallery and trio report | common-window CSVs, trio release CSVs, equity/drawdown chart |
| `notebooks/16_ai_factor_variants_tearsheet.ipynb` | AI-factor comparison report | `reports/nb16_ai_variants_tearsheet.csv`, charts |
| `notebooks/17_sjm_crowding_derisk.ipynb` | SJM overlay selection, replay, attribution, holdout and tail reporting | SJM equity/ledger artifacts, `reports/nb17_sjm_crowding_tearsheet.csv`, docs and figures |
| `notebooks/18_3_trio_10y.ipynb` | Ten-year trio tear sheet and Markowitz analysis | `tear_sheet_trio_10y*`, panel and frontier PNGs |
| `notebooks/18_4_trio_max_timeframe.ipynb` | Maximum-window trio tear sheet and Markowitz analysis | `tear_sheet_trio_max*`, panel and frontier PNGs |
| `scripts/build_tear_sheet.py` | Standalone release tear-sheet and risk-decomposition builder | release CSVs used or duplicated by the extended-stream pipeline |

There is no separate application or service layer. Executed notebooks and generated CSV, Parquet, JSON, Markdown, TeX, and PNG files are the product surfaces.

### 1.2 Shared numerical interfaces

| Interface | Existing capability | Gap relevant to this project |
|---|---|---|
| `evaluation.metric_block` | Elapsed-time CAGR; 252-day volatility, Sharpe and Sortino; separate legacy alternatives | Not consistently used by release producers; metadata is not carried into rows |
| `evaluation.crisis_analytics` | Crisis return, volatility and drawdown extraction | Slices before adding the preceding anchor and omits the first crisis return |
| `skill_metric.factor_returns_on` | Reindex price levels before differencing; catches absent labels | Does not reject present NaN/Inf values, does not explicitly require/handle the preceding anchor, and uses default `pct_change` filling behavior |
| `skill_metric.basket_residual` | HAC residual attribution with alpha, R², residual volatility, appraisal and observation count | Relies on caller-constructed economically aligned returns |
| `skill_metric.market_attribution` | HAC raw-return market-model regression | Not a CAPM/Jensen excess-return regression; result has limited window metadata |
| `ssr.ssr_inference` | Deterministic moving-block-bootstrap SSR inference and verdict | Accepts gross returns, lacks required argument validation, and omits some interpretation metadata |
| `factor_loop.verify` | Runs appraisal and SSR gates on OOS results | Does not forward `GateConfig.ssr_alpha` into inference |
| `factor_workbook.rederive.equity_metrics` | Intentional vectorbt/release parity with row-count and 365-day conventions | Used under generic reader-facing field names in parts of the release flow |

### 1.3 Existing validation patterns

The repository already favors deterministic synthetic tests and pure helper extraction:

- `tests/test_evaluation.py` pins every deliberate difference between reader-facing and legacy metric engines.
- `tests/test_skill_metric.py` covers known alpha/beta, one exchange-calendar gap, absent labels, appraisal floor behavior, and gate truth tables.
- `tests/test_ssr_inference.py` covers deterministic bootstrap output, directional discrimination, and block-length bounds.
- `tests/test_factor_loop.py` covers OOS/tuning separation, deterministic ledger replay, gate behavior, and artifact serialization.
- `tests/test_stream_ext2026.py` and `tests/test_stream_ext2026_derisk.py` already avoid network/API execution by targeting pure helpers.
- workbook tests enforce root/vendored SSR source parity and released-artifact conventions.

This means the project can follow established patterns rather than introduce a new testing framework.

---

## 2. Requirement-to-Asset Gap Map

### Requirement 1: Consistent Performance Measurement

| Asset | Current state | Gap tag | Required capability |
|---|---|---|---|
| `macro_framework/evaluation.py::metric_block` | Correct reader-facing and legacy alternatives coexist | Reusable | Make it the source for reader-facing report rows and preserve legacy fields under explicit names |
| `macro_framework/evaluation.py::head_to_head_report` | Uses generic vectorbt annualized metrics on full frames | Constraint | Generic fields can expose legacy basis and flat pre-start stubs without disclosure |
| `scripts/extend_stream_2026.py` headline, tear and luck-vs-skill blocks | Three incompatible measurement paths | Missing | One active window and one declared basis per reader-facing field |
| `scripts/build_tear_sheet.py` | Publishes 365/row-count metrics under generic names | Missing | Explicitly label legacy basis or use reader-facing metrics |
| differential row in `extend_stream_2026.py` | Total return is an endpoint wealth difference while Sharpe/SSR use the daily spread | Missing | Use one differential portfolio definition or separate and rename the endpoint gap |

Concrete current divergence for the same root PIT equity includes roughly 13.9% elapsed CAGR versus 21.3% row-count/365 CAGR. The implementation must preserve intentional legacy values without allowing them to masquerade as reader-facing metrics.

**Complexity:** Medium. Numerical primitives exist; the work is caller migration, schema labeling, and artifact churn.

### Requirement 2: Financially Valid SSR Inference

| Asset | Current state | Gap tag | Required capability |
|---|---|---|---|
| `macro_framework/ssr.py` | Deterministic point estimate and moving-block inference | Reusable | Validate alpha, bootstrap count, rolling window and finite benchmark; preserve complete inference metadata |
| notebook SSR callers | Pass raw portfolio returns | Missing | Construct matching-session portfolio-minus-total-return-cash returns before inference |
| `scripts/factor_loop.py::verify` | Uses default inference alpha | Missing | Forward the configured gate alpha |
| workbook vendored SSR | Byte-for-byte root copy | Constraint | Any shared SSR change must be synchronized and parity-tested |
| differential SSR | Uses a return spread | Constraint | Differential returns must not have cash subtracted a second time |

The statistical core should remain independent of market-data fetching. A separate explicit excess-return constructor or portfolio SSR adapter is needed so callers cannot confuse portfolio-excess and differential return semantics.

**Research needed during design:** Decide whether short but otherwise valid series remain an “insufficient observations” result while invalid configuration raises. Requirement 2.5 clearly requires invalid windows to raise, but a valid window with too few post-alignment observations may still warrant a non-passing result rather than an exception.

**Complexity:** Medium; risk rises because every report and workbook parity surface consumes SSR.

### Requirement 3: Calendar-Consistent Attribution

| Asset | Current state | Gap tag | Required capability |
|---|---|---|---|
| `skill_metric.factor_returns_on` | Correct direction: reindex levels before differencing | Partial | Reject NaN/Inf levels, disable implicit fill, require/provide the preceding anchor, validate exact requested observation set |
| `skill_metric._align` | Inner join followed by `dropna` | Constraint | Strict report paths must not silently shorten requested samples |
| notebooks 16 and 17 | Intersect requested dates with factor coverage | Missing | Fail, extend data, or disclose separate attribution window and count in the artifact |
| notebooks 18.3 and 18.4 | Pass an already-differenced return index | Missing | Include the equity/price anchor so the first valid strategy return remains |
| `build_tear_sheet.py` and `extend_stream_2026.py` | Difference factor prices before alignment | Missing | Route through strict price-level alignment |
| release regression labels | Raw-on-raw regression fields retain CAPM names | Missing | Add explicit excess-return CAPM/Jensen path or rename all raw outputs |

The repository should retain a raw-return market-model regression because some notebooks already label it honestly. A separate CAPM/Jensen path is preferable to silently changing the existing function's meaning.

**Complexity:** Medium. The primary risk is breaking callers that currently depend on silent inner joins.

### Requirement 4: Correct Cash-Sleeve and Crisis Returns

| Asset | Current state | Gap tag | Required capability |
|---|---|---|---|
| notebook 17 cash sleeve | Uses price-only BIL and `fillna(0)` | Missing | Use total-return BIL over matching intervals and fail on unresolved required cash data |
| notebook 17 overlay/control | Both use the same cash convention | Reusable | Preserve symmetry after correcting the series |
| total-return basket panel | Ends before some strategy outputs | Constraint | Extend append-only through the required report endpoint or shorten/fail the report |
| `evaluation.crisis_analytics` | Omits entry return | Missing | Prepend the last observation before crisis start |
| `factor_workbook.rederive.equity_metrics` | Duplicates crisis omission | Missing | Correct in parity with shared evaluation or retain only under explicit legacy semantics |
| workbook crisis test | Pins incorrect boundary | Constraint | Replace fixture expectation and regenerate affected historical outputs |

Correcting the cash sleeve changes the strategy itself, not merely a label. The existing deterministic selection flow must be rerun, and downstream SJM, trio, paper, and figure artifacts may change.

**Complexity:** Large due to data extension and full downstream regeneration.

### Requirement 5: Coherent Markowitz Inputs

| Asset | Current state | Gap tag | Required capability |
|---|---|---|---|
| notebooks 18.3/18.4 opportunity set | Mixes GBp SWDA.L and USD ETFs | Missing | Convert all asset levels/returns to one disclosed base currency |
| basket calendar | LSE/NYSE complete-case intersection | Constraint | Use consistent economic intervals across assets |
| annualization | Fixed 252 on about 246.8 effective observations/year | Missing | Match the selected grid/frequency |
| strategy points | Calculated on different daily grids and from mixed-currency strategy histories | Constraint | Rebuild in the same currency/grid or omit from the corrected frontier |
| notebook 18.4 full window | Claims June 30 while benchmark/frontier input ends May 29 | Missing | Use one shared end date or label the shorter frontier |
| notebook 18.4 parity block | Disabled with a literal false condition | Missing | Reinstate an executable parity guard |

Two feasible market-calendar approaches exist:

1. **Common weekly base-currency grid:** convert price levels to USD, sample all assets and strategy values on the same weekly periods, and annualize near 52. This best satisfies consistent intervals without adding exchange-calendar infrastructure.
2. **Measured common-session grid:** retain the complete-case intersection but use its effective annualization rate. This is lower change but leaves variable interval lengths around exchange holidays.

Strategy points cannot remain on a USD frontier unless the underlying strategy histories are rebuilt in the same numeraire. The lower-blast-radius honest output is an asset-only USD frontier until those strategy histories are regenerated.

**Research needed during design:** Confirm the authoritative FX source and historical snapshot policy. The repository currently has no FX conversion helper or pinned GBPUSD total-return-compatible input.

**Complexity:** Large, medium-high risk because it can require strategy-history regeneration or a deliberate figure-scope reduction.

### Requirement 6: Dated Replay Integrity

| Asset | Current state | Gap tag | Required capability |
|---|---|---|---|
| `extend_stream_2026.py::factor_meta` | Correctly retains dates and prompts | Reusable | Preserve date identity through consumption |
| `_assemble` | Persists dated rows but creates prompt-keyed response/score maps | Missing | Key replay evidence by date and variant |
| `_ReplayScorer` and generator callbacks | Receive prompt only | Constraint | Bind the current rebalance date to the selected response and score |
| generated factor artifacts | Store parsed loadings/scores but not raw response identity | Missing | Persist prompt hash, response, origin and dated linkage |
| replay validation | Warns on duplicate prompts and accepts “later date wins” | Missing | Treat misassociation as a publication-blocking validation failure |

Duplicate prompts are legitimate because macro inputs are rounded for prompt rendering. Current artifacts prove that October 2025 consumes November values and April/May 2026 consume June values, for both PIT and non-PIT streams. Date must therefore be part of the replay identity without being injected into the anonymized prompt.

Two implementation scopes are viable:

- extension-local dated record lookup keyed by `(variant, date)`;
- a broader callback-contract change so `factor_rebalance` supplies `(prompt, as_of)`.

The local option has the lower blast radius; the shared option fixes the underlying API limitation for all future dated replay users.

**Complexity:** Medium, high correctness priority.

### Requirement 7: Producer-to-Artifact Consistency

| Asset | Current state | Gap tag | Required capability |
|---|---|---|---|
| extended-stream output flow | Writes directly to final paths throughout execution | Constraint | Prevent partial mixed generations from appearing complete |
| run header | Contains some windows and source names | Partial | Add annualization, cash, currency, performance/attribution windows and counts, SSR settings, input/output identity, and completion status |
| artifact families | Current timestamps and schemas are mixed | Missing | Regenerate as one validated release set |
| parquet/CSV mirrors | Existing luck-vs-skill mirror is stale and schema-divergent | Missing | Enforce numeric/schema parity across mirrors |
| notebook output families | Most lack sidecar provenance | Missing | Add compact provenance per output family and direct disclosures in tables/figures |
| hardcoded line descriptions | Incorrect for 10-year and max streams | Missing | Derive descriptions from actual stream configuration |

An immutable run directory plus a manifest is the simplest auditable model. A staging directory with atomic promotion preserves existing paths but adds publication plumbing. Either model must write the completed marker last and block success if any parity check fails.

**Complexity:** Large because the output graph is broad, although the underlying mechanism can remain small.

### Requirement 8: Regression and Financial-Parity Validation

Existing tests cover some primitives but not the full 15-defect matrix. The following deterministic gaps remain:

1. gross versus cash-excess SSR construction;
2. differential SSR without double-subtracting cash;
3. non-default valid alpha through `factor_loop.verify`;
4. invalid alpha, bootstrap count, window and Sharpe benchmark;
5. present-but-NaN/Inf benchmark levels;
6. first-return anchor retention;
7. strict or disclosed attribution windows;
8. total-return cash sleeve and missing-cash failure;
9. overlay/control cash-series identity;
10. crisis entry-return inclusion;
11. raw market-model versus excess-return CAPM labeling;
12. base-currency Markowitz construction;
13. cross-exchange common-period returns and matching annualization;
14. duplicate prompts on different dates retaining distinct evidence;
15. source-versus-consumed replay equality;
16. reader versus explicit legacy annualization fields;
17. consistent differential portfolio fields;
18. artifact schema/value parity and stale-generation detection;
19. figure/report provenance matching actual inputs and windows.

Notebook JSON should not be the primary unit-test target. Pure calculations should live in shared modules or small producer helpers, with notebook execution retained as an optional end-to-end validation layer. PNGs should be checked through provenance and input parity rather than brittle pixel hashes.

**Complexity:** Large but low conceptual risk; the volume comes from breadth.

---

## 3. Mapping of the 15 Confirmed Findings

| # | Confirmed defect | Owning correction boundary | Required end-to-end validation |
|---:|---|---|---|
| 1 | Duplicate prompts overwrite dated replies/scores | Dated replay record and callback boundary in `extend_stream_2026.py` or shared factor-scoring API | Distinct same-prompt dates; source-to-decision-log equality |
| 2 | Business-daily metrics annualized with 365 under generic fields | Reader-facing report producer using `metric_block` | Exact 252/elapsed formulas and explicit legacy fields |
| 3 | Portfolio SSR uses raw returns | Shared excess-return construction plus all SSR callers | BIL-excess synthetic example and regenerated verdict metadata |
| 4 | Markowitz mixes GBP/GBp and USD | Base-currency opportunity-set producer | FX-converted synthetic and artifact provenance |
| 5 | SJM cash sleeve uses price-only BIL and zero fill | Notebook 17 portfolio-return producer | Total-return cash identity, missing-data failure, regenerated strategy chain |
| 6 | Release attribution differences before reindexing | Strict price-level alignment helper | Exchange-holiday compounding and release/notebook parity |
| 7 | Raw SPY intercept labeled CAPM/Jensen alpha | Separate explicit CAPM path or field renaming | Known-alpha excess-return regression and label contract |
| 8 | Crisis analytics omit first crisis return | Shared crisis window construction | Synthetic boundary case and exported-field parity |
| 9 | Differential row mixes endpoint and spread definitions | Single differential-return producer | Compounded spread equality across total return, Sharpe and SSR |
| 10 | Report rows mix 1,845-day performance and 1,824-day attribution | Report window contract | Common window or explicit attribution window/count |
| 11 | Markowitz intersection calendar scaled by 252 | Common-period grid and annualization contract | Effective frequency matches scaling |
| 12 | `factor_returns_on` permits present missing values | Strict level validation | NaN/Inf rejection with complete labels |
| 13 | `factor_loop.verify` ignores configured SSR alpha | Verification call boundary | Non-default alpha succeeds end to end |
| 14 | SSR inference accepts invalid parameters | `ssr_inference` validation boundary | Invalid argument matrix with clear errors |
| 15 | Trio regression drops first return | Anchored benchmark-return construction | Full observation count and first-return retention |

---

## 4. Implementation Approach Options

### Option A: Minimal Extension of Existing Components

#### Scope

- Strengthen `factor_returns_on` in place.
- Validate `ssr_inference` arguments in place.
- Pass configured alpha through `factor_loop.verify`.
- Correct `crisis_analytics` and workbook parity implementation.
- Patch each report/notebook caller to construct excess returns, use total-return BIL, align calendars, and label bases.
- Fix replay locally in `extend_stream_2026.py` with date-keyed maps.

#### Advantages

- Fewest new files and interfaces.
- Fastest path to corrected outputs.
- Reuses established tests and helper modules.

#### Disadvantages

- Measurement metadata remains caller-owned and easy to omit.
- Broad notebook-specific edits can recreate drift.
- Report rows may still be assembled from unrelated scalar blocks.
- Direct writes to final artifact paths remain vulnerable to partial publication unless separately addressed.

#### Fit

Viable for a narrow hotfix, but weak against Requirement 7 and future recurrence.

### Option B: New Financial Measurement and Publication Components

#### Scope

- Introduce a new unified measurement-result contract covering return definition, annualization, cash benchmark, currency, window, observation count, attribution, and SSR settings.
- Introduce a dated replay store abstraction.
- Introduce an artifact publisher with staging, manifest, hashes, and promotion.
- Migrate all scripts and notebooks to the new surfaces.

#### Advantages

- Strongest semantic and provenance guarantees.
- Report and artifact tests become straightforward.
- Prevents mixed-window and mixed-definition rows structurally.

#### Disadvantages

- Largest migration and review surface.
- Risks speculative abstraction beyond the confirmed defects.
- More difficult to preserve existing release-parity interfaces and notebook readability.

#### Fit

Technically clean but disproportionate for this repository unless future expansion is planned.

### Option C: Hybrid Shared Invariants with Thin Producer Contracts

#### Scope

- Extend existing shared numerical primitives at their natural boundaries:
  - strict aligned level-to-return construction;
  - explicit portfolio-excess return construction;
  - explicit raw-market-model and CAPM attribution paths;
  - SSR validation and metadata;
  - boundary-inclusive crisis windows.
- Add a small dated evidence record/helper in the extended-stream producer or shared factor-scoring module.
- Add small pure row builders for reader metrics, differential metrics, attribution metadata, and provenance.
- Keep legacy metric engines but expose them only under explicit names.
- Generate into an isolated location, validate parity, then publish with one manifest/completion marker.
- Reduce notebooks to IO, assertions, rendering, and exports over shared producers.

#### Advantages

- Fixes each invariant once while avoiding a large new framework.
- Supports deterministic unit and artifact tests.
- Preserves intentional raw-regression and legacy-metric outputs under honest names.
- Provides enough metadata to prevent mixed rows and stale artifact families.

#### Disadvantages

- Requires coordinated caller migration and artifact regeneration.
- Needs careful compatibility handling for workbook parity and write-once ledgers.
- Publication staging adds some operational plumbing.

#### Fit

Best alignment with the existing codebase and the approved requirements.

---

## 5. Complexity and Risk Assessment

| Workstream | Effort | Risk | Rationale |
|---|---|---|---|
| SSR validation and configured alpha | S–M | Low | Local numerical/API changes with deterministic tests |
| Excess-return SSR migration | M | Medium | Many callers and benchmark-window dependencies |
| Strict attribution alignment and CAPM split | M | Medium | Existing silent-inner-join assumptions must be removed carefully |
| Crisis boundary correction | S–M | Medium | Simple formula change but workbook fixtures and artifacts intentionally pin the old value |
| Dated replay integrity | M | High | Current simulations consumed wrong dated evidence; corrected outputs and ledgers may change materially |
| SJM total-return cash correction | L | High | Changes strategy economics, selection objective, ledger, and downstream reports |
| Markowitz currency/calendar correction | L | Medium-High | Requires authoritative FX data and a decision about strategy-point comparability |
| Artifact regeneration and provenance | L | Medium | Broad output graph and existing mixed/stale generations |
| Full regression/parity suite | L | Low-Medium | Established testing patterns, but many boundaries to cover |

**Overall effort:** Large, approximately 1–2 focused weeks.

**Overall risk:** Medium-high. The largest risk is not implementation complexity; it is that correct inputs alter published strategies, ledgers, and downstream artifacts, requiring disciplined regeneration and honest provenance.

---

## 6. Design-Phase Decisions and Research Needed

### 6.1 Decisions that should be explicit in design

1. **SSR caller contract**
   - Keep `ssr_inference` low-level over preconstructed returns.
   - Define separate portfolio-excess and differential-return adapters.

2. **Attribution API semantics**
   - Preserve raw-return market-model regression.
   - Add a distinct excess-return CAPM/Jensen path instead of redefining the existing function silently.

3. **First-return anchor contract**
   - Prefer accepting a value/price index containing the anchor and returning strategy-date returns.
   - Avoid heuristic previous-session lookup hidden inside notebooks.

4. **Dated replay scope**
   - Choose extension-local `(variant, date)` records for minimum blast radius, or extend shared callbacks to receive `as_of` for a root API fix.
   - Do not inject dates into the anonymized PIT prompt.

5. **Markowitz time grid**
   - Prefer a common weekly USD grid for coherent cross-exchange intervals.
   - Decide whether to remove strategy points temporarily or rebuild their underlying return streams in USD.

6. **Publication model**
   - Prefer immutable run/staging output plus a manifest and completion marker written last.
   - Decide whether existing stable paths are atomically promoted or point to a completed run.

7. **Legacy metrics**
   - Preserve existing 365/vectorbt parity fields only under explicit names or in a separate legacy output.

### 6.2 Research needed

1. **Authoritative FX source and snapshot**
   - Select a reproducible GBPUSD source and define units (`USD per GBP`) and append-only storage rules.

2. **Total-return benchmark extension**
   - Extend BIL, SPY, and factor total-return levels through the required June 2026 endpoint while preserving historical values and recording overlap checks.

3. **Strategy-point currency comparability**
   - Confirm whether the underlying strategy simulations can be rerun in USD within this project or whether the corrected Markowitz plots should be asset-only.

4. **Artifact publication compatibility**
   - Inventory downstream consumers that require current stable paths before choosing immutable run directories versus atomic replacement.

5. **Short-series SSR behavior**
   - Distinguish invalid configuration from valid but insufficient data in the final error/result contract.

---

## 7. Recommended Design Direction

Proceed with **Option C: Hybrid Shared Invariants with Thin Producer Contracts**.

The design should follow this dependency order:

1. harden price-level alignment, finite-value checks, and anchor handling;
2. define total-return cash alignment and explicit portfolio-excess returns;
3. validate SSR arguments and propagate configured alpha;
4. split raw market-model attribution from excess-return CAPM attribution;
5. correct boundary-inclusive crisis calculations in root and workbook implementations;
6. replace prompt-keyed replay consumption with dated evidence identity and equality checks;
7. correct reader-facing report rows and the differential portfolio definition;
8. extend required total-return and FX inputs under append-only validation;
9. rerun notebook 17's deterministic strategy-selection and artifact chain;
10. rebuild Markowitz inputs on one currency and one common period grid;
11. generate all affected artifacts into an isolated run, validate parity/provenance, and publish only after every required check passes;
12. retain notebook execution as an end-to-end check over shared pure producers rather than the only test boundary.

This direction follows the repository's existing patterns, keeps new abstractions minimal, fixes defects at their owning boundaries, and provides enough metadata to prevent the same classes of reporting error from recurring.

---

## 8. Full Design Discovery and Synthesis Decisions

### Summary

- **Discovery scope:** Complex integration across shared finance calculations, deterministic factor replay, market-data snapshots, SJM production, notebook rendering, and immutable publication.
- **Method:** Eight specialist investigations, two adversarial challenge reviews, per-domain distillation, and one final synthesis.
- **Steering limitation:** No `.kiro/steering/` directory exists; approved requirements and repository contracts are authoritative.

### External methodology conclusions

- Portfolio Sharpe and SSR consume matched-period excess returns, not raw portfolio returns. The designated investable benchmark is pinned BIL adjusted total return and must be described as a cash benchmark rather than a theoretical risk-free rate.
- Jensen/CAPM alpha requires excess-on-excess regression in one currency and frequency. Existing mixed-local strategy returns cannot support that label, so the current release retains only explicitly named raw market-model attribution.
- Crisis return intervals include the valuation immediately preceding the first requested session.
- CAGR uses elapsed calendar time. Square-root annualization requires a regular disclosed period grid; it must not be inferred from padded or irregular common rows.
- Covariance inputs must be converted to one base currency before return construction.

References informing these conclusions include Sharpe (1994), Jensen (1968), CFA Institute GIPS 2020, Lo (2002), Federal Reserve H.10/FRED `DEXUSUK`, the official BIL fund description, and pandas resampling/pct-change documentation.

### Synthesis decisions

1. **Architecture pattern:** Contract-hardened deterministic producers with immutable, manifest-addressed release bundles.
2. **Shared numerical boundary:** Extend `evaluation.py`, `skill_metric.py`, and `ssr.py`; do not add a finance-contract framework or tagged return hierarchy.
3. **Reporting boundary:** Add one small `macro_framework/reporting.py` module so reader, legacy, and differential rows are constructed once and carry their measurement metadata.
4. **Replay boundary:** Keep scorer callbacks unchanged. Resolve immutable evidence locally by `(variant, rebalance_date)`, then prove source-to-consumption equality before publication.
5. **SJM boundary:** Move canonical selection and artifact production into `scripts/build_sjm_crowding.py`; notebook 17 becomes a renderer and validator of a completed run.
6. **Markowitz boundary:** Add `macro_framework/markowitz.py`; use a Friday-ending common weekly USD grid, USD/GBP FRED input, `52.1775` periods/year, and asset-only frontiers.
7. **Market data:** Pin immutable append-only snapshots with IDs and hashes. The initial reviewed snapshot is `market_total_return_fx_2026-06-30_v1`.
8. **Publication boundary:** Add a remediation-specific coordinator rather than a generic framework. Build and validate a local `data-v4` upload set, then publish it only through the repository's existing immutable GitHub Release-tag contract after explicit approval.
9. **Legacy compatibility:** Preserve historical GitHub Release tags and vectorbt/365 semantics under explicit schema IDs. The immutable `data-v4` GitHub Release tag and its publication manifest are authoritative for corrected outputs.
10. **Dependency policy:** Add no new runtime package. Use explicit `pct_change(fill_method=None)`, unique monotonic indexes, current pandas/statsmodels/vectorbt-compatible APIs, and existing GitHub Release tooling for public distribution.

### Resolved research items

- **FX source:** Federal Reserve H.10/FRED `DEXUSUK`, quoted USD per GBP.
- **Weekly grid:** Friday valuation cutoff at 22:00 UTC with backward-as-of price and FX observations and a three-calendar-day staleness ceiling.
- **Strategy dots:** Omitted from USD frontiers until strategies are rebuilt in USD; separate strategy figures remain labeled as legacy local-quote simulations.
- **SSR short series:** Invalid configuration raises; valid input with fewer than ten rolling Sharpe observations returns the existing non-passing insufficient-inference result.
- **CAPM policy:** No CAPM/Jensen publication for current mixed-local strategies. Raw regression is retained under unambiguous market-model names.
- **Publication compatibility:** Immutable GitHub Release tag `data-v4` is authoritative. The workbook remains an explicit-tag client; historical tags and the data-v2 default remain unchanged.

### Principal architecture risks and mitigations

- **Correct cash changes the selected SJM strategy:** use a new immutable run ID, replay the existing deterministic search, and regenerate every dependent artifact.
- **Historical raw LLM replies are unavailable:** persist deterministic reconstructed responses with `response_origin="reconstructed_from_v1_loadings"`; never claim they are raw.
- **Vendor data can revise:** pin snapshots, quantify overlap revisions, store hashes, and prohibit overwriting a snapshot ID.
- **Mixed-local strategies cannot be placed on a USD frontier:** remove the dots instead of post-hoc aggregate FX conversion.
- **Dirty development tree undermines release provenance:** allow development staging, but require a clean Git tree before finalizing the `data-v4` upload set.
- **Root/workbook numerical environments differ:** keep vendored SSR byte parity and run both locked compatibility suites explicitly.
