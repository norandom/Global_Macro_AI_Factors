# Implementation Plan

- [x] 1. Establish remediation foundations and executable coverage gates
  - _Boundary: Remediation Foundation_

- [x] 1.1 Freeze the pre-remediation baseline and immutable-release inventory (3h)
  - Capture the source commit, working-tree state, schemas, hashes, row counts, measurement windows, and producer lineage for every affected artifact.
  - Record immutable baseline identities and hashes for `data-v1`, `data-v2`, and `data-v3`; historical assets remain read-only throughout remediation.
  - Map each of the 15 confirmed defects to its owning producer and affected downstream outputs.
  - Completion is observable when a machine-readable baseline detects historical mutation, stale output, missing files, unexpected schema changes, or changed lineage.
  - _Boundary: Baseline Inventory_
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.1, 8.8_

- [x] 1.2 Verify the locked root and workbook environments (2h)
  - Reconcile declared dependencies with actual imports in the root and workbook projects without adding a new runtime package.
  - Update project declarations or lockfiles only when demonstrated dependency drift requires it.
  - Verify Python 3.12 installation and imports independently from each lockfile, including the workbook PyXLL stub path.
  - Record the exact locked commands used by all later root and workbook validation tasks.
  - Completion is observable when both clean environments install and import the remediation surfaces without undeclared cross-environment dependencies.
  - _Boundary: Dependency Manifests and Lockfiles_
  - _Requirements: 7.4, 8.2, 8.8_

- [x] 1.3 Establish the 52-criterion and 15-defect evidence matrix (2h)
  - Map every acceptance criterion to an owning task, executable check, and final validation command.
  - Map every confirmed defect to a shared-boundary regression and at least one downstream report or artifact check where the defect affects shared behavior.
  - Include deterministic fixture coverage for anchors, holiday gaps, invalid SSR settings, dated prompt collisions, cash returns, mixed currencies, stale artifacts, and manifest failures.
  - Make the matrix validator fail when any criterion or defect loses required evidence.
  - Completion is observable when all 52 criteria and all 15 defects have executable, non-placeholder evidence assignments.
  - _Boundary: Remediation Coverage Matrix_
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

- [x] 2. Harden shared return construction and attribution
  - _Boundary: Strict Return Construction and Attribution_

- [x] 2.1 Enforce anchored price-level alignment before return construction (3h)
  - Reject empty, duplicate, unordered, or timezone-aware source and requested indexes.
  - Require an explicit price anchor strictly before the first requested return date.
  - Select the anchor and every requested price level before calculating returns with filling disabled.
  - Reject absent labels and present but non-finite selected values without sorting, intersecting, filling, or dropping observations.
  - Preserve the exact requested return index and compound benchmark movement across exchange-holiday gaps.
  - Completion is observable when every valid requested strategy session receives exactly one aligned return and every malformed calendar or value fixture fails at the boundary.
  - _Boundary: Strict Return Construction_
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 8.3_

- [x] 2.2 Add exact portfolio-excess and differential-return construction (2h)
  - Construct portfolio excess returns only from portfolio and cash series with identical validated indexes and finite values.
  - Construct comparison-minus-reference returns under the same exact-index contract.
  - Preserve the original index and series length without silent sorting, intersection, filling, or dropping.
  - Cover mismatched labels, mismatched order, duplicate indexes, and non-finite values with deterministic failures.
  - Completion is observable when valid inputs produce exact element-wise subtraction and every index or value mismatch fails before a partial result is returned.
  - _Boundary: Strict Return Construction_
  - _Requirements: 1.4, 2.1, 8.3_

- [x] 2.3 Replace permissive attribution with one strict raw market-model path (3h)
  - Route market and basket regressions through exact ordered observations with finite values and explicit HAC settings.
  - Retain the complete supplied window and report native-period and annualized intercepts, HAC uncertainty, beta where applicable, residual statistics, observation count, dates, annualization, and lag metadata.
  - Publish current mixed-local-currency regressions only as raw market-model attribution, never as CAPM or Jensen alpha.
  - Preserve unavailable appraisal when residual volatility is below the established floor.
  - Completion is observable when injected-coefficient fixtures reproduce their inputs and a one-date mismatch or non-finite value raises instead of shortening the regression.
  - _Boundary: Attribution_
  - _Requirements: 3.3, 3.4, 3.6, 3.8, 3.9, 8.2, 8.3_

- [x] 2.4 Add the strict return and attribution public surface without retiring callers (1h)
  - Export the completed strict return constructors, typed attribution results, and raw market-model attribution under unambiguous names.
  - Retain the ambiguous attribution export temporarily so existing callers can migrate in a controlled later task.
  - Add package-import assertions for these completed contracts only; SSR, crisis, and reporting exports remain owned by their later implementation tasks.
  - Completion is observable when callers can import the explicit return and attribution contracts while existing callers remain operational pending repository-wide migration.
  - _Boundary: Shared Finance Public API_
  - _Depends: 2.3_
  - _Requirements: 3.8, 3.9, 7.6, 8.2_

- [x] 3. Implement independent shared-finance corrections
  - _Boundary: SSR, Crisis, and Verification Core_

- [x] 3.1 (P) Validate SSR inference and persist reproducibility metadata (3h)
  - Reject non-finite returns or benchmarks, invalid indexes, alpha outside the open interval from zero to one, boolean or non-positive bootstrap counts, invalid windows, and invalid seeds.
  - Preserve the existing insufficient-inference outcome for valid settings with fewer than ten rolling Sharpe observations.
  - Persist alpha, rolling window, periods per year, bootstrap count, seed, block length, benchmark, both Monte Carlo tail probabilities, and the underlying SSR result.
  - Preserve daily 252-period scaling and deterministic behavior for valid existing calls.
  - Completion is observable when repeated valid calls are field-for-field identical and every invalid configuration raises a parameter-specific error.
  - _Boundary: SSR Inference_
  - _Depends: 1.3_
  - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6, 2.8, 8.4_

- [x] 3.2 (P) Implement boundary-inclusive typed crisis metrics (2h)
  - Anchor each episode at the last portfolio value strictly before the requested crisis start.
  - Include returns whose right endpoints fall inside the requested crisis interval, including the return entering the first crisis session.
  - Return immutable requested and actual bounds, anchor, first return date, episode return, anchored maximum drawdown, annualized volatility, count, and annualization basis.
  - Keep the multi-portfolio adapter as a projection of the shared result rather than an independent calculation.
  - Completion is observable when hand-calculated entry-return fixtures and adapter outputs match exactly.
  - _Boundary: Crisis Metrics_
  - _Depends: 1.3_
  - _Requirements: 4.4, 4.5, 8.1, 8.2_

- [x] 3.3 Integrate cash-excess SSR and configured alpha into Factor verification (2h)
  - Require matching-session cash returns as a verification input.
  - Construct portfolio excess returns once and pass the exact configured significance level through inference and gate evaluation.
  - Keep basket residual attribution on portfolio and factor returns because it is a separate factor-model calculation.
  - Cover missing cash, mismatched cash coverage, exact excess-return input, differential no-double-subtraction, and a valid non-default alpha.
  - Completion is observable when aligned BIL returns with alpha `0.10` complete verification and omitted or misaligned cash fails before SSR inference.
  - _Boundary: Factor Verification Integration_
  - _Depends: 2.2, 3.1_
  - _Requirements: 2.1, 2.2, 2.7, 2.8, 8.2, 8.4_

- [x] 4. Establish canonical reporting semantics
  - _Boundary: Canonical Reporting Rows_

- [x] 4.1 Define immutable row schemas and measurement provenance (2h)
  - Define separate schema identities for reader-facing elapsed-time/252 metrics, explicit vectorbt/365 legacy metrics, differential metrics, attribution records, crisis records, and monthly-return tables.
  - Require portfolio identity, return basis, window, count, annualization, cash benchmark, currency basis, and source lineage on every applicable row.
  - Prohibit CAPM and Jensen field names from current mixed-local-currency schemas.
  - Completion is observable when mixed schemas or missing provenance fail before a report row can be emitted.
  - _Boundary: Reporting Schema Contracts_
  - _Requirements: 1.1, 1.2, 1.3, 1.6, 3.6, 3.8, 3.9, 7.3, 7.4, 7.6_

- [x] 4.2 Build coherent reader-facing and explicit legacy rows (3h)
  - Derive reader-facing CAGR from elapsed time and annualize volatility-based measures on the 252-day convention.
  - Build Sharpe, Sortino, and SSR from the same validated BIL-excess return stream.
  - Preserve intentional vectorbt/365 values only in separately named legacy rows.
  - Require full reader rows to use one portfolio definition, return stream, performance window, and attribution observation set.
  - Completion is observable when reader and legacy fixtures reproduce their distinct formulas without sharing ambiguous field names.
  - _Boundary: Reader and Legacy Reporting_
  - _Requirements: 1.1, 1.2, 1.3, 1.6, 2.1, 2.8, 3.6, 3.9, 7.3, 7.4, 8.2, 8.6_

- [x] 4.3 Build one coherent differential-reporting contract (2h)
  - Derive differential total return, risk statistics, and SSR inputs from one exact daily comparison-minus-reference series.
  - Preserve endpoint wealth difference only as a separately named descriptive measure.
  - Attach one common window, count, annualization basis, and inference metadata to every differential statistic.
  - Reject rows that combine endpoint wealth difference with spread-derived risk statistics as though they described one portfolio.
  - Completion is observable when changing either source line changes every differential portfolio statistic through one producer.
  - _Boundary: Differential Reporting_
  - _Requirements: 1.4, 1.5, 1.6, 2.8, 7.2, 7.4, 8.2, 8.6_

- [x] 4.4 Separate shortened attribution and project shared crisis results (2h)
  - Emit a performance-only reader row when strict attribution cannot cover the performance end date.
  - Emit shortened attribution as a separate record with its actual start, end, count, annualization, and model identity.
  - Project shared crisis results into report rows without recalculating boundaries.
  - Reject full rows whose attribution dates or observations differ from performance coverage.
  - Completion is observable when every shortened window is explicit and crisis exports equal the shared typed result.
  - _Boundary: Reporting Window Integration_
  - _Requirements: 3.6, 3.7, 4.5, 7.2, 7.4, 7.6, 8.2, 8.6_

- [x] 4.5 Validate assembled tables and monthly-return semantics (2h)
  - Validate stable column meaning, one schema per row, coherent repeated portfolio/window metadata, and required provenance.
  - Add a canonical monthly-return schema derived from the same validated strategy return stream as the corresponding performance row.
  - Reject stale generated values, mixed portfolio definitions, mixed windows, undisclosed annualization, and prohibited attribution labels.
  - Completion is observable when a valid table round-trips with stable semantics and each deliberate mixed-row mutation fails deterministically.
  - _Boundary: Reporting Table Validation_
  - _Requirements: 1.3, 3.6, 3.7, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.2, 8.6_

- [x] 4.6 Expose completed SSR, crisis, and reporting contracts (1h)
  - Export the finalized SSR inference, typed crisis result, and canonical reporting builders under explicit public names.
  - Preserve the temporary ambiguous attribution export only until the repository-wide caller migration task removes it.
  - Add package-import assertions for every completed shared contract without changing historical metric semantics.
  - Completion is observable when all new callers can import the complete canonical finance surface and existing callers remain intact pending migration.
  - _Boundary: Shared Finance Public API_
  - _Depends: 2.4, 3.1, 3.2, 4.5_
  - _Requirements: 1.6, 2.8, 3.9, 4.5, 7.3, 7.6, 8.2_

- [x] 5. Implement the immutable market snapshot producer
  - _Boundary: Market Snapshot_

- [x] 5.1 (P) Freeze the total-return and FX acquisition contract (2h)
  - Pin snapshot identity, requested coverage, symbol universe, adjusted-total-return semantics, quote currencies, quote units, and FX direction before network access.
  - Define SWDA.L as GBp scaled to GBP by `0.01` and DEXUSUK as `USD_per_GBP`, multiplied rather than inverted.
  - Define reproducible FX retrieval through an explicit ALFRED real-time vintage where `realtime_start` and `realtime_end` equal one configured vintage date.
  - Require the vintage date, request parameters, source identifiers, requested end, and raw-response hash before acquisition can start.
  - Completion is observable when acquisition refuses incomplete or ambiguous source, unit, coverage, or vintage configuration.
  - _Boundary: Market Snapshot Acquisition Contract_
  - _Depends: 1.3_
  - _Requirements: 2.1, 4.1, 4.2, 5.1, 5.4, 5.6, 7.4, 7.6_

- [x] 5.2 Acquire and normalize pinned ETF and FX source data (3h)
  - Acquire adjusted total-return levels without applying a global complete-case filter across LSE and US calendars.
  - Preserve source FX observation dates and normalize the field as `USD_per_GBP`.
  - Keep SWDA.L in local GBp so conversion provenance remains explicit at the Markowitz boundary.
  - Reject duplicate, unordered, missing, timezone-aware, or non-finite source observations instead of silently repairing them.
  - Completion is observable when normalized tables reproduce the pinned source contract and contain finite, unique, ordered observations within the declared bounds.
  - _Boundary: Market Snapshot Acquisition_
  - _Requirements: 2.1, 4.1, 4.2, 5.1, 5.6, 7.4, 7.6_

- [x] 5.3 Persist and validate append-only market snapshots (3h)
  - Write only into a new empty staging directory and refuse to overwrite a completed or non-empty snapshot identity.
  - Quantify overlap revisions against the preceding compatible snapshot, including overlap counts and changed-cell statistics.
  - Inventory schema IDs, source and vintage metadata, requested and actual coverage, quote metadata, hashes, sizes, and row counts.
  - Write the snapshot completion marker only after data, manifest, coverage, overlap, and hash validation pass.
  - Completion is observable when mutation, overwrite, non-finite data, absent revision disclosure, or premature completion is rejected.
  - _Boundary: Market Snapshot State and Manifest_
  - _Requirements: 4.2, 5.4, 5.6, 7.4, 7.5, 7.6, 8.2, 8.8_

- [x] 5.4 Add offline snapshot boundary tests (3h)
  - Exercise adjusted-total-return fields, DEXUSUK direction, immutable identities, coverage validation, overlap revisions, and byte-level hash checks with injected source frames.
  - Prove the completion marker is absent after failures and written last after a successful build.
  - Verify that no test requires network access or mutable vendor data.
  - Completion is observable when a temporary snapshot validates offline and every invariant mutation independently blocks completion.
  - _Boundary: Market Snapshot Tests_
  - _Requirements: 5.1, 5.4, 5.6, 7.4, 7.5, 8.1, 8.2, 8.5, 8.8_

- [x] 6. Implement dated Factor evidence and immutable replay
  - _Boundary: Factor Producer_

- [x] 6.1 (P) Define and validate dated Factor evidence (2h)
  - Key evidence by the natural identity of variant and rebalance date.
  - Include prompt and response text and hashes, response and score origins, parsed loadings, segment, source artifact identity, and deterministic evidence identity.
  - Validate supported variants and origins, finite nullable values, parse-state consistency, sorted dates, unique keys, canonical hashes, and complete expected date-by-variant coverage.
  - Completion is observable when one valid entry exists for every expected key and any duplicate, missing, cross-date, or hash-mismatched record blocks publication.
  - _Boundary: Dated Factor Evidence_
  - _Depends: 1.3_
  - _Requirements: 6.1, 6.2, 6.3, 7.4, 8.7_

- [x] 6.2 Persist exact per-date PIT and non-PIT evidence (3h)
  - Preserve one record per rebalance date even when rendered prompt text is identical across dates.
  - Keep anonymized PIT prompt text unchanged and preserve newly generated response bytes exactly under UTF-8 hashing.
  - Mark historical deterministic reconstructions explicitly as reconstructed from v1 loadings with source hashes, never as raw responses.
  - Retain generation and scoring failures as dated evidence with explicit origins and failure reasons.
  - Treat historical v1 artifacts as read-only inputs and write new evidence only into a new empty run staging directory.
  - Completion is observable when duplicate-prompt dates retain independent responses, scores, loadings, origins, sources, and evidence identities.
  - _Boundary: Factor Evidence Persistence_
  - _Requirements: 6.1, 6.2, 6.3, 7.3, 7.4, 7.6_

- [x] 6.3 Replace prompt-keyed replay with exact dated resolution (3h)
  - Resolve immutable evidence by exact variant and rebalance date before invoking the existing sequential rebalance path.
  - Close the selected response and score over the rebalance call without changing the scorer API or PIT prompt renderer.
  - Require the simulation-rendered prompt and hash to equal the dated evidence while permitting identical text on other dates.
  - Remove all later-date-wins and warning-and-continue behavior.
  - Completion is observable when identical prompts on two dates consume their own evidence with zero live model calls during replay.
  - _Boundary: Dated Factor Replay_
  - _Requirements: 6.1, 6.2, 6.3, 8.7_

- [x] 6.4 Add source-to-consumption replay auditing (2h)
  - Record a run-local consumption fingerprint for every simulated variant and date.
  - Compare expected and consumed keys and prove equality of prompt, response, score, loadings, and resulting decision identities.
  - Fail before targets, equity, decision logs, metrics, or completion state when a key is absent, duplicated, cross-associated, or altered.
  - Persist the successful audit summary without mutating immutable source evidence.
  - Completion is observable when a deliberate cross-date response or score swap blocks every publishable portfolio output.
  - _Boundary: Factor Replay Audit_
  - _Requirements: 6.3, 6.4, 7.4, 7.5, 8.2, 8.7, 8.8_

- [x] 6.5 Add duplicate-prompt and immutable-evidence regression tests (3h)
  - Build offline fixtures with identical prompt text on different dates but distinct responses, scores, and loadings.
  - Cover natural-key uniqueness, exact hashes, reconstructed origin, generation-failure retention, variant isolation, and deterministic evidence identities.
  - Verify missing, duplicate, cross-date, cross-variant, or hash-mismatched evidence fails before portfolio publication.
  - Completion is observable when each integrity mutation produces its own publication-blocking failure without network credentials.
  - _Boundary: Factor Evidence Tests_
  - _Depends: 6.4_
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 8.2, 8.7, 8.8_

- [x] 6.6 Integrate run-local reader, legacy, differential, and SSR records (2h)
  - Replace local performance, annualization, differential, and SSR formulas with the approved shared contracts.
  - Use completed-snapshot BIL total-return observations for portfolio excess-return SSR and preserve the daily spread for differential rows without a second cash subtraction.
  - Emit manifest-owned run-local records with distinct schema identities, exact performance windows, counts, cash benchmark, currency basis, and deterministic inference settings.
  - Keep published table assembly outside the Factor producer; these records are the immutable source consumed later by the canonical report producer.
  - Completion is observable when the Factor producer contains no duplicate reader, legacy, differential, or SSR formula and every run-local record reconstructs from its source stream.
  - _Boundary: Factor Metric Record Integration_
  - _Depends: 3.3, 4.3, 5.4, 6.4_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.8, 7.1, 7.3, 7.4, 7.6_

- [x] 6.7 Integrate run-local attribution, crisis, and window records (2h)
  - Replace local attribution and crisis calculations with the strict shared contracts.
  - Emit exact attribution and crisis records with actual dates, counts, raw market-model labels, annualization, and source lineage.
  - Emit performance-only records when strict attribution coverage is shorter and stop on invalid required benchmark coverage.
  - Keep publication projection and cross-strategy table assembly outside the Factor producer.
  - Completion is observable when no duplicate attribution or crisis implementation remains and every run-local record has one explicit financial window.
  - _Boundary: Factor Attribution and Crisis Record Integration_
  - _Depends: 4.5, 5.4, 6.6_
  - _Requirements: 3.6, 3.7, 3.9, 4.5, 7.1, 7.2, 7.4, 7.6_

- [x] 6.8 Verify Factor run-local record parity (2h)
  - Recompute representative reader, legacy, differential, SSR, crisis, and attribution fields from fixture equity, cash, and benchmark inputs.
  - Compare the producer records field-for-field within documented tolerances.
  - Reject mixed windows, stale values, endpoint/spread substitution, and missing provenance.
  - Completion is observable when every run-local record can be reproduced from its manifest-owned inputs and each deliberate mismatch fails.
  - _Boundary: Factor Reporting Tests_
  - _Depends: 6.7_
  - _Requirements: 1.3, 1.4, 1.6, 2.1, 2.8, 3.6, 4.5, 7.2, 7.4, 8.2, 8.6_

- [x] 6.9 Assemble the immutable Factor run manifest and bundle (3h)
  - Assign a stable run identity and record configuration, source commit, prompt renderer identity, model metadata, input manifests, expected evidence counts, and replay-audit result.
  - Inventory evidence, scores, loadings, targets, equity, decision logs, contrasts, metrics, and report records with hashes, schemas, counts, dates, and lineage.
  - Refuse completed destination overwrite and write the run completion marker only after every inventory and audit validation passes.
  - Completion is observable when the bundle validates from its manifest alone and any missing or altered artifact prevents completion.
  - _Boundary: Factor Run Manifest_
  - _Requirements: 6.3, 6.4, 7.1, 7.2, 7.3, 7.4, 7.5, 8.8_

- [x] 6.10 Add Factor manifest, immutability, and completion-order tests (2h)
  - Verify manifest inventory, hashes, lineage, expected key counts, completion ordering, and destination immutability.
  - Reject incomplete, stale, or tampered bundles at every downstream reader.
  - Confirm failures leave diagnosable staging output without a completion marker.
  - Completion is observable when a temporary valid bundle passes and every manifest mutation leaves the candidate incomplete.
  - _Boundary: Factor Bundle Tests_
  - _Requirements: 6.3, 6.4, 7.2, 7.4, 7.5, 8.2, 8.7, 8.8_

- [x] 7. Implement the deterministic corrected SJM producer
  - _Boundary: SJM Producer_

- [x] 7.1 Gate SJM inputs on completed immutable manifests (2h)
  - Require exact completed Factor and market-snapshot manifest identities and hashes before selection begins.
  - Validate declared coverage, unique ordered indexes, and required observations through the endpoint.
  - Treat the pinned SJM limit table as a read-only input.
  - Reject fallback to loose legacy artifacts or incomplete runs.
  - Completion is observable when valid manifests load into one typed SJM input set and every incomplete or mismatched fixture fails before calculation.
  - _Boundary: SJM Input Gate_
  - _Depends: 5.4, 6.10_
  - _Requirements: 4.2, 7.4, 7.5, 7.6, 8.2, 8.8_

- [x] 7.2 Encode and hash the frozen SJM selection protocol (2h)
  - Pin the approved development and holdout dates, development Calmar objective, CAGR budget, seed, dry rounds, iterations, signal cadence, seed configuration, limit table, and ordered mutation registry.
  - Treat `derisk_cash_pin` as the authoritative control identity for the maximum-drawdown gate and reject any conflicting control alias.
  - Hash canonical protocol fields, the read-only limit table, and the fully ordered mutation registry.
  - Completion is observable when identical constructions produce identical hashes and any value or candidate-order change invalidates the protocol.
  - _Boundary: SJM Protocol_
  - _Requirements: 4.3, 7.3, 7.4, 7.6, 8.2, 8.8_

- [x] 7.3 Construct exact factor-calendar BIL total returns (2h)
  - Select adjusted BIL levels on the Factor calendar plus one explicit preceding anchor.
  - Calculate returns only after strict level alignment and with filling disabled.
  - Reject missing labels, non-finite values, duplicate dates, or absent anchors rather than substituting zero.
  - Record cash identity, snapshot identity, anchor, dates, and count.
  - Completion is observable when cash returns exactly equal the Factor return index and retain the first Factor return.
  - _Boundary: SJM Cash Alignment_
  - _Requirements: 4.1, 4.2, 7.4, 7.6, 8.2, 8.8_

- [x] 7.4 Implement exact overlay and control return equations (2h)
  - Apply risky exposure to Factor returns and residual exposure to the aligned cash return.
  - Apply the same cash series and timing to the `derisk_cash_pin` control.
  - Preserve lagged drawdown arming and the frozen crowding-signal cadence.
  - Reject non-finite values, index differences, or exposure-bound violations before compounding.
  - Completion is observable when overlay and control returns reconstruct exactly from supplied exposures and one shared cash vector.
  - _Boundary: SJM Portfolio Equations_
  - _Requirements: 4.1, 4.2, 4.3, 7.2, 7.6, 8.2, 8.8_

- [x] 7.5 Replay the ordered deterministic mutation registry (3h)
  - Generate candidates only from the frozen seed configuration and ordered mutation registry.
  - Preserve alternate-signal ordering and all configured mutation groups exactly.
  - Record every candidate, mutation, metric, and keep-or-revert decision.
  - Do not force or restore the previous winning configuration.
  - Completion is observable when repeated runs over identical inputs produce equivalent canonical ledgers and the same visited candidate order.
  - _Boundary: SJM Selection Replay_
  - _Requirements: 7.2, 7.4, 7.6, 8.2, 8.8_

- [x] 7.6 Apply development-only gates and select the corrected winner (2h)
  - Evaluate candidates only through the approved development end using development Calmar.
  - Enforce the CAGR budget and maximum drawdown no worse than the same-cash `derisk_cash_pin` control.
  - Keep holdout observations unavailable until after the final configuration is fixed.
  - Completion is observable when every retained candidate passes both gates and changing holdout-only values cannot change selection.
  - _Boundary: SJM Selection Gates_
  - _Requirements: 4.3, 7.2, 7.4, 7.6, 8.2, 8.8_

- [x] 7.7 Assemble and validate the immutable SJM v3 run (3h)
  - Persist the selected configuration, ledger, targets, exposures, overlay returns, control returns, anchored equity, protocol hashes, and input manifest lineage.
  - Require the selected configuration in the ledger to equal the configuration represented by targets, returns, equity, and manifest provenance.
  - Reconstruct persisted returns and equity with maximum absolute error below `1e-9`.
  - Completion is observable when one validator proves hash, protocol, configuration, equation, and reconstruction equality across the complete run.
  - _Boundary: SJM Run Manifest_
  - _Requirements: 4.1, 4.2, 4.3, 7.1, 7.2, 7.4, 7.5, 7.6, 8.2, 8.8_

- [x] 7.8 Add immutable SJM staging and command behavior (2h)
  - Accept exact Factor and market manifest inputs and write only into a new run-specific staging directory.
  - Refuse completed-run overwrite or reuse of files from an incomplete prior attempt.
  - Write run data and manifest first, validate them, and write the completion marker last.
  - Completion is observable when successful builds are immutable and an injected late failure cannot be consumed as a valid run.
  - _Boundary: SJM Build Command_
  - _Requirements: 7.1, 7.4, 7.5, 7.6, 8.2, 8.8_

- [x] 7.9 Add serialized SJM protocol, equation, manifest, and reconstruction tests (3h)
  - Cover strict cash alignment, first-return retention, missing cash, overlay/control equality, protocol hashes, candidate order, gate rejection, holdout isolation, and selected-configuration provenance.
  - Verify completed-run overwrite rejection and completion-marker ordering.
  - Keep the tests in one serialized boundary so the producer and shared test file have one owner.
  - Completion is observable when protocol drift, cash substitution, forced old winners, ledger/equity disagreement, or incomplete manifests fail deterministically.
  - _Boundary: SJM Producer Tests_
  - _Requirements: 4.1, 4.2, 4.3, 7.2, 7.4, 7.5, 7.6, 8.1, 8.2, 8.8_

- [x] 8. Implement coherent USD weekly Markowitz analytics
  - _Boundary: Markowitz Producer_

- [x] 8.1 (P) Build Friday as-of USD valuation grids (3h)
  - Load only completed, hash-valid snapshots with explicit quote specifications.
  - Construct Friday valuation cutoffs at 22:00 UTC and select the latest eligible asset and FX observations at or before each cutoff.
  - Persist asset and FX source dates and reject look-ahead or observations more than three calendar days stale.
  - Convert SWDA.L as GBp divided by 100 and multiplied by `USD_per_GBP`; leave USD assets unchanged.
  - Require a complete common USD level matrix for the requested window.
  - Completion is observable when synthetic mixed-calendar inputs produce auditable common Friday values and four-day staleness fails with the offending asset and cutoff.
  - _Boundary: Weekly USD Valuations_
  - _Depends: 5.4_
  - _Requirements: 5.1, 5.2, 5.4, 5.6, 7.4, 7.6_

- [x] 8.2 Compute coherent weekly returns and annualized moments (2h)
  - Calculate returns only between consecutive common Friday valuations with filling disabled.
  - Annualize arithmetic means and covariance using exactly `365.2425 / 7`.
  - Report actual return dates, count, snapshot identity, base currency, valuation rule, and annualization.
  - Validate finite moments, symmetry, and positive semidefiniteness within a documented tolerance.
  - Completion is observable when hand-computable weekly fixtures match expected means and covariance and incomplete or invalid matrices fail.
  - _Boundary: Markowitz Moments_
  - _Requirements: 5.2, 5.3, 5.4, 5.6, 7.4, 7.6, 8.5_

- [x] 8.3 Produce feasible long-only frontiers with diagnostics (3h)
  - Solve deterministic attainable target returns under fully invested zero-to-one weight bounds.
  - Retain success, status, message, iterations, objective, budget residual, target residual, bound violation, and weights for every target.
  - Surface failed or infeasible targets explicitly rather than silently dropping them.
  - Validate every publishable point against its stored weights, moments, and residual tolerances.
  - Completion is observable when all published points are feasible and an induced solver failure remains visible in the result.
  - _Boundary: Markowitz Frontier_
  - _Requirements: 5.1, 5.3, 5.4, 5.6, 7.4, 7.5, 7.6, 8.5, 8.8_

- [x] 8.4 Add offline Markowitz numerical and validation tests (3h)
  - Cover GBp scaling, multiplication by USD-per-GBP, and a counterexample that catches FX inversion.
  - Cover cross-exchange holidays, Friday source dates, no look-ahead, exact three-day acceptance, four-day rejection, and complete weekly intervals.
  - Verify weekly annualization, finite symmetric positive-semidefinite moments, all frontier residuals, and every stored weight vector.
  - Completion is observable when the focused suite runs without network access and deliberate direction, staleness, scaling, coverage, or feasibility regressions fail.
  - _Boundary: Markowitz Tests_
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.6, 8.1, 8.2, 8.5, 8.8_

- [x] 9. Build canonical report tables and locale projections
  - _Boundary: Canonical Report Producer_

- [x] 9.1 Add manifest-aware canonical input loading (2h)
  - Read Factor, SJM, market, and Markowitz inputs only through completed manifests and verified hashes.
  - Reject loose, incomplete, stale, or schema-incompatible inputs.
  - Preserve one explicit owner for assembled report tables and keep strategy producers limited to immutable strategy outputs and run-local records.
  - Completion is observable when every canonical row traces to verified producer lineage and tampered inputs fail before table assembly.
  - _Boundary: Canonical Report Input Gate_
  - _Depends: 4.5, 6.10, 7.9, 8.4_
  - _Requirements: 7.1, 7.2, 7.4, 7.5, 7.6, 8.2, 8.8_

- [x] 9.2 Assemble canonical Factor and AI-variant tables (3h)
  - Load and project the completed Factor bundle's manifest-owned run-local reader, legacy, differential, attribution, crisis, and SSR records.
  - Validate those records against their declared source streams and preserve performance-only rows plus separate shortened attribution records.
  - Produce the canonical cross-variant and AI-variant publication tables without independently recalculating a second Factor row family.
  - Completion is observable when representative published rows equal the validated run-local records and trace to their Factor manifest lineage.
  - _Boundary: Canonical Factor Reports_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.8, 3.6, 3.7, 3.9, 4.5, 7.1, 7.2, 7.3, 7.4, 7.6_

- [x] 9.3 Assemble canonical SJM report tables (2h)
  - Build SJM performance, holdout, tail, SSR, crisis, and raw market-model rows from the completed SJM v3 run.
  - Carry the selected-configuration hash, protocol identity, cash benchmark, input manifests, dates, counts, and annualization.
  - Produce the canonical SJM tear-sheet report consumed by notebook 17.
  - Completion is observable when every displayed SJM field can be reproduced without notebook-local finance formulas.
  - _Boundary: Canonical SJM Reports_
  - _Requirements: 3.6, 3.7, 3.9, 4.1, 4.2, 4.3, 4.5, 7.1, 7.2, 7.3, 7.4, 7.6, 8.2, 8.8_

- [x] 9.4 Assemble trio, static-window, and dashboard tables (3h)
  - Produce canonical trio, static buy-and-hold window, and window-dashboard tables with exact portfolio and window identity.
  - Own `tear_sheet_trio_ext2026.csv`, the static-window tables, the dashboard tables, and their canonical schema definitions outside notebooks.
  - Require repeated portfolio/window rows to agree on dates, count, cash benchmark, currency, annualization, and values.
  - Completion is observable when notebooks 15.2, 15.3, 18.2, and the final dashboard can render without recalculating canonical metrics.
  - _Boundary: Canonical Trio and Window Reports_
  - _Requirements: 1.1, 1.2, 1.3, 1.6, 2.1, 2.8, 3.6, 3.7, 4.5, 7.1, 7.2, 7.3, 7.4, 7.6, 8.6_

- [x] 9.5 Persist canonical ten-year and maximum-window Markowitz tables (2h)
  - Produce canonical ten-year and maximum-supported USD weekly moment and frontier tables outside notebooks.
  - Include snapshot identity, base currency, requested and actual windows, weekly count, annualization, source-date hashes, asset moments, weights, and solver diagnostics.
  - Produce `tear_sheet_trio_10y.csv`, `tear_sheet_trio_max.csv`, and their German-locale source schemas without including mixed-local strategy points on USD frontiers.
  - Completion is observable when notebooks 18.3 and 18.4 can consume immutable tables and perform no optimization or currency conversion.
  - _Boundary: Canonical Markowitz Reports_
  - _Depends: 8.4_
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 7.1, 7.2, 7.3, 7.4, 7.6, 8.5_

- [x] 9.6 Assemble canonical risk, attribution, crisis, and monthly-return tables (2h)
  - Build shared risk, attribution, crisis, differential, and monthly-return tables from the same validated streams used by portfolio rows.
  - Keep shortened attribution windows separate and preserve boundary-inclusive crisis values.
  - Require source schema and portfolio/window identity on every record.
  - Completion is observable when all secondary tables reconcile to their corresponding canonical portfolio rows.
  - _Boundary: Canonical Auxiliary Reports_
  - _Requirements: 1.3, 1.4, 1.5, 2.8, 3.6, 3.7, 3.9, 4.5, 7.1, 7.2, 7.4, 7.6, 8.6_

- [x] 9.7 Generate deterministic US and German locale mirrors (3h)
  - Generate locale projections only from canonical in-memory tables, never through independent financial calculations.
  - Use deterministic ordering, dates, null handling, and eight-decimal formatting.
  - Produce comma/dot US files and semicolon/comma German files with required matching basenames.
  - Verify round-trip numeric parity within `5e-9`.
  - Completion is observable when every cataloged canonical table has required mirrors and a locale parser reproduces the source values.
  - _Boundary: Locale Mirror Producer_
  - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 7.1, 7.2, 7.3, 7.6, 8.2, 8.6_

- [x] 9.8 Add canonical report and mirror integration tests (3h)
  - Verify schema isolation, exact windows, row provenance, monthly-return parity, shortened attribution, crisis equality, and locale round trips.
  - Reject stale values, mixed definitions, mixed annualization, missing lineage, and incompatible completed manifests.
  - Confirm report producers and locale exporters are the sole owners of canonical tables and mirrors; notebook no-write behavior is verified after notebook migration.
  - Completion is observable when every canonical family passes offline and each injected report or mirror defect fails deterministically.
  - _Boundary: Canonical Report Tests_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.8, 3.6, 3.7, 3.9, 4.5, 7.2, 7.4, 7.5, 8.2, 8.6, 8.8_

- [x] 10. Implement publication contracts and workbook integration tooling
  - _Boundary: Publication Contract and Workbook_

- [x] 10.1 Freeze the local `data-v4` asset catalog (2h)
  - Define exact canonical payloads, allowed projections, figures, formatted reports, schemas, locale requirements, compatibility paths, producers, and lineage.
  - Reject duplicate public basenames, missing required assets, undeclared aliases, and inputs from unexpected producer manifests.
  - Define the non-recursive checksum rule: the manifest inventories payload assets and `SHA256SUMS` covers payloads plus the final manifest but never itself.
  - Exclude any release-directory `COMPLETED` marker; completion is represented only by the final manifest.
  - Add the local release-asset directory to build-artifact ignore policy without ignoring source, tests, contracts, or documentation.
  - Completion is observable when the catalog produces one deterministic, collision-free inventory without consulting a mutable current-release pointer.
  - _Boundary: Publication Asset Catalog_
  - _Requirements: 7.1, 7.3, 7.4, 7.5, 7.6, 8.2, 8.6, 8.8_

- [x] 10.2 (P) Add manifest-aware explicit-tag release integrity checks (3h)
  - Keep release resolution exclusively on explicit immutable tags with no repository-local fallback.
  - For `data-v4`, reject unlisted asset names and verify downloaded bytes against manifest hashes before caching or loading.
  - Invalidate cache entries after integrity failure and record tag, URL, checksum, cache status, and verification state without exposing credentials.
  - Preserve direct historical behavior for `data-v1` through `data-v3`.
  - Completion is observable when mocked valid assets load, mismatched or unlisted assets fail, and no stale cache survives a failed integrity check.
  - _Boundary: Workbook Release Client_
  - _Depends: 10.1_
  - _Requirements: 7.3, 7.4, 7.5, 8.2, 8.6, 8.8_

- [x] 10.3 (P) Register tag-aware `data-v4` workbook schemas (3h)
  - Register canonical reader, legacy, differential, attribution, crisis, Factor, SJM, Markowitz, manifest, and compatibility schemas.
  - Bind schemas to explicit tags so historical contracts remain unchanged.
  - Validate required windows, counts, annualization, cash benchmark, currency basis, schema identity, and SSR settings.
  - Use compact schema-true fixtures rather than copying the release payload into tests.
  - Completion is observable when all `data-v4` fixtures validate, historical fixtures remain unchanged, and cross-tag substitution fails with field-specific errors.
  - _Boundary: Workbook Data Contracts_
  - _Depends: 4.5, 9.8, 10.1_
  - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 2.8, 3.6, 3.7, 3.8, 3.9, 4.5, 5.4, 5.5, 5.6, 7.2, 7.3, 7.4, 8.2, 8.6_

- [x] 10.4 (P) Synchronize workbook SSR and crisis primitives (2h)
  - Copy the finalized root SSR implementation using the existing byte-equivalent vendor convention.
  - Correct workbook crisis boundaries to include the pre-window anchor while retaining unrelated historical vectorbt/365 and day-zero behavior.
  - Verify cash-excess metadata, non-default alpha, invalid settings, deterministic verdicts, crisis parity, and root/vendor identity.
  - Completion is observable when root and workbook results match exactly and historical non-crisis fixtures remain unchanged.
  - _Boundary: Workbook Finance Parity_
  - _Depends: 3.1, 3.2_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 4.4, 4.5, 7.2, 7.3, 8.2, 8.4, 8.6_

- [x] 10.5 Migrate workbook step assembly to corrected `data-v4` semantics (3h)
  - Construct `data-v4` SSR from portfolio returns minus aligned BIL total returns and surface all deterministic inference metadata.
  - Consume canonical reader, legacy, differential, attribution, crisis, Factor, SJM, and Markowitz tables without local alternative calculations.
  - Preserve explicit historical-tag behavior and immutable S0–S5 audit semantics.
  - Surface missing cash, shortened attribution, incomplete windows, schema errors, and manifest failures as visible checks.
  - Completion is observable when representative `data-v4` views agree with canonical published rows while historical views retain their captured results.
  - _Boundary: Workbook Step Assembly_
  - _Depends: 10.2, 10.3, 10.4_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.8, 3.6, 3.7, 3.8, 3.9, 4.4, 4.5, 5.4, 5.5, 5.6, 7.2, 7.3, 7.4, 7.5, 8.2, 8.4, 8.5, 8.6, 8.8_

- [x] 10.6 Add fixture-driven S0–S5 and historical-isolation tests (3h)
  - Exercise canonical `data-v4` table loading, cash-excess SSR, crisis values, attribution coverage, Factor, SJM, and Markowitz views.
  - Verify historical tags remain isolated from corrected schemas and calculations.
  - Reject missing, stale, corrupt, cross-tag, or unmanifested assets.
  - Completion is observable when `data-v4` fixtures match canonical values and all historical fixtures pass unchanged.
  - _Boundary: Workbook Integration Tests_
  - _Requirements: 2.1, 2.8, 4.4, 4.5, 5.4, 5.5, 5.6, 7.2, 7.4, 7.5, 8.2, 8.4, 8.5, 8.6, 8.8_

- [x] 10.7 Preserve the `data-v2` default and enable explicit `data-v4` selection (2h)
  - Keep generated thesis workbooks pinned to `data-v2`.
  - Require explicit user selection of `data-v4` and expose active tag, publication identity, manifest status, source URLs, and per-asset provenance.
  - Create a fresh release client when tags change and prohibit cross-tag cache substitution.
  - Completion is observable when default-generation tests remain on `data-v2` and explicit `data-v4` selection uses manifest-verified loading.
  - _Boundary: Workbook Tag Selection_
  - _Requirements: 7.3, 7.4, 7.5, 8.2, 8.6, 8.8_

- [x] 10.8 Implement incomplete candidate staging and direct validation (3h)
  - Stage only cataloged assets into a new empty destination and verify every source against its completed producer manifest.
  - Write a provisional publication manifest with `completed=false`; prohibit `SHA256SUMS` and final completion claims.
  - Provide direct staging validators for schema, values, windows, inventory, lineage, duplicate basenames, extra files, and source hashes.
  - Refuse overwrite of an existing or completed candidate and leave failures diagnosable but incomplete.
  - Completion is observable when valid fixture assets stage deterministically and every missing, extra, stale, corrupt, or unowned asset blocks staging validation.
  - _Boundary: Release Publisher Staging_
  - _Depends: 10.1_
  - _Requirements: 7.1, 7.2, 7.4, 7.5, 7.6, 8.2, 8.6, 8.8_

- [x] 10.9 Implement manifest finalization, checksums, read-only verification, and atomic promotion (3h)
  - Replace a validated provisional manifest exactly once with canonical `completed=true` metadata.
  - Generate `SHA256SUMS` last, covering payloads plus the final manifest and excluding itself.
  - Verify finalized bytes through manifest and checksum readers without mutation, then atomically promote to an absent final local destination.
  - Refuse overwrite and leave the final destination absent after any verification or promotion failure.
  - Completion is observable when a fixture candidate self-verifies and promotes atomically while every finalization, checksum, or overwrite fault remains incomplete.
  - _Boundary: Release Publisher Finalization_
  - _Depends: 10.8_
  - _Requirements: 7.1, 7.2, 7.4, 7.5, 8.2, 8.6, 8.8_

- [x] 10.10 Add offline clean-room upload-set smoke tooling (2h)
  - Serve a finalized local candidate through a temporary HTTP endpoint that behaves like the public release asset endpoint.
  - Run the real explicit-tag release client, checksum verification, schema loaders, and representative S0–S5 builds from a clean temporary directory.
  - Ensure no repository data path or mutable current-release pointer is available.
  - Completion is observable when a fixture upload set passes independently and any missing, extra, stale, corrupt, or schema-invalid asset fails.
  - _Boundary: Local Release Smoke Tooling_
  - _Depends: 10.9_
  - _Requirements: 7.2, 7.4, 7.5, 8.2, 8.6, 8.8_

- [x] 10.11 Implement read-only public-release smoke-test hooks (3h)
  - Accept only an explicit tag and public base URL and provide no release-creation or upload capability.
  - Enumerate public assets, compare them with the publication manifest, verify checksums and schemas, and use a clean cache.
  - Verify immutable historical-tag metadata or frozen hashes.
  - Treat absent public `data-v4` as a clear not-yet-published state.
  - Completion is observable when mocked endpoints cover success, missing assets, duplicates, hash mismatch, historical stability, and not-yet-published behavior.
  - _Boundary: Public Release Verification Tooling_
  - _Depends: 10.10_
  - _Requirements: 7.2, 7.4, 7.5, 8.2, 8.6, 8.8_

- [ ] 11. Migrate callers and notebooks to presentation-only consumption
  - _Boundary: Caller and Notebook Migration_

- [ ] 11.1 Migrate notebook 14 to completed dated Factor inputs (2h)
  - Read completed Factor evidence, replay audit, and canonical contrast tables instead of producing canonical evidence or scores.
  - Preserve prompt alternatives and contamination narrative without changing prompt identities.
  - Require manifest and source-to-consumption validation before rendering.
  - Completion is observable when a network-disabled execution writes presentation outputs only and leaves canonical Factor artifacts unchanged.
  - _Boundary: Notebook 14 Presentation_
  - _Depends: 6.10, 9.8_
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 7.1, 7.4, 7.5, 7.6, 8.2, 8.7, 8.8_

- [ ] 11.2 Migrate notebook 15.3 to canonical extended-window tables (2h)
  - Consume canonical static-window and trio tables instead of calculating or writing them.
  - Validate the canonical US and German static-window files, labels, dates, counts, and source hashes.
  - Retain presentation-only `nb15_3_long_window_equity.png` and `nb15_3_sharpe_vs_window.png`.
  - Completion is observable when the notebook renders without owning canonical financial tables.
  - _Boundary: Notebook 15.3 Presentation_
  - _Requirements: 1.1, 1.2, 1.3, 1.6, 2.1, 2.8, 3.6, 3.7, 3.9, 4.5, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [ ] 11.3 Convert notebook 17 into an SJM renderer and validator (3h)
  - Remove market acquisition, selection, limit creation, ledger production, and canonical SJM writes.
  - Load the completed SJM v3 run and canonical SJM report rows.
  - Validate selected configuration, protocol hashes, total-return BIL identity, reconstruction, and report parity before display.
  - Completion is observable when the notebook executes offline and manifest mutation causes a visible failure rather than local recalculation.
  - _Boundary: Notebook 17 Presentation_
  - _Requirements: 3.6, 3.7, 3.9, 4.1, 4.2, 4.3, 4.4, 4.5, 7.1, 7.2, 7.4, 7.5, 7.6, 8.2, 8.8_

- [ ] 11.4 Migrate notebook 16 to canonical AI-variant rows (2h)
  - Replace notebook-local metrics and attribution with the completed canonical AI-variant report.
  - Retain plots and explanatory comparisons while validating the canonical report schema and hash.
  - Remove notebook ownership of the AI-variant tear-sheet CSV.
  - Completion is observable when every displayed value traces to one canonical row and no notebook-local finance calculation remains.
  - _Boundary: Notebook 16 Presentation_
  - _Requirements: 1.1, 1.2, 1.3, 1.6, 2.1, 2.8, 3.6, 3.7, 3.9, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [ ] 11.5 Migrate notebook 15.2 to the canonical trio (2h)
  - Load corrected Factor, SJM v3, and static rows from completed manifests and canonical trio tables.
  - Remove ownership of the trio table, German mirror, and canonical equity mirrors.
  - Keep gallery rendering and presentation-only exports with explicit common-window assertions.
  - Completion is observable when displayed rows and charts match canonical sources without hidden legacy or mixed-window substitutions.
  - _Boundary: Notebook 15.2 Presentation_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.8, 3.6, 3.7, 3.9, 4.5, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.6_

- [ ] 11.6 Migrate notebook 15.4 to presentation-only paper and thesis exports (2h)
  - Reshape only the completed canonical trio table into paper and thesis outputs.
  - Produce US and German CSV, Markdown, and TeX derivatives with source-table provenance.
  - Prohibit financial recalculation or SJM configuration selection.
  - Completion is observable when every exported value round-trips to the canonical trio and stale source hashes block generation.
  - _Boundary: Notebook 15.4 Presentation_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.8, 3.6, 3.7, 3.9, 4.5, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.6_

- [ ] 11.7 Migrate notebook 18.3 to canonical ten-year USD frontier data (3h)
  - Consume canonical ten-year USD weekly moments, frontier diagnostics, and trio tables.
  - Remove notebook-local currency conversion, daily returns, 252 scaling, and optimization.
  - Render only temporary `nb18_3_markowitz_plane_10y.png` and `nb18_3_panels_10y.png` during validation.
  - Validate `tear_sheet_trio_10y.csv` and its German mirror without rewriting them.
  - Keep the USD frontier asset-only and label separate strategy panels as legacy local-quote simulation.
  - Completion is observable when displayed window, count, base currency, annualization, and source hash match the canonical producer.
  - _Boundary: Notebook 18.3 Presentation_
  - _Depends: 8.4, 9.5_
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.5_

- [ ] 11.8 Migrate notebook 18.4 to canonical maximum-window USD frontier data (3h)
  - Consume the canonical maximum-supported USD weekly moments, frontier diagnostics, and trio tables.
  - Remove notebook-local optimization and reinstate an executable window-parity guard.
  - Render only temporary `nb18_4_markowitz_plane_max.png` and `nb18_4_panels_max.png` during validation.
  - Validate `tear_sheet_trio_max.csv` and its German mirror without rewriting them.
  - Stop or label shorter coverage rather than claiming unavailable endpoint data.
  - Completion is observable when all displayed boundaries and canonical tables agree exactly.
  - _Boundary: Notebook 18.4 Presentation_
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.5, 8.6_

- [ ] 11.9 Migrate notebook 18.2 to canonical window-dashboard data (2h)
  - Load canonical window-dashboard tables and completed benchmark lineage.
  - Remove ownership of both canonical dashboard locale files.
  - Render `nb18_2_risk_return_map.png` and `nb18_2_ratio_ladder.png` from validated rows.
  - Reject source date, count, window, or hash mismatches before plotting.
  - Completion is observable when dashboard figures reproduce from canonical rows without private metric paths.
  - _Boundary: Notebook 18.2 Presentation_
  - _Depends: 11.8_
  - _Requirements: 1.1, 1.2, 1.3, 1.6, 3.6, 3.7, 4.5, 5.4, 5.5, 5.6, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.6_

- [ ] 11.10 Migrate the final trio dashboard to corrected canonical sources (2h)
  - Replace superseded SJM inputs with the completed SJM v3 line.
  - Validate every plotted point and KPI against canonical trio rows, including window, count, cash benchmark, currency, and source hashes.
  - Retain `nb18_risk_return_maps.png`, `nb18_ratio_ladder.png`, and `nb18_metric_profile.png` as presentation artifacts only.
  - Remove stale predecessor paths from the current output inventory.
  - Completion is observable when all dashboard values trace to one completed canonical source and no canonical table is written by the notebook.
  - _Boundary: Final Trio Dashboard Presentation_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.8, 3.6, 3.7, 3.9, 4.5, 5.4, 5.5, 5.6, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.6_

- [ ] 11.11 Retire the ambiguous attribution API after caller migration (2h)
  - Search all root, script, workbook, test, and notebook callers for the ambiguous attribution name.
  - Include foundation import tests and every migrated presentation notebook in the zero-reference check.
  - Remove the old public export only after all callers use the explicit raw market-model name.
  - Do not add a compatibility alias that preserves ambiguous semantics.
  - Completion is observable when repository-wide search and package-import tests find every canonical public name and no remaining ambiguous attribution reference.
  - _Boundary: Shared Finance API Retirement_
  - _Depends: 10.7, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 11.10_
  - _Requirements: 1.6, 3.8, 3.9, 7.1, 7.6, 8.2_

- [ ] 12. Run provisional producer and presentation integration
  - _Boundary: Provisional End-to-End Integration_

- [ ] 12.1 Build and validate a provisional market-snapshot candidate (3h)
  - Before writing staging output, verify the configured yfinance and ALFRED/FRED endpoints are reachable and any required source credentials are present without logging secrets.
  - Freeze the actual requested dates and ALFRED vintage configuration before the one-time network acquisition.
  - Persist immutable raw yfinance/FRED response bytes or canonical source frames with source identity, request parameters, retrieval time, path, and SHA-256 so the snapshot can be rebuilt offline after approval.
  - Write a completed candidate under an isolated provisional identity and root that cannot be mistaken for or consumed as the final reviewed snapshot.
  - Validate source metadata, persisted source-artifact hashes, overlap revisions, coverage, output hashes, and completion ordering.
  - Completion is observable when the provisional snapshot validates independently and any unavailable required interval blocks downstream candidate production.
  - _Boundary: Market Snapshot Production_
  - _Requirements: 2.1, 4.1, 4.2, 5.1, 5.2, 5.4, 5.6, 7.4, 7.5, 8.2, 8.5, 8.8_

- [ ] 12.2 Build the provisional corrected Factor-bundle candidate (3h)
  - Before writing staging output, verify the existing provider runtime, required credentials, configured model availability, and network reachability for dates that require raw response generation; fail without logging secrets.
  - Keep deterministic reconstruction-only paths offline and do not change provider, model, or prompt behavior.
  - Write under an isolated provisional identity and root that cannot be consumed as the final reviewed Factor bundle.
  - Generate or reconstruct dated evidence according to declared origins.
  - Run exact dated replay, source-to-consumption validation, canonical finance rows, and bundle inventory validation.
  - Write completion only after all evidence, audit, metric, hash, and manifest checks pass.
  - Completion is observable when a completed Factor bundle is self-validating and contains one consumed record per expected variant and date.
  - _Boundary: Factor Run Production_
  - _Requirements: 2.1, 6.1, 6.2, 6.3, 6.4, 7.1, 7.4, 7.5, 7.6, 8.2, 8.7, 8.8_

- [ ] 12.3 Build the provisional corrected SJM candidate (3h)
  - Replay the frozen protocol against the completed provisional Factor and market candidates.
  - Write under an isolated provisional identity and root; reserve `sjm_crowding_v3_total_return_bil` for the final clean-source run.
  - Validate same-cash overlay/control equations, development-only selection, protocol hashes, selected configuration, and reconstruction.
  - Write completion only after every run invariant passes.
  - Completion is observable when the SJM v3 run validates from its manifest and contains no forced reference to the prior winner.
  - _Boundary: SJM Run Production_
  - _Requirements: 4.1, 4.2, 4.3, 7.1, 7.2, 7.4, 7.5, 7.6, 8.2, 8.8_

- [ ] 12.4 Produce provisional canonical Markowitz and report outputs (3h)
  - Build ten-year and maximum-window Markowitz tables from the completed snapshot.
  - Build Factor, SJM, trio, window, risk, attribution, crisis, differential, and monthly-return tables from completed manifests.
  - Generate deterministic locale mirrors and validate source-table parity.
  - Completion is observable when every canonical output has one owning producer, complete lineage, and no stale SJM v2 current-line reference.
  - _Boundary: Canonical Output Production_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.8, 3.6, 3.7, 3.9, 4.5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 7.1, 7.2, 7.3, 7.4, 7.6_

- [ ] 12.5 Execute provisional notebook wave A with network disabled (3h)
  - Execute temporary copies in order: notebook 14, notebook 15.3, notebook 17, then notebook 16.
  - Require completed manifests, fixed seeds, no network access, and no canonical producer writes.
  - Compare displayed values and source hashes to canonical inputs after each notebook.
  - Completion is observable when all four notebooks execute successfully and canonical evidence, snapshot, ledger, equity, and report hashes remain unchanged.
  - _Boundary: Notebook Execution Wave A_
  - _Requirements: 6.4, 7.1, 7.2, 7.4, 7.5, 7.6, 8.2, 8.6, 8.7, 8.8_

- [ ] 12.6 Execute provisional notebook wave B with network disabled (3h)
  - Execute temporary copies in order: notebook 15.2, notebook 15.4, notebook 18.3, then notebook 18.4.
  - Validate locale exports, report parity, USD weekly frontier metadata, asset-only frontiers, and executable maximum-window parity.
  - Keep all generated figures in isolated temporary output roots.
  - Completion is observable when all four notebooks execute successfully and their temporary outputs match canonical tables.
  - _Boundary: Notebook Execution Wave B_
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 7.1, 7.2, 7.4, 7.5, 7.6, 8.2, 8.5, 8.6, 8.8_

- [ ] 12.7 Execute provisional notebook wave C with network disabled (2h)
  - Execute temporary copies in order: notebook 18.2, then the final trio dashboard.
  - Validate source hashes, windows, counts, labels, and stale-artifact removal.
  - Keep canonical writes prohibited.
  - Completion is observable when both dashboards execute and every output traces to the same completed provisional source family.
  - _Boundary: Notebook Execution Wave C_
  - _Requirements: 1.3, 3.6, 3.7, 5.4, 5.5, 5.6, 7.1, 7.2, 7.4, 7.5, 7.6, 8.2, 8.6, 8.8_

- [ ] 13. Validate provisional outputs and freeze a committed source
  - _Boundary: Validation and Source Freeze_

- [ ] 13.1 Run the targeted root regression suite (2h)
  - Run the focused evaluation, return-construction, SSR, Factor verification, dated replay, reporting, snapshot, SJM, Markowitz, and publication-artifact tests.
  - Require each confirmed defect to pass at its owning boundary and each shared defect to pass a downstream check.
  - Record deterministic seeds and exact commands for later publication metadata.
  - Completion is observable when the targeted command passes with no skipped required criterion or defect check.
  - _Boundary: Focused Root Validation_
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

- [ ] 13.2 Run the locked workbook suite independently (2h)
  - Install and test from the workbook lockfile without relying on root-environment dependencies.
  - Verify release integrity, tag-aware schemas, finance parity, step assembly, S0–S5 behavior, explicit tag switching, and historical isolation.
  - Retain Excel-only UI behavior in the established manual boundary rather than treating Linux stubs as UI automation.
  - Completion is observable when the complete locked workbook suite passes independently.
  - _Boundary: Workbook Validation_
  - _Requirements: 2.8, 4.5, 7.2, 7.4, 8.2, 8.4, 8.8_

- [ ] 13.3 Validate cross-output tabular parity and stale-output detection (3h)
  - Compare Parquet, US CSV, German CSV, JSON, Markdown, TeX, and approved projections using the documented tolerance.
  - Verify schemas, windows, counts, annualization, cash benchmark, currency, run identities, inference settings, and lineage.
  - Inject stale, missing, duplicated, mixed-window, mixed-definition, and hash-mismatched cases.
  - Completion is observable when valid outputs agree and every injected defect reports path, schema, field, expected value, actual value, tolerance, and lineage.
  - _Boundary: Tabular Artifact Validation_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.8, 3.6, 3.7, 3.9, 4.5, 5.4, 5.5, 5.6, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.2, 8.5, 8.6, 8.8_

- [ ] 13.4 Validate and inventory provisional presentation artifacts (3h)
  - Compare executed notebook values, titles, windows, dimensions, and source-table hashes without pixel hashing.
  - Verify notebook 18.3 and 18.4 use distinct windows and asset-only USD frontiers.
  - Verify the candidate inventory includes the intended notebook 15.3, 15.4, 18.2, 18.3, 18.4, and final-dashboard outputs and excludes stale predecessors.
  - Keep validated presentation outputs in isolated provisional roots; final staging and promotion remain exclusively in Task 15.
  - Completion is observable when every presentation artifact is catalog-ready and any stale or mismatched output is excluded from later staging.
  - _Boundary: Presentation Artifact Validation_
  - _Requirements: 5.4, 5.5, 5.6, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.2, 8.5, 8.6, 8.8_

- [ ] 13.5 Run the full locked root test suite (2h)
  - Run the complete root suite only after focused, workbook, notebook, and artifact validations pass.
  - Treat failures, required-test skips, project-policy warnings, or lock drift as blockers.
  - Record command, lock hash, result, and duration.
  - Completion is observable when the entire repository suite passes against the provisional corrected outputs.
  - _Boundary: Full Root Validation_
  - _Requirements: 7.5, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

- [ ] 13.6 Reconcile the executable coverage matrix (2h)
  - Update the matrix only from actual test, producer, notebook, and artifact evidence.
  - Require both shared and downstream evidence for shared defects.
  - Leave publication-finalization entries pending until the local candidate is built.
  - Completion is observable when no non-publication criterion or confirmed defect remains without passing evidence.
  - _Boundary: Remediation Coverage Gate_
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

> **External approval prerequisite before Task 14:** The user reviews and commits the intended source and tracked generated changes. Automation must not create, clean, discard, or amend that commit.

- [ ] 13.7 Verify the reviewed source-and-input freeze (2h)
  - Accept a user-provided reviewed commit SHA and verify the checkout is clean without modifying it.
  - Freeze the approved provisional snapshot request/vintage and every immutable raw-source artifact identity, retained location, and SHA-256; also freeze the dated Factor evidence hash and origins plus provisional snapshot/Factor/SJM manifest IDs and hashes alongside the source SHA.
  - Record that any later vendor, vintage, provider-response, model, evidence, coverage, or hash drift returns the project to provisional generation and human review rather than flowing into final outputs.
  - Verify no unrelated user changes are hidden, discarded, or included outside the reviewed commit.
  - Completion is observable when one reviewed freeze record pins source and all external inputs required for the final rebuild.
  - _Boundary: Clean Source Preflight_
  - _Requirements: 7.4, 7.5, 8.8_

- [ ] 14. Rebuild and revalidate from the frozen clean source
  - _Boundary: Clean-Source Reproducibility_

- [ ] 14.1 Build the final market snapshot from approved pinned bytes (2h)
  - Consume the exact reviewed raw yfinance/FRED bytes or canonical source frames from their frozen retained locations; content hashes alone are not accepted as rebuild inputs and no vendor reacquisition is permitted.
  - Build the reserved final snapshot identity and compare every input and normalized output against the approved provisional candidate.
  - Treat any vendor, vintage, coverage, or hash difference as a hard return to Tasks 12–13.7; never assign a new final identity after approval.
  - Re-run coverage, revision, hash, and completion checks.
  - Completion is observable when the final snapshot is input-equivalent to the approved candidate and identifies the frozen source commit.
  - _Boundary: Clean-Source Snapshot Rebuild_
  - _Depends: 13.7_
  - _Requirements: 4.2, 5.4, 5.6, 7.4, 7.5, 8.2, 8.8_

- [ ] 14.2 Build the final Factor bundle from approved dated evidence (3h)
  - Reuse the reviewed dated-evidence table, exact response bytes, origins, and source hashes with zero live provider calls.
  - Re-run evidence validation, exact replay, source-to-consumption audit, run-local records, and immutable bundle validation under the reserved final identity.
  - Require exact approved input hashes and deterministic equality with the provisional decision/equity outputs; any provider-response, model, evidence, or input-driven drift returns to Tasks 12–13.7.
  - Completion is observable when the completed final Factor bundle is hash-valid, input-equivalent to the approved candidate, and all dated records remain distinct.
  - _Boundary: Clean-Source Factor Rebuild_
  - _Requirements: 2.1, 6.1, 6.2, 6.3, 6.4, 7.1, 7.4, 7.5, 7.6, 8.7, 8.8_

- [ ] 14.3 Build the final SJM v3 run from approved inputs (3h)
  - Re-run the frozen protocol against the final Factor bundle and final snapshot under the reserved `sjm_crowding_v3_total_return_bil` identity.
  - Revalidate candidate order, gates, selected configuration, cash equations, hashes, and reconstruction.
  - Require exact approved input hashes and value equality with the provisional candidate; any difference returns to Tasks 12–13.7 rather than being accepted post-review.
  - Completion is observable when the final SJM run reproduces the approved candidate deterministically and validates from its manifest alone.
  - _Boundary: Clean-Source SJM Rebuild_
  - _Requirements: 4.1, 4.2, 4.3, 7.1, 7.2, 7.4, 7.5, 7.6, 8.2, 8.8_

- [ ] 14.4 Rebuild canonical Markowitz tables, reports, and mirrors (3h)
  - Recompute all canonical tables from the final completed manifests.
  - Regenerate locale mirrors and verify all values and schemas.
  - Require output hashes and lineage to identify the frozen source commit.
  - Completion is observable when every canonical table family is internally consistent and no provisional lineage remains.
  - _Boundary: Clean-Source Canonical Reports_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.8, 3.6, 3.7, 3.9, 4.5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 7.1, 7.2, 7.3, 7.4, 7.6_

- [ ] 14.5 Re-execute clean-source notebook wave A (3h)
  - Execute notebooks 14, 15.3, 17, and 16 in the mandated order with network disabled.
  - Verify deterministic equality with final canonical inputs and prohibit canonical writes.
  - Completion is observable when wave A passes and the committed presentation outputs remain byte- or value-equivalent as appropriate.
  - _Boundary: Clean-Source Notebook Wave A_
  - _Requirements: 6.4, 7.1, 7.2, 7.4, 7.5, 7.6, 8.2, 8.6, 8.7, 8.8_

- [ ] 14.6 Re-execute clean-source notebook wave B (3h)
  - Execute notebooks 15.2, 15.4, 18.3, and 18.4 in the mandated order with isolated temporary outputs.
  - Revalidate locale exports, frontier tables, windows, annualization, and presentation metadata before promotion.
  - Completion is observable when wave B reproduces final tracked presentation artifacts without introducing a working-tree difference.
  - _Boundary: Clean-Source Notebook Wave B_
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 7.1, 7.2, 7.4, 7.5, 7.6, 8.2, 8.5, 8.6, 8.8_

- [ ] 14.7 Re-execute clean-source notebook wave C (2h)
  - Execute notebook 18.2 and the final trio dashboard in order.
  - Revalidate all final dashboard values, labels, source hashes, and stale-output exclusions.
  - Completion is observable when wave C reproduces the approved final outputs without dirtying the committed checkout.
  - _Boundary: Clean-Source Notebook Wave C_
  - _Requirements: 1.3, 3.6, 3.7, 5.4, 5.5, 5.6, 7.1, 7.2, 7.4, 7.5, 7.6, 8.2, 8.6, 8.8_

- [ ] 14.8 Re-run clean-source focused and workbook gates (2h)
  - Run the required focused root command and complete locked workbook command against final regenerated inputs.
  - Reject any divergence from the frozen source, final producer manifests, or historical-tag behavior.
  - Completion is observable when both commands pass and their exact results are ready for publication metadata.
  - _Boundary: Clean-Source Focused Validation_
  - _Requirements: 7.5, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

- [ ] 14.9 Re-run clean-source full and artifact-parity gates (3h)
  - Run the full root suite and all cross-output table and presentation checks.
  - Confirm the checkout remains clean and every final producer and artifact identifies the frozen commit.
  - Completion is observable when all clean-source gates pass with no generated drift or unmanifested output.
  - _Boundary: Clean-Source Final Validation_
  - _Requirements: 7.1, 7.2, 7.4, 7.5, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

- [ ] 15. Build and finalize the validated local `data-v4` upload set
  - _Boundary: Local Publication Candidate_

- [ ] 15.1 Stage an incomplete flat `data-v4` payload (3h)
  - Require the frozen clean commit, unused public tag preflight, completed input manifests, and an absent or empty destination.
  - Copy only cataloged canonical payloads, locale mirrors, figures, and formatted reports into an isolated staging directory.
  - Write a provisional manifest with `completed=false`; do not write `SHA256SUMS` or add a completion-marker file.
  - Reject duplicate basenames, unexpected files, incomplete inputs, or an existing completed candidate.
  - Completion is observable when the staging directory exactly matches the frozen catalog but cannot yet be mistaken for a finalized upload set.
  - _Boundary: Publication Staging_
  - _Depends: 14.9_
  - _Requirements: 6.3, 7.1, 7.4, 7.5, 7.6, 8.2, 8.8_

- [ ] 15.2 Validate staged tabular schemas, values, and windows (3h)
  - Validate every staged Parquet, JSON, US CSV, and German CSV against its cataloged schema and media type.
  - Compare columns, rows, dates, nulls, values, allowed projections, portfolio definitions, windows, annualization, cash benchmarks, and currency bases.
  - Reject stale, missing, duplicated, mixed-definition, or compatibility-path-divergent tabular assets.
  - Completion is observable when all staged tables pass and any single schema or value mutation blocks finalization.
  - _Boundary: Staged Tabular Validation_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.8, 3.6, 3.7, 3.8, 3.9, 4.5, 5.4, 5.5, 5.6, 7.2, 7.3, 7.4, 7.5, 7.6, 8.2, 8.5, 8.6, 8.8_

- [ ] 15.3 Validate staged formatted reports, figures, and inventory (2h)
  - Validate Markdown, TeX, PNG, compatibility aliases, public basenames, media types, and lineage.
  - Check PNG dimensions, title and window metadata, and source-table hashes without pixel hashes.
  - Reject unmanifested, stale, missing, duplicated, or wrongly named presentation assets.
  - Completion is observable when every presentation asset is catalog-owned and any inventory or provenance mutation blocks finalization.
  - _Boundary: Staged Presentation Validation_
  - _Requirements: 5.4, 5.5, 5.6, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.2, 8.5, 8.6, 8.8_

- [ ] 15.4 Run direct staging validation without finalizing (2h)
  - Validate the provisional `completed=false` manifest and staged files directly through staging-specific schema, value, inventory, and lineage validators.
  - Do not invoke the finalized release-client contract and do not require `SHA256SUMS` at this stage.
  - Capture pre-finalization validation results without mutating canonical source artifacts.
  - Completion is observable when the incomplete staging payload passes direct validation and every missing, extra, corrupt, or stale mutation fails.
  - _Boundary: Staged Pre-Finalization Validation_
  - _Requirements: 7.2, 7.4, 7.5, 8.2, 8.6, 8.8_

- [ ] 15.5 Close the pre-finalization evidence gate (2h)
  - Reconcile all clean-source producer, workbook, notebook, artifact, and direct staging evidence available before final manifest and checksum generation.
  - Require shared-boundary and downstream evidence for every shared defect while leaving manifest, checksum, clean-room, and promotion evidence pending.
  - Keep the candidate incomplete while any pre-finalization criterion, defect, command, or parity result is failing.
  - Completion is observable when every pre-finalization evidence slot passes and the only pending matrix entries are the explicitly post-finalization checks.
  - _Boundary: Pre-Finalization Remediation Gate_
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

- [ ] 15.6 Verify public-release smoke tooling readiness without publishing (2h)
  - Run the read-only public hook against mocked complete, missing, duplicated, and hash-mismatched endpoints.
  - Verify immutable historical-tag baselines and clean-cache workbook loading behavior.
  - Confirm that an absent real `data-v4` endpoint reports not-yet-published and performs no creation, tag, upload, or documentation action.
  - Completion is observable when public-verification tooling is ready for a separately approved future release action.
  - _Boundary: Public Smoke-Test Readiness_
  - _Requirements: 7.2, 7.4, 7.5, 8.2, 8.6, 8.8_

- [ ] 15.7 Write the final publication manifest exactly once (2h)
  - Replace the provisional manifest with canonical UTF-8 JSON containing sorted keys, finite numbers, the frozen commit, `git_dirty=false`, commands, seeds, input manifests, conventions, schemas, tolerances, validation results, compatibility paths, and superseded run identities.
  - Inventory every payload with public path, hash, size, media type, schema, rows, dates, locale, producer, and lineage.
  - Set `completed=true` only after the preceding gates pass.
  - Do not mutate the manifest after this task.
  - Completion is observable when the final manifest is semantically reproducible and rejects non-finite data, incomplete validation, or corrupt lineage.
  - _Boundary: Publication Manifest Finalization_
  - _Requirements: 2.8, 5.4, 6.3, 7.1, 7.2, 7.3, 7.4, 7.5, 8.2, 8.6, 8.8_

- [ ] 15.8 Generate `SHA256SUMS` last (1h)
  - Hash every payload and the final publication manifest using sorted public basenames.
  - Exclude `SHA256SUMS` from hashing itself.
  - Cross-check entries against manifest inventory and reject missing, extra, duplicated, or mismatched digests.
  - Completion is observable when the checksum file closes exactly over the manifest and payload set without requiring any later file mutation.
  - _Boundary: Publication Checksums_
  - _Depends: 15.7_
  - _Requirements: 7.2, 7.4, 7.5, 8.2, 8.6, 8.8_

- [ ] 15.9 Read-only verify and atomically promote the local candidate (2h)
  - Serve the finalized staging bytes through the implemented temporary HTTP clean-room tool and require the real explicit-tag release client to download and validate them from a clean directory.
  - Verify the finalized staging directory solely from its manifest and checksum file without changing manifest, checksums, or payloads.
  - Atomically move the verified directory to the final local `release_assets/data-v4` location.
  - Refuse overwrite of an existing finalized candidate.
  - Completion is observable when the promoted upload set is independently self-verifying and any failure leaves the final destination absent.
  - _Boundary: Local Candidate Promotion_
  - _Depends: 15.8_
  - _Requirements: 7.1, 7.2, 7.4, 7.5, 8.2, 8.6, 8.8_

- [ ] 15.10 Reconcile the final 52-criterion and 15-defect completion matrix (2h)
  - Add final-manifest, checksum, clean-room verification, local promotion, and public-smoke-readiness evidence to the matrix.
  - Require passing shared-boundary and downstream evidence for every shared defect and every acceptance criterion.
  - Keep the remediation incomplete if any producer, workbook, notebook, artifact, finalization, or verification result is missing or failing.
  - Confirm no outward-facing release, tag, upload, or current-release documentation change occurred.
  - Write the completion-matrix report outside the promoted upload set and tracked checkout so final manifest, checksums, and payload bytes remain unchanged.
  - Completion is observable when all 52 acceptance criteria and all 15 confirmed defects have passing executable evidence and the validated local upload set is the only publication result.
  - _Boundary: Final Remediation Gate_
  - _Depends: 15.6, 15.9_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 6.1, 6.2, 6.3, 6.4, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

## Implementation Notes

- **3.1 insufficiency boundary (2026-07-28):** The pre-remediation guard `len(r) < window + 10` returned the insufficient-inference result at exactly ten rolling Sharpe observations. Per the approved contract ("fewer than ten rolling Sharpe observations" retains insufficiency; R2.5 rejects a window invalid for the supplied series), inference now RUNS at exactly ten rolling observations (`test_exactly_ten_rolling_observations_runs_inference`) and `window > len(returns)` raises a validation error. Consequence sequenced to later work: `workbook/tests/test_parity_root_env.py` byte- and behavior-parity stays red until task 10.4 re-syncs `vendored_ssr.py` from the corrected root module.
- **Foundations dependency evidence (2026-07-28):** recall-guard was bumped v0.1.0 → v0.2.0 in the root manifest and lock by explicit user direction (upstream release, no API change). `dependency_evidence.json` root hashes were re-captured; no other dependency changed and no new runtime package was introduced.
- **Workbook env guard (2026-07-28):** `workbook/tests/test_imports.py::test_stats_extra_dep_available_in_root_env` asserted a root-env property but ran red under the locked `--project workbook` command (scikit-learn is behind the optional `stats` extra there). It now uses the same root-env detection as `test_parity_root_env.py`: runs (and passes) in the root env, auto-skips in the lean workbook env. The recorded locked workbook command is green: 228 passed, 20 skipped.
- **Mutation-hardening (2026-07-28):** Reviewer mutation testing found seven survivors; all are now killed by new tests: crisis anchor last-before-start and strictly-before boundary (`test_crisis_anchor_is_the_last_value_before_the_start`, `test_crisis_anchor_is_strictly_before_the_requested_start`), HAC covariance verified numerically against an independent nonrobust fit (`test_hac_covariance_is_actually_used_not_nonrobust`), fixed 252-day rolling-Sharpe scaling on non-default windows (`test_rolling_sharpe_annualizes_on_252_even_for_nondefault_window`), invalid periods/lags rejection, the `factor_returns_on` DataFrame branch with per-column error naming, tz-aware anchor and crisis NaT-label rejection, and a pin documenting NaN volatility for single-return crisis episodes. Kill-checks were verified by applying each mutation and observing the test fail.
- **Review outcome for tasks 2.1-3.3 (2026-07-28):** The adversarial review confirmed exactly two blockers: the recall-guard evidence freeze (remediated: user-sanctioned bump recorded in design.md Allowed Dependencies, `dependency_evidence.json` re-captured, `validate_foundations.py` exit 0, all frozen evidence commands green) and Task 4.1 work present ahead of its gate. Task 4.1 is hereby formally OPEN: its implementation (macro_framework/reporting.py, tests/test_reporting.py, tests/test_publication_artifacts.py) must pass its own reviewer gate against the Reporting Schema Contracts boundary before 4.1 is marked complete; it is never accepted as a side effect of the 2.1-3.3 wave.
- **Task 4.1 gate (2026-07-28):** Separately reviewed and APPROVED after one remediation round: the pre-emission gate initially accepted empty-string/NaN provenance ('is None' check only); it now treats None/NaN as missing, requires non-blank strings for the five string provenance keys, and folds non-string schema ids into the clean unknown-schema error. Reviewer RED-phase evidence reconstructed the defective gate and confirmed the 20-parameter regression matrix kills it. Full root suite at approval: 480 passed.
- **Task 4.2 gate (2026-07-28):** APPROVED after one remediation round. Initial rejection: an equal-length SSR computed on the RAW return stream passed the length-only check and could emit a self-contradictory reader row. Remediation (also adopting the reviewer's API suggestion ahead of 4.6): `build_reader_metric_row` now takes `cash_returns` and derives the excess stream internally through `portfolio_excess_returns` (misaligned/NaN cash fails structurally), and a stream-identity gate requires the recomputed excess Sharpe to equal `ssr.result.sr_full` (isclose 1e-9, both-NaN allowed) outside the documented insufficient-inference carve-out — which the reviewer confirmed can only leak NaN inference values. Five frozen matrix nodes green; 264-test frozen suite green at approval.
- **Task 4.3 gate (2026-07-28):** APPROVED first pass. `build_differential_metric_row` derives every statistic from the one `differential_returns` spread (base-anchored curve for total return/CAGR/maxDD/Calmar; spread moments for vol/Sharpe/Sortino; full ssr_* metadata under the sr_full stream-identity gate, which provably rejects raw-source and sign-flipped SSRs); `endpoint_total_return_difference` is a separately named descriptive field unemittable in single-portfolio schemas. Reviewer ran ten live adversarial probes, all rejected/behaved correctly. Non-blocking suggestion queued: return downside RMS from `_stream_sharpe_sortino` to remove the one duplicated expression (apply after the 4.4/4.5 gates land to avoid mid-review tree movement).
- **Task 4.4 gate (2026-07-28):** APPROVED first pass. `build_attribution_record` (actual window + model identity), `build_crisis_record` (verbatim CrisisMetrics projection over anchor..actual_end), and gate rejection of mixed-window/partial-binding/performance_only-contradiction full rows. Reviewer verified RED by scratchpad mutation reconstruction (4/4 mutations detected by exactly the frozen tests, including a deliberately compromised builder caught independently by the gate). Suggestions applied post-approval: meta type-check ordering; a crisis-row provenance cross-check is noted for a later reporting task.
- **Task 4.5 gate (2026-07-28):** APPROVED after one remediation round. Initial rejection: monthly rows' annualization basis was unpinned and their year/month labels unbound to the row window. Remediated: MONTHLY_SCHEMA pins 12 periods/year; the gate's monthly branch requires typed year/month/finite monthly_return, binds start/end to the labeled month, and caps n_obs at the month's day count; the stale-value check names a missing reader total_return. Reviewer confirmed all 21 mutation variants die with clean ValueErrors. Post-approval suggestion applied: duplicate (portfolio, window, year, month) monthly rows are rejected by report_table.
- **Task 4.6 gate (2026-07-28):** APPROVED first pass. All completed shared contracts export through `macro_framework` with `__all__`/attribute agreement verified, no circular imports, no unintended exports (TRADING_DAYS/cagr etc. stay module-internal), and the temporary `market_attribution` alias preserved for task 11.11. Package-import assertions split across tests/test_reporting.py (SSR/crisis/reporting names) and tests/test_skill_metric.py (attribution names). Full root suite 512 passed at approval. Reporting wave 4.1-4.6 complete.
- **Task 5.1 producer lineage (2026-07-28):** Adding the frozen `AcquisitionContract` to `scripts/build_basket_long.py` (the design-pinned location) changed the file's hash while leaving `main()` and its generated `data/basket_close_2009_2026.parquet` byte-identical; the baseline producer sha256 was re-captured with an in-inventory note and `validate_foundations.py` is green (exit 0).
- **Task 5.1 gate (2026-07-28):** APPROVED first pass (reviewer verified import-with-sockets-blocked, all refusal probes, fingerprint order-stability, and the legitimacy of the producer-hash re-capture). Post-approval hardening applied with tests: symbols frozen to a tuple at construction, scale_to_major type-pinned to float, vintage_date must not precede requested_end, total_return_field pinned to the module constant, non-dict quotes get the uniform ValueError. Producer sha256 re-captures now record old -> new in the inventory note per the reviewer's FYI.
- **Task 5.2 gate (2026-07-28):** APPROVED after one remediation round. The reviewer probed the LIVE FRED service and proved `fredgraph.csv` ignores `vintage_date`; the fetcher now uses `alfredgraph.csv` with cosd/coed/vintage_date and requires the vintage-suffixed response column as capture-time proof the vintage was honored. Also remediated: date-granular index guard (intraday labels rejected), complex-dtype rejection on both paths, parser drops ONLY the '.' unpublished sentinel (any other token raises), clean MultiIndex/duplicate-column errors, and an injected-disclosure consistency invariant (now also netting outside_requested_window_dropped per the re-gate suggestion). Mocked-transport probes confirmed every path; 544-test root suite green.
- **Tasks 5.3/5.4 gate (2026-07-28):** APPROVED first pass (combined gate). Live probes confirmed COMPLETED can never precede validation (an injected invalid frame left a dirty staging dir without the marker), NaN-pair overlap cells are not counted as revisions, and every refusal (overwrite, immutable identity, byte mutation, re-inventoried non-finite, absent overlap disclosure) fires independently. Suggestions applied post-approval with tests: build_time validated as tz-aware ISO-8601 and normalized to UTC, predecessor selection compares parsed timestamps, cash/benchmark role labels added to the manifest, COMPLETED carries the manifest sha256, delete-to-retry recovery documented. At-rest manifest tamper evidence is owned by the tasks 13-15 release-checksum layer per the reviewer's scope ruling. Section 5 complete.
- **Wave 6.1-6.5 (2026-07-28):** Executed as a fan-out (5 parallel spec briefs, 5 chained implementers, 5 parallel reviewer gates). ALL FIVE GATES APPROVED FIRST PASS with only informational findings. Delivered: the frozen `DatedFactorEvidence` contract with deterministic evidence identities and `validate_evidence_records` (6.1); per-date persistence preserving independent records for identical prompt text, exact response bytes under UTF-8 hashing, explicit reconstructed/failure origins, and refuse-non-empty staging (6.2); exact (variant, rebalance_date) replay resolution with the later-date-wins and warn-and-continue paths removed and zero live model calls (6.3); source-to-consumption audit fingerprints failing before any publishable output on cross-date/cross-variant swaps (6.4); and the duplicate-prompt/immutability regression matrix incl. all frozen nodes (test_ac_6_1..6_4, test_ac_8_7, defect-1 shared+downstream) (6.5). Every implementer captured genuine RED evidence (tests failing before the production change). Producer sha256 for scripts/extend_stream_2026.py re-captured per task with old->new notes. Post-gate cleanup: one unused test import removed. Fresh final evidence: 7/7 frozen wave nodes, 58 focused, 593 full root suite, foundation validator exit 0. Known pre-existing caveat (all gates): the frozen final_validation_command names tests/test_sjm_crowding.py and tests/test_markowitz.py, owned by later tasks 7.x/8.x.
- **Consolidated wave 6.7–6.10 / 8.2–8.4 / 10.2 (2026-07-29):** Per explicit user directive to consolidate and speed up, executed as one three-track parallel workflow with a single blocker-focused review per track (max 4 targeted adversarial probes, no mutation matrices) instead of per-task gates. ALL THREE REVIEWS APPROVED FIRST PASS with zero blockers. Factor track: 6.7 audit-closed with no code change required (strict shared contracts only, no local OLS/crisis math, performance_only shortened-SPY handling, publication assembly excluded, one explicit window per record); 6.8 parity harness with fully independent test-local recomputation (incl. hand-rolled Newey–West/Bartlett HAC, no statsmodels) plus four deliberate-mismatch tests; 6.9 immutable `factor_run.v1` bundle (FACTOR_RUN_ARTIFACTS 15-role catalog, build/validate/load, COMPLETED-last carrying manifest sha256, refuses overwrite/dirty staging, validates from the run directory alone) with `main()` stage S11; 6.10 nine bundle tests (valid end-to-end + eight rejections incl. coherently re-signed manifests and forged snapshot lineage). Markowitz track: 8.2 audit-closed (frozen-node collectability gap fixed test-first); 8.3 deterministic long-only SLSQP frontier with full per-target diagnostics and visible (never dropped) failures plus `validate_frontier_point`; 8.4 fifteen offline numerical tests incl. an FX-inversion counterexample; annualization hard-pinned to 365.2425/7. Workbook track: 10.2 close-out confirmed — explicit-immutable-tag-only resolution, allowlist-before-download, verify-before-cache (`_verify_bytes` recomputes SHA-256 from observed bytes, breaking the caller-controlled self-authentication loop), per-entry and whole-tag cache invalidation on integrity failure, credential-free provenance, data-v1..v3 path unchanged; reviewer independently reproduced the RED-bite probe. Fresh close evidence: root suite 1039 passed / 3 skipped; locked workbook command 253 passed / 22 skipped. Non-blocking reviewer suggestions recorded for later: `make_data_v4_manifest` completed=/release_tag= kwargs untested (invariants enforced in `_parse_manifest`, covered ad hoc by probes) and the data-v4 tag comparison is case-sensitive.
- **Consolidated wave 7.1+7.3–7.9 / 10.8–10.11 (2026-07-29):** Same consolidated pattern (chained implementers per track, one blocker-focused review per track, max 4 adversarial probes). BOTH REVIEWS APPROVED FIRST PASS with zero blockers. SJM track (scripts/build_sjm_crowding.py + tests/test_sjm_crowding.py, serialized boundary): 7.1 `load_sjm_inputs`/`SJMInputs` manifest gate (exact pinned Factor run_id/snapshot_id + manifest sha256s, Factor→snapshot lineage binding, loose-legacy rejection); 7.3 `cash_returns_on_factor_calendar` (strict anchored BIL selection, fill disabled, exact Factor index incl. first return); 7.4 exact overlay/control equations sharing one cash vector, `derisk_cash_pin`-only control alias, lagged drawdown arming, frozen signal cadence from the single hashed `_SIGNAL_STEPS` source; 7.5 `select_sjm_config` deterministic frozen-order mutation replay with canonical ledger + `ledger_sha256` (never restores the previous winner — proven with a spectacular-holdout-old-winner test); 7.6 development-only metrics/gates (holdout physically excluded before compounding; CAGR-budget and same-cash control max-DD gates; holdout-rewrite invariance test); 7.7 `build_sjm_run`/`validate_sjm_run` — one validator proving hash/protocol/configuration/equation/reconstruction equality at <1e-9, persisted cash leg per defect 5; 7.8 staging per repo convention (refuse completed overwrite and dirty staging, manifest first, COMPLETED-last carrying manifest sha256, injected-late-failure unconsumable); 7.9 the serialized test matrix (41 tests in file). Reviewer probes additionally confirmed: mid-staging write failure leaves no manifest and refuses retry; orphan COMPLETED-only destination stays immutable; iteration-0 seed KEEP mirrors the shipped factor_loop.run_loop baseline convention. Publisher track (scripts/publish_finance_remediation.py + tests/test_publication_artifacts.py): 10.8 `stage_publication_candidate` + direct staging validators (provisional manifest completed=false last, SHA256SUMS prohibited, per-source completed-producer-manifest verification, casefold duplicate-basename rejection); 10.9 exactly-once finalization, SHA256SUMS generated last covering payloads + final manifest and never itself, read-only verification proven under chmod 0444/0555 with byte+mtime snapshot equality, os.rename promotion to an absent-only destination; 10.10 `serve_publication_candidate` clean-room localhost endpoint (GET-only, 405 writes, 404 mutable tags) driving the REAL workbook ReleaseClient with clean cache plus representative S0–S5 builds; 10.11 `run_public_release_smoke` read-only hooks (explicit tag + http(s) base URL only, zero upload/creation capability grep-verified, historical frozen-hash pins, `TagNotPublishedError` not-yet-published state). Fresh close evidence: root suite 1072 passed / 3 skipped; locked workbook command 253 passed / 22 skipped; all coverage-matrix check_ids in both owned test files pass under their exact commands.
- **Consolidated wave 9.1–9.8 / 10.7 (2026-07-29):** Same consolidated pattern; BOTH REVIEWS APPROVED with zero blockers. Reports track (scripts/build_tear_sheet.py, scripts/export_csv_mirrors.py, tests/test_publication_artifacts.py): 9.1 four manifest-gated family loaders (Factor via load_completed_factor_run, SJM via validate_sjm_run, market/Markowitz via validate_market_snapshot; pinned identities + 64-hex manifest sha256s; test_ac_7_6 proves artifact-only edits — even with re-signed manifest and COMPLETED marker — can never stand in for the owning producer); 9.2 canonical Factor/AI-variant tables as pure projections of run-local records through report_table (six data-v4-pinned tables; exact-catalog gate against a second recalculated row family); 9.3 tear_sheet_sjm_crowding_ext2026 (tear_sheet.sjm.v3) from one completed SJM v3 run with dev/holdout window roles never fabricated, cash-excess SSR on the run's persisted BIL stream, post-load byte-mutation detection; 9.4 trio/static-window/dashboard tables where the trio's Factor row IS the run-local reader record and the SJM row IS the SJM full-window row, plus triple lineage binding and repeated-row agreement gates; 9.5 Markowitz 10y/max moment+frontier tables exclusively through macro_framework.markowitz with sealed source-date sha256 and no mixed-local strategy points on USD frontiers; 9.6 auxiliary monthly-return/risk tables recompounding to canonical reader rows at 1e-9 with raw-market-model-only vocabulary; 9.7 export_csv_mirrors.py rewritten as the zero-finance-formula locale-mirror producer (X.csv/X_de.csv, 8-decimal, 5e-9 round-trip); 9.8 the integration matrix. Workbook track: 10.7 tag-selection plumbing in workbook/build_workbook.py + workbook/tests/test_tag_selection.py — data-v2 default preserved (environment never consulted), explicit-only data-v4 selection through manifest-verified loading, fresh ReleaseClient per tag change, cross-tag cache substitution prohibited.
- **Consolidated wave 10.3/10.5/10.6 + provisional 12.1 (2026-07-29):** Workbook chain APPROVED (consolidated review): 10.3 tag-bound data-v4 registry (24 keys) in contract.py mirroring the canonical producer schemas with load_v4_frame/load_v4_json, per-row semantic validation (windows, counts, pinned annualization 252/365/12, cash benchmark, currency basis, deterministic SSR settings), bidirectional cross-tag refusal, historical v1–v3 specs byte-unchanged, root-env mirror-parity tests pinning the workbook vocabulary to macro_framework.reporting, build_tear_sheet.py, and the frozen data-v4 catalog; 10.5 build_v4 StepView in steps.py consuming all 22 canonical tables verbatim, cash-excess SSR rebuilt through the vendored root ssr_inference with bit-for-bit bootstrap agreement against published rows, missing-cash/shortened-attribution/incomplete-window/schema/manifest failures surfaced as named visible checks, S0–S5 and the data-v2 default untouched; 10.6 fixture-driven suite (test_steps_v4.py, 13 tests incl. real-ReleaseClient mocked-transport stale/unmanifested/manifest-failure scenarios). Provisional production 12.1 COMPLETE: one-time live acquisition (yfinance + ALFRED, vintage 2026-07-29, coverage 2009-09-01..2026-06-30) under the frozen committed AcquisitionContract; immutable raw response bytes persisted with SHA-256 and the snapshot REBUILT FULLY OFFLINE from them; completed self-validating candidate at data/provisional_remediation/market_snapshots/provisional_market_total_return_fx_2026-06-30_v1 (final identity market_total_return_fx_2026-06-30_v1 remains reserved). TWO ADVISORIES RECORDED FOR TASK 14.1: (1) live ALFRED full-history payloads contain 183 BLANK cells (unpublished H.10 bank-holiday dates) that the committed parse_fredgraph_csv rejects — production used the committed injectable-frames path with a deterministic blank→'.' canonicalization disclosed with exact counts in both manifests; the 14.1 rebuild MUST consume the persisted raw bytes (as 14.1 already mandates) or the parser needs a spec-approved fix; (2) the stlouisfed /graph/ CSV backends reject plain requests/curl TLS fingerprints — only browser impersonation (curl_cffi via the locked yfinance dependency) is served; transport disclosed in frozen_acquisition_config.json and raw_sources_manifest.json.
- **BLOCKER at task 12.2 — Factor simulation price input is undeclared and unreproducible (2026-07-29):** The provisional Factor bundle `data/provisional_remediation/factor_runs/factor_ext2026_2019-01-01_2026-06-30_v1` is internally complete and self-validating (COMPLETED pins manifest sha256 `1aaaf6de…`, build_time identical, `replay_audit.result=pass` over 180/180 keys, all 15 artifacts hash-inventoried, source_commit `e9aed6d`), and it was produced with ZERO live provider calls: all 36 loadings generations, 36 scorings, and 18 naive generations were replayed byte-for-byte from the persisted dated evidence, with 180 evidence_ids verified byte-identical to the original live run. HOWEVER: the 5-symbol yfinance price frame consumed by the walk-forward simulation is neither persisted, hashed, nor listed in `manifest.input_manifests` (which correctly pins the six v1 artifacts and the market snapshot); `config` records only `price_fetch_end`. A plain run of the committed `main()` using a LIVE price fetch HARD-FAILED at `build_factor_metric_records` with `ValueError: PIT and non-PIT performance returns must have identical indexes` (scratchpad log prov_12_4_factor_run.log); the completed bundle exists only because the remediation relaunch monkeypatched `ext.fetch_prices` to return a capture stored in the EPHEMERAL session scratchpad (`prov_repro_prices.parquet`). Consequences: task 13.7 cannot freeze this input's retained location and SHA-256, and task 14.2 cannot rebuild the bundle deterministically once that scratchpad is cleaned. The root cause of the live-fetch index divergence is unexplained. Downstream 12.3/12.4 and the notebook wave are HELD pending a decision: (A) persist the capture into `data/provisional_remediation/raw_sources/` as a hashed, manifest-disclosed raw source (provenance addition only), or (B) change the producer so simulation prices come from the immutable market snapshot (re-opens section 6 producer work and the design's Factor-input boundary). Nothing downstream was built on this foundation.
