# Implementation Plan

- [ ] 1. Foundation: workbook package scaffold
- [x] 1.1 Scaffold the lean workbook package with dual install surfaces
  - Create the new top-level workbook package with its own lean project manifest (runtime deps + optional token/statistics extras) and an importable package skeleton with no heavy work at import time
  - Add the two Excel-support libraries (the add-in stub and the spreadsheet writer) to the repository's dev dependency group only, so the root test run can import every workbook module; verify the statistics library needed by the optional certification extra is importable in the root environment (add it to the dev group only if it is not)
  - Create every planned module as a stub with the complete package re-export list up front, so the later parallel computation tasks each touch only their own module and test file (no shared-file conflicts)
  - Observable: the root test run collects and passes a new import-smoke test proving every planned workbook module imports in the root environment; the lean manifest lists only the allowed runtime dependencies (no portfolio/inference stack); nothing outside the new directory, the root dev group, and the ignore file is touched
  - _Requirements: 7.4, 7.5_

- [ ] 2. Core: data access
- [x] 2.1 Versioned release client with provenance and error taxonomy
  - Fetch release assets by explicit version tag with an on-disk cache, archive-member extraction for the bundled tarballs, and a provenance record (tag, asset, resolved address, checksum, cache origin) for every retrieval
  - Enforce the failure rules: a failed refresh raises a typed per-asset error (network/missing/authorization/unpack) and never silently serves substitute or stale data; switching version constructs a new client
  - Implement the dormant authenticated path: a token provider (system keychain then environment) consulted only when the unauthenticated address is refused, with the token never appearing in provenance, errors, or any persisted artifact
  - Observable: offline tests (mocked transport + fixture files) cover successful fetch with provenance, every error class, the no-stale-substitution rule, and the token lookup order; no live network in the default test run
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

- [x] 2.2 Schema-contract registry and typed loaders
  - Encode the captured per-asset schema contract (columns, dtypes, index, minimum rows) for every consumed release asset as a typed registry with loader functions returning validated tables plus provenance
  - Fail fast on any mismatch with an asset- and column-specific message (the foundation of the discrepancy detector)
  - Build the offline fixture set: schema-true subsets of every consumed asset, checked into the test tree
  - Observable: a test validates every registry entry against its fixture; a corrupted-fixture test produces the asset+column-specific failure message; fixtures cover all consumed assets including one evidence bundle and the screen results
  - _Requirements: 1.4, 2.1, 2.3, 3.1, 4.1, 5.1, 6.1_
  - _Depends: 2.1_

- [ ] 3. Core: computation (independent pure modules)
- [x] 3.1 (P) Re-derivation formulas
  - Implement the pure re-derivation set: the binomial interval for the naive eval, the guard formula (raw times one minus score), the paired effect size, the contamination premium, the loading-stability measures, the equity-line metrics (returns/risk/drawdown plus the fixed crisis window), and the evidence class statistics
  - Observable: each formula is tested against hand-computed values AND against the published values in the fixtures (agreement within display precision)
  - _Boundary: rederive_
  - _Requirements: 2.6, 3.2, 4.3, 4.4, 5.3, 6.1_
  - _Depends: 2.2_

- [x] 3.2 (P) Vendored Sharpe-stability computation with parity proof
  - Vendor the existing repo's Sharpe-stability computation verbatim (with a provenance header naming source path and commit) into the lean package
  - Observable: a parity test asserts the vendored copy reproduces the original module's output on the released equity fixture, auto-skipped outside the root environment; the vendored module imports without the heavy environment
  - _Boundary: vendored_ssr_
  - _Requirements: 6.2_
  - _Depends: 2.2_

- [x] 3.3 (P) Optional certification re-derivation (statistics extra)
  - Vendor the deterministic certification statistics (separation, resampled interval, permutation p) behind the optional statistics extra, importable only when that extra is installed
  - Observable: with the extra installed (root environment), a test reproduces the published controlled separation values from the evidence fixture's standardized feature columns; without it, the package imports and the deeper re-derivation reports itself unavailable rather than failing
  - _Boundary: certification_
  - _Requirements: 2.6_
  - _Depends: 2.2_

- [x] 3.4 (P) Verification framework (published vs re-derived)
  - Implement the comparison record and tolerance check that pairs a published figure with its re-derived value and yields a visible flag on disagreement — returned as data, never raised, never silently resolved in favor of either value
  - Observable: tests show agreement produces an ok record and an injected discrepancy produces a flagged record carrying a human-readable message
  - _Boundary: verify_
  - _Requirements: 7.2_

- [ ] 4. Core: storyboard step assembly (single shared module — sequential)
- [x] 4.1 Step 1 view — certification with evidence drill-down
  - Assemble the certification table, per-candidate raw-evidence tables, and the honest outcome framing (empty certified set; fallback selection runs recall-guarded); render the known gaps as explicit states — evidence-pending for the candidate missing raw evidence and unscreenable/not-exonerated for the unservable one — never as failures
  - Include the class-count/feature-statistics consistency check against the published summary (deeper separation re-derivation only where the statistics extra is present)
  - Observable: tests over fixtures assert the table contents, both gap markers, the mandated framing text, and the consistency checks attached to the view
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 7.3_
  - _Depends: 3.1, 3.3, 3.4_

- [x] 4.2 Step 2 view — coin-flip naive evaluation
  - Assemble the per-call table and the accuracy-with-interval summary re-derived from the per-call records, framed as the expected, correct no-alpha outcome (never a performance target)
  - Observable: tests assert the re-derived accuracy/interval matches the fixture records, the interval-contains-half statement, and the framing text
  - _Requirements: 3.1, 3.2, 3.3, 7.3_
  - _Depends: 3.1, 3.4_

- [x] 4.3 Step 3 view — factor development with the guard re-derived
  - Assemble loadings-with-parse-status, memorization scores with distribution summary, raw-vs-guarded views with the guard formula re-derived and checked against published values, per-version stability, and the refinement accept-gate inputs plus recorded rejection with both versions preserved
  - Observable: tests assert the guard check passes on the fixture, the distribution summary matches, the gate view carries the recorded decision and inputs, and both prompt versions appear
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 7.3_
  - _Depends: 3.1, 3.4_

- [x] 4.4 Step 4 view — two-line simulation
  - Assemble both lines (equity, targets, decision log) over the same stream with head-to-head metrics recomputed from the equity series, the deployable/diagnostic labeling rule, reference-track context from the published release, and per-date guard detail
  - Observable: tests assert both lines present with matching date coverage, the diagnostic label on the recall-enabled line (and never on any deployable field), recomputed metrics matching published figures within tolerance, and per-date steered/score detail
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 7.3_
  - _Depends: 3.1, 3.4_

- [x] 4.5 Step 5 view — luck versus skill
  - Assemble the paired contrast with premium and paired effect size re-derived from per-date records, the Sharpe-stability table for both lines and the differential re-derived via the vendored computation, the PIT-vs-non-PIT loading-stability comparison, and the mandated conclusion wording (luck-compatible; lookahead/recall bias, never attainable skill)
  - Observable: tests assert the re-derived premium/effect size and stability rows match published values within tolerance and the conclusion wording is present verbatim
  - _Requirements: 6.1, 6.2, 6.3, 7.3_
  - _Depends: 3.1, 3.2, 3.4_

- [ ] 5. Integration: Excel surface and workbook generation
- [x] 5.1 Excel function surface (stub-safe wrappers)
  - Expose the thin worksheet-function layer: an asynchronous versioned load returning a cached client handle, per-step view builders, on-demand table expansion, the verification-flags table, and the provenance table; version change is an explicit re-load driven by the tag input
  - No function accepts a token argument; error states surface as per-asset messages in the sheet
  - Observable: under the platform stub (no Excel), every exposed function is importable and callable end-to-end on fixtures in the root test run; the provenance and checks tables render as data frames
  - _Requirements: 1.2, 1.3, 1.5, 7.2, 7.4_
  - _Depends: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 5.2 Workbook skeleton generator
  - Generate the five-sheet workbook skeleton (plus a navigation index) on the development machine: titles, mandated framing text, provenance/checks areas, and formulas referencing only the exposed worksheet functions
  - Observable: a test opens the generated file and asserts five step sheets plus index exist, every formula references an existing exposed function name, and the framing text blocks are present; generation runs headlessly on the development machine
  - _Requirements: 2.2, 3.3, 5.2, 6.3, 7.3, 7.4_
  - _Depends: 5.1_

- [ ] 5.3 Deployment documentation and configuration example
  - Write the Windows deployment guide: add-in configuration example (module list, python path), install of the lean package, the commercial license/trial prerequisite stated explicitly, the token setup for a future private repo, and the manual Excel verification checklist (async load fills cells without freezing, tables spill, provenance populates)
  - Observable: the README and configuration example exist in the package directory, state the runtime prerequisite and that the data/computation layer remains verifiable without Excel, and the checklist enumerates the Excel-only behaviors excluded from automated tests
  - _Requirements: 7.6, 1.3_
  - _Depends: 5.1_

- [ ] 6. Validation
- [ ] 6.1 End-to-end validation and opt-in live-network test
  - Run the full fixture-driven pipeline (load → contract → re-derive → verify → all five step views) asserting every verification check passes on the shipped fixtures, and add one opt-in (marker-excluded) live test fetching a real release asset end-to-end
  - Confirm the whole repository test suite passes with the workbook tests collected, and that the working tree remains additive (new directory + dev-group and ignore-file lines only)
  - Observable: default root test run is green including all workbook tests with the live test excluded; enabling the marker fetches the real asset and validates it against the contract; a final review confirms no existing file beyond the declared two was modified
  - _Requirements: 7.1, 7.4, 7.5_
  - _Depends: 5.2, 5.3_

## Implementation Notes
- 1.1: workbook/tests has NO __init__.py (deliberate — with it, pytest maps the dir to package "tests", colliding with the root tests/ package and breaking full-suite collection). Do not reintroduce it. Smoke command: `PYTHONPATH=workbook uv run python -c "import factor_workbook"` (the package is intentionally outside the root uv workspace; root pytest imports it via workbook/tests/conftest.py sys.path insert).
- 2.1: cache is keyed (tag, asset) and served without re-fetch on hit (tags immutable); the no-stale rule = a FAILED fetch never falls back to any cache entry. Reviewer minors (acceptable, revisit only if the repo goes private): cache-hit provenance replays the public URL even for API-fetched bytes (from_cache=True flags it); cache writes are non-atomic and tag/asset are raw path segments (inputs come from the fixed registry).
- 2.2: an entirely-null column satisfies its contract dtype (parquet writers persist all-null as object OR float64 — the real 120b evidence raw_ref_delta is all-null float64 while 20b/phi-4 are str). Evidence tests parametrize over EVIDENCE_MODELS; fixture tarball carries both flavors. Registry keys are logical names; evidence loader is model-slug-parameterized.
- 3.1: paired_cohens_d uses POPULATION std (ddof=0) + 1e-12 zero-variance guard, matching macro_framework.factor_scoring._paired_cohens_d (the producer of the published 1.9252251) — ddof=1 misses it by ~0.7%. Fixtures are ROW SUBSETS: full-data published-figure agreement is exercised in 4.2/4.4/4.5. REVISED in 4.4 (approved boundary extension into rederive.py): equity_metrics mirrors the vectorbt producer convention — 365-day calendar-year annualization, day-0 zero return included (pct_change().fillna(0)), sharpe = arithmetic mean/std(ddof=1)*sqrt(365), sortino = mean/downside-RMS*sqrt(365), crisis_vol_ann stays 252 (evaluation.crisis_analytics). Verified against published pit/nonpit_metrics at rel err <= 1.71e-6; the earlier 252-day geometric convention fails 10/18 checks. Do NOT revert to 252/geometric. S4 metric tol = 1e-5 (documented).
- 3.3: full-data reproduction test runs on the REAL local evidence parquet (guarded skip elsewhere) and reproduces all four published values at 1e-9 — tolerance is sklearn-version-sensitive (1.8.0; loosen on upgrade). Do NOT module-level-import macro_framework.factor_scoring in workbook tests (breaks test_factor_scoring's sys.modules assertion) — import lazily inside test bodies.
- 4.1: gap markers are DATA-DRIVEN (screen_failed -> unscreenable; any other verdict with missing evidence member -> pending_evidence) — phi-4-mini is also pending in fixtures since the fixture tar ships only 20b/120b. Class-count checks flag ok=False on fixture subsets by design (R7.2 renders them); live agreement lands in 6.1.
- 4.5: SSR re-derivation slices equity at the first contrast date (2019-01-02), mirroring nb14; differential row uses documented _S5_DIFF_TOL=1e-3 (near-zero cancellation amplifies the parquet's ~1e-7 storage rounding to ~1.5e-4 rel) — non-differential rows hold 1e-6. paired_d reproduces exactly (1.9252251).
- 5.1: canonical in-cell error format is '#ERROR <asset>: <cause> — <detail>' (supersedes the design sketch's '#ERROR: <asset>: <cause>'). Surface is 7 functions: the design's five + FW_FRAMING (R7.3 sheet headers) + FW_VERSION (R1.2) — 5.2's formula-name check must target these.
