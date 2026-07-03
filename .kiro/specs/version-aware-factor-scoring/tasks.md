# Implementation Plan

- [ ] 1. Foundation: confirm the public primitive surface
- [x] 1.1 Confirm the recall_guard public MIA-primitive surface and module/test scaffold
  - Add an import-smoke test asserting the public primitives the design relies on are importable
    (`NvidiaLM`, `compute_mia_features`, `MiaFeatures`, `build_baseline`, `standardise`, `ControlBaseline`,
    `MCSCalibrator`, `LOGPROB_FLOOR`, and `train` from the mcs submodule) and that the host package imports
  - The directional façade is NOT imported (factor scoring uses the lower-level primitives only)
  - Observable: a new test passes confirming every relied-on primitive imports and the façade is absent from the new code path; no existing module/notebook/data is modified
  - _Requirements: 6.5_

- [ ] 2. Core: factor scoring module
  > All sub-tasks below live in the single new module file `macro_framework/factor_scoring.py`, so they are sequential (shared file), not parallel; the first sub-task creates the module. Scope is obvious from the descriptions, so per-task boundary annotations are omitted.
- [x] 2.1 Regime-loadings prompt renderer
  - Create the module and a single renderer that emits the regime-as-loadings factor task, in either an anonymized (point-in-time, z-scored, no date/ticker) or an identifying form (real tickers + as-of date + raw levels), differing only by the identifying additions
  - The prompt requests continuous loadings in `[-1, +1]` on the fixed named macro axes and never asks for a buy/sell direction or an expected return
  - The anonymized form carries only as-of, date-free macro content (the point-in-time property the scorer relies on)
  - Observable: unit tests confirm the anonymized form contains no date/ticker, the identifying form adds exactly the identity/date/raw-level tokens (otherwise token-identical), the prompt requests `[-1,1]` loadings on the named axes, contains no expected-return/forecast ask, and is deterministic for equal inputs
  - _Requirements: 1.4, 2.1, 2.2, 2.3, 2.5, 7.1, 7.6_

- [x] 2.2 Loadings parser
  - Parse a model reply into a per-axis loadings vector clipped to `[-1, +1]`, returning a typed result that flags parse failure rather than fabricating values
  - Observable: unit tests parse a well-formed reply into bounded loadings and return a not-parsed result on malformed output; the parsed result is keyed by rebalance date
  - _Requirements: 2.1, 2.4_

- [x] 2.3 Factor scorer — number-native calibration + configuration errors
  - Build the number-native calibrator from the macro panel on the factor task: an identifying recall corpus and an anonymized honest corpus (same task, framing-only difference), then the control baseline and the contamination calibrator, holding the calibrator + baseline + inference client; the reference run is disabled (so the reference-delta feature is inert)
  - Define and raise the module's own configuration error on an empty credential, on a zero-usable-row baseline, and on an authentication-class failure from the inference client (the bypassed façade does not provide one); expose the held-out separation and weak-calibrator flag
  - Observable: with the inference client and calibration primitives mocked, calibration builds the identifying/anonymized corpora and returns a scorer exposing the separation score and weak flag; an empty credential and a zero-usable-row baseline each raise the module's configuration error; tests assert the reference run is disabled
  - _Requirements: 1.1, 1.5, 1.6, 6.5_

- [x] 2.4 Factor scorer — number-native scoring path
  - Score any prompt for memorization via the public primitives (inference client logprobs → MIA features → calibrated probability), with no buy/sell direction parse; return a typed score carrying the memorization probability and a failure reason, and never read a directional signal
  - Surface the module's configuration error if the credential is rejected while scoring; return an unscored result (memorization probability absent) on a logprob/feature failure without crashing
  - Observable: with primitives mocked, scoring returns a probability computed via the feature path (not a direction parse) and differs across distinct prompts; a rejected credential raises the configuration error; a logprob failure yields an absent probability with a reason
  - _Requirements: 1.1, 1.2, 1.3, 1.5, 6.5_
  - _Depends: 2.3_

- [x] 2.5 Factor scorer — persistence
  - Persist and reload the trained calibrator and baseline standardisation statistics (and the calibration stats) without the credential; reload re-attaches a fresh inference client from a supplied credential
  - Observable: a save/load round-trip reproduces identical scores for the same inputs, the persisted artifact contains no credential, and the reloaded scorer reports the same separation score / weak flag
  - _Requirements: 1.6_
  - _Depends: 2.3_

- [x] 2.6 Tilt-as-exposure
  - Map a regime-loadings vector to per-asset dimensionless exposure tilts via a documented, non-predictive axis→category exposure table, packed so the existing unchanged Black-Litterman view-to-input conversion yields a magnitude of tilt times a dimensionless, non-return-bearing conviction
  - Observable: unit tests show each view carries a dimensionless tilt and conviction (no expected return), the existing conversion produces the expected `tilt × conviction` magnitude, and no predictive-return objective is introduced
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 2.7 Honesty-adjusted exposure
  - Down-weight each exposure tilt as a function of its measured memorization (tilt times one-minus-probability), returning a new view list and changing only magnitude; pass views through unchanged when the probability is unavailable or the calibrator is weak; apply only the discount (no hard exclusion gate)
  - Observable: unit tests show the adjusted exposure falls monotonically as memorization rises, equals the raw exposure at zero memorization, passes through unchanged on missing/weak, leaves asset legs/rationale untouched, and returns a new list
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 2.8 Factor-stability metric
  - Compute a per-version factor-stability metric defined as the variability of the version's factor loadings across the point-in-time stream
  - Observable: a unit test over a crafted loadings stream returns the expected per-axis variability summary
  - _Requirements: 5.2_

- [x] 2.9 PIT-vs-non-PIT contrast computation
  - Compute the contrast result from two variants' memorization values and head-to-head metrics: per-variant distributions and the non-PIT-minus-PIT contamination premium plus a paired effect size over the stream length (not a single-date point estimate)
  - Observable: a unit test over synthetic per-variant inputs returns the correct premium deltas, the paired effect size, and the backing stream length
  - _Requirements: 7.2, 7.3, 7.5_

- [ ] 3. Integration
- [x] 3.1 Steered factor weight-fn composition for walk-forward
  - Compose renderer → score → parse → tilt-as-exposure → honesty-adjust → the existing unchanged view-to-input conversion into a walk-forward-compatible factor decision step, holding the agent instance, sourcing real symbols from the sliced price columns, and injecting the existing base-allocation/blend math (not duplicating it); keep the composition agent-type-agnostic
  - Apply gating fallbacks so a parse failure, an unavailable memorization probability, or a weak calibrator yields the base/unadjusted behavior for that date
  - Observable: an integration test on a small fixture (mocked agent + mocked scorer) runs end-to-end through the unchanged conversion to a valid target row, and separately exercises the parse-fail / weak / missing fallback to base
  - _Requirements: 1.4, 3.1, 3.2, 3.3, 4.3_
  - _Depends: 2.1, 2.2, 2.4, 2.6, 2.7_

- [x] 3.2 Live number-native calibration build and persistence
  - Run the one-time number-native calibration end-to-end against the live inference endpoint (identifying recall corpus vs anonymized honest corpus on the factor task, from the macro panel; reuse the validated scoring model and its cutoff) and persist the calibrator + baseline + stats; record the held-out separation and weak flag
  - Append the calibration outcome (separation score, weak flag) as a new dated research-log entry
  - Observable: the persisted calibrator artifact exists (no credential in it), reloads and scores, and the separation score / weak flag are recorded; a new research-log entry documents the outcome without altering earlier entries
  - _Requirements: 1.1, 1.6, 6.3_
  - _Depends: 2.1, 2.3, 2.5_

- [x] 3.3 Certified no-recall model screen (module + live run) — _added 2026-07-03 (R8 amendment)_
  - Extend the factor-scoring module additively with a certification screen: gather the standardized per-prompt MIA features for the controlled identifying/anonymized classes, compute the held-out separation with a resampled confidence interval and a permutation p-value (offline, on the gathered features — no extra live calls), render a deliberately prose-confounded positive-control framing (diagnostic, explicitly non-R7.6), measure the factor-task parse rate, and produce a typed per-candidate certification verdict (certified-no-recall / recalls / detector-unvalidated / inconclusive)
  - Run the screen live across the logprob-bearing NIM candidates at a conservative common cutoff (pre-cutoff states trained-on for every candidate); persist per-candidate results (no credential) and append a dated research-log entry; select the certified model per the R8.4 rule and persist its calibrator at its own true cutoff
  - Observable: mocked unit tests cover the statistics (CI/permutation on synthetic features), the positive-control renderer, the parse-rate measure, and the verdict rule; the live run writes a results artifact + research-log entry; a certified model is selected and its calibrator persisted
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.6_
  - _Depends: 2.1, 2.3, 2.5_

- [ ] 4. Validation: playbooks
- [ ] 4.1 (P) Regime-loadings + recall-guarded factor playbook
  - Add a new numbered playbook that builds the per-rebalance regime-loadings artifact, scores each rebalance for memorization, runs the recall-guarded factor variant through the existing walk-forward and simulation, and persists the loadings, score log, and steered targets/equity/decision-log under new filenames
  - Open with the Excel-storyboard S2 section: the naive directional eval of the certified model (per-prompt records persisted, accuracy ≈ coin flip with binomial CI — the expected, correct no-alpha result)
  - Evaluate the variant with the existing head-to-head framework by adding a "Track A (factor)" entry alongside the existing tracks, and additionally report the memorization distribution; state the non-predictive success definition (factor stability + lower-or-equal contamination + non-degraded head-to-head)
  - Append the run's findings as a new dated research-log entry; if run concurrently with the refinement playbook, serialize the research-log append
  - Observable: the playbook runs end-to-end producing the new loadings/score/steered artifacts and a head-to-head table including the factor entry plus the memorization distribution; no existing notebook, module, or data artifact is modified
  - _Requirements: 2.4, 3.3, 4.1, 5.1, 5.2, 6.1, 6.2, 6.3, 6.4, 8.5_
  - _Boundary: notebook 13 macro factor scoring (shared append-only research log, serialized)_
  - _Depends: 3.1, 3.2, 3.3_
  - _Resumed: 2026-07-03 (user directive). Was deferred 2026-06-27 after the 3.2 weak-calibrator finding; now gated on the R8 certified no-recall model from task 3.3, which the playbook uses as both loadings generator and scored model (8.5)._

- [ ] 4.2 (P) Version-aware prompt refinement + PIT-vs-non-PIT contrast playbook
  - Add a new numbered playbook that evaluates at least two prompt versions over the same point-in-time stream, reporting each version's memorization distribution (which now differs by version) and factor-stability and the head-to-head deltas, with an accept-gate that adopts a refinement only at no-greater contamination and no-worse metrics, preserving prior versions
  - Run the PIT-vs-non-PIT contrast over the full rebalance stream and report the contamination premium with a paired effect size; the non-PIT variant is a diagnostic control and is never used to produce the deployable portfolio
  - Close with the Excel-storyboard S5 section: `compute_ssr` (Newey-West HAC) on the PIT line, the non-PIT line, and the recall-minus-no-recall return differential — luck-vs-skill stated with robust inference; persist the comparison table
  - Append the comparison and contrast outcomes as a new dated research-log entry; if run concurrently with the factor playbook, serialize the research-log append
  - Observable: the playbook reports per-version memorization distributions (differing across versions), factor-stability, head-to-head deltas, an accept/reject decision, and the PIT-vs-non-PIT premium over the full stream; new versioned artifacts are written without overwriting prior ones; the non-PIT control is not persisted as the deployable portfolio
  - _Requirements: 1.2, 5.1, 5.2, 5.3, 5.4, 5.5, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.5_
  - _Boundary: notebook 14 prompt refinement and contrast (shared append-only research log, serialized)_
  - _Depends: 2.8, 2.9, 3.1, 3.2, 3.3_
  - _Resumed: 2026-07-03 (user directive). Was deferred 2026-06-27; now gated on the R8 certified model (3.3). The controlled contrast is expected to show little separable premium for a certified no-recall model per the 3.2 finding — reporting that honestly is the point._

## Implementation Notes
- Carried forward from track-a-macro-steering: `recall_guard@v0.1.0` + pytest/pytest-mock are installed; run tests with `uv run pytest -q`. Offline core tasks mock `NvidiaLM`/the MIA primitives (no network). Reviewers must never `git checkout`/`reset` uncommitted task work.
- Number-native calibration uses the validated model `meta/llama-4-maverick-17b-128e-instruct` @ cutoff `2024-08-01` (research.md 2026-06-26: holdout_auc≈0.96, is_weak=False). The calibrator is trained on the FRED panel (identifying IS vs anonymized OOS on the factor task) — no news/FMP.
- Live tasks (3.2 calibration build, 4.1 nb13, 4.2 nb14) need `NVIDIA_API_KEY` (+ `OPENROUTER_KEY` for the agent). The Postgres price DB is absent → notebooks fetch yfinance prices in-cell; price-dependent equity/targets + any variant LLM caches are gitignored (regenerable). Persist the trained calibrator (joblib + JSON, no API key) so nb13/nb14 don't rebuild (~135 NIM calls).
- 2.2: `parse_loadings` clips to [-1,1] and returns None on malformed (no fabrication). Edge: non-standard JSON `NaN`/`Infinity` bypass the clip (NaN compares False); the tilt step (2.6) and any loadings consumer should guard non-finite values.
- 2026-07-03 Excel storyboard (user directive — the 5-step narrative the persisted data must support, one artifact set per step):
  - **S1 select model without recall** → `data/norecall_screen/results.json` (committed) + per-candidate raw evidence (`evidence/<model>/evidence.parquet`, `baseline.json`, `summary.json`).
  - **S2 coin-flip prediction on a naive prompt (no predictive alpha)** → nb13 opens with a naive directional eval of the CERTIFIED model over the rebalance stream: per-prompt records (date, prompt, predicted direction, confidence, realized direction, correct) + accuracy with binomial CI vs 0.5 → `data/naive_directional_eval_<model>.parquet`. Coin-flip is the EXPECTED, correct result (recall_guard purpose framing).
  - **S3 macro-factor development (numbers)** → nb13's per-rebalance loadings + p_memorized score log + factor-stability tables.
  - **S4 portfolio sim, two lines** → the PIT/no-recall deployable line (nb13) and the non-PIT/recall diagnostic line (nb14 contrast): targets + equity + decision logs per line (price-dependent files gitignored locally, shipped via the data release).
  - **S5 luck-vs-skill comparison** → head_to_head_report rows for both lines PLUS `compute_ssr` (macro_framework/ssr.py — Sharpe Stability Ratio with Newey-West HAC variance, Andrews bandwidth) per line and on the recall-minus-no-recall return differential → one comparison table artifact. Reuses existing ssr.py; no new stats code.
- 2026-07-03 data contract (user directive, for the follow-up PyXLL/Excel spec): every pipeline step this spec produces is persisted as a tidy, stable-named table under `data/` (parquet for tabular streams, JSON for run headers/certification evidence) — screen results (`data/norecall_screen/results.json`, committed), per-rebalance loadings, p_memorized score log, raw-vs-recall-guarded tilt views, targets/equity/decision log (price-dependent ones stay gitignored locally), and the PIT-vs-non-PIT contrast rows. Distribution: the parquet/JSON artifacts are uploaded to a GitHub Release on this repo (mirroring recall_guard's release-only distribution) so the Excel spec loads them from GH by URL instead of from a working tree.
- 2026-07-03 amendment: the 103-note's "validated holdout_auc≈0.96" was the prose-confounded probe; the controlled 3.2 run came back weak (holdout_auc=0.338 on a 30-prompt holdout — statistically consistent with chance). R8 + task 3.3 formalize this: screen candidates with bootstrap CI + permutation p (offline on gathered features), positive-control (prose-confounded) framing must fire, n_per_class raised (panel has ~168 pre-2024 months → up to ~120/class, holdout ~60). API rename (e9b1ed9): `honesty_*` → `recall_guarded_*` — spec prose "honesty/recall-guarded adjustment" refers to the `recall_guarded_*` API.
