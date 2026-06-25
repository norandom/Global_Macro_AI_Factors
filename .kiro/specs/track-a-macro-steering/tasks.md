# Implementation Plan

- [ ] 1. Foundation: declare the scoring dependency
- [x] 1.1 Add recall_guard as a pinned project dependency
  - Add the released `recall_guard @ v0.1.0` git dependency to the project's dependency list and lock it
  - Confirm the environment resolves and `recall_guard`'s public scorer surface is importable
  - This dependency manifest is the only existing file the whole feature is permitted to edit; all other work is additive
  - Observable: a fresh sync resolves the dependency and importing the scorer succeeds without pulling plotting/backtest extras; no existing module or notebook is modified beyond the dependency manifest
  - _Requirements: 1.1, 6.1_

- [ ] 2. Core: steering module
  > All sub-tasks below live in the single new steering module file, so they are sequential (shared file) rather than parallel; the first sub-task creates the module. Scope is obvious from the descriptions, so per-task boundary annotations are omitted.
- [x] 2.1 Directional point-in-time prompt rendering
  - Render the same anonymized, z-scored macro state and asset snapshot the agent saw into a directional-forecast prompt that elicits a parseable direction and confidence
  - Use the same prompt template that the calibration corpus uses so the scored features are comparable
  - Keep it deterministic and free of any calendar date or real ticker
  - Observable: equal inputs produce an identical prompt string containing no date/ticker, and a unit test pins the template and determinism
  - _Requirements: 1.2, 1.4_

- [x] 2.2 Calibration adapter over the released scorer
  - Build the dated FMP calibration corpus, read the resulting on-disk corpora back into in-memory prompt lists, and calibrate the chosen NIM model from them
  - Surface calibrator quality (held-out separation and the weak flag) and propagate the library's configuration error unchanged when the credential is missing or rejected at calibrate time
  - Observable: with the library and FMP builder mocked, calibration yields a scorer plus quality flags from corpora read back off disk, and an empty/rejected credential raises the configuration error; unit tests cover each path
  - _Requirements: 1.2, 1.5, 1.7_

- [x] 2.3 Scoring path on the separate inference path
  - Score rebalance prompts through the calibrated scorer on the separate logprob-bearing inference path, preserving input order
  - Expose only the memorization probability and failure reason to callers — never the scorer's own directional output — and propagate the configuration error if the credential is rejected while scoring
  - Observable: with the library mocked, scoring returns one result per prompt in input order carrying only the memorization probability/failure reason, and a rejected credential mid-scoring raises the configuration error; unit tests cover success and failure records
  - _Requirements: 1.2, 1.3, 1.5_

- [x] 2.4 Point-in-time macro characterization and steering signal
  - Produce, for a rebalance date, a deterministic regime label and a per-series as-of z-score summary computed only from observations strictly before that date, reusing the existing macro panel without regenerating it
  - Emit a per-rebalance steering signal that includes a macro-to-view consistency rule (a documented heuristic mapping regime to preferred asset categories), bounded between a configured floor and one
  - Persist the per-rebalance steering signals as a new artifact and define no forecasting target
  - Observable: a unit test confirms observations dated on/after the rebalance date are excluded, the regime label is deterministic, the consistency value stays within the floor-to-one band, and the signals artifact is written under a new filename
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 2.5 View confidence shaping and gating
  - Shape each view's confidence as base confidence times one-minus-memorization times the macro consistency factor, returning a new view list and leaving expected-excess, asset legs, and the inputs untouched
  - Exclude a rebalance's views when memorization meets or exceeds the configured threshold; pass the views through unchanged when scoring is disabled, the calibrator is weak, or the memorization value is unavailable
  - Introduce only confidence/inclusion shaping — no return objective
  - Observable: unit tests show confidence falls monotonically as memorization rises, views are excluded at the threshold, and the disabled/weak/missing cases return the original views unchanged
  - _Requirements: 1.6, 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 2.6 Prompt-version variant agent
  - Provide a variant agent that subclasses the existing agent and supplies alternative prompt instructions by overriding the agent's private readiness step, using a per-variant cache directory so different versions never collide in one cache
  - Preserve prior prompt versions additively and never write the base cache or modify the base agent module
  - Observable: a unit test confirms distinct versions resolve to distinct cache directories, the base cache is untouched, and the variant's instructions drive the prompt
  - _Requirements: 4.1, 4.4_

- [x] 2.7 Score distribution reporting
  - Summarize the memorization-probability distribution and parse-failure rate across a set of scores for evaluation use
  - Observable: a unit test over mixed pass/fail scores returns correct distribution aggregates and failure rate without any network calls
  - _Requirements: 4.2, 5.2_

- [ ] 3. Integration
- [x] 3.1 Steered rebalance composition for walk-forward
  - Compose the agent, characterizer, scoring path, and view steerer into a walk-forward-compatible steered decision step that holds the agent instance, sources real symbols from the sliced price columns, and feeds the shaped views into the existing unchanged view-to-Black-Litterman conversion
  - Keep the composition agent-type-agnostic: it accepts the base agent or any subclass (including the prompt-variant agent), so prompt refinement reuses it without changes and does not hard-depend on the variant
  - Apply the gating fallbacks so an unavailable/weak/disabled score yields the original Track A behavior for that date
  - Observable: an integration test on a small fixture with a mocked agent and mocked scorer runs end-to-end through the unchanged conversion to a valid target row, and separately exercises the unsteered-fallback path
  - _Requirements: 1.6, 3.1, 3.2, 3.3, 3.4_
  - _Depends: 2.1, 2.3, 2.4, 2.5_

- [ ] 3.2 Live calibration build and directional-template verification
  - Run the one-time live calibration end-to-end: build the dated FMP corpus, read it back, calibrate the chosen NIM model, and persist the corpora plus a calibration header recording the held-out separation score and weak flag
  - On a small live sample, confirm the chosen NIM model returns a parseable direction and confidence from the directional template before any full run
  - If the calibrator is weak, document the unsteered-fallback outcome as a valid contamination finding and append it to the append-only research log as a new dated entry
  - Observable: the persisted calibration artifacts (corpora files plus a header with separation score and weak flag) and a handful of parse-successful sample scores exist; any fallback decision is written as a new dated research-log entry without altering earlier entries
  - _Requirements: 1.5, 1.7, 6.3_
  - _Depends: 2.1, 2.2, 2.3_

- [ ] 4. Validation: playbooks and evaluation
- [ ] 4.1 (P) Macro characterization and steered-variant playbook
  - Add a new numbered playbook that builds the point-in-time steering signals, scores the existing rebalance stream, runs the steered variant through the existing walk-forward and simulation, and persists steered targets, equity, decision log, and the score log under new filenames
  - Evaluate the steered variant with the existing head-to-head framework by adding it as an additional comparison entry, and additionally report its memorization-probability distribution; define success as lower-or-equal contamination with non-degraded head-to-head metrics
  - Append the run's findings as a new dated research-log entry without altering earlier entries; if this runs concurrently with the prompt-refinement playbook, serialize the research-log append so each writes its own dated entry
  - Observable: the playbook runs end-to-end producing the new steered artifacts, a head-to-head table that includes the steered entry alongside the existing tracks, and the memorization distribution; no existing notebook, module, or data artifact is modified
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.3, 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 6.4_
  - _Boundary: notebook 11 macro characterization and steering (shared append-only research log, serialized)_
  - _Depends: 3.1, 3.2_

- [ ] 4.2 (P) Point-in-time prompt-refinement playbook
  - Add a new numbered playbook that evaluates at least two prompt versions over the same point-in-time prompt stream via the variant agent, reporting each version's memorization distribution and view-stability metrics and the head-to-head deltas
  - Accept a refined prompt only when its head-to-head metrics are no worse than the current prompt's, and preserve all prior versions under versioned, additive filenames
  - Append the comparison outcome as a new dated research-log entry without altering earlier entries; if this runs concurrently with the steered-variant playbook, serialize the research-log append so each writes its own dated entry
  - Observable: the playbook reports per-version contamination, view stability, and head-to-head deltas, records an accept/reject decision honoring the no-worse gate, and writes new versioned artifacts without overwriting prior ones
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 5.3, 6.1, 6.2, 6.3, 6.4_
  - _Boundary: notebook 12 prompt refinement (shared append-only research log, serialized)_
  - _Depends: 2.6, 3.1, 3.2_

## Implementation Notes
- 2.6: A reviewer subagent ran `git checkout` on the uncommitted `steering.py` during a RED-phase mutation, which can wipe in-progress work. It restored from a backup; the parent then independently verified the working tree (VariantMacroAgent present, 62 tests green, llm_agent.py untouched) before committing. Future reviews must never reset/checkout uncommitted task work.
- Env: pytest/pytest-mock are dev deps; run tests with `uv run pytest -q`. `recall_guard@v0.1.0` resolves from the git tag and is importable; tests mock it (no network). Live tasks (3.2/4.1/4.2) need `NVIDIA_API_KEY` + `FMP_API_KEY` (+ `OPENROUTER_KEY`); no `.env` is present.
