# Requirements Document

## Project Description (Input)

A quant researcher (the project owner) has completed the `version-aware-factor-scoring` spec: the
full assessment — model certification screen, coin-flip naive eval, AI macro-factor development,
two-line portfolio simulation, and the luck-vs-skill comparison — currently lives only in Jupyter
notebooks (nb13/nb14) and in the `data-v1` GitHub Release
(https://github.com/norandom/Global_Macro_AI_Factors/releases/tag/data-v1 — 27 tidy, stable-named
parquet/JSON assets plus the raw per-prompt evidence tarball and the persisted calibrator).
Reviewing or presenting the assessment requires a Python/Jupyter environment; there is no
Excel-native way to walk the analysis step by step or to audit it from the raw evidence data.

What should change: a PyXLL-based Excel workbook that loads the `data-v1` release assets directly
from GitHub by URL (no repo working tree, token header supported while the repo is private) and
presents the assessment as the established five-step storyboard: S1 model certification (with raw
per-prompt evidence drill-down), S2 coin-flip naive eval, S3 AI macro-factor development, S4 the
two-line portfolio simulation (PIT recall-guarded deployable vs non-PIT diagnostic), S5 the
luck-vs-skill comparison (contamination premium + SSR/Newey-West-HAC).

Constraints and context: PyXLL is a commercial Excel add-in (Windows Excel required for runtime;
the development machine is Linux, so the spec must separate testable Python logic from the
Excel-only layer). Data access is read-only from the release URLs (versioned; a future data-v2
must not silently change the workbook's meaning). The workbook is presentation/audit tooling only:
it must not re-run inference, must preserve the non-predictive framing (no forecast-accuracy
claims; coin-flip is correct; the recall premium is lookahead bias, never attainable skill), and
must keep the diagnostic-control labeling of the non-PIT line. Additive to the existing repo; no
changes to the completed specs, modules, or notebooks.

## Introduction

The feature turns the completed factor-scoring assessment into a step-by-step, Excel-native audit
workbook. Its subject ("the factor workbook") loads the published, versioned data release and
walks a reviewer through the five-step storyboard, with every headline number re-derivable in
front of the reviewer from the granular evidence tables. It adds presentation and audit
capability only — it produces no new analysis results and never re-runs model inference.

## Boundary Context

- **In scope**: read-only loading of the published data release by URL (including authenticated
  access while the repository is private); the five storyboard steps as navigable workbook
  content; raw-evidence drill-down; in-workbook re-derivation of headline figures from granular
  tables; a Python data/computation layer that is fully testable on the Linux development machine
  without Excel.
- **Out of scope**: any model inference or scoring runs; producing or modifying release data;
  changes to the completed specs, modules, notebooks, or committed data of this repository;
  Excel-side automated UI testing (the Excel-only layer is verified manually on Windows);
  publishing future data releases (owned by the producing spec).
- **Adjacent expectations**: the `data-v1` GitHub Release (owned by `version-aware-factor-scoring`)
  is the sole data source and its asset names/schemas are treated as a stable contract; future
  releases (data-v2+) may exist but are never adopted silently. The known upstream data gap — raw
  per-prompt evidence missing for one screened candidate (its summary verdict stands) — is a fact
  the workbook must present honestly, not repair.

## Requirements

### Requirement 1: Versioned release data access

**Objective:** As a reviewer, I want the workbook to load the published assessment data directly
from the versioned release, so that I audit exactly the data that was published, without needing a
repository checkout or a Python environment.

#### Acceptance Criteria

1. When the workbook loads data, the factor workbook shall retrieve assets exclusively from the
   published data release addressed by an explicit release version identifier (initially
   `data-v1`).
2. The factor workbook shall display the active release version and the source address of every
   loaded dataset, so the provenance of every displayed number is visible to the reviewer.
3. While the repository hosting the release is private, the factor workbook shall support
   authenticated retrieval using a reviewer-supplied access token, and shall never store that
   token inside the workbook file or any shared artifact.
4. If a release asset cannot be retrieved (network failure, missing asset, or authorization
   failure), the factor workbook shall present a clear per-asset error state identifying the
   failed asset and cause, and shall not display substitute or stale data as if it were current.
5. When a reviewer changes the release version, the factor workbook shall treat this as an
   explicit action, shall reload all datasets from the newly selected version, and shall display
   the changed version; it shall never switch versions silently.
6. The factor workbook shall treat the release data as read-only and shall not modify, re-publish,
   or write back any release content.

### Requirement 2: Step 1 — model certification with raw-evidence drill-down

**Objective:** As a reviewer, I want the model-selection step shown with its statistical evidence
and drill-down to the raw per-prompt records, so that I can verify the certification verdicts from
the evidence rather than trusting summary numbers.

#### Acceptance Criteria

1. When the certification step is opened, the factor workbook shall present the per-candidate
   certification table: controlled separation with its confidence interval and permutation
   p-value, the positive-control result, the parse rate, the sample size, and the verdict.
2. The factor workbook shall state the certification outcome honestly: the certified set is empty,
   every screenable candidate recalls the identified macro history with statistical certainty, and
   the selected model is a documented fallback choice that runs recall-guarded.
3. When a reviewer drills into a candidate, the factor workbook shall present that candidate's raw
   per-prompt evidence records — the prompt, the model reply, the raw and standardized
   memorization features, the inclusion flag, and the dropped reason — for every evidence arm.
4. If raw evidence is unavailable for a candidate (the known upstream gap), the factor workbook
   shall display that candidate's summary verdict together with an explicit "raw evidence
   pending" marker, and shall not fail or hide the candidate.
5. Where a candidate could not be screened at all, the factor workbook shall display it as
   unscreenable with the recorded reason, presented as "not exonerated" rather than as recall-free.
6. When the certification step displays a candidate's controlled separation, the factor workbook
   shall re-derive at least the class counts and feature summary statistics from the raw evidence
   records in front of the reviewer, demonstrating consistency with the published summary.

### Requirement 3: Step 2 — coin-flip naive prediction

**Objective:** As a reviewer, I want the naive directional evaluation shown per prompt with its
statistical framing, so that I can see there is no predictive alpha and understand that this is
the expected, correct result.

#### Acceptance Criteria

1. When the naive-prediction step is opened, the factor workbook shall present the per-call
   records of the directional evaluation: date, prompt, model reply, predicted direction,
   confidence, realized direction, and correctness.
2. The factor workbook shall display the overall accuracy together with a confidence interval
   against the coin-flip level, re-derived in the workbook from the per-call records.
3. The factor workbook shall frame the coin-flip outcome as the expected, correct result of an
   honesty measurement, and shall not present the accuracy figure as a performance target to be
   improved.

### Requirement 4: Step 3 — AI macro-factor development on the numbers

**Objective:** As a reviewer, I want the factor-development step shown from loadings through
guarded tilts, so that I can follow how continuous macro exposures are built from the numbers and
how the recall guard adjusts them.

#### Acceptance Criteria

1. When the factor step is opened, the factor workbook shall present the per-rebalance regime
   loadings on the five macro axes together with each date's parse status.
2. The factor workbook shall present the per-rebalance memorization scores alongside the loadings,
   including the distribution summary of those scores.
3. When guarded exposures are displayed, the factor workbook shall re-derive the guarded tilt from
   the raw tilt and the memorization score (raw times one minus score) in front of the reviewer
   and show it matches the published guarded values.
4. The factor workbook shall present the factor-stability metrics per prompt version.
5. When the prompt-refinement decision is displayed, the factor workbook shall show the accept-gate
   inputs and the recorded decision (the refinement was rejected on contamination), and shall show
   both prompt versions' data as preserved alternatives.

### Requirement 5: Step 4 — two-line portfolio simulation

**Objective:** As a reviewer, I want both simulation lines shown side by side in the familiar
walk-forward presentation, so that I can compare the deployable recall-guarded line against the
recall-enabled control.

#### Acceptance Criteria

1. When the simulation step is opened, the factor workbook shall present two lines over the same
   rebalance stream: the point-in-time recall-guarded line and the recall-enabled control line,
   each with its equity curve, target weights, and per-date decision log.
2. The factor workbook shall label the point-in-time recall-guarded line as the deployable line
   and the recall-enabled line as a diagnostic control, and shall never present the diagnostic
   control as a deployable or recommended portfolio.
3. The factor workbook shall present the head-to-head comparison of the two lines alongside the
   existing tracks' reference metrics from the published release.
4. When per-date detail is requested, the factor workbook shall show, for each rebalance date, the
   memorization score and whether the guard adjusted that date's exposures on the deployable line.

### Requirement 6: Step 5 — luck versus skill

**Objective:** As a reviewer, I want the contamination premium contrasted with robust performance
inference, so that I can distinguish what the model remembered from what would have been skill.

#### Acceptance Criteria

1. When the luck-vs-skill step is opened, the factor workbook shall present the paired per-date
   contrast of memorization scores between the two variants, and shall re-derive the contamination
   premium and its paired effect size from those per-date records in front of the reviewer.
2. The factor workbook shall present the robust performance-stability results for the deployable
   line, the diagnostic line, and their return differential, including the long-run variance
   treatment behind them.
3. The factor workbook shall state the published conclusion in its recorded terms: the return
   differential is statistically indistinguishable from zero, so the recall premium is
   luck-compatible; where any excess of the recall-enabled line is shown, it shall be labeled
   lookahead/recall bias and never attainable skill.

### Requirement 7: Integrity, framing, and verifiability

**Objective:** As a maintainer, I want the workbook constrained to honest presentation and its
logic testable on the development machine, so that the audit tool cannot itself distort the
assessment and can be maintained without an Excel license on every machine.

#### Acceptance Criteria

1. The factor workbook shall perform no model inference and no scoring runs; every number it
   displays shall be loaded from the release or re-derived from loaded release data.
2. Where the workbook re-derives a headline figure, if the re-derived value disagrees with the
   published summary value beyond display precision, the factor workbook shall flag the
   discrepancy visibly rather than silently preferring either value.
3. The factor workbook shall preserve the non-predictive framing throughout: no forecast-accuracy
   claims, coin-flip presented as correct, the recall premium presented as bias, and the
   diagnostic control never presented as deployable.
4. The workbook's data-loading and computation logic shall be exercisable and testable on the
   Linux development machine without Excel, with the Excel-dependent layer isolated to
   presentation; its automated tests shall run in the repository's existing test workflow.
5. The work shall be additive to the repository: no modification of the completed specs, existing
   modules, notebooks, or committed data artifacts.
6. If the reviewer's environment lacks the commercial Excel add-in license or runs on a platform
   without Excel, the documentation shall state this runtime prerequisite explicitly; the
   data/computation layer shall remain usable for verification in that environment.
