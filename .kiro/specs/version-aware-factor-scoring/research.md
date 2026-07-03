# Research & Design Log — version-aware-factor-scoring

> Append-only. The research log grows in one direction: add new dated entries at the bottom;
> never rewrite earlier entries.

## 2026-06-26 — Gap analysis (requirements → codebase)

_Source: verified read of the released `recall_guard@v0.1.0` public MIA primitives
(`recall_guard/mia/{features,mcs,control}.py`, `core/nvidia_lm.py`), the completed
`track-a-macro-steering` engine (`macro_framework/steering.py`), `macro_framework/anonymize.py`,
`llm_agent.py` (`views_to_bl`), `evaluation.py`, the macro panel, and a **live empirical probe** that
built the validated calibrator and scored paired PIT vs non-PIT prompts._

### Analysis summary
- **Contract feasibility CONFIRMED.** `compute_mia_features(response, logprobs, ref_logprobs)` is a pure
  public primitive that takes the model's response + per-token logprobs (incl. `top_logprobs`) and
  returns the 5 MIA features **with no `direction+confidence` parse gate** (that gate lives only in the
  `MemoryGuardedScorer` façade). With `build_baseline` → `train` (`MCSCalibrator`) →
  `predict_proba(features, baseline)`, a version-aware scorer is fully buildable on public primitives.
- **R1.2 EMPIRICALLY CONFIRMED.** The probe scored several distinct prompts through this path and got
  genuinely different `p_memorized` (0.05–0.52). Scoring the model's **response to the version-specific
  prompt** differentiates prompts — fixing track-a-macro-steering's state-only invariance.
- **R7 is real but NOISY + CONFOUNDED.** Paired PIT/non-PIT over 3 dates: mean p_mem PIT=0.222 vs
  non-PIT=0.251 (delta +0.029, hypothesized direction) BUT per-date noise is huge and 2019-01 reversed
  (PIT 0.516 > non-PIT 0.154). Two causes: (a) n=3 is far too small; (b) **calibration-domain mismatch**
  — the calibrator is trained on FMP **news articles**, but we score **factor prompts/reasoning**, so
  `p_memorized` partly measures "news-article-likeness," and the probe's non-PIT prompt also varied
  *prose framing* beyond the identity/date/level swap. This is exactly why R7.6 (hold all else equal)
  and a **full 72-pair stream with distributional stats** (not point estimates) are required.
- **All consumption surfaces exist and are additively reusable.** `AssetMap.deanonymize_*` /
  `pseudo_to_real` give the non-PIT de-anonymization; the macro panel carries raw **and** z columns for
  PIT-vs-non-PIT levels; `views_to_bl` already computes `Q = expected_excess · confidence / 252` — so
  **tilt-as-exposure maps cleanly** by reinterpreting the fields as `tilt × conviction` (no edit);
  `steer_views` is the template for the honesty-adjusted discount; `head_to_head_report` /
  `view_stability` provide R5/R7 metrics; nb11/nb12 are the playbook templates.
- **Net: mostly additive, one genuine research risk (domain mismatch for R7).**

### Verified dependency-contract facts
- `compute_mia_features(response, logprobs, ref_logprobs, k=0.2) -> MiaFeatures` (loss, min_k, min_k_pp,
  zlib_ratio, ref_delta). Requires non-empty `logprobs` and a non-empty `top_logprobs` per token; raises
  `ValueError` otherwise. llama-4-maverick on NIM supplies `top_logprobs` (3.2 calibration succeeded).
- `MCSCalibrator.predict_proba(features, baseline) -> float in [0,1]` — pure; needs **both** the trained
  calibrator AND the `ControlBaseline` (for standardisation). So a version-aware scorer must hold both.
- `build_baseline(lm, oos_rows, ref_lm, min_valid=50, max_workers=...) -> ControlBaseline` and
  `mcs.train(model_lm, is_memorized, oos_control, baseline, ref_lm, min_auc=0.6, seed=0, max_workers=...)`
  → `MCSCalibrator` (with `.holdout_auc`, `.is_weak`). The probe rebuilt this from the on-disk 3.2
  corpora: `holdout_auc=0.94, is_weak=False` (consistent with 3.2's 0.91–0.99 range).
- `EvalRow(prompt, target_direction=0, metadata={})` adapts corpus prompt strings.
- `AssetMap` (anonymize.py): `pseudo_to_real` = {Asset_A→SWDA.L, B→XLK, C→IAU, D→BIL}; `deanonymize_*`
  helpers exist. Categories: world_equity / tech_sector / gold_commodity / short_treasury_cash.
- `views_to_bl(views, real_symbols)` (llm_agent.py): `Q = expected_excess_annualized · clip(confidence,0,1) / 252`.
  Reinterpreting `expected_excess_annualized := tilt` and `confidence := conviction` yields `Q = tilt·conviction/252`
  — the BL-tilt-as-exposure (R3) reuses the UNCHANGED method by field reinterpretation.

### Requirement → asset map (gaps tagged Missing / Unknown / Constraint)
| Req | Existing asset to reuse | Gap |
|---|---|---|
| **R1.1–1.6** version-aware scorer | public MIA primitives (build_baseline/train/compute_mia_features/predict_proba), NvidiaLM | **Missing**: new scorer that holds (baseline, calibrator) and scores per-prompt via the primitives; **R1.2 confirmed**; **R1.6** via `MCSCalibrator.is_weak`; **R1.5** via `NvidiaLM` auth error |
| **R1.4** PIT scoring | walk_forward `<rb`; as-of macro panel | **Constraint**: feed only as-of content into the scored prompt |
| **R2** regime-as-loadings | macro panel z-cols (PIT); LLM agent path | **Missing**: factor prompt + a `[-1,1]` loadings parser + per-rebalance artifact |
| **R3** BL-tilt-as-exposure | `views_to_bl` (unchanged), `MacroView` | **Missing**: tilt-factor construction; **reuse by field reinterpretation** (tilt→expected_excess, conviction→confidence) |
| **R4** honesty-adjusted exposure | `steer_views` pattern (`base·(1-p_mem)`) | **Missing**: new function mirroring it with the version-aware p_mem (R6 forbids editing steering.py) |
| **R5** prompt refinement | nb12 pattern, `view_stability`, `head_to_head_report`, `VariantMacroAgent` | **Missing**: nb`14_`; wire the version-aware scorer (so contamination now differs by version) |
| **R6** additive/non-predictive | conventions | **Constraint**: new module + nb`13_`/`14_`; no edits to recall_guard/01–12/modules/data |
| **R7** PIT vs non-PIT contrast | `AssetMap.deanonymize_*`, raw panel levels, the new scorer, `head_to_head_report` | **Missing**: nb-level contrast; **Unknown/RISK**: domain mismatch + controlled token-identical prompts + full-stream stats (see probe) |

### Implementation approaches
- **Option A — extend `steering.py` in place.** ❌ R6 forbids editing it. Rejected.
- **Option B — new `macro_framework/factor_scoring.py` + nb`13_`/`14_` (recommended MVP).** New symbols:
  a `FactorScorer` (builds/holds baseline+calibrator from the 3.2 corpora; `score(prompt) -> p_memorized`
  via the public primitives; exposes `is_weak`); `render_regime_loadings_prompt` + `parse_loadings`;
  a tilt-as-exposure builder feeding the unchanged `views_to_bl`; `honesty_adjust(loading, p_mem)`;
  the PIT/non-PIT contrast harness. nb`13_` = regime-loadings + honesty-adjusted steered variant;
  nb`14_` = prompt refinement (version-aware) + the PIT-vs-non-PIT contrast. **Effort M, Risk Medium**
  (risk concentrated in the R7 domain mismatch, not wiring). Accepts the news-trained calibrator as a
  documented **proxy**, relying on **relative** version comparison (same factor-prompt distribution, so
  the news-likeness bias largely cancels between versions).
- **Option C — Option B + a factor-prompt-distribution calibration corpus.** Same surface, but also
  build a calibration corpus whose text resembles factor prompts/reasoning (not news), to make absolute
  `p_memorized` meaningful and de-confound R7. Higher cost (new corpus source + re-calibration);
  the **right answer if absolute PIT-vs-non-PIT contamination must be trusted**, vs B's relative use.
  **Effort L, Risk Medium–High.**

### Effort & risk (by workstream)
- Version-aware scorer (R1): **M, Medium** — wire public primitives; calibrator reuse confirmed; interpretation caveat (domain).
- Regime-loadings + tilt-as-exposure (R2/R3): **M, Low–Medium** — prompt + `[-1,1]` parser + field-reinterpret into `views_to_bl`.
- Honesty-adjusted exposure (R4): **S, Low** — mirror `steer_views`.
- Prompt refinement nb14 (R5): **M, Medium** — reuse nb12; version-aware scoring now differentiates (probe-confirmed).
- **PIT vs non-PIT contrast (R7): M, High** — calibration-domain confound + controlled (token-identical) prompts + full 72-pair stream + distributional significance (n=3 was noisy with a reversal).

### Research items to carry into design
1. **Calibration-domain mismatch (the key decision).** The calibrator is news-trained; we score factor
   prompts. Decide: (B) accept as a **proxy** and use **relative** version comparison only (R1.2/R5 — the
   bias cancels between same-distribution prompts), explicitly NOT trusting absolute cross-distribution
   deltas; or (C) build a factor-prompt-like calibration corpus so absolute `p_memorized` (and R7's
   premium) is meaningful. This directly governs how much weight R7's number can bear.
2. **R7 controlled-prompt construction (R7.6).** PIT and non-PIT prompts must be **token-identical except
   the de-anonymization / dating / raw-vs-z swap** — the probe's prompts varied prose framing and that
   confounded the delta. Run the **full 72-pair stream** and report the **distribution + a paired
   significance/effect size**, not point estimates (per-date noise ≫ mean delta at n=3).
3. **What text carries the version signal.** Confirmed: scoring the model's **response** (output+logprobs)
   to the version-specific prompt — that is what differs by version. Decide whether to also score the
   prompt text itself; design the scored unit explicitly.
4. **Loadings parsing vs scoring.** `compute_mia_features` needs NO parse, but **consuming** the regime
   loadings (R2) needs a robust `[-1,1]` parser; expect format-adherence failures (cf. nb11's ~0.8
   directional parse-fail) → graceful fallback to unsteered when loadings don't parse.
5. **Honesty-adjusted exposure under weak/missing score** — reuse the R4.3/R1.6 fallback (leave raw
   unadjusted), mirroring `steer_views`.

**Document status:** written (new research log for this spec). **Next:**
`/kiro-spec-design version-aware-factor-scoring` (or `-y` to auto-approve requirements and proceed).

---

## 2026-06-26 — NUMBER-NATIVE calibration validated (drops news; supersedes B-vs-C)

_Direction from the user: "for my AI factors I am not interested in news, but in using the models to
perform inference on the numbers." The news calibration corpus was a recall_guard general-purpose
artifact; it measures "news-likeness memorization," which is the wrong target and was the source of the
gap-analysis probe's noise/reversal. Re-grounded calibration on the macro numbers themselves and
validated live._

### Design: number-native, no news
- The contamination target for "inference on the numbers" is: **does revealing the period in the numbers
  trigger recall, vs forcing inference?** Operationalized as the calibration labels, on the factor task:
  - **IS / recall class** = pre-cutoff macro states presented **identifyingly** (real date + raw levels +
    real tickers) — recall-enabled.
  - **OOS / honest class** = the **same** states presented **anonymized** (z-scores, no date, Asset_A–D) —
    recall-disabled by construction.
  - Calibrate on exactly the distribution we score → **no domain mismatch** (the news problem). Uses only
    the FRED panel (134 pre-cutoff months → ample); **no FMP / no news**.

### Live validation (`scratchpad/probe_number_native.py`, llama-4-maverick @ cutoff 2024-08-01)
- Corpus: 67 identifying (IS) + 67 anonymized (OOS), same regime-loadings task.
- **`holdout_auc = 0.9619`, `is_weak = False`** — identifying-vs-anonymized **separates strongly**;
  recall-from-numbers is detectable. (vs the news calibrator's domain-mismatched, noisy result.)
- Held-out anon (PIT) vs identifying (non-PIT) `p_memorized`:
  - 2020-03-31: **0.007 vs 0.832** (Δ +0.825) — anonymized = pure inference; revealing the COVID-crash
    date/levels = strong recall. The thesis in one row.
  - 2022-06-30: **0.757 vs 0.940** (Δ +0.183) — even anonymized 2022 scores high: extreme/unique regimes
    are **self-identifying** (the z-scores alone pin the period). Anonymization is not a binary guard;
    the contamination measure adds signal beyond it (and the honesty-adjusted exposure will correctly
    discount such states).
  - 2019-07-31: **0.404 vs 0.430** (Δ +0.026) — unremarkable month, little to recall; small delta.
- All deltas non-negative (identifying ≥ anon) — directionally consistent, unlike the news probe.

### Decisions
- **Adopt number-native calibration; drop news/FMP entirely.** This supersedes the gap analysis's B
  (news-proxy) vs C (factor-corpus) framing — the anon-vs-identifying split on the factor task is better
  than both (no mismatch, AUC 0.96, directionally consistent).
- The R7 PIT-vs-non-PIT contrast and the calibration now share the **same identifying-vs-anonymized
  axis**: calibration trains the boundary; R7 applies it over the full rebalance stream to report the
  premium. The +0.825 (2020) / +0.183 (2022) deltas show the premium is real and number-native.
- Self-identifying-regime finding (2022 anon ≈ 0.76) is a **feature**: honesty-adjusted exposure
  down-weights regimes the model recognizes even when anonymized — exactly the residual-recall guard.
- Requirements updated: Boundary Context locked decisions replace the news-calibrator reuse with this
  number-native calibrator; out-of-scope now excludes news-based calibration and FMP; ACs unchanged
  (they were written tool-neutral and survive the pivot).

### Carry to design
- Calibration cost: ~135 NIM calls (build_baseline on 67 OOS + train on 67+67) per run; consider
  persisting the trained calibrator + baseline to avoid recalibrating each notebook run.
- The IS/OOS split is **same-date identifying-vs-anonymized** (framing-only difference) — the cleanest
  control. Date-split (pre/post cutoff) is unnecessary and was thin post-cutoff; the framing split uses
  abundant pre-cutoff data.
- Verify the regime-loadings parser separately (scoring needs no parse; consuming the loadings does).

---

## 2026-06-26 — Design generated + adversarial review

_design.md written (Extension; light discovery = the two probes + gap analysis above). New leaf module
`macro_framework/factor_scoring.py` (FactorScorer, regime-loadings renderer+parser, tilt-as-exposure,
honesty_adjust, contrast harness, factor_stability) + nb13/nb14. Two parallel reviewers (gate +
technical adversary)._

- **Technical adversary: SOUND (with fixes).** Verified against real source: `compute_mia_features`
  bypasses the parse gate; the `build_baseline`/`train`/`predict_proba` flow matches; **`ControlBaseline`
  persistence is feasible** (verified by execution — fields JSON-serializable, `MCSCalibrator`/LR pickle
  round-trip, reconstructed baseline feeds `predict_proba`); `views_to_bl` field-reinterpretation
  (`Q=tilt·conviction/252`) correct with no edit; nb09 reuse (`hrp_cvar_weights_with_fixed`,
  `bl_mv_weights` Utility, `build_walk_forward_targets`) + `AssetMap` + raw/z panel all confirmed; no
  drift from the validated number-native probe.
- **Gate reviewer: FAIL → repaired.** All 35 ACs traced; boundary + file-structure + non-predictive
  integrity clean. Applied must-fixes: (1) **R1.5** — `ConfigurationError` lives only in the bypassed
  façade, so `factor_scoring` defines its OWN and raises it (empty key / `baseline.n_valid==0` / auth
  `RuntimeError`); (2) pin `from recall_guard.mia.mcs import train` (top-level alias `train_mcs`);
  (3) `calibrate` passes `ref_lm=None` (ref_delta inert). Precision fixes: `feature_order` rides on the
  pickled `MCSCalibrator` (not `ControlBaseline`); `head_to_head_report` reuse = add a "Track A (factor)"
  entry to the input dicts; `view_stability` consumes dict views (log via `.to_dict()`); `honesty_adjust`
  mirrors only the `steer_views` **discount** limb (no hard gate; R4 magnitude-only); `ContrastResult`
  gains `n_pairs` + a paired effect size (R7 over the full stream, not a noisy point estimate);
  persisted calibrator = joblib + JSON.
- Phase → `design-generated`; requirements auto-approved (`-y`).

---

## 2026-06-27 — Task 3.2 result: live calibration is WEAK under the controlled renderer (key finding)

_Ran `scripts/calibrate_factor_scorer.py` live: `FactorScorer.calibrate` (number-native, identifying IS
vs anonymized OOS on the factor task, n=60/class) for `meta/llama-4-maverick-17b-128e-instruct` @ cutoff
2024-08-01. Result: **holdout_auc = 0.338, is_weak = True** (below the 0.6 gate). Persisted; smoke score
of an anonymized factor prompt: parse_ok=True, p_memorized≈0.49._

### Why this differs from the gap-analysis probe (0.96 → 0.34)
- The 2026-06-26 number-native probe scored **0.96**, but its identifying form used a *different prose
  framing* ("As of <date>: US CPI YoY=… Assets: SWDA.L …") that differs from the anonymized form by MORE
  than the identity/date/levels — so part of that separation was **prose-style confound** (which the gap
  analysis explicitly warned about).
- The PRODUCTION renderer (task 2.1) enforces **R7.6**: the identifying form is **token-identical to the
  anonymized form except the appended date/ticker/raw-level blocks**. Under this *rigorously controlled*
  contrast, appending those blocks barely changes llama-4-maverick's response, so the MIA features for IS
  (identifying) vs OOS (anonymized) do not separate → **weak calibrator**.

### Interpretation (honest, thesis-consistent)
- Under a properly controlled contrast, the model's macro-factor reasoning shows **no separable
  period-recall** via the MIA features. That is the project thesis stated rigorously: **controlled
  contamination is (here) undetectable — "inference, not recall."** The probe's strong signal was
  partly an artifact of an uncontrolled prompt difference.
- There is a genuine tension between **R7.6 controlledness** (clean attribution, but weak signal) and a
  **stronger but confounded** identifying framing (strong signal, but the delta is partly prose). This is
  a real research result about measuring LLM contamination on numeric factor reasoning.

### Consequence (spec-compliant graceful degradation)
- `is_weak=True` ⇒ by design (R1.6 surface; R4.3 fallback) the **honesty adjustment is skipped** and the
  factor variant runs **UNSTEERED** (raw exposures). nb13/nb14 will report the weak calibrator and the
  per-prompt `p_memorized` distribution (version-aware scoring still differentiates prompts, ~0.05–0.49)
  but will NOT steer on the unvalidated score. The factor pipeline (regime-loadings → tilt → BL) is
  unaffected.
- 3.2 is COMPLETE: it calibrated, **surfaced is_weak (R1.6)**, persisted (no credential), and recorded
  this entry (R6.3). The persisted calibrator dir is gitignored (regenerable via the runner; weak/env-
  specific); the runner + this finding are committed.

### Follow-up options (user decision)
- (a) **Accept** — the controlled-contamination-undetectable result is the desired/honest outcome; run
  nb13/nb14 unsteered and document it.
- (b) **Recover separation** — relax the identifying form (a stronger, less-controlled framing) to get a
  validated calibrator, trading R7.6 controlledness for signal; would re-open the 2.1 renderer.

---

## 2026-07-03 — Directive: certified no-recall model selection (R8) + recall_guarded rename

_User directive: nb13/nb14 are un-deferred. Before running them, SELECT the model from recall_guard
evidence — "the one AI model that doesn't recall and where we are certain" — and run the factor
pipeline on it (the certified model is BOTH the loadings generator and the scored model)._

### Spec amendments
- **Requirement 8** (certified no-recall model selection) appended to requirements.md; **task 3.3**
  (certification screen: module + live run) inserted; 4.1/4.2 **resumed**, now `_Depends: … 3.3_`.
- **API rename** (commit `e9b1ed9`): `HonestyConfig`→`RecallGuardedConfig`,
  `honesty_adjust`→`recall_guarded_adjust`, `honesty_config`→`recall_guarded_config`. No behavior
  change; 164 tests green. Spec prose "honesty-adjusted" refers to the `recall_guarded_*` API.

### Certification design (the "where we are certain" machinery)
- **Reframing of the 3.2 finding:** `holdout_auc=0.338` sat on a 30-prompt holdout
  (`n_per_class=60`, `mcs.train` holds out 25%) — ~1.5 SE below 0.5, i.e. statistically consistent
  with chance ("no detectable recall"), not evidence of anti-signal. `is_weak` was a threshold
  verdict; R8 demands a statistical one.
- **Certainty stats, offline:** gather the standardized per-prompt MIA features once (live), then
  bootstrap-resample the train/holdout split for an AUC confidence interval and permute labels for a
  p-value against AUC=0.5. No extra live calls for the statistics.
- **Positive control per candidate:** the deliberately prose-confounded identifying framing (the
  2026-06-26 probe form that hit 0.96 on maverick) must fire (perm p < 0.05); otherwise the verdict
  is "detector unvalidated", never "no recall".
- **Common screening cutoff 2023-12-01:** pre-cutoff states are trained-on for every candidate
  (llama-3.1 cutoff 2023-12, gpt-oss ~2024-06, maverick ~2024-08) and AUCs are comparable across
  candidates. Panel has 196 monthly rows (2010-01→2026-04) → ~167 pre-2024 states → n_per_class up
  to ~120, holdout ~60 (double the 3.2 run).
- **Verdict rule (R8.4):** certified-no-recall ⇔ controlled AUC indistinguishable from chance
  (CI ∋ 0.5, perm p > 0.1) ∧ positive control fires ∧ factor-task parse rate ≥ 0.9. Significant
  above-chance controlled AUC ⇒ rejected (recalls). Winner's calibrator re-persisted at its own true
  cutoff.
- **Candidates:** the logprob-bearing NIM pool from the 2026-06-26 re-probe (maverick,
  llama-3.1-8b-instruct, gpt-oss-20b, + whatever else the live probe admits), capped for cost.

---

## 2026-07-03 — Task 3.3 live result: the certified-no-recall set is EMPTY

_The R8 screen ran evidence-grade (raw per-prompt evidence persisted per candidate under
`data/norecall_screen/evidence/<model>/` — gitignored, ships via the GH data release) across every
servable logprob-bearing NIM candidate, common cutoff 2023-12-01, bootstrap CI + permutation p,
prose-confounded positive control, parse gate._

### Certification table (data/norecall_screen/results.json)
| model | verdict | controlled AUC [95% CI] | perm p | pos.ctrl AUC (p) | parse | n/class |
|---|---|---|---|---|---|---|
| openai/gpt-oss-120b | recalls | 0.633 [0.598, 0.731] | 0.004 | 0.729 (0.002) | 0.4* | 167 |
| meta/llama-4-maverick-17b-128e | recalls | 0.637 [0.561, 0.721] | 0.008 | 0.881 (0.002) | 0.0** | 120 |
| microsoft/phi-4-mini-instruct | recalls | 0.695 [0.646, 0.781] | 0.002 | 0.885 (0.002) | 1.0 | 120 |
| openai/gpt-oss-20b | recalls | **0.926 [0.885, 0.961]** | 0.002 | 0.799 (0.002) | 1.0 | 167 |
| meta/llama-3.3-70b-instruct | screen_failed | — | — | — | — | — |

\* timeout attrition at n=200 serving load, not format failure. \** maverick narrates 512 tokens
without emitting the loadings JSON (live-verified); a format failure, not attrition.

### Findings
1. **R8 answer: NO servable NIM candidate is certifiably recall-free on the macro numbers.** We are
   statistically certain of the OPPOSITE for all four screenable models (perm p ≤ 0.008 each,
   detector validated per model). The identifying additions (date + tickers + raw levels) shift
   token-probability behavior detectably in every model.
2. **The 2026-06-27 3.2 "weak/no-signal" finding is superseded as under-powered**: same model
   (maverick), controlled renderer, but n=60/class on a 30-prompt single split → AUC 0.34; at
   n=120/class with CV + permutation the same setup shows AUC 0.637, p=0.008. The earlier entry
   stands as written; its interpretation ("controlled contamination undetectable") does not survive
   the power upgrade.
3. **Size gradient inside one family**: gpt-oss-20b (0.926) ≫ gpt-oss-120b (0.633). The smaller
   model carries the far sharper memorization signature — consistent with less capacity forcing
   more verbatim storage relative to generalization.
4. gpt-oss-120b resolved from `inconclusive` (n=120, p=0.098) to `recalls` (n=167, p=0.004) — the
   max-power rescreen settled the fence exactly as the trend suggested.
5. llama-3.3-70b is not servable non-streaming at screen volume on this endpoint (15–120 s
   timeouts); recorded as unscreenable, not exonerated.

### Consequence — the strong-calibrator branch
With recall demonstrably present, the number-native calibrators are NOT weak → per R4.3 the
`recall_guarded_adjust` discount ENGAGES (no passthrough). The Excel storyboard S1 becomes:
"no model is certifiably recall-free — recall is quantified per model; the pipeline runs
RECALL-GUARDED." Fallback model selection for nb13/nb14 (generator + scored model) is a user
decision: phi-4-mini (parse 1.0, fastest serving, AUC 0.695), gpt-oss-20b (parse 1.0, sharpest
signature 0.926 → most vivid guarding + premium measurement), gpt-oss-120b (numerically least
recall 0.633 but ~4× slower serving with attrition).

---

## 2026-07-03 — Selection: gpt-oss-20b (user-confirmed) + production calibrator persisted

- With the certified set empty, the user selected **openai/gpt-oss-20b** for the recall-guarded
  pipeline (recommended option: sharpest recall signature 0.926 → most measurable guarding/premium;
  parse 1.0; workable serving). phi-4-mini (ops-best) and gpt-oss-120b (least-recall, slow) declined.
- Production calibrator persisted at the model's TRUE cutoff **2024-06-01**, n=170/class:
  **holdout_auc = 0.9668, is_weak = False** → `data/factor_calibrator_openai_gpt-oss-20b/`
  (joblib + JSON, credential-free; run_header.json committed context lives in the dir, gitignored).
- Smoke: anonymized factor prompt scores p_memorized = 0.384 (guard scales such tilts ×0.616);
  identifying prompts are expected to score far higher (screen CV AUC 0.926).
- Strong-calibrator branch CONFIRMED: per R4.3 the `recall_guarded_adjust` discount ENGAGES —
  nb13's deployable line is genuinely guarded, nb14's premium is measured with a validated detector.
