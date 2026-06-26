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
