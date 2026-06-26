# Research & Design Log — track-a-macro-steering

> Append-only. The research log grows in one direction: add new dated entries at the bottom;
> never rewrite earlier entries. Each entry is a baseline snapshot or a decision/finding.

## Summary
- **Feature**: `track-a-macro-steering`
- **Discovery Scope**: Extension of an existing system (Global_Macro_AI_Factors) + new analysis
  playbooks; integrates the external `recall_guard` library.
- **Key Findings (2026-06-25)**:
  - PIT discipline already exists structurally (`walk_forward.py` slices `< rebalance_date`;
    rolling 60-month z-scores, not full-sample).
  - Track A's LLM path (DSPy → OpenRouter Claude) does not expose logprobs; the memorization
    score must come from a parallel NIM call on the same prompt.
  - The macro signal is sparse/coarse: ~72 unique monthly macro states over 2019–2024.

---

## 2026-06-25 — Baseline capture (existing state)

_Source: codebase scout of `/home/mc/projects/Global_Macro_AI_Factors` (commit on `main`)._

### Track A — `macro_framework/llm_agent.py`, `notebooks/09_track_a_llm_agent.ipynb`
- `LlmMacroAgent.views_for_state(macro_state: dict, asset_snapshot: list) -> (list[MacroView], reasoning_text)`.
  - `macro_state = {cpi_yoy_z, t10y2y_z, hy_oas_z}` (rolling 5y z-scores, monthly, rounded ~2dp).
  - `asset_snapshot = [{id: "Asset_A", category, trailing_12m_return, trailing_vol_ann}, …]` — no ticker/date.
  - `MacroView(asset_long, asset_short|None, expected_excess_annualized, confidence∈~0.4–0.6, rationale)`.
- `views_to_bl(views, real_symbols) -> (P, Q)`; `Q = (expected_excess * confidence) / 252`.
- LLM: DSPy 3.1.3 → OpenRouter `claude-sonnet-4.6`, temperature 0, max_tokens 1024, diskcache keyed
  by `(prompt_version, macro_state, asset_snapshot)`. Prompt text = `AGENT_INSTRUCTIONS` + the
  `MacroViewSignature` docstring. **Logprobs NOT exposed.**
- Notebook 09: 72 monthly rebalances (2019–2024); base = HRP-CVaR (BIL pinned 25%); posterior = BL-MV;
  final blend 0.7·HRP + 0.3·BL; simulate via vectorbt; persists targets/equity/`track_a_agent_log.json`.

### Macro data + stats — `macro_framework/macro.py`, `notebooks/06_macro_zscores.ipynb`
- `build_macro_panel(series_map, freq="ME", zscore_window=60)`; `rolling_zscore(s, window=60, min_periods≈30)`.
- Series: CPIAUCSL→cpi_yoy, T10Y2Y→t10y2y, BAMLH0A0HYM2→hy_oas (+ `_z`). ~196 months (2010–2026).
- `data/macro_panel_monthly.parquet` holds raw + z columns. Existing stats: |z| overlay vs drawdown (nb06);
  Track B rule-based regimes (nb08). **No existing macro→agent steering / causal regression.**

### PIT / walk-forward — `macro_framework/walk_forward.py`
- `build_walk_forward_targets(prices, rebalance_dates, weight_fns, macro_panel, lookback_days=756, min_history=60)`.
- At each `rb`: `prices[index < rb].tail(lookback)`, `macro_panel[index < rb]` → `StrategyContext`. Lookahead
  is structurally impossible (slicing, not discipline). Monthly rebalance = first trading day.

### Anonymization — `macro_framework/anonymize.py`
- `AssetMap` fixed: SWDA.L↔Asset_A (world_equity), XLK↔Asset_B (tech), IAU↔Asset_C (gold), BIL↔Asset_D (cash).
  `pseudo_assets()` → `{id, category}` only. Qualitative recall-avoidance.

### Evaluation — `macro_framework/evaluation.py`, `notebooks/10_head_to_head_evaluation.ipynb`
- Metrics across Baseline / Track A / Track B: total/annualized return, vol, Sharpe/Sortino/Calmar, max DD,
  crisis (2022) analytics, anticipation lead time (BIL+IAU ≥ 40%), turnover, Track-A `view_stability()`.

### Conventions
- Python 3.12 (`<3.13`); deps include dspy==3.1.3, openai, riskfolio-lib, vectorbt, diskcache.
- Notebooks numbered 01–10 → new ones are `11_…`, `12_…`. `.llm_cache/` persists Claude responses.
- Regenerable: macro panel, baseline/track_a/track_b targets+equity (track_a needs cached LLM responses).
  Precious: `track_a_agent_log.json` (rationales).

### recall_guard integration surface (the new dependency)
- `MemoryGuardedScorer.calibrate(api_key, model, is_memorized: Sequence[str], oos_control: Sequence[str],
  reference_model=None, …)` → `.score(prompt)` / `.score_many(...)` → `GuardedScore(signal, p_memorized,
  memguard_confidence, features, fail_reason)`; `ConfigurationError` on missing/rejected NIM key.
- Recommended: **parallel NIM scorer** — after the DSPy call, score the identical anonymized, z-scored,
  PIT prompt on a NIM model; attach `p_memorized` per view; steer/gate BL confidence
  (`confidence * (1 - p_memorized)`) before `views_to_bl`. recall_guard's NIM endpoint is live.

## Open decisions / Research Needed (carry to requirements/design)
1. **IS/OOS calibration corpus.** recall_guard splits prompts by the NIM model's training cutoff
   (IS = before, OOS = after). For anonymized macro prompts, the split key is the macro state's as-of
   date. Candidate sources: mine the 2019–2024 agent log split by the chosen NIM model's cutoff; or
   synthesize macro-state prompts spanning pre/post cutoff. Risk: small/homogeneous corpus → unstable AUC.
2. **NIM model for scoring.** Which model (e.g. `meta/llama-3.1-8b-instruct`, `openai/gpt-oss-20b`)? Its
   cutoff defines the IS/OOS split and the cost/latency (~72 calls, then cached).
3. **Steering objective (a).** Confirm: steer toward statistically-grounded, low-`p_memorized` views, NOT
   alpha. Define the concrete steering levers (input shaping vs confidence post-processing) and a
   non-predictive success metric (e.g. lower `p_memorized` + unchanged/improved head-to-head, stable views).
4. **Prompt-refinement loop (b).** How to compare prompt versions: `p_memorized` distribution + view
   stability + head-to-head deltas, under PIT. Avoid overfitting prompts to the eval window.
5. **Z-score quantization.** Rounding z to 2dp can flip views at threshold edges; consider binning
   (e.g. 0.5σ) for prompt/cache stability — design decision.
6. **Confidence-adjustment function.** memguard linear `conf*(1-p_mem)` vs a gate vs sigmoid — pick and
   justify in design.

---

## 2026-06-25 — Gap analysis (requirements → codebase)

_Source: verified read of `macro_framework/{llm_agent,walk_forward,evaluation,macro}.py`,
`pyproject.toml`, `data/track_a_agent_log.json`, the notebook tree, and the released
`recall_guard@v0.1.0` public API. Extends the baseline entry above; does not replace it._

### Analysis summary
- **Mostly additive, low structural risk.** Every existing integration surface is reusable
  *without modification*: the agent's `views_to_bl` (Q = `expected_excess·confidence/252`),
  `evaluation.head_to_head_report(pfs, targets)`, `evaluation.view_stability`, the PIT slicing in
  `walk_forward.build_walk_forward_targets`, and the macro panel parquet. Steering plugs in as a
  **new wrapper module + two playbooks (`11_`/`12_`)**, satisfying R6 (no edits to `01–10` /
  existing modules / existing `data/`).
- **The 72-state agent log is a ready-made PIT corpus.** `data/track_a_agent_log.json` holds, per
  the 72 monthly rebalances 2019-01→2024-12, the `macro_state`, `reasoning`, `views`, and
  `weights`. nb11/nb12 can run offline against it (reuses data, no re-calls to the OpenRouter
  agent) — only the **NIM scoring** path needs live calls.
- **The single hard feasibility risk is calibrator validity, not wiring.** `MemoryGuardedScorer`
  needs an IS/OOS corpus split by the NIM model's training cutoff with `min_valid=50`/`min_auc=0.6`
  defaults. 72 *intentionally anonymized, dateless* macro prompts are small, homogeneous, and (by
  design) poorly memorizable → likely weak/invalid AUC. R1 AC7 already anticipates this ("surface
  weak calibrator, don't steer"). Resolving the calibration-corpus source is the #1 design input.
- **Dependency is compatible.** `recall-guard@v0.1.0` (floor `>=3.12`) fits this project's
  `>=3.12,<3.13`; its lean runtime adds `scikit-learn` (new here) + numpy/rich/pyyaml/requests/
  python-dotenv and pulls **no** matplotlib/vectorbt. Add via
  `uv add "recall-guard @ git+https://github.com/norandom/memguard_alpha.git@v0.1.0"`.

### Verified integration surface (new facts beyond the baseline)
- **`LlmMacroAgent.views_to_bl(views, real_symbols)`** is the steering hook. It already
  confidence-gates: `Q = expected_excess_annualized · clip(confidence,0,1) / 252`. The whole R3
  lever is therefore "shape `confidence` (or `Q`) before BL" — achievable by building a **new
  `MacroView` list with adjusted confidence and calling the *unchanged* `views_to_bl`**, so
  `llm_agent.py` is never touched (R6 AC1).
- **Per-rebalance, not per-view, `p_memorized`.** The scored object is the *input prompt* for a
  rebalance (one prompt → 1–3 views). So one `p_memorized` per rebalance is broadcast to that
  rebalance's views (R1 AC2 = "for that decision"; R3 AC1 applies it to "each view").
- **The "same PIT prompt" is reconstructable but not byte-identical to DSPy's wire format.** The
  agent sends `AGENT_INSTRUCTIONS` + the `MacroViewSignature` doc + `macro_state` (rounded 2dp) +
  `assets` (rounded 3dp). R1 AC2 says *content*, so a faithful textual reconstruction is
  acceptable; exact-bytes would require capturing DSPy's rendered messages (`inspect_history`).
- **Evaluation is purely additive.** `head_to_head_report(pfs, targets)` and `crisis_analytics`
  iterate over dict entries — adding a `"Track A (steered)"` key to both dicts slots the variant in
  with **zero change to `evaluation.py`** (R5 AC1). `view_stability(views_log)` already exists →
  reuse directly for R4 AC2.
- **Prompt versioning already exists.** `PROMPT_VERSION = "v1"` is baked into the agent's cache key,
  so adding `"v2"/"v3"` prompt variants is additive and cache-isolated (R4 AC4).
- **recall_guard surfaces the R1 AC5/AC7 signals natively:** `ConfigurationError` on
  missing/rejected NIM key; `scorer.holdout_auc` / `scorer.is_weak` for calibrator quality. Wire a
  guard that falls back to *unsteered* views when `is_weak`.

### Requirement → asset map (gaps tagged Missing / Unknown / Constraint)
| Req | Existing asset to reuse | Gap |
|---|---|---|
| **R1.1** dep pinned | `pyproject.toml` PEP 621 `dependencies` | trivial add; **Constraint**: git dep + NIM key at runtime |
| **R1.2** PIT `p_memorized`/decision | `recall_guard.MemoryGuardedScorer`, agent-log `macro_state` | **Missing**: scoring layer; **Unknown**: prompt reconstruction fidelity |
| **R1.3** separate inference path | recall_guard NIM path (≠ DSPy/OpenRouter) | satisfied by construction |
| **R1.4** PIT scoring | `walk_forward` `<rb` slicing; log is already as-of | **Constraint**: feed only as-of content |
| **R1.5** config error | `recall_guard.ConfigurationError` | propagate only |
| **R1.6** additive/disabled | — | **Missing**: enable/disable switch (design) |
| **R1.7** weak calibrator surfaced | `scorer.is_weak` / `.holdout_auc` | wire a guard |
| **R2.1–.2** PIT regime + z-summary | `macro_panel_monthly.parquet`, `rolling_zscore`, `walk_forward` | **Missing**: nb`11_`; **Unknown**: regime taxonomy (reuse Track B `mc_regime.py`/nb08 vs fresh z-bin) |
| **R2.3** reuse macro artifacts | macro panel parquet | satisfied (read-only) |
| **R2.4** steering-signal artifact | — | **Missing**: new `macro_steering_signals_*` artifact + macro→view consistency rule |
| **R3.1–.4** confidence shaping | `views_to_bl` (unchanged) + new wrapper | **Missing**: steering module; **Unknown**: adj. function (linear/gate/sigmoid) + threshold |
| **R3.3/3.5/R6** own targets+log | `build_walk_forward_targets`, vectorbt sim | **Missing**: new `track_a_steered_*` artifacts |
| **R4.1–.5** prompt refinement | `PROMPT_VERSION`, diskcache key, `view_stability`, `head_to_head_report` | **Missing**: nb`12_`; **Unknown**: prompt-version set + overfit guard |
| **R5.1–.3** non-predictive eval | `head_to_head_report` (additive dict key) | **Missing**: `p_memorized`-distribution reporter (nb-local, since nb10 untouched) |
| **R6.1–.4** additive delivery | numbering `11_`/`12_`, append-only log | **Constraint**: new filenames only; no edits to `01–10`/modules/`data/` |

### Implementation approaches
- **Option A — extend the agent / notebooks in place.** ❌ Conflicts with R6 (no edits to
  `llm_agent.py` or `01–10`). Rejected.
- **Option B — one new `macro_framework/steering.py` module + nb`11_`/`12_` (recommended).**
  A new module owns: (1) `score_rebalance(prompt) → GuardedScore` (thin recall_guard adapter +
  weak-calibrator guard), (2) `steer_views(views, p_memorized, macro_signal, threshold) →
  views'` (returns a new `MacroView` list; existing `views_to_bl` consumes it unchanged), (3)
  `macro_steering_signal(macro_hist) → regime+confidence-adjustment`. nb`11_` builds the PIT macro
  characterization + steering-signal artifact; nb`12_` runs the prompt-version sweep and the
  head-to-head deltas. Clean separation, fully additive, isolated tests. **Effort M, Risk Medium**
  (risk concentrated in calibration, not wiring).
- **Option C — hybrid: B's module, but calibrate on a separate purpose-built corpus.** Same module
  surface as B; differs only in *where the calibrator's IS/OOS corpus comes from* (a dedicated
  pre/post-cutoff text corpus rather than the 72 macro prompts), to clear `min_auc`. Likely the
  real shape once the calibration risk is assessed in design. **Effort M–L, Risk Medium–High**
  (depends on sourcing a valid corpus).

### Effort & risk (by workstream)
- Dependency + scoring adapter (R1): **S, Low** — add dep, thin recall_guard wrapper, propagate errors.
- **Calibration corpus + NIM model/cutoff (R1.2/1.7): M, High** — the project's pivotal unknown.
- nb`11_` macro characterization + steering signal (R2): **M, Medium** — PIT stats exist; regime taxonomy + macro→view consistency rule are new.
- Steering module + steered variant artifacts (R3): **M, Medium** — wiring is clean; the adjustment function/threshold need justification.
- nb`12_` prompt refinement + eval integration (R4/R5): **M, Medium** — reuses versioning + eval; overfit-to-window guard is the watch-item.

### Research items to carry into design
1. **Calibration-corpus source & NIM model (blocking).** Decide between: (a) calibrate on the 72
   anonymized macro prompts split by the NIM model's cutoff — simplest, but expect weak AUC →
   R1 AC7 fallback dominates; (b) calibrate on a separate, clearly pre/post-cutoff corpus, then
   *score* the macro prompts — more likely to clear `min_auc`. Pick the NIM model (its cutoff is
   the split key; e.g. `meta/llama-3.1-8b-instruct`) and record cost (~72 cached calls/sweep).
   Define what the pipeline does when `scorer.is_weak` (almost certainly: run unsteered, report the
   weakness — which is itself a valid, honest research finding).
2. **Prompt reconstruction fidelity.** Faithful textual rebuild of the agent prompt vs. capturing
   DSPy's exact rendered messages. Confirm the chosen rendering is what gets scored and is stable.
3. **Macro→view consistency rule (R2.4/R3.4).** How a regime label maps to a per-view confidence
   multiplier (e.g. down-weight long-risk views in a stagflation/credit-stress regime). Reuse Track
   B's `mc_regime.py` taxonomy or define a fresh z-bin regime?
4. **Confidence-adjustment function & threshold (R3.1/3.2).** linear `conf·(1-p_mem)` vs gate vs
   sigmoid; the exclusion threshold on `p_memorized`; how the macro multiplier and the contamination
   multiplier compose.
5. **z-score quantization** (carry-over): binning (e.g. 0.5σ) for prompt/cache stability.
6. **Prompt-version set & overfit guard (R4).** Which `v2/v3` prompts; accept a refinement only on
   non-degraded head-to-head **out of the tuning window** to avoid fitting the eval period.

**Document status:** appended (append-only log preserved). **Next step:**
`/kiro-spec-design track-a-macro-steering` (or `-y` to auto-approve requirements and proceed).

---

## 2026-06-25 — Design discovery & synthesis

_Source: verified read of the released `recall_guard@v0.1.0` source — `recall_guard/__init__.py`,
`recall_guard/harness/scorer.py`, `recall_guard/dataset/__init__.py`. Light discovery (Extension):
the codebase integration surface was already mapped in the two entries above; this entry records the
**dependency-contract** facts that shape the design plus the synthesis decisions._

### Decisive dependency-contract findings (new)
1. **The scorer gates on a parseable directional response.** `MemoryGuardedScorer._build_guarded_score`
   calls `_parse_direction(content)` and `_parse_confidence(content)` **before** computing MIA
   features; if either is `None` it returns `FAIL_PARSE` with `p_memorized = None`. The Track A agent
   prompt asks for a **JSON array of Black-Litterman view triples**, which would not parse as a single
   `direction∈{-1,0,1}` + `confidence∈[0,1]`. **⇒ The scored prompt must reframe the same PIT macro
   content into recall_guard's directional-forecast format**, not reproduce the BL-view prompt
   verbatim. R1 AC2 says "same … prompt *content*" — content (the anonymized z-scores + anonymized
   asset snapshot) is preserved; only the *response framing* differs to satisfy the parser. This
   **resolves** open research item "prompt reconstruction fidelity" in a specific direction: do NOT
   replicate DSPy's wire format; build a recall_guard-compatible directional prompt carrying the same
   macro content, using the **same template as the calibration corpus** so features are comparable.
2. **`min_valid=50` per class ⇒ the 72-state log cannot self-calibrate.** `calibrate` builds the OOS
   baseline (`build_baseline(... min_valid=50)`) and trains the calibrator from `is_memorized` /
   `oos_control`; 72 dateless prompts cannot yield two ≥50-valid classes split by cutoff.
   **⇒ Two-corpus design is mandatory**, not optional: calibrate on a separate dated corpus, then
   *score* the 72 macro prompts.
3. **`recall_guard.dataset.fmp_corpora` is a public, lazy API** — `build_calibration`, `update_oos`,
   `ArticleRecord`. It is the natural calibration-corpus source (FMP financial text, split by the NIM
   model's training cutoff → real IS/OOS separation → a calibrator that can clear `min_auc`). It is
   lazily re-exported, so importing it does not pull matplotlib/vectorbt. **Constraint:** needs an FMP
   key (already provisioned in this project's `.env`).
4. **Exact `calibrate` contract:** `calibrate(*, api_key, model, is_memorized, oos_control,
   reference_model=None, min_auc=0.6, min_valid=50, seed=0, max_workers=8, timeout_s=45.0,
   min_call_interval_s=0.0, lm_factory=None)`. Quality via properties `holdout_auc` / `is_weak`
   (R1.7). Scoring via `score(prompt)` / `score_many(prompts)`; `score_many` is the path for the 72
   rebalances (one NIM call each, order-preserving).
5. **Only `p_memorized` steers — never the scorer's `signal`/`raw_confidence`.** Those two fields are
   byproducts of the parse contract (the scorer runs a directional micro-task to elicit logprobs).
   Using them as portfolio factors would be a predictive use, which is explicitly out of scope. The
   steering layer consumes `p_memorized` (and may *report* `memguard_confidence`) only.

### Synthesis decisions
- **Generalization.** R1 (measure `p_memorized`), R2.4 (macro→view consistency), and R3 (shape
  confidence) are one capability: a **confidence-shaping pipeline**
  `adjusted = base · (1 − p_memorized) · macro_consistency`, with a hard gate at a `p_memorized`
  threshold. R4's prompt comparison is the same pipeline run over different prompt streams. One
  `steer_views(views, signal, config)` interface covers all three (contamination-only, macro+
  contamination, and A/B prompt) without bespoke variants.
- **Build vs adopt.** Adopt recall_guard for scoring + calibrator quality; adopt `fmp_corpora` for the
  calibration corpus; adopt existing `build_walk_forward_targets` + vectorbt for the steered backtest;
  adopt `head_to_head_report` + `view_stability` for eval (additive dict key, no edits). **Build**
  (small, new module): the directional `PromptRenderer`, a deterministic **z-bin regime** labeler, the
  `ViewSteerer`, and a `VariantMacroAgent` subclass for prompt refinement. _Rejected:_ reusing Track
  B's `mc_regime.py` — it is a stochastic Monte-Carlo/Nash regime coupled to Track B's portfolio
  logic; a deterministic, explainable z-bin regime is the right fit for a steering signal and keeps
  the boundary clean.
- **Prompt refinement without editing `llm_agent.py` (R6.1).** `AGENT_INSTRUCTIONS` / the signature /
  `PROMPT_VERSION` are baked into the existing module. Vary prompts by **subclassing**
  `LlmMacroAgent` (`VariantMacroAgent`) that injects custom `instructions` and uses a **per-variant
  `cache_dir`** (e.g. `.llm_cache_v2/`). Per-variant cache dirs are required because the base
  `_cache_key` keys on the module-level `PROMPT_VERSION="v1"` only — distinct variants would otherwise
  collide in one diskcache. Subclassing + separate cache dirs is fully additive; the base module is
  untouched.
- **Confidence-adjustment function (R3.1).** Chosen: **linear discount with hard gate** —
  `adjusted = base · (1 − p_memorized) · consistency`, and exclude the view when
  `p_memorized ≥ threshold`. Rationale: `(1 − p_memorized)` mirrors recall_guard's own MemGuard
  discount (`raw·(1−p_mem)`), is monotone and interpretable; the gate satisfies R3.2's hard
  exclusion. _Rejected:_ sigmoid (extra opaque parameters, no requirement need).
- **Simplification.** No plugin registry/config framework. A frozen `SteeringConfig`
  (threshold, consistency floor, enabled flag) + one steering module + two notebooks. If scoring is
  disabled or `scorer.is_weak`, `steer_views` returns the views unchanged → the steered variant
  degrades exactly to Track A (satisfies R1.6 additivity and R1.7 "don't steer on an unvalidated
  score").

### Residual research/design risks (carried into design Open Questions)
- Calibration AUC depends on the FMP corpus + chosen NIM model cutoff; if `is_weak`, the honest
  outcome is "run unsteered, report the weakness" — itself a valid contamination finding.
- The directional scoring template must elicit a parseable direction+confidence from the chosen NIM
  model; verify on a small sample before the full 72-state run.
- Macro→view consistency rule (regime → preferred asset categories) is a deliberate, documented
  heuristic, not a fitted model — must stay non-predictive.

### Design review outcome (2 parallel adversarial reviewers, applied)
- **Gate reviewer** (coverage/boundary/executability): FAIL → repaired. All 30 IDs traced; boundary +
  file-structure populated. Must-fixes applied: (1) `fmp_corpora.build_calibration` contract corrected,
  (2) `VariantMacroAgent` documented as overriding private `_ensure_ready`, (3) per-component symbol
  ownership table added to File Structure Plan (no orphans), (4) `views_to_bl` instance ownership +
  `real_symbols` sourcing made explicit in the steered flow.
- **Technical adversary** (claims vs real recall_guard + Track A source): HAS-ERRORS → repaired.
  Confirmed: parse-before-features gate is real (directional-reframe is correct); `views_to_bl` Q-lever;
  `_cache_key` collision (per-variant cache dirs needed); additive dict-keyed eval; strict `<rb` PIT
  slicing; all relied-on recall_guard exports/properties exist. Corrected: real
  `build_calibration(out_dir, cutoffs: dict[str, date], …) -> tuple[Path, Path]` (writes JSONL, read
  back to lists); `min_valid=50` gates only the **OOS control baseline** (IS/OOS training needs ≥2/class)
  — two-corpus conclusion preserved, rationale fixed.
- Nice-to-haves folded in: `score_rebalances` naming unified; `<nim_model>` filename slugification;
  gate-inertness documented as expected; `memguard_confidence` marked report-only (never a steering input).
- **Phase set to `design-generated`; requirements auto-approved (`-y`).**

### Tasks outcome (2026-06-25)
- `tasks.md` generated: 4 majors / 12 sub-tasks; all 30 requirement IDs (1.1–6.4) mapped; phase
  `tasks-generated`, requirements + design + tasks auto-approved (`-y`), `ready_for_implementation: true`.
- Structure: (1) dependency foundation; (2) core steering module — 7 sequential sub-tasks (one shared
  module file → no `(P)`); (3) integration — steered walk-forward composition + live calibration/template
  smoke; (4) validation — nb11 / nb12 (the only `(P)` pair: separate notebook files).
- One independent task-graph sanity review (fresh subagent): NEEDS_FIXES → repaired (coverage + the
  non-predictive/additive constraints were confirmed clean). Applied: split the scoring adapter into
  calibration (2.2) vs scoring path (2.3); promoted the one-time live calibration build to a discrete
  persisted-artifact task (3.2); made the steered composition agent-type-agnostic so prompt refinement
  does not hard-depend on the variant; added the append-only research-log serialization caveat to the
  parallel notebooks (4.1/4.2). Re-verified inline: 30/30 IDs, dependencies consistent.

---

## 2026-06-25 — Implementation outcome (offline engine complete; live tasks blocked)

_Autonomous `/kiro-impl` run: fresh implementer + independent adversarial reviewer per task, one at a
time, parent-side selective commits between. All offline tasks landed on `main`._

### Done (9/12 tasks; 82 tests green)
- **1.1** foundation — `recall_guard@v0.1.0` git dep (resolves + builds) + pytest/pytest-mock dev tooling + `tests/` (`8c4c7c8`).
- **2.1** `render_directional` + shared directional template (parses under recall_guard's real evaluator) (`45718d8`).
- **2.2** `ScoringAdapter.calibrate_from_fmp` (build FMP corpus → read JSONL back → calibrate) + `is_weak`/`holdout_auc` + ConfigurationError (`f4f061a`).
- **2.3** `score_rebalances` (order-preserving wrapper; ConfigurationError propagation) (`288ad13`).
- **2.4** `SteeringSignal`/`characterize`/`write_steering_signals` — defensive PIT z-bin regime + real-AssetMap consistency (`1a60cb6`).
- **2.5** `SteeringConfig`/`steer_views` — linear discount + hard gate + passthrough; single-flooring; non-mutating (`9a4c665`).
- **2.6** `VariantMacroAgent` — overrides private `_ensure_ready` (one-line behavioral diff vs base) + per-variant cache (`5062577`).
- **2.7** `score_distribution_report` — pure p_memorized distribution + parse_fail_rate; non-predictive (`63a9c4f`).
- **3.1** `SteeredDecision`/`steer_rebalance`/`make_steered_weight_fn` — composes the leaf pieces + the unchanged `views_to_bl`; HRP/BL injected (R6.1); all gating fallbacks → base (`27d1f7d`).
- The whole steering engine lives in the single leaf module `macro_framework/steering.py`; only `pyproject.toml`
  (the dep) and new test files were touched. Notebooks 01–10, existing modules, and `data/` are unchanged (R6.1).
- Every task passed an independent adversarial reviewer (mechanical + judgment, verifying claims against the
  REAL recall_guard/Track A source — directional parse-gate, `score_many` order, AssetMap categories, the
  `_ensure_ready` byte-diff, single-flooring, injection-not-duplication, non-predictive boundary).

### Blocked (3/12 — live credentials)
- **3.2** (live calibration build + directional smoke), **4.1** (nb11 steered-variant playbook), **4.2** (nb12
  prompt-refinement playbook) require live `NVIDIA_API_KEY` + `FMP_API_KEY` (+ `OPENROUTER_KEY`). No `.env` /
  credentials are provisioned in this environment, and their observables are live-run artifacts that cannot be
  produced or verified offline. All of their code prerequisites are complete and unit-tested; unblock by
  providing the keys, then 3.2 runs as-is and the two notebooks can be authored against the finished engine.

### Process note
- One reviewer subagent ran `git checkout` on uncommitted work (2.6) and restored from backup; the parent
  independently re-verified the tree before committing. Subsequent reviewers were instructed never to
  reset/checkout uncommitted task work. (Also recorded in tasks.md Implementation Notes.)

---

## 2026-06-26 — Live-credential diagnostics (task 3.2 attempt)

_A `.env` with `NVIDIA_API_KEY` + `FMP_API_KEY` + `OPENROUTER_KEY` was provided. Before spending on
NIM calibration, each dependency was probed. Result: 3.2 is **blocked on NIM inference auth**, not code._

### NIM (NVIDIA) — BLOCKER
- `GET /v1/models` → **HTTP 200, 121 models** (read access works).
- `POST /v1/chat/completions` (meta/llama-3.1-8b-instruct, gpt-oss-20b, llama-4-maverick, llama-3.3-70b)
  → **HTTP 403 `{"status":403,"title":"Forbidden","detail":"Authorization failed"}`** for ALL models.
- Diagnosis: the provided `nvapi-…` key can **list** models but is **not entitled to run inference** on
  `integrate.api.nvidia.com`. This is a key-level entitlement/rotation issue (the models exist; inference
  auth fails), exactly as anticipated. **Remediation: rotate to / provide a NIM key with inference access
  (build.nvidia.com personal key with credits, or an inference-entitled NGC key).** Until then, every
  NIM-dependent step (3.2 calibration + scoring inside 4.1/4.2) cannot run.

### FMP (ultra plan) — WORKS
- The earlier `HTTP 403` was on the **legacy** `/api/v3/profile` endpoint (deprecated by FMP's API
  migration), NOT an access issue (confirmed by the user; ultra plan).
- recall_guard's `fmp_corpora` targets the **stable** API (`/stable/fmp-articles`,
  `/stable/news/general-latest`); both returned **HTTP 200** with articles.
- **Endpoint behaviour (important for corpus building):** `fmp-articles` IGNORES the `from`/`to` window
  (returns only the latest ~today articles → they bucket as OOS); `news/general-latest` DOES honour the
  window (returned 2023-dated rows for an IS window, 2024-dated for an OOS window). So the date-split IS
  corpus must come from `news/general-latest`.
- **History depth:** `news/general-latest` is **thin before ~2019** (sub-windows 2010–2018 returned 0).
  A small `build_calibration(target=20, cutoff=2023-12-01)` probe filled OOS 20/20 but IS only **8/20**.

### Model + cutoff selection (decision for when NIM inference works)
- recall_guard ships NO packaged cutoffs (`load_cutoffs(path)` needs a file); the cutoff is a task input.
- Because FMP news is dense only ~2019→2026, pick a NIM model whose **real training cutoff sits inside
  that dense window** so BOTH IS (pre-cutoff) and OOS (post-cutoff) fill: a ~mid-2024 cutoff is ideal
  (IS 2019–2024, OOS 2024–2026). A 2023-12 cutoff (llama-3.1/3.3) leaves IS data-thin → likely weak
  calibrator (R1.7 fallback). Plan: once inference works, re-probe **logprobs support** (recall_guard's
  MIA features require them) across candidates and prefer a logprobs-capable model with a ~2024 cutoff
  (e.g. gpt-oss-20b ~2024-06, or llama-4-maverick ~2024-08); fall back to llama-3.1-8b (known-good
  logprobs, cutoff 2023-12) and accept a likely-weak calibrator if the 2024-cutoff models lack logprobs.

### Status
- 3.2 / 4.1 / 4.2 remain BLOCKED — now specifically on **NIM inference authorization** (FMP + OpenRouter
  keys are usable; FMP confirmed, OpenRouter not yet exercised). No code changes needed to unblock.

---

## 2026-06-26 — Task 3.2 result: calibration VALID (NIM key rotated)

_The NIM key was rotated to an inference-enabled key; `/v1/chat/completions` now returns 200. Ran
`scripts/calibrate_nim_scorer.py` (the reproducible 3.2 runner)._

### Model selection (empirical)
- Re-probed logprobs across candidates — ALL return logprobs now. Chose
  **`meta/llama-4-maverick-17b-128e-instruct`** @ cutoff **2024-08-01**: it returns logprobs, emits a
  clean `Direction: 1, Confidence: 0.52` answer (parses under recall_guard's evaluator), and its
  ~2024-08 training cutoff puts both IS (pre-cutoff) and OOS (post-cutoff) inside FMP's dense news
  window. (llama-3.1-8b refused the format; gpt-oss-20b emitted verbose reasoning; both also have a
  2023-12 cutoff → thinner IS.)

### Corpus
- `build_calibration(cutoff=2024-08-01, target=100/side)` → **IS=40, OOS=100**. IS capped at 40 because
  `news/general-latest` has essentially no history before ~2018 (the 2010–2018 sub-windows returned 0);
  the 2018–2024 windows supplied 40. OOS (2024-08 → 2026) filled to 100. Raw corpora are gitignored
  (FMP article text); regenerable via the runner.

### Result — calibrator is VALID (not weak)
- **`holdout_auc = 0.924`, `is_weak = False`** (gate `min_auc=0.6`). Despite the 40/100 imbalance, the
  llama-4-maverick memorization signal cleanly separates pre- vs post-cutoff financial news.
- **Directional smoke** on a sample anonymized macro state (`cpi_yoy_z=0.55, t10y2y_z=-1.59,
  hy_oas_z=0.65`): `parse_ok=True, signal=-1, p_memorized ≈ 0.051`. The directional template parses,
  and the dateless/anonymized macro prompt reads as **low contamination (~0.05)**.
- **Interpretation (the thesis, confirmed):** the detector demonstrably works on real dated text
  (AUC 0.92) yet the anonymized PIT macro prompts score low — so steering applies a gentle
  `(1 − 0.05) ≈ 0.95` confidence discount and the 0.8 exclusion gate does not fire. This is the desired
  low-contamination regime, not a failure. **Steering is enabled (the R1.7 weak fallback is NOT triggered).**
- Persisted: `data/track_a_scores_meta_llama-4-maverick-17b-128e-instruct.json` (metadata header:
  model, cutoff, auc, is_weak, n_is/n_oos, smoke). 4.1 / 4.2 are now unblocked (NIM inference works,
  calibrator valid; OpenRouter still to be exercised by the agent runs).
