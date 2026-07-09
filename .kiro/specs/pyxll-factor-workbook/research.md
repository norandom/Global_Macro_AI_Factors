# Research Log — pyxll-factor-workbook

## 2026-07-09 — Gap analysis (requirements → existing codebase)

_Parallel research: (a) codebase reuse surface + data-v1 asset schemas (read live with pandas),
(b) PyXLL external research (pyxll.com docs). Plus a live stability probe answering the user's
question "does PIT have better stability?" — see §5._

### 1. Current state

**Data source (R1) — fully in place.** The `data-v1` GitHub Release exists with 27 assets;
URL pattern `https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v1/<asset>`.
The repo is **PUBLIC** (`gh repo view --json isPrivate` → false): plain unauthenticated GET works
today. R1.3's token support is therefore a *dormant* requirement — needed only if the repo goes
private (then: API asset endpoint + `Authorization: Bearer` + `Accept: application/octet-stream`;
the browser URL stops working). Naming deltas vs the working tree: `norecall_screen/results.json`
→ asset `norecall_screen_results.json`; the evidence dirs and the calibrator ship as `.tar.gz`
bundles that the loader must unpack.

**Schema contract (R2–R6) — captured in full.** Every asset was read and its exact
columns/dtypes/index/rows recorded (full table in the workflow transcript; load-bearing rows):
- Monthly streams are 72 rows: `factor_loadings_v1/v2` (date-indexed, parse_ok + 5 float axes;
  v2 adds `prompt_version`), `factor_scores_v1/v2` (p_memorized, fail_reason),
  `factor_contrast_v1` (pit_p, nonpit_p, delta), `naive_directional_eval_*` (72 rows:
  date, prompt, reply, predicted/realized direction, confidence, correct).
- `factor_views_v1`: 284 rows (date, asset, raw_tilt, p_memorized, guarded_tilt, conviction) —
  the guard re-derivation surface (R4.3).
- Daily: `factor_targets_*` 2828 rows (4 ETF weight columns), `factor_equity_*` 2717 rows
  (`value`), for both lines.
- `factor_luck_vs_skill_v1`: 3 rows (line-indexed) with sharpe/ssr/nw_* columns mapping 1:1 onto
  `ssr.SSRResult` fields (sharpe←sr_full, nw_sigma_hac←sigma_hac, nw_bandwidth_L←L_hac).
- Decision logs (JSON): meta + per-date dicts (p_memorized/parse_ok/steered/conviction/loadings/
  views) with "YYYY-MM-DD HH:MM:SS" string keys — v1/v2/nonpit metas differ slightly (v2 lacks
  cutoff/holdout fields; nonpit carries `variant`).
- Screen evidence (per model dir in the tarball): `evidence.parquet` 521 rows — arm,
  prompt, reply, n_tokens, included, dropped_reason, raw_* and std_* MIA feature columns —
  + `baseline.json` (feature means/stds) + `summary.json` (one results row). Present for 3 of 4
  recalling candidates; **maverick has no evidence dir** (R2.4's "raw evidence pending" case is
  real, not hypothetical).

**Reusable re-derivation code (R7.1/R7.2):**
- `macro_framework/ssr.py` — pure numpy/pandas, ~80 lines; `compute_ssr(returns)` re-derives the
  luck-vs-skill rows from equity `pct_change()` exactly.
- `macro_framework/evaluation.py` — targets-only helpers reusable from parquet
  (`turnover_stats`, `anticipation_lead_time`, `view_stability`); **`head_to_head_report` /
  `crisis_analytics` are NOT reusable** (they require live `vbt.Portfolio` objects) — the workbook
  must recompute total_return/sharpe/sortino/calmar/max_dd from the equity series itself (simple
  pandas; the crisis-window logic is ~8 lines to copy).
- `macro_framework/factor_scoring.py` — `factor_stability`, `certification_stats`,
  `recall_guarded_adjust`, `_paired_cohens_d` are offline-pure, **but the module imports
  recall_guard + sklearn at module level** — importing it drags the full inference stack into the
  workbook env. `certification_stats` is the one piece worth vendoring verbatim (deterministic
  seed=0 reproduction of the released AUC/CI/perm-p from `std_*` evidence columns; needs sklearn).
  Wilson CI exists only inside nb13 (~4 lines to reimplement); guarded=raw·(1−p) is 1 line;
  paired Cohen's d is 2 lines.

**Packaging reality.** `macro_framework` is a non-installable top-level dir (no `[build-system]`);
everything runs `uv run` from repo root; the env carries vectorbt/numba/llvmlite/riskfolio/
jupyter — hostile to a lean Excel deploy. `factor_scoring`/`ssr` are not in the package
`__init__` (which eagerly imports heavy modules), so targeted imports are possible but still
require repo-root sys.path.

### 2. PyXLL feasibility (R7.4/R7.6)

- **The Linux problem is solved by PyXLL itself**: `pip install pyxll` on any platform installs a
  stub module whose `@xl_func`/`@xl_macro` are pass-through decorators. The entire core AND the
  Excel-facing wrapper module import and pytest-run on Linux with zero mocking. Only the real
  add-in (Windows Excel) executes them live.
- Current PyXLL 5.12.x; Python ≤3.12 supported (repo pins 3.12 — compatible). Commercial
  per-user annual license, 30-day trial (R7.6's documented prerequisite).
- Canonical split: pure core package (fetch/parse/re-derive) + one thin `@xl_func` wrapper module
  listed in `pyxll.cfg` (`modules=`, `pythonpath=`).
- Remote loads: plain UDFs block Excel's UI → use `async def @xl_func` (PyXLL runs its own asyncio
  loop) or RTD for long loads; big DataFrames stay in PyXLL's object cache (return a handle,
  expand to grid on demand) — dumping the 2828-row daily tables into cells is fine, the 521-row
  evidence tables too; avoid spilling everything at once.
- Tokens (dormant, R1.3): `keyring` → Windows Credential Manager (maps to Secret Service on
  Linux, so the same code tests in CI); never tokens in cell formulas or saved workbook content.

### 3. Requirements feasibility summary

| Req | Feasibility | Notes |
|---|---|---|
| R1 data access | HIGH | Public release; URL pattern verified; tar.gz unpack needed; token path dormant but designable via keyring |
| R2 certification + evidence | HIGH | Evidence schema captured; `certification_stats` vendorable for re-derivation; maverick gap = the R2.4 case |
| R3 coin-flip | HIGH | 72-row parquet; Wilson CI reimplemented (~4 lines) |
| R4 factor development | HIGH | All tables exist; guard math 1 line; stability re-derivable |
| R5 two lines | HIGH | Equity/targets/logs for both lines; head-to-head metrics must be recomputed from equity (no vbt) |
| R6 luck-vs-skill | HIGH | `compute_ssr` vendorable ~80 lines pure numpy/pandas; contrast parquet complete |
| R7 integrity/testability | HIGH | PyXLL stub → full Linux testability; framing rules are content, not tech |

### 4. Implementation approach options

**Option A — extend `macro_framework` in-repo.** New `macro_framework/workbook/` package imported
by the PyXLL wrapper. ✅ zero code duplication (ssr, evaluation helpers importable). ❌ the Excel
machine must carry the full env (vectorbt/numba/recall_guard...) and repo-root sys.path; violates
lean-deploy; factor_scoring import drags the inference stack — collides with R7.1's "no inference"
spirit and makes the Windows install fragile.

**Option B — new lean package in-repo (recommended direction).** New top-level dir (e.g.
`workbook/` with its own `pyproject.toml`, deps: pandas, pyarrow, requests, keyring, pyxll-stub;
optional sklearn extra for the certification re-derivation). Vendors `compute_ssr` (~80 pure
lines) + the 3 trivial formulas; hard-codes the schema contract as typed loaders with checks
(which doubles as the R7.2 discrepancy detector). ✅ lean Windows install (pip install one
package), Linux-testable end-to-end, additive to repo. ❌ ~90 lines vendored from ssr.py (dual
maintenance — acceptable: ssr.py is stable/complete).

**Option C — hybrid.** Lean package that *tries* `from macro_framework.ssr import compute_ssr`
and falls back to vendored copy. ❌ two code paths to test; the fallback IS the primary on any
Excel machine; complexity without benefit. Rejected unless design finds a reason.

**Research needed at design time:** (a) whether re-deriving certification stats needs sklearn on
the Excel machine or should be a Linux-side "verification report" shipped as a workbook sheet;
(b) RTD vs async-UDF choice per sheet; (c) how the "step-by-step" navigation is realized (one
sheet per step + a ribbon/menu vs pure worksheet functions) — presentation decision, Windows
validation manual.

### 5. Analytical probe — does PIT have better stability? (live-computed 2026-07-09)

Computed `factor_stability` on the non-PIT diagnostic loadings (not previously persisted) vs the
persisted PIT v1 stats — same model, same 72 dates, prompts differ only by the identifying
additions (R7.6-controlled, so the delta is attributable to PIT discipline):

| measure | PIT v1 | non-PIT | winner |
|---|---|---|---|
| mean loading dispersion (std) | **0.544** | 0.637 | PIT |
| mean month-to-month change (mac) | **0.388** | 0.510 | PIT |

PIT is more stable on **all 5 axes on both measures** (extremes: credit_stress mac 0.516 vs
0.823; risk_appetite mac 0.294 vs 0.485). Interpretation: identifying context makes the loadings
jumpier — recall injects date-specific pattern-matching noise; the anonymized numbers-only form
yields smoother state-driven inference. SSR stability is a statistical tie (0.124 vs 0.130, both
≪1.96, both luck-compatible). Caveats: n=72, no formal test on the stability delta (bootstrap =
natural workbook addition); prompt version also moves the metric (v2 mac 0.406) but the
PIT-vs-non-PIT gap is ~3× the v1-vs-v2 gap. → The workbook's S5 sheet should include this
loading-stability comparison (computable from released parquets alone).

---

## 2026-07-09 — Design synthesis (decisions on top of the gap analysis)

- **Option B refined → "one source tree, two install surfaces"**: the package lives once in-repo
  (`workbook/factor_workbook/`), collected by the repo's root `uv run pytest` (root env is a
  superset, satisfying R7.4's "existing test workflow"), while `workbook/pyproject.toml` defines
  the LEAN install (pandas, pyarrow, requests, pyxll-stub; `keyring` + `scikit-learn` optional
  extras) for the Windows Excel machine — no vectorbt/numba/recall_guard ever reaches Excel.
- **Presentation is generated, not hand-built**: a Linux-runnable `build_workbook.py` uses
  openpyxl to emit the five-sheet workbook skeleton (labels, framing text, `=FW_*()` formulas);
  PyXLL evaluates those functions live in Excel. This moves most of the "Excel-only" layer back
  into Linux-testable code; the manual Windows checklist shrinks to type-conversion + ribbon.
- **Vendoring over importing**: `compute_ssr` (~80 pure lines) is vendored verbatim with a
  provenance header (source path + commit) instead of importing `macro_framework` (drags heavy
  deps, needs repo-root sys.path). Wilson CI (~4 lines), guarded=raw·(1−p) (1 line), paired
  Cohen's d (2 lines) are reimplemented in `rederive.py`. `certification_stats` is vendored too
  but behind the `sklearn` extra — R2.6 only REQUIRES class counts + feature summary stats, so
  the full AUC/CI/p re-derivation is an optional deepening, not a hard dependency.
- **The schema contract IS the discrepancy detector's foundation**: `contract.py` encodes the
  captured per-asset schema (columns/dtypes/index/row-counts) as typed loaders that fail fast
  with asset+column-specific messages; `verify.py` compares re-derived vs published figures with
  a display-precision tolerance and returns flag rows (R7.2) — never silently preferring either.
- **Token path (dormant)**: `keyring` (Credential Manager on Windows / Secret Service on Linux)
  → `GITHUB_TOKEN` env fallback; only consulted when an unauthenticated GET fails with 404/403 on
  a private repo. Never accepted as a UDF argument (tokens must not live in cells; R1.3).
- **Async UDF over RTD**: loads are `async def @xl_func` returning object-cache handles (Excel UI
  stays live); per-table expansion functions spill DataFrames on demand. RTD deferred — nothing
  in this feature streams.
