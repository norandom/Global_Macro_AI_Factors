# SSR verdict rule — review and amendment (2026-07-23)

## What was wrong

Until 2026-07-23 every SSR verdict in this repo came from some copy of

```python
"stably > 0" if abs(ssr) >= 1.96 else "NOT distinguishable from zero under HAC — luck-compatible"
```

introduced without derivation in commit `5ba201c` and later hardened into a spec
acceptance criterion (regime-steered-ai-factors R2.3) and the factor-loop gate
default (`GateConfig.ssr_min = 1.96`).

The rule is a category error. SSR = mean(Z)/σ_HAC(Z) is a **signal-to-noise
effect size**: its denominator is the long-run *standard deviation of the
rolling-Sharpe path*, not a standard error — there is no √n anywhere. Because
overlapping 252-day windows make Z near-integrated, σ_HAC is ≈ 8–12 for every
real series, so SSR ≥ 1.96 demands a mean rolling Sharpe of ~18–23. Nothing can
pass: fifteen years of QQQ scores SSR 0.12 and was branded "luck-compatible"
alongside everything else. A verdict that fails everything measures its own
strictness, not luck.

Secondary defects found in the audit:

- the rule existed in **four independently-worded code copies**
  (`build_tear_sheet.py`, `extend_stream_2026.py` ×2, nb15_2, nb14) plus the
  gate in `skill_metric.py`;
- sidedness was inconsistent (`abs(ssr) >= 1.96` two-sided in the tear sheet,
  `ssr >= 1.96` one-sided in the gate and nb14);
- `abs(ssr)` would brand a significantly **negative** Sharpe "stably > 0";
- the vendored SSR paper (`Papers/the_sharp_stability_ratio_may_2026.pdf`,
  §3.3) prescribes a different procedure entirely and was not followed.

## The amended rule

`macro_framework.ssr.ssr_inference` is now the single verdict authority,
implementing the paper's preferred inference (§3.3.2–3.3.3, Test 1):

- **Moving-block bootstrap on the return series** (Künsch 1989): resample
  contiguous return blocks, block length from the Politis-White (2004)
  automatic procedure (Patton-Politis-White 2009 correction), rebuild the full
  rolling-Sharpe → Newey-West/Andrews → SSR pipeline per replicate. Resampling
  *returns* reconstructs the mechanical overlap autocorrelation inside every
  replicate instead of trusting the truncated Andrews bandwidth on the Z path.
- **One-sided test** H₀: μ_Z ≤ SR* vs H₁: μ_Z > SR*, p = #{SSR_b ≤ 0}/B,
  B = 1000, deterministic seed. Verdict "stably > 0" iff p < 0.05 **and**
  mean(Z) > SR* (the sign condition kills the abs() bug).
- **SSR itself is reported as the effect size**, never thresholded.

The stability gate (`GateConfig.ssr_alpha = 0.05`, `_stability_gate`) and every
tear sheet / luck-vs-skill table now call this one implementation.

### Why not the simpler candidates

- *HAC t = SSR·√n on the Z path*: Monte Carlo size 15–22 % at nominal 5 % —
  the Andrews AR(1) plug-in (ρ clipped at 0.97, L capped at n/4 — both binding)
  truncates the Bartlett sum inside the 252-day mechanical overlap, understating
  σ_HAC ~1.4×. Anti-conservative in the opposite direction of the old rule.
- *t-test on non-overlapping annual Sharpes*: well calibrated (4.8–5.5 %) and a
  reasonable fallback, but lower power (6–15 observations) and it abandons the
  SSR machinery the paper defines.

### Calibration of the adopted rule (Monte Carlo, 250 reps, B=300)

| null / alternative | n | rejection rate |
|---|---|---|
| iid zero-mean returns | 1500 | **5.2 %** |
| iid zero-mean returns | 3900 | **5.6 %** |
| GARCH(1,1) zero-mean (vol clustering) | 1500 | **3.6 %** |
| true annual Sharpe 1.0 (power) | 1500 | 75 % |

Nominal level 5 %: essentially exact under iid, mildly conservative under
realistic volatility clustering.

## What the corrected verdicts say

Same effect sizes, meaningful verdicts (one-sided MBB p in brackets):

| line | SSR | verdict |
|---|---|---|
| Static B&H 16-26 | 0.15 | stably > 0 (0.000) |
| Factor PIT v1 / v2 / ext26 | 0.11–0.13 | stably > 0 (≤ 0.001) |
| Track A / Track A steered / Track B | 0.07–0.13 | stably > 0 (≤ 0.011) |
| SJM×crowding de-risk v2 ext26 | 0.14 | stably > 0 (0.000) |
| **Baseline HRP+momentum (both spans)** | 0.04–0.08 | **luck-compatible (0.08–0.22)** |
| **Recall-premium differential (v1 / ext26)** | 0.11 / 0.02 | **luck-compatible (0.056 / 0.59)** |
| QQQ, SPY, IVV, DIA, IWM, XLF, GLD (15y) | 0.04–0.12 | stably > 0 (≤ 0.016) |
| EFA, XLE, EEM, GDX, FXI, HYG, EWZ, TLT (15y) | 0.00–0.05 | luck-compatible (0.056–0.47) |

> **Read the v1 differential as marginal, not comfortable.** The 2026-07-27 nb14
> re-run moved it from SSR 0.03 at p = 0.22 to **SSR 0.11 at p = 0.056** — it still
> fails to reject at α = 0.05, so "luck-compatible" stands, but it is now a
> near-miss rather than a clear non-result. The ext26 differential (0.02, p = 0.59)
> is unaffected and remains the stronger evidence. Both are single draws from a
> non-deterministic generator; neither should be quoted as a stable constant.

Two things the change does **not** alter:

1. **The no-recall conclusion stands.** The PIT-vs-non-PIT recall premium
   remains statistically indistinguishable from zero under the corrected test —
   the thesis's central honesty claim survives with a calibrated instrument
   behind it instead of an unpassable one.
2. **Passing is not a skill claim.** "Stably > 0" says the trailing-year Sharpe
   ran persistently above zero *in this sample* — long-only beta in a bull
   decade does exactly that (SPY passes too). Skill attribution stays with the
   basket-residual alpha gate; the verdict strings say so explicitly.

## Scope of the amendment

Code: `macro_framework/ssr.py` (new: `politis_white_block_length`,
`SSRInference`, `ssr_inference`), `skill_metric.py` (gate), `factor_loop.py`,
`build_tear_sheet.py`, `extend_stream_2026.py` (both sites), nb14, nb15_2,
appendix D. Specs: regime-steered-ai-factors R2.3 + design/tasks;
pyxll-factor-workbook R§ wording + tasks. Docs: README, workbook
ASSESSMENT/S0/SIMULATIONS, nb16 prose. Artifacts regenerated offline from the
persisted equity curves (a full `extend_stream_2026.py` rerun would trigger
live NIM calls); nb14's embedded outputs refresh on its next API-backed run.
Tests: `tests/test_ssr_inference.py` (new), `test_skill_metric.py`,
`workbook/tests/test_steps.py` updated.
