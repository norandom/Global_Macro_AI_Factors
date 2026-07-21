# SJM × crowding de-risk overlay — approach and results

*Companion to `notebooks/17_sjm_crowding_derisk.ipynb` (commits `18762b7`, `12c72a5` and successors). Goal: reduce the Factor PIT line's max drawdown at little return cost, with AI used for calibration only and a deterministic function at runtime.*

## 1. The Sparse Jump Model (SJM)

The regime detector is a **statistical jump model** (Shu & Mulvey 2024; Nystrup et al.), ported from the sibling `facdrone` project (`macro_framework/jump_regime.py`). It clusters temporal feature vectors `x_0..x_{T-1}` into `K` states while **penalising state transitions**:

```
min over centroids {θ_k} and states {s_t}:
    Σ_t ½·‖x_t − θ_{s_t}‖²_w  +  λ · Σ_t 1{s_t ≠ s_{t−1}}
```

solved by coordinate descent: fix centroids → exact dynamic-programming (Viterbi) pass over states; fix states → centroids = cluster means; iterate to a fixed point. The jump penalty `λ` buys **regime persistence** — the property a plain threshold or k-means classifier lacks (λ=50 default ⇒ a handful of switches per decade, not per month). The *Sparse* extension (`κ`) re-weights features by between-cluster variance so uninformative features get weight 0; we run the plain JM (`κ=None`).

Deliberate properties kept from the port:

- **Deterministic, no RNG**: quantile-based initialisation on the trend feature; DP tie-breaks prefer the lower state index and prefer *staying*; labels assigned by centroid order on the trend axis — never by fit order. Two identical fits give identical output (unit-tested).
- **Fail-open to `neutral`**: `neutral` is a *confidence guard* (centroids too close on the trend axis, or the latest point assigned with too small a margin), not a learned third cluster.
- **Labels**: `bull` / `bear` / `neutral` (facdrone's `trending`/`mean_reverting`/`neutral`, relabelled; ordering logic identical).

**Features** (`sjm_features`, 3 columns from the world-equity sleeve SWDA.L): EWMA daily log-return (the trend axis, halflife 21), EWMA volatility, EWMA downside volatility.

**Walk-forward fitting is non-negotiable.** `fit_labels_walk_forward` refits the model at every monthly rebalance using **only strictly-prior data** (min 504 obs, else `neutral`). The facdrone incident report (`research/lookahead_sjm_incident.md`) documents how easily a regime model refit on full history becomes a look-ahead vector — and that SJM used as a *return-seeking* tilt was measured **net-harmful** (2024: +2.89% vs +28.84% for the alternative). The defensible use is the one built here: **de-risk only** (caps ≤ 1, never winner-picking), and the PIT unit test bites (appending a future crash must not change a past label).

## 2. Crowding detection

We hold no positioning data, so crowding is proxied in **return space** over the full 112-ETF cross-section (`macro_framework/crowding.py`):

- **Absorption ratio** (Kritzman–Li–Page–Rigobon 2011): share of cross-sectional variance absorbed by the top `n/5` eigenvectors of the trailing 252-day correlation matrix. High AR = variance concentrated in few factors = a crowded, fragile market where shocks propagate broadly.
- **Financial turbulence** (Kritzman–Li 2010): Mahalanobis distance of the day's return vector from its trailing distribution (baseline strictly *before* the scored day).
- **PIT bucketing**: expanding-quantile terciles — a date's bucket uses only data up to that date (full-sample quantiles would be look-ahead).

**Evidence on the Factor PIT drawdown episodes** (AR percentile = PIT percentile of the absorption ratio at episode start; unconditional AR median 0.936, turbulence median 92):

| episode start | trough | AR pctile (PIT) | turbulence |
|---|---|---|---|
| 2020-03-12 (COVID) | −11.0% | **1.00** | **5299** (57× median) |
| 2022-06-13 → 2022-12 (rates) | −12.1% | 0.88–0.89 | 48–245 |
| 2025-11-04 → 2025-11-26 (AI concentration) | −8.3% | 0.79–0.80 | 79–96 |
| 2026-01-30 → 2026-04 | −11.8% | 0.81–0.84 | 95–501 |
| 2026-06-10 | −7.0% | 0.28 | 2 | 

Every major episode except 2026-06 began with the absorption ratio in the **top ~10–20% of its own history** — consistent with the crowding hypothesis. The honest caveat: these are concentration/fragility proxies, not positioning. True crowding measurement needs CFTC COT, short interest, 13F overlap, or ETF flows — the `fmp_etf_holdings` table is an untapped in-house source for holdings-overlap crowding.

## 3. Architecture: AI calibrates, a deterministic function applies

- **One offline LLM call** (NIM `openai/gpt-oss-20b`, temperature 0) sees only **anonymized dev-window statistics** per (regime × crowding bucket) cell — no tickers, no dates, no identifying framing (the recall-guard lesson) — and returns a **cap table** from the menu {1.0, 0.9, 0.8, 0.65, 0.5}, clamped deterministically (monotone non-increasing in crowding; bear ≤ neutral ≤ bull). The reply is **persisted** (`data/sjm_crowding_limits_sjm_crowding_v1.json`) and *replayed* on every re-execution — the artifact is authoritative because the LLM is not perfectly deterministic even at temperature 0 (observed: table drift across identical calls moved dev Calmar by ±0.04).
- **Calibrated table** (source `nim:openai/gpt-oss-20b`): bull 1.0/0.9/0.8, neutral 1.0/0.9/0.8, bear 1.0/0.8/0.5 across crowding buckets 0/1/2.
- **Runtime is pure and deterministic**: at each monthly rebalance, `cap = table[SJM regime][crowding bucket]`; the book holds `cap · FactorPIT + (1−cap) · BIL` until the next rebalance. No AI at runtime.

## 4. The /loop protocol and its verdict

The search reused the shipped `factor_loop.run_loop` engine (one deterministic mutation per iteration, keep-or-revert, auditable ledger):

- **Objective**: shallowest dev-window max drawdown (the stated goal).
- **Constraint**: CAGR cost ≤ **2pp** vs the unhedged Factor PIT (budget approved after the initial 1pp run refused the frontier candidate).
- **Control gate**: the candidate must beat the *dumb* correlation-overlay control (`derisk_cash_pin`, task 3.1) on the same objective — the SJM/crowding machinery has to out-earn simplicity to be adopted.
- **Hygiene**: tuned **only** on the dev window (2019-01 → 2024-06); the holdout (2024-07 → 2026-06) was never consulted by any keep/revert decision. Ledger: `data/factor_loop_ledger_sjm_crowding_v1_2pp.json`.

**Dev-window frontier** (baseline: CAGR 10.73%, maxDD −12.1%, Calmar 0.89; control: maxDD −11.0%, Calmar 0.92):

| candidate | dev maxDD | dev CAGR | cost | verdict |
|---|---|---|---|---|
| seed (AI table, λ=50) | −8.8% | 7.21% | 3.5pp | over budget |
| **λ=20** | **−8.3%** | 7.58% | 3.2pp | over budget |
| turbulence signal | −8.7% | 7.40% | 3.3pp | over budget |
| scale 1.2 (milder caps) | −10.5% | 8.61% | 2.1pp | over budget (by 0.1pp) |
| scale 1.3 | −11.0% | 9.07% | 1.7pp | within budget — **ties the control's −11.0%**, not adopted |

**Verdict: 0 of 11 mutations adopted — the dev trade-off frontier runs through the control point.** Under a 2pp budget, no SJM×crowding configuration *dominates* the simple correlation dampener: candidates within budget don't cut the drawdown more than the control; candidates that cut it decisively (λ=20: −8.3%, a 3.8pp reduction) cost ~3pp of CAGR. The loop refused to adopt a tie. This mirrors the facdrone finding at a different scale: regime machinery must clear the simplest alternative, and mostly it doesn't.

**Full-span reference (2019-01 → 2026-06, seed variant shown for transparency):**

| | Factor PIT | SJM×crowding | Corr control |
|---|---|---|---|
| CAGR | 14.0% | 10.0% | 12.8% |
| Ann. vol | 9.7% | 7.1% | 9.1% |
| Sharpe | 1.43 | 1.39 | 1.39 |
| Max DD | −12.1% | **−9.5%** | −11.8% |
| Calmar | **1.16** | 1.05 | 1.09 |
| CVaR 95 (daily) | −1.4% | **−1.0%** | −1.3% |
| Beta (SPY) | 0.23 | **0.15** | 0.22 |
| Basket t(α) HAC | 0.86 | 1.26 | 0.49 |

## 5. What would change the answer

1. **A bigger budget.** At ~3pp of CAGR, λ=20 delivers a −8.3% dev maxDD (vs −12.1%) and clearly dominates the control. If the mandate prioritises drawdown over return, rerun the loop with a 3pp budget.
2. **A better-calibrated table.** The dominant regime cell (bull × crowded, 736 of ~1380 dev days) carries cap 0.8 — most of the return cost. A second AI calibration round fed the *loop's own frontier evidence* (cap that cell nearer 0.9–1.0, keep bear × crowded at 0.5) is the obvious next mutation class.
3. **Real crowding data.** Holdings-overlap from `fmp_etf_holdings`, short interest, or flows would replace the return-space proxy with actual positioning — the biggest methodological upgrade available.
4. **Denser regime × crowding variation.** Most dev days sit in one cell; a 2-bucket crowding split or vol-scaled continuous caps would give the optimiser more usable variation.

## 6. Reproducibility

- All signals are PIT by construction (walk-forward SJM refits; expanding-quantile buckets; trailing-window AR/turbulence) with biting unit tests (suite: 520 passing).
- The AI table is called once and replayed from the persisted artifact; the loop ledger records every candidate's mutation, metrics, gate outcomes, and decision.
- Artifacts: `data/sjm_crowding_limits_sjm_crowding_v1.json`, `data/factor_loop_ledger_sjm_crowding_v1_2pp.json` (+ CSV mirror), `reports/nb17_sjm_crowding_tearsheet.csv`.
- Rebuild: run `notebooks/17_sjm_crowding_derisk.ipynb` top-to-bottom (needs `DATABASE_URL`; the NIM key is only needed if the limits artifact is absent).
