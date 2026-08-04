# Ensemble prompt consensus — asking N times and keeping the answer that survives

*Companion to `notebooks/appendix_i_factor_dispersion.ipynb` and the measurements in
`data/appendix_i_factor_dispersion/`.*

> **Status: shipped in `recall-guard` 0.3.0; this repo is on it.** The proposal in
> §5 was implemented upstream as `generate_ensemble` / `EnsembleSpec` /
> `EnsembleResult` plus the `recall_guard.core.consensus` primitives, and the §9
> concurrency defect is fixed. This project is pinned to `v0.3.0` and consumes the
> ensemble path via `scripts/run_factor_dispersion_study.py --ensemble`.
>
> **Not yet wired into production.** Every factor run in
> `data/provisional_remediation/factor_runs/` still takes one draw per rebalance.
> Moving the deployed line onto ensembles still needs the evidence/replay decision
> in §7, which is unchanged.
>
> §11 records what the integration itself turned up — two behaviours a caller has
> to get right that are not obvious from the API surface.

## 1. What this proposes

Ask the model the **same prompt N times**, then reduce the N replies to one answer
plus an explicit confidence, instead of taking a single draw and hoping it was
representative. The reduction has to survive the failure modes the data actually
exhibits — which are not the ones a textbook outlier filter expects.

The entry point is an optional per-prompt flag on the `recall_guard` client, so a
caller can opt in where it matters (a rebalance decision) and stay on the cheap
single-draw path everywhere else.

## 2. Why this is worth building — the measurement

Appendix I ran the identical anonymized PIT loadings prompt **1000 times** at one
rebalance date (2020-03-02, the COVID onset), at the production setting
(`temperature=0`, `max_tokens=2048`, `openai/gpt-oss-20b`). Of those, 977 parsed to a
full five-axis vector.

| observation | value |
|---|---|
| distinct loading vectors | **652 of 977** |
| draws implying a defensive posture | **98.6%** (random answerer: 50.0%) |
| unparsed replies | 23 (2.3%) |

The headline is the tension between those two rows. **The model almost never repeats
itself exactly, yet almost always reaches the same decision.** A single draw is
therefore a poor estimate of the parameter vector and a good estimate of the
decision. An ensemble is how you get both.

Temperature is 0 in production. The dispersion above is *not* sampling temperature —
it is nondeterminism in the serving stack. Turning temperature down is not available
as a mitigation because it is already down.

## 3. Dispersion is not uniform across axes

| axis | median | sd | IQR | sign agreement |
|---|---|---|---|---|
| inflation | +0.80 | 0.033 | 0.05 | 100.0% |
| risk_appetite | −0.70 | 0.170 | 0.11 | 98.7% |
| credit_stress | +0.60 | 0.293 | 0.47 | 97.6% |
| growth | −0.60 | 0.370 | 0.30 | 87.4% |
| policy | +0.50 | 0.660 | 1.30 | **63.3%** |

Two axes are effectively pinned (`inflation` never moves more than 0.15 from its
median across a thousand draws, and half the draws sit within 0.05 of it). One axis —
`policy` — is not converged at all. Any aggregator that
treats these five the same is discarding the most useful thing the ensemble knows,
which is *which parts of its own answer the model is unsure about*.

## 4. Why the obvious outlier filters are wrong here

This is the part that motivates a purpose-built design rather than
`scipy.stats.trim_mean` and a modified z-score.

**4.1 The MAD rule divides by zero on the tightest axis.** The standard robust
outlier test is the modified z-score, `0.6745·|x − median| / MAD`. Measured MADs:

| axis | MAD | modified-z rule flags | Tukey 1.5·IQR flags |
|---|---|---|---|
| inflation | **0.000** | undefined (÷0) | 1.4% |
| growth | 0.200 | 0.1% | **12.4%** |
| credit_stress | 0.200 | 2.0% | 2.4% |
| policy | 0.200 | **29.2%** | 0.0% |
| risk_appetite | 0.100 | 1.3% | 5.6% |

The model emits on a coarse 0.1 grid and concentrates hard, so the median absolute
deviation collapses to exactly zero whenever more than half the draws share a value.
On `inflation` the canonical robust rule is undefined. Any implementation needs a
grid-aware floor on the scale estimate, not the textbook formula.

**4.2 On the one axis that genuinely disagrees, the MAD rule deletes the
disagreement.** It flags **29.2%** of `policy` draws as outliers. That is not noise
rejection; a third of the distribution is being thrown away precisely where the
ensemble is carrying real information. Tukey's fence has the mirror-image failure —
0.0% on `policy`, but 12.4% on `growth`.

**4.3 `policy` is bimodal, and the mean lands in the trough.** The full observed
distribution (all 977 parsed draws):

```
-1.0 ████ 44          +0.0 ▏ 6
-0.9 ████ 44          +0.1 ▏ 5     ← the mean, +0.142, rounds into this bin
-0.8 █████ 50         +0.2 ▏ 8
-0.7 ███████ 73       +0.3 ███ 34
-0.6 ███████ 74       +0.4 ██ 27
-0.5 ████ 44          +0.5 █████████ 89
-0.4 ▏ 9              +0.6 ███████████████ 156
-0.3 ▏ 8              +0.7 ████████████████████ 205
-0.2 ▏ 6              +0.8 ████ 47
-0.1 ▏ 1              +0.9 ██ 26
                      +1.0 ██ 21
```

Two clusters — **36.1%** of mass below zero, **63.3%** above — separated by a trough
holding only **4.4%** (43 draws) across the whole `[−0.4, +0.2]` band. The model is
not producing a noisy estimate around one value; it is choosing between two
incompatible readings of the policy axis (tightening vs. easing).

The arithmetic mean is **+0.142**, which falls in the `+0.1` bin — a value the model
emitted **5 times in 977 attempts (0.5%)**. Averaging here does not denoise; it
returns a near-abstention that misrepresents a genuine 36/63 split as a weak positive
reading.

An ensemble feature that silently returns +0.142 is worse than a single draw, because
it launders a real disagreement into false precision.

**4.4 Aggregators disagree materially.** Same 977 draws, different reductions,
carried through to the defensive spread `(gold+cash) − (equity+tech)`:

| reduction | vector | spread |
|---|---|---|
| mean | [0.823, −0.504, 0.614, **0.142**, −0.662] | +6.77 |
| median | [0.80, −0.60, 0.60, 0.50, −0.70] | +7.68 |
| 10% trimmed mean | [0.822, −0.561, 0.636, 0.185, −0.682] | +7.13 |
| modal (0.1 grid) | [0.80, −0.70, 0.40, 0.70, −0.70] | +7.76 |
| median *draw* (decision space) | [0.80, −0.70, 0.40, 0.50, −0.60] | +7.17 |
| *single recorded production draw* | [0.80, 0.40, 0.90, 0.70, −0.80] | +6.11 |

The spread ranges +6.77 to +7.76 depending on choice of estimator — and the single
draw that production actually used sits **below all of them**, at +6.11. Note its
`growth` of **+0.40** against an ensemble median of −0.60: the deployed run took a
draw whose growth sign was the minority one, and no downstream artifact records that.

## 5. The design, as shipped

Implemented in `recall-guard` 0.3.0. The sketch below is kept because it states the
intent; the shipped names differ in detail, and where they do, the shipped API wins.

### 5.1 The flag

```python
@dataclass(frozen=True)
class EnsembleSpec:
    draws: int = 128                  # upper bound, not a fixed cost (see 5.4)
    max_workers: int = 32
    min_draws: int = 24               # never stop before this
    agreement_target: float = 0.95    # stop when Wilson lower bound clears this
    grid: float = 0.1                 # the emission grid; drives the scale floor
    multimodal_action: str = "flag"   # "flag" | "raise" | "collapse"

class NvidiaLM:
    def generate(self, prompt, temperature=0.0, max_tokens=512,
                 ensemble: EnsembleSpec | None = None) -> CompletionResult | EnsembleResult: ...
```

`ensemble=None` preserves today's behaviour byte-for-byte. This matters for the
project's replay contract: existing runs must remain reproducible.

### 5.2 Reduce in two spaces, not one

The ensemble must answer two different questions, and one statistic cannot do both:

- **Parameter space** — "what is the answer?" Reduce per-component with a
  grid-aware trimmed mean, then snap to the emission grid. Trimming (not the MAD
  z-score) because §4.1 shows the scale estimate is unusable on concentrated axes.
- **Decision space** — "how much should you trust it?" Push every individual draw
  through the caller's downstream map and measure agreement on the *outcome*. For
  the factor line that map is linear (`tilt = loadings · REGIME_ASSET_EXPOSURE`), so
  the caller supplies it as a projection function.

Reporting only the parameter-space answer would have hidden the actual Appendix I
result. The loadings disagree constantly (652 distinct vectors); the decision agrees
98.6% of the time. **The confidence lives in decision space.**

```python
@dataclass(frozen=True)
class EnsembleResult:
    consensus: CompletionResult      # the representative draw, not a synthetic one
    location: dict[str, float]       # per-component robust location, grid-snapped
    agreement: float                 # share of draws agreeing on the decision
    agreement_ci: tuple[float, float]  # Wilson interval
    multimodal: tuple[str, ...]      # components that failed the unimodality test
    n_requested: int
    n_parsed: int
    draws: tuple[CompletionResult, ...]  # retained for the evidence table
```

`consensus` is deliberately **an actual draw** — the one nearest the decision-space
median — never a synthesized vector. §4.3 is the reason: a synthesized vector can be
a point the model never considered.

### 5.3 Refuse to collapse a multimodal component

Detect the §4.3 pattern — a dip test, or simply two clusters each holding >15% of
mass separated by a trough of several grid steps holding an order of magnitude less
(the measured `policy` split is 36% / 4% / 63%) — and **do not average across it**.
Default `multimodal_action="flag"` returns the modal cluster as the
location, names the component in `multimodal`, and lets the caller decide. For a
rebalance the sensible policy is to attenuate exposure on a flagged component — the
same instinct the recall guard already applies to `p_memorized`.

Silently averaging is the one behaviour that must not be the default.

### 5.4 Stop early — 1000 draws is not the operating point

Bootstrap over the stored draws (400 resamples per row):

| n | trimmed-mean spread, 95% band | width | Wilson lower bound on agreement |
|---|---|---|---|
| 10 | +7.06 [+5.85, +8.18] | 2.33 | 0.596 |
| 25 | +7.05 [+6.18, +7.82] | 1.64 | 0.805 |
| 50 | +7.11 [+6.57, +7.61] | 1.04 | 0.895 |
| 100 | +7.13 [+6.73, +7.52] | 0.79 | 0.930 |
| 200 | +7.13 [+6.84, +7.39] | 0.55 | 0.957 |
| 977 | +7.13 [+7.00, +7.25] | 0.26 | 0.975 |

The point estimate is stable by **n≈50**; everything after that buys interval width
at the usual `1/√n` rate. Going from 100 to 977 draws costs 10× for a 3× narrower
band. **Sequential stopping at an agreement target, with `min_draws≈24` and a cap
around 128, captures nearly all the value.** 1000 was the right number for a
one-off study of the distribution's shape; it is the wrong number for a production
loop.

## 6. The recall guard needs this more than the loadings do

`p_memorized` was scored **100 times on the identical prompt**:

> mean **0.211**, sd **0.218**, range **0.000 – 0.760**

The guard is noisier in relative terms than the signal it is guarding. It directly
scales deployed exposure (`guarded_tilt = raw_tilt · (1 − p_memorized)`), so this
dispersion passes straight into position sizing. Bootstrapped precision of the
*median* guard score:

| n | median p_memorized | 95% band | width |
|---|---|---|---|
| 1 | 0.206 | [0.000, 0.662] | **0.662** |
| 10 | 0.158 | [0.007, 0.384] | 0.378 |
| 25 | 0.141 | [0.013, 0.323] | 0.310 |
| 100 | 0.136 | [0.061, 0.268] | 0.208 |

**A single guard scoring is close to uninformative** — its 95% band spans two thirds
of the unit interval. The canonical run recorded `p_memorized = 0.0` for this date;
the ensemble median is ≈0.14 and the mean 0.211. Those imply materially different
attenuation (100% of the tilt vs. ~79–86%).

If only one consumer gets the ensemble treatment, it should be the guard, not the
loadings. This also argues against using a raw `p_memorized` as a hard gate anywhere
until it is ensembled.

## 7. Interaction with the replay and audit contract

The project's integrity model (`factor_run.v1`, dated evidence, `replay_audit`)
assumes **one persisted reply per `(variant, rebalance_date)`**. An ensemble breaks
that assumption N-fold and needs an explicit decision before implementation:

- **Evidence volume.** A full run is 126 dates × 2 variants = 252 prompts. At 128
  draws that is 32,256 replies per run versus 252 today. The existing evidence table
  stores full reply text; it would grow by roughly the same factor.
- **Proposed shape.** Keep the evidence table keyed as it is now, storing the
  `consensus` draw plus the ensemble's summary statistics and a `draws_sha256` over
  the canonicalized draw set. Persist the full draw set as a **separate, per-run
  ensemble artifact** referenced by that hash. This keeps `replay_audit` working
  unchanged on the existing key while making the ensemble fully auditable.
- **Replay determinism.** Replaying an ensemble run must re-derive the same consensus
  from the stored draws without re-querying — so the reduction has to be a pure
  function of the persisted draw set, with the tie-breaking rule pinned (stable sort,
  explicit seed for any resampling).

## 8. Cost

At the measured throughput (260 draws/min, §9) a 128-draw ensemble is ≈30 s per
prompt. A full 252-prompt run is ≈2 hours of wall clock against ≈1 minute today.
Sequential stopping should cut that substantially on the many prompts that converge
early. This is affordable for a periodic canonical rebuild; it is not affordable as
a default for every exploratory run, which is why the flag is opt-in.

## 9. The client used to serialise concurrent calls — fixed in 0.3.0

**A real defect in `recall_guard` 0.2.0, found while building Appendix I. It gated
the whole feature and is now fixed upstream.**

`recall_guard/core/nvidia_lm.py` holds a per-instance pacing lock across the entire
HTTP request:

```python
def _paced_post() -> requests.Response:
    with self._pace_lock:                    # ← held across the network round-trip
        if self.min_call_interval_s > 0 and self._last_call_t is not None:
            ...
        response = requests.post(self.api_base, headers=headers,
                                 json=payload, timeout=self.timeout_s)
```

The lock exists to honour `min_call_interval_s`, but because it wraps the blocking
POST rather than just the rate-limiter bookkeeping, **every concurrent call through
one client serialises**. `generate_many(..., max_workers=8)` and
`FactorScorer.score_many(..., max_workers=8)` therefore run effectively sequentially:
the worker count is silently inert.

Measured on the Appendix I workload (32 workers, same prompt):

| configuration | throughput | 1000 draws |
|---|---|---|
| one shared client (as shipped) | ~0.7 draws/min | ~25 hours |
| one client per worker thread | **~260 draws/min** | **<4 minutes** |

A ~370× difference, and the reason a 1000-draw study is a coffee break rather than
an overnight job.

**Fixed in 0.3.0** by `NvidiaLM._reserve_call_slot()`, which holds the lock only for
the send-slot bookkeeping and returns the wait, so the POST happens outside it —
the shape proposed here:

```python
with self._pace_lock:
    wait = self._compute_wait()
    self._last_call_t = time.monotonic() + max(wait, 0.0)
if wait > 0:
    time.sleep(wait)
response = requests.post(...)          # outside the lock
```

This preserves the call-interval guarantee while allowing genuine concurrency.
Verified after upgrading: 16 calls at 16 workers through **one shared client** ran
10.9× faster than the sum of their latencies (88 calls/min), where 0.2.0 would have
serialised them.

The thread-local workarounds in the two runner scripts have been removed accordingly.
**The other callers get the fix for free** — `FactorScorer.score_many` in
`macro_framework/factor_scoring.py` and `_generate_big` / `_score_with_retry` in
`scripts/extend_stream_2026.py` now actually honour their `max_workers` arguments,
which were inert under 0.2.0.

## 10. What this does not solve

- **Consistency is not correctness.** An ensemble makes a repeatable answer legible;
  it cannot make a wrong answer right. Appendix I's result is meaningful only because
  the posture the draws agree on is also the one the subsequent drawdown rewarded.
- **It does not address contamination.** 2020 sits inside the model's training
  window. Ensembling sharpens the estimate of what the model says, not whether the
  model is reasoning or recalling. That remains the guard's job — see §6 for why the
  guard's own estimate needs this first.
- **One date is not a distribution over regimes.** Every number here comes from a
  single rebalance at a crisis onset, chosen precisely because it is the hard case.
  Whether `policy` is bimodal in calm regimes too is unmeasured, and worth checking
  before the multimodality policy in §5.3 is tuned.
- **The 2.3% unparsed rate is unmodelled.** Today a parse failure falls back to the
  base allocation. Under an ensemble, parse failures should be reported as a
  first-class outcome (`n_parsed` vs `n_requested`), since a rising failure rate is
  itself a signal about prompt or model health.

## 11. Integration notes (`recall-guard` 0.3.0)

Two behaviours cost a debugging cycle each when wiring
`scripts/run_factor_dispersion_study.py --ensemble`. Neither is a defect; both are
easy to get silently wrong.

**11.1 The ensemble draws with the client's defaults, not production's.**
`generate_ensemble` issues each draw as `lm.generate(prompt)` with no per-call
overrides, so a draw inherits `max_tokens=512` — while the deployed factor stream
generates at 2048. For a reasoning model that gap is decisive: the chain of thought
consumes the budget and the reply is truncated before the JSON object.

Measured on the same prompt and 64 draws:

| draw budget | parsed | failures |
|---|---|---|
| `max_tokens=512` (client default) | 31/64 (48%) | 28 projection, 5 http |
| `max_tokens=2048` (production) | **61/64 (95%)** | 3 total (breakdown not retained) |

The 95% figure is consistent with the raw study's 977/1000 (97.7%) at the same
budget, so the parse collapse is attributable to the token budget rather than to
the ensemble path.

An ensemble that silently samples under different generation settings than production
is not measuring the production decision. This repo passes a `ProductionDefaultsLM`
subclass whose `generate` defaults match the deployed stream. A `max_tokens` field on
`EnsembleSpec` would remove the need for the subclass and make the mismatch
impossible to introduce by accident — worth raising upstream.

**11.2 A flagged component is dropped from `location`, and the caller must decide
what that means.** With `MultimodalAction.FLAG`, `_reduce_components` skips a
separated component entirely — "refusing to reduce it to one location" — so
`result.location` simply has no key for it. Indexing it raises `KeyError`, which is
the API doing its job: it will not hand back a centre the draws do not support.

This project treats a missing axis as **abstention**: it contributes nothing to the
tilt, so the ensemble's refusal propagates into a *smaller* exposure rather than a
fabricated one, and the abstained axes are recorded in the consensus JSON.

**11.3 The separated-cluster test needs enough draws to fire.** On the full 977-draw
set the library flags exactly the axis this document identified by hand:

```
policy  separated=True  lower_mass=0.360  upper_mass=0.627  trough_mass=0.012  gap=(-0.2, +0.2)
```

— matching the measured 36.1% / 63.3% split, from an independent implementation. But
across two 64-draw runs the flag was unstable: one run flagged `policy` and abstained,
the next did not and returned a *sign-flipped* location for it (`−0.5`, against `+0.5`
from the 977-draw reduction).

So at n≈64 the bimodality is real but not reliably detected, and the reduction can
return a confident-looking value on the one axis that has no single answer. The §5.4
sizing table was derived for the precision of the *decision*; detecting a split
component evidently needs more draws than estimating a location does. Until that
threshold is measured, `min_cluster_draws` should be treated as unvalidated for small
ensembles, and small-n consensus on a known-bimodal axis should not be trusted.
