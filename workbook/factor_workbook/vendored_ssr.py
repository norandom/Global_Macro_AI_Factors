# Vendored VERBATIM from macro_framework/ssr.py (R6.2).
# Source: /home/mc/projects/Global_Macro_AI_Factors/macro_framework/ssr.py
#         (repo-relative: macro_framework/ssr.py)
# Do not edit by hand — re-sync from the source module whenever it changes.
# "Verbatim" is enforced, not asserted: in the root environment
# workbook/tests/test_parity_root_env.py compares this file byte for byte
# against the source and runs both implementations on identical inputs.
"""Sharpe Stability Ratio — Bajo Traver & Rodríguez Domínguez (2026).

Z_t     = rolling Sharpe (window τ) of excess returns, annualized
SSR     = (mean(Z) - SR*) / sigma_HAC(Z)
sigma_HAC uses Newey-West Bartlett kernel with Andrews (1991) automatic bandwidth.

Inference (the paper's §3.3.2-3.3.3, Test 1): SSR is an EFFECT SIZE, not a test
statistic — its denominator carries no sqrt(n), so comparing SSR itself to a normal
critical value (the repo's pre-2026-07 rule, |SSR| >= 1.96) demands a mean rolling
Sharpe of ~2·sigma_HAC ≈ 18-23 and brands everything, 15 years of SPY included,
"luck-compatible". The paper's preferred procedure is a moving-block bootstrap on
the RETURN series (Künsch 1989; Politis-White 2004 automatic block length):
resample contiguous return blocks, recompute the full rolling-Sharpe → SSR pipeline
per replicate — which reconstructs the mechanical overlap autocorrelation inside
each replicate instead of trusting the truncated Andrews bandwidth — and report the
one-sided p-value  p = #{SSR_b <= 0} / B  for H0: mu_Z <= SR*.  ``ssr_inference``
implements exactly that and is the single verdict authority for the repo
(tear sheets, luck-vs-skill tables, the factor-loop stability gate).
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def rolling_sharpe(returns: pd.Series, window: int = TRADING_DAYS) -> pd.Series:
    mu = returns.rolling(window).mean()
    sigma = returns.rolling(window).std(ddof=1)
    sr = (mu / sigma) * np.sqrt(TRADING_DAYS)
    return sr.dropna()


def andrews_bandwidth(z: np.ndarray) -> int:
    """Andrews (1991) data-dependent bandwidth for Bartlett kernel (AR(1) plug-in)."""
    n = len(z)
    if n < 4:
        return 1
    zc = z - z.mean()
    gamma0 = float((zc * zc).mean())
    if gamma0 <= 0:
        return 1
    gamma1 = float((zc[1:] * zc[:-1]).mean())
    rho = float(np.clip(gamma1 / gamma0, -0.97, 0.97))
    alpha = (4.0 * rho**2) / ((1.0 - rho) ** 2 * (1.0 + rho) ** 2)
    L = 1.1447 * (alpha * n) ** (1.0 / 3.0)
    return max(1, min(int(np.floor(L)), n // 4))


def newey_west_var(z: np.ndarray, L: int | None = None) -> tuple[float, int]:
    """Newey-West HAC long-run variance with Bartlett kernel."""
    z = np.asarray(z, dtype=float)
    if L is None:
        L = andrews_bandwidth(z)
    zc = z - z.mean()
    total = float((zc * zc).mean())
    for k in range(1, L + 1):
        gk = float((zc[k:] * zc[:-k]).mean())
        total += 2.0 * (1.0 - k / (L + 1)) * gk
    return max(total, 0.0), L


@dataclass(frozen=True)
class SSRResult:
    n_obs: int
    n_rolling: int
    sr_full: float
    mean_rolling_sr: float
    sigma_hac: float
    L_hac: int
    ssr: float


def compute_ssr(returns: pd.Series, window: int = TRADING_DAYS, sr_star: float = 0.0) -> SSRResult:
    """Point estimate ONLY — the SSR effect size and its inputs, no verdict.

    SSR measures, inference decides: sigma_HAC is the long-run volatility of the
    rolling-Sharpe PATH, not a standard error (no sqrt(n) anywhere), so this
    number has no critical value and must never be thresholded or ranked on.
    Any stable/luck-compatible claim comes from ``ssr_inference``'s MBB p-value.
    """
    r = returns.dropna()
    rolling = rolling_sharpe(r, window=window)
    if len(rolling) < 10:
        return SSRResult(
            n_obs=int(len(r)),
            n_rolling=int(len(rolling)),
            sr_full=np.nan,
            mean_rolling_sr=np.nan,
            sigma_hac=np.nan,
            L_hac=0,
            ssr=np.nan,
        )
    z = rolling.to_numpy()
    z_bar = float(z.mean())
    sigma2, L = newey_west_var(z)
    sigma_hac = float(np.sqrt(sigma2)) if sigma2 > 0 else np.nan
    ssr = (z_bar - sr_star) / sigma_hac if sigma_hac and sigma_hac > 0 else np.nan
    sr_full = float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return SSRResult(
        n_obs=int(len(r)),
        n_rolling=int(len(rolling)),
        sr_full=sr_full,
        mean_rolling_sr=z_bar,
        sigma_hac=sigma_hac,
        L_hac=int(L),
        ssr=float(ssr) if np.isfinite(ssr) else np.nan,
    )


# --- MBB inference (paper §3.3.2-3.3.3, Test 1) ---------------------------------- #


def politis_white_block_length(x: np.ndarray) -> int:
    """Politis-White (2004) automatic MBB block length, Patton-Politis-White (2009)
    correction. Flat-top lag-window selection of the autocorrelation length, then
    b_opt = (2 g^2 / D_MBB)^(1/3) n^(1/3) with D_MBB = (4/3) sigma^4. White noise
    collapses to b=1 (iid bootstrap); the cap is ceil(min(3 sqrt(n), n/3))."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 30:
        return max(1, n // 4)
    xc = x - x.mean()
    gamma0 = float((xc * xc).mean())
    if gamma0 <= 0:
        return 1
    k_n = max(5, int(np.ceil(np.sqrt(np.log10(n)))))
    maxlag = min(n - 2, int(np.ceil(np.sqrt(n))) + k_n)
    rho = np.array([(xc[k:] * xc[:-k]).mean() / gamma0 for k in range(1, maxlag + 1)])
    thresh = 2.0 * np.sqrt(np.log10(n) / n)
    m_hat = maxlag - k_n  # fallback: everything correlated
    for m in range(0, maxlag - k_n + 1):
        if np.all(np.abs(rho[m : m + k_n]) < thresh):
            m_hat = m
            break
    big_m = min(2 * m_hat, maxlag)
    lags = np.arange(1, big_m + 1)
    lam = np.where(lags / max(big_m, 1) <= 0.5, 1.0,
                   np.clip(2.0 * (1.0 - lags / max(big_m, 1)), 0.0, 1.0))
    gam = gamma0 * rho[:big_m] if big_m else np.array([])
    g_hat = float(2.0 * np.sum(lam * lags * gam))
    sigma2 = float(gamma0 + 2.0 * np.sum(lam * gam))
    if sigma2 <= 0 or g_hat == 0.0:
        return 1
    d_mbb = (4.0 / 3.0) * sigma2 ** 2
    b = ((2.0 * g_hat ** 2) / d_mbb) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    b_max = int(np.ceil(min(3.0 * np.sqrt(n), n / 3.0)))
    return int(np.clip(round(b), 1, b_max))


def _rolling_sharpe_np(r: np.ndarray, window: int) -> np.ndarray:
    """Annualized rolling Sharpe via cumsums; matches ``rolling_sharpe`` (ddof=1)."""
    n = len(r)
    if n < window:
        return np.empty(0)
    s = np.concatenate([[0.0], np.cumsum(r)])
    q = np.concatenate([[0.0], np.cumsum(r * r)])
    mu = (s[window:] - s[:-window]) / window
    ex2 = (q[window:] - q[:-window]) / window
    var = (ex2 - mu * mu) * window / (window - 1)
    out = np.full(n - window + 1, np.nan)
    ok = var > 0
    out[ok] = mu[ok] / np.sqrt(var[ok]) * np.sqrt(TRADING_DAYS)
    return out


@dataclass(frozen=True)
class SSRInference:
    """SSR point estimates plus the paper's one-sided MBB p-value (Test 1).

    SSR measures, inference decides. The bare SSR is an effect size — "how far
    above the benchmark sits the mean rolling Sharpe, in units of the path's own
    long-run volatility" — with no sampling distribution attached, so a verdict
    needs a null: the MBB resamples the observed RETURNS in dependence-preserving
    blocks and rebuilds the whole rolling-Sharpe -> SSR pipeline per replicate.
    ``stable`` is the verdict bit: p < alpha AND mean rolling Sharpe above the
    benchmark. ``verdict(differential=...)`` renders the repo's single canonical
    verdict string — every tear sheet / luck-vs-skill table / gate must use it
    rather than re-encoding a rule. The bundled settings (window, n_boot, seed,
    alpha, block_len) make every verdict reproducible (R2.8)."""

    result: SSRResult
    sr_star: float
    p_value: float
    block_len: int
    n_boot: int
    seed: int
    alpha: float
    #: mirror-tail p for H0: mu_Z >= sr_star. ``p_value`` alone can only reject
    #: UPWARD, so without this a decisively negative sample is indistinguishable
    #: from a genuinely inconclusive one and both render as "luck-compatible".
    p_value_lower: float = float("nan")
    window: int = TRADING_DAYS
    periods_per_year: int = TRADING_DAYS

    @property
    def stable(self) -> bool:
        # bool(...) — np.isfinite returns np.bool_, which leaks through `and` and
        # fails callers asserting `is False`.
        return bool(
            np.isfinite(self.result.ssr)
            and np.isfinite(self.p_value)
            and self.p_value < self.alpha
            and self.result.mean_rolling_sr > self.sr_star
        )

    @property
    def stably_below(self) -> bool:
        """Mirror of ``stable``: the sample rejects DOWNWARD at ``alpha``."""
        return bool(
            np.isfinite(self.result.ssr)
            and np.isfinite(self.p_value_lower)
            and self.p_value_lower < self.alpha
            and self.result.mean_rolling_sr < self.sr_star
        )

    def verdict(self, *, differential: bool = False) -> str:
        if not np.isfinite(self.result.ssr) or not np.isfinite(self.p_value):
            return "insufficient rolling observations for inference"
        ssr, p, q = self.result.ssr, self.p_value, self.p_value_lower
        subject = "differential" if differential else "rolling Sharpe"
        star = "zero" if differential else f"{self.sr_star:g}"
        if self.stable:
            if differential:
                return (f"SSR={ssr:.2f}: differential stably > 0 (one-sided MBB "
                        f"p={p:.3f} < {self.alpha:g}) — QUANTIFIED LOOKAHEAD/RECALL "
                        f"BIAS, never attainable skill")
            return (f"SSR={ssr:.2f}: rolling Sharpe stably > {self.sr_star:g} in this "
                    f"sample (one-sided MBB p={p:.3f} < {self.alpha:g}) — temporal "
                    f"consistency, not a skill claim")
        # Not stable upward. Distinguish "decisively on the wrong side" from
        # "genuinely inconclusive" — calling the former luck-compatible is false.
        if self.stably_below:
            return (f"SSR={ssr:.2f}: {subject} stably BELOW {star} in this sample "
                    f"(mirror-tail MBB p={q:.3f} < {self.alpha:g}) — decisively "
                    f"negative, NOT merely inconclusive")
        tail = f", mirror p={q:.3f}" if np.isfinite(q) else ""
        return (f"SSR={ssr:.2f}: {subject} NOT distinguishable from {star} "
                f"(one-sided MBB p={p:.3f}{tail}) — luck-compatible"
                + (", not skill" if differential else ""))


def _finite_real(name: str, value: Real) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be a finite real number")
    return normalized


def _positive_int(name: str, value: Integral, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return normalized


def _validate_returns(returns: pd.Series) -> None:
    if not isinstance(returns, pd.Series):
        raise ValueError("returns must be a pandas Series")
    if returns.empty:
        raise ValueError("returns index must be non-empty")
    index = returns.index
    if isinstance(index, pd.MultiIndex):
        raise ValueError("returns index must be one-dimensional")
    if index.hasnans:
        raise ValueError("returns index must not contain missing labels")
    if index.has_duplicates:
        raise ValueError("returns index must be unique")
    if not index.is_monotonic_increasing:
        raise ValueError("returns index must be strictly increasing")
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        raise ValueError("returns index must be timezone-naive")
    if (
        not pd.api.types.is_numeric_dtype(returns.dtype)
        or pd.api.types.is_bool_dtype(returns.dtype)
        or pd.api.types.is_complex_dtype(returns.dtype)
    ):
        raise ValueError("returns values must be finite real numeric values")
    try:
        values = returns.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("returns values must be finite real numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError("returns values must be finite")


def ssr_inference(
    returns: pd.Series,
    *,
    window: int = TRADING_DAYS,
    sr_star: float = 0.0,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> SSRInference:
    """Paper-faithful MBB inference on the SSR (§3.3.2-3.3.3, Test 1).

    Resamples contiguous blocks of the OBSERVED RETURNS (block length from
    Politis-White on the return series), rebuilds the rolling-Sharpe path and the
    Newey-West/Andrews SSR per replicate, and reports p = #{SSR_b <= 0} / B for
    H0: mu_Z <= sr_star. Deterministic for fixed (returns, kwargs): one seeded
    Generator drives all draws. Replicates whose SSR is undefined count AGAINST
    stability (conservative). Measured size: 5.2 % (n=1500) / 5.6 % (n=3900) under an
    iid null and 3.6 % under a GARCH null, at nominal 5 % (B=300, 250 reps); power 75 %
    against a true annual Sharpe of 1.0 on 6y — evidence in docs/ssr_verdict_review.md."""
    _validate_returns(returns)
    window = _positive_int("window", window, minimum=2)
    if window > len(returns):
        raise ValueError(f"window={window} must not exceed returns length {len(returns)}")
    n_boot = _positive_int("n_boot", n_boot, minimum=1)
    seed = _positive_int("seed", seed, minimum=0)
    sr_star = _finite_real("sr_star", sr_star)
    alpha = _finite_real("alpha", alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")

    res = compute_ssr(returns, window=window, sr_star=sr_star)
    r = returns.to_numpy(dtype=np.float64)
    if not np.isfinite(res.ssr):
        return SSRInference(
            result=res,
            sr_star=sr_star,
            p_value=np.nan,
            block_len=0,
            n_boot=n_boot,
            seed=seed,
            alpha=alpha,
            p_value_lower=np.nan,
            window=window,
            periods_per_year=TRADING_DAYS,
        )

    block_len = politis_white_block_length(r)
    t = len(r)
    n_blocks = int(np.ceil(t / block_len))
    starts_max = t - block_len + 1
    rng = np.random.default_rng(seed)
    offsets = np.arange(block_len)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max, n_blocks)
        idx = (starts[:, None] + offsets[None, :]).ravel()[:t]
        z = _rolling_sharpe_np(r[idx], window)
        z = z[np.isfinite(z)]
        if len(z) < 10:
            draws[b] = np.nan
            continue
        lr_var, _ = newey_west_var(z)
        sig = np.sqrt(lr_var) if lr_var > 0 else np.nan
        draws[b] = (z.mean() - sr_star) / sig if np.isfinite(sig) and sig > 0 else np.nan
    # undefined replicates count as "failed to exceed the benchmark" (conservative)
    ok = np.isfinite(draws)
    n_bad = int(np.sum(~ok))
    # (r + 1) / (B + 1), not r / B: the naive ratio reports p == 0.0 exactly whenever
    # no replicate crosses, which is not a valid Monte-Carlo p-value — it asserts
    # more evidence than B draws can carry. The add-one estimator (Davison & Hinkley
    # 1997; Phipson & Smyth 2010) floors p at 1/(B+1) and stays unbiased under H0.
    p = float((np.sum(draws[ok] <= 0.0) + n_bad + 1) / (n_boot + 1))
    # mirror tail (H0: mu_Z >= sr_star); undefined replicates again count against
    # rejection, so both tails stay conservative rather than summing to 1.
    p_lower = float((np.sum(draws[ok] >= 0.0) + n_bad + 1) / (n_boot + 1))
    return SSRInference(
        result=res,
        sr_star=sr_star,
        p_value=p,
        block_len=block_len,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
        p_value_lower=p_lower,
        window=window,
        periods_per_year=TRADING_DAYS,
    )
