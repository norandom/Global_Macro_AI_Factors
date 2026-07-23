"""Return-space crowding / fragility proxies (pure numpy/pandas, PIT-clean).

Concentration and fragility signals computed from a daily-returns cross-section
(callers fetch the data; this module never touches a DB). All estimators are
point-in-time by construction: absorption_ratio and turbulence read only
trailing windows, crowding_bucket ranks only against the expanding history up
to and including each date. All functions are pure and deterministic.

Estimators:
- absorption_ratio: Kritzman, Li, Page & Rigobon (2011), "Principal Components
  as a Measure of Systemic Risk".
- turbulence: Kritzman & Li (2010), "Skulls, Financial Turbulence, and Risk
  Management" (Mahalanobis distance of the day's return vector).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def absorption_ratio(
    returns: pd.DataFrame,
    *,
    window: int = 252,
    n_components: int | None = None,
    step: int = 5,
) -> pd.Series:
    """Kritzman-Li-Page-Rigobon (2011) absorption ratio, trailing-window PIT.

    For each computation date t (starting after the first full ``window``),
    take the trailing ``window`` rows up to and including t, keep only columns
    with no NaN and nonzero variance in the window, and eigendecompose the
    correlation matrix of the (standardized) returns. The absorption ratio is
    the fraction of total variance absorbed by the top ``n_components``
    eigenvectors (default ``ceil(n_kept_assets / 5)`` per the paper).

    For speed the ratio is computed only every ``step`` trading days and
    forward-filled between computation dates, so intermediate dates carry the
    most recent computed value.

    High AR = variance concentrated in few factors = crowded/fragile market.
    Values lie in (0, 1]. Returns a Series indexed by date (from the first
    full-window date onward); a window that keeps no asset computes NaN, but the
    final forward-fill replaces it with the most recent computed value, so only
    leading no-asset windows surface as NaN.
    """
    idx = returns.index
    out: dict = {}
    for pos in range(window - 1, len(returns), step):
        win = returns.iloc[pos - window + 1 : pos + 1]
        keep = win.notna().all() & (win.std(ddof=0) > 0)
        sub = win.loc[:, keep]
        n_kept = sub.shape[1]
        if n_kept == 0:
            out[idx[pos]] = np.nan
            continue
        if n_kept == 1:
            out[idx[pos]] = 1.0
            continue
        k = math.ceil(n_kept / 5) if n_components is None else min(n_components, n_kept)
        corr = np.corrcoef(sub.values, rowvar=False)
        eig = np.linalg.eigvalsh(corr)  # ascending
        eig = np.clip(eig, 0.0, None)  # correlation matrices are PSD; drop float noise
        out[idx[pos]] = float(eig[-k:].sum() / eig.sum())
    ser = pd.Series(out, dtype=float, name="absorption_ratio")
    return ser.reindex(idx[window - 1 :]).ffill()


def turbulence(
    returns: pd.DataFrame,
    *,
    window: int = 252,
    ridge: float = 1e-6,
    step: int = 1,
) -> pd.Series:
    """Kritzman-Li (2010) financial turbulence: PIT Mahalanobis distance.

    For each scored date t, the baseline mean ``mu`` and covariance ``Sigma``
    come from the trailing ``window`` rows strictly BEFORE t (the day being
    scored is never in its own baseline), and

        d_t = (r_t - mu) @ pinv(Sigma + ridge * I) @ (r_t - mu)

    Only columns with no NaN in the window and on day t are kept; a date is
    NaN when ``window < 2 * n_kept_assets`` (covariance too poorly determined)
    or no column survives. Returns a daily Series (with ``step > 1`` only
    every step-th day is scored). High turbulence = statistically unusual
    cross-sectional return day.
    """
    idx = returns.index
    out: dict = {}
    for pos in range(window, len(returns), step):
        base = returns.iloc[pos - window : pos]
        today = returns.iloc[pos]
        keep = base.notna().all() & today.notna()
        sub = base.loc[:, keep]
        n_kept = sub.shape[1]
        if n_kept == 0 or window < 2 * n_kept:
            out[idx[pos]] = np.nan
            continue
        x = today[keep].to_numpy() - sub.mean().to_numpy()
        sigma = np.atleast_2d(np.cov(sub.to_numpy(), rowvar=False))
        d = x @ np.linalg.pinv(sigma + ridge * np.eye(n_kept)) @ x
        out[idx[pos]] = float(d)
    return pd.Series(out, dtype=float, name="turbulence")


def crowding_bucket(
    signal: pd.Series,
    *,
    n_buckets: int = 3,
    min_obs: int = 252,
) -> pd.Series:
    """PIT quantile bucket of a signal against its own expanding history.

    For each date t the bucket (0..n_buckets-1) is the quantile rank of
    ``signal_t`` within the EXPANDING window of the signal up to and including
    t — never full-sample quantiles (that would be lookahead). Vectorized:
    with expanding rank r_t of signal_t among signal[:t] inclusive and
    expanding non-NaN count c_t, ``bucket = floor(r_t / c_t * n_buckets)``
    clipped to ``n_buckets - 1``. Dates with fewer than ``min_obs``
    observations (or a NaN signal) get the middle bucket ``n_buckets // 2``.
    Returns an integer Series.
    """
    rank = signal.expanding().rank()
    count = signal.expanding().count()
    bucket = np.floor(rank / count * n_buckets).clip(upper=n_buckets - 1)
    middle = n_buckets // 2
    return bucket.where((count >= min_obs) & rank.notna(), middle).astype(int)
