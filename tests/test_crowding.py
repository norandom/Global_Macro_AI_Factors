"""Tests for macro_framework.crowding — absorption ratio, turbulence, PIT bucketing.

Deterministic seeded synthetic data, no DB. The PIT tests append a violent
regime flip that WOULD change historical values if the estimators leaked
future data, so they genuinely bite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from macro_framework.crowding import absorption_ratio, crowding_bucket, turbulence


def _bdays(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2018-01-01", periods=n)


def _one_factor_returns(
    rng: np.random.Generator, n_days: int, n_assets: int, idio_vol: float = 0.0005
) -> np.ndarray:
    """One dominant common factor + tiny idio noise -> near-rank-1 correlation."""
    common = rng.normal(0.0, 0.01, (n_days, 1))
    return common + rng.normal(0.0, idio_vol, (n_days, n_assets))


def test_absorption_ratio_rises_on_regime_concentration():
    rng = np.random.default_rng(7)
    n_assets = 20
    # first 250 days: 20 independent factors (each asset its own) -> low AR
    r1 = rng.normal(0.0, 0.01, (250, n_assets))
    # next 250 days: ONE dominant common factor -> AR near 1
    r2 = _one_factor_returns(rng, 250, n_assets)
    rets = pd.DataFrame(
        np.vstack([r1, r2]),
        index=_bdays(500),
        columns=[f"A{i}" for i in range(n_assets)],
    )
    ar = absorption_ratio(rets, window=120, step=5)
    assert ar.iloc[-1] - ar.iloc[0] >= 0.2
    assert ((ar > 0) & (ar <= 1)).all()


def test_turbulence_spikes_on_common_shock():
    rng = np.random.default_rng(11)
    n, n_assets, window, shock_pos = 400, 5, 100, 350
    rets = pd.DataFrame(
        rng.normal(0.0, 0.01, (n, n_assets)),
        index=_bdays(n),
        columns=[f"A{i}" for i in range(n_assets)],
    )
    rets.iloc[shock_pos] = 0.10  # one massive common shock day
    turb = turbulence(rets, window=window)
    shock_date = rets.index[shock_pos]
    quiet = turb.loc[turb.index < shock_date]
    assert turb.loc[shock_date] > 5 * quiet.median()


def test_absorption_ratio_is_pit():
    rng = np.random.default_rng(3)
    n_assets, n_hist, n_future = 10, 300, 120
    idx = _bdays(n_hist + n_future)
    cols = [f"A{i}" for i in range(n_assets)]
    hist = pd.DataFrame(
        rng.normal(0.0, 0.01, (n_hist, n_assets)), index=idx[:n_hist], columns=cols
    )
    # violent regime flip appended: near-perfectly common returns that WOULD
    # raise historical AR values if any future data leaked into the estimator
    fut = pd.DataFrame(
        _one_factor_returns(rng, n_future, n_assets), index=idx[n_hist:], columns=cols
    )
    full = pd.concat([hist, fut])
    ar_trunc = absorption_ratio(hist, window=120, step=5)
    ar_full = absorption_ratio(full, window=120, step=5)
    pd.testing.assert_series_equal(ar_full.loc[ar_trunc.index], ar_trunc)
    # sanity: the flip genuinely moves the signal, so leakage would have shown
    assert ar_full.iloc[-1] > ar_trunc.iloc[-1] + 0.2


def test_crowding_bucket_is_pit():
    rng = np.random.default_rng(5)
    n_hist, n_future = 400, 200
    idx = _bdays(n_hist + n_future)
    hist = pd.Series(rng.normal(0.0, 1.0, n_hist), index=idx[:n_hist])
    # violent flip: extreme future values; full-sample quantiles would push
    # historical top-bucket dates down, so leakage would change b_trunc dates
    fut = pd.Series(rng.normal(50.0, 1.0, n_future), index=idx[n_hist:])
    full = pd.concat([hist, fut])
    b_trunc = crowding_bucket(hist, min_obs=100)
    b_full = crowding_bucket(full, min_obs=100)
    pd.testing.assert_series_equal(b_full.loc[b_trunc.index], b_trunc)
    # sanity: historical top buckets exist, which full-sample quantiles would erase
    assert (b_trunc == 2).any()


def test_crowding_bucket_monotone_signal_and_min_obs():
    n = 600
    sig = pd.Series(np.arange(n, dtype=float), index=_bdays(n))
    b = crowding_bucket(sig)  # n_buckets=3, min_obs=252
    assert (b.iloc[: 252 - 1] == 1).all()  # before min_obs -> middle bucket
    assert b.iloc[-1] == 2  # monotone increasing signal ends in top bucket
    assert b.between(0, 2).all()
    assert pd.api.types.is_integer_dtype(b)
