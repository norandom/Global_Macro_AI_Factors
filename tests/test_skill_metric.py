"""Tests for macro_framework.skill_metric — own-basket appraisal ratio + market attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_framework.skill_metric import (
    IDIO_FLOOR,
    BasketResidual,
    MarketAttribution,
    basket_residual,
    market_attribution,
)

RNG = np.random.default_rng(20260720)
FACTORS = ("SWDA.L", "XLK", "IAU", "BIL")


def _factor_frame(n: int = 750) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n)
    data = {c: RNG.normal(0.0003, 0.01, n) for c in FACTORS}
    return pd.DataFrame(data, index=idx)


def test_recovers_injected_alpha_and_reports_hac_t():
    factors = _factor_frame()
    known_alpha_daily = 0.0004  # ~10%/yr
    beta = 0.8
    strat = (
        known_alpha_daily
        + beta * factors["XLK"]
        + RNG.normal(0.0, 0.0005, len(factors))  # small idio noise, above floor
    )
    res = basket_residual(strat, factors)
    assert isinstance(res, BasketResidual)
    assert res.alpha_ann == pytest.approx(known_alpha_daily * 252, abs=0.02)
    assert np.isfinite(res.t_alpha_hac) and res.t_alpha_hac > 3.0
    assert 0.0 <= res.r2 <= 1.0
    assert res.idio_vol_ann >= IDIO_FLOOR
    assert res.appraisal == pytest.approx(res.alpha_ann / res.idio_vol_ann)
    assert res.n_obs == len(factors)
    assert res.hac_maxlags == 5


def test_appraisal_none_when_residual_below_floor():
    factors = _factor_frame()
    # strategy is an exact linear combo of the factors → zero residual
    strat = 0.5 * factors["SWDA.L"] + 0.3 * factors["IAU"] + 0.2 * factors["BIL"]
    res = basket_residual(strat, factors)
    assert res.idio_vol_ann < IDIO_FLOOR
    assert res.appraisal is None


def test_inner_join_drops_mismatched_dates():
    factors = _factor_frame()
    strat = 0.0004 + factors["XLK"] + RNG.normal(0, 0.0005, len(factors))
    # simulate LSE-holiday style gaps: drop a few factor rows
    factors_gapped = factors.drop(factors.index[[10, 20, 30]])
    res = basket_residual(strat, factors_gapped)
    assert res.n_obs == len(factors_gapped)


def test_market_attribution_self_regression():
    idx = pd.bdate_range("2020-01-01", periods=500)
    mkt = pd.Series(RNG.normal(0.0003, 0.01, 500), index=idx)
    res = market_attribution(mkt, mkt)
    assert isinstance(res, MarketAttribution)
    assert res.beta == pytest.approx(1.0, abs=1e-8)
    assert res.r2 == pytest.approx(1.0, abs=1e-8)
    assert res.alpha_ann == pytest.approx(0.0, abs=1e-8)


def test_market_attribution_recovers_beta_and_alpha():
    idx = pd.bdate_range("2020-01-01", periods=750)
    mkt = pd.Series(RNG.normal(0.0003, 0.01, 750), index=idx)
    alpha_daily, beta = 0.0003, 1.3
    strat = alpha_daily + beta * mkt + RNG.normal(0, 0.0005, 750)
    res = market_attribution(strat, mkt)
    assert res.beta == pytest.approx(beta, abs=0.05)
    assert res.alpha_ann == pytest.approx(alpha_daily * 252, abs=0.02)


def test_deterministic():
    factors = _factor_frame()
    strat = 0.0004 + factors["XLK"] + RNG.normal(0, 0.0005, len(factors))
    a = basket_residual(strat, factors)
    b = basket_residual(strat, factors)
    assert a == b
