"""Tests for macro_framework.skill_metric — own-basket appraisal ratio + market attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_framework.skill_metric import (
    IDIO_FLOOR,
    BasketResidual,
    GateConfig,
    GateVerdict,
    MarketAttribution,
    basket_residual,
    evaluate_gates,
    market_attribution,
)
from macro_framework.ssr import SSRResult

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


# --- Gate verdict truth table (Requirement 2) ------------------------------------


def _residual(t=4.0, appraisal=0.9):
    return BasketResidual(
        alpha_ann=0.1,
        t_alpha_hac=t,
        r2=0.9,
        idio_vol_ann=0.11,
        appraisal=appraisal,
        n_obs=500,
        hac_maxlags=5,
    )


def _ssr(value=2.5):
    return SSRResult(
        n_obs=500,
        n_rolling=250,
        sr_full=1.0,
        mean_rolling_sr=1.0,
        sigma_hac=0.4,
        L_hac=5,
        ssr=value,
    )


# baseline all-pass gate inputs
_PASS = dict(
    recall_premium=0.0,
    oos_calmar=1.2,
    baseline_calmar=1.0,
    oos_maxdd=-0.10,
    baseline_maxdd=-0.15,
)


def test_all_gates_pass():
    v = evaluate_gates(_residual(), _ssr(), **_PASS)
    assert isinstance(v, GateVerdict)
    assert v.passed is True
    assert (v.skill_pass, v.stability_pass, v.recall_pass, v.risk_shape_pass) == (
        True,
        True,
        True,
        True,
    )
    assert v.first_failure is None
    assert v.values["skill_t"] == 4.0
    assert v.values["ssr"] == 2.5


def test_skill_gate_flip():
    v = evaluate_gates(_residual(t=1.5), _ssr(), **_PASS)
    assert v.passed is False
    assert v.skill_pass is False
    assert v.stability_pass and v.recall_pass and v.risk_shape_pass
    assert v.first_failure.startswith("skill:")


def test_stability_gate_flip():
    v = evaluate_gates(_residual(), _ssr(0.14), **_PASS)
    assert v.passed is False
    assert v.stability_pass is False
    assert v.first_failure.startswith("stability:")
    assert "0.14" in v.first_failure and "1.96" in v.first_failure


def test_recall_gate_flip():
    args = {**_PASS, "recall_premium": 0.30}
    v = evaluate_gates(_residual(), _ssr(), **args)
    assert v.passed is False
    assert v.recall_pass is False
    assert v.first_failure.startswith("recall:")


def test_recall_gate_symmetric():
    args = {**_PASS, "recall_premium": -0.30}
    v = evaluate_gates(_residual(), _ssr(), **args)
    assert v.recall_pass is False  # |premium| tested, sign-independent


def test_risk_shape_calmar_flip():
    args = {**_PASS, "oos_calmar": 0.5}  # below baseline 1.0
    v = evaluate_gates(_residual(), _ssr(), **args)
    assert v.passed is False
    assert v.risk_shape_pass is False
    assert v.first_failure.startswith("risk_shape:")


def test_risk_shape_maxdd_flip():
    args = {**_PASS, "oos_maxdd": -0.40}  # worse than baseline -0.15
    v = evaluate_gates(_residual(), _ssr(), **args)
    assert v.passed is False
    assert v.risk_shape_pass is False
    assert v.first_failure.startswith("risk_shape:")


def test_first_failure_reports_skill_before_stability():
    # both skill and stability fail; skill is reported first
    v = evaluate_gates(_residual(t=0.5), _ssr(0.1), **_PASS)
    assert v.first_failure.startswith("skill:")


def test_nan_ssr_fails_stability():
    v = evaluate_gates(_residual(), _ssr(np.nan), **_PASS)
    assert v.passed is False
    assert v.stability_pass is False
    assert v.first_failure.startswith("stability:")


def test_nan_tstat_fails_skill():
    v = evaluate_gates(_residual(t=np.nan), _ssr(), **_PASS)
    assert v.passed is False
    assert v.skill_pass is False
    assert v.first_failure.startswith("skill:")


def test_none_appraisal_fails_skill():
    v = evaluate_gates(_residual(appraisal=None), _ssr(), **_PASS)
    assert v.passed is False
    assert v.skill_pass is False
    assert v.first_failure.startswith("skill:")


def test_relative_mode_passes_where_absolute_fails():
    res = _residual(t=1.5, appraisal=0.4)  # 0 < t < 2, positive appraisal
    assert evaluate_gates(res, _ssr(), **_PASS).skill_pass is False  # absolute default
    cfg = GateConfig(mode="relative_improvement")
    v = evaluate_gates(res, _ssr(), **_PASS, config=cfg)
    assert v.skill_pass is True
    assert v.passed is True


def test_relative_mode_fails_on_negative_t():
    res = _residual(t=-0.5, appraisal=-0.2)
    v = evaluate_gates(res, _ssr(), **_PASS, config=GateConfig(mode="relative_improvement"))
    assert v.skill_pass is False


def test_custom_thresholds():
    cfg = GateConfig(skill_t_min=3.0, ssr_min=1.0)
    v = evaluate_gates(_residual(t=2.5), _ssr(1.5), **_PASS, config=cfg)
    assert v.skill_pass is False  # 2.5 < 3.0
    assert v.stability_pass is True  # 1.5 >= 1.0
