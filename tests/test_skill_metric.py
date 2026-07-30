"""Tests for strict return construction and raw-return HAC attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_framework.skill_metric import (
    IDIO_FLOOR,
    AttributionKind,
    BasketResidual,
    GateConfig,
    GateVerdict,
    MarketAttribution,
    basket_residual,
    differential_returns,
    evaluate_gates,
    factor_returns_on,
    market_attribution,
    portfolio_excess_returns,
    raw_market_model_attribution,
)
from macro_framework.ssr import SSRInference, SSRResult

RNG = np.random.default_rng(20260720)
FACTORS = ("SWDA.L", "XLK", "IAU", "BIL")


def _factor_frame(n: int = 750) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n)
    data = {c: RNG.normal(0.0003, 0.01, n) for c in FACTORS}
    return pd.DataFrame(data, index=idx)


def test_recovers_injected_basket_intercept_and_complete_hac_metadata():
    factors = _factor_frame()
    known_intercept = 0.0004
    strat = known_intercept + 0.8 * factors["XLK"] + RNG.normal(0.0, 0.0005, len(factors))

    res = basket_residual(strat, factors, periods_per_year=252, hac_maxlags=7)

    assert isinstance(res, BasketResidual)
    assert res.intercept_native_period == pytest.approx(known_intercept, abs=0.02 / 252)
    assert res.intercept_ann_arithmetic == pytest.approx(known_intercept * 252, abs=0.02)
    assert np.isfinite(res.intercept_t_hac) and res.intercept_t_hac > 3.0
    assert 0.0 <= res.r2 <= 1.0
    assert res.idio_vol_ann >= IDIO_FLOOR
    assert res.appraisal == pytest.approx(res.intercept_ann_arithmetic / res.idio_vol_ann)
    assert (res.n_obs, res.start, res.end) == (len(factors), factors.index[0], factors.index[-1])
    assert (res.periods_per_year, res.hac_maxlags) == (252, 7)
    assert res.alpha_ann == res.intercept_ann_arithmetic
    assert res.t_alpha_hac == res.intercept_t_hac


def test_appraisal_none_when_residual_below_floor():
    factors = _factor_frame()
    strat = 0.5 * factors["SWDA.L"] + 0.3 * factors["IAU"] + 0.2 * factors["BIL"]
    res = basket_residual(strat, factors)
    assert res.idio_vol_ann < IDIO_FLOOR
    assert res.appraisal is None


def test_basket_regression_rejects_one_date_mismatch_instead_of_shortening():
    factors = _factor_frame()
    strat = 0.0004 + factors["XLK"]
    with pytest.raises(ValueError, match="identical indexes"):
        basket_residual(strat, factors.drop(factors.index[10]))


def test_factor_returns_on_selects_anchor_before_compounding_holiday_gap():
    idx = pd.bdate_range("2020-01-01", periods=6)
    prices = pd.Series([100.0, 101.0, 103.0, 102.0, 104.0, 105.0], index=idx)
    return_index = idx[[1, 3, 4, 5]]

    got = factor_returns_on(prices, return_index, anchor=idx[0])

    assert got.index.equals(return_index)
    assert got.notna().all()
    assert got.loc[idx[3]] == pytest.approx(102.0 / 101.0 - 1.0)
    assert prices.pct_change(fill_method=None).loc[idx[3]] == pytest.approx(102.0 / 103.0 - 1.0)
    assert (1 + got).prod() == pytest.approx(105.0 / 100.0)


def test_factor_returns_on_requires_explicit_strictly_preceding_anchor():
    idx = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.Series([100.0, 101.0, 102.0, 103.0], index=idx)
    with pytest.raises(TypeError, match="anchor"):
        factor_returns_on(prices, idx[1:])
    with pytest.raises(ValueError, match="strictly before"):
        factor_returns_on(prices, idx[1:], anchor=idx[1])
    with pytest.raises(ValueError, match="must not be NaT"):
        factor_returns_on(prices, idx[1:], anchor=pd.NaT)
    with pytest.raises(ValueError, match="timezone-naive"):
        factor_returns_on(prices, idx[1:], anchor=idx[0].tz_localize("UTC"))


def test_factor_returns_on_dataframe_matches_per_column_series():
    idx = pd.bdate_range("2020-01-01", periods=6)
    frame = pd.DataFrame(
        {"A": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0],
         "B": [50.0, 51.0, 52.0, 53.0, 54.0, 55.0]},
        index=idx,
    )
    requested = idx[[1, 3, 5]]

    got = factor_returns_on(frame, requested, anchor=idx[0])

    assert list(got.columns) == ["A", "B"]
    assert got.index.equals(requested)
    for col in frame:
        pd.testing.assert_series_equal(
            got[col],
            factor_returns_on(frame[col], requested, anchor=idx[0]),
            check_names=False,
        )


def test_factor_returns_on_dataframe_names_the_offending_column():
    idx = pd.bdate_range("2020-01-01", periods=4)
    frame = pd.DataFrame(
        {"A": [100.0, 101.0, 102.0, 103.0], "B": [50.0, np.nan, 52.0, 53.0]},
        index=idx,
    )
    with pytest.raises(ValueError, match="column 'B'"):
        factor_returns_on(frame, idx[1:], anchor=idx[0])


def test_factor_returns_on_rejects_absent_anchor_or_requested_label():
    idx = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.Series(np.arange(5, dtype=float) + 100.0, index=idx)
    with pytest.raises(ValueError, match="first missing"):
        factor_returns_on(prices, idx[2:], anchor=idx[0] - pd.offsets.BDay())
    requested = idx[1:].append(pd.DatetimeIndex([idx[-1] + pd.offsets.BDay()]))
    with pytest.raises(ValueError, match="first missing"):
        factor_returns_on(prices, requested, anchor=idx[0])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_factor_returns_on_rejects_present_non_finite_selected_levels(bad):
    idx = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.Series([100.0, bad, 102.0, 103.0], index=idx)
    with pytest.raises(ValueError, match="selected price levels"):
        factor_returns_on(prices, idx[1:], anchor=idx[0])


@pytest.mark.parametrize("values", [[True, False, True], [100 + 0j, 101 + 1j, 102 + 2j]])
def test_factor_returns_on_rejects_non_real_price_levels(values):
    idx = pd.bdate_range("2020-01-01", periods=3)
    prices = pd.Series(values, index=idx)

    with pytest.raises(ValueError, match="finite real numeric"):
        factor_returns_on(prices, idx[1:], anchor=idx[0])


@pytest.mark.parametrize(
    ("bad_index", "match"),
    [
        (pd.DatetimeIndex([]), "must not be empty"),
        (pd.DatetimeIndex(["2020-01-01", "2020-01-01"]), "unique"),
        (pd.DatetimeIndex(["2020-01-02", "2020-01-01"]), "strictly increasing"),
        (pd.DatetimeIndex([pd.NaT, "2020-01-02"]), "NaT"),
        (pd.date_range("2020-01-01", periods=2, tz="UTC"), "timezone-naive"),
    ],
)
def test_factor_returns_on_rejects_malformed_source_index(bad_index, match):
    prices = pd.Series(np.arange(len(bad_index), dtype=float) + 100.0, index=bad_index)
    with pytest.raises(ValueError, match=match):
        factor_returns_on(
            prices,
            pd.DatetimeIndex(["2020-01-02"]),
            anchor=pd.Timestamp("2020-01-01"),
        )


@pytest.mark.parametrize(
    ("bad_index", "match"),
    [
        (pd.DatetimeIndex([]), "must not be empty"),
        (pd.DatetimeIndex(["2020-01-02", "2020-01-02"]), "unique"),
        (pd.DatetimeIndex(["2020-01-03", "2020-01-02"]), "strictly increasing"),
        (pd.DatetimeIndex([pd.NaT, "2020-01-02"]), "NaT"),
        (pd.date_range("2020-01-02", periods=2, tz="UTC"), "timezone-naive"),
    ],
)
def test_factor_returns_on_rejects_malformed_requested_index(bad_index, match):
    source = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.Series(np.arange(4, dtype=float) + 100.0, index=source)
    with pytest.raises(ValueError, match=match):
        factor_returns_on(prices, bad_index, anchor=source[0])


@pytest.mark.parametrize("helper", [portfolio_excess_returns, differential_returns])
def test_exact_return_helpers_subtract_elementwise_and_preserve_index(helper):
    idx = pd.bdate_range("2020-01-01", periods=3)
    left = pd.Series([0.03, -0.01, 0.02], index=idx, name="left")
    right = pd.Series([0.01, 0.005, -0.02], index=idx, name="right")
    got = helper(left, right)
    pd.testing.assert_series_equal(got, pd.Series([0.02, -0.015, 0.04], index=idx, name="left"))


@pytest.mark.parametrize("helper", [portfolio_excess_returns, differential_returns])
def test_exact_return_helpers_reject_index_or_value_defects(helper):
    idx = pd.bdate_range("2020-01-01", periods=3)
    valid = pd.Series([0.01, 0.02, 0.03], index=idx)
    with pytest.raises(ValueError, match="identical indexes"):
        helper(valid, valid.set_axis(idx + pd.offsets.BDay()))
    duplicate = valid.set_axis(pd.DatetimeIndex([idx[0], idx[0], idx[2]]))
    with pytest.raises(ValueError, match="unique"):
        helper(duplicate, duplicate)
    non_finite = valid.copy()
    non_finite.iloc[1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        helper(valid, non_finite)
    with pytest.raises(ValueError, match="finite real numeric"):
        helper(valid.astype(complex), valid.astype(complex))
    unordered = valid.set_axis(pd.DatetimeIndex([idx[1], idx[0], idx[2]]))
    with pytest.raises(ValueError, match="strictly increasing"):
        helper(unordered, unordered)


def test_raw_market_model_recovers_coefficients_and_reports_complete_metadata():
    idx = pd.bdate_range("2020-01-01", periods=750)
    mkt = pd.Series(RNG.normal(0.0003, 0.01, len(idx)), index=idx)
    intercept, beta = 0.0003, 1.3
    portfolio = intercept + beta * mkt + RNG.normal(0, 0.0005, len(idx))

    res = raw_market_model_attribution(portfolio, mkt, periods_per_year=252, hac_maxlags=9)

    assert isinstance(res, MarketAttribution)
    assert res.kind == "raw_market_model"
    assert res.intercept_native_period == pytest.approx(intercept, abs=0.02 / 252)
    assert res.intercept_ann_arithmetic == pytest.approx(intercept * 252, abs=0.02)
    assert np.isfinite(res.intercept_se_hac)
    assert np.isfinite(res.intercept_t_hac)
    assert res.beta == pytest.approx(beta, abs=0.05)
    assert 0.0 <= res.r2 <= 1.0
    assert (res.n_obs, res.start, res.end) == (len(idx), idx[0], idx[-1])
    assert (res.periods_per_year, res.hac_maxlags) == (252, 9)
    assert res.alpha_ann == res.intercept_ann_arithmetic


def test_hac_covariance_is_actually_used_not_nonrobust():
    """Kill-check for silently dropping cov_type='HAC': with strongly
    autocorrelated errors the Newey-West intercept se must be far above the
    nonrobust OLS se computed independently here."""
    import statsmodels.api as sm

    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2018-01-01", periods=1500)
    market = pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)
    noise = np.empty(len(idx))
    noise[0] = rng.normal(0.0, 0.001)
    for t in range(1, len(idx)):  # AR(1), rho=0.9 — long-run variance ~19x
        noise[t] = 0.9 * noise[t - 1] + rng.normal(0.0, 0.001)
    portfolio = 0.0005 + 1.2 * market + noise

    res = raw_market_model_attribution(portfolio, market, hac_maxlags=30)

    design = np.column_stack((np.ones(len(market)), market.to_numpy()))
    nonrobust_se = float(sm.OLS(portfolio.to_numpy(), design).fit().bse[0])
    assert res.intercept_se_hac > 2.0 * nonrobust_se


@pytest.mark.parametrize(
    "kwargs",
    [
        {"periods_per_year": True},
        {"periods_per_year": 0},
        {"periods_per_year": 2.5},
        {"hac_maxlags": -1},
        {"hac_maxlags": True},
        {"hac_maxlags": 1.5},
    ],
)
def test_regressions_reject_invalid_periods_or_lags(kwargs):
    factors = _factor_frame(50)
    strat = 0.0004 + factors["XLK"]
    with pytest.raises(ValueError):
        basket_residual(strat, factors, **kwargs)
    with pytest.raises(ValueError):
        raw_market_model_attribution(strat, factors["XLK"], **kwargs)


def test_raw_market_model_rejects_mismatch_and_non_finite_values():
    idx = pd.bdate_range("2020-01-01", periods=10)
    market = pd.Series(np.linspace(-0.01, 0.01, len(idx)), index=idx)
    portfolio = 0.001 + 0.8 * market
    with pytest.raises(ValueError, match="identical indexes"):
        raw_market_model_attribution(portfolio, market.drop(idx[3]))
    market.iloc[4] = np.inf
    with pytest.raises(ValueError, match="finite"):
        raw_market_model_attribution(portfolio, market)


def test_market_attribution_is_the_temporary_compatibility_export():
    assert market_attribution is raw_market_model_attribution


def test_raw_market_model_has_unambiguous_labels():
    # R3.8/R3.9: a CAPM/Jensen label would claim excess-on-excess regression with
    # a shared cash benchmark; the current raw-on-raw model must never carry one.
    import dataclasses
    import typing

    assert typing.get_args(AttributionKind) == ("raw_market_model",)
    for contract in (MarketAttribution, BasketResidual):
        names = {f.name for f in dataclasses.fields(contract)}
        assert not {n for n in names if "capm" in n.lower() or "jensen" in n.lower()}, contract
    assert {"intercept_native_period", "intercept_ann_arithmetic"} <= {
        f.name for f in dataclasses.fields(MarketAttribution)
    }


def test_ac_3_8():
    test_raw_market_model_has_unambiguous_labels()


def test_basket_residual_rejects_positional_construction():
    with pytest.raises(TypeError):
        BasketResidual(0.1, 4.0, 0.9, 0.11, 0.9, 500, 5)  # legacy 7-field shape


# --- Frozen coverage-matrix aliases (tasks 2.1/2.3): thin delegators onto the
#     semantic tests above, named exactly as coverage_matrix.json expects. --------


def test_holiday_gap_levels_align_before_returns():  # defect 6 shared boundary
    test_factor_returns_on_selects_anchor_before_compounding_holiday_gap()


def test_factor_returns_reject_present_nonfinite_levels():  # defect 12 shared boundary
    for bad in (np.nan, np.inf, -np.inf):
        test_factor_returns_on_rejects_present_non_finite_selected_levels(bad)


def test_anchor_retains_first_strategy_return():  # defect 15 shared boundary
    test_factor_returns_on_requires_explicit_strictly_preceding_anchor()
    test_factor_returns_on_selects_anchor_before_compounding_holiday_gap()


def test_ac_3_1():
    test_factor_returns_on_selects_anchor_before_compounding_holiday_gap()


def test_ac_3_2():
    test_holiday_gap_levels_align_before_returns()


def test_ac_3_3():
    test_factor_returns_on_rejects_absent_anchor_or_requested_label()


def test_ac_3_4():
    test_factor_returns_reject_present_nonfinite_levels()


def test_ac_3_5():
    test_anchor_retains_first_strategy_return()


def test_ac_3_9():
    test_raw_market_model_has_unambiguous_labels()


def test_ac_8_3():
    # calendar edge cases: holiday gap, first-return anchor, absent labels,
    # present-but-non-finite values (R8.3)
    test_holiday_gap_levels_align_before_returns()
    test_anchor_retains_first_strategy_return()
    test_factor_returns_on_rejects_absent_anchor_or_requested_label()
    test_factor_returns_reject_present_nonfinite_levels()


def test_full_report_rows_bind_attribution_to_the_performance_window():
    # R3.6: one common end date and observation set for performance + attribution
    from macro_framework.reporting import build_reader_metric_row
    from tests.test_reporting import _line

    meta, _, metrics, cash, excess, ssr, attr, market = _line()
    port = metrics["returns"]

    full = build_reader_metric_row(meta, metrics, cash, ssr, source="t", attribution=attr)
    assert full["row_kind"] == "full"
    assert full["raw_market_model_end"] == full["end"]
    assert full["raw_market_model_n_obs"] == full["n_obs"]

    short_attr = raw_market_model_attribution(port.iloc[:-30], market.iloc[:-30])
    short = build_reader_metric_row(meta, metrics, cash, ssr, source="t", attribution=short_attr)
    assert short["row_kind"] == "performance_only"
    assert not any(key.startswith("raw_market_model_") for key in short)


def test_ac_3_6():
    test_full_report_rows_bind_attribution_to_the_performance_window()


def test_strict_finance_contracts_export_through_package():
    import macro_framework as mf

    for name in (
        "AttributionKind",
        "BasketResidual",
        "MarketAttribution",
        "factor_returns_on",
        "portfolio_excess_returns",
        "differential_returns",
        "basket_residual",
        "raw_market_model_attribution",
        "market_attribution",
    ):
        assert name in mf.__all__
        assert hasattr(mf, name)


def test_deterministic():
    factors = _factor_frame()
    strat = 0.0004 + factors["XLK"] + RNG.normal(0, 0.0005, len(factors))
    assert basket_residual(strat, factors) == basket_residual(strat, factors)


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


def _ssr(p=0.01, value=0.14, alpha=0.05):
    return SSRInference(
        result=SSRResult(
            n_obs=500,
            n_rolling=250,
            sr_full=1.0,
            mean_rolling_sr=1.0,
            sigma_hac=0.4,
            L_hac=5,
            ssr=value,
        ),
        sr_star=0.0,
        p_value=p,
        block_len=5,
        n_boot=1000,
        seed=0,
        alpha=alpha,
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
    assert v.values["ssr"] == 0.14
    assert v.values["ssr_p"] == 0.01


def test_skill_gate_flip():
    v = evaluate_gates(_residual(t=1.5), _ssr(), **_PASS)
    assert v.passed is False
    assert v.skill_pass is False
    assert v.stability_pass and v.recall_pass and v.risk_shape_pass
    assert v.first_failure.startswith("skill:")


def test_stability_gate_flip():
    v = evaluate_gates(_residual(), _ssr(p=0.31), **_PASS)
    assert v.passed is False
    assert v.stability_pass is False
    assert v.first_failure.startswith("stability:")
    assert "0.31" in v.first_failure and "0.05" in v.first_failure


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
    cfg = GateConfig(skill_t_min=3.0, ssr_alpha=0.10)
    # the inference must be BUILT at the gate's alpha — the gate delegates to
    # SSRInference.stable rather than re-encoding the rule, so the two can no
    # longer disagree on the same object.
    ssr = _ssr(p=0.08, alpha=0.10)
    v = evaluate_gates(_residual(t=2.5), ssr, **_PASS, config=cfg)
    assert v.skill_pass is False  # 2.5 < 3.0
    assert v.stability_pass is True  # p=0.08 < alpha=0.10
    assert ssr.stable is True  # gate and verdict agree


def test_stability_gate_rejects_alpha_mismatch():
    """A gate alpha that disagrees with the inference's alpha is a caller error."""
    with pytest.raises(ValueError, match="ssr_alpha"):
        evaluate_gates(_residual(), _ssr(p=0.08, alpha=0.05), **_PASS,
                       config=GateConfig(ssr_alpha=0.10))


def test_verdict_distinguishes_negative_from_inconclusive():
    """A decisively negative sample must not render as 'luck-compatible'."""
    neg = SSRInference(
        result=SSRResult(n_obs=500, n_rolling=250, sr_full=-1.5, mean_rolling_sr=-1.5,
                         sigma_hac=0.4, L_hac=5, ssr=-0.30),
        sr_star=0.0, p_value=1.0, block_len=5, n_boot=1000, seed=0, alpha=0.05,
        p_value_lower=0.0,
    )
    assert neg.stable is False and neg.stably_below is True
    assert "stably BELOW" in neg.verdict()
    assert "luck-compatible" not in neg.verdict()
    # genuinely inconclusive: neither tail rejects
    incon = _ssr(p=0.40, value=0.02)
    incon = SSRInference(**{**incon.__dict__, "p_value_lower": 0.60})
    assert incon.stably_below is False
    assert "luck-compatible" in incon.verdict()


def test_shortened_attribution_is_identified_as_a_separate_window():
    # R3.7: coverage short of the performance end date -> performance_only row
    # plus a separate attribution record naming its actual start/end/count
    from macro_framework.reporting import build_attribution_record, build_reader_metric_row
    from tests.test_reporting import _line

    meta, _, metrics, cash, excess, ssr, attr, market = _line()
    port = metrics["returns"]
    short_attr = raw_market_model_attribution(port.iloc[:-30], market.iloc[:-30])

    reader = build_reader_metric_row(meta, metrics, cash, ssr, source="t", attribution=short_attr)
    record = build_attribution_record(meta, short_attr, source="t")

    assert reader["row_kind"] == "performance_only"
    assert (record["start"], record["end"], record["n_obs"]) == (
        short_attr.start, short_attr.end, short_attr.n_obs
    )


def test_ac_3_7():
    test_shortened_attribution_is_identified_as_a_separate_window()
