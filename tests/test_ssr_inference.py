"""MBB inference on the SSR (paper §3.3.2-3.3.3, Test 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_framework.ssr import (
    TRADING_DAYS,
    politis_white_block_length,
    rolling_sharpe,
    ssr_inference,
)


def _series(mu: float, n: int = 1600, seed: int = 5) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, 0.01, n))


def test_deterministic_given_seed_with_complete_metadata():
    r = _series(0.0005)
    a = ssr_inference(r, n_boot=100, seed=3, alpha=0.10)
    b = ssr_inference(r, n_boot=100, seed=3, alpha=0.10)

    assert a == b
    assert a.window == TRADING_DAYS
    assert a.periods_per_year == TRADING_DAYS
    assert a.sr_star == 0.0
    assert a.n_boot == 100
    assert a.seed == 3
    assert a.alpha == 0.10
    assert a.block_len == 1
    assert a.p_value == pytest.approx(0.009900990099009901)
    assert a.p_value_lower == 1.0
    assert a.result.ssr == pytest.approx(0.19479300760422577)


def test_nondefault_window_metadata_keeps_fixed_daily_scaling():
    inf = ssr_inference(
        _series(0.001, n=134),
        window=126,
        sr_star=0.25,
        n_boot=50,
        seed=7,
        alpha=0.10,
    )

    assert inf.result.n_rolling == 9
    assert inf.window == 126
    assert inf.periods_per_year == TRADING_DAYS
    assert inf.sr_star == 0.25
    assert inf.n_boot == 50
    assert inf.seed == 7
    assert inf.alpha == 0.10


def test_rolling_sharpe_annualizes_on_252_even_for_nondefault_window():
    """Kill-check for sqrt(TRADING_DAYS) -> sqrt(window): with window=126 the
    single rolling value must scale on 252, hand-computed here from raw moments."""
    from macro_framework.ssr import _rolling_sharpe_np

    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.001, 0.01, 126))
    expected = float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))

    z = rolling_sharpe(r, window=126)
    assert len(z) == 1
    assert z.iloc[0] == pytest.approx(expected, rel=1e-12)

    np_z = _rolling_sharpe_np(r.to_numpy(dtype=float), 126)
    assert len(np_z) == 1
    assert np_z[0] == pytest.approx(expected, rel=1e-9)


def test_strong_drift_is_stable_and_noise_is_not():
    strong = ssr_inference(_series(0.0012), n_boot=200)  # ann. Sharpe ~1.9
    noise = ssr_inference(_series(0.0), n_boot=200)
    assert strong.stable and strong.p_value < 0.05
    assert not noise.stable and noise.p_value > 0.05
    assert "luck-compatible" in noise.verdict()
    assert "not a skill claim" in strong.verdict()


def test_negative_drift_never_passes():
    down = ssr_inference(_series(-0.0012), n_boot=200)
    assert not down.stable  # the old abs(SSR) rule would have branded this "stably > 0"


def test_valid_short_series_preserves_insufficient_inference():
    inf = ssr_inference(_series(0.001, n=TRADING_DAYS + 8), n_boot=50)

    assert inf.result.n_rolling == 9
    assert not inf.stable and np.isnan(inf.p_value) and np.isnan(inf.p_value_lower)
    assert inf.block_len == 0
    assert inf.window == TRADING_DAYS
    assert inf.periods_per_year == TRADING_DAYS
    assert inf.verdict() == "insufficient rolling observations for inference"


def test_exactly_ten_rolling_observations_runs_inference():
    inf = ssr_inference(_series(0.001, n=TRADING_DAYS + 9), n_boot=10)

    assert inf.result.n_rolling == 10
    assert inf.block_len > 0
    assert np.isfinite(inf.p_value) and np.isfinite(inf.p_value_lower)


def test_politis_white_bounds():
    rng = np.random.default_rng(0)
    white = rng.normal(size=2000)
    ar = np.empty(2000)
    ar[0] = 0.0
    for t in range(1, 2000):
        ar[t] = 0.9 * ar[t - 1] + rng.normal()
    b_white = politis_white_block_length(white)
    b_ar = politis_white_block_length(ar)
    assert 1 <= b_white <= 5  # white noise -> (near-)iid bootstrap
    assert b_ar > b_white  # persistence -> longer blocks
    assert b_ar <= int(np.ceil(min(3 * np.sqrt(2000), 2000 / 3)))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_return_values_are_rejected(value):
    returns = _series(0.0, n=300)
    returns.iloc[10] = value

    with pytest.raises(ValueError, match="returns values"):
        ssr_inference(returns, n_boot=10)


@pytest.mark.parametrize("values", [[True, False] * 150, [0.01 + 0.02j] * 300])
def test_non_real_return_values_are_rejected(values):
    returns = pd.Series(values, index=pd.bdate_range("2020-01-01", periods=300))

    with pytest.raises(ValueError, match="finite real numeric"):
        ssr_inference(returns, n_boot=10)


@pytest.mark.parametrize(
    "returns, message",
    [
        (pd.Series(dtype=float), "non-empty"),
        (pd.Series([0.0, 0.0, 0.0], index=[0, 1, 1]), "unique"),
        (pd.Series([0.0, 0.0, 0.0], index=[0, 2, 1]), "strictly increasing"),
        (
            pd.Series(
                [0.0, 0.0, 0.0],
                index=pd.date_range("2026-01-01", periods=3, tz="UTC"),
            ),
            "timezone-naive",
        ),
    ],
)
def test_invalid_return_indexes_are_rejected(returns, message):
    with pytest.raises(ValueError, match=message):
        ssr_inference(returns, window=2, n_boot=10)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1, np.nan, np.inf, True])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        ssr_inference(_series(0.0, n=300), alpha=alpha, n_boot=10)


@pytest.mark.parametrize("n_boot", [True, False, 0, -1, 1.5])
def test_invalid_bootstrap_count_is_rejected(n_boot):
    with pytest.raises(ValueError, match="n_boot"):
        ssr_inference(_series(0.0, n=300), n_boot=n_boot)


@pytest.mark.parametrize("window", [True, False, 1, 0, -1, 2.5, 301])
def test_invalid_window_is_rejected(window):
    with pytest.raises(ValueError, match="window"):
        ssr_inference(_series(0.0, n=300), window=window, n_boot=10)


@pytest.mark.parametrize("seed", [True, False, -1, 1.5, None])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        ssr_inference(_series(0.0, n=300), seed=seed, n_boot=10)


@pytest.mark.parametrize("sr_star", [np.nan, np.inf, -np.inf, True, "zero"])
def test_invalid_sharpe_benchmark_is_rejected(sr_star):
    with pytest.raises(ValueError, match="sr_star"):
        ssr_inference(_series(0.0, n=300), sr_star=sr_star, n_boot=10)


# --- Frozen coverage-matrix aliases (task 3.1): thin delegators onto the
#     semantic tests above, named exactly as coverage_matrix.json expects. --------

_INVALID_ALPHAS = (0.0, 1.0, -0.1, 1.1, np.nan, np.inf, True)
_INVALID_BOOTS = (True, False, 0, -1, 1.5)
_INVALID_WINDOWS = (True, False, 1, 0, -1, 2.5, 301)
_INVALID_SEEDS = (True, False, -1, 1.5, None)
_INVALID_SR_STARS = (np.nan, np.inf, -np.inf, True, "zero")


def test_invalid_inference_parameter_matrix():  # defect 14 shared boundary
    for alpha in _INVALID_ALPHAS:
        test_invalid_alpha_is_rejected(alpha)
    for n_boot in _INVALID_BOOTS:
        test_invalid_bootstrap_count_is_rejected(n_boot)
    for window in _INVALID_WINDOWS:
        test_invalid_window_is_rejected(window)
    for seed in _INVALID_SEEDS:
        test_invalid_seed_is_rejected(seed)
    for sr_star in _INVALID_SR_STARS:
        test_invalid_sharpe_benchmark_is_rejected(sr_star)


def test_ac_2_3():
    for alpha in _INVALID_ALPHAS:
        test_invalid_alpha_is_rejected(alpha)


def test_ac_2_4():
    for n_boot in _INVALID_BOOTS:
        test_invalid_bootstrap_count_is_rejected(n_boot)


def test_ac_2_5():
    for window in _INVALID_WINDOWS:
        test_invalid_window_is_rejected(window)


def test_ac_2_6():
    for sr_star in _INVALID_SR_STARS:
        test_invalid_sharpe_benchmark_is_rejected(sr_star)


def test_ac_2_8():
    test_deterministic_given_seed_with_complete_metadata()


def test_ac_8_4():
    # SSR suite coverage: excess-return construction is pinned at the
    # factor-loop boundary (test_ac_2_1); here determinism, metadata,
    # insufficiency, and invalid settings (R8.4)
    test_deterministic_given_seed_with_complete_metadata()
    test_nondefault_window_metadata_keeps_fixed_daily_scaling()
    test_valid_short_series_preserves_insufficient_inference()
    test_invalid_inference_parameter_matrix()
