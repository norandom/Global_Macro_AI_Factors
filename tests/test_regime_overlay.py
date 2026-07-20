"""Tests for macro_framework.regime_overlay — correlation de-risk scale + cash pin.

Drives Requirement 6 (6.1-6.4) and 8.2: monotone de-risk-only cash pin, bounded
in [base_cash_pin, 1), and PIT-clean (only the passed window matters).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_framework.regime_overlay import (
    avg_pairwise_correlation,
    correlation_scale,
    derisk_cash_pin,
    ewma_correlation_matrix,
)

RNG = np.random.default_rng(20260720)
RISKY = ("SWDA.L", "XLK", "IAU")


def _corr_frame(rho: float, n: int = 250) -> pd.DataFrame:
    """Risky-sleeve returns with a target pairwise correlation `rho`.

    Each column = shared common factor (weight sqrt(rho)) + idio (weight sqrt(1-rho)),
    so every off-diagonal population correlation is ~rho.
    """
    idx = pd.bdate_range("2020-01-01", periods=n)
    common = RNG.normal(0.0, 0.01, n)
    cols = {}
    for c in RISKY:
        idio = RNG.normal(0.0, 0.01, n)
        cols[c] = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idio
    return pd.DataFrame(cols, index=idx)


def test_calm_market_scale_is_one_and_pin_is_base():
    calm = _corr_frame(0.05)
    assert correlation_scale(calm) == pytest.approx(1.0)
    pin = derisk_cash_pin(calm, base_risky_symbols=RISKY, base_cash_pin=0.25)
    assert pin == pytest.approx(0.25)


def test_crisis_market_scale_near_min_and_pin_raised():
    crisis = _corr_frame(0.90)
    scale = correlation_scale(crisis, min_scale=0.20)
    assert scale == pytest.approx(0.20, abs=1e-9)
    pin = derisk_cash_pin(crisis, base_risky_symbols=RISKY, base_cash_pin=0.25, min_scale=0.20)
    # bil_pin = 1 - 0.20 * (1 - 0.25) = 0.85
    assert pin == pytest.approx(0.85, abs=1e-9)
    assert pin < 1.0


def test_pin_monotone_non_decreasing_in_correlation():
    pins = [
        derisk_cash_pin(_corr_frame(rho), base_risky_symbols=RISKY)
        for rho in (0.05, 0.30, 0.50, 0.70, 0.90)
    ]
    assert pins == sorted(pins)
    assert pins[0] < pins[-1]  # actually moves


def test_pin_bounded_and_never_lifts_risky_above_no_overlay():
    for rho in (0.0, 0.20, 0.45, 0.65, 0.85, 0.99):
        pin = derisk_cash_pin(_corr_frame(rho), base_risky_symbols=RISKY, base_cash_pin=0.25)
        assert 0.25 <= pin < 1.0
        # risky sleeve = 1 - pin, never above no-overlay level (1 - base_cash_pin)
        assert (1.0 - pin) <= (1.0 - 0.25) + 1e-12


def test_pit_clean_future_rows_do_not_change_pin():
    hist = _corr_frame(0.60, n=200)
    pin_now = derisk_cash_pin(hist, base_risky_symbols=RISKY)
    # append future rows to a copy and re-slice to the same pre-date window
    future = _corr_frame(0.05, n=50)
    future.index = pd.bdate_range(hist.index[-1] + pd.offsets.BDay(1), periods=50)
    extended = pd.concat([hist, future])
    pin_windowed = derisk_cash_pin(extended.loc[: hist.index[-1]], base_risky_symbols=RISKY)
    assert pin_now == pytest.approx(pin_windowed, abs=1e-12)


def test_degenerate_short_window_no_derisk():
    short = _corr_frame(0.90, n=5)  # fewer than halflife rows
    assert correlation_scale(short) == pytest.approx(1.0)
    assert derisk_cash_pin(short, base_risky_symbols=RISKY) == pytest.approx(0.25)


def test_degenerate_single_risky_column_no_derisk():
    one = _corr_frame(0.90)[[RISKY[0]]]
    assert avg_pairwise_correlation(one) == 0.0
    assert derisk_cash_pin(one, base_risky_symbols=(RISKY[0],)) == pytest.approx(0.25)


def test_uses_only_base_risky_symbols_columns():
    # A calm risky sleeve with an extra highly self-correlated non-risky column
    # present in the frame must NOT trigger de-risk.
    calm = _corr_frame(0.05)
    calm["BIL"] = calm["XLK"]  # perfectly correlated noise column, but not risky
    pin = derisk_cash_pin(calm, base_risky_symbols=RISKY, base_cash_pin=0.25)
    assert pin == pytest.approx(0.25)


def test_ewma_correlation_matrix_shape_and_diagonal():
    frame = _corr_frame(0.40)
    corr = ewma_correlation_matrix(frame)
    assert list(corr.columns) == list(RISKY)
    assert np.allclose(np.diag(corr.values), 1.0)
    assert (corr.values <= 1.0 + 1e-9).all() and (corr.values >= -1.0 - 1e-9).all()
