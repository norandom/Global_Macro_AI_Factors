"""Correlation de-risk overlay: a walk-forward-fit, de-risk-only control leg.

Ported from the sibling facdrone `regime_overlays` (pure numpy/pandas): an EWMA
average pairwise correlation across the risky sleeve maps to a continuous
dampening scale in ``[min_scale, 1.0]``, and that scale is converted to a cash
(BIL) pin via ``bil_pin = 1 - scale * (1 - base_cash_pin)``.

The overlay only ever scales the risky sleeve *down* (higher measured
correlation -> lower scale -> higher cash pin); it never picks winners and never
lifts the risky sleeve above its no-overlay level. It is PIT-clean by
construction: every function reads only the return history passed to it, so
feeding a strictly-pre-rebalance window makes the signal walk-forward and use no
post-decision information. All functions are pure and deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_correlation_matrix(returns: pd.DataFrame, halflife: int = 10) -> pd.DataFrame:
    """EWMA-weighted correlation matrix from the most recent return window."""
    ewma_cov = returns.ewm(halflife=halflife, min_periods=halflife).cov()
    last_date = returns.index[-1]
    cov_block = ewma_cov.loc[last_date]

    std = np.sqrt(np.diag(cov_block.values))
    std_outer = np.outer(std, std)
    std_outer[std_outer == 0] = 1e-10
    corr = cov_block.values / std_outer
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)
    return pd.DataFrame(corr, index=cov_block.index, columns=cov_block.columns)


def avg_pairwise_correlation(returns: pd.DataFrame, halflife: int = 10) -> float:
    """Mean off-diagonal EWMA pairwise correlation across the sleeve.

    Returns 0.0 (no stress) for the degenerate case of <2 columns or fewer than
    ``halflife`` rows, so the overlay cannot de-risk on an unfit signal.
    """
    if len(returns.columns) < 2 or len(returns) < halflife:
        return 0.0
    corr = ewma_correlation_matrix(returns, halflife=halflife)
    n = len(corr)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pairwise = corr.values[mask]
    if len(pairwise) == 0:
        return 0.0
    return float(np.mean(pairwise))


def correlation_scale(
    returns: pd.DataFrame,
    *,
    halflife: int = 10,
    normal_corr: float = 0.30,
    crisis_corr: float = 0.70,
    min_scale: float = 0.20,
) -> float:
    """Continuous de-risk scale in ``[min_scale, 1.0]``.

    ``1.0`` when avg pairwise correlation <= ``normal_corr`` (calm), ``min_scale``
    when >= ``crisis_corr`` (crisis), linear in between. Higher correlation ->
    lower scale -> smaller risky sleeve.
    """
    avg_corr = avg_pairwise_correlation(returns, halflife=halflife)
    if avg_corr <= normal_corr:
        return 1.0
    if avg_corr >= crisis_corr:
        return min_scale
    frac = (avg_corr - normal_corr) / (crisis_corr - normal_corr)
    return 1.0 - frac * (1.0 - min_scale)


def derisk_cash_pin(
    returns_hist: pd.DataFrame,
    *,
    base_risky_symbols: tuple[str, ...],
    base_cash_pin: float = 0.25,
    min_scale: float = 0.20,
) -> float:
    """Map the correlation stress signal to a cash (BIL) pin.

    Computes ``correlation_scale`` over only the ``base_risky_symbols`` columns of
    ``returns_hist`` (PIT-clean: reads only the passed history), then
    ``bil_pin = 1 - scale * (1 - base_cash_pin)``.

    Guaranteed monotonically non-decreasing in measured correlation and bounded in
    ``[base_cash_pin, 1)``: at scale=1.0 (calm) the pin equals ``base_cash_pin``;
    at scale=``min_scale`` (crisis) it is raised to ``1 - min_scale*(1-base_cash_pin)``,
    still strictly below 1 so ``hrp_cvar_weights_with_fixed``'s sum<1 guard holds.
    This only raises cash in stress and never lifts risky above ``1 - base_cash_pin``.
    """
    risky = [s for s in base_risky_symbols if s in returns_hist.columns]
    scale = correlation_scale(returns_hist[risky], min_scale=min_scale)
    return 1.0 - scale * (1.0 - base_cash_pin)
