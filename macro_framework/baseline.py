"""Baseline (Control) strategy — HRP-CVaR + 12m momentum filter, no macro overlay.

Weight logic per rebalance date:
  1. 12-month momentum = price[t] / price[t - 252 trading days] - 1
  2. Eligible assets = those with positive 12m momentum
  3. If none eligible: 100% into BIL (cash fallback)
  4. Otherwise: HRP-CVaR on eligible assets using trailing 3y daily returns
"""

from __future__ import annotations

import pandas as pd

from .allocation import hrp_cvar_weights

CASH_TICKER = "BIL"
MOMENTUM_LOOKBACK = 252  # trading days


def hrp_momentum_weights(
    returns: pd.DataFrame,
    prices_window: pd.DataFrame,
    mom_lookback: int = MOMENTUM_LOOKBACK,
    cash_ticker: str = CASH_TICKER,
) -> pd.Series:
    """HRP-CVaR on positive-momentum subset; full cash if nothing qualifies.

    `returns` must be daily returns aligned to `prices_window.index`. Both must
    end on (or before) the rebalance date.
    """
    if len(prices_window) < mom_lookback + 1:
        raise ValueError(f"prices_window has {len(prices_window)} rows; need ≥ {mom_lookback + 1}")

    latest = prices_window.iloc[-1]
    past = prices_window.iloc[-(mom_lookback + 1)]
    momentum = latest / past - 1.0
    eligible = [c for c in prices_window.columns if momentum[c] > 0]

    weights = pd.Series(0.0, index=prices_window.columns)

    if not eligible:
        if cash_ticker in weights.index:
            weights[cash_ticker] = 1.0
        return weights

    if len(eligible) == 1:
        weights[eligible[0]] = 1.0
        return weights

    rets_elig = returns[eligible].dropna(how="any")
    if rets_elig.empty or len(rets_elig) < 60:
        if cash_ticker in weights.index:
            weights[cash_ticker] = 1.0
        return weights

    w_hrp = hrp_cvar_weights(rets_elig)
    weights.loc[w_hrp.index] = w_hrp.values
    return weights
