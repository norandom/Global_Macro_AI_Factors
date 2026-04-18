from __future__ import annotations

import pandas as pd


def daily_returns(prices: pd.DataFrame, rf_annual: float = 0.0) -> pd.DataFrame:
    """Arithmetic daily excess returns from a wide price frame.

    rf_annual is a constant annualized risk-free rate; converted to daily via rf_annual/252.
    Default rf=0 treats raw returns as excess (common when comparing ETFs apples-to-apples).
    """
    rets = prices.pct_change()
    if rf_annual:
        rets = rets - rf_annual / 252.0
    return rets.dropna(how="all")
