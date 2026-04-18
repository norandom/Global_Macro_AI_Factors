"""Annual rebalancing: build target-weight DataFrame + vbt simulation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .returns import daily_returns

if TYPE_CHECKING:  # pragma: no cover
    import vectorbt as vbt

WeightFn = Callable[[pd.DataFrame], pd.Series]


def annual_rebalance_dates(prices: pd.DataFrame, start: str | pd.Timestamp) -> pd.DatetimeIndex:
    """First trading day on/after `start`, then the first trading day of each subsequent year."""
    idx = prices.index
    start_ts = pd.Timestamp(start)
    first = idx[idx >= start_ts][0]
    later_years = sorted({d.year for d in idx if d.year > first.year})
    per_year = [idx[idx.year == y][0] for y in later_years]
    return pd.DatetimeIndex([first, *per_year])


def build_target_weights(
    prices: pd.DataFrame,
    weight_fn: WeightFn,
    rebalance_dates: pd.DatetimeIndex,
    lookback_days: int = 756,
) -> pd.DataFrame:
    """Return a DataFrame aligned to `prices.index` where rows on `rebalance_dates`
    hold target weights computed by `weight_fn` on the prior `lookback_days` of
    returns. All other rows are NaN (vectorbt: no order)."""
    all_returns = daily_returns(prices)
    target = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    for rb in rebalance_dates:
        hist = all_returns.loc[all_returns.index < rb].tail(lookback_days)
        if len(hist) < max(60, lookback_days // 4):
            continue  # not enough history
        w = weight_fn(hist.dropna(how="any"))
        target.loc[rb, w.index] = w.values

    return target


def run_rebalance_sim(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    init_cash: float = 10_000.0,
    freq: str = "D",
) -> "vbt.Portfolio":
    """vectorbt grouped portfolio rebalancing to the target weights on each non-NaN row."""
    import vectorbt as vbt

    aligned_prices = prices[target_weights.columns].dropna(how="any").copy()
    aligned_prices.index = pd.DatetimeIndex(aligned_prices.index.astype("datetime64[ns]"))
    aligned_targets = target_weights.copy()
    aligned_targets.index = pd.DatetimeIndex(aligned_targets.index.astype("datetime64[ns]"))
    aligned_targets = aligned_targets.loc[aligned_prices.index]
    return vbt.Portfolio.from_orders(
        close=aligned_prices,
        size=aligned_targets,
        size_type="targetpercent",
        group_by=True,
        cash_sharing=True,
        init_cash=init_cash,
        freq=freq,
    )
