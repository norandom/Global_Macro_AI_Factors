"""Walk-forward orchestrator — builds target-weight schedules with strict no-lookahead slicing.

A strategy fn receives only data with dates **strictly before** the rebalance date.
That makes lookahead structurally impossible regardless of what the fn computes internally.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .returns import daily_returns

StrategyContext = dict[str, Any]
WeightFn = Callable[[StrategyContext], pd.Series]


def monthly_rebalance_dates(
    prices: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """First trading day of each calendar month in [start, end]."""
    idx = prices.index
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end is not None else idx.max()
    in_range = idx[(idx >= start_ts) & (idx <= end_ts)]
    month_keys = pd.DataFrame({"d": in_range, "k": in_range.to_period("M")})
    first_per_month = month_keys.groupby("k", sort=True)["d"].min()
    return pd.DatetimeIndex(first_per_month.values)


def build_walk_forward_targets(
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    weight_fns: dict[str, WeightFn],
    macro_panel: pd.DataFrame | None = None,
    lookback_days: int = 756,
    min_history: int = 60,
) -> dict[str, pd.DataFrame]:
    """Build a target-weight DataFrame per strategy.

    Each `weight_fn` gets a context dict:
      - rebalance_date: pd.Timestamp
      - prices:         DataFrame with rows strictly before rebalance_date (tail of lookback_days)
      - returns:        daily returns aligned to `prices`
      - macro_panel:    rows strictly before rebalance_date (if provided)
    and must return a Series of target weights summing to 1 over `prices.columns`.
    """
    all_returns = daily_returns(prices)
    targets = {name: pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
               for name in weight_fns}

    for rb in rebalance_dates:
        price_hist = prices.loc[prices.index < rb].tail(lookback_days)
        ret_hist = all_returns.loc[all_returns.index < rb].tail(lookback_days).dropna(how="any")
        if len(price_hist) < min_history or len(ret_hist) < min_history:
            continue
        macro_hist = None
        if macro_panel is not None:
            macro_hist = macro_panel.loc[macro_panel.index < rb]

        ctx: StrategyContext = {
            "rebalance_date": rb,
            "prices":         price_hist,
            "returns":        ret_hist,
            "macro_panel":    macro_hist,
        }

        for name, fn in weight_fns.items():
            try:
                w = fn(ctx)
            except Exception as exc:  # noqa: BLE001
                print(f"[{name} @ {rb.date()}] weight_fn failed: {exc!r}; holding previous")
                continue
            if w.sum() == 0:
                continue
            w = w.reindex(prices.columns).fillna(0.0)
            targets[name].loc[rb, :] = w.values

    return targets
