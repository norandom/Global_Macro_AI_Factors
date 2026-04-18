"""Buy-and-hold backtests via vectorbt.

vectorbt is imported lazily inside each function — top-level import of
`macro_framework` must stay cheap (vectorbt's numba warm-up is ~10–30s).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # type-check only; never imports vectorbt at runtime unless a function is called
    import vectorbt as vbt


def buy_and_hold(
    prices: pd.DataFrame,
    weights: dict[str, float] | pd.Series,
    init_cash: float = 10_000.0,
    freq: str = "D",
) -> "vbt.Portfolio":
    """Grouped buy-and-hold: allocate `weights` on the first price date, never rebalance."""
    import vectorbt as vbt

    w = pd.Series(weights) if isinstance(weights, dict) else weights.copy()
    aligned = prices[w.index].dropna(how="any")
    if aligned.empty:
        raise ValueError("no overlapping price history across the requested symbols")

    size = pd.DataFrame(np.nan, index=aligned.index, columns=aligned.columns)
    size.iloc[0] = w.reindex(aligned.columns).values

    return vbt.Portfolio.from_orders(
        close=aligned,
        size=size,
        size_type="targetpercent",
        group_by=True,
        cash_sharing=True,
        init_cash=init_cash,
        freq=freq,
    )


def single_asset_buy_and_hold(
    prices: pd.Series | pd.DataFrame,
    init_cash: float = 10_000.0,
    freq: str = "D",
) -> "vbt.Portfolio":
    """Buy-and-hold a single asset (for benchmarks like SPY)."""
    import vectorbt as vbt

    series = prices.iloc[:, 0] if isinstance(prices, pd.DataFrame) else prices
    series = series.dropna()
    return vbt.Portfolio.from_holding(series, init_cash=init_cash, freq=freq)


def summary(
    portfolio: "vbt.Portfolio",
    benchmark: "vbt.Portfolio | None" = None,
    label: str = "portfolio",
    benchmark_label: str = "benchmark",
) -> pd.DataFrame:
    """Side-by-side key stats table."""
    def _row(pf: "vbt.Portfolio") -> dict[str, float]:
        return {
            "total_return": float(pf.total_return()),
            "annualized_return": float(pf.annualized_return()),
            "annualized_volatility": float(pf.annualized_volatility()),
            "sharpe_ratio": float(pf.sharpe_ratio()),
            "sortino_ratio": float(pf.sortino_ratio()),
            "calmar_ratio": float(pf.calmar_ratio()),
            "max_drawdown": float(pf.max_drawdown()),
        }

    cols: dict[str, dict[str, float]] = {label: _row(portfolio)}
    if benchmark is not None:
        cols[benchmark_label] = _row(benchmark)
    return pd.DataFrame(cols)
