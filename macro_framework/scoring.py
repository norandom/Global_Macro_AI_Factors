from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from .etf import coverage_summary, get_prices
from .returns import daily_returns
from .ssr import TRADING_DAYS, compute_ssr


def score_universe(
    start: str,
    end: str,
    window: int = TRADING_DAYS,
    sr_star: float = 0.0,
    rf_annual: float = 0.0,
) -> pd.DataFrame:
    """Score every ETF whose history covers [start, end] by SSR; return 0..100 percentile scores.

    Columns: rank, symbol, name, category, ssr, score, mean_rolling_sr, sigma_hac,
             sr_full, n_obs, n_rolling, L_hac
    """
    start_ts = pd.Timestamp(start).date()
    cov = coverage_summary()
    eligible = cov.loc[cov["first_date"] <= start_ts].copy()
    symbols = eligible["symbol"].tolist()
    if not symbols:
        raise ValueError(f"no ETFs with first_date <= {start}")

    prices = get_prices(symbols, start=start, end=end)
    rets = daily_returns(prices, rf_annual=rf_annual)

    rows: list[dict] = []
    for sym in symbols:
        if sym not in rets.columns:
            continue
        r = rets[sym].dropna()
        if len(r) < window * 2:
            continue
        res = compute_ssr(r, window=window, sr_star=sr_star)
        rows.append({"symbol": sym, **asdict(res)})

    df = (
        pd.DataFrame(rows)
        .merge(eligible[["symbol", "name", "category"]], on="symbol", how="left")
        .dropna(subset=["ssr"])
        .reset_index(drop=True)
    )
    df["score"] = df["ssr"].rank(method="average", pct=True) * 100.0
    df = df.sort_values("ssr", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    cols = [
        "rank", "symbol", "name", "category",
        "ssr", "score",
        "mean_rolling_sr", "sigma_hac", "sr_full",
        "n_obs", "n_rolling", "L_hac",
    ]
    return df[cols]


def select_top_per_category(
    scored: pd.DataFrame,
    categories: list[str],
    n_per_category: int = 1,
) -> pd.DataFrame:
    """Pick the top `n_per_category` ETFs (by SSR) from each requested category and
    attach equal weights summing to 1. `scored` must be sorted descending by SSR.
    """
    picks: list[pd.DataFrame] = []
    for cat in categories:
        sub = scored.loc[scored["category"] == cat].head(n_per_category)
        if len(sub) < n_per_category:
            raise ValueError(f"category {cat!r}: only {len(sub)} candidate(s), need {n_per_category}")
        picks.append(sub)
    sel = pd.concat(picks, ignore_index=True)
    sel["weight"] = 1.0 / len(sel)
    return sel
