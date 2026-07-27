"""Total-return (dividend-adjusted) close panel for appendix D.

Appendix D ranked the most-traded ETFs by SSR off ``etf_prices.close`` — RAW closes,
no dividend adjustment. That is fine for QQQ/SPY/IVV/DIA, whose return is mostly capital
appreciation, but it is structurally unfair to the income-heavy funds in the same table:
HYG and TLT earn essentially all of their return as coupon, and EFA/EEM/XLE/EWZ are
high-dividend. On a price-only series their main return source is deleted by construction,
so a "luck-compatible" verdict for them measured the wrong series rather than the fund.

This persists yfinance auto-adjusted closes (dividends and splits reinvested) for the
liquid universe, so the notebook can compute total returns offline.

Universe: top ``N_FETCH`` by median daily dollar volume from ``etf_prices`` — a superset of
the notebook's ``TOP_N``, so its coverage floor can still bind without a missing symbol.
Volume ranking stays on the DB (dollar volume needs raw close x volume, not an adjusted one).

Window END is PINNED to the DB feed's last date so the two sources describe the same span
and a later yfinance run cannot silently extend every published figure.

Reproducible: ``uv run python scripts/build_appendix_d_total_return.py``.
Gitignored; shipped via the GH data release.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402
from sqlalchemy import text  # noqa: E402

from macro_framework.db import get_engine  # noqa: E402

N_FETCH = 25
START, END = "2009-12-01", "2026-05-30"  # end pinned to the etf_prices feed's last session
OUT = REPO / "data" / "appendix_d_total_return_2010_2026.parquet"


def main() -> None:
    with get_engine().connect() as conn:
        px = pd.read_sql(text("SELECT symbol, date, close, volume FROM etf_prices"),
                         conn, parse_dates=["date"])
    px["dollar_vol"] = px["close"] * px["volume"]
    ranked = px.groupby("symbol")["dollar_vol"].median().sort_values(ascending=False)
    symbols = list(ranked.index[:N_FETCH])
    db_end = px["date"].max()
    print(f"fetching {len(symbols)} symbols by median $vol/day; db feed ends {db_end:%Y-%m-%d}")

    raw = yf.download(symbols, start=START, end=END, auto_adjust=True, progress=False)["Close"]
    tr = raw[symbols]
    tr.index = pd.to_datetime(tr.index).tz_localize(None)
    tr.index.name = "date"
    tr = tr.loc[:db_end]

    assert tr.index.is_unique and tr.index.is_monotonic_increasing
    tr.to_parquet(OUT)
    print(f"{OUT.name}: {tr.index.min():%Y-%m-%d} -> {tr.index.max():%Y-%m-%d} "
          f"({len(tr)} rows, {tr.shape[1]} symbols)")

    # The adjustment is the whole point — show it landed. On a total-return series a
    # coupon-paying fund's cumulative growth must exceed its raw-price growth.
    with get_engine().connect() as conn:
        rawdb = pd.read_sql(text("SELECT symbol, date, close FROM etf_prices "
                                 "WHERE symbol IN ('HYG','TLT','SPY')"),
                            conn, parse_dates=["date"]).pivot(
                                index="date", columns="symbol", values="close")
    print("\ncumulative growth over the common span, raw close vs total return:")
    for s in ["HYG", "TLT", "SPY"]:
        if s not in tr.columns:
            continue
        i = tr[s].dropna().index.intersection(rawdb[s].dropna().index)
        g_tr = tr[s].loc[i].iloc[-1] / tr[s].loc[i].iloc[0] - 1
        g_raw = rawdb[s].loc[i].iloc[-1] / rawdb[s].loc[i].iloc[0] - 1
        print(f"  {s:4s} raw {g_raw:+7.1%}   total-return {g_tr:+7.1%}   "
              f"income contribution {(g_tr - g_raw) * 100:+.1f}pp")


if __name__ == "__main__":
    main()
