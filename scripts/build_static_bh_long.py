"""EXTENDED-TIMEFRAME static buy-and-hold lines — the datasets behind nb15_3.

Companion to ``build_static_bh.py``: same 4-ETF basket, same construction (its
``static_bh`` is imported, not re-implemented), only the windows differ. Two
additional lines are persisted so the extended-timeframe notebook stays fully
offline:

- ``data/static_bh_equity_2009_2026.parquet`` — the LONGEST line the data
  admits: bought 2009-09-25 (SWDA.L's inception, the binding constraint) and
  held. 16.7 years.
- ``data/static_bh_equity_2019_2026.parquet`` — the same basket bought FRESH at
  the trio's start, so the window ladder compares like with like.

Why a fresh buy per window instead of slicing one long line: buy-and-hold holds
SHARES, so a slice inherits weights that have already drifted for years and
silently answers a different question. Slicing the 2009 line to 2016 gives
CAGR 17.5% / maxDD -24.2%; buying fresh in 2016 gives 15.4% / -19.6%.
``data/static_bh_equity_2016_2026.parquet`` (built by ``build_static_bh.py``) is
already a fresh 2016 buy and is reused as the 10-year rung untouched.

Both windows end at a PINNED date, not at "today": these are release artifacts
that published figures are pinned against, and a moving end silently moves every
number in the tear sheet. Prices run further; the pin does not.

IN-SAMPLE CAVEAT (carried verbatim from ``build_static_bh.py``): the four ETFs
were selected by the Sharpe Stability Ratio computed over an overlapping window.
The line is a hindsight-selected static portfolio, never attainable skill — and
the extended window does not fix that, it only shows the same basket over more
history.

Prices: yfinance (auto-adjusted Close), the documented substitution for the
absent Postgres price DB. Reproducible: ``uv run python scripts/build_static_bh_long.py``.
Additive: writes only the artifacts listed above (gitignored via ``data/static_bh_*``
and ``data/csv_mirrors/``; shipped via the GH data release).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

from build_static_bh import static_bh  # noqa: E402  — one construction, imported

WINDOWS = {
    # tag -> (buy date, PINNED hold-to date)
    "2009_2026": ("2009-09-25", "2026-05-29"),  # SWDA.L inception -> release pin
    "2019_2026": ("2019-01-02", "2026-01-30"),  # trio start -> trio end
}
CSV_OUT = REPO / "data" / "csv_mirrors"


def main() -> None:
    spec = pd.read_parquet(REPO / "data" / "portfolio_ssr_top_per_category.parquet")
    weights = dict(zip(spec["symbol"], spec["weight"]))
    symbols = list(weights)
    print(f"[1/2] extended-timeframe static B&H of {weights} (yfinance substitution)")

    raw = yf.download(symbols, start="2009-09-01", end="2026-05-30",
                      auto_adjust=True, progress=False)["Close"]
    CSV_OUT.mkdir(parents=True, exist_ok=True)

    for tag, (start, end) in WINDOWS.items():
        equity, _ = static_bh(raw.loc[start:end][symbols], weights)
        equity.index.name = "Date"
        stem = f"static_bh_equity_{tag}"
        equity.to_frame().to_parquet(REPO / "data" / f"{stem}.parquet")

        # CSV mirrors, both locales — same shape as export_csv_mirrors.EQUITY_TABLES
        mirror = equity.to_frame()
        mirror["daily_return"] = equity.pct_change()
        mirror["drawdown"] = equity / equity.cummax() - 1
        mirror.to_csv(CSV_OUT / f"{stem}.csv", float_format="%.8f")
        mirror.to_csv(CSV_OUT / f"{stem}_de.csv", sep=";", decimal=",", float_format="%.8f")

        years = (equity.index[-1] - equity.index[0]).days / 365.25
        print(f"[2/2] {tag}: {equity.index[0]:%Y-%m-%d} -> {equity.index[-1]:%Y-%m-%d} "
              f"({years:.2f}y, {len(equity)} rows) total_return="
              f"{equity.iloc[-1] / equity.iloc[0] - 1:.4f} -> data/{stem}.parquet")


if __name__ == "__main__":
    main()
