"""Long 4-ETF close panel for the Markowitz risk-return planes (nb18_3).

The offline cache ``data/etf_prices_wide_2013_2026.parquet`` starts 2013-01-02, so the
max-timeframe frontier has no opportunity set before then. This persists the same
basket ``scripts/build_static_bh_long.py`` trades — SWDA.L / XLK / IAU / BIL — back to
SWDA.L's inception, from the identical source (yfinance auto-adjusted Close), so the
frontier and the buy-and-hold line it is plotted against share one price basis.

Window end is PINNED: yfinance keeps extending, and an unpinned end would silently move
every published figure on the next run.

Reproducible: ``uv run python scripts/build_basket_long.py``. Gitignored, shipped via
the GH data release beside its ``spy_close_2009_2026`` sibling.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

REPO = Path(__file__).resolve().parent.parent
BASKET = ["SWDA.L", "XLK", "IAU", "BIL"]
START, END = "2009-09-01", "2026-05-30"  # SWDA.L lists 2009-09-25; end pinned
OUT = REPO / "data" / "basket_close_2009_2026.parquet"


def main() -> None:
    raw = yf.download(BASKET, start=START, end=END, auto_adjust=True, progress=False)["Close"]
    px = raw[BASKET].dropna(how="any")
    px.index = pd.to_datetime(px.index).tz_localize(None)
    px.index.name = "Date"

    first = {s: raw[s].first_valid_index() for s in BASKET}
    binding = max(first, key=lambda s: first[s])
    assert px.index.min() == max(first.values()), "common span must start at the last inception"

    px.to_parquet(OUT)
    print(f"{OUT.name}: {px.index.min():%Y-%m-%d} -> {px.index.max():%Y-%m-%d} "
          f"({(px.index.max() - px.index.min()).days / 365.25:.2f}y, {len(px)} rows)")
    print(f"binding constraint: {binding} inception {first[binding]:%Y-%m-%d}")
    for s in BASKET:
        print(f"  {s:8s} first {first[s]:%Y-%m-%d}")


if __name__ == "__main__":
    main()
