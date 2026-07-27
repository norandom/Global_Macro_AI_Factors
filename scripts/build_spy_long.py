"""LONG SPY reference line — the regression benchmark nb18_2's window ladder needs.

``data/etf_prices_wide_2013_2026.parquet`` starts 2013-01-02, so the SPY-regression
appraisal ratio cannot be computed on the 2009 rungs of the static buy-and-hold
window ladder. This persists the one missing series:

- ``data/spy_close_2009_2026.parquet`` — Date-indexed ``SPY`` close, 2009-09-25
  (SWDA.L's inception, the ladder's earliest buy) → 2026-05-29 (the release pin
  ``build_static_bh_long.py`` uses).

SPLICED, not re-downloaded. From 2013-01-02 onward the series is the published
``etf_prices_wide_2013_2026.parquet`` column VERBATIM, so nb18_2's rungs regress
against byte-identically the same benchmark the trio in nb15_2/nb18 does; only
2009-09-25 → 2012-12-31 is fetched. A fresh full pull would silently re-adjust the
overlap (yfinance had already revised two 2026-04 closes by build time) and put
the two dashboards on different SPY.

PRICE-ONLY, deliberately. The published artifact is yfinance ``auto_adjust=False``
Close — dividends missing — verified here to the cent over the 3371 shared rows.
The prepend uses the same basis, and the splice asserts a ~1.0 price ratio across
the junction month (a split would show up as a step). Carrying the repo's caveat
forward: with rf=0 and a price-only benchmark, the regression intercept is an
internal same-source comparison, not an excess-return CAPM/Jensen alpha.

Reproducible: ``uv run python scripts/build_spy_long.py``. Additive: writes only
the artifact above (+ ``data/csv_mirrors/spy_close_2009_2026{,_de}.csv``).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

REPO = Path(__file__).resolve().parent.parent
START = "2009-09-25"          # ladder's earliest buy (SWDA.L inception)
CSV_OUT = REPO / "data" / "csv_mirrors"


def main() -> None:
    published = pd.read_parquet(REPO / "data" / "etf_prices_wide_2013_2026.parquet")["SPY"].dropna()
    splice_at = published.index.min()
    print(f"[1/3] published SPY {splice_at:%Y-%m-%d} → {published.index.max():%Y-%m-%d} "
          f"({len(published)} rows) kept verbatim")

    # price-only Close, matching the published artifact's basis; one month of overlap
    # is fetched purely to verify the splice, then dropped.
    raw = yf.download("SPY", start=START, end="2013-02-01", auto_adjust=False,
                      progress=False)["Close"]["SPY"].dropna()
    overlap = raw.index.intersection(published.index)
    ratio = (raw.loc[overlap] / published.loc[overlap])
    assert len(overlap) >= 15, f"too little overlap to verify the splice: {len(overlap)}"
    assert ((ratio - 1.0).abs() < 5e-4).all(), (
        f"splice basis mismatch (split or adjustment change): ratio range "
        f"{ratio.min():.6f}–{ratio.max():.6f}")
    print(f"[2/3] splice verified on {len(overlap)} shared rows, "
          f"price ratio {ratio.min():.6f}–{ratio.max():.6f}")

    px = pd.concat([raw.loc[raw.index < splice_at], published]).rename("SPY")
    px.index.name = "Date"
    assert px.index.is_monotonic_increasing and not px.index.has_duplicates

    out = REPO / "data" / "spy_close_2009_2026.parquet"
    px.to_frame().to_parquet(out)
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    px.to_frame().to_csv(CSV_OUT / "spy_close_2009_2026.csv", float_format="%.8f")
    px.to_frame().to_csv(CSV_OUT / "spy_close_2009_2026_de.csv", sep=";", decimal=",",
                         float_format="%.8f")
    print(f"[3/3] {px.index[0]:%Y-%m-%d} → {px.index[-1]:%Y-%m-%d} ({len(px)} rows, "
          f"{(px.index < splice_at).sum()} prepended) -> data/{out.name}")


if __name__ == "__main__":
    main()
