#!/usr/bin/env python
"""Vet the FMP-sourced ``etf_prices`` table for fat fingers, jitter, stale runs
and structural OHLC errors. Read-only: prints per-symbol and dataset error
percentages. Optionally writes a repaired copy (mask + forward-fill, PIT-safe).

Detectors (each flag marks a bad CLOSE row, so percentages are rows-affected):
  structural   high<low, close/open outside [low,high], non-positive prices
  fat_finger   idiosyncratic |robust z| > K_MAD move >= FAT_PCT, >=60% undone next bar
  jitter       same as fat_finger but smaller than FAT_PCT (bad-tick noise)
  level_shift  huge non-reversed jump (likely missed split/adjustment — do NOT delete)
  stale        run of >= STALE_RUN identical closes with zero volume (dead feed)

Real market-wide moves (e.g. 2020-03) are protected twice: a spike must revert
next bar AND must not occur on a day when >= MARKET_FRAC of the universe also
moved |z|>3.

Usage:
  uv run python scripts/vet_prices.py                    # QC report from the DB
  uv run python scripts/vet_prices.py --write-clean data/etf_prices_clean.parquet
  uv run python scripts/vet_prices.py --selfcheck        # synthetic test, no DB
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# --- thresholds ------------------------------------------------------------ #
K_MAD = 6.0                 # robust z on daily log returns to call a move suspect
REVERSAL = 0.6              # next bar must undo >= this fraction of the move
FAT_PCT = 0.05              # reversed move >= 5% -> fat_finger, smaller -> jitter
RET_FLOOR = 0.002           # ignore sub-20bp moves (penny ticks on low-vol funds)
STALE_RUN = 5               # >= N consecutive identical closes with zero volume
LEVEL_SHIFT = np.log(1.8)   # non-reversed one-day jump this big -> missed split?
MARKET_FRAC = 0.5           # day is a market event if this share of universe |z|>3
ROLL = 252                  # trailing window for rolling median/MAD
MIN_OBS = 60                # min history before z is defined (early rows unflagged)
TOL = 0.01                  # close/open outside [low,high] by more -> hard error;
                            # by less -> ohlc_quirk (thin-LSE auction-print pattern)

FLAG_COLS = ["structural", "fat_finger", "jitter", "level_shift", "stale"]


def flag_symbol(g: pd.DataFrame) -> pd.DataFrame:
    """Boolean flag columns + robust z for one symbol's date-indexed OHLCV."""
    o, h, l, c, v = (g[k] for k in ("open", "high", "low", "close", "volume"))
    out = pd.DataFrame(index=g.index)
    # worst relative excursion of close/open outside [low, high]
    breach = pd.concat([c / h - 1, 1 - c / l, o / h - 1, 1 - o / l], axis=1).max(axis=1).clip(lower=0)
    close_breach = pd.concat([c / h - 1, 1 - c / l], axis=1).max(axis=1).clip(lower=0)
    nonpos = (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)
    out["structural"] = (h < l) | nonpos | (breach > TOL)
    out["ohlc_quirk"] = (breach > 0) & ~out["structural"]

    r = np.log(c.where(c > 0)).diff()
    med = r.rolling(ROLL, min_periods=MIN_OBS).median()
    mad = (r - med).abs().rolling(ROLL, min_periods=MIN_OBS).median()
    z = (r - med) / (1.4826 * mad.where(mad > 0))
    nxt = r.shift(-1)
    undone = (np.sign(nxt) == -np.sign(r)) & (nxt.abs() >= REVERSAL * r.abs())
    suspect = (z.abs() > K_MAD) & (r.abs() >= RET_FLOOR)
    out["fat_finger"] = suspect & undone & (r.abs() >= np.log1p(FAT_PCT))
    out["jitter"] = suspect & undone & ~out["fat_finger"]
    out["level_shift"] = (r.abs() >= LEVEL_SHIFT) & ~undone

    flat = c.eq(c.shift()) & v.fillna(0).eq(0)
    run_len = flat.groupby((~flat).cumsum()).transform("sum")
    out["stale"] = flat & (run_len >= STALE_RUN)
    out["z"] = z
    # close itself provably wrong (used by clean_prices; open-only breaches spared)
    out["bad_close"] = (c <= 0) | (h < l) | (close_breach > TOL)
    return out


def run_qc(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """prices: long OHLCV (symbol, date, open, high, low, close, volume).

    Returns (flags long-frame aligned to prices, per-symbol report with a TOTAL row).
    """
    prices = prices.sort_values(["symbol", "date"]).reset_index(drop=True)
    flags = (
        prices.groupby("symbol", group_keys=False)[
            ["open", "high", "low", "close", "volume"]
        ]
        .apply(flag_symbol)
        .reset_index(drop=True)
    )
    flags[["symbol", "date"]] = prices[["symbol", "date"]]

    # market-wide guard: don't call it a fat finger if half the universe moved too
    zwide = flags.pivot(index="date", columns="symbol", values="z")
    if zwide.shape[1] >= 5:
        frac = zwide.abs().gt(3).sum(axis=1) / zwide.notna().sum(axis=1).clip(lower=1)
        event_dates = frac.index[(frac >= MARKET_FRAC) & (zwide.notna().sum(axis=1) >= 5)]
        on_event = flags["date"].isin(event_dates)
        flags.loc[on_event, ["fat_finger", "jitter"]] = False

    # "any" = data errors only; ohlc_quirk is a benign feed characteristic
    flags["any"] = flags[FLAG_COLS].any(axis=1)
    return flags, summarize(flags)


def summarize(flags: pd.DataFrame) -> pd.DataFrame:
    rep = flags.groupby("symbol")[FLAG_COLS + ["ohlc_quirk", "any"]].mean().mul(100)
    rep.insert(0, "n_rows", flags.groupby("symbol").size())
    total = rep.drop(columns="n_rows").mul(rep["n_rows"], axis=0).sum() / rep["n_rows"].sum()
    rep.loc["TOTAL"] = [rep["n_rows"].sum(), *total]
    return rep


def clean_prices(prices: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Mask confirmed-bad closes to NaN and forward-fill per symbol (PIT-safe:
    no future information is used). level_shift and stale rows are left alone —
    a missed split needs back-adjustment, not deletion."""
    bad = flags[["bad_close", "fat_finger", "jitter"]].any(axis=1).to_numpy()
    out = prices.sort_values(["symbol", "date"]).reset_index(drop=True).copy()
    out.loc[bad, ["open", "high", "low", "close"]] = np.nan
    out["close"] = out.groupby("symbol")["close"].ffill()
    return out


def report(flags: pd.DataFrame, rep: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    print("\n=== error rate, % of rows flagged (10y FMP etf_prices) ===")
    print(rep.round(3).to_string())
    worst = flags.loc[flags["any"] & flags["z"].notna()].copy()
    worst["absz"] = worst["z"].abs()
    worst = worst.sort_values("absz", ascending=False).head(20)
    if len(worst):
        print("\n=== worst 20 flagged rows ===")
        cols = ["symbol", "date", "z"] + FLAG_COLS + ["ohlc_quirk"]
        print(worst[cols].to_string(index=False))
    ls = int(flags["level_shift"].sum())
    if ls:
        print(f"\nWARNING: {ls} level_shift rows — verify corporate actions against a "
              "second source (yfinance) and back-adjust; do not interpolate these.")


def write_report(flags: pd.DataFrame, rep: pd.DataFrame, span: str, universe: str,
                 tag: str = "") -> None:
    """reports/vet_prices{tag}.{csv,md} + row-level flagged CSV (the audit trail)."""
    out = REPO / "reports"
    out.mkdir(exist_ok=True)
    rep.round(4).to_csv(out / f"vet_prices{tag}.csv")
    cols = ["symbol", "date"] + FLAG_COLS + ["ohlc_quirk", "z"]
    flagged = flags.loc[flags["any"] | flags["ohlc_quirk"], cols]
    flagged.to_csv(out / f"vet_prices{tag}_flagged_rows.csv", index=False)

    worst = flagged.loc[flags["any"]].reindex(
        flagged.loc[flags["any"], "z"].abs().sort_values(ascending=False).index
    ).head(20)
    n_ls = int(flags["level_shift"].sum())
    md = [
        f"# Price data QC — FMP `etf_prices`{' (full feed)' if tag else ''}",
        "",
        f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} by `scripts/vet_prices.py`.",
        f"Data span: {span}. Universe: {universe}.",
        "",
        "## Error rate, % of rows flagged",
        "",
        rep.round(3).to_markdown(),
        "",
        "Buckets: `structural` OHLC impossible (high<low, close/open >1% outside range, "
        "price<=0); `fat_finger` idiosyncratic reversed spike >=5%; `jitter` same but "
        "smaller (bad ticks); `level_shift` big non-reversed jump (missed split — "
        "back-adjust, never delete); `stale` >=5 flat closes on zero volume; "
        "`ohlc_quirk` close/open <=1% outside range with single-print bars (benign "
        "thin-LSE auction pattern — close usable, range not). "
        "`any` counts errors only, quirks excluded.",
        "",
        f"Detector: rolling {ROLL}d median/MAD z > {K_MAD:g} with >= {REVERSAL:.0%} "
        f"next-bar reversal, {RET_FLOOR:.1%} absolute floor; market-wide days "
        f"(>= {MARKET_FRAC:.0%} of the full {112}-symbol cross-section at |z|>3) are "
        "never flagged.",
    ]
    if len(worst):
        md += ["", "## Worst flagged rows (by |z|)", "", worst.to_markdown(index=False)]
    if n_ls:
        md += ["", f"**WARNING**: {n_ls} level_shift rows — verify corporate actions "
                   "against a second source and back-adjust; do not interpolate."]
    (out / f"vet_prices{tag}.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote reports/vet_prices{tag}.csv, .md, _flagged_rows.csv")


def main_db(write_clean: str | None, symbols: list[str] | None) -> None:
    from sqlalchemy import text

    from macro_framework.db import get_engine

    # always QC the full feed: the market-wide guard needs the whole cross-section
    with get_engine().connect() as conn:
        prices = pd.read_sql(
            text("SELECT symbol, date, open, high, low, close, volume FROM etf_prices"),
            conn, parse_dates=["date"],
        )
    print(f"loaded {len(prices):,} rows, {prices['symbol'].nunique()} symbols, "
          f"{prices['date'].min().date()} .. {prices['date'].max().date()}")
    flags, rep = run_qc(prices)
    if symbols:
        print(f"reporting universe: {', '.join(symbols)}")
        prices = prices.sort_values(["symbol", "date"]).reset_index(drop=True)
        keep = flags["symbol"].isin(symbols)
        prices, flags = prices.loc[keep], flags.loc[keep]
        rep = summarize(flags)
    report(flags, rep)
    span = f"{prices['date'].min():%Y-%m-%d} .. {prices['date'].max():%Y-%m-%d}"
    write_report(flags, rep, span, ", ".join(symbols) if symbols else "all symbols",
                 tag="" if symbols else "_all")
    if write_clean:
        cleaned = clean_prices(prices, flags)
        cleaned.to_parquet(write_clean, index=False)
        n = int(flags[["structural", "fat_finger", "jitter"]].any(axis=1).sum())
        print(f"\nwrote {write_clean} ({n} closes masked+ffilled)")


def selfcheck() -> None:
    """Inject known defects into synthetic GBM series; assert the detectors find
    them and (almost) nothing else."""
    rng = np.random.default_rng(7)
    n, sigma = 2500, 0.005
    dates = pd.bdate_range("2015-01-02", periods=n)
    frames = []
    for sym in ("AAA", "BBB"):
        logp = np.cumsum(rng.normal(0.0003, sigma, n))
        c = 100 * np.exp(logp)
        if sym == "AAA":
            c[500] *= 1.15          # fat finger: +15%, reverts next bar
            c[1200] *= 1.045        # jitter: +4.5%, reverts next bar
            c[1800:] *= 2.0         # level shift: missed 2:1 adjustment
        v = rng.integers(1e4, 1e6, n).astype(float)
        df = pd.DataFrame({"symbol": sym, "date": dates, "open": c, "high": c * 1.001,
                           "low": c * 0.999, "close": c, "volume": v})
        if sym == "AAA":
            # thin-LSE pattern: single-print bar, close 0.3% outside -> quirk not error
            df.loc[2000, ["open", "high", "low"]] = df.loc[2000, "close"] * 1.003
        if sym == "BBB":
            df.loc[900:907, ["close", "volume"]] = [df.loc[899, "close"], 0.0]  # stale run
            df.loc[1500, "high"] = df.loc[1500, "low"] - 1.0                    # structural
        frames.append(df)
    prices = pd.concat(frames, ignore_index=True)
    flags, rep = run_qc(prices)

    f = flags.set_index(["symbol", "date"])
    assert f.loc[("AAA", dates[500]), "fat_finger"], "fat finger missed"
    assert f.loc[("AAA", dates[1200]), "jitter"], "jitter missed"
    assert f.loc[("AAA", dates[1800]), "level_shift"], "level shift missed"
    assert f.loc[("BBB", dates[1500]), "structural"], "structural error missed"
    q = f.loc[("AAA", dates[2000])]
    assert q["ohlc_quirk"] and not q["structural"], "quirk misclassified"
    assert flags.loc[flags["symbol"] == "BBB", "stale"].sum() >= STALE_RUN, "stale run missed"
    injected = 4 + int(flags["stale"].sum())
    false_pos = int(flags["any"].sum()) - injected
    assert false_pos <= 2, f"too many false positives: {false_pos}"

    cleaned = clean_prices(prices, flags)
    a = cleaned[cleaned["symbol"] == "AAA"].reset_index(drop=True)
    r = np.log(a["close"]).diff().abs()
    assert r[499:502].max() < 0.05, "fat finger not repaired"
    report(flags, rep)
    print(f"\nselfcheck OK (false positives beyond injected: {false_pos})")


def default_universe() -> list[str] | None:
    """The simulated 4-ETF universe (+SPY benchmark), as the notebooks source it."""
    p = REPO / "data" / "portfolio_ssr_top_per_category.parquet"
    if not p.exists():
        return None
    return [*pd.read_parquet(p)["symbol"], "SPY"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selfcheck", action="store_true", help="run synthetic test, no DB")
    ap.add_argument("--write-clean", metavar="PATH", help="write repaired parquet")
    ap.add_argument("--all", action="store_true", help="scan the full etf_prices feed")
    ap.add_argument("--symbols", nargs="+", help="explicit symbol list")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        syms = None if args.all else (args.symbols or default_universe())
        main_db(args.write_clean, syms)
