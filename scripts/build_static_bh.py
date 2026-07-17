"""Static buy-and-hold line of the 4-ETF portfolio, persisted as data artifacts.

Reproduces notebook 04's static run (25% per top-SSR-per-category ETF, bought
once, never rebalanced) and persists it as tidy artifacts for the data release
and the Excel workbook's comparison lines:

- ``data/static_bh_equity_2014_2024.parquet``  — Date-indexed ``value`` series
  on the walk-forward-matching window (joins against ``factor_equity_v1``).
- ``data/static_bh_targets_2014_2024.parquet`` — the DRIFTING weights (buy-and
  -hold holds shares, not weights; the drift is part of the story).
- ``data/static_bh_equity_2016_2026.parquet``  — nb04's original 10-year window.
- ``data/static_bh_stats.json``                — per-window metrics (vectorbt
  convention, matching the published head-to-head figures) + the Sharpe
  Stability Ratio of the static line, + the in-sample caveat verbatim.

Prices: yfinance (auto-adjusted Close), the documented substitution for the
absent Postgres price DB, matching notebooks 11/13/14. In-sample caveat
(nb04, carried verbatim): the four ETFs were SELECTED by SSR computed over the
same window being simulated. A hindsight test, not an out-of-sample result.

Reproducible: ``uv run python scripts/build_static_bh.py``. Additive: writes
only the four artifacts above (gitignored; shipped via the GH data release).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workbook"))

import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

from factor_workbook.rederive import equity_metrics  # noqa: E402
from macro_framework.ssr import compute_ssr  # noqa: E402

INIT = 10_000.0  # matches factor_equity_v1's start value (scale-invariant stats)
WINDOWS = {
    "2014_2024": ("2014-01-02", "2024-12-30"),  # joins the S4 walk-forward lines
    "2016_2026": ("2016-01-31", "2026-01-31"),  # nb04's original 10-year window
}


def static_bh(prices: pd.DataFrame, weights: dict[str, float]) -> tuple[pd.Series, pd.DataFrame]:
    """Buy once at the first close, hold: equity value + drifting weights."""
    prices = prices.dropna(how="any")
    shares = {s: INIT * w / prices[s].iloc[0] for s, w in weights.items()}
    values = pd.DataFrame({s: prices[s] * n for s, n in shares.items()})
    equity = values.sum(axis=1).rename("value")
    drift = values.div(equity, axis=0)
    return equity, drift


def main() -> None:
    spec = pd.read_parquet(REPO / "data" / "portfolio_ssr_top_per_category.parquet")
    weights = dict(zip(spec["symbol"], spec["weight"]))
    symbols = list(weights)
    print(f"[1/3] static B&H of {weights} (yfinance substitution for the price DB)")

    raw = yf.download(symbols + ["SPY"], start="2013-12-01", end="2026-02-01",
                      auto_adjust=True, progress=False)["Close"]

    stats: dict[str, dict] = {}
    for tag, (start, end) in WINDOWS.items():
        window = raw.loc[start:end]
        equity, drift = static_bh(window[symbols], weights)
        equity.index.name = "Date"
        drift.index.name = "Date"

        out_eq = REPO / "data" / f"static_bh_equity_{tag}.parquet"
        equity.to_frame().to_parquet(out_eq)
        if tag == "2014_2024":
            drift.to_parquet(REPO / "data" / f"static_bh_targets_{tag}.parquet")

        m = equity_metrics(equity)
        ssr = compute_ssr(equity.pct_change().dropna())
        spy_equity = (window["SPY"].dropna().pct_change().fillna(0).add(1).cumprod() * INIT).rename("value")
        spy_m = equity_metrics(spy_equity)

        # Crisis drawdown episodes — the event-level observables (window
        # return, max drawdown, annualized vol inside the episode), for the
        # static line AND the SPY reference, per named macro-crisis window.
        episodes = {}
        for name, (c_start, c_end) in {
            "covid_2020": ("2020-02-19", "2020-04-30"),
            "inflation_2022": ("2022-01-01", "2022-12-31"),
        }.items():
            em = equity_metrics(equity, crisis=(c_start, c_end))
            es = equity_metrics(spy_equity, crisis=(c_start, c_end))
            episodes[name] = {
                "window": [c_start, c_end],
                "static_bh": {"crisis_return": em.crisis_return,
                               "crisis_max_drawdown": em.crisis_max_drawdown,
                               "crisis_vol_ann": em.crisis_vol_ann},
                "spy_bh": {"crisis_return": es.crisis_return,
                            "crisis_max_drawdown": es.crisis_max_drawdown,
                            "crisis_vol_ann": es.crisis_vol_ann},
            }

        stats[tag] = {
            "window": [start, end],
            "weights_at_inception": weights,
            "static_bh": {k: getattr(m, k) for k in (
                "total_return", "annualized_return", "annualized_vol", "sharpe",
                "sortino", "calmar", "max_drawdown")},
            "static_bh_ssr": {"ssr": ssr.ssr, "mean_rolling_sr": ssr.mean_rolling_sr,
                               "sigma_hac": ssr.sigma_hac, "L_hac": ssr.L_hac,
                               "n_rolling": ssr.n_rolling},
            "spy_bh": {k: getattr(spy_m, k) for k in (
                "total_return", "annualized_return", "sharpe", "max_drawdown")},
            "crisis_episodes": episodes,
            "weight_drift_final": drift.iloc[-1].round(4).to_dict() if tag == "2014_2024" else None,
        }
        print(f"[2/3] {tag}: total_return={m.total_return:.4f} sharpe={m.sharpe:.4f} "
              f"max_dd={m.max_drawdown:.4f} SSR={ssr.ssr:.4f} -> {out_eq.name}")

    stats["caveat"] = (
        "IN-SAMPLE BY CONSTRUCTION: the four ETFs were selected by the Sharpe "
        "Stability Ratio computed over the same window being simulated (nb02/nb03). "
        "This line illustrates how strong a hindsight-selected static portfolio looks "
        "— the lookahead/contamination problem the recall-guarded pipeline measures. "
        "Its performance is a hindsight artifact, never attainable skill."
    )
    stats["source"] = "yfinance auto-adjusted Close (documented substitution for the price DB)"
    stats["built_at"] = datetime.now(timezone.utc).isoformat()
    (REPO / "data" / "static_bh_stats.json").write_text(json.dumps(stats, indent=2))
    print("[3/3] stats -> data/static_bh_stats.json")


if __name__ == "__main__":
    main()
