"""Tear-sheet data pack for every equity line, as Excel-ready CSVs.

Per line (static B&H both windows, the factor PIT/non-PIT/v2 lines, and the
2019-2024 tracks): return/risk metrics under the repo's published conventions
(365-day annualization, day-0 zero return; mirrors head_to_head_report via
vectorbt), tail statistics, the Sharpe Stability Ratio block (Newey-West HAC,
Andrews bandwidth; not reproducible in native Excel, hence precomputed), and
a two-level risk decomposition:

- CAPM vs SPY: beta, annualized alpha, R², the systematic share relative
  to the broad equity market; residual vol is "idiosyncratic-to-the-market"
  (for a multi-asset book much of it is OTHER systematic factors: gold, rates).
- Basket (4-ETF) regression: R² against the portfolio's own asset-class
  factors (SWDA.L/XLK/IAU/BIL daily returns). For a static basket this is ~1:
  single-name idiosyncratic risk is already diversified inside the ETF
  wrappers, and what remains for dynamic lines is allocation-timing residual,
  not stock-picking risk.

Outputs (data/tear_sheet/, gitignored; upload to the data release):
- tear_sheet.csv            — one row per line, all metrics as columns
- risk_decomposition.csv    — CAPM + basket regression per line
- monthly_returns_<key>.csv — month x year matrices for the headline lines

Prices for SPY + the 4 ETFs via yfinance (documented DB substitution).
Reproducible: ``uv run python scripts/build_tear_sheet.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workbook"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

from factor_workbook.rederive import equity_metrics  # noqa: E402
from macro_framework.ssr import compute_ssr  # noqa: E402

OUT = REPO / "data" / "tear_sheet"
BASKET = ["SWDA.L", "XLK", "IAU", "BIL"]

LINES = {  # key -> (equity parquet, label)
    "static_bh_2016_2026": ("static_bh_equity_2016_2026.parquet", "Static B&H 25% EW (nb04 10y, in-sample)"),
    "static_bh_2014_2024": ("static_bh_equity_2014_2024.parquet", "Static B&H 25% EW (walk-forward window)"),
    "factor_pit_v1": ("factor_equity_v1.parquet", "PIT recall-guarded factor (deployable)"),
    "factor_nonpit_diag": ("factor_nonpit_diagnostic_equity_v1.parquet", "Non-PIT recall-enabled (DIAGNOSTIC)"),
    "factor_pit_v2": ("factor_equity_v2.parquet", "PIT factor, rejected prompt v2"),
    "baseline": ("baseline_equity_2019_2024.parquet", "Baseline HRP+momentum"),
    "track_a_llm": ("track_a_equity_2019_2024.parquet", "Track A (LLM directional)"),
    "track_a_steered": ("track_a_steered_equity_2019_2024.parquet", "Track A memory-guarded"),
    "track_b": ("track_b_equity_2019_2024.parquet", "Track B (MC/Nash)"),
}
MONTHLY_KEYS = ["static_bh_2016_2026", "factor_pit_v1", "factor_nonpit_diag"]

ANNUAL = 365  # repo convention (vectorbt calendar-year basis)


def _active_value(value: pd.Series) -> pd.Series:
    """The line from the last flat day before it first moves (skips pre-start stubs).

    Several 2019-start tracks are stored on 2014-anchored frames with a flat
    stub; including the stub dilutes CAGR/vol/Sharpe (the published
    contrast-summary metrics embed that full-frame convention — the tear sheet
    deliberately reports the ACTIVE span instead, disclosed in the window
    columns, matching the luck-vs-skill table's slice convention).
    """
    moving = value[value.ne(value.iloc[0])]
    if moving.empty:
        return value
    first_move = moving.index.min()
    prior = value.index[value.index < first_move]
    start = prior.max() if len(prior) else first_move
    return value.loc[start:]


def _ols(y: pd.Series, x: pd.DataFrame) -> tuple[np.ndarray, float, pd.Series]:
    x_ = np.column_stack([np.ones(len(x)), x.to_numpy()])
    coef, *_ = np.linalg.lstsq(x_, y.to_numpy(), rcond=None)
    fitted = x_ @ coef
    resid = y - fitted
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coef, r2, resid


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    px = yf.download(BASKET + ["SPY"], start="2013-12-01", end="2026-02-01",
                     auto_adjust=True, progress=False)["Close"]
    factor_returns = px.pct_change().dropna(how="all")

    tear_rows, risk_rows = [], []
    for key, (fname, label) in LINES.items():
        path = REPO / "data" / fname
        if not path.exists():
            print(f"  SKIP {key} (missing {fname})")
            continue
        value = _active_value(pd.read_parquet(path)["value"])
        r = value.pct_change().dropna()
        m = equity_metrics(value)
        ssr = compute_ssr(r)

        neg = r[r < 0]
        dd = value / value.cummax() - 1
        dd_end = dd.idxmin()
        dd_start = value.loc[:dd_end].idxmax()
        recovery = dd.loc[dd_end:][dd.loc[dd_end:] >= -1e-12]
        monthly = (1 + r).resample("ME").prod() - 1

        tear_rows.append({
            "line": key, "label": label,
            "start": r.index.min().date(), "end": r.index.max().date(),
            "n_days": len(r),
            "total_return": m.total_return, "cagr": m.annualized_return,
            "ann_vol": m.annualized_vol, "sharpe": m.sharpe,
            "sortino": m.sortino, "calmar": m.calmar,
            "max_drawdown": m.max_drawdown,
            "max_dd_peak": dd_start.date(), "max_dd_trough": dd_end.date(),
            "max_dd_recovered": recovery.index.min().date() if len(recovery) else None,
            "skew": float(r.skew()), "excess_kurtosis": float(r.kurtosis()),
            "var_95_daily": float(r.quantile(0.05)),
            "cvar_95_daily": float(r[r <= r.quantile(0.05)].mean()),
            "best_day": float(r.max()), "worst_day": float(r.min()),
            "positive_day_rate": float((r > 0).mean()),
            "best_month": float(monthly.max()), "worst_month": float(monthly.min()),
            "crisis_2022_return": m.crisis_return,
            "crisis_2022_max_dd": m.crisis_max_drawdown,
            "ssr": ssr.ssr, "mean_rolling_sharpe": ssr.mean_rolling_sr,
            "nw_sigma_hac": ssr.sigma_hac, "nw_bandwidth_L": ssr.L_hac,
            "ssr_verdict": ("stably > 0" if abs(ssr.ssr) >= 1.96 else
                             "NOT distinguishable from zero under HAC — luck-compatible"),
        })

        spy = factor_returns["SPY"].reindex(r.index).dropna()
        y = r.reindex(spy.index)
        coef, r2_capm, resid = _ols(y, spy.to_frame())
        basket = factor_returns[BASKET].reindex(r.index).dropna()
        yb = r.reindex(basket.index)
        _, r2_basket, resid_b = _ols(yb, basket)
        risk_rows.append({
            "line": key, "label": label,
            "beta_spy": float(coef[1]),
            "alpha_ann_vs_spy": float(coef[0] * ANNUAL),
            "r2_capm": r2_capm,
            "corr_spy": float(y.corr(spy)),
            "systematic_share_capm": r2_capm,
            "idio_vol_ann_capm": float(resid.std(ddof=1) * np.sqrt(ANNUAL)),
            "r2_basket_4etf": r2_basket,
            "residual_vol_ann_basket": float(resid_b.std(ddof=1) * np.sqrt(ANNUAL)),
            "note": ("basket R2 ~ 1: single-name idiosyncratic risk is diversified inside the "
                      "ETF wrappers; residual on dynamic lines is allocation-timing, not stock-picking"),
        })

        if key in MONTHLY_KEYS:
            mt = monthly.to_frame("ret")
            mt["year"], mt["month"] = mt.index.year, mt.index.month
            (mt.pivot_table(index="year", columns="month", values="ret")
               .to_csv(OUT / f"monthly_returns_{key}.csv", float_format="%.6f"))

        print(f"  {key}: sharpe {m.sharpe:.2f}  ssr {ssr.ssr:.3f}  beta {coef[1]:.2f}  "
              f"r2_capm {r2_capm:.2f}  r2_basket {r2_basket:.4f}")

    pd.DataFrame(tear_rows).to_csv(OUT / "tear_sheet.csv", index=False, float_format="%.8f")
    pd.DataFrame(risk_rows).to_csv(OUT / "risk_decomposition.csv", index=False, float_format="%.8f")
    print(f"[done] -> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
