"""Head-to-head evaluation metrics for Baseline / Track A / Track B."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    import vectorbt as vbt

#: Trading-day annualization basis (sqrt(252) volatility, the notebooks' headline).
TRADING_DAYS = 252
#: Calendar-year basis — vectorbt's ``year_freq`` and what
#: ``factor_workbook.rederive`` used to build the RELEASED tear sheet.
CALENDAR_DAYS = 365


def cagr(value: pd.Series) -> float:
    """Geometric growth rate on ELAPSED CALENDAR TIME. The preferred basis: it is
    invariant to how many sessions the exchange happened to open in the window."""
    years = max((value.index[-1] - value.index[0]).days / 365.25, 1 / 365.25)
    return float((value.iloc[-1] / value.iloc[0]) ** (1 / years) - 1)


def cagr_rows(value: pd.Series, *, periods_per_year: float = CALENDAR_DAYS) -> float:
    """Geometric growth rate annualized on ROW COUNT — vectorbt's convention, kept
    for parity with the released artifact. Sensitive to calendar coverage: the same
    curve on a 252-row/yr trading calendar vs a 365-row/yr padded one differs."""
    n = len(value)
    if n < 2:
        return 0.0
    return float((value.iloc[-1] / value.iloc[0]) ** (periods_per_year / n) - 1)


def max_drawdown(value: pd.Series) -> float:
    return float((value / value.cummax() - 1).min())


def calmar(value: pd.Series) -> float:
    dd = max_drawdown(value)
    return float(cagr(value) / abs(dd)) if dd else np.nan


def metric_block(value: pd.Series) -> dict:
    """One tear-sheet metric implementation shared by nb15_2 / nb16 / nb17 / nb18.

    BOTH annualization bases are returned, explicitly named, because the repo needs
    both: row-count/365 for vectorbt and released-artifact parity, elapsed-calendar
    /252 for everything reader-facing. The bare keys (``cagr``, ``ann_vol``,
    ``sharpe``, ``sortino``, ``calmar``) are the preferred calendaric/trading-day
    primaries; the ``*_rows`` and ``*_cal`` keys are the alternates.

    Never mix bases within one table. Doing so inflates vol-scaled figures by
    sqrt(365/252)=1.20 and annualized means by 365/252=1.45 — the exact failure
    ``tests/test_evaluation.py`` now pins against.
    """
    r = value.pct_change().dropna()
    downside = np.minimum(r.to_numpy(dtype=float), 0.0)
    downside_rms = float(np.sqrt(np.mean(downside ** 2))) if len(downside) else np.nan
    std = float(r.std(ddof=1))
    mean = float(r.mean())
    dd = value / value.cummax() - 1
    mdd = float(dd.min())
    cg, cg_rows = cagr(value), cagr_rows(value)

    def _ann(basis: int) -> tuple[float, float]:
        root = np.sqrt(basis)
        return (
            float(mean / std * root) if std > 0 else np.nan,
            float(mean / downside_rms * root) if downside_rms > 0 else np.nan,
        )

    sharpe_252, sortino_252 = _ann(TRADING_DAYS)
    sharpe_365, sortino_365 = _ann(CALENDAR_DAYS)
    return {
        "returns": r,
        "dd": dd,
        # basis-free
        "total_return": float(value.iloc[-1] / value.iloc[0] - 1),
        "maxdd": mdd,
        "downside_rms": downside_rms,
        # preferred: elapsed-calendar growth, sqrt(252) risk
        "cagr": cg,
        "ann_vol": std * np.sqrt(TRADING_DAYS),
        "sharpe": sharpe_252,
        "sortino": sortino_252,
        "calmar": float(cg / abs(mdd)) if mdd else np.nan,
        # alternates: row-count growth, sqrt(365) risk (vectorbt / released artifact)
        "cagr_rows": cg_rows,
        "ann_vol_cal": std * np.sqrt(CALENDAR_DAYS),
        "sharpe_cal": sharpe_365,
        "sortino_cal": sortino_365,
        "calmar_rows": float(cg_rows / abs(mdd)) if mdd else np.nan,
    }


def anticipation_lead_time(
    target_weights: pd.DataFrame,
    defensive_cols: tuple[str, ...] = ("BIL", "IAU"),
    threshold: float = 0.40,
) -> pd.Timestamp | None:
    """First rebalance date where defensive share CROSSES up to the threshold.

    Returns the first date where the sum of ``defensive_cols`` moves from below
    ``threshold`` to at-or-above it. A portfolio already defensive on the first
    observed row does NOT count as an anticipatory rotation and returns ``None``.
    """
    tgt = target_weights.dropna(how="all")
    defensive = [c for c in defensive_cols if c in tgt.columns]
    if not defensive:
        return None
    defensive_share = tgt[defensive].sum(axis=1)
    above = defensive_share >= threshold
    if above.empty:
        return None
    crossed = above & ~above.shift(1, fill_value=above.iloc[0])
    hit = crossed[crossed]
    return hit.index[0] if len(hit) else None


def crisis_analytics(
    pfs: dict[str, "vbt.Portfolio"],
    crisis_start: str = "2022-01-01",
    crisis_end: str = "2022-12-31",
) -> pd.DataFrame:
    """Within-crisis DD + period return + vol per portfolio."""
    rows: dict[str, dict[str, float]] = {}
    for name, pf in pfs.items():
        val = pf.value()
        window = val.loc[crisis_start:crisis_end]
        if window.empty:
            continue
        peak = window.cummax()
        dd = (window / peak) - 1.0
        period_return = window.iloc[-1] / window.iloc[0] - 1.0
        rows[name] = {
            "crisis_return": float(period_return),
            "crisis_max_drawdown": float(dd.min()),
            "crisis_vol_ann": float(window.pct_change().std(ddof=1) * np.sqrt(252)),
        }
    return pd.DataFrame(rows).T


def turnover_stats(target_weights: pd.DataFrame) -> dict[str, float]:
    """Weight turnover at each rebalance: sum(|Δw|). Average + max."""
    tgt = target_weights.dropna(how="all")
    if tgt.empty or len(tgt) < 2:
        return {"avg_turnover": 0.0, "max_turnover": 0.0}
    diffs = tgt.diff().abs().sum(axis=1).iloc[1:]
    return {"avg_turnover": float(diffs.mean()), "max_turnover": float(diffs.max())}


def view_stability(views_log: dict) -> dict[str, float]:
    """Track-A only: month-to-month stability of agent view magnitudes.

    Returns count of views per month, mean |expected_excess|, and how often the
    largest-confidence view changes its long leg.
    """
    if not views_log:
        return {"mean_n_views": 0.0, "mean_abs_expected": 0.0, "long_switch_rate": 0.0}
    dates = sorted(views_log.keys())
    n_views, magnitudes, top_longs = [], [], []
    for d in dates:
        views = views_log[d]
        n_views.append(len(views))
        if views:
            magnitudes.extend([abs(v["expected_excess_annualized"]) for v in views])
            top = max(views, key=lambda v: v.get("confidence", 0))
            top_longs.append(top["asset_long"])
    switches = sum(1 for a, b in zip(top_longs, top_longs[1:]) if a != b)
    denom = max(1, len(top_longs) - 1)
    return {
        "mean_n_views":      float(np.mean(n_views)) if n_views else 0.0,
        "mean_abs_expected": float(np.mean(magnitudes)) if magnitudes else 0.0,
        "long_switch_rate":  switches / denom,
    }


def head_to_head_report(
    pfs: dict[str, "vbt.Portfolio"],
    targets: dict[str, pd.DataFrame],
    crisis_start: str = "2022-01-01",
    crisis_end: str = "2022-12-31",
    defensive_cols: tuple[str, ...] = ("BIL", "IAU"),
    defensive_threshold: float = 0.40,
) -> pd.DataFrame:
    """Single side-by-side comparison table across all three tracks."""
    rows: dict[str, dict[str, float]] = {}
    crisis = crisis_analytics(pfs, crisis_start, crisis_end)
    for name, pf in pfs.items():
        tgt = targets.get(name, pd.DataFrame()).dropna(how="all")
        lead = anticipation_lead_time(
            tgt,
            defensive_cols=defensive_cols,
            threshold=defensive_threshold,
        )
        turnover = turnover_stats(tgt)
        crow = crisis.loc[name].to_dict() if name in crisis.index else {"crisis_return": float("nan"), "crisis_max_drawdown": float("nan"), "crisis_vol_ann": float("nan")}
        defensive = [c for c in defensive_cols if c in tgt.columns]
        starts_defensive = bool(
            defensive and not tgt.empty and tgt[defensive].sum(axis=1).iloc[0] >= defensive_threshold
        )
        lead_label = (
            pd.Timestamp(lead).date().isoformat() if lead is not None
            else "already defensive at sample start" if starts_defensive
            else "—"
        )
        rows[name] = {
            "total_return":             float(pf.total_return()),
            "annualized_return":        float(pf.annualized_return()),
            "annualized_vol":           float(pf.annualized_volatility()),
            "sharpe":                   float(pf.sharpe_ratio()),
            "sortino":                  float(pf.sortino_ratio()),
            "calmar":                   float(pf.calmar_ratio()),
            "max_drawdown":             float(pf.max_drawdown()),
            "crisis_return":            crow["crisis_return"],
            "crisis_max_drawdown":      crow["crisis_max_drawdown"],
            "defensive_lead_date":      lead_label,
            "avg_turnover":             turnover["avg_turnover"],
        }
    return pd.DataFrame(rows).T
