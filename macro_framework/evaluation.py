"""Head-to-head evaluation metrics for Baseline / Track A / Track B."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    import vectorbt as vbt


def anticipation_lead_time(
    target_weights: pd.DataFrame,
    defensive_cols: tuple[str, ...] = ("BIL", "IAU"),
    threshold: float = 0.40,
) -> pd.Timestamp | None:
    """First rebalance date where the sum of `defensive_cols` weights ≥ threshold.
    Returns None if the strategy never crosses the threshold.
    """
    tgt = target_weights.dropna(how="all")
    defensive = [c for c in defensive_cols if c in tgt.columns]
    if not defensive:
        return None
    defensive_share = tgt[defensive].sum(axis=1)
    hit = defensive_share[defensive_share >= threshold]
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
        lead = anticipation_lead_time(
            targets.get(name, pd.DataFrame()),
            defensive_cols=defensive_cols,
            threshold=defensive_threshold,
        )
        turnover = turnover_stats(targets.get(name, pd.DataFrame()))
        crow = crisis.loc[name].to_dict() if name in crisis.index else {"crisis_return": float("nan"), "crisis_max_drawdown": float("nan"), "crisis_vol_ann": float("nan")}
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
            "defensive_lead_date":      (pd.Timestamp(lead).date().isoformat() if lead is not None else "—"),
            "avg_turnover":             turnover["avg_turnover"],
        }
    return pd.DataFrame(rows).T
