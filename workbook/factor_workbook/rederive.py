"""Pure re-derivation formulas shown to the reviewer in the workbook.

The binomial (Wilson) interval for the naive eval (R3.2), the recall-guard
formula (R4.3), the paired effect size and contamination premium (R6.1), the
loading-stability measures (R4.4), the equity-line metrics with the fixed
2022 crisis window (R5.3), and the evidence class statistics (R2.6).

Every function is pure, deterministic, and typed: equal inputs yield equal
outputs, no I/O, no randomness. This module imports only pandas/numpy and
the stdlib — never ``release``/``contract`` (architecture layering) and never
``macro_framework`` (the formulas mirror its published semantics instead:
``factor_scoring.factor_stability``, ``ContrastResult.contamination_premium``,
``evaluation.crisis_analytics``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Result records                                                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PremiumResult:
    """Contamination premium re-derived from the paired per-date records (R6.1).

    Attributes:
        n_pairs: Number of valid index-paired ``p_memorized`` pairs.
        mean_delta: mean(non-PIT) - mean(PIT) over the paired records.
        median_delta: median(non-PIT) - median(PIT) over the paired records.
        paired_d: Cohen's d for paired samples over the per-date deltas.
    """

    n_pairs: int
    mean_delta: float
    median_delta: float
    paired_d: float


@dataclass(frozen=True)
class EquityMetrics:
    """Equity-line metrics recomputed from a value series (R5.3).

    Attributes:
        total_return: Final over initial value minus one.
        annualized_return: Geometric annualization on 252 trading days.
        annualized_vol: Sample std of daily returns, annualized.
        sharpe: annualized_return / annualized_vol (rf = 0).
        sortino: annualized_return / annualized downside std (ddof=0 over
            negative returns).
        calmar: annualized_return / abs(max_drawdown).
        max_drawdown: Minimum of the running-max drawdown of the value line.
        crisis_return: Period return inside the fixed crisis window (NaN when
            the series does not overlap the window).
        crisis_max_drawdown: Within-window running-max drawdown minimum (NaN
            when no overlap).
        crisis_vol_ann: Annualized sample std of within-window returns (NaN
            when no overlap).
    """

    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    crisis_return: float
    crisis_max_drawdown: float
    crisis_vol_ann: float


@dataclass(frozen=True)
class ClassStats:
    """Class counts + feature summaries re-derived from raw evidence (R2.6).

    Attributes:
        arm_counts: Number of evidence records per evidence arm (class).
        feature_stats: One row per arm; ``<feature>_mean`` / ``<feature>_std``
            (sample std) columns for every standardized ``std_*`` feature.
    """

    arm_counts: dict[str, int]
    feature_stats: pd.DataFrame


# --------------------------------------------------------------------------- #
# Formulas                                                                     #
# --------------------------------------------------------------------------- #


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% score interval for a binomial proportion (R3.2, nb13 form).

    Center ``(p + z^2/2n) / (1 + z^2/n)`` with half-width
    ``z * sqrt(p(1-p)/n + z^2/4n^2) / (1 + z^2/n)``.

    Args:
        successes: Number of correct calls.
        n: Number of trials.
        z: Normal quantile (1.96 for the 95% interval).

    Returns:
        The ``(low, high)`` interval; a zero-trial input degrades to the
        uninformative ``(0.0, 1.0)`` rather than raising.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half_width = z * math.sqrt(p * (1.0 - p) / n + z**2 / (4 * n**2)) / denom
    return (center - half_width, center + half_width)


def guarded_tilt(raw: pd.Series, p_mem: pd.Series) -> pd.Series:
    """Re-derive the guarded tilt: raw times one minus the score (R4.3).

    Args:
        raw: Raw per-asset tilt values.
        p_mem: Memorization scores, clipped into ``[0, 1]``.

    Returns:
        The guarded tilt series ``raw * (1 - clip(p_mem, 0, 1))``.
    """
    return raw * (1.0 - p_mem.clip(0.0, 1.0))


def paired_cohens_d(deltas: pd.Series) -> float:
    """Cohen's d for paired samples: mean(deltas) / population std(deltas).

    Mirrors ``macro_framework.factor_scoring._paired_cohens_d`` (the producer
    of the published ``p_memorized_paired_d``): population std (ddof=0), with
    fewer than two pairs or (near-)zero variance degrading to ``0.0`` rather
    than NaN or a division error (R6.1).

    Args:
        deltas: Per-date paired deltas (non-PIT minus PIT).

    Returns:
        The standardized paired effect size.
    """
    values = deltas.dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return 0.0
    std = float(np.std(values))
    if std <= 1e-12:
        return 0.0
    return float(np.mean(values)) / std


def contamination_premium(pit: pd.Series, nonpit: pd.Series) -> PremiumResult:
    """Re-derive the contamination premium from paired per-date scores (R6.1).

    Pairs the two ``p_memorized`` streams on their common index, drops pairs
    with a missing side, and reports the non-PIT minus PIT gap over the
    stream: mean delta, median delta, and the paired effect size.

    Args:
        pit: Per-date memorization scores of the point-in-time variant.
        nonpit: Per-date memorization scores of the recall-enabled variant.

    Returns:
        The premium summary; an empty pairing degrades to all-zero values.
    """
    paired = pd.DataFrame({"pit": pit, "nonpit": nonpit}).dropna()
    if paired.empty:
        return PremiumResult(n_pairs=0, mean_delta=0.0, median_delta=0.0, paired_d=0.0)
    deltas = paired["nonpit"] - paired["pit"]
    return PremiumResult(
        n_pairs=len(paired),
        mean_delta=float(paired["nonpit"].mean() - paired["pit"].mean()),
        median_delta=float(paired["nonpit"].median() - paired["pit"].median()),
        paired_d=paired_cohens_d(deltas),
    )


def loading_stability(
    loadings: pd.DataFrame, parse_ok: pd.Series
) -> dict[str, float]:
    """Re-derive the per-version loading-stability measures (R4.4).

    Mirrors ``macro_framework.factor_scoring.factor_stability``: per axis the
    population std (ddof=0) and the mean absolute consecutive change over the
    DATE-SORTED parsed stream, plus the ``mean_std`` / ``mean_mac`` per-axis
    means. ``parse_ok=False`` rows are skipped (a not-parsed rebalance carries
    no factor vector); a missing (NaN) axis value contributes nothing to that
    axis. Fewer than two points per axis degrade to ``0.0``, never NaN.

    Args:
        loadings: Per-rebalance axis loadings (one column per macro axis,
            date index).
        parse_ok: Per-rebalance parse status aligned with ``loadings``.

    Returns:
        ``{<axis>_std, <axis>_mac, ...}`` plus ``mean_std`` / ``mean_mac``.
    """
    parsed = loadings[parse_ok.astype(bool)].sort_index()
    summary: dict[str, float] = {}
    per_axis_std: list[float] = []
    per_axis_mac: list[float] = []
    for axis in loadings.columns:
        values = parsed[axis].dropna().to_numpy(dtype=float)
        if len(values) < 2:
            axis_std = axis_mac = 0.0
        else:
            axis_std = float(np.std(values))
            axis_mac = float(np.mean(np.abs(np.diff(values))))
        summary[f"{axis}_std"] = axis_std
        summary[f"{axis}_mac"] = axis_mac
        per_axis_std.append(axis_std)
        per_axis_mac.append(axis_mac)
    n_axes = len(loadings.columns)
    summary["mean_std"] = sum(per_axis_std) / n_axes if n_axes else 0.0
    summary["mean_mac"] = sum(per_axis_mac) / n_axes if n_axes else 0.0
    return summary


def equity_metrics(
    value: pd.Series, crisis: tuple[str, str] = ("2022-01-01", "2022-12-31")
) -> EquityMetrics:
    """Recompute the equity-line metrics from a value series (R5.3).

    Daily returns are ``value.pct_change().dropna()``; annualization uses 252
    trading days with geometric compounding for the return. The crisis block
    mirrors ``macro_framework.evaluation.crisis_analytics`` on the fixed 2022
    window; a series with no window overlap reports NaN crisis fields.

    Args:
        value: The equity value series (datetime index, e.g. the released
            ``factor_equity_v1`` value column).
        crisis: The fixed crisis window as ``(start, end)`` date strings.

    Returns:
        The recomputed metrics; degenerate inputs (constant line, no downside)
        degrade ratio metrics to ``0.0`` rather than NaN.
    """
    returns = value.pct_change().dropna()
    n = len(returns)
    total_return = float(value.iloc[-1] / value.iloc[0] - 1.0) if len(value) else 0.0
    annualized_return = (
        (1.0 + total_return) ** (_TRADING_DAYS / n) - 1.0 if n else 0.0
    )
    annualized_vol = (
        float(returns.std(ddof=1)) * math.sqrt(_TRADING_DAYS) if n >= 2 else 0.0
    )
    sharpe = annualized_return / annualized_vol if annualized_vol > 0.0 else 0.0
    downside = returns[returns < 0].to_numpy(dtype=float)
    downside_vol = (
        float(np.std(downside)) * math.sqrt(_TRADING_DAYS) if len(downside) else 0.0
    )
    sortino = annualized_return / downside_vol if downside_vol > 0.0 else 0.0
    drawdown = value / value.cummax() - 1.0
    max_drawdown = float(drawdown.min()) if len(value) else 0.0
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0.0 else 0.0

    window = value.loc[crisis[0] : crisis[1]]
    if window.empty:
        crisis_return = crisis_max_dd = crisis_vol = float("nan")
    else:
        crisis_return = float(window.iloc[-1] / window.iloc[0] - 1.0)
        crisis_max_dd = float((window / window.cummax() - 1.0).min())
        crisis_vol = float(window.pct_change().std(ddof=1)) * math.sqrt(_TRADING_DAYS)

    return EquityMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_vol=annualized_vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_drawdown,
        crisis_return=crisis_return,
        crisis_max_drawdown=crisis_max_dd,
        crisis_vol_ann=crisis_vol,
    )


def evidence_class_stats(evidence: pd.DataFrame) -> ClassStats:
    """Re-derive class counts + std_* feature summaries from raw evidence (R2.6).

    Args:
        evidence: Raw per-prompt evidence records carrying an ``arm`` column
            (the evidence class) and standardized ``std_*`` feature columns.

    Returns:
        Per-arm record counts and, per arm, the mean and sample std of every
        ``std_*`` feature (``<feature>_mean`` / ``<feature>_std`` columns).
    """
    arm_counts = {str(arm): int(count) for arm, count in evidence["arm"].value_counts().items()}
    std_cols = [c for c in evidence.columns if c.startswith("std_")]
    grouped = evidence.groupby("arm")[std_cols].agg(["mean", "std"])
    grouped.columns = [f"{feature}_{stat}" for feature, stat in grouped.columns]
    return ClassStats(arm_counts=arm_counts, feature_stats=grouped)
