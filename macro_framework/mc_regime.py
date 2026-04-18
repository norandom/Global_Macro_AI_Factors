"""Track B — Monte Carlo macro bootstrap + regime classification + Nash-minimax allocator.

Pipeline at each rebalance date t:
  1. Block-bootstrap N forward paths of the macro z-score panel (strictly from dates < t).
  2. Classify each path point into a discrete regime.
  3. Aggregate regime probabilities across all simulated path points.
  4. Estimate per-asset expected daily return conditional on each regime (from history < t).
  5. Build payoff matrix (candidate portfolios × regimes).
  6. Solve minimax LP — mixed strategy over candidate portfolios that maximizes
     worst-case regime return. Final ETF weights = convex combo of candidate portfolios.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import linprog

REGIMES = ("normal", "inflationary_shock", "recession")

# Default rule-based classifier — tweak by passing a different callable to regime_probabilities.
def classify_regime(row: pd.Series) -> str:
    """Given a row with ['cpi_yoy_z', 't10y2y_z', 'hy_oas_z'], return a regime label.

    Priority: inflation > recession > normal.
    """
    if row["cpi_yoy_z"] > 1.0:
        return "inflationary_shock"
    if row["hy_oas_z"] > 1.0 or row["t10y2y_z"] < -1.5:
        return "recession"
    return "normal"


def block_bootstrap_paths(
    panel_z: pd.DataFrame,
    horizon: int = 3,
    n_paths: int = 10_000,
    block_size: int = 3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Block bootstrap — resample contiguous blocks of length `block_size` from `panel_z`
    and tile them to a horizon of length `horizon`. Preserves cross-series and
    short-run dependence. Returns an (n_paths, horizon, n_vars) array.
    """
    rng = rng or np.random.default_rng(42)
    X = panel_z.to_numpy(dtype=float)
    T, n_vars = X.shape
    if T < block_size + horizon:
        raise ValueError(f"panel too short: {T} rows, need ≥ {block_size + horizon}")

    n_blocks_needed = int(np.ceil(horizon / block_size))
    max_start = T - block_size
    starts = rng.integers(0, max_start + 1, size=(n_paths, n_blocks_needed))

    paths = np.empty((n_paths, horizon, n_vars), dtype=float)
    for p in range(n_paths):
        chunks = [X[s:s + block_size] for s in starts[p]]
        path = np.concatenate(chunks, axis=0)[:horizon]
        paths[p] = path
    return paths


def regime_probabilities(
    paths: np.ndarray,
    column_names: list[str],
    classifier: Callable[[pd.Series], str] = classify_regime,
) -> dict[str, float]:
    """Flatten all simulated path points into a single bag, classify each, return
    empirical regime frequencies."""
    n_paths, horizon, n_vars = paths.shape
    flat = paths.reshape(-1, n_vars)
    df = pd.DataFrame(flat, columns=column_names)
    labels = df.apply(classifier, axis=1)
    counts = labels.value_counts(normalize=True)
    return {r: float(counts.get(r, 0.0)) for r in REGIMES}


def candidate_portfolios(symbols: list[str]) -> dict[str, pd.Series]:
    """8 standard portfolios for the Nash payoff matrix.
    Symbols must be ordered [world, tech_sector, commodity_gold, cash_tbill]."""
    if len(symbols) != 4:
        raise ValueError("expected exactly 4 symbols: [world, sector, commodity, bond/cash]")
    w, s, c, b = symbols

    def v(d: dict[str, float]) -> pd.Series:
        out = pd.Series(0.0, index=symbols)
        for k, val in d.items():
            out[k] = val
        return out / out.sum()

    return {
        "all_cash":            v({b: 1.0}),                              # flight-to-safety
        "equal_weight":        v({w: 1, s: 1, c: 1, b: 1}),              # 25/25/25/25
        "defensive_60_40":     v({w: 0.40, b: 0.60}),                    # bond-heavy, classic defensive
        "inflation_hedge":     v({c: 0.50, b: 0.30, w: 0.20}),           # gold-heavy + cash + equity
        "growth_tilt":         v({s: 0.50, w: 0.35, b: 0.15}),           # equity-heavy
        "balanced_defensive":  v({w: 0.20, s: 0.20, c: 0.30, b: 0.30}),  # gold + bond ballast
        "aggressive_growth":   v({w: 0.35, s: 0.50, b: 0.15}),           # risk-on
        "risk_parity_approx":  v({w: 0.25, s: 0.15, c: 0.30, b: 0.30}),  # equal-risk-ish
    }


def per_asset_regime_returns(
    asset_returns: pd.DataFrame,
    macro_panel: pd.DataFrame,
    classifier: Callable[[pd.Series], str] = classify_regime,
) -> pd.DataFrame:
    """Mean daily return per (regime, asset), estimated from the overlap of
    asset_returns and macro_panel. Returns a DataFrame indexed by regime, columns = assets."""
    monthly_regimes = macro_panel.apply(classifier, axis=1).rename("regime")
    # Forward-fill the month-end regime label to each trading day in asset_returns
    regime_daily = monthly_regimes.reindex(asset_returns.index, method="ffill")
    merged = asset_returns.copy()
    merged["regime"] = regime_daily
    merged = merged.dropna(subset=["regime"])
    return merged.groupby("regime").mean()


def build_payoff_matrix(
    regime_returns: pd.DataFrame,
    candidates: dict[str, pd.Series],
    annualize: bool = True,
) -> pd.DataFrame:
    """Rows = candidate portfolio names, columns = regime labels, cells = expected return.

    Scales by 252 when `annualize=True` so payoff units are annualized returns.
    """
    scale = 252.0 if annualize else 1.0
    rows = {}
    for name, w in candidates.items():
        # align w to regime_returns columns
        w_aligned = w.reindex(regime_returns.columns).fillna(0.0)
        rows[name] = (regime_returns * w_aligned.values).sum(axis=1) * scale
    return pd.DataFrame(rows).T.reindex(columns=list(REGIMES))


def nash_minimax_weights(payoff: pd.DataFrame) -> pd.Series:
    """Two-player zero-sum game: maximizer picks a mixed strategy over rows (candidate
    portfolios); adversary picks the worst regime (column). Returns the investor's
    mixed strategy p (Series indexed by candidate names) that maximizes the guaranteed
    worst-case regime return.

    LP:
        maximize   v
        s.t.       sum_i p_i * A[i, j] >= v   for each j
                   sum_i p_i = 1
                   p_i >= 0

    Rewritten for scipy.linprog (minimization):
        x = [p_1, ..., p_m, v]
        minimize c.T x  where c = [0, ..., 0, -1]
        A_ub @ x <= b_ub  (flip the >= inequality)
        A_eq @ x = b_eq   (sum(p)=1)
    """
    A = payoff.to_numpy(dtype=float)  # shape (m, r)
    m, r = A.shape
    # Variables: p_1..p_m, v
    c = np.zeros(m + 1)
    c[-1] = -1.0  # maximize v == minimize -v
    # Inequality: for each regime j: -sum_i p_i * A[i,j] + v <= 0
    A_ub = np.hstack([-A.T, np.ones((r, 1))])
    b_ub = np.zeros(r)
    # Equality: sum p_i = 1, v unconstrained
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])
    # Bounds: p_i in [0, 1], v free
    bounds = [(0.0, 1.0)] * m + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        # Fallback: best row-wise minimax (pure strategy) if LP fails
        worst_case = payoff.min(axis=1)
        best_pure = worst_case.idxmax()
        p = pd.Series(0.0, index=payoff.index)
        p[best_pure] = 1.0
        return p
    p = pd.Series(res.x[:m], index=payoff.index, name="p").clip(lower=0.0)
    return p / p.sum() if p.sum() > 0 else p


def mc_nash_asset_weights(
    panel_z: pd.DataFrame,
    asset_returns: pd.DataFrame,
    macro_panel: pd.DataFrame,
    symbols: list[str],
    horizon: int = 3,
    n_paths: int = 10_000,
    block_size: int = 3,
    bootstrap_window_months: int = 12,
    rng: np.random.Generator | None = None,
    classifier: Callable[[pd.Series], str] = classify_regime,
) -> tuple[pd.Series, dict[str, float], pd.DataFrame, pd.Series]:
    """End-to-end Track B weight fn.

    `bootstrap_window_months` — use only the trailing N months of macro z-scores as
    the bootstrap source (captures the *current* regime dynamics, not full history).

    Returns: (etf_weights, regime_probs, payoff_matrix, candidate_mix)
    """
    trailing = panel_z.tail(bootstrap_window_months)
    paths = block_bootstrap_paths(
        trailing, horizon=horizon, n_paths=n_paths, block_size=block_size, rng=rng
    )
    probs = regime_probabilities(paths, column_names=list(panel_z.columns), classifier=classifier)
    regime_rets = per_asset_regime_returns(asset_returns[symbols], macro_panel, classifier=classifier)
    candidates = candidate_portfolios(symbols)
    payoff = build_payoff_matrix(regime_rets, candidates)
    p_mix = nash_minimax_weights(payoff)
    # Convex combine candidate portfolios to final ETF weights
    etf_w = pd.Series(0.0, index=symbols)
    for name, pi in p_mix.items():
        if pi <= 0:
            continue
        etf_w = etf_w + pi * candidates[name].reindex(symbols).fillna(0.0)
    if etf_w.sum() > 0:
        etf_w = etf_w / etf_w.sum()
    return etf_w, probs, payoff, p_mix
