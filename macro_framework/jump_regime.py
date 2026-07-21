"""Sparse Jump Model regime detector (port of Shu & Mulvey 2024 / Nystrup et al. SJM) — pure math.

Ported from facdrone ``library/models/jump_regime.py``. DEVIATION from the source: facdrone
labelled regimes ``trending`` / ``mean_reverting`` / ``neutral`` (imported from its
``trend_regime`` module); this repo relabels the SAME ordering logic as ``bull`` / ``bear`` /
``neutral`` — the centroid HIGHEST on the trend axis -> bull, lowest -> bear, and the confidence
guard -> neutral. Only the strings differ; the centroid-order logic is identical.

A statistical jump model clusters temporal feature vectors ``x_0..x_{T-1}`` into ``K`` states while
penalising state transitions, which buys regime *persistence* (the property a composite trend
threshold lacks). Objective:

    min over centroids {theta_k} and states {s_t}:
        sum_t 0.5 * ||x_t - theta_{s_t}||_w^2  +  lambda * sum_t 1{s_t != s_{t-1}}

solved by coordinate descent: (i) fix centroids -> exact DP/Viterbi over states (emission
``0.5||x-theta||_w^2``, transition cost ``lambda``) gives the global-optimal sequence; (ii) fix
states -> centroids = cluster means; iterate to a fixed point. The Sparse extension (``kappa`` set)
re-weights features by between-cluster sum of squares, soft-thresholded onto an L1 ball so
uninformative features get weight 0; ``kappa=None`` (default) is the plain JM (uniform weights).

DETERMINISM is a hard requirement — the sim is seeded and must stay byte-reproducible. This model
uses NO RNG at all: the init is a deterministic seeding at quantiles of the trend feature (robust to
noise features that would dominate a k-means++ distance), the DP tie-break prefers the lower state
index AND prefers *staying*, and the public label is assigned by centroid order on the trend axis
(never by fit/discovery order). The
model fails open to ``neutral`` — which here is a *confidence guard* (regimes too close on the trend
axis, or the latest point assigned with too small a margin), NOT a learned third cluster.

Repo-specific additions: ``sjm_features`` (3 EWMA features from a price series, trend first) and
``fit_labels_walk_forward`` (point-in-time refit wrapper — never fits on data at/after the
decision date).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]

_STD_FLOOR = 1e-8

BULL = "bull"
BEAR = "bear"
NEUTRAL = "neutral"


@dataclass(frozen=True)
class JumpRegimeConfig:
    """Jump-model hyperparameters (frozen)."""

    k: int = 2  # bull/bear (neutral is a confidence guard, not a learned state)
    lam: float = 50.0  # jump penalty — high => persistent regimes (the whole point)
    kappa: float | None = None  # L1 feature-budget; None => plain JM (uniform weights), SJM off
    max_iter: int = 30
    tol: float = 1e-6
    trend_feature: int = 0  # column used to order centroids -> labels (the lead EWMA return)
    min_centroid_gap: float = 0.5  # z-units: trend-axis separation below this => neutral
    min_assign_margin: float = 0.0  # latest-point assignment cost margin below this => neutral


@dataclass(frozen=True)
class JumpRegimeModel:
    """A fitted jump model (centroids/weights/scaler in standardized feature space)."""

    centroids: FloatArray  # (K, F)
    feature_weights: FloatArray  # (F,)
    scaler_mean: FloatArray  # (F,)
    scaler_std: FloatArray  # (F,)
    states: IntArray  # (T,) in-sample assignment
    objective: float


def _standardize(features: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    x = np.asarray(features, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < _STD_FLOOR, 1.0, std)
    return (x - mean) / std, mean, std


def _emission(x: FloatArray, centroids: FloatArray, w: FloatArray) -> FloatArray:
    """(T, K) weighted half-squared distance ``0.5 * sum_j w_j (x_tj - c_kj)^2``."""
    diff = x[:, None, :] - centroids[None, :, :]
    return np.asarray(0.5 * (diff**2 * w[None, None, :]).sum(axis=2), dtype=np.float64)


def _dp_states(emission: FloatArray, lam: float) -> IntArray:
    """Exact DP for the global-optimal state sequence under the jump penalty. Tie-break: prefer
    the lower predecessor index, and prefer *staying* in the same state (an epsilon nudge that
    affects only ties, never the reported objective)."""
    t_len, k = emission.shape
    cost = np.empty((t_len, k), dtype=np.float64)
    back = np.zeros((t_len, k), dtype=np.int_)
    cost[0] = emission[0]
    arange_k = np.arange(k)
    for t in range(1, t_len):
        prev = cost[t - 1]
        for state in range(k):
            trans = prev + lam * (arange_k != state)
            tie = trans.copy()
            tie[state] -= 1e-12  # prefer staying on exact ties (does not change the true cost)
            best = int(np.argmin(tie))  # np.argmin -> lowest index among remaining ties
            back[t, state] = best
            cost[t, state] = emission[t, state] + prev[best] + lam * float(best != state)
    states = np.empty(t_len, dtype=np.int_)
    states[-1] = int(np.argmin(cost[-1]))
    for t in range(t_len - 2, -1, -1):
        states[t] = back[t + 1, states[t + 1]]
    return states


def _objective(emission: FloatArray, states: IntArray, lam: float) -> float:
    fit = float(emission[np.arange(len(states)), states].sum())
    jumps = float((states[1:] != states[:-1]).sum())
    return fit + lam * jumps


def _quantile_init(x: FloatArray, k: int, trend_feature: int) -> FloatArray:
    """Deterministic, noise-robust init: seed the K centroids at evenly-spaced quantiles of the
    TREND feature (the designated regime axis), so the init separates along the signal rather than
    along noise features that can dominate a k-means++ distance. Picks the actual data points at
    those quantiles, so the init is a real observation in every dimension."""
    order = np.argsort(x[:, trend_feature], kind="stable")
    idx = [int(order[min(int((i + 0.5) / k * len(order)), len(order) - 1)]) for i in range(k)]
    return x[idx].copy()


def _reseed_empty(
    x: FloatArray, centroids: FloatArray, states: IntArray, w: FloatArray, lam: float
) -> tuple[FloatArray, IntArray, FloatArray]:
    """Safety net: re-seed any empty cluster to the current worst-fit point (max assigned emission,
    lowest index on tie) and re-assign. Deterministic; bounds at K passes."""
    emission = _emission(x, centroids, w)
    for c in range(centroids.shape[0]):
        if not (states == c).any():
            worst = int(np.argmax(emission[np.arange(len(states)), states]))
            centroids = centroids.copy()
            centroids[c] = x[worst]
            emission = _emission(x, centroids, w)
            states = _dp_states(emission, lam)
    return centroids, states, emission


def _fit_once(
    x: FloatArray, w: FloatArray, cfg: JumpRegimeConfig
) -> tuple[FloatArray, IntArray, float]:
    centroids = _quantile_init(x, cfg.k, cfg.trend_feature)
    states = _dp_states(_emission(x, centroids, w), cfg.lam)
    centroids, states, emission = _reseed_empty(x, centroids, states, w, cfg.lam)
    prev_obj = float("inf")
    obj = prev_obj
    for _ in range(cfg.max_iter):
        new = centroids.copy()
        for c in range(cfg.k):
            mask = states == c
            if mask.any():
                new[c] = x[mask].mean(axis=0)
        centroids = new
        states = _dp_states(_emission(x, centroids, w), cfg.lam)
        centroids, states, emission = _reseed_empty(x, centroids, states, w, cfg.lam)
        obj = _objective(emission, states, cfg.lam)
        if abs(prev_obj - obj) < cfg.tol:
            break
        prev_obj = obj
    return centroids, states, obj


def _sparse_weights(x: FloatArray, states: IntArray, kappa: float) -> FloatArray:
    """Sparse feature weights (Witten-Tibshirani style): maximise w.BCSS s.t. ||w||_2<=1,
    ||w||_1<=kappa, w>=0 — soft-threshold BCSS and L2-normalise, with the threshold found by
    bisection so ||w||_1 == kappa. Uninformative (low-BCSS) features go to exactly 0."""
    overall = x.mean(axis=0)
    bcss = np.zeros(x.shape[1], dtype=np.float64)
    for c in np.unique(states):
        members = x[states == c]
        bcss += members.shape[0] * (members.mean(axis=0) - overall) ** 2
    a = np.maximum(bcss, 0.0)
    norm = float(np.linalg.norm(a))
    if norm == 0.0:
        return np.ones(x.shape[1], dtype=np.float64)
    if float((a / norm).sum()) <= kappa:  # L1 already within budget -> no thresholding
        return a / norm

    def weights_for(delta: float) -> FloatArray:
        st = np.maximum(a - delta, 0.0)
        n = float(np.linalg.norm(st))
        return st / n if n > 0 else st

    lo, hi = 0.0, float(a.max())
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if float(weights_for(mid).sum()) > kappa:
            lo = mid
        else:
            hi = mid
    return weights_for(hi)


def fit_jump_model(features: FloatArray, config: JumpRegimeConfig) -> JumpRegimeModel:
    """Fit the (sparse) jump model on the trailing in-sample window. Pure + deterministic."""
    x, mean, std = _standardize(features)
    n_features = x.shape[1]
    w: FloatArray = np.ones(n_features, dtype=np.float64)
    centroids, states, obj = _fit_once(x, w, config)
    if config.kappa is not None:  # SJM: alternate feature-weight update with the JM fit
        for _ in range(10):
            w_new = _sparse_weights(x, states, config.kappa)
            converged = float(np.abs(w_new - w).sum()) < 1e-4
            w = w_new
            centroids, states, obj = _fit_once(x, w, config)
            if converged:
                break
    return JumpRegimeModel(
        centroids=centroids,
        feature_weights=w,
        scaler_mean=mean,
        scaler_std=std,
        states=states,
        objective=obj,
    )


def assign_latest(model: JumpRegimeModel, latest_features: FloatArray) -> int:
    """Nearest weighted centroid for one new (raw) feature vector, standardized by the fit stats."""
    x = (np.asarray(latest_features, dtype=np.float64) - model.scaler_mean) / model.scaler_std
    dist = ((x[None, :] - model.centroids) ** 2 * model.feature_weights[None, :]).sum(axis=1)
    return int(np.argmin(dist))


def model_label(model: JumpRegimeModel, state_idx: int, config: JumpRegimeConfig) -> str:
    """Map a state to the canonical vocabulary by CENTROID ORDER on the trend axis (never fit
    order): highest -> bull, lowest -> bear, middle (K>2) -> neutral. A trend-axis
    centroid gap below ``min_centroid_gap`` => neutral (regimes not meaningfully separated)."""
    centroids = model.centroids
    if centroids.shape[0] < 2:
        return NEUTRAL
    trend = centroids[:, config.trend_feature]
    order = np.argsort(trend, kind="stable")
    if float(trend[order[-1]] - trend[order[0]]) < config.min_centroid_gap:
        return NEUTRAL
    if state_idx == int(order[-1]):
        return BULL
    if state_idx == int(order[0]):
        return BEAR
    return NEUTRAL


def classify_latest(
    model: JumpRegimeModel, latest_features: FloatArray, config: JumpRegimeConfig
) -> str:
    """Online label for the current period: assign to the nearest weighted centroid, fail open to
    neutral when the two nearest states are within ``min_assign_margin`` (low confidence), then map
    via ``model_label`` (which also applies the trend-axis separation guard)."""
    x = (np.asarray(latest_features, dtype=np.float64) - model.scaler_mean) / model.scaler_std
    dist = ((x[None, :] - model.centroids) ** 2 * model.feature_weights[None, :]).sum(axis=1)
    state = int(np.argmin(dist))
    ordered = np.sort(dist)
    if ordered.size >= 2 and float(ordered[1] - ordered[0]) < config.min_assign_margin:
        return NEUTRAL
    return model_label(model, state, config)


# --- repo-specific helpers (not in the facdrone source) -------------------------------------


def sjm_features(
    prices: pd.Series, *, halflife_ret: int = 21, halflife_vol: int = 21
) -> pd.DataFrame:
    """The 3 standard SJM features from one price series. Column 0 is the TREND feature
    (matches ``JumpRegimeConfig.trend_feature = 0``): EWMA daily log-return. Column 1: EWMA vol
    of daily log-returns. Column 2: EWMA downside vol (negative returns only). Warm-up NaNs
    dropped. Pure pandas, deterministic."""
    ret = np.log(prices.astype(np.float64)).diff()
    ewma_ret = ret.ewm(halflife=halflife_ret).mean()
    ewma_vol = np.sqrt(ret.pow(2).ewm(halflife=halflife_vol).mean())
    downside = ret.where(ret < 0, 0.0)
    ewma_dvol = np.sqrt(downside.pow(2).ewm(halflife=halflife_vol).mean())
    out = pd.DataFrame(
        {"ewma_ret": ewma_ret, "ewma_vol": ewma_vol, "ewma_downside_vol": ewma_dvol}
    )
    return out.dropna()


def fit_labels_walk_forward(
    features: pd.DataFrame,
    dates,
    *,
    config: JumpRegimeConfig = JumpRegimeConfig(),
    min_obs: int = 504,
) -> pd.Series:
    """Point-in-time walk-forward labels: for each decision date d, fit ONLY on feature rows
    strictly before d and classify the last row of that window. Fewer than ``min_obs`` rows =>
    fail open to neutral. Never fits on data at/after d — the PIT discipline the facdrone
    lookahead incident demanded."""
    labels = []
    for d in dates:
        window = features.loc[features.index < d]
        if len(window) < min_obs:
            labels.append(NEUTRAL)
            continue
        model = fit_jump_model(window.to_numpy(dtype=np.float64), config)
        labels.append(classify_latest(model, window.iloc[-1].to_numpy(dtype=np.float64), config))
    return pd.Series(labels, index=pd.Index(dates))
