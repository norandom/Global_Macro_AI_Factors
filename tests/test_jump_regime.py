"""Tests for macro_framework.jump_regime — deterministic, seeded numpy only, no DB."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_framework.jump_regime import (
    BEAR,
    BULL,
    NEUTRAL,
    JumpRegimeConfig,
    classify_latest,
    fit_jump_model,
    fit_labels_walk_forward,
    sjm_features,
)

CFG = JumpRegimeConfig()


@pytest.fixture(scope="module")
def prices() -> pd.Series:
    """~500 calm-uptrend days (mean +0.08%, vol 0.6%) then ~500 downtrend days
    (mean -0.10%, vol 1.8%), cumulated to prices. Seeded => byte-reproducible."""
    rng = np.random.default_rng(7)
    ret = np.concatenate(
        [rng.normal(0.0008, 0.006, 500), rng.normal(-0.0010, 0.018, 500)]
    )
    idx = pd.bdate_range("2015-01-02", periods=1000)
    return pd.Series(100.0 * np.exp(np.cumsum(ret)), index=idx, name="px")


@pytest.fixture(scope="module")
def feats(prices: pd.Series) -> pd.DataFrame:
    return sjm_features(prices)


def test_sjm_features_shape_and_trend_first(feats: pd.DataFrame) -> None:
    assert feats.shape[1] == 3
    assert feats.columns[0] == "ewma_ret"  # trend feature at index 0 == trend_feature=0
    assert not feats.isna().any().any()


def test_full_fit_classifies_last_point_bear(feats: pd.DataFrame) -> None:
    x = feats.to_numpy(dtype=np.float64)
    model = fit_jump_model(x, CFG)
    assert classify_latest(model, x[-1], CFG) == BEAR


def test_bull_only_window_classifies_bull(feats: pd.DataFrame) -> None:
    x = feats.to_numpy(dtype=np.float64)[:400]  # entirely inside the calm uptrend
    model = fit_jump_model(x, CFG)
    assert classify_latest(model, x[-1], CFG) == BULL


def test_persistence_few_switches(feats: pd.DataFrame) -> None:
    model = fit_jump_model(feats.to_numpy(dtype=np.float64), CFG)
    switches = int((model.states[1:] != model.states[:-1]).sum())
    assert switches < 10  # lam=50 buys persistence over ~1000 obs


def test_walk_forward_pit_ignores_future(feats: pd.DataFrame) -> None:
    """The label at d must be identical with and without future rows — even when the
    future is a violent regime flip appended right after d."""
    d = feats.index[550]
    base = feats.loc[feats.index <= d]
    # violent flip strictly after d: crash-like trend, huge vol
    future_idx = pd.bdate_range(d + pd.Timedelta(days=1), periods=60)
    flip = pd.DataFrame(
        {"ewma_ret": -0.05, "ewma_vol": 0.10, "ewma_downside_vol": 0.10}, index=future_idx
    )
    extended = pd.concat([base, flip])

    lab_base = fit_labels_walk_forward(base, [d])
    lab_ext = fit_labels_walk_forward(extended, [d])
    assert lab_base.loc[d] == lab_ext.loc[d]
    assert lab_base.loc[d] in (BULL, BEAR, NEUTRAL)


def test_walk_forward_fail_open_neutral(feats: pd.DataFrame) -> None:
    d = feats.index[100]  # only 100 rows strictly before d < min_obs=504
    labels = fit_labels_walk_forward(feats, [d])
    assert labels.loc[d] == NEUTRAL


def test_determinism_identical_fits(feats: pd.DataFrame) -> None:
    x = feats.to_numpy(dtype=np.float64)
    a = fit_jump_model(x, CFG)
    b = fit_jump_model(x, CFG)
    assert np.array_equal(a.centroids, b.centroids)
    assert np.array_equal(a.states, b.states)
    assert a.objective == b.objective
