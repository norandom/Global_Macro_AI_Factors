"""MBB inference on the SSR (paper §3.3.2-3.3.3, Test 1): determinism + discrimination."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_framework.ssr import politis_white_block_length, ssr_inference


def _series(mu: float, n: int = 1600, seed: int = 5) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, 0.01, n))


def test_deterministic_given_seed():
    r = _series(0.0005)
    a = ssr_inference(r, n_boot=100, seed=3)
    b = ssr_inference(r, n_boot=100, seed=3)
    assert a.p_value == b.p_value and a.block_len == b.block_len


def test_strong_drift_is_stable_and_noise_is_not():
    strong = ssr_inference(_series(0.0012), n_boot=200)  # ann. Sharpe ~1.9
    noise = ssr_inference(_series(0.0), n_boot=200)
    assert strong.stable and strong.p_value < 0.05
    assert not noise.stable and noise.p_value > 0.05
    assert "luck-compatible" in noise.verdict()
    assert "not a skill claim" in strong.verdict()


def test_negative_drift_never_passes():
    down = ssr_inference(_series(-0.0012), n_boot=200)
    assert not down.stable  # the old abs(SSR) rule would have branded this "stably > 0"


def test_short_series_fails_open():
    inf = ssr_inference(_series(0.001, n=100), n_boot=50)
    assert not inf.stable and np.isnan(inf.p_value)
    assert inf.verdict() == "insufficient rolling observations for inference"


def test_politis_white_bounds():
    rng = np.random.default_rng(0)
    white = rng.normal(size=2000)
    ar = np.empty(2000)
    ar[0] = 0.0
    for t in range(1, 2000):
        ar[t] = 0.9 * ar[t - 1] + rng.normal()
    b_white = politis_white_block_length(white)
    b_ar = politis_white_block_length(ar)
    assert 1 <= b_white <= 5  # white noise -> (near-)iid bootstrap
    assert b_ar > b_white  # persistence -> longer blocks
    assert b_ar <= int(np.ceil(min(3 * np.sqrt(2000), 2000 / 3)))
