"""Task 4.2: optional de-risk overlay at the weight-combination seam.

Tests the extracted `_regime_cash_pin` helper directly (no NIM/API, no main()).
The overlay defaults OFF and must be byte-identical (float ==) to the published
constant 0.25; when ON it raises the BIL pin in a high-correlation window and
leaves it at the base pin in a calm window.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

RISKY = ("SWDA.L", "XLK", "IAU")
OVERLAY = {"min_scale": 0.20}


def _mod():
    import extend_stream_2026 as ext

    return ext


def _returns(corr_target: str, n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    if corr_target == "high":
        common = rng.standard_normal(n)
        cols = {s: 0.02 * (common + 0.15 * rng.standard_normal(n)) for s in RISKY}
    else:  # calm: independent risky sleeves
        cols = {s: 0.01 * rng.standard_normal(n) for s in RISKY}
    cols["BIL"] = 0.0002 * rng.standard_normal(n)
    return pd.DataFrame(cols, index=idx)


def test_overlay_off_is_byte_identical_base_pin() -> None:
    ext = _mod()
    r = _returns("high")
    assert ext._regime_cash_pin(r, None) == 0.25  # exact float equality


def test_overlay_on_raises_pin_in_high_correlation_window() -> None:
    ext = _mod()
    r = _returns("high")
    pin = ext._regime_cash_pin(r, OVERLAY)
    assert 0.25 < pin < 1.0


def test_overlay_on_stays_at_base_pin_in_calm_window() -> None:
    ext = _mod()
    r = _returns("calm")
    pin = ext._regime_cash_pin(r, OVERLAY)
    assert pin == 0.25


def test_module_default_overlay_flag_is_off() -> None:
    ext = _mod()
    assert ext.REGIME_OVERLAY is None
