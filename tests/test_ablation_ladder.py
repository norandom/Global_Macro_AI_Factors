"""Tests for scripts/ablation_ladder — pure ladder-scoring core (no DB/NIM).

Synthetic daily returns per rung are known linear combinations of synthetic
4-ETF factor returns plus an injected timing alpha: the AI-PIT rung gets a
LARGER injected alpha than the fixed rung, and the overlay control gets a
smaller/zero alpha. Asserts the per-rung skill table, the AI-view marginal
delta, and the AI-minus-control skill difference (Req 4.1-4.3, 6.5).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
ablation_ladder = importlib.import_module("ablation_ladder")

score_ladder = ablation_ladder.score_ladder
ai_view_marginal_delta = ablation_ladder.ai_view_marginal_delta
ai_minus_control = ablation_ladder.ai_minus_control
HRP_ONLY = ablation_ladder.HRP_ONLY
HRP_BL_FIXED = ablation_ladder.HRP_BL_FIXED
HRP_BL_AI_PIT = ablation_ladder.HRP_BL_AI_PIT
HRP_BL_AI_NONPIT = ablation_ladder.HRP_BL_AI_NONPIT
OVERLAY_CONTROL = ablation_ladder.OVERLAY_CONTROL

RNG = np.random.default_rng(20260720)
FACTORS = ("SWDA.L", "XLK", "IAU", "BIL")


def _factor_frame(n: int = 900) -> pd.DataFrame:
    idx = pd.bdate_range("2019-01-01", periods=n)
    data = {c: RNG.normal(0.0003, 0.01, n) for c in FACTORS}
    return pd.DataFrame(data, index=idx)


def _rung(factors: pd.DataFrame, *, alpha_daily: float, beta: float = 0.8) -> pd.Series:
    """Strategy = injected alpha + beta*XLK + small idio noise (above the vol floor)."""
    return (
        alpha_daily
        + beta * factors["XLK"]
        + RNG.normal(0.0, 0.0006, len(factors))
    )


def _rung_returns(factors: pd.DataFrame) -> dict[str, pd.Series]:
    # AI-PIT gets the largest alpha; fixed less; overlay control ~zero.
    return {
        HRP_ONLY: _rung(factors, alpha_daily=0.00005),
        HRP_BL_FIXED: _rung(factors, alpha_daily=0.00015),
        HRP_BL_AI_PIT: _rung(factors, alpha_daily=0.00040),
        HRP_BL_AI_NONPIT: _rung(factors, alpha_daily=0.00045),
        OVERLAY_CONTROL: _rung(factors, alpha_daily=0.00002),
    }


def test_row_per_rung_with_finite_skill_and_risk_shape():
    factors = _factor_frame()
    table = score_ladder(_rung_returns(factors), factors)
    # a row per rung, including the overlay control (4.1, 4.2, 6.5)
    assert set(table.index) == {
        HRP_ONLY, HRP_BL_FIXED, HRP_BL_AI_PIT, HRP_BL_AI_NONPIT, OVERLAY_CONTROL
    }
    for col in ("appraisal", "alpha_ann", "t_alpha_hac", "r2", "idio_vol_ann",
                "calmar", "max_drawdown"):
        assert col in table.columns
    assert np.isfinite(table["appraisal"]).all()
    assert np.isfinite(table["calmar"]).all()
    # overlay control specifically has finite appraisal + calmar
    assert np.isfinite(table.loc[OVERLAY_CONTROL, "appraisal"])
    assert np.isfinite(table.loc[OVERLAY_CONTROL, "calmar"])


def test_ai_view_marginal_delta_positive_when_ai_beats_fixed():
    factors = _factor_frame()
    table = score_ladder(_rung_returns(factors), factors)
    delta = ai_view_marginal_delta(table)
    # AI-PIT injected more alpha than fixed -> positive marginal skill delta (4.3)
    assert delta > 0.0
    assert np.isclose(
        delta,
        table.loc[HRP_BL_AI_PIT, "appraisal"] - table.loc[HRP_BL_FIXED, "appraisal"],
    )


def test_ai_minus_control_positive_when_ai_beats_control():
    factors = _factor_frame()
    table = score_ladder(_rung_returns(factors), factors)
    diff = ai_minus_control(table)
    # AI-PIT beats the near-zero-alpha overlay control on skill (6.5)
    assert diff > 0.0
    assert np.isclose(
        diff,
        table.loc[HRP_BL_AI_PIT, "appraisal"] - table.loc[OVERLAY_CONTROL, "appraisal"],
    )


def test_deterministic_same_inputs_same_table():
    factors = _factor_frame()
    rungs = _rung_returns(factors)
    t1 = score_ladder(rungs, factors)
    t2 = score_ladder(rungs, factors)
    pd.testing.assert_frame_equal(t1, t2)


def test_imports_without_db():
    # The module must import and expose the pure core without any DB/API key.
    assert hasattr(ablation_ladder, "score_ladder")
    assert callable(ablation_ladder.score_ladder)
