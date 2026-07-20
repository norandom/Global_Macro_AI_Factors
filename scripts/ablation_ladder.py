"""Ablation ladder: score each pipeline rung on the own-basket skill metric.

Scores five rungs on ONE out-of-sample window (Req 4.1, 4.2, 6.5):
  HRP-only -> HRP+BL(fixed view) -> HRP+BL(AI point-in-time view)
  -> HRP+BL(AI non-point-in-time diagnostic view), PLUS the non-LLM regime
  de-risk overlay as a control rung.

Each rung is scored on the own-basket appraisal ratio (timing skill, via
``macro_framework.skill_metric.basket_residual``) AND risk-shape metrics
(Calmar, max drawdown, via ``workbook.factor_workbook.rederive.equity_metrics``).
Reports the AI-view marginal delta (AI-PIT minus fixed, 4.3) and the
AI-minus-control skill difference (AI-PIT minus overlay control, 6.5).

The scoring core (``score_ladder`` and the two delta helpers) is pure,
deterministic, and unit-tested on synthetic series — equal inputs yield equal
output, no IO. Producing the rung return series from the walk-forward needs the
DB/NIM pipeline, so it lives under ``if __name__ == "__main__"`` with lazy
imports and is NOT imported or unit-tested here. Diagnostic-only labeling and
artifact writing are task 5.2, not this module.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

from macro_framework.skill_metric import basket_residual

# ``factor_workbook`` is a lean, uninstalled sibling package whose __init__ pulls in
# pyxll (addin); ``rederive`` itself is standalone (numpy/pandas/stdlib only), so we
# load it in isolation by path to reuse equity_metrics without that heavy chain and
# keep `import ablation_ladder` DB/GUI-free (Req 8.2: reuse, don't re-implement).
_REDERIVE_PATH = (
    Path(__file__).resolve().parents[1] / "workbook" / "factor_workbook" / "rederive.py"
)
_spec = importlib.util.spec_from_file_location("_fw_rederive", _REDERIVE_PATH)
_rederive = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _rederive  # dataclass forward-ref resolution needs this
_spec.loader.exec_module(_rederive)
equity_metrics = _rederive.equity_metrics

# Canonical rung names (task 5.2 labels HRP_BL_AI_NONPIT diagnostic-only).
HRP_ONLY = "HRP_ONLY"
HRP_BL_FIXED = "HRP_BL_FIXED"
HRP_BL_AI_PIT = "HRP_BL_AI_PIT"
HRP_BL_AI_NONPIT = "HRP_BL_AI_NONPIT"
OVERLAY_CONTROL = "OVERLAY_CONTROL"

#: Canonical row order for the ladder table.
RUNG_ORDER = (HRP_ONLY, HRP_BL_FIXED, HRP_BL_AI_PIT, HRP_BL_AI_NONPIT, OVERLAY_CONTROL)


def _equity_from_returns(returns: pd.Series) -> pd.Series:
    """Cumulative equity line from a daily-return series (start at 1.0)."""
    return (1.0 + returns.fillna(0.0)).cumprod()


def score_ladder(
    rung_returns: dict[str, pd.Series],
    factor_returns: pd.DataFrame,
    *,
    benchmark_returns: pd.Series | None = None,  # ponytail: reserved for 5.2 market attr; unused here
    periods_per_year: int = 252,
    hac_maxlags: int = 5,
) -> pd.DataFrame:
    """Score every rung on skill + risk-shape metrics over one OOS window.

    Args:
        rung_returns: ``{rung_name: daily_return_series}``. Any subset of the
            canonical rungs (including ``OVERLAY_CONTROL``) is accepted; rows are
            ordered by ``RUNG_ORDER`` with unknown names appended.
        factor_returns: The own 4-ETF daily returns the rungs are regressed on.
        periods_per_year: Annualization basis for the appraisal metric (√252).
        hac_maxlags: Newey-West HAC lag count for the alpha t-statistic.

    Returns:
        A DataFrame indexed by rung name with columns: ``appraisal``,
        ``alpha_ann``, ``t_alpha_hac``, ``r2``, ``idio_vol_ann`` (own-basket
        skill) and ``calmar``, ``max_drawdown`` (risk-shape). Pure/deterministic.
    """
    rows: dict[str, dict[str, float]] = {}
    for name, returns in rung_returns.items():
        res = basket_residual(
            returns, factor_returns,
            periods_per_year=periods_per_year, hac_maxlags=hac_maxlags,
        )
        eq = equity_metrics(_equity_from_returns(returns))
        rows[name] = {
            "appraisal": float("nan") if res.appraisal is None else res.appraisal,
            "alpha_ann": res.alpha_ann,
            "t_alpha_hac": res.t_alpha_hac,
            "r2": res.r2,
            "idio_vol_ann": res.idio_vol_ann,
            "calmar": eq.calmar,
            "max_drawdown": eq.max_drawdown,
        }
    order = [n for n in RUNG_ORDER if n in rows] + [n for n in rows if n not in RUNG_ORDER]
    return pd.DataFrame.from_dict(rows, orient="index").loc[order]


def ai_view_marginal_delta(table: pd.DataFrame) -> float:
    """AI-view marginal contribution: appraisal(AI-PIT) - appraisal(BL-fixed) (4.3)."""
    return float(table.loc[HRP_BL_AI_PIT, "appraisal"] - table.loc[HRP_BL_FIXED, "appraisal"])


def ai_minus_control(table: pd.DataFrame) -> float:
    """AI-minus-control skill difference: appraisal(AI-PIT) - appraisal(overlay control) (6.5)."""
    return float(table.loc[HRP_BL_AI_PIT, "appraisal"] - table.loc[OVERLAY_CONTROL, "appraisal"])


if __name__ == "__main__":  # pragma: no cover -- DB/NIM data run; NOT unit-tested
    # ponytail: the full 5-rung walk-forward reuses the extend_stream_2026 pipeline
    # (build_walk_forward_targets + the combine closure + _ReplayScorer for the AI
    # rungs + derisk_cash_pin for the control). Task 5.2 owns the run+artifacts; this
    # block only wires already-built rung return series into score_ladder and prints
    # the table. All heavy imports stay lazy so `import ablation_ladder` needs no DB.
    import argparse
    import json

    import numpy as np

    parser = argparse.ArgumentParser(description="Score the ablation ladder on one OOS window.")
    parser.add_argument(
        "--rung-returns", required=True,
        help="JSON {rung_name: {isodate: daily_return}} for the OOS window.",
    )
    parser.add_argument(
        "--factor-returns", required=True,
        help="JSON {etf: {isodate: daily_return}} of the own 4-ETF factors.",
    )
    args = parser.parse_args()

    def _load_series_frame(path: str) -> pd.DataFrame:
        with open(path) as fh:
            payload = json.load(fh)
        return pd.DataFrame(payload).apply(pd.to_numeric).sort_index().set_axis(
            pd.to_datetime(pd.DataFrame(payload).index), axis=0
        )

    factors = _load_series_frame(args.factor_returns)
    rungs = {name: col.dropna() for name, col in _load_series_frame(args.rung_returns).items()}

    ladder = score_ladder(rungs, factors)
    with pd.option_context("display.float_format", lambda v: f"{v: .4f}"):
        print(ladder.to_string())
    print(f"\nAI-view marginal delta (AI-PIT - BL-fixed):   {ai_view_marginal_delta(ladder): .4f}")
    print(f"AI-minus-control skill diff (AI-PIT - overlay): {ai_minus_control(ladder): .4f}")
