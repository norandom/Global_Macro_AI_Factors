from __future__ import annotations

import pandas as pd
import pytest

from macro_framework.evaluation import anticipation_lead_time


def _targets(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([d for d, *_ in rows])
    return pd.DataFrame(
        {"BIL": [bil for _, bil, _ in rows], "IAU": [iau for _, _, iau in rows]},
        index=idx,
    )


def test_anticipation_lead_time_requires_upward_crossing() -> None:
    tgt = _targets(
        [
            ("2019-01-02", 0.10, 0.10),
            ("2019-02-01", 0.15, 0.10),
            ("2019-03-01", 0.25, 0.20),
        ]
    )
    assert anticipation_lead_time(tgt, threshold=0.40) == pd.Timestamp("2019-03-01")


def test_anticipation_lead_time_ignores_already_defensive_start() -> None:
    tgt = _targets(
        [
            ("2019-01-02", 0.25, 0.20),
            ("2019-02-01", 0.22, 0.20),
            ("2019-03-01", 0.30, 0.15),
        ]
    )
    assert anticipation_lead_time(tgt, threshold=0.40) is None


def test_anticipation_lead_time_returns_none_without_crossing() -> None:
    tgt = _targets(
        [
            ("2019-01-02", 0.10, 0.05),
            ("2019-02-01", 0.12, 0.08),
            ("2019-03-01", 0.14, 0.09),
        ]
    )
    assert anticipation_lead_time(tgt, threshold=0.40) is None


# --- metric convention binding (the anti-drift check) ------------------------- #


def _equity(n: int = 800, seed: int = 0) -> pd.Series:
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-02", periods=n)
    return pd.Series(10_000.0 * (1 + rng.normal(3e-4, 8e-3, n)).cumprod(), index=idx)


def test_the_two_metric_conventions_differ_only_as_documented() -> None:
    """The repo runs exactly two metric engines. Pin every way they differ.

    ``factor_workbook.rederive.equity_metrics`` built the released tear sheet and
    must keep vectorbt parity; ``macro_framework.evaluation.metric_block`` is what
    the notebooks use. Three differences are deliberate and enumerated below. Any
    FOURTH difference — or a change to one of these three — fails here, which is
    the point: a table assembled from both engines silently inflates every
    vol-scaled figure by sqrt(365/252) and every annualized mean by 365/252.
    """
    import math
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workbook"))
    from factor_workbook.rederive import equity_metrics

    from macro_framework.evaluation import metric_block

    eq = _equity()
    mine, theirs = metric_block(eq), equity_metrics(eq)

    # (1) rederive injects a synthetic day-0 return of 0.0 (vectorbt convention);
    #     metric_block drops it.
    r = pd.concat([pd.Series([0.0], index=eq.index[:1]), mine["returns"]])
    std = float(r.std(ddof=1))

    # (2) rederive annualizes on the 365-day calendar year, metric_block on 252.
    assert theirs.annualized_vol == pytest.approx(std * math.sqrt(365), rel=1e-12)
    assert theirs.sharpe == pytest.approx(
        float(r.mean()) / std * math.sqrt(365), rel=1e-12
    )

    # (3) rederive's CAGR annualizes on ROW COUNT; metric_block on elapsed days.
    assert theirs.annualized_return == pytest.approx(
        (1.0 + mine["total_return"]) ** (365 / len(eq)) - 1.0, rel=1e-12
    )

    # everything basis-free must agree exactly
    assert mine["maxdd"] == pytest.approx(theirs.max_drawdown, rel=1e-12)
    assert mine["total_return"] == pytest.approx(theirs.total_return, rel=1e-12)


def test_metric_block_carries_both_bases_with_the_exact_ratio() -> None:
    """Both annualizations ship side by side; the ratio between them is fixed."""
    import math

    from macro_framework.evaluation import CALENDAR_DAYS, TRADING_DAYS, metric_block

    m = metric_block(_equity())
    ratio = math.sqrt(CALENDAR_DAYS / TRADING_DAYS)
    assert m["ann_vol_cal"] == pytest.approx(m["ann_vol"] * ratio, rel=1e-12)
    assert m["sharpe_cal"] == pytest.approx(m["sharpe"] * ratio, rel=1e-12)
    assert m["sortino_cal"] == pytest.approx(m["sortino"] * ratio, rel=1e-12)
    # the two growth bases are genuinely different numbers, not aliases
    assert m["cagr"] != pytest.approx(m["cagr_rows"], rel=1e-6)
    assert m["calmar"] == pytest.approx(m["cagr"] / abs(m["maxdd"]), rel=1e-12)
    assert m["calmar_rows"] == pytest.approx(m["cagr_rows"] / abs(m["maxdd"]), rel=1e-12)


def test_metric_block_cagr_uses_elapsed_calendar_not_row_count() -> None:
    """rederive annualizes on ROW COUNT; metric_block on ELAPSED DAYS. Pin both."""
    from macro_framework.evaluation import cagr, calmar, max_drawdown, metric_block

    eq = _equity()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    expected = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    assert cagr(eq) == pytest.approx(expected, rel=1e-12)
    assert metric_block(eq)["cagr"] == pytest.approx(expected, rel=1e-12)
    assert calmar(eq) == pytest.approx(cagr(eq) / abs(max_drawdown(eq)), rel=1e-12)
