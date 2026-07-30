from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from macro_framework.evaluation import (
    CrisisMetrics,
    anticipation_lead_time,
    crisis_analytics,
    crisis_metrics,
)


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


def test_crisis_metrics_include_the_return_entering_the_first_session() -> None:
    value = pd.Series(
        [100.0, 80.0, 88.0, 92.0],
        index=pd.to_datetime(["2021-12-31", "2022-01-03", "2022-01-04", "2022-01-10"]),
    )

    result = crisis_metrics(value, "2022-01-01", "2022-01-04")

    assert isinstance(result, CrisisMetrics)
    assert result.requested_start == pd.Timestamp("2022-01-01")
    assert result.requested_end == pd.Timestamp("2022-01-04")
    assert result.anchor == pd.Timestamp("2021-12-31")
    assert result.first_return_date == pd.Timestamp("2022-01-03")
    assert result.actual_end == pd.Timestamp("2022-01-04")
    assert result.episode_return == pytest.approx(88.0 / 100.0 - 1.0)
    assert result.boundary_anchored_max_drawdown == pytest.approx(80.0 / 100.0 - 1.0)
    assert result.volatility_ann == pytest.approx(
        pd.Series([-0.20, 0.10]).std(ddof=1) * 252**0.5
    )
    assert result.n_returns == 2
    assert result.periods_per_year == 252
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.n_returns = 3  # type: ignore[misc]


def test_crisis_anchor_is_the_last_value_before_the_start() -> None:
    # two pre-crisis observations: the anchor must be the LAST one
    value = pd.Series(
        [90.0, 100.0, 80.0, 88.0],
        index=pd.to_datetime(["2021-12-29", "2021-12-31", "2022-01-03", "2022-01-04"]),
    )
    result = crisis_metrics(value, "2022-01-01", "2022-01-04")
    assert result.anchor == pd.Timestamp("2021-12-31")
    assert result.episode_return == pytest.approx(88.0 / 100.0 - 1.0)
    assert result.boundary_anchored_max_drawdown == pytest.approx(80.0 / 100.0 - 1.0)


def test_crisis_anchor_is_strictly_before_the_requested_start() -> None:
    # an observation exactly ON the requested start belongs to the window, not the anchor
    value = pd.Series(
        [100.0, 90.0, 81.0],
        index=pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"]),
    )
    result = crisis_metrics(value, "2022-01-04", "2022-01-05")
    assert result.anchor == pd.Timestamp("2022-01-03")
    assert result.first_return_date == pd.Timestamp("2022-01-04")
    assert result.episode_return == pytest.approx(81.0 / 100.0 - 1.0)
    assert result.n_returns == 2


def test_crisis_metrics_single_return_episode_has_nan_volatility() -> None:
    # ddof=1 volatility is undefined on one return; NaN is the documented outcome
    import numpy as np

    value = pd.Series(
        [100.0, 95.0], index=pd.to_datetime(["2021-12-31", "2022-01-03"])
    )
    result = crisis_metrics(value, "2022-01-01", "2022-01-05")
    assert result.n_returns == 1
    assert np.isnan(result.volatility_ann)


def test_crisis_metrics_include_entry_return() -> None:  # defect 8 shared boundary
    test_crisis_metrics_include_the_return_entering_the_first_session()




def test_crisis_metrics_reject_zero_level_producing_nonfinite_returns() -> None:
    # a finite-but-zero VALUE yields an infinite constructed return
    value = pd.Series(
        [100.0, 0.0, 88.0],
        index=pd.to_datetime(["2021-12-31", "2022-01-03", "2022-01-04"]),
    )
    with pytest.raises(ValueError, match="episode returns"):
        crisis_metrics(value, "2022-01-01", "2022-01-04")


def test_crisis_metrics_require_an_anchor_and_an_in_window_return() -> None:
    value = pd.Series(
        [100.0, 101.0], index=pd.to_datetime(["2022-01-03", "2022-01-04"])
    )
    assert crisis_metrics(value, "2022-01-01", "2022-01-04") is None
    assert crisis_metrics(value, "2022-02-01", "2022-02-28") is None


def test_crisis_metrics_reject_unsorted_values_instead_of_misdating_the_end() -> None:
    value = pd.Series(
        [100.0, 88.0, 80.0],
        index=pd.to_datetime(["2021-12-31", "2022-01-04", "2022-01-03"]),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        crisis_metrics(value, "2022-01-01", "2022-01-04")


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (pd.Series(index=pd.DatetimeIndex([]), dtype=float), "must not be empty"),
        (
            pd.Series([100.0, 101.0], index=["2021-12-31", "2022-01-03"]),
            "DatetimeIndex",
        ),
        (
            pd.Series(
                [100.0, 101.0],
                index=pd.to_datetime(["2021-12-31", "2021-12-31"]),
            ),
            "unique labels",
        ),
        (
            pd.Series(
                [100.0, float("nan")],
                index=pd.to_datetime(["2021-12-31", "2022-01-03"]),
            ),
            "non-finite value",
        ),
        (
            pd.Series(
                pd.to_datetime(["2021-12-31", "2022-01-03"]),
                index=pd.to_datetime(["2021-12-30", "2021-12-31"]),
            ),
            "real numeric values",
        ),
        (
            pd.Series(
                ["100", "101"],
                index=pd.to_datetime(["2021-12-31", "2022-01-03"]),
            ),
            "real numeric values",
        ),
        (
            pd.Series(
                [True, False],
                index=pd.to_datetime(["2021-12-31", "2022-01-03"]),
            ),
            "real numeric values",
        ),
        (
            pd.Series(
                [100 + 1j, 101 + 2j],
                index=pd.to_datetime(["2021-12-31", "2022-01-03"]),
            ),
            "real numeric values",
        ),
        (
            pd.Series(
                [100.0, 101.0],
                index=pd.to_datetime(["2021-12-31", "2022-01-03"], utc=True),
            ),
            "timezone-naive",
        ),
        (
            pd.Series(
                [100.0, 101.0],
                index=pd.DatetimeIndex([pd.NaT, pd.Timestamp("2022-01-03")]),
            ),
            "NaT",
        ),
    ],
)
def test_crisis_metrics_reject_malformed_values(value: pd.Series, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        crisis_metrics(value, "2022-01-01", "2022-01-04")


@pytest.mark.parametrize(
    ("start", "end", "periods_per_year", "match"),
    [
        ("2022-01-05", "2022-01-04", 252, "on or before"),
        (pd.NaT, "2022-01-04", 252, "must not be NaT"),
        ("not-a-date", "2022-01-04", 252, "valid timestamps"),
        ("2022-01-01T00:00:00Z", "2022-01-04T00:00:00Z", 252, "timezone-naive"),
        ("2022-01-01", "2022-01-04", 0, "positive integer"),
        ("2022-01-01", "2022-01-04", True, "positive integer"),
    ],
)
def test_crisis_metrics_reject_invalid_bounds_or_annualization(
    start: object, end: object, periods_per_year: object, match: str
) -> None:
    value = pd.Series(
        [100.0, 101.0], index=pd.to_datetime(["2021-12-31", "2022-01-03"])
    )
    with pytest.raises(ValueError, match=match):
        crisis_metrics(  # type: ignore[arg-type]
            value, start, end, periods_per_year=periods_per_year
        )


def test_crisis_analytics_is_a_legacy_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    import macro_framework.evaluation as evaluation

    expected = CrisisMetrics(
        requested_start=pd.Timestamp("2022-01-01"),
        requested_end=pd.Timestamp("2022-12-31"),
        anchor=pd.Timestamp("2021-12-31"),
        first_return_date=pd.Timestamp("2022-01-03"),
        actual_end=pd.Timestamp("2022-12-30"),
        episode_return=-0.12,
        boundary_anchored_max_drawdown=-0.20,
        volatility_ann=0.34,
        n_returns=252,
        periods_per_year=252,
    )
    calls: list[tuple[pd.Series, str, str]] = []

    def fake_crisis_metrics(value: pd.Series, start: str, end: str) -> CrisisMetrics:
        calls.append((value, start, end))
        return expected

    class FakePortfolio:
        def __init__(self, value: pd.Series) -> None:
            self._value = value

        def value(self) -> pd.Series:
            return self._value

    value = pd.Series([100.0], index=pd.to_datetime(["2021-12-31"]))
    monkeypatch.setattr(evaluation, "crisis_metrics", fake_crisis_metrics)

    frame = crisis_analytics({"line": FakePortfolio(value)})  # type: ignore[dict-item]

    assert calls == [(value, "2022-01-01", "2022-12-31")]
    assert frame.loc["line"].to_dict() == {
        "crisis_return": -0.12,
        "crisis_max_drawdown": -0.20,
        "crisis_vol_ann": 0.34,
    }


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
