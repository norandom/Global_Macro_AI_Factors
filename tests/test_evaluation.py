from __future__ import annotations

import pandas as pd

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
