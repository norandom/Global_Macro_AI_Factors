"""Offline tests for scripts/extend_stream_2026.py (task 8.1 pure helpers).

No network, no NIM key, no price fetch: only the pure helpers the extension
script composes around the live pipeline (replay-reply synthesis, the
completed-months panel guard, and the in-training vs post-cutoff split table).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

# NOTE: extend_stream_2026 (and macro_framework.factor_scoring, which it pulls
# in) is imported INSIDE each test, and this file is named to sort AFTER
# tests/test_factor_scoring.py: that suite asserts factor_scoring is NOT yet in
# sys.modules when its foundation test runs (module-level imports here would
# break it at collection time).


def _mod():
    import extend_stream_2026 as ext

    return ext


def test_synth_loadings_reply_roundtrips_through_parse_loadings() -> None:
    from macro_framework import factor_scoring as fs

    ext = _mod()
    loadings = {
        "inflation": 0.4,
        "growth": -0.2,
        "credit_stress": 0.6,
        "policy": -0.1,
        "risk_appetite": -0.3,
    }
    reply = ext.synth_loadings_reply(loadings)
    rl = fs.parse_loadings(reply, pd.Timestamp("2020-03-02"))
    assert rl is not None and rl.parse_ok
    assert rl.loadings == loadings


def test_synth_loadings_reply_none_stays_unparsed() -> None:
    from macro_framework import factor_scoring as fs

    ext = _mod()
    reply = ext.synth_loadings_reply(None)
    assert reply == ""
    assert fs.parse_loadings(reply, pd.Timestamp("2020-03-02")) is None


def test_completed_months_only_drops_current_month_row() -> None:
    ext = _mod()
    idx = pd.DatetimeIndex(["2026-05-31", "2026-06-30", "2026-07-31"])
    panel = pd.DataFrame({"cpi_yoy": [1.0, 2.0, 3.0]}, index=idx)
    out = ext.completed_months_only(panel, today=pd.Timestamp("2026-07-17"))
    assert list(out.index) == [pd.Timestamp("2026-05-31"), pd.Timestamp("2026-06-30")]


def test_split_contrast_table_segments_and_stats() -> None:
    ext = _mod()
    idx = pd.DatetimeIndex(["2024-01-02", "2024-05-01", "2024-08-01", "2025-01-02", "2025-02-03"])
    df = pd.DataFrame(
        {
            "pit_p": [0.1, 0.2, 0.3, 0.4, float("nan")],
            "nonpit_p": [0.7, 0.9, 0.35, 0.45, 0.5],
        },
        index=idx,
    )
    table = ext.split_contrast_table(df, cutoff=pd.Timestamp("2024-06-01"))

    it = table["in_training"]
    assert it["n_pairs"] == 2
    assert abs(it["mean_delta"] - 0.65) < 1e-12
    assert abs(it["median_delta"] - 0.65) < 1e-12
    assert set(it) == {"n_pairs", "mean_delta", "median_delta", "paired_d"}

    pc = table["post_cutoff"]
    assert pc["n_pairs"] == 2  # the NaN pit_p pair is dropped
    assert abs(pc["mean_delta"] - 0.05) < 1e-12

    assert isinstance(table["prediction_outcome"], str)
    assert "collapsed" in table["prediction_outcome"]


def test_classify_premium_outcome_either_way() -> None:
    ext = _mod()
    assert "collapsed" in ext.classify_premium_outcome(0.53, 0.02)
    persisted = ext.classify_premium_outcome(0.53, 0.48)
    assert "did NOT collapse" in persisted
