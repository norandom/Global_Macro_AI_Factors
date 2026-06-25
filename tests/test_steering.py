"""Unit tests for macro_framework/steering.py (track-a-macro-steering).

Task 2.1 — directional point-in-time prompt rendering. The renderer turns the
same anonymized, z-scored macro content the agent saw into a directional-forecast
prompt that recall_guard's parsers can read (Direction / Confidence lines).

Requirements: 1.2 (same anonymized z-scored content), 1.4 (only as-of info,
inherited from the caller — here exercised as "no date / no real ticker").
"""

from __future__ import annotations

import re

import pytest

from macro_framework.steering import (
    DIRECTIONAL_PROMPT_TEMPLATE,
    render_directional,
)

# recall_guard's own strict parsers — the rendered prompt must request precisely
# the answer format these accept, so we assert against the real expressions.
from recall_guard.harness.evaluator import _parse_confidence, _parse_direction


# A representative PIT macro state + anonymized asset snapshot. Numbers carry
# more precision than the agent rounds to, so we can prove the renderer rounds.
MACRO_STATE = {
    "cpi_yoy_z": 1.23456,
    "t10y2y_z": -0.98765,
    "hy_oas_z": 0.50049,
}
ASSET_SNAPSHOT = [
    {"id": "Asset_A", "category": "world_equity",
     "trailing_12m_return": 0.1234567, "trailing_vol_ann": 0.1875432},
    {"id": "Asset_B", "category": "tech_sector",
     "trailing_12m_return": -0.0501234, "trailing_vol_ann": 0.2499876},
    {"id": "Asset_C", "category": "gold_commodity",
     "trailing_12m_return": 0.0789012, "trailing_vol_ann": 0.1500987},
    {"id": "Asset_D", "category": "short_treasury_cash",
     "trailing_12m_return": 0.0204999, "trailing_vol_ann": 0.0050001},
]

REAL_TICKERS = ("SWDA", "SWDA.L", "XLK", "IAU", "BIL")


def test_render_directional_is_deterministic() -> None:
    """Equal inputs ⇒ byte-identical prompt string (design 1.2/1.4 invariant)."""
    a = render_directional(MACRO_STATE, ASSET_SNAPSHOT)
    b = render_directional(dict(MACRO_STATE), [dict(x) for x in ASSET_SNAPSHOT])
    assert a == b
    assert isinstance(a, str) and a


def test_render_directional_embeds_rounded_macro_z_scores() -> None:
    """Macro z-scores appear rounded to 2dp, mirroring llm_agent rounding (1.2)."""
    out = render_directional(MACRO_STATE, ASSET_SNAPSHOT)
    assert "1.23" in out          # cpi_yoy_z 1.23456 -> 1.23
    assert "-0.99" in out         # t10y2y_z -0.98765 -> -0.99
    assert "0.5" in out           # hy_oas_z 0.50049 -> 0.5
    # The raw over-precise values must NOT leak.
    assert "1.23456" not in out
    assert "0.98765" not in out


def test_render_directional_embeds_anonymized_assets_rounded() -> None:
    """Asset ids + categories present; numeric fields rounded to 3dp (1.2)."""
    out = render_directional(MACRO_STATE, ASSET_SNAPSHOT)
    for asset in ASSET_SNAPSHOT:
        assert asset["id"] in out
        assert asset["category"] in out
    assert "0.123" in out         # trailing_12m_return 0.1234567 -> 0.123
    assert "0.188" in out         # trailing_vol_ann 0.1875432 -> 0.188
    assert "0.1234567" not in out  # raw precision must not leak


def test_render_directional_has_no_date_or_year() -> None:
    """No 4-digit year and no ISO date anywhere in the prompt (1.4)."""
    out = render_directional(MACRO_STATE, ASSET_SNAPSHOT)
    assert not re.search(r"\b(19|20)\d{2}\b", out), "leaked a 4-digit year"
    assert not re.search(r"\b\d{4}-\d{2}-\d{2}\b", out), "leaked an ISO date"


def test_render_directional_has_no_real_ticker() -> None:
    """Only Asset_A..D pseudo ids — no real ETF ticker leaks (1.4)."""
    out = render_directional(MACRO_STATE, ASSET_SNAPSHOT)
    for ticker in REAL_TICKERS:
        assert not re.search(rf"\b{re.escape(ticker)}\b", out), f"leaked {ticker}"


def test_render_directional_requests_parseable_direction_and_confidence() -> None:
    """Prompt uses the literal Direction/Confidence tokens recall_guard accepts."""
    out = render_directional(MACRO_STATE, ASSET_SNAPSHOT)
    assert "Direction:" in out
    assert "Confidence:" in out
    assert "-1" in out and "0" in out and "1" in out   # the {-1, 0, 1} set
    # A model echoing the requested format must parse cleanly through the real
    # recall_guard parsers (this is the contract the template exists to satisfy).
    sample_answer = "Direction: 1\nConfidence: 0.5"
    assert _parse_direction(sample_answer) == 1
    assert _parse_confidence(sample_answer) == 0.5
    # And the template's own example lines must themselves be parser-valid.
    assert _parse_direction(DIRECTIONAL_PROMPT_TEMPLATE) in {-1, 0, 1}
    assert _parse_confidence(DIRECTIONAL_PROMPT_TEMPLATE) is not None


def test_template_is_shared_source_of_truth() -> None:
    """The renderer is built from the module-level template constant (1.2)."""
    out = render_directional(MACRO_STATE, ASSET_SNAPSHOT)
    # Template's fixed instructional spine must survive into every rendering.
    spine = DIRECTIONAL_PROMPT_TEMPLATE.split("{")[0].strip()
    assert spine
    assert spine in out


@pytest.mark.parametrize(
    "state",
    [
        {"cpi_yoy_z": 0.0, "t10y2y_z": 0.0, "hy_oas_z": 0.0},
        {"cpi_yoy_z": -2.5, "t10y2y_z": 3.14159, "hy_oas_z": -1.4949},
    ],
)
def test_render_directional_always_parseable_format(state: dict[str, float]) -> None:
    """Across macro states the requested answer format stays parser-ready (1.2)."""
    out = render_directional(state, ASSET_SNAPSHOT)
    assert "Direction:" in out and "Confidence:" in out


# ---------------------------------------------------------------------------
# Task 2.2 — Calibration adapter over the released scorer
#
# The ScoringAdapter owns the CALIBRATION half: it builds a dated FMP corpus
# (writing two JSONL files), reads them back via _read_corpus_jsonl into
# in-memory prompt lists, and calibrates the chosen NIM model from them. It
# surfaces calibrator quality (holdout_auc / is_weak) and lets recall_guard's
# ConfigurationError propagate when the NIM credential is empty/rejected.
#
# Both external calls are MOCKED here (no FMP, no NIM):
#   * build_calibration  — patched where the steering module binds it, so it
#     writes two tiny temp JSONL files into tmp_path and returns their paths.
#   * MemoryGuardedScorer.calibrate — patched to return a fake scorer exposing
#     holdout_auc / is_weak (or, for the error path, to raise the real
#     ConfigurationError on an empty key).
#
# Requirements: 1.2 (same-content prompts read back & fed to calibrate),
# 1.5 (clear ConfigurationError on bad credential), 1.7 (surface weak/AUC).
# ---------------------------------------------------------------------------

import json
from dataclasses import dataclass
from pathlib import Path

from recall_guard import ConfigurationError

from macro_framework.steering import (
    CalibrationResult,
    ScoringAdapter,
    _read_corpus_jsonl,
)


@dataclass
class _FakeScorer:
    """Stand-in for recall_guard.MemoryGuardedScorer exposing the quality props."""

    holdout_auc: float = 0.83
    is_weak: bool = False


def _write_corpus_jsonl(path: Path, prompts: list[str], label: int) -> None:
    """Write a build_calibration-shaped JSONL file (prompt/label/metadata)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for i, prompt in enumerate(prompts):
            rec = {
                "prompt": prompt,
                "label": label,
                "metadata": {
                    "published_at": "2020-01-01",
                    "source": "test",
                    "url": f"https://example.test/{label}/{i}",
                },
            }
            fh.write(json.dumps(rec) + "\n")


def _patch_build_calibration(mocker, tmp_path, is_prompts, oos_prompts):
    """Patch build_calibration as bound in steering: writes 2 JSONL, returns paths."""
    is_path = tmp_path / "is_memorized.jsonl"
    oos_path = tmp_path / "oos_control.jsonl"

    def _fake_build(out_dir, cutoffs, target_per_corpus=100, api_key=None, **kwargs):
        _write_corpus_jsonl(is_path, is_prompts, label=1)
        _write_corpus_jsonl(oos_path, oos_prompts, label=0)
        return is_path, oos_path

    return mocker.patch(
        "macro_framework.steering.build_calibration",
        side_effect=_fake_build,
    )


def test_read_corpus_jsonl_returns_prompts_in_file_order(tmp_path: Path) -> None:
    """_read_corpus_jsonl projects each record's prompt field, order preserved (1.2)."""
    prompts = ["first prompt", "second prompt", "third prompt"]
    path = tmp_path / "corpus.jsonl"
    _write_corpus_jsonl(path, prompts, label=1)
    out = _read_corpus_jsonl(path)
    assert out == prompts
    assert all(isinstance(p, str) for p in out)


def test_calibrate_from_fmp_reads_back_and_passes_correct_lists(
    mocker, tmp_path: Path
) -> None:
    """Adapter reads both JSONL files back and feeds the right IS/OOS lists (1.2)."""
    is_prompts = ["is-A", "is-B", "is-C"]
    oos_prompts = ["oos-1", "oos-2"]
    build_mock = _patch_build_calibration(mocker, tmp_path, is_prompts, oos_prompts)
    cal_mock = mocker.patch(
        "macro_framework.steering.MemoryGuardedScorer.calibrate",
        return_value=_FakeScorer(holdout_auc=0.77, is_weak=False),
    )

    import datetime as _dt

    adapter = ScoringAdapter.calibrate_from_fmp(
        nim_model="meta/llama-3.1-8b-instruct",
        cutoff_date=_dt.date(2023, 1, 1),
        out_dir=tmp_path,
        api_key="nim-key",
        fmp_api_key="fmp-key",
        reference_model=None,
        min_auc=0.6,
        target_per_corpus=50,
    )

    # build_calibration called with the cutoff dict keyed by the NIM model.
    assert build_mock.call_count == 1
    _, bkwargs = build_mock.call_args
    assert bkwargs["cutoffs"] == {"meta/llama-3.1-8b-instruct": _dt.date(2023, 1, 1)}
    assert bkwargs["target_per_corpus"] == 50
    assert bkwargs["api_key"] == "fmp-key"

    # calibrate received the prompt lists read back off disk, in file order.
    assert cal_mock.call_count == 1
    _, ckwargs = cal_mock.call_args
    assert ckwargs["model"] == "meta/llama-3.1-8b-instruct"
    assert ckwargs["api_key"] == "nim-key"
    assert list(ckwargs["is_memorized"]) == is_prompts
    assert list(ckwargs["oos_control"]) == oos_prompts
    assert ckwargs["min_auc"] == 0.6

    assert isinstance(adapter, ScoringAdapter)


def test_calibrate_from_fmp_surfaces_holdout_auc_and_is_weak(
    mocker, tmp_path: Path
) -> None:
    """is_weak / holdout_auc reflect the calibrated scorer (1.7)."""
    _patch_build_calibration(mocker, tmp_path, ["is-A", "is-B"], ["oos-1", "oos-2"])
    mocker.patch(
        "macro_framework.steering.MemoryGuardedScorer.calibrate",
        return_value=_FakeScorer(holdout_auc=0.55, is_weak=True),
    )

    import datetime as _dt

    adapter = ScoringAdapter.calibrate_from_fmp(
        nim_model="meta/llama-3.1-8b-instruct",
        cutoff_date=_dt.date(2023, 1, 1),
        out_dir=tmp_path,
        api_key="nim-key",
    )
    assert adapter.is_weak is True
    assert adapter.holdout_auc == 0.55


def test_calibration_result_is_returned_with_quality(mocker, tmp_path: Path) -> None:
    """The adapter exposes a CalibrationResult carrying scorer + quality (1.7)."""
    fake = _FakeScorer(holdout_auc=0.91, is_weak=False)
    _patch_build_calibration(mocker, tmp_path, ["is-A", "is-B"], ["oos-1", "oos-2"])
    mocker.patch(
        "macro_framework.steering.MemoryGuardedScorer.calibrate",
        return_value=fake,
    )

    import datetime as _dt

    adapter = ScoringAdapter.calibrate_from_fmp(
        nim_model="meta/llama-3.1-8b-instruct",
        cutoff_date=_dt.date(2023, 1, 1),
        out_dir=tmp_path,
        api_key="nim-key",
    )
    assert isinstance(adapter.calibration, CalibrationResult)
    assert adapter.calibration.scorer is fake
    assert adapter.calibration.holdout_auc == 0.91
    assert adapter.calibration.is_weak is False


def test_calibrate_from_fmp_propagates_configuration_error_on_empty_key(
    mocker, tmp_path: Path
) -> None:
    """An empty NIM api_key surfaces ConfigurationError (let recall_guard raise) (1.5)."""
    _patch_build_calibration(mocker, tmp_path, ["is-A", "is-B"], ["oos-1", "oos-2"])
    # Do NOT mock calibrate — exercise the real empty-key guard in recall_guard,
    # which raises ConfigurationError. The adapter must not catch/swallow it.

    import datetime as _dt

    with pytest.raises(ConfigurationError):
        ScoringAdapter.calibrate_from_fmp(
            nim_model="meta/llama-3.1-8b-instruct",
            cutoff_date=_dt.date(2023, 1, 1),
            out_dir=tmp_path,
            api_key="",
        )


# ---------------------------------------------------------------------------
# Task 2.3 — Scoring path on the separate inference path
#
# score_rebalances is a thin, order-preserving wrapper over the calibrated
# scorer's score_many (one NIM call per prompt — the separate, logprob-bearing
# inference path, NOT the agent's DSPy path). It returns the full list of
# GuardedScore (task 2.7's reporter needs p_memorized / parse_ok / fail_reason /
# memguard_confidence); the "expose only p_memorized + failure reason" rule is a
# downstream consumption contract (the 2.4 ViewSteerer reads only p_memorized),
# not a narrowing of this return type.
#
# The scorer is a FAKE here — no NIM, no FMP. We mirror the real
# recall_guard.GuardedScore field shape and the real score_many contract
# (Sequence[str] -> list[GuardedScore], order preserved).
#
# Requirements: 1.2 (same-content rebalance prompts scored), 1.3 (separate
# logprob-bearing inference path via the calibrated scorer), 1.5 (a rejected
# credential mid-scoring surfaces ConfigurationError, not swallowed).
# ---------------------------------------------------------------------------


def _fake_guarded_score(
    *,
    prompt: str,
    p_memorized: float,
    parse_ok: bool = True,
    fail_reason: str | None = None,
):
    """Build a recall_guard-faithful GuardedScore for the fake scorer.

    Uses the REAL GuardedScore so the test stays honest about the dataclass the
    wrapper returns (prompt_hash / parse_ok / signal / raw_confidence /
    p_memorized / memguard_confidence / features / fail_reason).
    """
    from hashlib import sha256

    from recall_guard import GuardedScore

    return GuardedScore(
        prompt_hash=sha256(prompt.encode("utf-8")).hexdigest()[:16],
        parse_ok=parse_ok,
        signal=1 if parse_ok else None,
        raw_confidence=0.5 if parse_ok else None,
        p_memorized=p_memorized if parse_ok else None,
        memguard_confidence=(0.5 * (1.0 - p_memorized)) if parse_ok else None,
        features=None,
        fail_reason=fail_reason,
    )


class _FakeCalibratedScorer:
    """Stand-in for a calibrated MemoryGuardedScorer: only score_many is used.

    Mirrors the real signature ``score_many(prompts, *, max_workers=8) ->
    list[GuardedScore]`` and preserves input order. Records the prompts it was
    called with so the test can assert delegation.
    """

    holdout_auc: float = 0.83
    is_weak: bool = False

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def score_many(self, prompts, *, max_workers: int = 8):
        self.calls.append(list(prompts))
        return [
            _fake_guarded_score(prompt=p, p_memorized=round(0.1 * i, 3))
            for i, p in enumerate(prompts)
        ]


class _RaisingCalibratedScorer:
    """Calibrated scorer whose score_many rejects the credential mid-scoring."""

    holdout_auc: float = 0.83
    is_weak: bool = False

    def score_many(self, prompts, *, max_workers: int = 8):
        raise ConfigurationError("NIM credential rejected")


def _adapter_around(scorer) -> ScoringAdapter:
    """Wrap a fake calibrated scorer in a ScoringAdapter (no calibration call)."""
    return ScoringAdapter(
        CalibrationResult(scorer=scorer, holdout_auc=scorer.holdout_auc, is_weak=scorer.is_weak)
    )


def test_score_rebalances_returns_one_score_per_prompt_in_order() -> None:
    """Results align 1:1 with input prompts, in input order (1.2, 1.3)."""
    scorer = _FakeCalibratedScorer()
    adapter = _adapter_around(scorer)
    prompts = ["prompt-A", "prompt-B", "prompt-C", "prompt-D"]

    out = adapter.score_rebalances(prompts)

    assert len(out) == len(prompts)
    # Order preserved: prompt_hash of each result matches the prompt at that index.
    from hashlib import sha256

    for prompt, score in zip(prompts, out, strict=True):
        assert score.prompt_hash == sha256(prompt.encode("utf-8")).hexdigest()[:16]


def test_score_rebalances_delegates_to_scorer_score_many() -> None:
    """score_rebalances delegates to the calibrated scorer's score_many (1.3)."""
    scorer = _FakeCalibratedScorer()
    adapter = _adapter_around(scorer)
    prompts = ["p1", "p2", "p3"]

    adapter.score_rebalances(prompts)

    assert scorer.calls == [prompts]


def test_score_rebalances_returns_full_guardedscore_for_reporter() -> None:
    """Full GuardedScore is returned (2.7 reporter reads p_memorized/parse_ok/...)."""
    scorer = _FakeCalibratedScorer()
    adapter = _adapter_around(scorer)

    out = adapter.score_rebalances(["only-prompt"])

    score = out[0]
    # The fields task 2.7 consumes must be present (not narrowed away).
    assert hasattr(score, "p_memorized")
    assert hasattr(score, "parse_ok")
    assert hasattr(score, "fail_reason")
    assert hasattr(score, "memguard_confidence")
    assert score.parse_ok is True
    assert score.p_memorized == 0.0


def test_score_rebalances_preserves_failure_records() -> None:
    """A failed (parse) score is returned verbatim with its fail_reason (1.2)."""

    class _MixedScorer(_FakeCalibratedScorer):
        def score_many(self, prompts, *, max_workers: int = 8):
            self.calls.append(list(prompts))
            return [
                _fake_guarded_score(prompt="ok", p_memorized=0.2, parse_ok=True),
                _fake_guarded_score(
                    prompt="bad", p_memorized=0.0, parse_ok=False, fail_reason="parse_failure"
                ),
            ]

    adapter = _adapter_around(_MixedScorer())
    out = adapter.score_rebalances(["ok", "bad"])

    assert out[0].parse_ok is True
    assert out[1].parse_ok is False
    assert out[1].fail_reason == "parse_failure"
    assert out[1].p_memorized is None


def test_score_rebalances_propagates_configuration_error() -> None:
    """A ConfigurationError raised mid-scoring propagates out, not swallowed (1.5)."""
    adapter = _adapter_around(_RaisingCalibratedScorer())

    with pytest.raises(ConfigurationError):
        adapter.score_rebalances(["p1", "p2"])


def test_score_rebalances_handles_empty_prompts() -> None:
    """No prompts ⇒ empty result, still a single delegated call (order-preserving)."""
    scorer = _FakeCalibratedScorer()
    adapter = _adapter_around(scorer)

    out = adapter.score_rebalances([])

    assert out == []
    assert scorer.calls == [[]]


# ---------------------------------------------------------------------------
# Task 2.4 — Point-in-time macro characterization and steering signal
#
# characterize(macro_hist, rebalance_date) reads the EXISTING macro panel
# strictly < rebalance_date (R2.2, R2.3), takes the latest as-of z-scores per
# series (R2.1), and assigns a DETERMINISTIC, non-predictive z-bin regime label
# (R2.5). The returned frozen SteeringSignal exposes a pure consistency(view)
# heuristic in [consistency_floor, 1.0] (R2.4 — used downstream by ViewSteerer).
# write_steering_signals persists the per-rebalance signals to a parquet
# artifact under a NEW filename (R2.4 / R6.4 — caller chooses the name).
#
# Tests use synthetic pandas panels and the REAL MacroView; no I/O beyond
# tmp_path, no NIM/FMP, no panel regeneration.
# ---------------------------------------------------------------------------

import pandas as pd

from macro_framework.llm_agent import MacroView
from macro_framework.steering import (
    SteeringSignal,
    characterize,
    write_steering_signals,
)


def _macro_panel(rows: dict[str, list[float]], dates: list[str]) -> pd.DataFrame:
    """Build a date-indexed macro panel carrying the *_z columns characterize reads."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="date")
    return pd.DataFrame(rows, index=idx)


def _view(asset_long: str, *, confidence: float = 0.5) -> MacroView:
    """A real MacroView with the given long-leg pseudo asset id."""
    return MacroView(
        asset_long=asset_long,
        asset_short=None,
        expected_excess_annualized=0.05,
        confidence=confidence,
        rationale="test",
    )


# Crafted z-scores per regime. Thresholds documented in steering.characterize.
_GOLDILOCKS = {"cpi_yoy_z": [-0.5], "t10y2y_z": [0.5], "hy_oas_z": [-1.2]}
_CREDIT_STRESS = {"cpi_yoy_z": [0.1], "t10y2y_z": [0.2], "hy_oas_z": [1.6]}
_STAGFLATION = {"cpi_yoy_z": [1.4], "t10y2y_z": [-1.1], "hy_oas_z": [0.3]}
_NEUTRAL = {"cpi_yoy_z": [0.1], "t10y2y_z": [0.1], "hy_oas_z": [0.1]}


def test_characterize_latest_as_of_zscore_summary() -> None:
    """zscore_summary is the last row strictly before rebalance_date (R2.1)."""
    panel = _macro_panel(
        {
            "cpi_yoy_z": [0.10, 0.20, 1.40],
            "t10y2y_z": [0.10, 0.20, -1.10],
            "hy_oas_z": [0.10, 0.20, 0.30],
        },
        ["2020-01-31", "2020-02-29", "2020-03-31"],
    )
    sig = characterize(panel, pd.Timestamp("2020-03-31"))
    # The 2020-03-31 row is NOT before 2020-03-31; latest as-of is 2020-02-29.
    assert sig.zscore_summary == {
        "cpi_yoy_z": 0.20,
        "t10y2y_z": 0.20,
        "hy_oas_z": 0.20,
    }


def test_characterize_is_point_in_time(monkeypatch=None) -> None:
    """Appending a row dated >= rebalance_date does NOT change the output (R2.2)."""
    base = _macro_panel(
        _STAGFLATION,
        ["2020-02-29"],
    )
    rb = pd.Timestamp("2020-03-31")
    sig_before = characterize(base, rb)

    # Append a future row (on or after rb) carrying a totally different regime.
    future = _macro_panel(
        {"cpi_yoy_z": [-2.0], "t10y2y_z": [2.0], "hy_oas_z": [-2.0]},
        ["2020-03-31"],  # == rb, must be excluded
    )
    later = _macro_panel(
        {"cpi_yoy_z": [3.0], "t10y2y_z": [-3.0], "hy_oas_z": [3.0]},
        ["2020-04-30"],  # > rb, must be excluded
    )
    contaminated = pd.concat([base, future, later])

    sig_after = characterize(contaminated, rb)
    assert sig_after.regime_label == sig_before.regime_label
    assert sig_after.zscore_summary == sig_before.zscore_summary


def test_characterize_deterministic_equal_inputs_equal_signal() -> None:
    """Equal inputs ⇒ equal SteeringSignal (determinism)."""
    panel = _macro_panel(_GOLDILOCKS, ["2020-02-29"])
    rb = pd.Timestamp("2020-03-31")
    a = characterize(panel, rb)
    b = characterize(panel.copy(), rb)
    assert a == b


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (_GOLDILOCKS, "goldilocks"),
        (_CREDIT_STRESS, "credit_stress"),
        (_STAGFLATION, "stagflation_risk"),
        (_NEUTRAL, "neutral"),
    ],
)
def test_characterize_regime_labels(rows: dict, expected: str) -> None:
    """Crafted z-scores map to specific deterministic labels (R2.1, ≥3 regimes)."""
    panel = _macro_panel(rows, ["2020-02-29"])
    sig = characterize(panel, pd.Timestamp("2020-03-31"))
    assert sig.regime_label == expected


def test_consistency_aligned_view_returns_one() -> None:
    """A view whose long leg is in the regime's preferred set ⇒ 1.0 (R2.4)."""
    # credit_stress prefers defensive categories (gold + cash); Asset_C = gold.
    panel = _macro_panel(_CREDIT_STRESS, ["2020-02-29"])
    sig = characterize(panel, pd.Timestamp("2020-03-31"))
    assert sig.regime_label == "credit_stress"
    assert sig.consistency(_view("Asset_C")) == 1.0


def test_consistency_contradicting_view_returns_floor() -> None:
    """A view contradicting the regime ⇒ consistency_floor (R2.4)."""
    # credit_stress does NOT prefer risk assets; Asset_B = tech_sector.
    panel = _macro_panel(_CREDIT_STRESS, ["2020-02-29"])
    sig = characterize(panel, pd.Timestamp("2020-03-31"), consistency_floor=0.5)
    assert sig.regime_label == "credit_stress"
    assert sig.consistency(_view("Asset_B")) == 0.5


def test_consistency_always_within_floor_and_one() -> None:
    """consistency is always in [consistency_floor, 1.0] across regimes/views (R2.4)."""
    floor = 0.3
    for rows in (_GOLDILOCKS, _CREDIT_STRESS, _STAGFLATION, _NEUTRAL):
        panel = _macro_panel(rows, ["2020-02-29"])
        sig = characterize(panel, pd.Timestamp("2020-03-31"), consistency_floor=floor)
        for asset in ("Asset_A", "Asset_B", "Asset_C", "Asset_D"):
            c = sig.consistency(_view(asset))
            assert floor <= c <= 1.0


def test_consistency_goldilocks_prefers_risk_assets() -> None:
    """goldilocks prefers equities/tech; cash is not preferred (documented map)."""
    panel = _macro_panel(_GOLDILOCKS, ["2020-02-29"])
    sig = characterize(panel, pd.Timestamp("2020-03-31"))
    assert sig.regime_label == "goldilocks"
    assert sig.consistency(_view("Asset_A")) == 1.0  # world_equity preferred
    assert sig.consistency(_view("Asset_D")) == sig._consistency_floor  # cash not


def test_consistency_neutral_prefers_all() -> None:
    """neutral regime is non-committal: every category is preferred ⇒ 1.0 (R2.4)."""
    panel = _macro_panel(_NEUTRAL, ["2020-02-29"])
    sig = characterize(panel, pd.Timestamp("2020-03-31"))
    assert sig.regime_label == "neutral"
    for asset in ("Asset_A", "Asset_B", "Asset_C", "Asset_D"):
        assert sig.consistency(_view(asset)) == 1.0


def test_write_steering_signals_round_trip(tmp_path: Path) -> None:
    """Writer emits a parquet with regime_label + z-summary; round-trips back (R2.4)."""
    signals = [
        characterize(_macro_panel(_GOLDILOCKS, ["2020-01-31"]), pd.Timestamp("2020-02-29")),
        characterize(_macro_panel(_CREDIT_STRESS, ["2020-02-29"]), pd.Timestamp("2020-03-31")),
        characterize(_macro_panel(_STAGFLATION, ["2020-03-31"]), pd.Timestamp("2020-04-30")),
    ]
    out = tmp_path / "macro_steering_signals_test.parquet"
    write_steering_signals(signals, out)
    assert out.exists()

    df = pd.read_parquet(out)
    # One row per rebalance, regime + the three z columns present.
    assert len(df) == 3
    assert "regime_label" in df.columns
    for col in ("cpi_yoy_z", "t10y2y_z", "hy_oas_z"):
        assert col in df.columns
    labels = list(df["regime_label"])
    assert labels == ["goldilocks", "credit_stress", "stagflation_risk"]
    # rebalance_date is recoverable (index or column).
    dates = df.index if df.index.name == "rebalance_date" else df["rebalance_date"]
    assert pd.Timestamp("2020-03-31") in set(pd.to_datetime(dates))


def test_write_steering_signals_empty(tmp_path: Path) -> None:
    """An empty signal list still writes a readable (empty) parquet (R2.4)."""
    out = tmp_path / "macro_steering_signals_empty.parquet"
    write_steering_signals([], out)
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 0


def test_steering_signal_is_frozen() -> None:
    """SteeringSignal is a frozen dataclass (immutable per design)."""
    panel = _macro_panel(_NEUTRAL, ["2020-02-29"])
    sig = characterize(panel, pd.Timestamp("2020-03-31"))
    assert isinstance(sig, SteeringSignal)
    with pytest.raises(Exception):
        sig.regime_label = "tampered"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Task 2.5 — View confidence shaping and gating (ViewSteerer)
#
# steer_views(views, p_memorized, signal, config) produces a NEW list of
# MacroView objects whose ONLY changed field is `confidence`:
#   adjusted = clip(base * (1 - p_memorized) * signal.consistency(view), 0, 1)
# (R3.1, R3.4). The macro-consistency floor is already applied INSIDE
# signal.consistency (built with a floor by characterize in task 2.4) — it is
# NOT re-applied here (no double-flooring).
#
# Gating (R3.2): p_memorized >= config.threshold ⇒ the whole rebalance's views
# are dropped (empty list returned).
#
# Passthrough (R1.6 additive / R1.7 weak / disabled): when `not config.enabled`
# OR `p_memorized is None`, the input views are returned UNCHANGED. The
# composition (task 3.1) supplies p_memorized=None when the calibrator is_weak,
# so the weak-calibrator case is handled here as the None case (steer_views does
# not receive the scorer).
#
# steer_views never introduces a return/forecast objective (R3.5) and never
# mutates the input list or its views (R3.4). Tests use the REAL MacroView and a
# REAL SteeringSignal from characterize on a small synthetic panel.
# ---------------------------------------------------------------------------

from macro_framework.steering import (
    SteeringConfig,
    steer_views,
)


def _signal_for(rows: dict, *, consistency_floor: float = 0.5) -> SteeringSignal:
    """A real SteeringSignal from characterize on a one-row synthetic panel."""
    panel = _macro_panel(rows, ["2020-02-29"])
    return characterize(panel, pd.Timestamp("2020-03-31"), consistency_floor=consistency_floor)


def test_steering_config_defaults() -> None:
    """SteeringConfig is a frozen dataclass with the design's defaults."""
    cfg = SteeringConfig()
    assert cfg.enabled is True
    assert cfg.threshold == 0.8
    assert cfg.consistency_floor == 0.5
    with pytest.raises(Exception):
        cfg.enabled = False  # type: ignore[misc]


def test_steer_views_monotonic_in_p_memorized() -> None:
    """Higher p_memorized ⇒ strictly lower adjusted confidence (R3.1)."""
    # Fixed view + signal; vary only p_memorized (all below the 0.8 threshold).
    sig = _signal_for(_CREDIT_STRESS)  # prefers gold/cash
    view = _view("Asset_C", confidence=0.6)  # gold -> consistency 1.0
    confs = [
        steer_views([view], p_mem, sig)[0].confidence
        for p_mem in (0.0, 0.2, 0.5, 0.7)
    ]
    # Strictly decreasing as p_memorized rises (pairwise; consecutive slices).
    assert all(earlier > later for earlier, later in zip(confs, confs[1:]))


def test_steer_views_exact_formula_preferred_view() -> None:
    """adjusted == base*(1-p_mem)*consistency, clipped to [0,1] (R3.1, R3.4)."""
    sig = _signal_for(_CREDIT_STRESS)
    base = 0.6
    p_mem = 0.25
    # Asset_C = gold -> preferred under credit_stress -> consistency 1.0.
    view = _view("Asset_C", confidence=base)
    cons = sig.consistency(view)
    assert cons == 1.0
    out = steer_views([view], p_mem, sig)
    expected = base * (1.0 - p_mem) * cons
    assert out[0].confidence == pytest.approx(expected)


def test_steer_views_exact_formula_contradicting_view_uses_signal_floor() -> None:
    """Consistency floor comes from the SIGNAL, applied once (no double-floor)."""
    floor = 0.5
    sig = _signal_for(_CREDIT_STRESS, consistency_floor=floor)
    base = 0.6
    p_mem = 0.25
    # Asset_B = tech -> NOT preferred under credit_stress -> consistency == floor.
    view = _view("Asset_B", confidence=base)
    cons = sig.consistency(view)
    assert cons == floor
    out = steer_views([view], p_mem, sig)
    expected = base * (1.0 - p_mem) * floor
    assert out[0].confidence == pytest.approx(expected)


def test_steer_views_clips_to_unit_interval() -> None:
    """Adjusted confidence is clipped into [0, 1] (R3.4)."""
    sig = _signal_for(_NEUTRAL)  # neutral prefers all -> consistency 1.0
    # A pathological over-1 base confidence with p_mem=0 would exceed 1 unclipped.
    view = _view("Asset_A", confidence=1.5)
    out = steer_views([view], 0.0, sig)
    assert out[0].confidence == 1.0
    # And it never goes below 0 (consistency/(1-p_mem) are non-negative anyway).
    view0 = _view("Asset_A", confidence=0.0)
    out0 = steer_views([view0], 0.5, sig)
    assert out0[0].confidence == 0.0


def test_steer_views_excludes_at_threshold(monkeypatch=None) -> None:
    """p_memorized >= threshold ⇒ empty list (whole rebalance dropped) (R3.2)."""
    sig = _signal_for(_NEUTRAL)
    cfg = SteeringConfig(threshold=0.8)
    views = [_view("Asset_A"), _view("Asset_B")]
    # At the threshold.
    assert steer_views(views, 0.8, sig, cfg) == []
    # Above the threshold.
    assert steer_views(views, 0.95, sig, cfg) == []


def test_steer_views_below_threshold_keeps_views() -> None:
    """p_memorized just below threshold ⇒ views retained (shaped, not dropped)."""
    sig = _signal_for(_NEUTRAL)
    cfg = SteeringConfig(threshold=0.8)
    views = [_view("Asset_A", confidence=0.6)]
    out = steer_views(views, 0.79, sig, cfg)
    assert len(out) == 1


def test_steer_views_passthrough_when_disabled() -> None:
    """enabled=False ⇒ input views returned unchanged (R1.6)."""
    sig = _signal_for(_CREDIT_STRESS)
    cfg = SteeringConfig(enabled=False)
    views = [_view("Asset_C", confidence=0.6), _view("Asset_B", confidence=0.4)]
    out = steer_views(views, 0.3, sig, cfg)
    assert [v.confidence for v in out] == [0.6, 0.4]
    assert [v.to_dict() for v in out] == [v.to_dict() for v in views]


def test_steer_views_passthrough_when_p_memorized_none() -> None:
    """p_memorized is None (fail / weak calibrator) ⇒ unchanged (R1.6/R1.7)."""
    sig = _signal_for(_CREDIT_STRESS)
    views = [_view("Asset_C", confidence=0.6), _view("Asset_B", confidence=0.4)]
    out = steer_views(views, None, sig)
    assert [v.confidence for v in out] == [0.6, 0.4]
    assert [v.to_dict() for v in out] == [v.to_dict() for v in views]


def test_steer_views_does_not_mutate_input() -> None:
    """Input list and its views are never mutated; output is fresh objects (R3.4)."""
    sig = _signal_for(_CREDIT_STRESS)
    view = _view("Asset_C", confidence=0.6)
    views = [view]
    before = view.to_dict()
    out = steer_views(views, 0.25, sig)
    # Input view object untouched.
    assert view.to_dict() == before
    assert view.confidence == 0.6
    # New list and new object identity.
    assert out is not views
    assert out[0] is not view


def test_steer_views_changes_only_confidence() -> None:
    """Only confidence differs; legs/expected_excess/rationale identical (R3.4)."""
    sig = _signal_for(_CREDIT_STRESS)
    view = MacroView(
        asset_long="Asset_C",
        asset_short="Asset_B",
        expected_excess_annualized=0.07,
        confidence=0.6,
        rationale="defensive tilt",
    )
    out = steer_views([view], 0.25, sig)[0]
    assert out.asset_long == view.asset_long
    assert out.asset_short == view.asset_short
    assert out.expected_excess_annualized == view.expected_excess_annualized
    assert out.rationale == view.rationale
    assert out.confidence != view.confidence


def test_steer_views_deterministic() -> None:
    """Equal inputs ⇒ equal output (determinism)."""
    sig = _signal_for(_GOLDILOCKS)
    views = [_view("Asset_A", confidence=0.6), _view("Asset_D", confidence=0.4)]
    a = steer_views(views, 0.3, sig)
    b = steer_views([_view("Asset_A", confidence=0.6), _view("Asset_D", confidence=0.4)], 0.3, sig)
    assert [v.to_dict() for v in a] == [v.to_dict() for v in b]


def test_steer_views_empty_views_returns_empty() -> None:
    """No views in ⇒ empty list out (and not the same object)."""
    sig = _signal_for(_NEUTRAL)
    views: list[MacroView] = []
    out = steer_views(views, 0.3, sig)
    assert out == []


# ---------------------------------------------------------------------------
# Task 2.6 — Prompt-version variant agent (VariantMacroAgent)
#
# VariantMacroAgent subclasses the COMMITTED LlmMacroAgent (read-only import) to
# run alternative prompt versions WITHOUT editing llm_agent.py. It stores custom
# `instructions` + `prompt_version` and overrides the PRIVATE `_ensure_ready` to
# rebuild the DSPy signature's docstring from `self.instructions` instead of the
# module-level AGENT_INSTRUCTIONS, while keeping the LM config, dspy.Predict, and
# the diskcache wiring equivalent to the base. Each variant must use a DISTINCT
# cache_dir so versions never alias each other (the base _cache_key keys on the
# module-level PROMPT_VERSION="v1" only), and must never write the base
# .llm_cache/ (the module-level CACHE_DIR). Prior versions are preserved by
# distinct cache dirs (R4.4).
#
# _ensure_ready is OFFLINE-SAFE to run: dspy.LM(...) construction and
# dspy.Predict(...) do NOT call the network — only views_for_state -> _predict
# would. We monkeypatch the api-key env var so the base key check passes; no
# network occurs. Requirements: 4.1 (variant instructions drive the prompt),
# 4.4 (distinct caches / prior versions preserved).
# ---------------------------------------------------------------------------

import macro_framework.llm_agent as _llm_agent_mod
from macro_framework.llm_agent import (
    AGENT_INSTRUCTIONS,
    CACHE_DIR,
    LlmMacroAgent,
)
from macro_framework.steering import VariantMacroAgent

_VARIANT_INSTRUCTIONS_V2 = (
    "You are a disciplined macro strategist (variant v2). Reason only from the "
    "numeric state in front of you and emit Black-Litterman view triples. Never "
    "reference calendar dates, years, or real tickers."
)
_VARIANT_INSTRUCTIONS_V3 = (
    "You are a cautious macro risk officer (variant v3). Prefer fewer, higher "
    "confidence views and avoid any forecast objective. Never reference dates."
)


def test_variant_is_subclass_of_base() -> None:
    """VariantMacroAgent subclasses LlmMacroAgent (additive, no base edit) (R4.1)."""
    assert issubclass(VariantMacroAgent, LlmMacroAgent)


def test_variant_stores_instructions_and_prompt_version(tmp_path: Path) -> None:
    """instructions + prompt_version are stored/retrievable for artifact naming (R4.4)."""
    agent = VariantMacroAgent(
        instructions=_VARIANT_INSTRUCTIONS_V2,
        prompt_version="v2",
        cache_dir=tmp_path / ".llm_cache_v2",
    )
    assert agent.instructions == _VARIANT_INSTRUCTIONS_V2
    assert agent.prompt_version == "v2"


def test_variant_reuses_base_wiring_via_super(tmp_path: Path) -> None:
    """super().__init__ wires model/api_base/api_key_env/cache_dir from the base (R4.1)."""
    agent = VariantMacroAgent(
        instructions=_VARIANT_INSTRUCTIONS_V2,
        prompt_version="v2",
        cache_dir=tmp_path / ".llm_cache_v2",
        model="openrouter/test/model",
        api_base="https://example.test/v1",
        api_key_env="MY_KEY",
        temperature=0.0,
        max_tokens=512,
    )
    # Base attributes set by LlmMacroAgent.__init__ are present and forwarded.
    assert agent.model == "openrouter/test/model"
    assert agent.api_base == "https://example.test/v1"
    assert agent.api_key_env == "MY_KEY"
    assert agent.max_tokens == 512
    assert agent.cache_dir == Path(tmp_path / ".llm_cache_v2")
    # Lazy wiring not built yet (mirrors the base contract).
    assert agent._predict is None


def test_variant_distinct_versions_use_distinct_cache_dirs(tmp_path: Path) -> None:
    """Two variants with different prompt_version resolve to different cache_dirs (R4.4)."""
    a = VariantMacroAgent(
        instructions=_VARIANT_INSTRUCTIONS_V2,
        prompt_version="v2",
        cache_dir=tmp_path / ".llm_cache_v2",
    )
    b = VariantMacroAgent(
        instructions=_VARIANT_INSTRUCTIONS_V3,
        prompt_version="v3",
        cache_dir=tmp_path / ".llm_cache_v3",
    )
    assert a.cache_dir != b.cache_dir


def test_variant_cache_dir_is_not_base_cache_dir(tmp_path: Path) -> None:
    """A variant's cache_dir never equals the base module-level CACHE_DIR (R4.4)."""
    a = VariantMacroAgent(
        instructions=_VARIANT_INSTRUCTIONS_V2,
        prompt_version="v2",
        cache_dir=tmp_path / ".llm_cache_v2",
    )
    b = VariantMacroAgent(
        instructions=_VARIANT_INSTRUCTIONS_V3,
        prompt_version="v3",
        cache_dir=tmp_path / ".llm_cache_v3",
    )
    assert a.cache_dir != Path(CACHE_DIR)
    assert b.cache_dir != Path(CACHE_DIR)


def test_variant_ensure_ready_uses_variant_instructions(
    monkeypatch, tmp_path: Path
) -> None:
    """After _ensure_ready, the built signature carries self.instructions, NOT the
    module AGENT_INSTRUCTIONS (R4.1)."""
    # Offline-safe: dspy.LM/dspy.Predict do not hit the network; satisfy the base
    # api-key guard so _ensure_ready reaches signature construction.
    monkeypatch.setenv("OPENROUTER_KEY", "test")
    agent = VariantMacroAgent(
        instructions=_VARIANT_INSTRUCTIONS_V2,
        prompt_version="v2",
        cache_dir=tmp_path / ".llm_cache_v2",
    )
    agent._ensure_ready()

    # The predict's signature docstring carries the variant instructions.
    sig = agent._predict.signature
    docstring = sig.__doc__
    assert _VARIANT_INSTRUCTIONS_V2 in docstring
    # The base instructions must NOT drive the variant prompt.
    assert docstring != AGENT_INSTRUCTIONS
    assert AGENT_INSTRUCTIONS not in docstring


def test_variant_ensure_ready_builds_predict_and_cache(
    monkeypatch, tmp_path: Path
) -> None:
    """_ensure_ready wires _predict + a diskcache at self.cache_dir (mirrors base)."""
    import diskcache

    monkeypatch.setenv("OPENROUTER_KEY", "test")
    cache_dir = tmp_path / ".llm_cache_v2"
    agent = VariantMacroAgent(
        instructions=_VARIANT_INSTRUCTIONS_V2,
        prompt_version="v2",
        cache_dir=cache_dir,
    )
    agent._ensure_ready()
    assert agent._predict is not None
    assert isinstance(agent._cache, diskcache.Cache)
    # The diskcache lives at the variant's cache_dir, not the base CACHE_DIR.
    assert Path(agent._cache.directory) == cache_dir
    assert Path(agent._cache.directory) != Path(CACHE_DIR)


def test_variant_ensure_ready_is_idempotent(monkeypatch, tmp_path: Path) -> None:
    """Calling _ensure_ready twice keeps the same _predict (lazy-init guard, like base)."""
    monkeypatch.setenv("OPENROUTER_KEY", "test")
    agent = VariantMacroAgent(
        instructions=_VARIANT_INSTRUCTIONS_V2,
        prompt_version="v2",
        cache_dir=tmp_path / ".llm_cache_v2",
    )
    agent._ensure_ready()
    first = agent._predict
    agent._ensure_ready()
    assert agent._predict is first


def test_variant_missing_api_key_raises(monkeypatch, tmp_path: Path) -> None:
    """The base credential guard is preserved: missing key ⇒ RuntimeError."""
    monkeypatch.delenv("OPENROUTER_KEY", raising=False)
    agent = VariantMacroAgent(
        instructions=_VARIANT_INSTRUCTIONS_V2,
        prompt_version="v2",
        cache_dir=tmp_path / ".llm_cache_v2",
        api_key_env="DEFINITELY_UNSET_KEY_FOR_TEST",
    )
    monkeypatch.delenv("DEFINITELY_UNSET_KEY_FOR_TEST", raising=False)
    with pytest.raises(RuntimeError):
        agent._ensure_ready()


def test_variant_does_not_write_base_cache(monkeypatch, tmp_path: Path) -> None:
    """Constructing/preparing a variant never creates/writes the base .llm_cache/ (R4.4)."""
    monkeypatch.setenv("OPENROUTER_KEY", "test")
    base_cache = Path(CACHE_DIR)
    base_existed = base_cache.exists()

    agent = VariantMacroAgent(
        instructions=_VARIANT_INSTRUCTIONS_V2,
        prompt_version="v2",
        cache_dir=tmp_path / ".llm_cache_v2",
    )
    agent._ensure_ready()
    # Variant wired its own cache; the base cache dir must not be created here.
    if not base_existed:
        assert not base_cache.exists(), "variant must not create the base .llm_cache/"


def test_variant_does_not_alter_base_module_instructions(tmp_path: Path) -> None:
    """Constructing a variant never mutates the base module AGENT_INSTRUCTIONS (R6.1)."""
    snapshot = _llm_agent_mod.AGENT_INSTRUCTIONS
    VariantMacroAgent(
        instructions=_VARIANT_INSTRUCTIONS_V2,
        prompt_version="v2",
        cache_dir=tmp_path / ".llm_cache_v2",
    )
    assert _llm_agent_mod.AGENT_INSTRUCTIONS == snapshot


def test_variant_base_agent_unaffected_by_variant(tmp_path: Path) -> None:
    """A base LlmMacroAgent built alongside a variant keeps the base cache_dir (R4.4/R6.1)."""
    base = LlmMacroAgent()
    VariantMacroAgent(
        instructions=_VARIANT_INSTRUCTIONS_V2,
        prompt_version="v2",
        cache_dir=tmp_path / ".llm_cache_v2",
    )
    # The base agent still points at the base CACHE_DIR, untouched by the variant.
    assert base.cache_dir == Path(CACHE_DIR)


# ---------------------------------------------------------------------------
# Task 2.7 — Score distribution reporting (score_distribution_report)
#
# score_distribution_report(scores) is a PURE summary over a list of REAL
# recall_guard.GuardedScore objects. It computes the p_memorized distribution
# (mean / median / p90) over the parse-OK scores ONLY, the parse_fail_rate
# (fraction with parse_ok False / p_memorized None), and n_scored (total count).
# It MAY also include a report-only memguard_confidence mean (report-only per the
# design Data Models note — never a steering input). It must NOT read the
# directional `signal` or `raw_confidence` (non-predictive boundary).
#
# holdout_auc is NOT a GuardedScore field (it is a scorer property); it is
# accepted ONLY as an optional keyword arg and surfaced in the dict when given.
#
# The empty-list and all-failed cases are handled gracefully (no ZeroDivision /
# no NaN): well-defined values with a documented sentinel for the undefined mean.
#
# Requirements: 4.2 (report the p_memorized distribution per prompt version),
# 5.2 (additionally report the p_memorized distribution for the variant).
# ---------------------------------------------------------------------------

from recall_guard import GuardedScore

from macro_framework.steering import score_distribution_report

# Documented sentinel returned for distribution stats when there is no parse-OK
# score to summarize (empty list / all-failed). Mirrors steering's contract.
_UNDEFINED_STAT = 0.0


def _ok_score(prompt: str, p_memorized: float, *, raw_confidence: float = 0.5) -> GuardedScore:
    """A REAL parse-OK GuardedScore with a known p_memorized and memguard_confidence."""
    from hashlib import sha256

    return GuardedScore(
        prompt_hash=sha256(prompt.encode("utf-8")).hexdigest()[:16],
        parse_ok=True,
        signal=1,
        raw_confidence=raw_confidence,
        p_memorized=p_memorized,
        memguard_confidence=raw_confidence * (1.0 - p_memorized),
        features=None,
        fail_reason=None,
    )


def _fail_score(prompt: str, *, fail_reason: str = "parse_failure") -> GuardedScore:
    """A REAL parse-failure GuardedScore (parse_ok False, p_memorized None)."""
    from hashlib import sha256

    return GuardedScore(
        prompt_hash=sha256(prompt.encode("utf-8")).hexdigest()[:16],
        parse_ok=False,
        signal=None,
        raw_confidence=None,
        p_memorized=None,
        memguard_confidence=None,
        features=None,
        fail_reason=fail_reason,
    )


def test_score_distribution_report_mix_aggregates_over_ok_only() -> None:
    """mean/median/p90 over parse-OK scores only; fail rate + n_scored correct (4.2, 5.2)."""
    # 4 OK scores with known p_memorized + 1 failure record.
    ok_p = [0.1, 0.2, 0.3, 0.4]
    scores = [_ok_score(f"ok-{i}", p) for i, p in enumerate(ok_p)] + [_fail_score("bad")]
    rep = score_distribution_report(scores)

    assert rep["n_scored"] == 5  # total count, OK + failed
    # parse_fail_rate = 1 failed / 5 total.
    assert rep["parse_fail_rate"] == pytest.approx(1 / 5)
    # Distribution computed over the OK p_memorized values ONLY.
    assert rep["p_mem_mean"] == pytest.approx(sum(ok_p) / len(ok_p))
    assert rep["p_mem_median"] == pytest.approx(0.25)  # median of [.1,.2,.3,.4]
    # p90 of the OK values (lies in [max-1 step, max]).
    assert 0.3 <= rep["p_mem_p90"] <= 0.4


def test_score_distribution_report_ignores_failed_in_distribution() -> None:
    """A failure record (p_memorized None) must not pull the mean toward 0 (4.2)."""
    scores = [_ok_score("a", 0.5), _ok_score("b", 0.5), _fail_score("c")]
    rep = score_distribution_report(scores)
    # Mean over OK only = 0.5 (NOT (0.5+0.5+0)/3).
    assert rep["p_mem_mean"] == pytest.approx(0.5)
    assert rep["n_scored"] == 3
    assert rep["parse_fail_rate"] == pytest.approx(1 / 3)


def test_score_distribution_report_all_ok_zero_fail_rate() -> None:
    """All parse-OK ⇒ parse_fail_rate 0.0; n_scored = count (5.2)."""
    scores = [_ok_score(f"p{i}", p) for i, p in enumerate([0.0, 0.5, 1.0])]
    rep = score_distribution_report(scores)
    assert rep["parse_fail_rate"] == 0.0
    assert rep["n_scored"] == 3
    assert rep["p_mem_mean"] == pytest.approx(0.5)
    assert rep["p_mem_median"] == pytest.approx(0.5)


def test_score_distribution_report_empty_list_is_graceful() -> None:
    """Empty input ⇒ well-defined output, no ZeroDivision / NaN (graceful)."""
    rep = score_distribution_report([])
    assert rep["n_scored"] == 0
    assert rep["parse_fail_rate"] == 0.0  # 0 failed / 0 total -> documented 0.0
    # Distribution stats are the documented sentinel (no OK scores to summarize).
    for key in ("p_mem_mean", "p_mem_median", "p_mem_p90"):
        assert rep[key] == _UNDEFINED_STAT
    # No NaN anywhere.
    import math

    assert all(not math.isnan(v) for v in rep.values())


def test_score_distribution_report_all_failed_is_graceful() -> None:
    """All-failed ⇒ parse_fail_rate 1.0; distribution stats are the sentinel."""
    scores = [_fail_score("x"), _fail_score("y"), _fail_score("z")]
    rep = score_distribution_report(scores)
    assert rep["n_scored"] == 3
    assert rep["parse_fail_rate"] == pytest.approx(1.0)
    for key in ("p_mem_mean", "p_mem_median", "p_mem_p90"):
        assert rep[key] == _UNDEFINED_STAT
    import math

    assert all(not math.isnan(v) for v in rep.values())


def test_score_distribution_report_makes_no_network_calls(monkeypatch) -> None:
    """The function only reads dataclass fields — zero external/network calls."""
    import urllib.request

    def _boom(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("score_distribution_report made a network call")

    monkeypatch.setattr(urllib.request, "urlopen", _boom, raising=False)
    # recall_guard's scorer would call out; ensure the reporter never touches it.
    scores = [_ok_score("a", 0.2), _fail_score("b")]
    rep = score_distribution_report(scores)
    assert rep["n_scored"] == 2


def test_score_distribution_report_reports_memguard_confidence_mean() -> None:
    """Report-only memguard_confidence mean over OK scores (never a steering input)."""
    # memguard_confidence = raw_confidence * (1 - p_memorized).
    s1 = _ok_score("a", 0.0, raw_confidence=0.6)   # mgc = 0.6
    s2 = _ok_score("b", 0.5, raw_confidence=0.6)   # mgc = 0.3
    rep = score_distribution_report([s1, s2, _fail_score("c")])
    assert rep["memguard_confidence_mean"] == pytest.approx((0.6 + 0.3) / 2)


def test_score_distribution_report_holdout_auc_optional_kwarg() -> None:
    """holdout_auc appears only when passed (it is a scorer property, not a field)."""
    scores = [_ok_score("a", 0.2)]
    # Absent by default — the function must NOT invent it from the scores.
    rep_default = score_distribution_report(scores)
    assert "holdout_auc" not in rep_default
    # Present and equal to the value when supplied as a keyword arg.
    rep_with = score_distribution_report(scores, holdout_auc=0.83)
    assert rep_with["holdout_auc"] == pytest.approx(0.83)


def test_score_distribution_report_does_not_read_directional_signal() -> None:
    """The non-predictive boundary: report keys never expose signal/raw_confidence."""
    scores = [_ok_score("a", 0.3, raw_confidence=0.9), _ok_score("b", 0.4)]
    rep = score_distribution_report(scores)
    # No raw_confidence / signal mean leaks into the report dict.
    assert "raw_confidence_mean" not in rep
    assert "signal_mean" not in rep
    assert "signal" not in rep


def test_score_distribution_report_returns_plain_float_dict() -> None:
    """Values are plain floats (dict[str, float]); JSON/serialization-friendly."""
    scores = [_ok_score("a", 0.1), _ok_score("b", 0.9), _fail_score("c")]
    rep = score_distribution_report(scores, holdout_auc=0.7)
    assert isinstance(rep, dict)
    for key, value in rep.items():
        assert isinstance(key, str)
        assert isinstance(value, float), f"{key} -> {type(value)} not float"


# ---------------------------------------------------------------------------
# Task 3.1 — Steered rebalance composition for walk-forward (integration)
#
# steer_rebalance(...) composes the already-built pieces — render_directional
# (2.1), ScoringAdapter.score_rebalances (2.3), characterize (2.4), steer_views
# (2.5) — plus the agent's UNCHANGED views_to_bl, into one steered decision step,
# and make_steered_weight_fn(...) wraps it into a walk_forward weight_fn(ctx) ->
# pd.Series. nb09's HRP/BL/blend math is INJECTED (build_inputs / combine), never
# owned here (R6.1).
#
# The agent and scorer are MOCKED (no DSPy/OpenRouter, no NIM/FMP, no network):
#   * A FAKE agent mirrors the real signatures: views_for_state(macro_state,
#     asset_snapshot) -> (list[MacroView], reasoning); views_to_bl(views,
#     real_symbols) -> (P, Q) or (None, None) when no view survives.
#   * A FAKE scorer mirrors ScoringAdapter: is_weak + score_rebalances([...]) ->
#     [GuardedScore] (REAL recall_guard.GuardedScore).
#
# Asserts (design "Steered rebalance" sequence + ViewSteerer notes):
#   (a) STEERED path (is_weak False, real p_memorized) runs end-to-end through
#       views_to_bl to a valid target row, steered=True, confidences shaped.
#   (b) UNSTEERED-FALLBACK (is_weak True OR a failure GuardedScore OR
#       config.enabled False) ⇒ steered=False with the original (unsteered) views.
#   (c) All views excluded by the gate ⇒ views_to_bl -> (None, None) and combine
#       still produces a valid (base) row.
#   real_symbols sourced from price columns in the weight_fn path.
#
# Requirements: 1.6 (additive/passthrough), 3.1 (confidence falls with
# p_memorized), 3.2 (gate/exclude), 3.3 (composition feeds nb11's own targets),
# 3.4 (only confidence/inclusion shaping).
# ---------------------------------------------------------------------------

from macro_framework.anonymize import AssetMap
from macro_framework.steering import (
    SteeredDecision,
    make_steered_weight_fn,
    steer_rebalance,
)


class _FakeAgent:
    """Stand-in for LlmMacroAgent: mirrors views_for_state + views_to_bl.

    views_for_state returns a fixed list of REAL MacroView objects (and a
    reasoning string). views_to_bl emits a deterministic, simple (P, Q) keyed by
    the real symbols for the surviving views, or (None, None) when no view maps
    (mirroring the real method's empty-views contract).
    """

    def __init__(self, views: list[MacroView], asset_map: AssetMap | None = None) -> None:
        self._views = views
        self.asset_map = asset_map or AssetMap.default()
        self.calls_views_for_state: list[tuple] = []
        self.calls_views_to_bl: list[list[MacroView]] = []

    def views_for_state(self, macro_state, asset_snapshot):
        self.calls_views_for_state.append((macro_state, asset_snapshot))
        return list(self._views), "fake reasoning"

    def views_to_bl(self, views, real_symbols):
        self.calls_views_to_bl.append(list(views))
        pseudo_to_real = self.asset_map.pseudo_to_real
        rows_P, rows_Q = [], []
        for v in views:
            long_real = pseudo_to_real.get(v.asset_long)
            if long_real is None or long_real not in real_symbols:
                continue
            row = [1.0 if s == long_real else 0.0 for s in real_symbols]
            rows_P.append(row)
            rows_Q.append([(v.expected_excess_annualized * v.confidence) / 252.0])
        if not rows_P:
            return None, None
        P = pd.DataFrame(rows_P, columns=list(real_symbols))
        Q = pd.DataFrame(rows_Q)
        return P, Q


class _FakeScoringAdapter:
    """Stand-in for ScoringAdapter: is_weak + score_rebalances -> [GuardedScore].

    Returns REAL recall_guard.GuardedScore objects so the composition's reads of
    p_memorized / parse_ok / fail_reason stay honest. ``fail`` forces a parse
    failure (p_memorized None) to exercise the unsteered fallback; ``p_memorized``
    sets the score on the success path.
    """

    def __init__(self, *, is_weak: bool = False, p_memorized: float = 0.2, fail: bool = False) -> None:
        self.is_weak = is_weak
        self._p_memorized = p_memorized
        self._fail = fail
        self.calls: list[list[str]] = []

    def score_rebalances(self, prompts):
        self.calls.append(list(prompts))
        out = []
        for p in prompts:
            if self._fail:
                out.append(_fail_score(p))
            else:
                out.append(_ok_score(p, self._p_memorized))
        return out


# A real, on-cutoff asset map maps Asset_A..D to real tickers; we source the real
# symbols from those so views_to_bl can place tilts.
_AMAP = AssetMap.default()
_REAL_SYMBOLS = list(_AMAP.pseudo_to_real.values())


def _credit_stress_hist() -> pd.DataFrame:
    """A one-row macro panel that characterize labels 'credit_stress'."""
    return _macro_panel(_CREDIT_STRESS, ["2020-02-29"])


def _gold_view(confidence: float = 0.6) -> MacroView:
    """A view long the gold pseudo asset (preferred under credit_stress)."""
    return MacroView(
        asset_long="Asset_C",
        asset_short=None,
        expected_excess_annualized=0.08,
        confidence=confidence,
        rationale="defensive tilt",
    )


def test_steer_rebalance_steered_path_runs_end_to_end() -> None:
    """is_weak False + real p_memorized ⇒ steered=True, shaped views, valid (P, Q) (3.1)."""
    view = _gold_view(confidence=0.6)
    agent = _FakeAgent([view])
    scorer = _FakeScoringAdapter(is_weak=False, p_memorized=0.2)
    rb = pd.Timestamp("2020-03-31")

    dec = steer_rebalance(
        agent=agent,
        scorer=scorer,
        macro_state=dict(MACRO_STATE),
        asset_snapshot=[dict(a) for a in ASSET_SNAPSHOT],
        macro_hist=_credit_stress_hist(),
        rebalance_date=rb,
        real_symbols=_REAL_SYMBOLS,
    )

    assert isinstance(dec, SteeredDecision)
    assert dec.steered is True
    assert dec.p_memorized == pytest.approx(0.2)
    assert dec.regime_label == "credit_stress"
    # Confidence actually shaped down: base 0.6 * (1-0.2) * consistency(1.0) = 0.48.
    assert len(dec.steered_views) == 1
    assert dec.steered_views[0].confidence == pytest.approx(0.6 * 0.8 * 1.0)
    # Ran through views_to_bl to a valid (P, Q).
    assert dec.P is not None and dec.Q is not None
    assert list(dec.P.columns) == _REAL_SYMBOLS
    assert len(dec.P) == 1
    # The scorer saw the directional prompt for the macro content.
    assert len(scorer.calls) == 1 and len(scorer.calls[0]) == 1


def test_steer_rebalance_weak_calibrator_falls_back_unsteered() -> None:
    """is_weak ⇒ p_memorized None ⇒ steered=False, original (unsteered) views (1.6/1.7)."""
    view = _gold_view(confidence=0.6)
    agent = _FakeAgent([view])
    scorer = _FakeScoringAdapter(is_weak=True)
    rb = pd.Timestamp("2020-03-31")

    dec = steer_rebalance(
        agent=agent,
        scorer=scorer,
        macro_state=dict(MACRO_STATE),
        asset_snapshot=[dict(a) for a in ASSET_SNAPSHOT],
        macro_hist=_credit_stress_hist(),
        rebalance_date=rb,
        real_symbols=_REAL_SYMBOLS,
    )

    assert dec.steered is False
    assert dec.p_memorized is None
    # Weak calibrator never scored (scorer skipped on the is_weak path).
    assert scorer.calls == []
    # Views are the unsteered originals (confidence unchanged).
    assert [v.confidence for v in dec.steered_views] == [0.6]
    # Still produces a decision through views_to_bl.
    assert dec.P is not None and dec.Q is not None


def test_steer_rebalance_scoring_failure_falls_back_unsteered() -> None:
    """A failure GuardedScore (p_memorized None) ⇒ steered=False, unsteered views (1.6)."""
    view = _gold_view(confidence=0.6)
    agent = _FakeAgent([view])
    scorer = _FakeScoringAdapter(is_weak=False, fail=True)
    rb = pd.Timestamp("2020-03-31")

    dec = steer_rebalance(
        agent=agent,
        scorer=scorer,
        macro_state=dict(MACRO_STATE),
        asset_snapshot=[dict(a) for a in ASSET_SNAPSHOT],
        macro_hist=_credit_stress_hist(),
        rebalance_date=rb,
        real_symbols=_REAL_SYMBOLS,
    )

    assert dec.steered is False
    assert dec.p_memorized is None
    assert [v.confidence for v in dec.steered_views] == [0.6]
    assert dec.P is not None and dec.Q is not None


def test_steer_rebalance_disabled_falls_back_unsteered() -> None:
    """config.enabled False ⇒ steered=False, no scoring, original views (1.6)."""
    view = _gold_view(confidence=0.6)
    agent = _FakeAgent([view])
    scorer = _FakeScoringAdapter(is_weak=False, p_memorized=0.2)
    rb = pd.Timestamp("2020-03-31")

    dec = steer_rebalance(
        agent=agent,
        scorer=scorer,
        macro_state=dict(MACRO_STATE),
        asset_snapshot=[dict(a) for a in ASSET_SNAPSHOT],
        macro_hist=_credit_stress_hist(),
        rebalance_date=rb,
        real_symbols=_REAL_SYMBOLS,
        config=SteeringConfig(enabled=False),
    )

    assert dec.steered is False
    assert dec.p_memorized is None
    assert scorer.calls == []  # disabled ⇒ never scored
    assert [v.confidence for v in dec.steered_views] == [0.6]


def test_steer_rebalance_no_scorer_falls_back_unsteered() -> None:
    """scorer is None ⇒ steered=False, p_memorized None, unsteered views (1.6)."""
    view = _gold_view(confidence=0.6)
    agent = _FakeAgent([view])
    rb = pd.Timestamp("2020-03-31")

    dec = steer_rebalance(
        agent=agent,
        scorer=None,
        macro_state=dict(MACRO_STATE),
        asset_snapshot=[dict(a) for a in ASSET_SNAPSHOT],
        macro_hist=_credit_stress_hist(),
        rebalance_date=rb,
        real_symbols=_REAL_SYMBOLS,
    )

    assert dec.steered is False
    assert dec.p_memorized is None
    assert [v.confidence for v in dec.steered_views] == [0.6]


def test_steer_rebalance_gate_excludes_all_views_yields_none_none() -> None:
    """p_memorized >= threshold ⇒ all views excluded ⇒ views_to_bl (None, None) (3.2)."""
    view = _gold_view(confidence=0.6)
    agent = _FakeAgent([view])
    scorer = _FakeScoringAdapter(is_weak=False, p_memorized=0.95)  # >= 0.8 threshold
    rb = pd.Timestamp("2020-03-31")

    dec = steer_rebalance(
        agent=agent,
        scorer=scorer,
        macro_state=dict(MACRO_STATE),
        asset_snapshot=[dict(a) for a in ASSET_SNAPSHOT],
        macro_hist=_credit_stress_hist(),
        rebalance_date=rb,
        real_symbols=_REAL_SYMBOLS,
    )

    # Gate fired: empty steered views; views_to_bl saw an empty list ⇒ (None, None).
    assert dec.steered_views == []
    assert dec.P is None and dec.Q is None
    # p_memorized was measured (gate is a steered outcome, not a fallback).
    assert dec.p_memorized == pytest.approx(0.95)
    assert dec.steered is True


def _build_inputs_stub(ctx):
    """Inject build_inputs: derive a (macro_state, asset_snapshot) from ctx."""
    return dict(MACRO_STATE), [dict(a) for a in ASSET_SNAPSHOT]


def _combine_stub(ctx, P, Q):
    """Inject combine: a tiny base-vs-tilt blend returning weights over real symbols.

    When the steered views collapse to (None, None) we fall back to an equal-weight
    base allocation — exactly the Track A behaviour (BL posterior -> base). When a
    tilt survives, we lean the weight toward the long leg(s) of P. Always returns a
    valid (sums-to-1) weight Series over the real-symbol columns of ctx['prices'].
    """
    symbols = list(ctx["prices"].columns)
    base = pd.Series(1.0 / len(symbols), index=symbols)
    if P is None:
        return base
    tilt = P.sum(axis=0)
    tilt = tilt.reindex(symbols).fillna(0.0)
    blended = base + 0.1 * tilt
    return blended / blended.sum()


def _ctx_with_prices(rb: pd.Timestamp) -> dict:
    """A minimal walk_forward ctx: prices over the real symbols + macro_panel."""
    idx = pd.DatetimeIndex([pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")])
    prices = pd.DataFrame(
        {s: [100.0, 101.0] for s in _REAL_SYMBOLS}, index=idx
    )
    return {
        "rebalance_date": rb,
        "prices": prices,
        "returns": prices.pct_change().dropna(),
        "macro_panel": _credit_stress_hist(),
    }


def test_make_steered_weight_fn_steered_path_returns_valid_row() -> None:
    """weight_fn sources real_symbols from price columns, steers, returns valid weights (3.3)."""
    view = _gold_view(confidence=0.6)
    agent = _FakeAgent([view])
    scorer = _FakeScoringAdapter(is_weak=False, p_memorized=0.2)
    rb = pd.Timestamp("2020-03-31")

    weight_fn = make_steered_weight_fn(
        agent=agent,
        scorer=scorer,
        real_symbols=_REAL_SYMBOLS,
        build_inputs=_build_inputs_stub,
        combine=_combine_stub,
    )
    w = weight_fn(_ctx_with_prices(rb))

    assert isinstance(w, pd.Series)
    assert list(w.index) == _REAL_SYMBOLS
    assert w.sum() == pytest.approx(1.0)
    # Steered tilt leaned weight toward the gold leg above equal weight.
    gold_real = _AMAP.pseudo_to_real["Asset_C"]
    assert w[gold_real] > 1.0 / len(_REAL_SYMBOLS)
    # real_symbols in the weight_fn path come from the price columns.
    assert agent.calls_views_to_bl  # views_to_bl was invoked


def test_make_steered_weight_fn_unsteered_fallback_returns_base_row() -> None:
    """is_weak ⇒ unsteered; weight_fn still returns a valid (base-blended) row (1.6, 3.3)."""
    view = _gold_view(confidence=0.6)
    agent = _FakeAgent([view])
    scorer = _FakeScoringAdapter(is_weak=True)
    rb = pd.Timestamp("2020-03-31")

    weight_fn = make_steered_weight_fn(
        agent=agent,
        scorer=scorer,
        real_symbols=_REAL_SYMBOLS,
        build_inputs=_build_inputs_stub,
        combine=_combine_stub,
    )
    w = weight_fn(_ctx_with_prices(rb))

    assert isinstance(w, pd.Series)
    assert w.sum() == pytest.approx(1.0)
    assert scorer.calls == []  # weak ⇒ unscored


def test_make_steered_weight_fn_gate_excludes_all_falls_back_to_base() -> None:
    """All views gated ⇒ views_to_bl (None, None) ⇒ combine returns the base row (3.2)."""
    view = _gold_view(confidence=0.6)
    agent = _FakeAgent([view])
    scorer = _FakeScoringAdapter(is_weak=False, p_memorized=0.95)
    rb = pd.Timestamp("2020-03-31")

    weight_fn = make_steered_weight_fn(
        agent=agent,
        scorer=scorer,
        real_symbols=_REAL_SYMBOLS,
        build_inputs=_build_inputs_stub,
        combine=_combine_stub,
    )
    w = weight_fn(_ctx_with_prices(rb))

    # Combine fell back to the equal-weight base (no surviving tilt).
    assert isinstance(w, pd.Series)
    assert w.sum() == pytest.approx(1.0)
    for s in _REAL_SYMBOLS:
        assert w[s] == pytest.approx(1.0 / len(_REAL_SYMBOLS))


def test_steered_decision_is_frozen() -> None:
    """SteeredDecision is a frozen dataclass carrying the steered/unsteered outcome."""
    view = _gold_view(confidence=0.6)
    agent = _FakeAgent([view])
    scorer = _FakeScoringAdapter(is_weak=False, p_memorized=0.2)
    dec = steer_rebalance(
        agent=agent,
        scorer=scorer,
        macro_state=dict(MACRO_STATE),
        asset_snapshot=[dict(a) for a in ASSET_SNAPSHOT],
        macro_hist=_credit_stress_hist(),
        rebalance_date=pd.Timestamp("2020-03-31"),
        real_symbols=_REAL_SYMBOLS,
    )
    with pytest.raises(Exception):
        dec.steered = False  # type: ignore[misc]
