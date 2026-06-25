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
