"""Unit tests for the version-aware-factor-scoring feature.

This is the feature's test file; later tasks append to it. Task 1.1 is a
test-only foundation: it confirms the recall_guard public MIA-primitive surface
the number-native scoring path depends on is importable, and that the directional
MemoryGuardedScorer facade is NOT on that path (Requirement 6.5).

The feature scores factor prompts via the lower-level public primitives
(NvidiaLM logprobs -> compute_mia_features -> MCSCalibrator.predict_proba, with
build_baseline / train for calibration), bypassing the directional facade so no
direction+confidence parse is required.
"""

from __future__ import annotations

import importlib
import sys


def test_public_mia_primitives_importable() -> None:
    """Every public primitive the feature relies on imports cleanly (6.5).

    These are the lower-level number-native primitives listed in the design's
    Allowed Dependencies. The feature uses these instead of the directional
    facade so contamination is measured without a buy/sell + confidence parse.
    """
    from recall_guard import (
        LOGPROB_FLOOR,
        ControlBaseline,
        MCSCalibrator,
        MiaFeatures,
        NvidiaLM,
        build_baseline,
        compute_mia_features,
        standardise,
    )

    # Calibration + scoring primitives are usable shapes, not just names.
    assert callable(compute_mia_features)
    assert callable(build_baseline)
    assert callable(standardise)
    assert hasattr(NvidiaLM, "generate")
    assert hasattr(MCSCalibrator, "predict_proba")
    assert isinstance(LOGPROB_FLOOR, float)

    import dataclasses

    assert dataclasses.is_dataclass(MiaFeatures)
    assert dataclasses.is_dataclass(ControlBaseline)


def test_mcs_train_submodule_function_importable() -> None:
    """`train` imports from the mcs submodule; `train_mcs` is its top-level alias.

    The design pins the import as `from recall_guard.mia.mcs import train`
    (the top-level name is the alias `train_mcs`).
    """
    from recall_guard.mia.mcs import train

    assert callable(train)

    import recall_guard

    assert recall_guard.train_mcs is train


def test_macro_framework_host_package_imports() -> None:
    """The existing host package still imports unchanged (additive delivery)."""
    import macro_framework  # noqa: F401


def test_feature_path_does_not_use_directional_facade() -> None:
    """The number-native scoring path bypasses the MemoryGuardedScorer facade (6.5).

    The public primitives above are sufficient to score a factor prompt, so the
    feature's code path must not import the directional facade
    (`recall_guard.harness.scorer`). The feature module does not exist yet
    (task 2.1 creates `macro_framework/factor_scoring.py`); this test asserts the
    foundation: the primitives are present and callable, while no new feature
    module has pulled the facade into `sys.modules` via the feature path.
    """
    from recall_guard import compute_mia_features

    # The number-native primitive is callable on its own: no facade needed to
    # turn a response + logprobs into MIA features.
    assert callable(compute_mia_features)

    # The directional facade lives in `recall_guard.harness.scorer`. The new
    # feature module is not created until task 2.1, so it cannot be imported here.
    assert "macro_framework.factor_scoring" not in sys.modules
    facade_spec = importlib.util.find_spec("recall_guard.harness.scorer")
    assert facade_spec is not None, "facade module exists in the library (we just do not depend on it)"


# --------------------------------------------------------------------------- #
# Task 2.1 — Regime-loadings prompt renderer (Requirements 1.4, 2.1, 2.2,     #
# 2.3, 2.5, 7.1, 7.6)                                                          #
# --------------------------------------------------------------------------- #

import re

import pandas as pd
import pytest

# Tokens that would reveal the period or the real assets to the model; the
# anonymized (PIT) form must contain NONE of these (R1.4, R2.3).
_REAL_TICKERS = ("SWDA", "XLK", "IAU", "BIL", "SWDA.L")
_DIRECTION_TOKENS = ("buy", "sell", "direction", "expected return", "forecast")


def _macro_state() -> dict[str, float]:
    # z-scored macro state (PIT, recall-disabled framing).
    return {"cpi_yoy_z": 1.42, "t10y2y_z": -0.83, "hy_oas_z": 0.57}


def _raw_levels() -> dict[str, float]:
    # raw non-normalized macro levels (recall-enabling addition for identifying).
    return {"cpi_yoy": 0.089, "t10y2y": -0.41, "hy_oas": 4.62}


def _asset_snapshot() -> list[dict[str, object]]:
    # anonymized asset descriptors: letter id + category only, no ticker.
    return [
        {"id": "Asset_A", "category": "world_equity", "trailing_12m_return": 0.18,
         "trailing_vol_ann": 0.14},
        {"id": "Asset_B", "category": "tech_sector", "trailing_12m_return": 0.31,
         "trailing_vol_ann": 0.22},
        {"id": "Asset_C", "category": "gold_commodity", "trailing_12m_return": 0.06,
         "trailing_vol_ann": 0.11},
        {"id": "Asset_D", "category": "short_treasury_cash", "trailing_12m_return": 0.02,
         "trailing_vol_ann": 0.01},
    ]


def test_macro_axes_constant() -> None:
    """The five named macro axes are exposed as the locked MACRO_AXES tuple."""
    from macro_framework.factor_scoring import MACRO_AXES

    assert MACRO_AXES == ("inflation", "growth", "credit_stress", "policy", "risk_appetite")


def test_anonymized_form_has_no_date_or_ticker() -> None:
    """Anonymized (PIT, default) form leaks no calendar date/year and no real ticker (1.4, 2.3)."""
    from macro_framework.factor_scoring import render_regime_loadings_prompt

    prompt = render_regime_loadings_prompt(_macro_state(), _asset_snapshot())

    # No 4-digit year and no ISO date.
    assert re.search(r"\b\d{4}\b", prompt) is None, "anonymized prompt must not contain a 4-digit year"
    assert re.search(r"\b\d{4}-\d{2}-\d{2}\b", prompt) is None, "anonymized prompt must not contain an ISO date"

    # No real tickers.
    for ticker in _REAL_TICKERS:
        assert ticker not in prompt, f"anonymized prompt leaked real ticker {ticker!r}"


def test_anonymized_asks_for_loadings_on_all_axes_no_direction() -> None:
    """Prompt requests [-1, 1] loadings on all five axes; never a buy/sell/return ask (2.1, 2.2, 2.5)."""
    from macro_framework.factor_scoring import MACRO_AXES, render_regime_loadings_prompt

    prompt = render_regime_loadings_prompt(_macro_state(), _asset_snapshot())
    lowered = prompt.lower()

    # All five named axes are present.
    for axis in MACRO_AXES:
        assert axis in prompt, f"prompt must name the macro axis {axis!r}"

    # The bounded [-1, 1] range is requested.
    assert ("-1" in prompt and "+1" in prompt) or "[-1, 1]" in prompt or "[-1,1]" in prompt

    # No directional / forecast ask.
    for token in _DIRECTION_TOKENS:
        assert token not in lowered, f"prompt must not ask for {token!r}"


def test_identifying_adds_tokens_and_otherwise_matches_anonymized() -> None:
    """Identifying form adds tickers + as_of + raw levels and is otherwise the same template (7.6)."""
    from macro_framework.factor_scoring import render_regime_loadings_prompt

    macro = _macro_state()
    assets = _asset_snapshot()
    as_of = pd.Timestamp("2022-06-30")
    raw = _raw_levels()

    anon = render_regime_loadings_prompt(macro, assets)
    ident = render_regime_loadings_prompt(
        macro, assets, identifying=True, as_of=as_of, raw_levels=raw
    )

    # The identifying form differs from the anonymized form.
    assert ident != anon

    # Exactly the recall-enabling additions appear in the identifying form.
    assert "2022-06-30" in ident
    for ticker in _REAL_TICKERS:
        if ticker == "SWDA.L":
            continue
        assert ticker in ident, f"identifying prompt must reveal real ticker {ticker!r}"
    # Raw levels are surfaced.
    assert any(str(v) in ident or f"{v:g}" in ident for v in raw.values())

    # Token-identical except the additions: every non-empty line of the anonymized
    # form must still appear verbatim in the identifying form (the identifying form
    # only ADDS the identity/date/raw-level lines, R7.6).
    for line in anon.splitlines():
        if line.strip():
            assert line in ident, f"identifying form dropped/altered anonymized line: {line!r}"


def test_renderer_is_deterministic() -> None:
    """Equal inputs produce an identical string (deterministic)."""
    from macro_framework.factor_scoring import render_regime_loadings_prompt

    macro = _macro_state()
    assets = _asset_snapshot()

    a1 = render_regime_loadings_prompt(macro, assets)
    a2 = render_regime_loadings_prompt(macro, assets)
    assert a1 == a2

    as_of = pd.Timestamp("2022-06-30")
    raw = _raw_levels()
    i1 = render_regime_loadings_prompt(macro, assets, identifying=True, as_of=as_of, raw_levels=raw)
    i2 = render_regime_loadings_prompt(macro, assets, identifying=True, as_of=as_of, raw_levels=raw)
    assert i1 == i2


def test_identifying_requires_as_of_and_raw_levels() -> None:
    """identifying=True without as_of/raw_levels raises a clear error (7.6 preconditions)."""
    from macro_framework.factor_scoring import render_regime_loadings_prompt

    macro = _macro_state()
    assets = _asset_snapshot()

    with pytest.raises(ValueError):
        render_regime_loadings_prompt(macro, assets, identifying=True)

    with pytest.raises(ValueError):
        render_regime_loadings_prompt(
            macro, assets, identifying=True, as_of=pd.Timestamp("2022-06-30")
        )

    with pytest.raises(ValueError):
        render_regime_loadings_prompt(macro, assets, identifying=True, raw_levels=_raw_levels())


# --------------------------------------------------------------------------- #
# Task 2.2 — Loadings parser + RegimeLoadings (Requirements 2.1, 2.4)         #
# --------------------------------------------------------------------------- #


def _well_formed_reply() -> str:
    """A clean JSON-object reply naming every MACRO_AXES axis (one in-range value
    deliberately exceeds +1 so the [-1, +1] clip is exercised, R2.1)."""
    return (
        "Here is the regime characterization:\n"
        '{"inflation": 0.6, "growth": -0.2, "credit_stress": 1.4, '
        '"policy": -0.1, "risk_appetite": -0.3}\n'
        "These are loadings, not trades."
    )


def test_parse_loadings_well_formed_all_axes_clipped() -> None:
    """A well-formed reply parses into all five axes, each clipped to [-1, 1] (2.1)."""
    from macro_framework.factor_scoring import (
        MACRO_AXES,
        RegimeLoadings,
        parse_loadings,
    )

    rb = pd.Timestamp("2022-06-30")
    result = parse_loadings(_well_formed_reply(), rb)

    assert isinstance(result, RegimeLoadings)
    assert result.parse_ok is True
    assert result.rebalance_date == rb

    # One loading per named axis, no extras, no missing.
    assert set(result.loadings) == set(MACRO_AXES)

    # Every loading is inside the closed [-1, +1] interval.
    for axis, value in result.loadings.items():
        assert -1.0 <= value <= 1.0, f"{axis}={value} not clipped to [-1, 1]"

    # The deliberately out-of-range value (1.4) was clipped to the +1 bound.
    assert result.loadings["credit_stress"] == 1.0
    # In-range values are preserved (not mangled by the clip).
    assert result.loadings["inflation"] == pytest.approx(0.6)
    assert result.loadings["growth"] == pytest.approx(-0.2)


def test_parse_loadings_tolerant_of_labeled_list_form() -> None:
    """A reasonable non-JSON labeled-list reply also parses into all five axes (2.1, tolerant)."""
    from macro_framework.factor_scoring import MACRO_AXES, parse_loadings

    rb = pd.Timestamp("2021-03-31")
    text = (
        "inflation: 0.6\n"
        "growth: -0.2\n"
        "credit_stress: 0.5\n"
        "policy: -2.0\n"  # out of range, must clip to -1
        "risk_appetite: 0.3\n"
    )
    result = parse_loadings(text, rb)

    assert result is not None
    assert result.parse_ok is True
    assert set(result.loadings) == set(MACRO_AXES)
    assert result.loadings["policy"] == -1.0  # clipped to the lower bound
    assert -1.0 <= result.loadings["risk_appetite"] <= 1.0


def test_parse_loadings_partial_reply_not_parsed_no_fabrication() -> None:
    """A reply missing axes yields the not-parsed result, NOT fabricated zeros (2.4)."""
    from macro_framework.factor_scoring import parse_loadings

    rb = pd.Timestamp("2022-06-30")
    # Only two of the five axes present.
    text = '{"inflation": 0.6, "growth": -0.2}'
    result = parse_loadings(text, rb)

    # Not-parsed: either None or a RegimeLoadings with parse_ok=False and no
    # fabricated full five-axis vector.
    if result is None:
        return
    assert result.parse_ok is False
    # Must NOT have fabricated the three missing axes (no zero-filled vector).
    assert "credit_stress" not in result.loadings
    assert "policy" not in result.loadings
    assert "risk_appetite" not in result.loadings


def test_parse_loadings_garbage_reply_not_parsed() -> None:
    """A garbage reply with no extractable loadings yields the not-parsed result (2.4)."""
    from macro_framework.factor_scoring import parse_loadings

    rb = pd.Timestamp("2022-06-30")
    result = parse_loadings("I cannot answer that question.", rb)

    if result is None:
        return
    assert result.parse_ok is False
    # No fabricated values.
    assert result.loadings == {} or all(
        axis not in result.loadings
        for axis in ("inflation", "growth", "credit_stress", "policy", "risk_appetite")
    )


def test_regime_loadings_is_frozen_and_keyed_by_date() -> None:
    """RegimeLoadings is a frozen dataclass keyed by rebalance_date (2.4)."""
    import dataclasses

    from macro_framework.factor_scoring import RegimeLoadings, parse_loadings

    assert dataclasses.is_dataclass(RegimeLoadings)

    rb = pd.Timestamp("2022-06-30")
    result = parse_loadings(_well_formed_reply(), rb)
    assert isinstance(result, RegimeLoadings)

    # Frozen: attribute assignment is rejected.
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.rebalance_date = pd.Timestamp("2099-01-01")  # type: ignore[misc]

    # Keyed by rebalance_date: the field round-trips the key it was built with,
    # so a per-date artifact (R2.4) can index on it.
    assert result.rebalance_date == rb
    by_date = {result.rebalance_date: result}
    assert by_date[rb] is result


def test_parse_loadings_is_deterministic() -> None:
    """Equal inputs produce identical parsed loadings (pure/deterministic)."""
    from macro_framework.factor_scoring import parse_loadings

    rb = pd.Timestamp("2022-06-30")
    r1 = parse_loadings(_well_formed_reply(), rb)
    r2 = parse_loadings(_well_formed_reply(), rb)

    assert r1 is not None and r2 is not None
    assert r1.loadings == r2.loadings
    assert r1.parse_ok == r2.parse_ok
    assert r1.rebalance_date == r2.rebalance_date


# --------------------------------------------------------------------------- #
# Task 2.3 — FactorScorer: number-native calibration + configuration errors    #
# (Requirements 1.1, 1.5, 1.6, 6.5)                                            #
#                                                                              #
# These tests MOCK the recall_guard calls AS BOUND IN macro_framework.         #
# factor_scoring (build_baseline / train / NvidiaLM) so nothing hits NIM/FMP.  #
# They cover: the number-native corpus construction (identifying IS vs          #
# anonymized OOS on the SAME factor task), the ref_lm=None contract, the        #
# is_weak / holdout_auc passthrough, and the module's OWN ConfigurationError.   #
# --------------------------------------------------------------------------- #

from datetime import date  # noqa: E402

from macro_framework.anonymize import AssetMap  # noqa: E402


def _synthetic_panel() -> pd.DataFrame:
    """A small synthetic macro panel with raw + z columns (R1 calibration input).

    Mirrors the real ``data/macro_panel_monthly.parquet`` shape: a DatetimeIndex
    of month-ends and the raw (``cpi_yoy``, ``t10y2y``, ``hy_oas``) + z-scored
    (``*_z``) columns the renderer reads. All rows are pre-2024-08-01 so the
    ``< cutoff_date`` slice keeps them.
    """
    idx = pd.date_range("2015-01-31", periods=24, freq="ME")
    n = len(idx)
    return pd.DataFrame(
        {
            "cpi_yoy": [0.02 + 0.001 * i for i in range(n)],
            "t10y2y": [0.5 - 0.02 * i for i in range(n)],
            "hy_oas": [3.0 + 0.05 * i for i in range(n)],
            "cpi_yoy_z": [(-1.0 + 0.08 * i) for i in range(n)],
            "t10y2y_z": [(1.0 - 0.07 * i) for i in range(n)],
            "hy_oas_z": [(-0.5 + 0.04 * i) for i in range(n)],
        },
        index=idx,
    )


class _FakeBaseline:
    """A stand-in ControlBaseline carrying only the attributes calibrate reads."""

    def __init__(self, n_valid: int) -> None:
        self.n_valid = n_valid
        self.model = "fake-model"
        self.is_calibrated = n_valid > 0
        self.min_valid = 1


class _FakeCalibrator:
    """A stand-in MCSCalibrator exposing the holdout_auc / is_weak surface."""

    def __init__(self, holdout_auc: float, is_weak: bool) -> None:
        self.holdout_auc = holdout_auc
        self.is_weak = is_weak
        self.model = "fake-model"


def _patch_recall_guard(
    monkeypatch: pytest.MonkeyPatch,
    *,
    baseline_n_valid: int = 50,
    holdout_auc: float = 0.96,
    is_weak: bool = False,
) -> dict[str, object]:
    """Patch build_baseline / train / NvidiaLM as bound in factor_scoring.

    Returns a dict capturing the calls so assertions can inspect what the
    corpus builder fed to the primitives (the IS/OOS prompts and the ref_lm
    kwarg). Nothing here issues a network call.
    """
    from macro_framework import factor_scoring as fs

    captured: dict[str, object] = {}

    class _FakeLM:
        def __init__(self, api_key: str, model: str, **kwargs: object) -> None:
            # Mirror NvidiaLM's own guard so an empty key would ValueError here
            # IF the module ever forgot its own ConfigurationError guard.
            if not api_key:
                raise ValueError("api_key must be a non-empty string")
            self.api_key = api_key
            self.model = model
            captured["lm"] = self

    def _fake_build_baseline(lm, oos_rows, ref_lm, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured["build_baseline_oos_rows"] = list(oos_rows)
        captured["build_baseline_ref_lm"] = ref_lm
        return _FakeBaseline(baseline_n_valid)

    def _fake_train(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured["train_args"] = args
        captured["train_kwargs"] = kwargs
        return _FakeCalibrator(holdout_auc=holdout_auc, is_weak=is_weak)

    monkeypatch.setattr(fs, "NvidiaLM", _FakeLM)
    monkeypatch.setattr(fs, "build_baseline", _fake_build_baseline)
    monkeypatch.setattr(fs, "train", _fake_train)
    return captured


def _prompt_of(row: object) -> str:
    """Extract the prompt string from an EvalRow (or a raw string)."""
    return getattr(row, "prompt", row)  # type: ignore[return-value]


def test_calibrate_builds_identifying_is_and_anonymized_oos_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """calibrate builds an identifying IS corpus and an anonymized OOS corpus (R1.1).

    The number-native split is framing-only on the SAME factor task: the IS
    prompts carry the recall-enabling tokens (real tickers + an ISO date), the
    OOS prompts carry none of them. Both come from render_regime_loadings_prompt.
    """
    from macro_framework import factor_scoring as fs

    captured = _patch_recall_guard(monkeypatch)

    scorer = fs.FactorScorer.calibrate(
        nim_model="meta/llama-4-maverick-17b-128e-instruct",
        cutoff_date=date(2024, 8, 1),
        macro_panel=_synthetic_panel(),
        asset_map=AssetMap.default(),
        api_key="test-key",
        n_per_class=5,
        max_workers=2,
    )
    assert isinstance(scorer, fs.FactorScorer)

    # OOS corpus reached build_baseline; IS corpus reached train (label-1 arg).
    oos_rows = captured["build_baseline_oos_rows"]
    assert oos_rows, "OOS corpus must be non-empty"
    oos_prompts = [_prompt_of(r) for r in oos_rows]  # type: ignore[union-attr]

    # The IS corpus is the train call's is_memorized argument. train was called
    # with keyword args per the design's pinned signature.
    train_kwargs = captured["train_kwargs"]
    is_rows = train_kwargs["is_memorized"]
    assert is_rows, "IS corpus must be non-empty"
    is_prompts = [_prompt_of(r) for r in is_rows]

    # SAME factor task: every prompt asks for the regime loadings on the axes.
    for p in oos_prompts + is_prompts:
        assert "loading" in p.lower()
        for axis in fs.MACRO_AXES:
            assert axis in p

    # Anonymized OOS: NO real ticker, NO ISO date (recall-disabled framing).
    for p in oos_prompts:
        for ticker in _REAL_TICKERS:
            assert ticker not in p, f"OOS leaked real ticker {ticker!r}"
        assert re.search(r"\b\d{4}-\d{2}-\d{2}\b", p) is None

    # Identifying IS: contains the recall-enabling tokens (real ticker + date).
    assert any("SWDA.L" in p for p in is_prompts), "IS must reveal a real ticker"
    assert any(re.search(r"\b\d{4}-\d{2}-\d{2}\b", p) for p in is_prompts), (
        "IS must reveal an as-of date"
    )


def test_calibrate_passes_ref_lm_none_to_train_and_build_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ref_lm=None on both build_baseline and train (ref_delta inert, locked)."""
    from macro_framework import factor_scoring as fs

    captured = _patch_recall_guard(monkeypatch)

    fs.FactorScorer.calibrate(
        nim_model="meta/llama-4-maverick-17b-128e-instruct",
        cutoff_date=date(2024, 8, 1),
        macro_panel=_synthetic_panel(),
        asset_map=AssetMap.default(),
        api_key="test-key",
        n_per_class=5,
    )

    assert captured["build_baseline_ref_lm"] is None
    assert captured["train_kwargs"]["ref_lm"] is None


def test_calibrate_exposes_holdout_auc_and_is_weak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The returned scorer delegates holdout_auc / is_weak to the calibrator (R1.6)."""
    from macro_framework import factor_scoring as fs

    _patch_recall_guard(monkeypatch, holdout_auc=0.42, is_weak=True)

    scorer = fs.FactorScorer.calibrate(
        nim_model="meta/llama-4-maverick-17b-128e-instruct",
        cutoff_date=date(2024, 8, 1),
        macro_panel=_synthetic_panel(),
        asset_map=AssetMap.default(),
        api_key="test-key",
        n_per_class=5,
    )

    assert scorer.holdout_auc == 0.42
    assert scorer.is_weak is True


def test_calibrate_empty_api_key_raises_module_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty api_key raises factor_scoring.ConfigurationError BEFORE NvidiaLM (R1.5).

    The module defines its OWN ConfigurationError (the bypassed recall_guard
    facade is the only place recall_guard raises one). The guard fires before
    NvidiaLM construction, which would otherwise raise a bare ValueError.
    """
    from macro_framework import factor_scoring as fs

    _patch_recall_guard(monkeypatch)

    assert issubclass(fs.ConfigurationError, RuntimeError)

    with pytest.raises(fs.ConfigurationError):
        fs.FactorScorer.calibrate(
            nim_model="meta/llama-4-maverick-17b-128e-instruct",
            cutoff_date=date(2024, 8, 1),
            macro_panel=_synthetic_panel(),
            asset_map=AssetMap.default(),
            api_key="",
            n_per_class=5,
        )


def test_calibrate_zero_valid_baseline_raises_module_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A baseline with n_valid == 0 raises factor_scoring.ConfigurationError (R1.5)."""
    from macro_framework import factor_scoring as fs

    _patch_recall_guard(monkeypatch, baseline_n_valid=0)

    with pytest.raises(fs.ConfigurationError):
        fs.FactorScorer.calibrate(
            nim_model="meta/llama-4-maverick-17b-128e-instruct",
            cutoff_date=date(2024, 8, 1),
            macro_panel=_synthetic_panel(),
            asset_map=AssetMap.default(),
            api_key="test-key",
            n_per_class=5,
        )


def test_factor_score_and_calibration_stats_dataclass_shapes() -> None:
    """FactorScore + CalibrationStats are frozen dataclasses with the spec fields."""
    import dataclasses

    from macro_framework.factor_scoring import CalibrationStats, FactorScore

    assert dataclasses.is_dataclass(FactorScore)
    assert dataclasses.is_dataclass(CalibrationStats)

    fscore = FactorScore(p_memorized=0.3, parse_ok=True, fail_reason=None)
    assert {f.name for f in dataclasses.fields(FactorScore)} == {
        "p_memorized",
        "parse_ok",
        "fail_reason",
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        fscore.p_memorized = 0.9  # type: ignore[misc]

    stats = CalibrationStats(holdout_auc=0.96, is_weak=False, n_is=60, n_oos=60)
    assert {f.name for f in dataclasses.fields(CalibrationStats)} == {
        "holdout_auc",
        "is_weak",
        "n_is",
        "n_oos",
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.holdout_auc = 0.1  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Task 2.4 — FactorScorer: number-native scoring path                          #
# (Requirements 1.1, 1.2, 1.3, 1.5, 6.5)                                       #
#                                                                              #
# These tests build a FactorScorer around a FAKE NvidiaLM + a fake calibrator  #
# + a fake baseline (no NIM, no FMP). The score path must be number-native:    #
#   generate -> compute_mia_features(content, logprobs, None)                  #
#            -> calibrator.predict_proba(features, baseline) -> FactorScore.    #
# It NEVER parses a buy/sell direction or reads a directional signal (R1.3),    #
# distinct prompts can yield distinct p_memorized (R1.2), an auth-class         #
# RuntimeError from generate raises the module's OWN ConfigurationError (R1.5), #
# and other failures degrade to FactorScore(p_memorized=None, parse_ok=False). #
# --------------------------------------------------------------------------- #


class _FakeTokenLogprob:
    """A minimal per-token logprob stand-in (only the attrs the path reads)."""

    def __init__(self, token: str, logprob: float) -> None:
        self.token = token
        self.logprob = logprob
        self.top_logprobs = [{"logprob": logprob}]


class _FakeCompletion:
    """A stand-in CompletionResult carrying content + logprobs."""

    def __init__(self, content: str, logprobs: list[object]) -> None:
        self.content = content
        self.logprobs = logprobs
        self.raw_temperature_observed = 0.0


def _logprobs_for(seed: float) -> list[object]:
    """Build a small distinct logprob list driven by ``seed`` (R1.2 driver)."""
    return [_FakeTokenLogprob(f"t{i}", -(0.1 + seed) * (i + 1)) for i in range(4)]


class _ScoreFakeLM:
    """Fake LM whose ``generate`` returns distinct logprobs per prompt.

    Distinct prompts -> distinct logprobs -> distinct features -> distinct
    p_memorized (the version-aware property, R1.2). A prompt may be configured
    to raise (auth/timeout/non-auth/empty-logprobs) to exercise the fail paths.
    """

    def __init__(self, *, behavior: dict[str, object] | None = None) -> None:
        self.model = "fake-model"
        self.api_key = "test-key"
        self._behavior = behavior or {}
        self.calls: list[str] = []

    def generate(self, prompt: str, *args: object, **kwargs: object) -> object:
        self.calls.append(prompt)
        action = self._behavior.get(prompt)
        if isinstance(action, BaseException):
            raise action
        if action == "empty_logprobs":
            return _FakeCompletion(content="reply", logprobs=[])
        # default: distinct logprobs derived from the prompt length (stable + distinct)
        return _FakeCompletion(
            content=f"reply::{prompt}", logprobs=_logprobs_for(len(prompt) * 0.01)
        )


class _ScoreFakeBaseline:
    """A stand-in ControlBaseline; only identity matters (passed through)."""

    def __init__(self) -> None:
        self.n_valid = 10
        self.is_calibrated = True
        self.model = "fake-model"


class _ScoreFakeCalibrator:
    """A fake MCSCalibrator whose predict_proba maps features -> a known p.

    Records every (features, baseline) pair so a test can assert the score
    path fed it the MIA features (NOT a parsed direction). The probability is
    derived from the features' ``loss`` so distinct features -> distinct p.
    """

    def __init__(self, *, is_weak: bool = False, holdout_auc: float = 0.96) -> None:
        self.is_weak = is_weak
        self.holdout_auc = holdout_auc
        self.model = "fake-model"
        self.calls: list[tuple[object, object]] = []

    def predict_proba(self, features: object, baseline: object) -> float:
        self.calls.append((features, baseline))
        # Map the MIA loss feature into (0, 1) deterministically; distinct
        # features (from distinct logprobs) therefore yield distinct p.
        loss = float(getattr(features, "loss"))
        return 1.0 / (1.0 + pow(2.718281828, -loss))


def _make_scorer(
    *,
    lm: object | None = None,
    calibrator: object | None = None,
    is_weak: bool = False,
):
    """Construct a FactorScorer directly around fakes (no calibrate / no NIM)."""
    from macro_framework import factor_scoring as fs

    return fs.FactorScorer(
        calibrator=calibrator or _ScoreFakeCalibrator(is_weak=is_weak),
        baseline=_ScoreFakeBaseline(),
        lm=lm or _ScoreFakeLM(),
        stats=fs.CalibrationStats(
            holdout_auc=0.96, is_weak=is_weak, n_is=5, n_oos=5
        ),
    )


def test_score_uses_feature_path_not_direction_parse() -> None:
    """score computes p_memorized via compute_mia_features -> predict_proba (R1.1, R1.3).

    The result must come from the MIA feature path, never from a buy/sell
    direction parse. We assert both primitives ran (the calibrator recorded a
    MiaFeatures object) and that the returned p matches predict_proba's output.
    """
    from recall_guard.mia.features import MiaFeatures

    calibrator = _ScoreFakeCalibrator()
    scorer = _make_scorer(calibrator=calibrator)

    result = scorer.score("Characterize the regime as loadings.")

    assert result.parse_ok is True
    assert result.fail_reason is None
    assert result.p_memorized is not None
    assert 0.0 <= result.p_memorized <= 1.0

    # predict_proba ran exactly once and was fed real MiaFeatures (the feature
    # path), not a parsed direction/confidence.
    assert len(calibrator.calls) == 1
    features, _baseline = calibrator.calls[0]
    assert isinstance(features, MiaFeatures)


def test_score_passes_none_ref_logprobs_to_compute_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The score path calls compute_mia_features(content, logprobs, None) (R1.1).

    No reference run on the score path: ref_logprobs is fixed at None (mirrors
    the ref_lm=None calibration contract). We patch compute_mia_features as
    bound in the module and assert the third positional arg is None.
    """
    from macro_framework import factor_scoring as fs

    captured: dict[str, object] = {}
    real = fs.compute_mia_features

    def _spy(content, logprobs, ref_logprobs, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured["content"] = content
        captured["logprobs"] = logprobs
        captured["ref_logprobs"] = ref_logprobs
        return real(content, logprobs, ref_logprobs, *args, **kwargs)

    monkeypatch.setattr(fs, "compute_mia_features", _spy)

    scorer = _make_scorer()
    result = scorer.score("a prompt")

    assert result.p_memorized is not None
    assert captured["ref_logprobs"] is None
    assert captured["content"] == "reply::a prompt"
    assert captured["logprobs"], "the model's logprobs must reach compute_mia_features"


def test_score_distinct_prompts_yield_distinct_p_memorized() -> None:
    """Distinct prompts -> distinct logprobs -> distinct p_memorized (R1.2).

    This is the version-aware property: two different prompt versions over the
    same state can score differently, driven by the model's distinct logprobs.
    """
    scorer = _make_scorer()

    a = scorer.score("version one prompt is short")
    b = scorer.score("version two prompt is considerably longer than the first one")

    assert a.p_memorized is not None
    assert b.p_memorized is not None
    assert a.p_memorized != b.p_memorized


def test_score_never_reads_a_directional_signal() -> None:
    """The FactorScore carries no direction/confidence/signal field (R1.3, R6.5).

    The number-native path bypasses the directional facade entirely; its result
    is a contamination score only — no buy/sell signal is ever read or returned.
    """
    import dataclasses

    from macro_framework.factor_scoring import FactorScore

    scorer = _make_scorer()
    result = scorer.score("any prompt")

    field_names = {f.name for f in dataclasses.fields(FactorScore)}
    assert "signal" not in field_names
    assert "direction" not in field_names
    assert "raw_confidence" not in field_names
    assert isinstance(result, FactorScore)


def test_score_auth_runtime_error_raises_module_configuration_error() -> None:
    """An auth-class RuntimeError from generate -> factor_scoring.ConfigurationError (R1.5).

    HTTP 401/403/unauthorized/forbidden markers in the RuntimeError message are
    a rejected credential; the module surfaces its OWN ConfigurationError rather
    than a silent failure record.
    """
    from macro_framework import factor_scoring as fs

    for msg in (
        "Model X request failed: 401 Client Error: Unauthorized",
        "Model X request failed: 403 Client Error: Forbidden",
        "authentication failed for the api key",
        "invalid api key supplied",
    ):
        lm = _ScoreFakeLM(behavior={"p": RuntimeError(msg)})
        scorer = _make_scorer(lm=lm)
        with pytest.raises(fs.ConfigurationError):
            scorer.score("p")


def test_score_non_auth_runtime_error_returns_unscored() -> None:
    """A non-auth RuntimeError degrades to p_memorized=None (R1.5, graceful)."""
    lm = _ScoreFakeLM(
        behavior={"p": RuntimeError("Model X request failed after 3 attempt(s): 500")}
    )
    scorer = _make_scorer(lm=lm)

    result = scorer.score("p")

    assert result.p_memorized is None
    assert result.parse_ok is False
    assert result.fail_reason is not None


def test_score_timeout_returns_unscored() -> None:
    """A TimeoutError degrades to p_memorized=None, never crashes (R1.5, graceful)."""
    lm = _ScoreFakeLM(behavior={"p": TimeoutError("Model X timed out")})
    scorer = _make_scorer(lm=lm)

    result = scorer.score("p")

    assert result.p_memorized is None
    assert result.parse_ok is False
    assert result.fail_reason is not None


def test_score_empty_logprobs_returns_unscored() -> None:
    """Empty logprobs (no per-token data) degrade to p_memorized=None (R1.5)."""
    lm = _ScoreFakeLM(behavior={"p": "empty_logprobs"})
    scorer = _make_scorer(lm=lm)

    result = scorer.score("p")

    assert result.p_memorized is None
    assert result.parse_ok is False
    assert result.fail_reason is not None


def test_score_feature_failure_returns_unscored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A compute_mia_features failure degrades to p_memorized=None (R1.5)."""
    from macro_framework import factor_scoring as fs

    def _boom(*args: object, **kwargs: object) -> object:
        raise ValueError("bad logprobs")

    monkeypatch.setattr(fs, "compute_mia_features", _boom)

    scorer = _make_scorer()
    result = scorer.score("p")

    assert result.p_memorized is None
    assert result.parse_ok is False
    assert result.fail_reason is not None


def test_score_predict_failure_returns_unscored() -> None:
    """A predict_proba failure degrades to p_memorized=None (R1.5)."""

    class _BoomCalibrator(_ScoreFakeCalibrator):
        def predict_proba(self, features: object, baseline: object) -> float:
            raise ValueError("standardisation produced None")

    scorer = _make_scorer(calibrator=_BoomCalibrator())
    result = scorer.score("p")

    assert result.p_memorized is None
    assert result.parse_ok is False
    assert result.fail_reason is not None


def test_score_many_preserves_order_one_result_per_prompt() -> None:
    """score_many is order-preserving with one FactorScore per prompt (R1.1)."""
    scorer = _make_scorer()
    prompts = [
        "alpha prompt",
        "a much longer beta prompt with more tokens than alpha",
        "g",
    ]

    results = scorer.score_many(prompts, max_workers=2)

    assert len(results) == len(prompts)
    # Each entry is a FactorScore with a usable p_memorized for these good prompts.
    for r in results:
        assert r.p_memorized is not None
        assert r.parse_ok is True

    # Order-preserving + version-aware: distinct prompts -> distinct scores, and
    # scoring the same prompts singly yields the SAME per-prompt p_memorized.
    singles = [scorer.score(p).p_memorized for p in prompts]
    assert [r.p_memorized for r in results] == singles


def test_score_many_mixed_success_and_failure_is_per_prompt() -> None:
    """score_many degrades per prompt: good ones score, bad ones return None (R1.5)."""
    lm = _ScoreFakeLM(
        behavior={"bad": RuntimeError("500 server error after 3 attempt(s)")}
    )
    scorer = _make_scorer(lm=lm)
    prompts = ["good one", "bad", "another good one"]

    results = scorer.score_many(prompts)

    assert len(results) == 3
    assert results[0].p_memorized is not None and results[0].parse_ok is True
    assert results[1].p_memorized is None and results[1].parse_ok is False
    assert results[2].p_memorized is not None and results[2].parse_ok is True


def test_score_many_auth_error_raises_module_configuration_error() -> None:
    """An auth-class RuntimeError in a batch raises ConfigurationError (R1.5)."""
    from macro_framework import factor_scoring as fs

    lm = _ScoreFakeLM(behavior={"bad": RuntimeError("401 Unauthorized")})
    scorer = _make_scorer(lm=lm)

    with pytest.raises(fs.ConfigurationError):
        scorer.score_many(["good", "bad"])


# --------------------------------------------------------------------------- #
# Task 2.5 — FactorScorer persistence (save / load) (Requirement 1.6)          #
#                                                                             #
# A trained scorer is persisted so the one-time number-native calibration      #
# (~135 NIM calls) is reused across notebooks. save writes a directory: the     #
# pickled MCSCalibrator (carrying the LogisticRegression + feature_order) via   #
# joblib, the ControlBaseline standardisation stats as JSON, and the           #
# CalibrationStats. The api_key / NvidiaLM are NEVER persisted; load           #
# re-attaches a fresh NvidiaLM(api_key, model). A loaded scorer must produce    #
# byte-identical predict_proba / scores for the same inputs (round-trip).       #
# These tests build a REAL MCSCalibrator + ControlBaseline from synthetic       #
# feature data (no network) so the round-trip fidelity is verified directly.    #
# --------------------------------------------------------------------------- #


from recall_guard import NvidiaLM  # noqa: E402
from recall_guard.mia.control import ControlBaseline as _RealControlBaseline  # noqa: E402
from recall_guard.mia.features import MiaFeatures as _RealMiaFeatures  # noqa: E402
from recall_guard.mia.mcs import MCSCalibrator as _RealMCSCalibrator  # noqa: E402

_PERSIST_MODEL = "meta/llama-4-maverick-17b-128e-instruct"
_PERSIST_DUMMY_KEY = "nvapi-DUMMY-SECRET-do-not-persist-1234567890"


def _real_baseline() -> _RealControlBaseline:
    """A real ControlBaseline with synthetic per-feature means/stds (no network).

    Mirrors what build_baseline produces with ref_lm=None: the four core
    features are populated and ref_delta is None (so feature_order excludes it,
    matching the calibration contract).
    """
    return _RealControlBaseline(
        model=_PERSIST_MODEL,
        n_valid=20,
        feature_means={
            "loss": 1.5,
            "min_k": -2.0,
            "min_k_pp": 0.25,
            "zlib_ratio": 0.8,
            "ref_delta": None,
        },
        feature_stds={
            "loss": 0.5,
            "min_k": 0.75,
            "min_k_pp": 0.1,
            "zlib_ratio": 0.2,
            "ref_delta": None,
        },
        is_calibrated=True,
        min_valid=2,
    )


def _real_calibrator() -> _RealMCSCalibrator:
    """A real MCSCalibrator: a tiny fitted LogisticRegression + feature_order.

    Fits sklearn's LogisticRegression on synthetic standardised vectors (the
    four core features, ref_delta excluded — the ref_lm=None contract) so the
    classifier round-trips through joblib exactly like the calibrated one.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    feature_order = ["loss", "min_k", "min_k_pp", "zlib_ratio"]
    rng = np.random.default_rng(0)
    x_oos = rng.normal(0.0, 1.0, size=(12, 4))
    x_is = rng.normal(1.5, 1.0, size=(12, 4))
    x = np.vstack([x_oos, x_is])
    y = np.array([0] * 12 + [1] * 12)
    clf = LogisticRegression(
        class_weight="balanced", solver="liblinear", random_state=0
    )
    clf.fit(x, y)
    return _RealMCSCalibrator(
        model=_PERSIST_MODEL,
        classifier=clf,
        feature_order=feature_order,
        holdout_auc=0.93,
        is_weak=False,
    )


def _real_features() -> _RealMiaFeatures:
    """A fixed real MiaFeatures vector (deterministic; no network)."""
    return _RealMiaFeatures(
        loss=2.1, min_k=-1.4, min_k_pp=0.35, zlib_ratio=0.9, ref_delta=None
    )


def _real_scorer():
    """A FactorScorer holding REAL recall_guard components (offline)."""
    from macro_framework import factor_scoring as fs

    calibrator = _real_calibrator()
    baseline = _real_baseline()
    # NvidiaLM construction is offline (no network at init); a dummy key is fine.
    lm = NvidiaLM(api_key=_PERSIST_DUMMY_KEY, model=_PERSIST_MODEL)
    stats = fs.CalibrationStats(
        holdout_auc=0.93, is_weak=False, n_is=12, n_oos=12
    )
    return fs.FactorScorer(
        calibrator=calibrator, baseline=baseline, lm=lm, stats=stats
    )


def test_save_then_load_round_trips_predict_proba_identically(tmp_path) -> None:
    """save -> load yields a scorer with IDENTICAL predict_proba (R1.6, persistence).

    The loaded calibrator + baseline must reproduce the original's calibrated
    p_memorized bit-for-bit for the same fixed MiaFeatures — the round-trip
    fidelity guarantee. Deterministic, needs no network.
    """
    scorer = _real_scorer()
    features = _real_features()

    original_p = scorer._calibrator.predict_proba(features, scorer._baseline)

    out = tmp_path / "scorer_artifact"
    scorer.save(out)

    loaded = type(scorer).load(out, api_key=_PERSIST_DUMMY_KEY)
    loaded_p = loaded._calibrator.predict_proba(features, loaded._baseline)

    assert loaded_p == original_p


def test_loaded_baseline_reconstructs_real_control_baseline_fields(tmp_path) -> None:
    """load reconstructs a ControlBaseline with the saved standardisation fields."""
    scorer = _real_scorer()
    out = tmp_path / "scorer_artifact"
    scorer.save(out)

    loaded = type(scorer).load(out, api_key=_PERSIST_DUMMY_KEY)

    assert isinstance(loaded._baseline, _RealControlBaseline)
    assert loaded._baseline.model == scorer._baseline.model
    assert loaded._baseline.n_valid == scorer._baseline.n_valid
    assert loaded._baseline.is_calibrated == scorer._baseline.is_calibrated
    assert loaded._baseline.min_valid == scorer._baseline.min_valid
    assert loaded._baseline.feature_means == scorer._baseline.feature_means
    assert loaded._baseline.feature_stds == scorer._baseline.feature_stds


def test_persisted_artifact_contains_no_api_key(tmp_path) -> None:
    """The dummy api key is absent from EVERY persisted file (no secret on disk).

    R1.5/persistence invariant: the API key is never written; load re-attaches a
    fresh NvidiaLM(api_key, model). Grep every saved file's raw bytes.
    """
    scorer = _real_scorer()
    out = tmp_path / "scorer_artifact"
    scorer.save(out)

    key_bytes = _PERSIST_DUMMY_KEY.encode("utf-8")
    files = [p for p in out.rglob("*") if p.is_file()]
    assert files, "save wrote no files"
    for path in files:
        assert key_bytes not in path.read_bytes(), (
            f"api key leaked into persisted artifact {path}"
        )


def test_loaded_scorer_reports_same_is_weak_and_holdout_auc(tmp_path) -> None:
    """The reloaded scorer reports the same is_weak / holdout_auc (R1.6)."""
    scorer = _real_scorer()
    out = tmp_path / "scorer_artifact"
    scorer.save(out)

    loaded = type(scorer).load(out, api_key=_PERSIST_DUMMY_KEY)

    assert loaded.is_weak == scorer.is_weak
    assert loaded.holdout_auc == scorer.holdout_auc
    assert loaded.stats == scorer.stats


def test_load_reattaches_fresh_nvidia_lm_with_given_key_and_model(tmp_path) -> None:
    """load re-attaches a fresh NvidiaLM(api_key, model) — key supplied at load."""
    scorer = _real_scorer()
    out = tmp_path / "scorer_artifact"
    scorer.save(out)

    new_key = "nvapi-A-DIFFERENT-KEY-supplied-at-load"
    loaded = type(scorer).load(out, api_key=new_key)

    assert isinstance(loaded._lm, NvidiaLM)
    assert loaded._lm.api_key == new_key
    assert loaded._lm.model == _PERSIST_MODEL


def test_save_then_load_round_trips_scores_identically(tmp_path) -> None:
    """A loaded scorer produces identical FactorScore.p_memorized via score (R1.1).

    Drives both the original and loaded scorer through the public score path with
    a fake LM (offline) so the only varying component is the persisted
    calibrator + baseline; the scores must be identical.
    """
    scorer = _real_scorer()
    out = tmp_path / "scorer_artifact"
    scorer.save(out)
    loaded = type(scorer).load(out, api_key=_PERSIST_DUMMY_KEY)

    # Swap in a deterministic offline LM on BOTH so score() makes no network call.
    fake_lm = _ScoreFakeLM()
    scorer._lm = fake_lm
    loaded._lm = _ScoreFakeLM()

    prompt = "Characterize the regime as loadings on the named axes."
    before = scorer.score(prompt)
    after = loaded.score(prompt)

    assert before.parse_ok and after.parse_ok
    assert before.p_memorized == after.p_memorized


# --------------------------------------------------------------------------- #
# Task 2.6 — TiltExposure: REGIME_ASSET_EXPOSURE + loadings_to_tilt_views      #
# (Requirements 3.1, 3.2, 3.3, 3.4)                                           #
#                                                                             #
# Map a parsed regime-loadings vector to per-asset DIMENSIONLESS exposure      #
# tilts via a documented, NON-PREDICTIVE axis->category exposure table, packed #
# into MacroView so the UNCHANGED views_to_bl yields Q = tilt·conviction/252.  #
# These tests use the REAL MacroView + a real LlmMacroAgent.views_to_bl        #
# (offline — views_to_bl is pure arithmetic, no network).                     #
# --------------------------------------------------------------------------- #


from macro_framework.llm_agent import LlmMacroAgent, MacroView  # noqa: E402


def _all_real_symbols() -> list[str]:
    """The four real tickers backing the default AssetMap (BL column order)."""
    return ["SWDA.L", "XLK", "IAU", "BIL"]


def _crafted_loadings() -> "object":
    """A crafted RegimeLoadings vector with a known per-axis value (no clipping).

    Chosen so the documented dot-product tilt for each category is exact and
    hand-verifiable in the assertions below.
    """
    from macro_framework.factor_scoring import RegimeLoadings

    return RegimeLoadings(
        rebalance_date=pd.Timestamp("2022-06-30"),
        loadings={
            "inflation": 0.5,
            "growth": -0.4,
            "credit_stress": 0.8,
            "policy": 0.2,
            "risk_appetite": -0.6,
        },
        parse_ok=True,
    )


def test_regime_asset_exposure_is_documented_non_predictive_table() -> None:
    """REGIME_ASSET_EXPOSURE maps each anonymized category to a per-axis exposure (3.1).

    It is a heuristic exposure profile (dimensionless), not a fitted return
    model: each of the four asset categories carries one loading per MACRO_AXES
    axis. The values must be finite, dimensionless numbers in a small bounded
    range (a heuristic profile, never a forecast return).
    """
    import math

    from macro_framework.factor_scoring import MACRO_AXES, REGIME_ASSET_EXPOSURE

    categories = {"world_equity", "tech_sector", "gold_commodity", "short_treasury_cash"}
    assert set(REGIME_ASSET_EXPOSURE) == categories

    for category, profile in REGIME_ASSET_EXPOSURE.items():
        # One exposure per named axis, no extras / no missing.
        assert set(profile) == set(MACRO_AXES), f"{category} profile axes mismatch"
        for axis, exposure in profile.items():
            assert isinstance(exposure, (int, float)) and not isinstance(exposure, bool)
            assert math.isfinite(exposure), f"{category}.{axis} not finite"
            # A heuristic profile — small, dimensionless, bounded magnitude.
            assert -1.0 <= float(exposure) <= 1.0


def test_views_carry_tilt_as_expected_excess_and_conviction_as_confidence() -> None:
    """Each view packs tilt=expected_excess_annualized, conviction=confidence (3.1, 3.2).

    No expected-return semantics: the field is a dimensionless tilt, and the
    confidence field is the dimensionless conviction passed in by the caller.
    The tilt equals the documented dot-product Σ_axis loadings·exposure.
    """
    from macro_framework.factor_scoring import (
        REGIME_ASSET_EXPOSURE,
        loadings_to_tilt_views,
    )

    asset_map = AssetMap.default()
    loadings = _crafted_loadings()
    asset_snapshot = asset_map.pseudo_assets()
    conviction = 0.5

    views = loadings_to_tilt_views(loadings, asset_snapshot, asset_map, conviction)

    assert views, "expected one view per mapped asset"
    assert all(isinstance(v, MacroView) for v in views)

    # category -> pseudo id (Asset_A..D), so we can recompute the expected tilt.
    pseudo_to_category = asset_map.categories
    by_long = {v.asset_long: v for v in views}

    for pseudo, category in pseudo_to_category.items():
        assert pseudo in by_long, f"no view emitted for {pseudo} ({category})"
        view = by_long[pseudo]

        expected_tilt = sum(
            loadings.loadings[axis] * REGIME_ASSET_EXPOSURE[category][axis]
            for axis in loadings.loadings
        )

        # The tilt is packed as expected_excess_annualized (field reinterpretation).
        assert view.expected_excess_annualized == pytest.approx(expected_tilt)
        # Conviction is packed as confidence (dimensionless, caller-supplied).
        assert view.confidence == pytest.approx(conviction)
        # Long-only single-leg exposure (no short, no direction).
        assert view.asset_short is None
        # The rationale is axis-grounded (mentions a macro axis), not a return.
        assert any(axis in view.rationale for axis in loadings.loadings)


def test_unchanged_views_to_bl_yields_q_equals_tilt_times_conviction_over_252() -> None:
    """The UNCHANGED views_to_bl produces Q = tilt·conviction/252 (3.2, 3.3).

    This is the core field-reinterpretation contract: loadings_to_tilt_views
    packs tilt into expected_excess_annualized and conviction into confidence,
    and the real LlmMacroAgent.views_to_bl (reused without modification) then
    yields Q = expected_excess_annualized · clip(confidence,0,1) / 252.
    """
    from macro_framework.factor_scoring import (
        REGIME_ASSET_EXPOSURE,
        loadings_to_tilt_views,
    )

    asset_map = AssetMap.default()
    loadings = _crafted_loadings()
    asset_snapshot = asset_map.pseudo_assets()
    conviction = 0.5
    real_symbols = _all_real_symbols()

    views = loadings_to_tilt_views(loadings, asset_snapshot, asset_map, conviction)

    # The agent that owns the UNCHANGED views_to_bl conversion.
    agent = LlmMacroAgent(asset_map=asset_map)
    P, Q = agent.views_to_bl(views, real_symbols)

    assert P is not None and Q is not None
    assert len(Q) == len(views)

    # Recompute the expected Q per view = tilt · conviction / 252 and match it to
    # the BL output row for that view's long leg (P has a single +1 there).
    pseudo_to_real = asset_map.pseudo_to_real
    for row_idx, view in enumerate(views):
        long_real = pseudo_to_real[view.asset_long]
        col = real_symbols.index(long_real)
        # P row marks the long leg.
        assert P.iloc[row_idx, col] == 1.0

        category = asset_map.categories[view.asset_long]
        tilt = sum(
            loadings.loadings[axis] * REGIME_ASSET_EXPOSURE[category][axis]
            for axis in loadings.loadings
        )
        expected_q = (tilt * conviction) / 252.0
        assert Q.iloc[row_idx, 0] == pytest.approx(expected_q)


def test_non_finite_loadings_do_not_produce_nan_tilts() -> None:
    """NaN/Inf axis loadings are guarded so no NaN tilt propagates into BL (2.2 note, 3.1).

    Per the tasks.md 2.2 implementation note, non-standard JSON NaN/Infinity can
    bypass the parser's clip. The tilt step must treat a non-finite loading as 0
    (or skip the asset) so no NaN/Inf tilt reaches the unchanged views_to_bl.
    """
    import math

    from macro_framework.factor_scoring import RegimeLoadings, loadings_to_tilt_views

    asset_map = AssetMap.default()
    asset_snapshot = asset_map.pseudo_assets()

    loadings = RegimeLoadings(
        rebalance_date=pd.Timestamp("2022-06-30"),
        loadings={
            "inflation": float("nan"),
            "growth": float("inf"),
            "credit_stress": -float("inf"),
            "policy": 0.2,
            "risk_appetite": -0.6,
        },
        parse_ok=True,
    )

    views = loadings_to_tilt_views(loadings, asset_snapshot, asset_map, conviction=0.5)

    # Every emitted view carries a finite tilt (non-finite loadings treated as 0).
    for view in views:
        assert math.isfinite(view.expected_excess_annualized), (
            f"non-finite tilt for {view.asset_long}: {view.expected_excess_annualized}"
        )

    # And the unchanged views_to_bl produces a finite Q (no NaN into BL).
    agent = LlmMacroAgent(asset_map=asset_map)
    P, Q = agent.views_to_bl(views, _all_real_symbols())
    if Q is not None:
        for q in Q.iloc[:, 0].tolist():
            assert math.isfinite(q), f"NaN/Inf Q reached BL: {q}"


def test_loadings_to_tilt_views_introduces_no_predictive_objective() -> None:
    """No predictive-return objective: tilt is a pure dot-product, no direction (3.4).

    Structural check: the produced views carry only the dimensionless tilt
    (as expected_excess_annualized), the caller-supplied conviction (as
    confidence), and no short leg / direction. The tilt is NOT derived from any
    return target — it is the documented loadings·exposure dot-product, which is
    invariant to scaling conviction (conviction never feeds the tilt itself).
    """
    from macro_framework.factor_scoring import loadings_to_tilt_views

    asset_map = AssetMap.default()
    asset_snapshot = asset_map.pseudo_assets()
    loadings = _crafted_loadings()

    views_a = loadings_to_tilt_views(loadings, asset_snapshot, asset_map, conviction=0.2)
    views_b = loadings_to_tilt_views(loadings, asset_snapshot, asset_map, conviction=0.9)

    # The tilt (expected_excess_annualized) is independent of conviction — it is a
    # characterization of the regime exposure, never a forecast scaled by belief.
    by_long_a = {v.asset_long: v for v in views_a}
    by_long_b = {v.asset_long: v for v in views_b}
    assert set(by_long_a) == set(by_long_b)
    for pseudo in by_long_a:
        assert by_long_a[pseudo].expected_excess_annualized == pytest.approx(
            by_long_b[pseudo].expected_excess_annualized
        )
        # Only conviction (confidence) changed.
        assert by_long_a[pseudo].confidence == pytest.approx(0.2)
        assert by_long_b[pseudo].confidence == pytest.approx(0.9)
        # No short leg / directional bet.
        assert by_long_a[pseudo].asset_short is None
