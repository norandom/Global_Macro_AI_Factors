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
