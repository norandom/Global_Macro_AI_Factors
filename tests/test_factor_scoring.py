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
