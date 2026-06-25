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
