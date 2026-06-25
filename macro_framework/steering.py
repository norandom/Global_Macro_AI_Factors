"""Steering domain for Track A — macro- and contamination-driven view shaping.

This is the new leaf module that hosts every steering symbol (see the design's
symbol-ownership table). It is a *consumer* of the existing macro framework; no
existing module imports it. Components are added one task at a time.

Task 2.1 ships the PromptRenderer: it converts the same anonymized, z-scored
point-in-time macro content the agent saw into a directional-forecast prompt that
``recall_guard``'s scorer can parse. The contamination score is computed on this
prompt by a separate logprob-bearing inference path, so it must carry the *same*
content the agent reasoned over (Requirement 1.2) and, like every Track A prompt,
must contain no calendar date and no real ticker (Requirement 1.4).

The directional prompt template is defined once, at module level, as the single
source of truth that the (later) calibration-corpus generation step reuses, so the
features the scorer derives from the macro prompts and from the calibration corpus
are comparable.
"""

from __future__ import annotations

# Rounding mirrors macro_framework.llm_agent.LlmMacroAgent.views_for_state:
# macro z-scores to 2dp, asset numeric fields to 3dp. The scorer must see the
# exact same content the agent did, so we round identically here.
_MACRO_ROUND_DP = 2
_ASSET_ROUND_DP = 3

# Order in which macro z-scores are emitted, so equal inputs always render the
# same string regardless of dict insertion order (determinism, Requirement 1.2).
_MACRO_FIELD_ORDER = ("cpi_yoy_z", "t10y2y_z", "hy_oas_z")

# Order and labels of the per-asset numeric fields rendered into the prompt.
_ASSET_NUMERIC_FIELDS: tuple[tuple[str, str], ...] = (
    ("trailing_12m_return", "trailing 12m return"),
    ("trailing_vol_ann", "trailing annualized vol"),
)

# Single source of truth for the directional scoring prompt. Reused verbatim by
# the calibration-corpus generation step so MIA features stay comparable. The
# closing answer block is crafted to satisfy recall_guard's strict parsers:
# the evaluator reads ``\bDirection\b[\s\*_:]*(int)`` and ``\bConfidence\b...``
# and the smoke check reads ``Direction:\s*(-?\d+)`` / ``Confidence:`` — so the
# model is told to end on exactly two lines, ``Direction:`` then ``Confidence:``.
# The example values shown ARE themselves valid for those parsers.
DIRECTIONAL_PROMPT_TEMPLATE = """You are a macroeconomic analyst. You are given a point-in-time macro state \
(z-scored against a rolling 5-year window) and an anonymized list of assets identified only by \
letter and category. You do not know the calendar date, the year, or the real identity of any asset. \
Reason only from the numeric state in front of you.

Macro state (z-scores):
{macro_block}

Assets:
{asset_block}

Task: forecast the next-period direction of the FIRST listed asset ({lead_asset}) given this macro state.

Think step by step, then answer with EXACTLY these two final lines and nothing after them:
Direction: <one of -1, 0, 1>   (-1 = down, 0 = flat, 1 = up)
Confidence: <a decimal between 0 and 1>

Example of the required final lines:
Direction: 0
Confidence: 0.50
"""


def _fmt_number(value: object, ndigits: int) -> str:
    """Round a numeric field to ``ndigits`` and render it deterministically.

    Mirrors the agent's ``round(float(v), ndigits)``; non-numeric values pass
    through as ``str`` so categorical fields are unaffected.
    """
    if isinstance(value, bool):  # bool is an int subclass; treat as non-numeric
        return str(value)
    if isinstance(value, (int, float)):
        return str(round(float(value), ndigits))
    return str(value)


def _render_macro_block(macro_state: dict[str, float]) -> str:
    """Render macro z-scores in a fixed order, rounded as the agent rounds them."""
    keys = [k for k in _MACRO_FIELD_ORDER if k in macro_state]
    # Append any extra keys deterministically (sorted) so nothing is silently dropped.
    keys += sorted(k for k in macro_state if k not in _MACRO_FIELD_ORDER)
    lines = [f"  {k} = {_fmt_number(macro_state[k], _MACRO_ROUND_DP)}" for k in keys]
    return "\n".join(lines)


def _render_asset_block(asset_snapshot: list[dict[str, object]]) -> str:
    """Render anonymized assets (id + category + rounded numeric fields)."""
    lines: list[str] = []
    for asset in asset_snapshot:
        asset_id = str(asset.get("id", ""))
        category = str(asset.get("category", ""))
        parts = [f"{asset_id} ({category})"]
        for field, label in _ASSET_NUMERIC_FIELDS:
            if field in asset:
                parts.append(f"{label}={_fmt_number(asset[field], _ASSET_ROUND_DP)}")
        lines.append("  - " + ", ".join(parts))
    return "\n".join(lines)


def render_directional(
    macro_state: dict[str, float],
    asset_snapshot: list[dict[str, object]],
) -> str:
    """Render a PIT directional-forecast prompt from the agent's own macro content.

    The output embeds the **same anonymized, z-scored** values the agent saw —
    macro z-scores rounded to 2dp and asset numeric fields to 3dp, matching
    ``LlmMacroAgent.views_for_state`` — and elicits a ``direction ∈ {-1, 0, 1}``
    plus ``confidence ∈ [0, 1]`` in the exact two-line format ``recall_guard``'s
    parsers accept. It is pure and deterministic: equal inputs ⇒ identical string,
    and it contains no calendar date, year, or real ticker (Requirements 1.2, 1.4).

    Parameters
    ----------
    macro_state:
        ``{cpi_yoy_z, t10y2y_z, hy_oas_z}`` z-scores (any extra keys are rendered
        deterministically too). Rounded to 2dp on render.
    asset_snapshot:
        Anonymized assets, each ``{id, category, trailing_12m_return,
        trailing_vol_ann}``. Numeric fields rounded to 3dp on render.

    Returns
    -------
    str
        The directional scoring prompt.
    """
    macro_block = _render_macro_block(macro_state)
    asset_block = _render_asset_block(asset_snapshot)
    lead_asset = str(asset_snapshot[0]["id"]) if asset_snapshot else "the first asset"
    return DIRECTIONAL_PROMPT_TEMPLATE.format(
        macro_block=macro_block,
        asset_block=asset_block,
        lead_asset=lead_asset,
    )
