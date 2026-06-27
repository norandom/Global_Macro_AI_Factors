"""Number-native AI macro factors — version-aware contamination scoring layer.

A new leaf module (no existing module imports it) hosting the
version-aware-factor-scoring symbols. It builds additively on the
``track-a-macro-steering`` engine and the released ``recall_guard`` public MIA
primitives; it never edits ``llm_agent.py`` / ``steering.py`` / ``recall_guard``.

This file is populated task-by-task (the symbol-ownership table in the design).
Task 2.1 defines the first symbols: the named macro axes and the one
regime-loadings prompt renderer (anonymized PIT default, identifying non-PIT
control) — the single source of truth for the factor task.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

# Named macro axes for the regime-as-loadings factor (locked decision, 2026-06-26):
# inflation pressure, growth/cycle, credit/liquidity stress, policy stance, risk
# appetite. Loadings are continuous, bounded in [-1, +1]; never a direction/return.
MACRO_AXES: tuple[str, ...] = ("inflation", "growth", "credit_stress", "policy", "risk_appetite")


def _fmt_macro_state(macro_state: dict[str, float]) -> str:
    """Deterministic rendering of the z-scored macro state (sorted by key)."""
    return "\n".join(f"  - {k}: {float(v):+.2f}" for k, v in sorted(macro_state.items()))


def _fmt_assets(asset_snapshot: list[dict[str, object]]) -> str:
    """Deterministic rendering of the anonymized assets (letter id + category only).

    Mirrors the ``llm_agent`` anonymized convention: an asset is identified by its
    pseudo letter and category — never a ticker — so this block is identical in the
    anonymized and identifying forms (the ticker reveal is a SEPARATE added block).
    """
    lines: list[str] = []
    for a in asset_snapshot:
        aid = a.get("id")
        category = a.get("category")
        extras = {
            k: v for k, v in a.items() if k not in ("id", "category")
        }
        extra_str = ""
        if extras:
            extra_str = " (" + ", ".join(
                f"{k}={float(v):+.3f}" if isinstance(v, (int, float)) and not isinstance(v, bool)
                else f"{k}={v}"
                for k, v in sorted(extras.items())
            ) + ")"
        lines.append(f"  - {aid}: {category}{extra_str}")
    return "\n".join(lines)


def _axes_block() -> str:
    return "\n".join(f"  - {axis}" for axis in MACRO_AXES)


def render_regime_loadings_prompt(
    macro_state: dict[str, float],
    asset_snapshot: list[dict[str, object]],
    *,
    identifying: bool = False,
    as_of: Any | None = None,
    raw_levels: dict[str, float] | None = None,
) -> str:
    """Render the regime-as-loadings factor task — one renderer, two framings.

    The model is asked to characterize the macro state as continuous loadings in
    ``[-1, +1]`` on the five named ``MACRO_AXES``. It is NEVER asked for a buy/sell
    direction or an expected/forecast return (R2.2, R2.5) — the output is a factor
    vector, not a bet, and no forecasting target is defined.

    Two framings share this single source of truth so a later PIT-vs-non-PIT
    contrast can attribute the contamination delta to the point-in-time discipline
    alone (R7.6 — hold all else equal):

    - **Anonymized (PIT, default ``identifying=False``)** — the z-scored macro state
      + anonymized assets (Asset_A–D + category). Contains NO calendar date/year and
      NO real ticker: the point-in-time, recall-disabled form (R1.4, R2.3).
    - **Identifying (``identifying=True``)** — adds exactly the recall-enabling tokens
      (the real tickers, the ``as_of`` date, and the ``raw_levels``) and is OTHERWISE
      token-identical to the anonymized form: the only difference is the
      identity/date/raw-level additions (R7.6). ``as_of`` and ``raw_levels`` are
      required in this framing.

    Deterministic: equal inputs produce an identical string.

    Args:
        macro_state: z-scored macro state (e.g. ``cpi_yoy_z``, ``t10y2y_z``, ``hy_oas_z``).
        asset_snapshot: anonymized asset descriptors (``id`` letter + ``category``, plus
            optional numeric trailing stats); no ticker.
        identifying: when True, render the recall-enabling (non-PIT) control form.
        as_of: the rebalance date to disclose (identifying only); required when identifying.
        raw_levels: the raw non-normalized macro levels to disclose (identifying only);
            required when identifying.

    Returns:
        The rendered prompt string.

    Raises:
        ValueError: when ``identifying=True`` but ``as_of`` or ``raw_levels`` is missing.
    """
    if identifying and (as_of is None or raw_levels is None):
        raise ValueError(
            "identifying=True requires both as_of and raw_levels "
            "(the recall-enabling date + raw macro levels); "
            f"got as_of={as_of!r}, raw_levels={raw_levels!r}"
        )

    # --- Anonymized (PIT) base form. Built first, line by line, so the identifying
    # form can be produced by INSERTING extra blocks without altering any base line
    # (guarantees R7.6 token-identity except the additions). ---
    base_lines: list[str] = [
        "You are a macroeconomic game theorist. Characterize the current macro regime "
        "as continuous factor loadings.",
        "",
        "You are given the macro state (z-scored against a rolling window) and an "
        "anonymized list of assets identified only by letter and category. You do not "
        "know what year it is; reason only from the numeric state in front of you.",
        "",
        "Macro state (z-scored):",
        _fmt_macro_state(macro_state),
        "",
        "Assets (anonymized):",
        _fmt_assets(asset_snapshot),
        "",
        "Characterize the regime as a continuous loading on each of these five macro axes:",
        _axes_block(),
        "",
        "Rules:",
        "  - Each loading is a continuous number in [-1, +1] (use -1 and +1 as the bounds).",
        "  - A loading describes how strongly the regime sits on that axis — it is NOT a "
        "trade, a bet, or a prediction.",
        "  - Do NOT output a long/short/neutral position and do NOT output a return for any "
        "asset; characterize the regime only.",
        "",
        'Output a JSON object mapping each axis name to its loading, e.g. '
        '{"inflation": 0.4, "growth": -0.2, "credit_stress": 0.6, "policy": -0.1, '
        '"risk_appetite": -0.3}.',
    ]

    if not identifying:
        return "\n".join(base_lines)

    # --- Identifying (non-PIT) additions. Inserted as SEPARATE blocks; every base
    # line above is reproduced verbatim, so the only difference is these additions. ---
    assert as_of is not None and raw_levels is not None  # narrowed by the guard above
    real_to_pseudo = _default_asset_identities()
    identity_lines = [
        f"  - {pseudo}: {real}" for real, pseudo in real_to_pseudo.items()
    ]
    raw_lines = [f"  - {k}: {float(v):g}" for k, v in sorted(raw_levels.items())]

    identifying_block = [
        "",
        f"As-of date: {as_of}",
        "",
        "Real asset identities:",
        *identity_lines,
        "",
        "Raw (non-normalized) macro levels:",
        *raw_lines,
    ]

    return "\n".join(base_lines + identifying_block)


def _default_asset_identities() -> dict[str, str]:
    """Real ticker -> pseudo letter for the identifying (non-PIT) reveal.

    Read-only mirror of ``macro_framework.anonymize.AssetMap.default()``; defined
    here (lazy import) so the renderer's identifying form can disclose the real
    tickers (SWDA.L, XLK, IAU, BIL) without the renderer hard-depending on AssetMap
    construction at module import time.
    """
    from .anonymize import AssetMap

    return dict(AssetMap.default().real_to_pseudo)


# --------------------------------------------------------------------------- #
# Task 2.2 — LoadingsParser: RegimeLoadings dataclass + parse_loadings         #
# (Requirements 2.1, 2.4)                                                      #
#                                                                              #
# Scoring needs NO parse (the MIA features come from logprobs); only factor    #
# *consumption* of the loadings parses the model reply. The parser is pure and #
# deterministic: it extracts one loading per MACRO_AXES axis, clips each to     #
# [-1, +1], and returns the not-parsed result when the reply does not yield     #
# the full five-axis vector — it never fabricates missing axes.                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RegimeLoadings:
    """A parsed regime-loadings factor vector keyed by its rebalance date (R2.4).

    The per-rebalance consumable artifact: one continuous loading per named
    ``MACRO_AXES`` axis, each bounded in ``[-1, +1]``. It is a characterization
    of the regime, never a direction or a return (R2.1, R2.2). Frozen so a stream
    of these can be safely indexed by ``rebalance_date``.

    Attributes:
        rebalance_date: the date this vector characterizes (the artifact key).
        loadings: axis name -> loading in ``[-1, +1]``. On a successful parse this
            holds exactly one entry per ``MACRO_AXES`` axis; on a not-parsed
            result it carries no fabricated full vector.
        parse_ok: whether the model reply yielded the full five-axis vector.
    """

    rebalance_date: pd.Timestamp
    loadings: dict[str, float]
    parse_ok: bool


def _clip_unit(value: float) -> float:
    """Clip a loading to the closed ``[-1, +1]`` interval (R2.1)."""
    if value < -1.0:
        return -1.0
    if value > 1.0:
        return 1.0
    return value


# An axis name followed by a number, e.g. ``inflation: 0.6``, ``"growth" = -0.2``,
# ``credit_stress -> 1.4``. Tolerant of JSON-ish, labeled-list, and "key value"
# formats; the axis name is matched literally so only MACRO_AXES axes are read.
_NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def _extract_axis_value(text: str, axis: str) -> float | None:
    """Find the first numeric loading associated with ``axis`` in ``text``.

    Matches the axis name (optionally quoted) followed by a separator
    (``:``, ``=``, ``->``) and a number. Returns ``None`` when the axis is not
    present with an associated number (so the caller never fabricates it).
    """
    pattern = re.compile(
        rf'["\']?{re.escape(axis)}["\']?\s*(?::|=|->)\s*({_NUMBER})',
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:  # pragma: no cover - regex guarantees a numeric group
        return None


def _loadings_from_json(text: str) -> dict[str, float] | None:
    """Try to read a JSON object (possibly embedded in prose) into axis floats.

    Returns the subset of ``MACRO_AXES`` present with numeric values, or ``None``
    if no JSON object can be located/decoded. Used as a first, exact pass before
    the tolerant regex fallback.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    out: dict[str, float] = {}
    for axis in MACRO_AXES:
        if axis in obj:
            raw = obj[axis]
            if isinstance(raw, bool):  # bool is an int subclass; reject it
                continue
            if isinstance(raw, (int, float)):
                out[axis] = float(raw)
    return out


def parse_loadings(text: str, rebalance_date: pd.Timestamp) -> RegimeLoadings | None:
    """Parse a model reply into a ``RegimeLoadings`` factor vector (R2.1, R2.4).

    Extracts one loading per ``MACRO_AXES`` axis from a model reply and clips each
    to ``[-1, +1]``. Tolerant of reasonable reply shapes — a JSON object (even
    embedded in prose) or a labeled list (``inflation: 0.6``). Pure and
    deterministic: equal inputs yield equal results.

    When the reply does NOT yield the full five-axis vector (missing axes,
    garbage), returns ``None`` rather than fabricating the missing axes — the
    caller falls back to the base allocation, exactly as Track A does on empty
    views. Missing axes are never zero-filled.

    Args:
        text: the model's reply text.
        rebalance_date: the date this vector characterizes (the artifact key).

    Returns:
        A ``RegimeLoadings`` with ``parse_ok=True`` and one clipped loading per
        axis when the full vector is present; otherwise ``None``.
    """
    values: dict[str, float] = {}

    # First pass: exact JSON decode (the prompt requests a JSON object).
    json_values = _loadings_from_json(text)
    if json_values:
        values.update(json_values)

    # Second pass: tolerant per-axis regex for any axis JSON did not yield (so a
    # labeled-list or mixed reply still resolves the full vector).
    for axis in MACRO_AXES:
        if axis in values:
            continue
        found = _extract_axis_value(text, axis)
        if found is not None:
            values[axis] = found

    # Not-parsed unless every named axis was found — never fabricate the rest.
    if len(values) != len(MACRO_AXES):
        return None

    loadings = {axis: _clip_unit(values[axis]) for axis in MACRO_AXES}
    return RegimeLoadings(
        rebalance_date=rebalance_date, loadings=loadings, parse_ok=True
    )
