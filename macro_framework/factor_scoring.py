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

from typing import Any

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
