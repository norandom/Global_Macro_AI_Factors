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
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# Number-native MIA calibration/scoring primitives from the released library.
# These are imported AT MODULE LEVEL (not lazily) so they are bound on this
# module's namespace and can be patched as `macro_framework.factor_scoring.<name>`
# in tests (the calibration path is mocked there — no NIM/FMP calls). The
# directional MemoryGuardedScorer facade is deliberately NOT imported (R6.5).
from recall_guard import NvidiaLM, build_baseline, compute_mia_features
from recall_guard.core.loader import EvalRow
from recall_guard.core.nvidia_lm import generate_many
from recall_guard.mia.mcs import train

# Read-only import of the existing agent's view dataclass (R3): the tilt-as-exposure
# builder PACKS regime exposures into MacroView so the UNCHANGED views_to_bl
# conversion yields Q = tilt·conviction/252. MacroView is a lightweight dataclass
# (the heavy DSPy wiring in LlmMacroAgent stays lazy), so this stays off the hot
# path while making the symbol patchable in tests. llm_agent is never edited.
from .llm_agent import MacroView

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from datetime import date
    from pathlib import Path

    import pandas as pd

    from recall_guard.mia.control import ControlBaseline
    from recall_guard.mia.mcs import MCSCalibrator

    from .anonymize import AssetMap

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


# An axis name, a separator, and a number, e.g. ``inflation: 0.6``,
# ``"growth" = -0.2``, ``credit_stress -> 1.4``. Tolerant of JSON-ish and
# labeled-list formats; a ``:``/``=``/``->`` separator is required (a bare
# "key value" pair does not match), and the axis name is matched literally so
# only MACRO_AXES axes are read.
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


# --------------------------------------------------------------------------- #
# Task 2.3 — FactorScorer: number-native calibration + configuration errors     #
# (Requirements 1.1, 1.5, 1.6, 6.5)                                            #
#                                                                              #
# The version-aware contamination scorer. It is NUMBER-NATIVE: the calibrator   #
# is trained on the macro numbers themselves on the regime-loadings factor task #
# (validated 2026-06-26: holdout_auc ~= 0.96, is_weak=False — see research.md).  #
#                                                                              #
#   - recall class (IS)   = pre-cutoff macro states presented IDENTIFYINGLY     #
#                           (real date + raw levels + real tickers).            #
#   - recall-guarded class (OOS)  = the SAME states presented ANONYMIZED                #
#                           (z-scores, no date, Asset_A-D).                     #
#                                                                              #
# The only difference between the two corpora is the identifying-vs-anonymized  #
# framing on one shared factor task (R7.6 — hold all else equal); that is the   #
# axis the calibrator learns. Calibration uses the released library's PUBLIC    #
# MIA primitives (build_baseline / train), never the directional facade (R6.5). #
#                                                                              #
# This task defines the calibration surface only: ConfigurationError, the       #
# FactorScore / CalibrationStats dataclasses, and FactorScorer.calibrate with   #
# its is_weak / holdout_auc properties. score / score_many (2.4) and            #
# save / load (2.5) are intentionally NOT implemented here.                     #
# --------------------------------------------------------------------------- #


class ConfigurationError(RuntimeError):
    """The factor-scoring layer's own fail-fast configuration error (R1.5).

    The released ``recall_guard`` library raises a ``ConfigurationError`` only
    from its directional ``MemoryGuardedScorer`` facade — and this module
    deliberately BYPASSES that facade (R6.5), using the lower-level public MIA
    primitives instead. The primitive path therefore never raises one. So this
    module DEFINES and OWNS the R1.5 contract: a clear configuration error is
    surfaced rather than returning an unscored or silently invalid result.

    Raised on:
      - an empty ``api_key`` (guarded BEFORE constructing ``NvidiaLM``, which
        would otherwise raise a bare ``ValueError``);
      - ``baseline.n_valid == 0`` after ``build_baseline`` (no control corpus to
        standardise against — scoring would be invalid);
      - an auth-class ``RuntimeError`` (401/403 / invalid-api-key markers, see
        ``_AUTH_MARKERS``) surfacing from ``score`` / ``score_many`` (task 2.4).

    Subclasses ``RuntimeError`` so callers that already catch the library's
    runtime errors keep working, while ``ConfigurationError`` callers get the
    specific, intent-revealing type.
    """


# Substrings that mark an authentication/authorisation failure in a ``RuntimeError``
# message raised by ``NvidiaLM.generate``. A small, module-local re-implementation of
# the marker idea used by recall_guard's bypassed directional façade — re-stated here
# (NOT imported, R6.5) so the number-native scoring path can map a rejected credential
# to THIS module's own ``ConfigurationError`` (R1.5) rather than a silent fail record.
_AUTH_MARKERS: tuple[str, ...] = (
    "401",
    "403",
    "unauthorized",
    "forbidden",
    "authentication",
    "invalid api key",
    "invalid_api_key",
)


def _is_auth_error(exc: BaseException) -> bool:
    """Return True iff ``exc``'s message looks like a NIM auth/authorisation failure.

    HTTP 401/403, ``unauthorized``/``forbidden``, or an ``invalid api key`` marker
    signals a rejected scoring credential — the R1.5 fail-fast configuration case.
    """
    message = str(exc).lower()
    return any(marker in message for marker in _AUTH_MARKERS)


@dataclass(frozen=True)
class FactorScore:
    """The version-aware contamination score for one scored factor prompt (R1.1).

    Produced per call by ``FactorScorer.score`` (task 2.4). Defined here so the
    calibration surface and the scoring surface share one contract.

    Attributes:
        p_memorized: the calibrated probability of memorization in ``[0, 1]``,
            or ``None`` when scoring failed for this prompt (e.g. a logprob /
            feature-computation failure) so the rebalance can fall back to an
            unadjusted exposure rather than crash.
        parse_ok: whether the model's reply was usable for scoring. (Scoring
            itself needs NO loadings parse — the MIA features come from the
            logprobs — so this reflects the scoring path's own success.)
        fail_reason: a short human-readable reason when ``p_memorized is None``;
            ``None`` on success.
    """

    p_memorized: float | None
    parse_ok: bool
    fail_reason: str | None


@dataclass(frozen=True)
class CalibrationStats:
    """Recorded statistics of a trained number-native calibrator (R1.6).

    Surfaced so a weak/uncalibrated calibrator is reported rather than reporting
    a contamination value as if validated. ``is_weak`` is ``True`` iff the
    held-out AUC fell below the configured ``min_auc`` at train time.

    Attributes:
        holdout_auc: ROC-AUC of the trained classifier on the held-out portion
            of the identifying-IS vs anonymized-OOS corpus.
        is_weak: ``True`` iff ``holdout_auc < min_auc`` — the fallback signal.
        n_is: number of identifying (recall-class) prompts built for calibration.
        n_oos: number of anonymized (recall-guarded-class) prompts built for calibration.
    """

    holdout_auc: float
    is_weak: bool
    n_is: int
    n_oos: int


# Raw macro level -> z-scored column name in the FRED macro panel. The renderer
# reads the z-scored state for the anonymized (PIT) framing and the raw levels
# for the identifying (recall-enabling) additions; both come from the same panel
# row, so revealing the period in the numbers is the only difference.
_RAW_TO_Z: dict[str, str] = {
    "cpi_yoy": "cpi_yoy_z",
    "t10y2y": "t10y2y_z",
    "hy_oas": "hy_oas_z",
}


def _asset_snapshot_from_map(asset_map: AssetMap) -> list[dict[str, object]]:
    """Build the anonymized asset descriptors (letter id + category) from the map.

    Identical content for the IS and OOS framings — the ticker reveal is a
    SEPARATE block added by the renderer's identifying form — so the calibration
    corpora differ ONLY by the identifying additions (R7.6).
    """
    snapshot: list[dict[str, object]] = []
    for pseudo, category in sorted(asset_map.categories.items()):
        snapshot.append({"id": pseudo, "category": category})
    return snapshot


def _build_corpus(
    *,
    cutoff_date: date,
    macro_panel: pd.DataFrame,
    asset_map: AssetMap,
    n_per_class: int,
) -> tuple[list[EvalRow], list[EvalRow]]:
    """Build the number-native calibration corpora from pre-cutoff macro rows.

    Selects up to ``n_per_class`` evenly-spaced pre-cutoff (``< cutoff_date``)
    rows from the macro panel and renders, for each, the SAME regime-loadings
    factor task in two framings:

      - **identifying IS** (``identifying=True``, ``as_of=<date>``,
        ``raw_levels=<raw>``) — the recall-enabled class;
      - **anonymized OOS** (``identifying=False``) — the recall-guarded (recall-disabled)
        class.

    Returns ``(is_rows, oos_rows)`` as ``EvalRow`` lists. The two corpora are
    token-identical except the identifying additions (R7.6), so the calibrator
    learns the identifying-vs-anonymized boundary alone.
    """
    cutoff_ts = _pd().Timestamp(cutoff_date)
    pre = macro_panel.loc[macro_panel.index < cutoff_ts]
    if pre.empty:
        return [], []

    # Evenly-spaced sample so the corpus spans the pre-cutoff history rather than
    # clustering at one end; deterministic (no RNG) for reproducible calibration.
    n = min(int(n_per_class), len(pre))
    if n <= 0:
        return [], []
    if n >= len(pre):
        rows = pre
    else:
        positions = [round(i * (len(pre) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
        rows = pre.iloc[sorted(set(positions))]

    asset_snapshot = _asset_snapshot_from_map(asset_map)

    is_rows: list[EvalRow] = []
    oos_rows: list[EvalRow] = []
    for as_of, row in rows.iterrows():
        macro_state = {
            z_col: float(row[z_col]) for z_col in _RAW_TO_Z.values() if z_col in row
        }
        raw_levels = {
            raw_col: float(row[raw_col]) for raw_col in _RAW_TO_Z if raw_col in row
        }

        oos_prompt = render_regime_loadings_prompt(
            macro_state, asset_snapshot, identifying=False
        )
        is_prompt = render_regime_loadings_prompt(
            macro_state,
            asset_snapshot,
            identifying=True,
            as_of=_pd().Timestamp(as_of).date().isoformat(),
            raw_levels=raw_levels,
        )

        date_tag = _pd().Timestamp(as_of).date().isoformat()
        oos_rows.append(
            EvalRow(prompt=oos_prompt, target_direction=0, metadata={"as_of": date_tag})
        )
        is_rows.append(
            EvalRow(prompt=is_prompt, target_direction=0, metadata={"as_of": date_tag})
        )

    return is_rows, oos_rows


def _pd():  # type: ignore[no-untyped-def]
    """Lazy ``pandas`` import (keeps it off the module-import hot path)."""
    import pandas as pd

    return pd


# Persisted-artifact filenames (task 2.5). The directory written by
# ``FactorScorer.save`` holds the pickled calibrator + the JSON baseline/stats;
# the api_key / NvidiaLM are never among them (the credential is supplied at load).
_CALIBRATOR_FILE = "calibrator.joblib"
_BASELINE_FILE = "baseline.json"
_STATS_FILE = "stats.json"


class FactorScorer:
    """Number-native, version-aware contamination scorer (R1.1, R1.5, R1.6, R6.5).

    Holds a trained ``MCSCalibrator`` + its ``ControlBaseline`` + the live
    ``NvidiaLM`` so any factor prompt can later be scored for ``p_memorized``
    via the public MIA primitives (``score`` / ``score_many`` are task 2.4).
    The calibrator is trained number-natively on the macro numbers themselves
    (identifying IS vs anonymized OOS on the regime-loadings task).

    Construct via :meth:`calibrate`; direct construction holds the three trained
    components. Persistence (``save`` / ``load``) is task 2.5.
    """

    def __init__(
        self,
        *,
        calibrator: MCSCalibrator,
        baseline: ControlBaseline,
        lm: NvidiaLM,
        stats: CalibrationStats,
    ) -> None:
        self._calibrator = calibrator
        self._baseline = baseline
        self._lm = lm
        self._stats = stats

    @classmethod
    def calibrate(
        cls,
        *,
        nim_model: str,
        cutoff_date: date,
        macro_panel: pd.DataFrame,
        asset_map: AssetMap,
        api_key: str,
        n_per_class: int = 60,
        min_auc: float = 0.6,
        max_workers: int = 8,
        lm_factory: Callable[[str, str], NvidiaLM] | None = None,
    ) -> FactorScorer:
        """Calibrate a number-native contamination scorer on the macro numbers.

        Builds the identifying-IS + anonymized-OOS corpora from the pre-cutoff
        (``< cutoff_date``) macro-panel rows on the shared regime-loadings task,
        builds the control baseline from the OOS corpus, and trains the
        ``MCSCalibrator`` to separate the two framings. ``ref_lm`` is fixed at
        ``None`` so the ``ref_delta`` feature stays inert and ``feature_order``
        excludes it (a locked, deterministic choice).

        Args:
            nim_model: the logprob-bearing NIM model id (its cutoff defines the
                pre-cutoff IS window).
            cutoff_date: the model's training cutoff; only rows before it are used.
            macro_panel: the FRED macro panel with raw + z-scored columns,
                indexed by date.
            asset_map: the identifying <-> anonymized asset map.
            api_key: the NIM scoring credential (non-empty).
            n_per_class: target number of prompts per class (IS and OOS).
            min_auc: weak-calibrator threshold (``is_weak`` iff below it).
            max_workers: parallelism for the baseline + train LM calls.
            lm_factory: optional ``(api_key, model) -> lm`` factory (e.g. an
                ``NvidiaLM`` with a longer timeout for slow-serving models).

        Returns:
            A calibrated ``FactorScorer``.

        Raises:
            ConfigurationError: when ``api_key`` is empty (R1.5, surfaced BEFORE
                constructing ``NvidiaLM`` so the caller gets the intent-revealing
                type rather than a bare ``ValueError``); or when the control
                baseline has ``n_valid == 0`` (no corpus to standardise against).
        """
        # R1.5: fail fast on a missing credential BEFORE NvidiaLM construction
        # (NvidiaLM itself would raise a bare ValueError — we own the contract).
        if not api_key:
            raise ConfigurationError(
                "FactorScorer.calibrate: a non-empty NIM api_key is required "
                "(the scoring credential is missing)."
            )

        is_rows, oos_rows = _build_corpus(
            cutoff_date=cutoff_date,
            macro_panel=macro_panel,
            asset_map=asset_map,
            n_per_class=n_per_class,
        )

        # lm_factory mirrors screen_candidate: inject a client with a longer
        # timeout for slow-serving models (NvidiaLM defaults to 15 s).
        if lm_factory is not None:
            lm = lm_factory(api_key, nim_model)
        else:
            lm = NvidiaLM(api_key=api_key, model=nim_model)

        baseline = build_baseline(
            lm,
            oos_rows,
            None,  # ref_lm=None -> ref_delta inert, feature_order excludes it
            min_valid=min(len(oos_rows), 2),
            max_workers=max_workers,
        )
        if baseline.n_valid == 0:
            raise ConfigurationError(
                "FactorScorer.calibrate: the control baseline has no valid rows "
                "(baseline.n_valid == 0); cannot standardise MIA features. "
                "Check the NIM credential/endpoint and that the panel yielded a "
                "non-empty pre-cutoff OOS corpus."
            )

        calibrator = train(
            model_lm=lm,
            is_memorized=is_rows,
            oos_control=oos_rows,
            baseline=baseline,
            ref_lm=None,  # ref_delta inert; feature_order excludes it (locked)
            min_auc=min_auc,
            max_workers=max_workers,
        )

        stats = CalibrationStats(
            holdout_auc=float(calibrator.holdout_auc),
            is_weak=bool(calibrator.is_weak),
            n_is=len(is_rows),
            n_oos=len(oos_rows),
        )
        return cls(calibrator=calibrator, baseline=baseline, lm=lm, stats=stats)

    @property
    def is_weak(self) -> bool:
        """Whether the trained calibrator is weak (``holdout_auc < min_auc``) (R1.6).

        The fallback signal: when weak, the recall-guarded adjustment leaves exposures
        unadjusted rather than applying an unvalidated discount.
        """
        return bool(self._calibrator.is_weak)

    @property
    def holdout_auc(self) -> float:
        """The trained calibrator's held-out AUC (R1.6)."""
        return float(self._calibrator.holdout_auc)

    @property
    def stats(self) -> CalibrationStats:
        """The recorded calibration statistics."""
        return self._stats

    # -- persistence (task 2.5) ------------------------------------------------ #

    def save(self, path: Path) -> None:
        """Persist the trained scorer to a directory — never the credential (R1.6).

        The number-native calibration is a one-time cost (~3 × n_per_class live
        NIM calls: the baseline pass plus both corpora again in training); this
        writes the trained components so they are reused across notebooks without
        a rebuild. Three artifacts are written under ``path`` (created if absent):

          - ``calibrator.joblib`` — the pickled :class:`MCSCalibrator`, which
            carries the fitted sklearn ``LogisticRegression`` **and** its
            ``feature_order`` (so ``predict_proba`` can never be fed a permuted
            vector). joblib round-trips the estimator exactly.
          - ``baseline.json`` — the :class:`ControlBaseline` standardisation
            stats (``feature_means`` / ``feature_stds`` plus ``model`` /
            ``n_valid`` / ``is_calibrated`` / ``min_valid``). These are the only
            fields :meth:`predict_proba` needs to standardise features, so the
            baseline reconstructs from them faithfully.
          - ``stats.json`` — the :class:`CalibrationStats`.

        The ``api_key`` and the live ``NvidiaLM`` are NEVER persisted — the model
        id is recorded inside ``baseline.json`` (``model``) and :meth:`load`
        re-attaches a fresh ``NvidiaLM(api_key, model)`` from a caller-supplied
        key. No secret is ever written to disk.

        Args:
            path: the destination directory (created with parents if needed).
        """
        import joblib

        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self._calibrator, path / _CALIBRATOR_FILE)

        baseline_payload = {
            "model": self._baseline.model,
            "n_valid": int(self._baseline.n_valid),
            "feature_means": dict(self._baseline.feature_means),
            "feature_stds": dict(self._baseline.feature_stds),
            "is_calibrated": bool(self._baseline.is_calibrated),
            "min_valid": int(self._baseline.min_valid),
        }
        (path / _BASELINE_FILE).write_text(
            json.dumps(baseline_payload, indent=2, sort_keys=True), encoding="utf-8"
        )

        stats_payload = {
            "holdout_auc": float(self._stats.holdout_auc),
            "is_weak": bool(self._stats.is_weak),
            "n_is": int(self._stats.n_is),
            "n_oos": int(self._stats.n_oos),
        }
        (path / _STATS_FILE).write_text(
            json.dumps(stats_payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        api_key: str,
        lm_factory: Callable[[str, str], NvidiaLM] | None = None,
    ) -> FactorScorer:
        """Reconstruct a fully usable scorer from a saved directory (R1.6).

        Reads the joblib-pickled :class:`MCSCalibrator`, reconstructs the
        :class:`ControlBaseline` from ``baseline.json``, reads the
        :class:`CalibrationStats`, and re-attaches a FRESH
        ``NvidiaLM(api_key, model)`` — the credential is supplied here, never
        loaded from disk. A loaded scorer produces identical
        ``predict_proba`` / scores to the original for the same inputs.

        Args:
            path: the directory previously written by :meth:`save`.
            api_key: the NIM scoring credential to attach to the fresh
                ``NvidiaLM`` (non-empty).
            lm_factory: optional ``(api_key, model) -> lm`` factory (e.g. an
                ``NvidiaLM`` with a longer timeout for slow-serving models).

        Returns:
            A fully usable :class:`FactorScorer`.

        Raises:
            ConfigurationError: when ``api_key`` is empty (R1.5; the loaded
                scorer would have no usable scoring credential).
        """
        # R1.5: fail fast on a missing credential BEFORE NvidiaLM construction
        # (NvidiaLM itself would raise a bare ValueError — we own the contract).
        if not api_key:
            raise ConfigurationError(
                "FactorScorer.load: a non-empty NIM api_key is required "
                "(the scoring credential is never persisted; supply it at load)."
            )

        import joblib

        from recall_guard.mia.control import ControlBaseline

        calibrator: MCSCalibrator = joblib.load(path / _CALIBRATOR_FILE)

        baseline_payload = json.loads(
            (path / _BASELINE_FILE).read_text(encoding="utf-8")
        )
        baseline = ControlBaseline(
            model=baseline_payload["model"],
            n_valid=int(baseline_payload["n_valid"]),
            feature_means=dict(baseline_payload["feature_means"]),
            feature_stds=dict(baseline_payload["feature_stds"]),
            is_calibrated=bool(baseline_payload["is_calibrated"]),
            min_valid=int(baseline_payload["min_valid"]),
        )

        stats_payload = json.loads(
            (path / _STATS_FILE).read_text(encoding="utf-8")
        )
        stats = CalibrationStats(
            holdout_auc=float(stats_payload["holdout_auc"]),
            is_weak=bool(stats_payload["is_weak"]),
            n_is=int(stats_payload["n_is"]),
            n_oos=int(stats_payload["n_oos"]),
        )

        # Re-attach a fresh NvidiaLM with the caller's key on the persisted model
        # (lm_factory injection for slow-serving models, mirroring calibrate).
        if lm_factory is not None:
            lm = lm_factory(api_key, baseline.model)
        else:
            lm = NvidiaLM(api_key=api_key, model=baseline.model)

        return cls(calibrator=calibrator, baseline=baseline, lm=lm, stats=stats)

    # -- number-native scoring (task 2.4) -------------------------------------- #

    def score(self, prompt: str) -> FactorScore:
        """Score one factor prompt for ``p_memorized`` number-natively (R1.1, R1.3).

        The version-aware contamination score for THIS prompt and the model's
        emitted factor reasoning, via the released library's public MIA
        primitives — never the directional façade (R6.5):

            ``self._lm.generate(prompt)`` (content + per-token logprobs)
            -> ``compute_mia_features(content, logprobs, None)``  (ref_logprobs
               is fixed at ``None`` — no reference run on the score path, mirroring
               the ``ref_lm=None`` calibration contract; ``ref_delta`` is inert)
            -> ``self._calibrator.predict_proba(features, self._baseline)``
            -> ``FactorScore(p_memorized=<float>, parse_ok=True, fail_reason=None)``.

        No buy/sell ``direction``/``confidence`` parse is performed and no
        directional ``signal`` is ever read (R1.3): the features come from the
        logprobs, so distinct prompts produce distinct features and hence
        distinct ``p_memorized`` (the version-aware property, R1.2).

        Failure handling (R1.5):
          - an **auth-class** ``RuntimeError`` from ``generate`` (HTTP
            401/403/unauthorized/forbidden) ⇒ raise this module's own
            :class:`ConfigurationError` (a rejected credential is a
            configuration fault, not a per-prompt data failure);
          - every other failure (timeout, non-auth ``RuntimeError``, empty
            logprobs, feature computation failure, ``predict_proba`` failure)
            ⇒ ``FactorScore(p_memorized=None, parse_ok=False, fail_reason=…)``
            so the rebalance can fall back to an unadjusted exposure rather
            than crash.

        Args:
            prompt: the factor prompt to score.

        Returns:
            A :class:`FactorScore` for this prompt.

        Raises:
            ConfigurationError: when ``generate`` raises an auth-class
                ``RuntimeError`` (the scoring credential was rejected).
        """
        try:
            completion = self._lm.generate(prompt)
        except TimeoutError:
            return FactorScore(p_memorized=None, parse_ok=False, fail_reason="timeout")
        except RuntimeError as exc:
            if _is_auth_error(exc):
                raise ConfigurationError(
                    "FactorScorer.score: the NIM endpoint rejected the scoring "
                    f"credential while scoring model {self._lm.model!r}: {exc}"
                ) from exc
            return FactorScore(p_memorized=None, parse_ok=False, fail_reason="error")

        return self._score_completion(completion)

    def score_many(
        self, prompts: Sequence[str], *, max_workers: int = 8
    ) -> list[FactorScore]:
        """Score many factor prompts; order-preserving, one result per prompt (R1.1).

        Fans the per-prompt ``generate`` calls out via the library's
        ``generate_many`` (parallel, input-order-preserving), then builds one
        :class:`FactorScore` per prompt through the SAME number-native path as
        :meth:`score`. Per-prompt failures degrade to
        ``p_memorized=None`` independently (R1.5); an auth-class ``RuntimeError``
        from any prompt raises :class:`ConfigurationError` (a rejected credential
        affects the whole batch, not one row).

        Args:
            prompts: the factor prompts to score.
            max_workers: parallelism for the LM calls.

        Returns:
            A list of :class:`FactorScore`, one per input prompt, in input order.

        Raises:
            ConfigurationError: when any ``generate`` raises an auth-class
                ``RuntimeError`` (the scoring credential was rejected).
        """
        results = generate_many(self._lm, list(prompts), max_workers=max_workers)
        scores: list[FactorScore] = []
        for completion in results:
            if isinstance(completion, TimeoutError):
                scores.append(
                    FactorScore(p_memorized=None, parse_ok=False, fail_reason="timeout")
                )
                continue
            if isinstance(completion, RuntimeError):
                if _is_auth_error(completion):
                    raise ConfigurationError(
                        "FactorScorer.score_many: the NIM endpoint rejected the "
                        f"scoring credential while scoring model {self._lm.model!r}: "
                        f"{completion}"
                    ) from completion
                scores.append(
                    FactorScore(p_memorized=None, parse_ok=False, fail_reason="error")
                )
                continue
            if isinstance(completion, BaseException):
                scores.append(
                    FactorScore(p_memorized=None, parse_ok=False, fail_reason="error")
                )
                continue
            scores.append(self._score_completion(completion))
        return scores

    def _score_completion(self, completion: Any) -> FactorScore:
        """Turn a successful ``generate`` result into a :class:`FactorScore`.

        The shared tail of :meth:`score` / :meth:`score_many`:
        ``compute_mia_features(content, logprobs, None)`` ->
        ``predict_proba(features, baseline)``. An empty-logprobs reply, a feature
        computation failure, or a ``predict_proba`` failure all degrade to
        ``p_memorized=None`` (R1.5) — never a crash, never a direction parse.
        """
        logprobs = getattr(completion, "logprobs", None)
        if not logprobs:
            return FactorScore(
                p_memorized=None, parse_ok=False, fail_reason="no_logprobs"
            )

        content = getattr(completion, "content", "")
        try:
            features = compute_mia_features(content, logprobs, None)
        except (ValueError, RuntimeError):
            return FactorScore(p_memorized=None, parse_ok=False, fail_reason="error")

        try:
            p_memorized = float(self._calibrator.predict_proba(features, self._baseline))
        except (ValueError, RuntimeError):
            return FactorScore(p_memorized=None, parse_ok=False, fail_reason="error")

        return FactorScore(p_memorized=p_memorized, parse_ok=True, fail_reason=None)


# --------------------------------------------------------------------------- #
# Task 2.6 — TiltExposure: REGIME_ASSET_EXPOSURE + loadings_to_tilt_views       #
# (Requirements 3.1, 3.2, 3.3, 3.4)                                            #
#                                                                              #
# Map a parsed regime-loadings vector to per-asset DIMENSIONLESS exposure       #
# tilts via a documented, NON-PREDICTIVE axis->category exposure table, then     #
# pack them into MacroView so the UNCHANGED LlmMacroAgent.views_to_bl yields     #
# Q = tilt·conviction/252 (field reinterpretation, no edit to views_to_bl):      #
#                                                                              #
#   tilt(asset) = Σ_axis loadings[axis] · REGIME_ASSET_EXPOSURE[category][axis]  #
#   MacroView(expected_excess_annualized := tilt, confidence := conviction)      #
#   views_to_bl: Q = expected_excess_annualized · clip(confidence,0,1) / 252     #
#              =  tilt · conviction / 252                                        #
#                                                                              #
# The tilt is a dimensionless characterization of the regime's exposure, NOT a   #
# forecast return; conviction is a dimensionless, non-return-bearing scalar      #
# supplied by the caller (it never feeds the tilt itself). No predictive-return  #
# objective and no direction is introduced (R3.4).                              #
# --------------------------------------------------------------------------- #


# Documented, NON-PREDICTIVE heuristic exposure profile: for each anonymized
# asset category, a dimensionless loading on each of the five MACRO_AXES. This is
# a deliberate, hand-specified exposure map (how a category's risk *sits on* each
# macro axis) — NOT a fitted return model and NOT a forecast. The signs encode the
# economic rationale below; the magnitudes are a coarse, bounded heuristic in
# [-1, +1] (a "strong" exposure is ±1.0, a "moderate" one ±0.5, a "mild" one
# ±0.3). Tuning these to any return/eval window is explicitly out of scope (R3.4).
#
# Rationale (per category):
#   - world_equity (broad equity beta): +growth (pro-cyclical risk asset that
#     benefits from expansion); −credit_stress and −inflation (drawdowns when
#     liquidity tightens / inflation erodes real earnings); +risk_appetite
#     (rallies with risk-on); slightly −policy (tighter policy is a headwind).
#   - tech_sector (long-duration growth equity): +growth (high-beta to the cycle);
#     −inflation (long-duration cash flows discount harder under inflation —
#     the "inflation-sensitivity" leg); −credit_stress and −policy (most exposed
#     to tightening / liquidity); +risk_appetite (the high-beta risk-on play).
#   - gold_commodity (real-asset / hedge): +inflation (classic inflation hedge);
#     +credit_stress (a flight-to-safety / systemic-stress hedge); −growth (less
#     attractive when the cycle is strong); −risk_appetite (a risk-off asset);
#     mildly +policy-neutral (small).
#   - short_treasury_cash (defensive cash-like): +policy (rewarded when policy is
#     tight / rates are high); +credit_stress (the defensive safe leg in stress);
#     −growth and −risk_appetite (gives up upside in expansions / risk-on); a
#     mild +inflation carry term (short cash rolls with higher nominal rates).
#
# All values are dimensionless. Each profile names exactly the five MACRO_AXES.
REGIME_ASSET_EXPOSURE: dict[str, dict[str, float]] = {
    "world_equity": {
        "inflation": -0.3,
        "growth": 1.0,
        "credit_stress": -0.8,
        "policy": -0.3,
        "risk_appetite": 0.6,
    },
    "tech_sector": {
        "inflation": -0.6,
        "growth": 1.0,
        "credit_stress": -0.7,
        "policy": -0.5,
        "risk_appetite": 0.8,
    },
    "gold_commodity": {
        "inflation": 0.8,
        "growth": -0.4,
        "credit_stress": 0.7,
        "policy": 0.1,
        "risk_appetite": -0.5,
    },
    "short_treasury_cash": {
        "inflation": 0.3,
        "growth": -0.6,
        "credit_stress": 0.6,
        "policy": 0.8,
        "risk_appetite": -0.6,
    },
}


def _finite_or_zero(value: float) -> float:
    """Return ``value`` if finite, else ``0.0`` (guard non-finite loadings).

    Non-standard JSON ``NaN`` can bypass the parser's clip (``Infinity`` is
    clipped to ±1 by ``_clip_unit``; an ``Inf`` here can only come from a
    manually built ``RegimeLoadings``). Treating a non-finite loading as ``0``
    keeps it inert in
    the dot-product so no ``NaN``/``Inf`` tilt propagates into the unchanged
    Black-Litterman conversion (R3.1).
    """
    return float(value) if math.isfinite(value) else 0.0


def loadings_to_tilt_views(
    loadings: RegimeLoadings,
    asset_snapshot: list[dict[str, object]],
    asset_map: AssetMap,
    conviction: float,
) -> list[MacroView]:
    """Map regime loadings to per-asset dimensionless exposure tilt views (R3.1–R3.4).

    For each asset in ``asset_snapshot``, computes the dimensionless tilt

        ``tilt = Σ_axis loadings[axis] · REGIME_ASSET_EXPOSURE[category][axis]``

    over the named ``MACRO_AXES`` (the documented, non-predictive exposure table),
    and packs it into a :class:`MacroView` by FIELD REINTERPRETATION so the
    UNCHANGED :meth:`LlmMacroAgent.views_to_bl` yields ``Q = tilt·conviction/252``:

      - ``expected_excess_annualized := tilt`` — a dimensionless exposure tilt,
        NOT a forecast return (R3.1);
      - ``confidence := conviction`` — the dimensionless, non-return-bearing
        conviction supplied by the caller; it scales the BL view magnitude via
        the unchanged conversion but NEVER feeds the tilt itself (R3.2);
      - ``asset_short = None`` — a single-leg exposure, no direction (R3.4).

    Non-finite axis loadings (``NaN`` can slip past the parser's clip via
    non-standard JSON; ``Inf`` only via a manually built ``RegimeLoadings``)
    are treated as ``0`` so no ``NaN`` tilt reaches
    BL (tasks.md 2.2 note). An asset whose category is absent from
    ``REGIME_ASSET_EXPOSURE`` or unmapped to a pseudo id is skipped (no view).

    No predictive-return objective is introduced: the tilt is a pure
    characterization of the regime's exposure (R3.4), and the conversion is the
    existing one reused verbatim (R3.3).

    Args:
        loadings: the parsed regime-loadings vector (axis -> loading).
        asset_snapshot: anonymized asset descriptors (``id`` pseudo letter +
            ``category``), e.g. ``AssetMap.pseudo_assets()``.
        asset_map: the identifying <-> anonymized asset map (used to confirm the
            pseudo id is mapped; ``views_to_bl`` resolves it to a real ticker).
        conviction: a dimensionless, non-return-bearing conviction in ``[0, 1]``
            packed as each view's ``confidence``.

    Returns:
        One :class:`MacroView` per mapped asset, in ``asset_snapshot`` order.
    """
    views: list[MacroView] = []
    for asset in asset_snapshot:
        pseudo = asset.get("id")
        category = asset.get("category")
        if category is None:
            category = asset_map.categories.get(str(pseudo)) if pseudo is not None else None
        if category is None or category not in REGIME_ASSET_EXPOSURE:
            continue
        if pseudo is None or str(pseudo) not in asset_map.pseudo_to_real:
            continue

        exposure = REGIME_ASSET_EXPOSURE[category]
        tilt = 0.0
        contributing: list[str] = []
        for axis, raw_loading in loadings.loadings.items():
            axis_exposure = exposure.get(axis)
            if axis_exposure is None:
                continue
            loading = _finite_or_zero(float(raw_loading))
            contribution = loading * float(axis_exposure)
            tilt += contribution
            if abs(contribution) > 1e-9:
                contributing.append(axis)

        # Guard once more: an all-non-finite vector would already be 0 above, but
        # keep the tilt finite under any arithmetic edge (e.g. 0·inf upstream).
        tilt = _finite_or_zero(tilt)

        axes_text = ", ".join(contributing) if contributing else "the macro axes"
        rationale = (
            f"Dimensionless exposure tilt for {pseudo} ({category}) from the "
            f"regime loadings on {axes_text} via the documented "
            f"REGIME_ASSET_EXPOSURE table; not a return forecast."
        )

        views.append(
            MacroView(
                asset_long=str(pseudo),
                asset_short=None,
                expected_excess_annualized=tilt,
                confidence=float(conviction),
                rationale=rationale,
            )
        )
    return views


# --------------------------------------------------------------------------- #
# Task 2.7 — RecallGuardedAdjust: RecallGuardedConfig + recall_guarded_adjust                      #
# (Requirements 4.1, 4.2, 4.3, 4.4)                                            #
#                                                                              #
# Down-weight each view's dimensionless exposure tilt by its measured           #
# contamination so recall-tainted reasoning is discounted while genuine         #
# inference is retained:                                                        #
#                                                                              #
#   adjusted_tilt = expected_excess_annualized · (1 − p_memorized)              #
#                                                                              #
# This mirrors ONLY the *discount* limb of steering.steer_views                 #
# (`raw·(1−p_mem)`) and is a RE-IMPLEMENTATION — steering.py is never imported   #
# or edited (R6.1/R6.5). The key difference from steer_views: there is NO hard  #
# exclusion gate — R4 is magnitude-only, so views are down-weighted, never      #
# dropped. The adjustment changes ONLY the exposure magnitude                   #
# (expected_excess_annualized); asset_long / asset_short / confidence /         #
# rationale are copied through unchanged (4.4), and no return/forecast          #
# objective is ever introduced (the discount scales the existing tilt only).    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RecallGuardedConfig:
    """Configuration for the recall-guarded (contamination-discount) adjustment (R4).

    Attributes:
        enabled: when ``False``, :func:`recall_guarded_adjust` is a passthrough that
            returns the input views UNCHANGED regardless of ``p_memorized``
            (the off switch; the weak-calibrator fallback is supplied instead by
            the caller as ``p_memorized=None``) (R4.3).
    """

    enabled: bool = True


def recall_guarded_adjust(
    views: list[MacroView],
    p_memorized: float | None,
    config: RecallGuardedConfig = RecallGuardedConfig(),
) -> list[MacroView]:
    """Down-weight exposure tilts by measured contamination — magnitude-only (R4).

    Returns a **new** ``list[MacroView]`` of **new** ``MacroView`` objects; the
    input list and its views are never mutated. Only the exposure magnitude
    (``expected_excess_annualized``) is changed — ``asset_long``,
    ``asset_short``, ``confidence`` and ``rationale`` are copied through
    unchanged (R4.4, magnitude-only; no return/forecast objective is introduced).

    Behaviour:

    * **Passthrough** (R4.3): if ``not config.enabled`` OR ``p_memorized is
      None``, the input views are returned UNCHANGED (returned as-is). The
      composition supplies ``p_memorized=None`` when the calibrator ``is_weak``,
      so the weak-calibrator case degrades here to the unadjusted raw exposure.
    * **Discount** (R4.1, R4.2): otherwise each view's tilt becomes
      ``expected_excess_annualized · (1 − clip(p_memorized, 0, 1))``. Higher
      ``p_memorized`` ⇒ a lower-or-equal adjusted tilt (R4.1, monotone);
      ``p_memorized == 0`` ⇒ the adjusted tilt equals the raw tilt (R4.2).

    This mirrors ONLY the *discount* limb of ``steering.steer_views`` and is a
    re-implementation, NOT an import/edit of ``steering.py``. Unlike
    ``steer_views``, there is **NO hard exclusion gate**: R4 is magnitude-only,
    so a high ``p_memorized`` down-weights the views but never drops them (the
    returned list is never made empty by the discount).

    Pure and deterministic: equal inputs ⇒ equal output (no randomness, no I/O).

    Args:
        views: the exposure-tilt views for ONE rebalance (must share that
            rebalance's ``p_memorized``).
        p_memorized: the measured contamination score in ``[0, 1]`` for this
            rebalance, or ``None`` when scoring failed / the calibrator is weak /
            the adjustment is off.
        config: the recall-guarded configuration (the ``enabled`` flag).

    Returns:
        Passthrough — the input views unchanged (disabled / ``p_memorized``
        ``None``); otherwise a new list of new views with the tilt magnitude
        discounted by ``(1 − p_memorized)``.
    """
    # Passthrough (R4.3): leave the raw exposure unadjusted rather than apply an
    # unvalidated discount (the weak-calibrator path arrives here as None).
    if not config.enabled or p_memorized is None:
        return views

    # Clip p_memorized into [0, 1] so the discount factor stays in [0, 1]
    # (a calibrated probability is already bounded; this is a defensive guard).
    p_clipped = min(1.0, max(0.0, float(p_memorized)))
    discount = 1.0 - p_clipped

    adjusted: list[MacroView] = []
    for view in views:
        # Magnitude-only (R4.4): scale the tilt; copy every other field unchanged.
        # NO hard gate — the view is always retained, only down-weighted (R4.1).
        adjusted_tilt = view.expected_excess_annualized * discount
        adjusted.append(
            MacroView(
                asset_long=view.asset_long,
                asset_short=view.asset_short,
                expected_excess_annualized=adjusted_tilt,
                confidence=view.confidence,
                rationale=view.rationale,
            )
        )
    return adjusted


# --------------------------------------------------------------------------- #
# Task 2.8 — FactorStability: factor_stability                                 #
# (Requirement 5.2)                                                            #
#                                                                              #
# Summarize the variability of ONE prompt version's regime loadings across the #
# point-in-time stream so refinement (nb14) can compare versions by factor      #
# stability (R5.2). Two complementary per-axis summaries are reported:          #
#                                                                              #
#   <axis>_std — population standard deviation of the axis's loading across the #
#                stream (overall spread of the axis around its own mean);       #
#   <axis>_mac — mean absolute month-to-month change of the axis's loading over #
#                the DATE-SORTED stream (run-to-run jitter of the PIT series).  #
#                                                                              #
# plus the overall summaries mean_std / mean_mac (the mean per-axis std /       #
# mean-abs-change across MACRO_AXES). Only parsed loadings count (parse_ok      #
# entries with the full axis vector); not-parsed entries are skipped. Empty /   #
# single-entry / all-not-parsed streams degrade to a well-defined all-zero      #
# summary (no crash, no NaN). Pure + deterministic; no I/O.                     #
# --------------------------------------------------------------------------- #


def _population_std(values: list[float]) -> float:
    """Population standard deviation of ``values`` (0.0 for <2 entries).

    Population (not sample) so a single observation is a well-defined ``0.0``
    rather than a ``ZeroDivisionError`` / ``NaN`` — the spread of one point is
    zero. Deterministic; no I/O.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(variance)


def _mean_abs_change(values: list[float]) -> float:
    """Mean absolute consecutive change of an ORDERED sequence (0.0 for <2).

    The caller passes the date-sorted axis series; with fewer than two points
    there is no transition, so the change is a well-defined ``0.0``.
    Deterministic; no I/O.
    """
    if len(values) < 2:
        return 0.0
    deltas = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    return sum(deltas) / len(deltas)


def factor_stability(
    loadings_by_date: dict[pd.Timestamp, RegimeLoadings],
) -> dict[str, float]:
    """Summarize a prompt version's loadings variability across the PIT stream (R5.2).

    The factor-stability metric: how much the version's regime-loadings factor
    moves across the point-in-time rebalance stream. For each named
    :data:`MACRO_AXES` axis it reports two complementary summaries, plus two
    overall collapses:

    - ``<axis>_std`` — the population standard deviation of that axis's loading
      across the stream (the overall spread around the axis's own mean);
    - ``<axis>_mac`` — the mean absolute month-to-month change of that axis's
      loading over the DATE-SORTED stream (the run-to-run jitter of the series);
    - ``mean_std`` / ``mean_mac`` — the mean per-axis ``std`` / ``mac`` across
      ``MACRO_AXES`` (single-number overall stability summaries).

    Only parsed loadings count: entries with ``parse_ok=False`` are skipped (a
    not-parsed rebalance falls back to the base allocation, so it carries no
    factor vector to measure). An entry that, despite ``parse_ok=True``, is
    missing an axis contributes nothing to that axis's series.

    Empty, single-entry, or all-not-parsed streams degrade to a well-defined
    all-zero summary — every key present, every value ``0.0`` — never a crash and
    never a ``NaN`` explosion (population std + the <2-point change guard).

    Pure and deterministic: equal inputs yield an identical dict; the
    month-to-month change is computed over the chronological (date-sorted)
    sequence, so the dict's insertion order does not affect the result. No I/O.

    Args:
        loadings_by_date: the per-rebalance regime-loadings stream for ONE prompt
            version, keyed by ``rebalance_date`` (e.g. the artifact emitted by
            :func:`parse_loadings`).

    Returns:
        A flat ``dict[str, float]`` with ``<axis>_std`` and ``<axis>_mac`` for
        every :data:`MACRO_AXES` axis plus the overall ``mean_std`` / ``mean_mac``.
    """
    # Consider only parsed entries (R5.2: a not-parsed rebalance carries no
    # factor vector). Sort by rebalance date so the month-to-month change is the
    # chronological run-to-run jitter, independent of dict insertion order.
    parsed = [
        rl for rl in loadings_by_date.values() if getattr(rl, "parse_ok", False)
    ]
    parsed.sort(key=lambda rl: rl.rebalance_date)

    summary: dict[str, float] = {}
    per_axis_std: list[float] = []
    per_axis_mac: list[float] = []

    for axis in MACRO_AXES:
        # The axis's date-ordered series across the parsed stream; an entry that
        # omits this axis contributes nothing to it (never a fabricated value).
        series = [
            float(rl.loadings[axis]) for rl in parsed if axis in rl.loadings
        ]
        axis_std = _population_std(series)
        axis_mac = _mean_abs_change(series)
        summary[f"{axis}_std"] = axis_std
        summary[f"{axis}_mac"] = axis_mac
        per_axis_std.append(axis_std)
        per_axis_mac.append(axis_mac)

    # Overall single-number summaries: the mean per-axis std / mean-abs-change.
    n_axes = len(MACRO_AXES)
    summary["mean_std"] = sum(per_axis_std) / n_axes if n_axes else 0.0
    summary["mean_mac"] = sum(per_axis_mac) / n_axes if n_axes else 0.0

    return summary


# --------------------------------------------------------------------------- #
# Task 2.9 — ContrastHarness: ContrastResult + run_pit_vs_nonpit_contrast       #
# (Requirements 7.2, 7.3, 7.5)                                                 #
#                                                                              #
# The OFFLINE PIT-vs-non-PIT contrast computation (the LIVE full-stream run is  #
# nb14, task 4.2). ContrastResult holds the paired per-variant p_memorized      #
# streams + the per-variant head-to-head metrics; contamination_premium()       #
# reports the non-PIT − PIT gap — per p_memorized stat AND per metric — WITH a   #
# paired effect size over n_pairs (R7.5: the contamination premium read over     #
# the stream, NOT a single-date point estimate; research.md: n=3 was noisy with  #
# a reversal, so the premium MUST be a distributional/paired summary).           #
#                                                                              #
# run_pit_vs_nonpit_contrast delegates scoring to the injected scorer's          #
# score_many for the supplied PIT and non-PIT prompts (which come from the SAME  #
# renderer in both modes — R7.6 — supplied by nb14; this function does not       #
# render), pairs the per-variant p_memorized BY INDEX, drops a pair only when    #
# either side is None (keeping the remaining pairing intact), and attaches the   #
# caller-supplied head-to-head metrics. Pure aside from the delegated scoring;   #
# no direct network.                                                            #
# --------------------------------------------------------------------------- #


def _median(values: list[float]) -> float:
    """Median of ``values`` (0.0 for an empty list).

    Deterministic; no I/O. An empty stream has no defined median, so it degrades
    to a well-defined ``0.0`` rather than raising — the zero-``n_pairs`` premium
    case (research.md: the premium is reported over the stream, never a point).
    """
    n = len(values)
    if n == 0:
        return 0.0
    ordered = sorted(values)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return (float(ordered[mid - 1]) + float(ordered[mid])) / 2.0


def _paired_cohens_d(deltas: list[float]) -> float:
    """Cohen's d for paired samples: ``mean(deltas) / population_std(deltas)``.

    The standardized paired effect size of the non-PIT − PIT gap OVER the stream
    (R7.5) — so the contamination premium is read distributionally, not as a
    single-date point estimate (research.md: n=3 was noisy with a reversal).

    Degrades to a well-defined ``0.0`` (never ``NaN`` / ``ZeroDivisionError``)
    when fewer than two pairs back the deltas OR the paired deltas have zero
    variance (a constant gap has no standardized magnitude to report).
    """
    if len(deltas) < 2:
        return 0.0
    std = _population_std(deltas)
    # A tolerance guard (not an exact == 0.0): a constant gap can leave a tiny
    # float-residual std, which would otherwise blow the ratio up to ~1e16.
    if std <= 1e-12:
        return 0.0
    mean = sum(deltas) / len(deltas)
    return mean / std


@dataclass(frozen=True)
class ContrastResult:
    """The offline PIT-vs-non-PIT contrast result (R7.2, R7.3, R7.5).

    Holds the two paired per-variant ``p_memorized`` streams and the per-variant
    head-to-head metrics so the non-PIT − PIT gap can be reported as the
    contamination premium OVER the rebalance stream (not a single-date point
    estimate; research.md: n=3 was noisy with a reversal). Frozen so a computed
    contrast can be safely logged/serialized.

    Attributes:
        pit_p_memorized: the point-in-time variant's per-pair ``p_memorized``
            (anonymized / z-scored / dateless framing).
        nonpit_p_memorized: the non-point-in-time variant's per-pair
            ``p_memorized``, index-aligned with ``pit_p_memorized`` (the
            recall-enabling framing: real tickers + date + raw levels).
        pit_metrics: the PIT variant's head-to-head metrics (caller-supplied).
        nonpit_metrics: the non-PIT variant's head-to-head metrics
            (caller-supplied), keyed identically to ``pit_metrics``.
        n_pairs: the number of valid index-paired ``p_memorized`` pairs backing
            the premium (the stream length the paired effect size is read over).
    """

    pit_p_memorized: list[float]
    nonpit_p_memorized: list[float]
    pit_metrics: dict[str, float]
    nonpit_metrics: dict[str, float]
    n_pairs: int

    def contamination_premium(self) -> dict[str, float]:
        """Report the non-PIT − PIT gap as the contamination premium (R7.2, R7.3, R7.5).

        Frames the non-PIT − PIT difference as lookahead/recall bias — NOT
        attainable skill (R7.5) — and reports it OVER the stream rather than as a
        single-date point estimate (research.md: n=3 was noisy with a reversal).
        The returned flat ``dict[str, float]`` carries:

        - ``n_pairs`` — the number of valid index-paired ``p_memorized`` pairs.
        - ``p_memorized_mean_delta`` / ``p_memorized_median_delta`` — the
          per-``p_memorized``-stat gap (non-PIT − PIT) (R7.2).
        - ``p_memorized_paired_d`` — Cohen's d for paired samples
          (``mean(non-PIT − PIT) / std(non-PIT − PIT)`` over ``n_pairs``): the
          standardized paired effect size, so the premium is read over the
          stream, not a single point (R7.5).
        - ``<metric>_delta`` for every metric present in BOTH ``pit_metrics`` and
          ``nonpit_metrics`` — the per-metric head-to-head gap (non-PIT − PIT)
          (R7.3).

        Small / zero ``n_pairs`` and zero-variance paired deltas degrade to a
        well-defined output (mean/median/effect size ``0.0``) — never ``NaN`` or
        a ``ZeroDivisionError`` crash. Pure and deterministic; no I/O.

        Returns:
            The contamination-premium summary as a flat ``dict[str, float]``.
        """
        # Pair p_memorized by index over whatever valid pairs survived (the
        # construction already dropped any pair with a None on either side).
        n = min(len(self.pit_p_memorized), len(self.nonpit_p_memorized))
        paired_deltas = [
            float(self.nonpit_p_memorized[i]) - float(self.pit_p_memorized[i])
            for i in range(n)
        ]

        # Per-p_memorized-stat gap (non-PIT − PIT). Empty streams -> 0.0 (R7.5:
        # reported over the stream; a no-pair stream carries no premium).
        pit_mean = (
            sum(self.pit_p_memorized) / len(self.pit_p_memorized)
            if self.pit_p_memorized
            else 0.0
        )
        nonpit_mean = (
            sum(self.nonpit_p_memorized) / len(self.nonpit_p_memorized)
            if self.nonpit_p_memorized
            else 0.0
        )
        premium: dict[str, float] = {
            "n_pairs": float(self.n_pairs),
            "p_memorized_mean_delta": nonpit_mean - pit_mean,
            "p_memorized_median_delta": _median(list(self.nonpit_p_memorized))
            - _median(list(self.pit_p_memorized)),
            # Paired effect size OVER the stream (0.0 for <2 pairs / zero variance).
            "p_memorized_paired_d": _paired_cohens_d(paired_deltas),
        }

        # Per-metric head-to-head gap (non-PIT − PIT), keyed "<metric>_delta",
        # for every metric reported by BOTH variants (R7.3).
        for metric in sorted(set(self.pit_metrics) & set(self.nonpit_metrics)):
            premium[f"{metric}_delta"] = float(self.nonpit_metrics[metric]) - float(
                self.pit_metrics[metric]
            )

        return premium


def run_pit_vs_nonpit_contrast(
    scorer: FactorScorer,
    pit_prompts: Sequence[str],
    nonpit_prompts: Sequence[str],
    *,
    pit_metrics: dict[str, float],
    nonpit_metrics: dict[str, float],
) -> ContrastResult:
    """Compute the offline PIT-vs-non-PIT contrast (R7.2, R7.3, R7.5).

    Scores the supplied point-in-time and non-point-in-time prompts via the
    injected ``scorer.score_many`` and pairs the per-variant ``p_memorized`` BY
    INDEX so the resulting :class:`ContrastResult` can report the non-PIT − PIT
    gap as the contamination premium over the stream. The PIT and non-PIT prompts
    come from the SAME renderer in both modes (the identifying flag — R7.6,
    "hold all else equal") and are supplied by the caller (nb14); this function
    does NOT render.

    A pair is dropped ONLY when either side's ``p_memorized`` is ``None`` (a
    failed score on that date); the surviving pairs keep their index pairing
    intact (PIT_i is always matched with non-PIT_i). ``n_pairs`` is the number of
    valid pairs that backs the paired effect size.

    The provided head-to-head ``pit_metrics`` / ``nonpit_metrics`` (computed
    elsewhere — the non-PIT variant is a diagnostic control, never the deployed
    portfolio, R7.4) are attached verbatim for the per-metric premium (R7.3).

    Pure aside from delegating scoring to the injected ``scorer``; no direct
    network. Deterministic for a deterministic scorer.

    Args:
        scorer: the (already-calibrated) :class:`FactorScorer`; only its
            ``score_many`` is used here.
        pit_prompts: the point-in-time (anonymized / z-scored / dateless) prompts.
        nonpit_prompts: the non-point-in-time (recall-enabling) prompts,
            index-aligned with ``pit_prompts``.
        pit_metrics: the PIT variant's head-to-head metrics.
        nonpit_metrics: the non-PIT variant's head-to-head metrics.

    Returns:
        A :class:`ContrastResult` with the paired ``p_memorized`` streams,
        ``n_pairs``, and the attached head-to-head metrics.
    """
    pit_scores = scorer.score_many(list(pit_prompts))
    nonpit_scores = scorer.score_many(list(nonpit_prompts))

    # Pair p_memorized by index; drop a pair only when EITHER side is None (a
    # failed score), keeping the rest of the pairing intact (R7.2).
    pit_p: list[float] = []
    nonpit_p: list[float] = []
    n = min(len(pit_scores), len(nonpit_scores))
    for i in range(n):
        pit_pm = pit_scores[i].p_memorized
        nonpit_pm = nonpit_scores[i].p_memorized
        if pit_pm is None or nonpit_pm is None:
            continue
        pit_p.append(float(pit_pm))
        nonpit_p.append(float(nonpit_pm))

    return ContrastResult(
        pit_p_memorized=pit_p,
        nonpit_p_memorized=nonpit_p,
        pit_metrics=dict(pit_metrics),
        nonpit_metrics=dict(nonpit_metrics),
        n_pairs=len(pit_p),
    )


# --------------------------------------------------------------------------- #
# Task 3.1 — Integration: steered factor weight-fn composition for walk-forward #
# (Requirements 1.4, 3.1, 3.2, 3.3, 4.3)                                       #
#                                                                              #
# Compose the finished pieces — render_regime_loadings_prompt (2.1),            #
# parse_loadings (2.2), FactorScorer.score (2.4), loadings_to_tilt_views (2.6), #
# recall_guarded_adjust (2.7) — plus the agent's UNCHANGED views_to_bl into one        #
# network-free, walk-forward-compatible factor decision step. The "Factor       #
# rebalance" sequence (design System Flows):                                    #
#                                                                              #
#   render anonymized PIT prompt (R1.4) → generate loadings (injected) → parse  #
#   → score the SAME prompt for p_memorized → loadings→tilt views → recall-guard #
#   adjust → UNCHANGED views_to_bl → HRP+BL blend (injected via `combine`).      #
#                                                                              #
# Gating fallbacks (design gating note): a loadings parse-fail ⇒ (None, None)   #
# ⇒ the caller's `combine` falls back to the base allocation; p_memorized       #
# unavailable / scorer.is_weak / scorer is None ⇒ the recall-guarded step passes       #
# through (unadjusted exposures) but P/Q are still produced from the raw tilt.   #
# No directional signal is read and no return objective is introduced (R3.4).    #
#                                                                              #
# Mirrors the predecessor steering.steer_rebalance / make_steered_weight_fn      #
# SHAPE (read for the pattern) — but steering.py is never imported or edited     #
# (R6.1/R6.5); this is the factor analogue. The composition owns NEITHER the     #
# LLM call (injected `generate_loadings`) NOR nb09's HRP/BL/blend (injected      #
# `combine`), keeping it agent-type-agnostic and unit-testable with mocks.       #
# --------------------------------------------------------------------------- #


def _conviction_from_loadings(loadings: RegimeLoadings) -> float:
    """A dimensionless, non-return-bearing conviction in ``[0, 1]`` from the loadings.

    The clipped L2 norm of the parsed loading vector, normalized by the maximum
    possible norm (``sqrt(len(MACRO_AXES))`` when every axis sits at ``±1``), so a
    flat/neutral regime yields a near-zero conviction and a saturated regime
    approaches ``1.0``. Non-finite loadings are treated as ``0`` (inert), mirroring
    the tilt step's guard, so no ``NaN`` conviction can scale the BL view.

    This scalar feeds ``loadings_to_tilt_views`` as the dimensionless ``confidence``
    that the UNCHANGED ``views_to_bl`` uses for ``Q = tilt·conviction/252`` (R3.2);
    it NEVER feeds the tilt itself and carries no return semantics (R3.4).
    """
    axes = MACRO_AXES
    sq_sum = 0.0
    for axis in axes:
        value = loadings.loadings.get(axis)
        if value is None:
            continue
        v = _finite_or_zero(float(value))
        sq_sum += v * v
    norm = math.sqrt(sq_sum)
    max_norm = math.sqrt(len(axes)) if axes else 1.0
    if max_norm <= 0.0:
        return 0.0
    conviction = norm / max_norm
    # Defensive clip — the construction already bounds this to [0, 1].
    return min(1.0, max(0.0, conviction))


@dataclass(frozen=True)
class FactorDecision:
    """The outcome of one factor rebalance decision (R3.1–R3.3, R4.3).

    Frozen so a decision is a stable value the (later) nb13 decision log can
    persist verbatim. It carries the UNCHANGED ``views_to_bl`` output
    (``P``/``Q``, ``(None, None)`` when the loadings failed to parse so the
    caller's ``combine`` falls back to the base allocation), the recall-guarded
    ``views`` handed to ``views_to_bl`` (``[]`` on the parse-fail path), the
    parsed ``loadings`` (``None`` on parse fail), the measured ``p_memorized``
    (``None`` on the weak / no-scorer / scoring-failure path — R4.3), a
    ``parse_ok`` flag, and a ``steered`` flag that is ``True`` whenever a score
    was measured (``p_memorized is not None``) — the adjustment itself can still
    be a passthrough when ``RecallGuardedConfig.enabled`` is ``False``.

    Attributes:
        P, Q: the ``views_to_bl`` BL view pair, or ``(None, None)`` when no view
            survived (parse fail) — the caller's ``combine`` then falls back to
            the base allocation, exactly as Track A (R3.2).
        views: the (recall-guarded or unadjusted) exposure-tilt views handed to
            ``views_to_bl``; ``[]`` on the parse-fail path.
        loadings: the parsed :class:`RegimeLoadings`, or ``None`` on parse fail.
        p_memorized: the measured contamination score in ``[0, 1]`` for this
            rebalance, or ``None`` when scoring was unavailable / weak / failed
            (R4.3) — the recall-guarded adjustment then passes through (unadjusted).
        parse_ok: whether the loadings reply yielded the full factor vector.
        steered: ``True`` iff ``p_memorized is not None`` — a measured score
            actually drove the recall-guard discount; ``False`` on every fallback.
    """

    P: pd.DataFrame | None
    Q: pd.DataFrame | None
    views: list[MacroView]
    loadings: RegimeLoadings | None
    p_memorized: float | None
    parse_ok: bool
    steered: bool


def factor_rebalance(
    *,
    generate_loadings: Callable[[str], str],
    scorer: object | None,
    agent: object,
    macro_state: dict[str, float],
    asset_snapshot: list[dict[str, object]],
    real_symbols: list[str],
    as_of: Any | None = None,
    raw_levels: dict[str, float] | None = None,
    recall_guarded_config: RecallGuardedConfig = RecallGuardedConfig(),
) -> FactorDecision:
    """Compose one factor rebalance decision (design "Factor rebalance" flow).

    Wires the finished pieces — :func:`render_regime_loadings_prompt` (2.1),
    :func:`parse_loadings` (2.2), :meth:`FactorScorer.score` (2.4),
    :func:`loadings_to_tilt_views` (2.6), :func:`recall_guarded_adjust` (2.7) — plus the
    agent's UNCHANGED ``views_to_bl`` into one network-free step. The composition
    owns NEITHER the LLM call (injected ``generate_loadings``) NOR the HRP/BL
    blend (the caller's ``combine``), so it is unit-testable with mocks and stays
    agent-type-agnostic (``agent`` need only expose ``views_to_bl`` + ``asset_map``).

    Steps:

    1. ``prompt = render_regime_loadings_prompt(macro_state, asset_snapshot)`` —
       the anonymized, point-in-time renderer (no date / no real ticker; R1.4).
    2. ``reply = generate_loadings(prompt)`` — the injected loadings-producing
       callable (the live agent / a replay); the composition does NOT own it.
    3. ``loadings = parse_loadings(reply, as_of)``; on ``None`` (a parse failure)
       return a decision with ``P=Q=None``, ``parse_ok=False``, ``steered=False``
       so the caller's ``combine`` falls back to the base allocation (R3.2).
    4. ``p_memorized = None``; only when ``scorer is not None and not
       scorer.is_weak`` is the SAME anonymized prompt scored
       (``scorer.score(prompt).p_memorized`` — ``None`` on a scoring failure).
       The weak / no-scorer / failure cases keep ``p_memorized = None`` so the
       recall-guarded step passes through (unadjusted exposures; R4.3).
    5. ``conviction`` is the dimensionless clipped-L2 norm of the loadings
       (``[0, 1]``, non-return-bearing); ``views = loadings_to_tilt_views(...)``.
    6. ``adjusted = recall_guarded_adjust(views, p_memorized, recall_guarded_config)`` — a
       passthrough when ``p_memorized`` is ``None`` (R4.3); otherwise the tilt is
       discounted by ``(1 − p_memorized)`` (R4.1).
    7. ``P, Q = agent.views_to_bl(adjusted, real_symbols)`` — the UNCHANGED method.
    8. ``steered = p_memorized is not None``.

    No directional ``signal`` is ever read and no return objective is introduced
    (R3.4): the tilt is a pure exposure characterization and the conversion is the
    existing one reused verbatim (R3.3).

    Args:
        generate_loadings: injected callable mapping the rendered prompt to a
            loadings reply (the LLM / agent / replay).
        scorer: the calibrated :class:`FactorScorer` (or any object exposing
            ``is_weak`` + ``score``); ``None`` disables scoring (passthrough).
        agent: any object exposing the UNCHANGED ``views_to_bl`` + ``asset_map``.
        macro_state: the z-scored macro state for this rebalance (PIT).
        asset_snapshot: the anonymized asset snapshot for this rebalance.
        real_symbols: the real tickers ``views_to_bl`` keys ``P``/``Q`` on.
        as_of: the rebalance date used to key the parsed :class:`RegimeLoadings`
            (carried into the artifact; the anonymized prompt discloses no date).
        raw_levels: accepted for signature parity with the renderer's identifying
            form; unused on the anonymized PIT path (kept ``None``).
        recall_guarded_config: the recall-guarded (contamination-discount) configuration.

    Returns:
        A :class:`FactorDecision` carrying the (possibly ``(None, None)``) BL view
        pair, the adjusted views, the parsed loadings, ``p_memorized``,
        ``parse_ok`` and ``steered``.
    """
    # 1. The anonymized, point-in-time renderer (no date / no real ticker; R1.4).
    prompt = render_regime_loadings_prompt(macro_state, asset_snapshot)

    # 2. The injected loadings-producing callable (the composition does not own it).
    reply = generate_loadings(prompt)

    # 3. Parse; a parse failure falls back to (None, None) so combine → base (R3.2).
    loadings = parse_loadings(reply, as_of)
    if loadings is None:
        return FactorDecision(
            P=None,
            Q=None,
            views=[],
            loadings=None,
            p_memorized=None,
            parse_ok=False,
            steered=False,
        )

    # 4. Measure p_memorized on the SAME anonymized prompt, but only when a scorer
    #    is present and the calibrator is not weak. Each gate degrades to the
    #    unadjusted (p_memorized=None) path so the factor exposure stays honest
    #    rather than carrying an unvalidated discount (R4.3 / R1.6).
    p_memorized: float | None = None
    if scorer is not None and not scorer.is_weak:
        score = scorer.score(prompt)
        # None on a scoring failure ⇒ recall_guarded_adjust passes through (R4.3).
        p_memorized = score.p_memorized

    # 5. Dimensionless conviction (clipped L2 norm) → per-asset exposure tilts.
    conviction = _conviction_from_loadings(loadings)
    views = loadings_to_tilt_views(loadings, asset_snapshot, agent.asset_map, conviction)

    # 6. Down-weight the exposure by the measured contamination (passthrough on None).
    adjusted = recall_guarded_adjust(views, p_memorized, recall_guarded_config)

    # 7. The UNCHANGED views_to_bl conversion (Q = tilt·conviction/252).
    P, Q = agent.views_to_bl(adjusted, real_symbols)

    # 8. Steered iff a score was measured; with the guard disabled the adjustment
    #    is a passthrough but the measurement is still recorded as steered.
    steered = p_memorized is not None

    return FactorDecision(
        P=P,
        Q=Q,
        views=list(adjusted),
        loadings=loadings,
        p_memorized=p_memorized,
        parse_ok=True,
        steered=steered,
    )


def make_factor_weight_fn(
    *,
    generate_loadings: Callable[[str], str],
    scorer: object | None,
    agent: object,
    build_inputs: Callable[[dict], tuple[dict[str, float], list[dict[str, object]], Any, dict[str, float] | None]],
    combine: Callable[[dict, pd.DataFrame | None, pd.DataFrame | None], pd.Series],
    recall_guarded_config: RecallGuardedConfig = RecallGuardedConfig(),
) -> Callable[[dict], pd.Series]:
    """Adapt :func:`factor_rebalance` into a ``walk_forward`` ``weight_fn(ctx) -> pd.Series``.

    Returns a closure holding ONE agent instance and the scorer, so the same agent
    runs the UNCHANGED ``views_to_bl`` for each rebalance. nb09's HRP / BL / blend
    math is INJECTED via ``combine`` rather than owned here (R3.3 / R6.1) and the
    LLM/replay is injected via ``generate_loadings`` — keeping the composition
    agent-type-agnostic. Mirrors the SHAPE of the predecessor
    ``steering.make_steered_weight_fn`` (read for the pattern; steering.py is never
    imported or edited).

    The weight_fn:

    * sources ``real_symbols`` from ``list(ctx["prices"].columns)`` (the real
      tickers, exactly as the Track A weight_fn in nb09);
    * derives ``(macro_state, asset_snapshot, as_of, raw_levels)`` from the
      injected ``build_inputs(ctx)`` (the PIT-sliced ctx → renderer inputs);
    * calls :func:`factor_rebalance`;
    * returns ``combine(ctx, dec.P, dec.Q)`` — the injected blend, which falls
      back to the base allocation on the ``(None, None)`` parse-fail path (R3.2).

    Args:
        generate_loadings: injected callable mapping the rendered prompt to a
            loadings reply (the LLM / agent / replay).
        scorer: the calibrated :class:`FactorScorer` (or ``None`` to disable
            scoring — the recall-guarded passthrough path, R4.3).
        agent: the agent instance reused across rebalances (only ``views_to_bl``
            + ``asset_map`` are used).
        build_inputs: injected; maps a walk-forward ctx to
            ``(macro_state, asset_snapshot, as_of, raw_levels)``.
        combine: injected; maps ``(ctx, P, Q)`` to a target-weight ``pd.Series``
            (owns the BL / base blend; falls back to base on ``(None, None)``).
        recall_guarded_config: the recall-guarded configuration shared across rebalances.

    Returns:
        A ``walk_forward``-compatible ``weight_fn(ctx) -> pd.Series``.
    """

    def weight_fn(ctx: dict) -> pd.Series:
        # Source the real tickers exactly as the Track A weight_fn does (R3.3).
        real_symbols = list(ctx["prices"].columns)
        macro_state, asset_snapshot, as_of, raw_levels = build_inputs(ctx)
        dec = factor_rebalance(
            generate_loadings=generate_loadings,
            scorer=scorer,
            agent=agent,
            macro_state=macro_state,
            asset_snapshot=asset_snapshot,
            real_symbols=real_symbols,
            as_of=as_of,
            raw_levels=raw_levels,
            recall_guarded_config=recall_guarded_config,
        )
        return combine(ctx, dec.P, dec.Q)

    return weight_fn


# --------------------------------------------------------------------------- #
# Task 3.3 — R8 certification screen (certified no-recall model selection)     #
# (Requirements 8.1, 8.2, 8.3, 8.4, 8.6)                                       #
#                                                                              #
# Screen a logprob-bearing NIM candidate for CERTIFIED absence of recall on     #
# the identified macro history. The screen reuses the EXACT controlled corpora  #
# the factor scorer calibrates on (_build_corpus: identifying IS vs anonymized  #
# OOS, token-identical except the identifying additions — R8.1) and adds:       #
#                                                                              #
#   - certification_stats — OFFLINE statistics on the gathered standardized     #
#     MIA features (R8.2): stratified-CV AUC point estimate + a per-class       #
#     bootstrap CI + a label-permutation two-sided p-value against chance.      #
#     No additional live inference for the statistics.                          #
#   - render_prose_confounded_prompt — the deliberately recall-enabling         #
#     POSITIVE-CONTROL framing (R8.3): dated narrative prose with real tickers  #
#     and raw levels. DIAGNOSTIC ONLY and explicitly NON-R7.6 (it differs from  #
#     the anonymized form by STYLE, not only additions) — never deployed.       #
#   - certification_verdict — the pure R8.4 decision rule mapping the           #
#     controlled + positive-control statistics and the factor-task parse rate   #
#     to one of: "recalls" / "detector_unvalidated" / "certified_no_recall" /   #
#     "inconclusive".                                                           #
#   - screen_candidate — the per-candidate orchestration (R8.1): build the      #
#     corpora, then the live-call steps (baseline, feature gathering, and the   #
#     fresh-reply parse-rate sample), compute the controlled and                #
#     positive-control statistics offline on the gathered matrices, and         #
#     return a typed CertificationResult (no credential in it, R8.6).           #
# --------------------------------------------------------------------------- #


# The allowed R8.4 verdict strings, in precedence order.
CERTIFICATION_VERDICTS: tuple[str, ...] = (
    "recalls",
    "detector_unvalidated",
    "certified_no_recall",
    "inconclusive",
)


def render_prose_confounded_prompt(
    macro_state: dict[str, float],
    asset_snapshot: list[dict[str, object]],
    *,
    as_of: Any,
    raw_levels: dict[str, float],
) -> str:
    """Render the deliberately recall-enabling PROSE-CONFOUNDED framing (R8.3).

    DIAGNOSTIC ONLY — the positive-control framing of the certification screen.
    A candidate model that recalls its training data at all should light up on
    THIS framing; if it does not, the detector itself is unvalidated
    ("detector_unvalidated", R8.3) and a null controlled result proves nothing.

    Explicitly NON-R7.6: unlike ``render_regime_loadings_prompt(identifying=True)``
    — which differs from the anonymized form ONLY by the identifying additions —
    this framing differs by STYLE as well: a dated market-commentary NARRATIVE
    ("It is {as_of}. ...") that names the real tickers and weaves the raw macro
    levels into sentences. It is therefore NEVER used for the deployable factor
    pipeline or for the controlled (R7.6) contrast; it exists solely to validate
    that the MIA detector can fire on this candidate at all.

    The task at the end is UNCHANGED: the same five ``MACRO_AXES`` loadings and
    the same JSON output instruction, so ``parse_loadings`` still applies to the
    reply. Deterministic: equal inputs produce an identical string.

    Args:
        macro_state: z-scored macro state (same dict the anonymized form renders).
        asset_snapshot: the asset descriptors (``id`` + ``category``); their
            categories are woven into the narrative.
        as_of: the calendar date to disclose in the opening sentence.
        raw_levels: the raw non-normalized macro levels woven into the prose.

    Returns:
        The rendered positive-control prompt string.
    """
    real_to_pseudo = _default_asset_identities()
    tickers_text = ", ".join(real_to_pseudo)
    categories_text = ", ".join(
        str(a.get("category")) for a in asset_snapshot if a.get("category") is not None
    )
    raw_text = "; ".join(
        f"{name} prints at {float(value):g}" for name, value in sorted(raw_levels.items())
    )
    z_text = "; ".join(
        f"{name} at {float(value):+.2f}" for name, value in sorted(macro_state.items())
    )

    lines: list[str] = [
        f"It is {as_of}. You are reading this morning's market commentary before "
        "the open, and every desk already knows how this chapter of the macro "
        "story goes.",
        "",
        f"The book holds the usual sleeves — {tickers_text} — covering "
        f"{categories_text}.",
        "",
        f"Today's tape is unambiguous: {raw_text}. Against their rolling history "
        f"the standardized readings sit at {z_text}.",
        "",
        "Write the regime the way the period's history is already told. "
        "Characterize it as a continuous loading in [-1, +1] on each of these "
        "five macro axes:",
        _axes_block(),
        "",
        'Output a JSON object mapping each axis name to its loading, e.g. '
        '{"inflation": 0.4, "growth": -0.2, "credit_stress": 0.6, "policy": -0.1, '
        '"risk_appetite": -0.3}.',
    ]
    return "\n".join(lines)


def certification_stats(
    x_is: Sequence[Sequence[float]],
    x_oos: Sequence[Sequence[float]],
    *,
    n_boot: int = 200,
    n_perm: int = 500,
    n_splits: int = 5,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Offline separation statistics on gathered standardized features (R8.2).

    Pure statistics — NO live inference. Operates on the two standardized MIA
    feature matrices gathered once per candidate (identifying vs anonymized, or
    prose-confounded vs anonymized) and reports how separable the two classes
    are, with statistical-certainty measures:

    - **Point estimate** — the mean fold AUC of a stratified ``n_splits``-fold
      cross-validated ``LogisticRegression(class_weight='balanced',
      solver='liblinear', random_state=seed)`` (the same classifier family the
      MCS calibrator uses, so the screen measures the separation the deployed
      calibrator could exploit).
    - **CI** — a bootstrap 2.5/97.5 percentile interval: rows are resampled
      with replacement PER CLASS (preserving both class sizes) ``n_boot``
      times and the CV AUC recomputed each time. Known ceiling (review
      2026-07-03): with-replacement duplicates span CV fold boundaries, so the
      bootstrap distribution is upward-biased (~+0.1 CI-midpoint on pure noise
      at n=20/class, shrinking with n). The bias is strictly CONSERVATIVE for
      the R8.4 gate — certification requires the CI to CONTAIN 0.5, so
      inflation can only block a certification, never grant a false one.
      Upgrade path if it ever blocks a true no-recall candidate: bootstrap the
      held-out fold predictions instead of the rows.
    - **Permutation p** — the class labels are shuffled ``n_perm`` times, the
      CV AUC recomputed each time, and the two-sided p-value reported as
      ``(1 + #{|auc_perm − 0.5| ≥ |auc_obs − 0.5|}) / (n_perm + 1)`` (the
      add-one permutation estimator; never exactly 0).

    Deterministic given ``seed``: one ``numpy`` Generator seeded from ``seed``
    drives all resampling in a fixed order, and the CV splitter / classifier
    share the same ``seed``.

    Args:
        x_is: the recall-class standardized feature rows (label 1).
        x_oos: the anonymized-class standardized feature rows (label 0).
        n_boot: bootstrap resamples for the CI.
        n_perm: label permutations for the p-value.
        n_splits: stratified CV folds.
        seed: the deterministic seed for resampling, splitting, and fitting.

    Returns:
        ``(auc, ci_low, ci_high, perm_p)``.

    Raises:
        ValueError: when either class has fewer than ``n_splits`` rows —
            stratified ``n_splits``-fold CV cannot guarantee both classes in
            every fold on such degenerate input.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    xi = np.asarray(x_is, dtype=np.float64)
    xo = np.asarray(x_oos, dtype=np.float64)
    if xi.ndim != 2 or xo.ndim != 2 or len(xi) < n_splits or len(xo) < n_splits:
        raise ValueError(
            "certification_stats: each class needs at least n_splits "
            f"(={n_splits}) feature rows of equal width for stratified CV; "
            f"got n_is={len(xi) if xi.ndim else 0}, n_oos={len(xo) if xo.ndim else 0}."
        )

    x = np.vstack([xi, xo])
    y = np.concatenate(
        [np.ones(len(xi), dtype=np.int64), np.zeros(len(xo), dtype=np.int64)]
    )

    def _cv_auc(xm: np.ndarray, ym: np.ndarray) -> float:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_aucs: list[float] = []
        for train_idx, test_idx in skf.split(xm, ym):
            clf = LogisticRegression(
                class_weight="balanced", solver="liblinear", random_state=seed
            )
            clf.fit(xm[train_idx], ym[train_idx])
            fold_aucs.append(
                float(
                    roc_auc_score(
                        ym[test_idx], clf.predict_proba(xm[test_idx])[:, 1]
                    )
                )
            )
        return float(np.mean(fold_aucs))

    auc_obs = _cv_auc(x, y)

    rng = np.random.default_rng(seed)

    # Bootstrap CI: resample rows with replacement PER CLASS so both class
    # sizes are preserved (stratified CV stays valid on every resample).
    n_i, n_o = len(xi), len(xo)
    boot_aucs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx_i = rng.integers(0, n_i, n_i)
        idx_o = rng.integers(0, n_o, n_o)
        boot_aucs[b] = _cv_auc(np.vstack([xi[idx_i], xo[idx_o]]), y)
    ci_low = float(np.percentile(boot_aucs, 2.5))
    ci_high = float(np.percentile(boot_aucs, 97.5))

    # Two-sided permutation p against chance separation (AUC = 0.5).
    obs_dev = abs(auc_obs - 0.5)
    hits = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        if abs(_cv_auc(x, y_perm) - 0.5) >= obs_dev:
            hits += 1
    perm_p = (1 + hits) / (n_perm + 1)

    return auc_obs, ci_low, ci_high, float(perm_p)


@dataclass(frozen=True)
class CertificationResult:
    """The typed per-candidate certification outcome (R8.2, R8.3, R8.4, R8.6).

    Frozen so a screened candidate's outcome is a stable value the live run can
    persist verbatim (via :meth:`to_dict`). Carries NO credential (R8.6).

    Attributes:
        model: the screened NIM model id.
        controlled_auc: the controlled (identifying vs anonymized) CV AUC.
        controlled_ci_low: the bootstrap CI lower bound of the controlled AUC.
        controlled_ci_high: the bootstrap CI upper bound of the controlled AUC.
        controlled_perm_p: the controlled permutation p-value against chance.
        positive_control_auc: the prose-confounded (vs anonymized) CV AUC, or
            ``None`` when the positive-control statistics could not be computed.
        positive_control_perm_p: the positive-control permutation p-value, or
            ``None`` when unavailable — which maps to "detector_unvalidated".
        parse_rate: the fraction of sampled anonymized factor replies that
            ``parse_loadings`` parsed, or ``None`` when no sample was taken.
        n_per_class: the number of controlled prompts actually built per class.
        verdict: one of :data:`CERTIFICATION_VERDICTS` (R8.4).
    """

    model: str
    controlled_auc: float
    controlled_ci_low: float
    controlled_ci_high: float
    controlled_perm_p: float
    positive_control_auc: float | None
    positive_control_perm_p: float | None
    parse_rate: float | None
    n_per_class: int
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable dict of this result (for the persisted artifact)."""
        return {
            "model": self.model,
            "controlled_auc": self.controlled_auc,
            "controlled_ci_low": self.controlled_ci_low,
            "controlled_ci_high": self.controlled_ci_high,
            "controlled_perm_p": self.controlled_perm_p,
            "positive_control_auc": self.positive_control_auc,
            "positive_control_perm_p": self.positive_control_perm_p,
            "parse_rate": self.parse_rate,
            "n_per_class": self.n_per_class,
            "verdict": self.verdict,
        }


def certification_verdict(
    *,
    controlled_auc: float,
    controlled_ci_low: float,
    controlled_ci_high: float,
    controlled_perm_p: float,
    positive_control_perm_p: float | None,
    parse_rate: float | None,
    alpha: float = 0.05,
    no_recall_p: float = 0.1,
    min_parse_rate: float = 0.9,
) -> str:
    """The pure R8.4 certification rule — precedence-ordered, deterministic.

    Precedence:

    1. **"recalls"** — the controlled separation is statistically significant
       AND above chance (``controlled_perm_p < alpha`` and
       ``controlled_auc > 0.5``): the candidate is rejected as recalling,
       regardless of the positive control (a firing controlled detector is
       itself the strongest possible detector validation).
    2. **"detector_unvalidated"** — the positive control did not fire
       (``positive_control_perm_p`` is ``None`` or ``>= alpha``): a null
       controlled result proves nothing when the detector cannot even see the
       deliberately recall-enabling framing (R8.3).
    3. **"certified_no_recall"** — the controlled separation is statistically
       indistinguishable from chance (``controlled_perm_p > no_recall_p`` and
       the CI contains 0.5) AND the factor-task replies parse at a usable rate
       (``parse_rate >= min_parse_rate``) (R8.4).
    4. **"inconclusive"** — everything else (e.g. a borderline p-value, a CI
       missing 0.5, or an unusable parse rate).

    Args:
        controlled_auc: the controlled CV AUC point estimate.
        controlled_ci_low: the controlled bootstrap CI lower bound.
        controlled_ci_high: the controlled bootstrap CI upper bound.
        controlled_perm_p: the controlled permutation p-value.
        positive_control_perm_p: the positive-control permutation p-value
            (``None`` when unavailable).
        parse_rate: the measured factor-task parse rate (``None`` when unmeasured).
        alpha: the significance level for "recalls" and the positive control.
        no_recall_p: the minimum controlled p-value for a no-recall certificate.
        min_parse_rate: the minimum usable factor-task parse rate.

    Returns:
        One of :data:`CERTIFICATION_VERDICTS`.
    """
    if controlled_perm_p < alpha and controlled_auc > 0.5:
        return "recalls"
    if positive_control_perm_p is None or positive_control_perm_p >= alpha:
        return "detector_unvalidated"
    if (
        controlled_perm_p > no_recall_p
        and controlled_ci_low <= 0.5 <= controlled_ci_high
        and parse_rate is not None
        and parse_rate >= min_parse_rate
    ):
        return "certified_no_recall"
    return "inconclusive"


def gather_certification_features(
    lm: NvidiaLM,
    rows: list[EvalRow],
    baseline: ControlBaseline,
    *,
    max_workers: int = 8,
    evidence: list[dict[str, Any]] | None = None,
) -> list[list[float]]:
    """Gather the standardized per-prompt MIA feature vectors for one corpus (R8.2).

    Mirrors the ``FactorScorer._score_completion`` primitive path exactly —
    ``lm.generate(prompt)`` (fanned out via the library's order-preserving
    ``generate_many``) → ``compute_mia_features(content, logprobs, None)``
    (``ref_logprobs=None``: the reference run is disabled, matching the
    calibration contract) → ``standardise(features, baseline)`` — then flattens
    each standardized dict into a vector in the deterministic order
    ``list(baseline.feature_means)`` restricted to the features the baseline
    actually carries (``ref_delta`` is excluded when its baseline mean is
    ``None``, mirroring the calibrator's ``feature_order`` resolution).

    Rows whose generation, feature computation, or standardisation fails are
    SKIPPED (dropped); the surviving vectors keep the input-row order.

    When ``evidence`` is given (a caller-owned list), one raw-evidence record is
    appended per INPUT row — surviving and dropped alike — carrying the prompt,
    the reply text, the token count, the raw MIA features (``raw_<key>``), the
    standardized values (``std_<key>``), an ``included`` flag, and the
    ``dropped_reason`` when the row did not survive. This is the audit trail
    the R8.6 evidence artifact persists so the assessment can be re-derived
    from raw data (no credential is ever recorded).

    Args:
        lm: the candidate's inference client (only ``generate`` is used).
        rows: the corpus rows to gather features for.
        baseline: the candidate's control baseline (standardisation stats).
        max_workers: parallelism for the LM calls.
        evidence: optional caller-owned list to append per-row raw-evidence
            records to (one per input row, in input order).

    Returns:
        One standardized feature vector per surviving row, in input order.
    """
    from dataclasses import asdict

    from recall_guard import standardise

    # The deterministic vector order: the baseline's feature keys, restricted
    # to features the baseline actually carries (ref_delta mean None -> excluded).
    order = [
        key for key, mean in baseline.feature_means.items() if mean is not None
    ]

    prompts = [row.prompt for row in rows]
    results = generate_many(lm, prompts, max_workers=max_workers)

    vectors: list[list[float]] = []
    for i, completion in enumerate(results):
        record: dict[str, Any] | None = None
        if evidence is not None:
            record = {
                "row_index": i,
                "as_of": rows[i].metadata.get("as_of"),
                "prompt": rows[i].prompt,
                "reply": None,
                "n_tokens": None,
                "included": False,
                "dropped_reason": None,
            }
            evidence.append(record)
        if isinstance(completion, BaseException) or completion is None:
            if record is not None:
                record["dropped_reason"] = (
                    f"generation failed: {type(completion).__name__}"
                    if isinstance(completion, BaseException)
                    else "generation failed"
                )
            continue  # generation failed for this row -> drop it
        logprobs = getattr(completion, "logprobs", None)
        content = getattr(completion, "content", "")
        if record is not None:
            record["reply"] = content
            record["n_tokens"] = len(logprobs) if logprobs else 0
        if not logprobs:
            if record is not None:
                record["dropped_reason"] = "logprobs missing"
            continue
        try:
            features = compute_mia_features(content, logprobs, None)
        except (ValueError, RuntimeError) as exc:
            if record is not None:
                record["dropped_reason"] = f"features failed: {type(exc).__name__}"
            continue
        standardised = standardise(features, baseline)
        if record is not None:
            for key, value in asdict(features).items():
                if value is None or isinstance(value, (int, float)):
                    record[f"raw_{key}"] = value
            for key in order:
                record[f"std_{key}"] = standardised.get(key)
        vector: list[float] = []
        for key in order:
            value = standardised.get(key)
            if value is None:
                break
            vector.append(float(value))
        else:
            vectors.append(vector)
            if record is not None:
                record["included"] = True
        if record is not None and not record["included"] and record["dropped_reason"] is None:
            record["dropped_reason"] = "standardised feature missing"
    return vectors


def _write_screen_evidence(
    evidence_dir: Any,
    *,
    result: CertificationResult,
    cutoff_date: date,
    baseline: ControlBaseline,
    arms: dict[str, list[dict[str, Any]]],
) -> None:
    """Persist the raw per-prompt evidence backing one candidate's screen (R8.6).

    Written for the follow-up PyXLL/Excel spec: the assessment must be
    re-derivable from RAW evidence, not just the summary statistics. Files
    (all credential-free) under ``evidence_dir``:

    - ``evidence.parquet`` — one row per prompt across every arm
      (``identifying`` / ``anonymized`` / ``prose_confounded`` /
      ``parse_sample``): prompt, reply, token count, raw + standardized MIA
      features, included flag, dropped reason.
    - ``baseline.json`` — the control-baseline standardisation stats.
    - ``summary.json`` — the ``CertificationResult`` plus the screen cutoff.
    """
    from pathlib import Path

    pd_mod = _pd()
    out = Path(evidence_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for arm, records in arms.items():
        if not records:
            continue
        frame = pd_mod.DataFrame.from_records(records)
        frame.insert(0, "arm", arm)
        frames.append(frame)
    if frames:
        pd_mod.concat(frames, ignore_index=True, sort=False).to_parquet(
            out / "evidence.parquet"
        )

    (out / "baseline.json").write_text(json.dumps({
        "model": baseline.model,
        "n_valid": baseline.n_valid,
        "min_valid": baseline.min_valid,
        "is_calibrated": baseline.is_calibrated,
        "feature_means": baseline.feature_means,
        "feature_stds": baseline.feature_stds,
    }, indent=2))
    (out / "summary.json").write_text(json.dumps(
        {**result.to_dict(), "cutoff_date": cutoff_date.isoformat()}, indent=2
    ))


def screen_candidate(
    *,
    nim_model: str,
    cutoff_date: date,
    macro_panel: pd.DataFrame,
    asset_map: AssetMap,
    api_key: str,
    n_per_class: int = 120,
    parse_sample: int = 20,
    max_workers: int = 8,
    lm_factory: Callable[[str, str], NvidiaLM] | None = None,
    evidence_dir: Any | None = None,
) -> CertificationResult:
    """Screen one logprob-bearing candidate for certified no-recall (R8.1–R8.4).

    The per-candidate orchestration of the certification screen:

    1. Build the CONTROLLED corpora via :func:`_build_corpus` — the exact
       identifying-IS vs anonymized-OOS framings the factor scorer calibrates
       on, from the pre-cutoff macro states (R8.1; ``n_per_class`` is capped at
       the available pre-cutoff rows by ``_build_corpus``).
    2. Re-render the SAME pre-cutoff states with
       :func:`render_prose_confounded_prompt` — the positive-control class
       (R8.3); the anonymized OOS rows are shared as its contrast class.
    3. ``build_baseline`` on the anonymized OOS corpus (``ref_lm=None``) —
       itself live generate calls.
    4. Gather standardized features for all three classes (more live calls;
       step 5's parse sample is live too), then compute the controlled
       (identifying vs anonymized) and
       positive-control (prose-confounded vs anonymized) statistics OFFLINE
       via :func:`certification_stats` (R8.2). A positive-control statistics
       failure (too few surviving rows) degrades to ``None`` — which the
       verdict maps to "detector_unvalidated" — rather than crashing the screen.
    5. Measure the factor-task parse rate: the fraction of ``parse_loadings``
       successes over up to ``parse_sample`` fresh anonymized factor replies.
    6. Map everything through :func:`certification_verdict` (R8.4).

    The returned :class:`CertificationResult` carries NO credential (R8.6).

    Args:
        nim_model: the candidate's logprob-bearing NIM model id.
        cutoff_date: the conservative screening cutoff; only rows before it are
            used (pre-cutoff states trained-on for every candidate).
        macro_panel: the FRED macro panel with raw + z-scored columns.
        asset_map: the identifying <-> anonymized asset map.
        api_key: the NIM scoring credential (non-empty; never persisted).
        n_per_class: target number of controlled prompts per class.
        parse_sample: number of anonymized replies sampled for the parse rate.
        max_workers: parallelism for the LM calls.
        lm_factory: optional ``(api_key, model) -> lm`` factory for test
            injection; defaults to constructing ``NvidiaLM``.
        evidence_dir: when set, the raw per-prompt evidence backing this
            screen (all four arms) is persisted under it via
            ``_write_screen_evidence`` — ``evidence.parquet`` +
            ``baseline.json`` + ``summary.json``, credential-free (R8.6 /
            the PyXLL data contract).

    Returns:
        The candidate's :class:`CertificationResult`.

    Raises:
        ConfigurationError: when ``api_key`` is empty (same fail-fast contract
            as :meth:`FactorScorer.calibrate`, R1.5) or when the control
            baseline ends up with zero usable rows.
        ValueError: when the CONTROLLED classes end up with too few gathered
            feature rows for :func:`certification_stats` (a candidate that
            cannot even be measured is a hard screen failure).
    """
    # R1.5-style fail-fast: a missing credential is a configuration fault.
    if not api_key:
        raise ConfigurationError(
            "screen_candidate: a non-empty NIM api_key is required "
            "(the scoring credential is missing)."
        )

    is_rows, oos_rows = _build_corpus(
        cutoff_date=cutoff_date,
        macro_panel=macro_panel,
        asset_map=asset_map,
        n_per_class=n_per_class,
    )

    # Positive-control corpus: the SAME selected pre-cutoff states, re-rendered
    # prose-confounded. The state is re-read from the panel via each OOS row's
    # as_of tag so the selection is exactly _build_corpus's (R8.3 shares the
    # anonymized OOS rows as the contrast class).
    pd_mod = _pd()
    asset_snapshot = _asset_snapshot_from_map(asset_map)
    prose_rows: list[EvalRow] = []
    for row in oos_rows:
        as_of_tag = row.metadata["as_of"]
        panel_row = macro_panel.loc[pd_mod.Timestamp(as_of_tag)]
        macro_state = {
            z_col: float(panel_row[z_col])
            for z_col in _RAW_TO_Z.values()
            if z_col in panel_row
        }
        raw_levels = {
            raw_col: float(panel_row[raw_col])
            for raw_col in _RAW_TO_Z
            if raw_col in panel_row
        }
        prose_prompt = render_prose_confounded_prompt(
            macro_state, asset_snapshot, as_of=as_of_tag, raw_levels=raw_levels
        )
        prose_rows.append(
            EvalRow(prompt=prose_prompt, target_direction=0, metadata={"as_of": as_of_tag})
        )

    if lm_factory is not None:
        lm = lm_factory(api_key, nim_model)
    else:
        lm = NvidiaLM(api_key=api_key, model=nim_model)

    baseline = build_baseline(
        lm,
        oos_rows,
        None,  # ref_lm=None -> ref_delta inert (matching calibrate)
        min_valid=min(len(oos_rows), 2),
        max_workers=max_workers,
    )
    if baseline.n_valid == 0:
        raise ConfigurationError(
            "screen_candidate: the control baseline has no valid rows "
            f"(baseline.n_valid == 0) for model {nim_model!r}; cannot "
            "standardise MIA features. Check the NIM credential/endpoint and "
            "that the panel yielded a non-empty pre-cutoff OOS corpus."
        )

    # Optional raw-evidence capture (R8.6 / PyXLL data contract): one record
    # per prompt per arm, so the assessment is re-derivable from raw data.
    ev: dict[str, list[dict[str, Any]]] | None = None
    if evidence_dir is not None:
        ev = {
            "identifying": [],
            "anonymized": [],
            "prose_confounded": [],
            "parse_sample": [],
        }

    # Gather the standardized features per class; the controlled and
    # positive-control statistics are computed OFFLINE on these matrices (R8.2).
    # The parse-rate block further below issues its own fresh live calls.
    x_is = gather_certification_features(
        lm, is_rows, baseline, max_workers=max_workers,
        evidence=ev["identifying"] if ev is not None else None,
    )
    x_oos = gather_certification_features(
        lm, oos_rows, baseline, max_workers=max_workers,
        evidence=ev["anonymized"] if ev is not None else None,
    )
    x_prose = gather_certification_features(
        lm, prose_rows, baseline, max_workers=max_workers,
        evidence=ev["prose_confounded"] if ev is not None else None,
    )

    controlled_auc, ci_low, ci_high, controlled_perm_p = certification_stats(
        x_is, x_oos
    )

    positive_control_auc: float | None
    positive_control_perm_p: float | None
    try:
        positive_control_auc, _, _, positive_control_perm_p = certification_stats(
            x_prose, x_oos
        )
    except ValueError:
        # Too few surviving positive-control rows: the detector cannot be
        # validated -> None, which the verdict maps to "detector_unvalidated".
        positive_control_auc = None
        positive_control_perm_p = None

    # Factor-task parse rate over fresh anonymized replies (R8.4's "usable rate").
    sample = oos_rows[: max(0, int(parse_sample))]
    parse_rate: float | None = None
    if sample:
        replies = generate_many(
            lm, [row.prompt for row in sample], max_workers=max_workers
        )
        n_parsed = 0
        for j, completion in enumerate(replies):
            content: str | None = None
            parsed_ok = False
            if not (isinstance(completion, BaseException) or completion is None):
                content = getattr(completion, "content", "")
                parsed_ok = parse_loadings(content, None) is not None
            if parsed_ok:
                n_parsed += 1
            if ev is not None:
                ev["parse_sample"].append({
                    "row_index": j,
                    "as_of": sample[j].metadata.get("as_of"),
                    "prompt": sample[j].prompt,
                    "reply": content,
                    "included": parsed_ok,
                    "dropped_reason": None if parsed_ok else "loadings parse failed",
                })
        parse_rate = n_parsed / len(sample)

    verdict = certification_verdict(
        controlled_auc=controlled_auc,
        controlled_ci_low=ci_low,
        controlled_ci_high=ci_high,
        controlled_perm_p=controlled_perm_p,
        positive_control_perm_p=positive_control_perm_p,
        parse_rate=parse_rate,
    )

    result = CertificationResult(
        model=nim_model,
        controlled_auc=controlled_auc,
        controlled_ci_low=ci_low,
        controlled_ci_high=ci_high,
        controlled_perm_p=controlled_perm_p,
        positive_control_auc=positive_control_auc,
        positive_control_perm_p=positive_control_perm_p,
        parse_rate=parse_rate,
        n_per_class=len(is_rows),
        verdict=verdict,
    )

    if ev is not None:
        _write_screen_evidence(
            evidence_dir,
            result=result,
            cutoff_date=cutoff_date,
            baseline=baseline,
            arms=ev,
        )

    return result
