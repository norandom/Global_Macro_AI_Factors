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

# Number-native MIA calibration/scoring primitives from the released library.
# These are imported AT MODULE LEVEL (not lazily) so they are bound on this
# module's namespace and can be patched as `macro_framework.factor_scoring.<name>`
# in tests (the calibration path is mocked there — no NIM/FMP calls). The
# directional MemoryGuardedScorer facade is deliberately NOT imported (R6.5).
from recall_guard import NvidiaLM, build_baseline, compute_mia_features
from recall_guard.core.loader import EvalRow
from recall_guard.core.nvidia_lm import generate_many
from recall_guard.mia.mcs import train

if TYPE_CHECKING:
    from collections.abc import Sequence
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
#   - honest class (OOS)  = the SAME states presented ANONYMIZED                #
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
        standardise against — scoring would be invalid).

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
        n_oos: number of anonymized (honest-class) prompts built for calibration.
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
      - **anonymized OOS** (``identifying=False``) — the honest, recall-disabled
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

        The fallback signal: when weak, the honesty adjustment leaves exposures
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

        The number-native calibration is a one-time cost (~135 NIM calls); this
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
    def load(cls, path: Path, *, api_key: str) -> FactorScorer:
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

        # Re-attach a fresh NvidiaLM with the caller's key on the persisted model.
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
