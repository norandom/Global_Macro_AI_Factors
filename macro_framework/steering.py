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

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from recall_guard import MemoryGuardedScorer
from recall_guard.dataset.fmp_corpora import build_calibration

from .anonymize import AssetMap
from .llm_agent import LlmMacroAgent

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from recall_guard import GuardedScore

    from .llm_agent import MacroView

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


# ---------------------------------------------------------------------------
# Task 2.2 — ScoringAdapter: calibration half.
#
# The adapter owns the two-corpus calibration of the released
# ``recall_guard`` scorer (design "ScoringAdapter" + "Two-corpus calibration"
# key decision). The 72-state Track A agent log is *not* a labelled
# directional IS/OOS corpus, so the NIM model is calibrated on a dated FMP
# corpus instead: ``fmp_corpora.build_calibration`` writes two JSONL files
# (IS = pre-cutoff, label 1; OOS = post-cutoff, label 0) and returns their
# paths; the adapter reads them back via ``_read_corpus_jsonl``, projects each
# record's ``prompt`` field to a ``list[str]`` (order preserved), and passes the
# IS list as ``is_memorized`` and the OOS list as ``oos_control`` to
# ``MemoryGuardedScorer.calibrate`` (Calibration flow diagram).
#
# Requirements:
#   1.2 — the same anonymized, z-scored PIT content is scored; here the
#         calibration corpora are read back and fed to the calibrator verbatim.
#   1.5 — an empty/rejected NIM credential surfaces ``ConfigurationError``;
#         ``calibrate`` already raises it on an empty key, so it is left to
#         propagate (no catch-and-swallow).
#   1.7 — calibrator quality is surfaced via ``is_weak`` / ``holdout_auc``.
#
# The per-prompt scoring path (``score_rebalances``) is intentionally NOT in
# this task — it ships in task 2.3.
# ---------------------------------------------------------------------------


def _read_corpus_jsonl(path: Path) -> list[str]:
    """Read a ``build_calibration`` JSONL file into a list of prompt strings.

    Each line is a JSON record shaped ``{"prompt": str, "label": int,
    "metadata": {...}}`` (the schema ``fmp_corpora.build_calibration`` writes).
    Only the ``prompt`` field is projected; the order of the records on disk is
    preserved so the IS/OOS labelling implied by the file stays aligned.

    Parameters
    ----------
    path:
        Path to one of the JSONL corpora returned by ``build_calibration``.

    Returns
    -------
    list[str]
        Each record's ``prompt`` field, in file order.
    """
    prompts: list[str] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            prompts.append(str(json.loads(line)["prompt"]))
    return prompts


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of a one-time two-corpus calibration of the NIM scorer.

    Carries the calibrated ``recall_guard`` scorer alongside the calibrator's
    held-out separation (``holdout_auc``) and the weak flag (``is_weak``) so the
    quality can be persisted in the score-log header and used to gate steering
    (Requirement 1.7).
    """

    scorer: MemoryGuardedScorer
    holdout_auc: float
    is_weak: bool


class ScoringAdapter:
    """Wrap a calibrated ``recall_guard`` scorer for the Track A scoring path.

    This task ships the calibration half only. ``calibrate_from_fmp`` builds the
    dated FMP corpus, reads both JSONL files back, calibrates the chosen NIM
    model, and wraps the resulting scorer; ``is_weak`` / ``holdout_auc`` delegate
    to the scorer so callers can surface calibrator quality (Requirement 1.7).
    The per-rebalance ``score_rebalances`` path is added in task 2.3.
    """

    def __init__(self, calibration: CalibrationResult) -> None:
        self._calibration = calibration

    @property
    def calibration(self) -> CalibrationResult:
        """The calibration result wrapping the scorer and its quality flags."""
        return self._calibration

    @property
    def scorer(self) -> MemoryGuardedScorer:
        """The underlying calibrated ``recall_guard`` scorer."""
        return self._calibration.scorer

    @property
    def is_weak(self) -> bool:
        """``True`` when the calibrator is weak (delegates to the scorer; R1.7)."""
        return self._calibration.scorer.is_weak

    @property
    def holdout_auc(self) -> float:
        """Held-out IS/OOS separation of the calibrator (delegates; R1.7)."""
        return self._calibration.scorer.holdout_auc

    @classmethod
    def calibrate_from_fmp(
        cls,
        *,
        nim_model: str,
        cutoff_date: date,
        out_dir: Path,
        api_key: str,
        fmp_api_key: str | None = None,
        reference_model: str | None = None,
        min_auc: float = 0.6,
        target_per_corpus: int = 100,
    ) -> ScoringAdapter:
        """Build the FMP corpus, calibrate the NIM model, and wrap the scorer.

        Steps (design "Calibration flow"):

        1. ``build_calibration(out_dir, cutoffs={nim_model: cutoff_date},
           target_per_corpus=target_per_corpus, api_key=fmp_api_key)`` writes the
           IS (pre-cutoff, label 1) and OOS (post-cutoff, label 0) JSONL corpora
           and returns ``(is_path, oos_path)``.
        2. Both files are read back via ``_read_corpus_jsonl`` into prompt lists,
           order preserved (Requirement 1.2).
        3. ``MemoryGuardedScorer.calibrate(...)`` trains the calibrator on the IS
           list (``is_memorized``) and the OOS list (``oos_control``).

        The calibrated scorer is wrapped in a :class:`CalibrationResult` whose
        ``holdout_auc`` / ``is_weak`` are surfaced by the adapter (Requirement
        1.7). An empty/rejected NIM ``api_key`` raises ``ConfigurationError`` from
        ``calibrate``; it is left to propagate unchanged (Requirement 1.5).

        Parameters
        ----------
        nim_model:
            NIM model id; also the cutoff-dict key (defines the IS/OOS split).
        cutoff_date:
            The ``nim_model``'s training cutoff.
        out_dir:
            Writable directory for the two JSONL corpora.
        api_key:
            NVIDIA NIM credential (non-empty; an empty value raises
            ``ConfigurationError`` from ``calibrate``).
        fmp_api_key:
            FMP credential for the corpus build (falls back to the env inside
            ``build_calibration`` when ``None``).
        reference_model:
            Optional NIM reference model for the control baseline.
        min_auc:
            Calibration gate passed through to ``calibrate``.
        target_per_corpus:
            Target row count per corpus passed through to ``build_calibration``.

        Returns
        -------
        ScoringAdapter
            Wrapping the calibrated scorer.
        """
        is_path, oos_path = build_calibration(
            out_dir,
            cutoffs={nim_model: cutoff_date},
            target_per_corpus=target_per_corpus,
            api_key=fmp_api_key,
        )
        is_memorized = _read_corpus_jsonl(is_path)
        oos_control = _read_corpus_jsonl(oos_path)

        scorer = MemoryGuardedScorer.calibrate(
            api_key=api_key,
            model=nim_model,
            is_memorized=is_memorized,
            oos_control=oos_control,
            reference_model=reference_model,
            min_auc=min_auc,
        )
        calibration = CalibrationResult(
            scorer=scorer,
            holdout_auc=scorer.holdout_auc,
            is_weak=scorer.is_weak,
        )
        return cls(calibration)

    def score_rebalances(self, prompts: Sequence[str]) -> list[GuardedScore]:
        """Score per-rebalance directional prompts on the separate inference path.

        A thin, order-preserving wrapper over the calibrated scorer's
        ``score_many``: one NIM call per prompt, run on the calibrated
        ``recall_guard`` scorer — the **separate, logprob-bearing inference
        path**, not the agent's DSPy/OpenRouter call, which stays untouched
        (Requirement 1.3). The prompts carry the same anonymized, z-scored PIT
        macro content the agent reasoned over (rendered by ``render_directional``;
        Requirement 1.2). Results align 1:1 with ``prompts`` in input order,
        because ``MemoryGuardedScorer.score_many`` preserves order.

        The full ``list[GuardedScore]`` is returned on purpose: task 2.7's
        ``score_distribution_report`` needs ``p_memorized``, ``parse_ok``,
        ``fail_reason``, and ``memguard_confidence``. The design Data Models note
        ("Only ``p_memorized`` and ``fail_reason`` are steering inputs … the
        scorer's ``signal`` / ``raw_confidence`` are never read"; design
        "ScoringAdapter" Responsibilities) is a **downstream consumption
        contract** enforced by the ViewSteerer (task 2.4 reads only
        ``p_memorized``) and the gating logic — never read from a decision path
        here — not a narrowing of this return type.

        If the NIM endpoint rejects the credential mid-scoring, ``recall_guard``
        raises ``ConfigurationError``; it is left to propagate unchanged so the
        caller fails fast with an actionable error rather than acting on an
        unscored result (Requirement 1.5).

        Parameters
        ----------
        prompts:
            Per-rebalance directional prompts (typically from
            ``render_directional``), one per decision to score.

        Returns
        -------
        list[GuardedScore]
            One ``GuardedScore`` per prompt, in input order; failed scores carry
            their ``fail_reason`` (the scorer does not drop them).
        """
        return self.scorer.score_many(prompts)


# ---------------------------------------------------------------------------
# Task 2.4 — MacroCharacterizer: PIT regime + z-summary + consistency inputs.
#
# ``characterize`` consumes the EXISTING macro panel strictly *before* the
# rebalance date (Requirements 2.2, 2.3 — never regenerates it), reads the
# latest as-of z-scores per series (Requirement 2.1), and assigns a
# DETERMINISTIC z-bin regime label (Requirement 2.5 — no fitted/forecasting
# target). The frozen :class:`SteeringSignal` it returns exposes a pure
# ``consistency(view)`` heuristic in ``[consistency_floor, 1.0]`` consumed
# downstream by the ViewSteerer (task 2.5). ``write_steering_signals`` persists
# the per-rebalance signals to a parquet artifact under a NEW filename
# (Requirements 2.4, 6.4 — the playbook chooses the actual name).
#
# The macro z-series this reads are the panel's z-scored columns
# (``macro_framework.macro.build_macro_panel`` emits cpi_yoy_z, t10y2y_z,
# hy_oas_z); the consistency map resolves view long-legs to categories via the
# FIXED ``AssetMap`` (read-only).
# ---------------------------------------------------------------------------

# The z-series the regime/summary is built from, in a fixed order so the signal
# is deterministic regardless of the panel's column order (Requirement 2.1).
_REGIME_Z_SERIES = ("cpi_yoy_z", "t10y2y_z", "hy_oas_z")

# DETERMINISTIC z-bin thresholds for the regime taxonomy. These are fixed,
# documented priors — NOT fitted to any forecasting target (Requirement 2.5).
# A series is "elevated" at/above +_Z_HIGH; the curve is "inverted" at/below
# _Z_INVERTED. The benign goldilocks bin uses the softer _Z_BENIGN: it asks only
# that inflation be below-average and credit spreads tight (clearly below
# average), not a full sigma depressed — matching the design's "low inflation +
# tight spreads" intent on the coarse ~72-state panel.
_Z_HIGH = 1.0       # z at/above which a series counts as elevated
_Z_INVERTED = -0.5  # t10y2y_z at/below which the curve counts as inverted
_Z_BENIGN = -0.5    # hy_oas_z at/below which spreads count as tight (goldilocks)

# Regime taxonomy (evaluated in priority order; first match wins). The order
# resolves overlaps deterministically: credit stress (a market-wide risk-off
# tell) outranks stagflation, which outranks the benign goldilocks bin; anything
# unclassified is the non-committal ``neutral``.
#   credit_stress    : hy_oas_z high                       (credit risk-off)
#   stagflation_risk : cpi_yoy_z high AND curve inverted   (high inflation + slowdown)
#   goldilocks       : cpi_yoy_z below-average AND spreads tight (cool + tight)
#   neutral          : none of the above

# Regime → preferred asset CATEGORIES (a deliberate, documented heuristic, NOT a
# learned/return-fitted mapping; Requirement 2.5). Categories are the FIXED
# ``AssetMap`` categories (world_equity, tech_sector, gold_commodity,
# short_treasury_cash). Risk-off regimes prefer defensives (gold + cash);
# goldilocks prefers risk assets (equities + tech); neutral prefers everything
# (non-committal). A view whose long-leg category is preferred scores 1.0;
# otherwise it falls to the ``consistency_floor``.
_REGIME_PREFERRED_CATEGORIES: dict[str, frozenset[str]] = {
    "credit_stress": frozenset({"gold_commodity", "short_treasury_cash"}),
    "stagflation_risk": frozenset({"gold_commodity", "short_treasury_cash"}),
    "goldilocks": frozenset({"world_equity", "tech_sector"}),
    "neutral": frozenset(
        {"world_equity", "tech_sector", "gold_commodity", "short_treasury_cash"}
    ),
}

# Default min consistency multiplier (mirrors SteeringConfig.consistency_floor in
# the later ViewSteerer task); a contradicting view never falls below this.
_DEFAULT_CONSISTENCY_FLOOR = 0.5


def _classify_regime(z: dict[str, float]) -> str:
    """Map latest as-of z-scores to a deterministic regime label (Requirement 2.5).

    Pure function of the three z-scores using the fixed ``_Z_HIGH`` /
    ``_Z_INVERTED`` thresholds and the documented priority order
    (credit_stress > stagflation_risk > goldilocks > neutral). No randomness, no
    fitting, no forecasting target.
    """
    cpi = z.get("cpi_yoy_z", 0.0)
    curve = z.get("t10y2y_z", 0.0)
    hy = z.get("hy_oas_z", 0.0)

    if hy >= _Z_HIGH:
        return "credit_stress"
    if cpi >= _Z_HIGH and curve <= _Z_INVERTED:
        return "stagflation_risk"
    if cpi < 0.0 and hy <= _Z_BENIGN:
        return "goldilocks"
    return "neutral"


@dataclass(frozen=True)
class SteeringSignal:
    """A point-in-time macro characterization for one rebalance date.

    Frozen and deterministic: built only from macro rows strictly before
    ``rebalance_date`` (Requirement 2.2). ``zscore_summary`` carries the latest
    as-of z per series (Requirement 2.1); ``regime_label`` is a deterministic
    z-bin label (Requirement 2.5). ``consistency`` is a pure heuristic mapping a
    view's long-leg category to ``1.0`` (preferred) or ``consistency_floor``
    (not preferred), consumed by the ViewSteerer (Requirement 2.4).
    """

    rebalance_date: pd.Timestamp
    regime_label: str
    zscore_summary: dict[str, float]
    _preferred_categories: frozenset[str] = field(default_factory=frozenset)
    _consistency_floor: float = _DEFAULT_CONSISTENCY_FLOOR
    _asset_map: AssetMap = field(default_factory=AssetMap.default)

    def consistency(self, view: MacroView) -> float:
        """Macro-consistency multiplier for ``view`` in ``[consistency_floor, 1.0]``.

        Resolves the view's long-leg pseudo asset (e.g. ``Asset_A``) to its
        category via the FIXED ``AssetMap`` and returns ``1.0`` when that
        category is in this regime's preferred set, else ``consistency_floor``.
        Pure function of (regime, view category): no randomness, no I/O, no
        forecasting (Requirement 2.4, 2.5). An unknown long-leg id (no category)
        is treated as not-preferred ⇒ floor.
        """
        category = self._asset_map.categories.get(view.asset_long)
        if category is not None and category in self._preferred_categories:
            return 1.0
        return self._consistency_floor


def characterize(
    macro_hist: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    consistency_floor: float = _DEFAULT_CONSISTENCY_FLOOR,
) -> SteeringSignal:
    """Characterize the macro panel as-of ``rebalance_date`` into a steering signal.

    DEFENSIVELY slices ``macro_hist`` to rows strictly before ``rebalance_date``
    so adding any row dated on/after it cannot change the result (Requirement
    2.2). The panel is consumed read-only — nothing is regenerated (Requirement
    2.3). The latest as-of row supplies the per-series z summary (Requirement
    2.1); a deterministic z-bin taxonomy assigns the regime label (Requirement
    2.5). The returned :class:`SteeringSignal` carries the regime's preferred
    categories so its ``consistency`` heuristic is self-contained.

    Parameters
    ----------
    macro_hist:
        Date-indexed macro panel carrying the z-scored columns
        (``cpi_yoy_z``, ``t10y2y_z``, ``hy_oas_z``), e.g. from
        ``macro_framework.macro.build_macro_panel``. Sliced read-only here.
    rebalance_date:
        The decision date. Only rows strictly before it are used.
    consistency_floor:
        Minimum consistency multiplier returned for a contradicting view
        (mirrors the later ``SteeringConfig.consistency_floor``).

    Returns
    -------
    SteeringSignal
        Frozen, deterministic per-rebalance signal.
    """
    rebalance_date = pd.Timestamp(rebalance_date)
    # Defensive PIT slice: strictly before the rebalance date (Requirement 2.2).
    as_of = macro_hist.loc[macro_hist.index < rebalance_date]

    zscore_summary: dict[str, float] = {}
    if not as_of.empty:
        latest = as_of.iloc[-1]
        for col in _REGIME_Z_SERIES:
            if col in as_of.columns:
                value = latest[col]
                if pd.notna(value):
                    zscore_summary[col] = float(value)

    regime_label = _classify_regime(zscore_summary)
    preferred = _REGIME_PREFERRED_CATEGORIES.get(regime_label, frozenset())

    return SteeringSignal(
        rebalance_date=rebalance_date,
        regime_label=regime_label,
        zscore_summary=zscore_summary,
        _preferred_categories=preferred,
        _consistency_floor=consistency_floor,
    )


def write_steering_signals(signals: Sequence[SteeringSignal], path: str | Path) -> None:
    """Persist per-rebalance steering signals to a parquet artifact (Requirement 2.4).

    Serializes a table indexed by ``rebalance_date`` with the ``regime_label``
    and the latest as-of z-summary columns (``cpi_yoy_z``, ``t10y2y_z``,
    ``hy_oas_z``). Only writes where told; the playbook chooses the
    ``data/macro_steering_signals_*.parquet`` name (Requirement 6.4 — new
    filename, never overwrites an existing artifact). An empty ``signals`` list
    writes a readable empty table with the same schema.

    Parameters
    ----------
    signals:
        The per-rebalance signals to persist (typically one per rebalance date).
    path:
        Destination parquet path (caller-chosen).
    """
    columns = ["rebalance_date", "regime_label", *_REGIME_Z_SERIES]
    records: list[dict[str, object]] = []
    for sig in signals:
        record: dict[str, object] = {
            "rebalance_date": pd.Timestamp(sig.rebalance_date),
            "regime_label": sig.regime_label,
        }
        for col in _REGIME_Z_SERIES:
            record[col] = sig.zscore_summary.get(col)
        records.append(record)

    df = pd.DataFrame(records, columns=columns)
    df = df.set_index("rebalance_date")
    df.to_parquet(Path(path))


# ---------------------------------------------------------------------------
# Task 2.5 — ViewSteerer: confidence shaping and gating.
#
# ``steer_views`` is the only place the measured contamination score and the
# macro-consistency signal touch the agent's views. Its sole effect is
# confidence/inclusion shaping (Requirement 3.4) — it NEVER changes a view's
# asset legs, expected excess, or rationale, and NEVER introduces a return /
# forecast objective (Requirement 3.5). The shaped views are consumed by the
# UNCHANGED ``LlmMacroAgent.views_to_bl``.
#
# Core formula (Requirements 3.1, 3.4):
#     adjusted_confidence = base_confidence * (1 - p_memorized) * consistency
# where ``consistency = signal.consistency(view)`` ALREADY carries the
# macro-consistency floor (task 2.4's ``characterize`` built the SteeringSignal
# with that floor). The floor is therefore NOT re-applied here — re-flooring
# would double-apply it. The result is clipped to ``[0, 1]``.
#
# Gating (Requirement 3.2): when ``p_memorized >= config.threshold`` the whole
# rebalance's views are excluded (an empty list is returned) — "down-weight or
# exclude that view" at the rebalance level.
#
# Passthrough (Requirements 1.6 additive / 1.7 weak): when ``not
# config.enabled`` OR ``p_memorized is None`` the input views are returned
# UNCHANGED. The composition (task 3.1) supplies ``p_memorized=None`` when the
# calibrator ``is_weak`` (steer_views never receives the scorer), so the
# weak-calibrator graceful-degradation case is handled here as the None case —
# the steered variant then equals plain Track A for that rebalance.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SteeringConfig:
    """Configuration for view confidence shaping and gating (Requirements 1.6, 3.1-3.5).

    Frozen so a config is a stable, hashable value shared across rebalances.

    Attributes
    ----------
    enabled:
        Master switch. ``False`` ⇒ ``steer_views`` returns the input views
        unchanged (scoring is additive; Requirement 1.6).
    threshold:
        ``p_memorized`` at/above which a rebalance's views are excluded — the
        hard contamination gate (Requirement 3.2). The research expects the
        anonymized macro prompts to yield uniformly low ``p_memorized``, so this
        gate may rarely or never fire; the ``(1 - p_memorized)`` discount applies
        continuously regardless.
    consistency_floor:
        The minimum macro-consistency multiplier. This is the floor the
        composition passes into ``characterize`` when building the
        :class:`SteeringSignal` (the single source of truth for the signal's
        floor). ``steer_views`` itself does NOT re-apply it: ``signal.consistency``
        already returns values in ``[consistency_floor, 1.0]``, so re-flooring
        here would double-apply it. The field lives on the config because the
        composition reads it from one place when constructing both the signal and
        this config.
    """

    enabled: bool = True
    threshold: float = 0.8
    consistency_floor: float = 0.5


def steer_views(
    views: list[MacroView],
    p_memorized: float | None,
    signal: SteeringSignal,
    config: SteeringConfig = SteeringConfig(),
) -> list[MacroView]:
    """Shape view confidence from contamination + macro consistency (R1.6, 3.1-3.5).

    Returns a **new** ``list[MacroView]`` of **new** ``MacroView`` objects; the
    input list and its views are never mutated. Only ``confidence`` is changed —
    ``asset_long``, ``asset_short``, ``expected_excess_annualized`` and
    ``rationale`` are copied through unchanged (Requirement 3.4). No
    return/forecast objective is ever introduced (Requirement 3.5).

    Behaviour:

    * **Passthrough** (Requirements 1.6, 1.7): if ``not config.enabled`` OR
      ``p_memorized is None``, the input views are returned UNCHANGED (the same
      content, returned as-is). The composition (task 3.1) supplies
      ``p_memorized=None`` when the calibrator ``is_weak``, so the weak-calibrator
      case degrades here to plain Track A.
    * **Gate** (Requirement 3.2): if ``p_memorized >= config.threshold`` the
      whole rebalance's views are excluded — an empty list is returned.
    * **Shape** (Requirements 3.1, 3.4): otherwise each view's confidence becomes
      ``clip(base * (1 - p_memorized) * signal.consistency(view), 0.0, 1.0)``.
      ``signal.consistency(view)`` already carries the macro-consistency floor
      (built by ``characterize``), so the floor is applied exactly once here.

    Pure and deterministic: equal inputs ⇒ equal output (no randomness, no I/O).

    Parameters
    ----------
    views:
        The agent's views for ONE rebalance (must share that rebalance's
        ``signal`` / ``p_memorized``).
    p_memorized:
        The measured contamination score in ``[0, 1]`` for this rebalance, or
        ``None`` when scoring failed / the calibrator is weak / scoring is off.
    signal:
        The per-rebalance :class:`SteeringSignal` from ``characterize``; supplies
        the floored ``consistency(view)`` multiplier.
    config:
        Steering configuration (enabled flag, gate threshold, consistency floor).

    Returns
    -------
    list[MacroView]
        Passthrough: the input views unchanged (disabled / ``p_memorized`` None).
        Gated: an empty list (``p_memorized >= threshold``).
        Shaped: a new list of new views with adjusted confidence only.
    """
    # Passthrough (Requirement 1.6 additive / 1.7 weak): return input unchanged.
    if not config.enabled or p_memorized is None:
        return views

    # Hard contamination gate (Requirement 3.2): drop the rebalance's views.
    if p_memorized >= config.threshold:
        return []

    from .llm_agent import MacroView  # read-only import (leaf consumer)

    contamination_discount = 1.0 - p_memorized
    steered: list[MacroView] = []
    for view in views:
        # consistency() already carries the macro-consistency floor (R3.1/3.4);
        # do NOT re-floor here (no double application).
        consistency = signal.consistency(view)
        adjusted = view.confidence * contamination_discount * consistency
        adjusted = min(1.0, max(0.0, adjusted))  # clip to [0, 1]
        steered.append(
            MacroView(
                asset_long=view.asset_long,
                asset_short=view.asset_short,
                expected_excess_annualized=view.expected_excess_annualized,
                confidence=adjusted,
                rationale=view.rationale,
            )
        )
    return steered


# ---------------------------------------------------------------------------
# Task 2.6 — Prompt-version variant agent (VariantMacroAgent).
#
# ``VariantMacroAgent`` runs ALTERNATIVE prompt versions over the same PIT prompt
# stream WITHOUT modifying the committed ``macro_framework.llm_agent`` module
# (Requirement 6.1). It subclasses the read-only ``LlmMacroAgent`` and supplies a
# custom ``instructions`` string and a ``prompt_version`` label.
#
# Override mechanics (design "VariantMacroAgent" — Override mechanics):
#   * The base bakes the module-level ``AGENT_INSTRUCTIONS`` into
#     ``MacroViewSignature.__doc__`` INSIDE the private ``_ensure_ready``; the
#     base ``__init__`` exposes no instructions hook. So the subclass OVERRIDES
#     ``_ensure_ready``, mirroring the base's LM / ``dspy.Predict`` / diskcache
#     wiring line-for-line, but building the signature docstring from
#     ``self.instructions`` instead. This duplicates the base's private setup —
#     accepted as the additive cost of not editing ``llm_agent.py``; it depends on
#     a private internal of a read-only module and is a Revalidation Trigger if
#     the base ``_ensure_ready`` changes.
#   * Per-variant cache (Requirement 4.4): the base ``_cache_key`` keys on the
#     module-level ``PROMPT_VERSION`` constant ("v1") ONLY — not an instance
#     attribute — so two variants sharing one cache dir would alias each other's
#     cached responses. Each variant therefore MUST use a DISTINCT ``cache_dir``
#     and must never write the base ``CACHE_DIR`` (``.llm_cache/``). ``__init__``
#     requires ``cache_dir`` (no default) to make this explicit; prior versions
#     are preserved by their distinct cache dirs.
# ---------------------------------------------------------------------------


class VariantMacroAgent(LlmMacroAgent):
    """A prompt-version variant of ``LlmMacroAgent`` (Requirements 4.1, 4.4).

    Runs an alternative prompt version over the same point-in-time prompt stream
    without editing the base agent module. Stores a custom ``instructions``
    string and a ``prompt_version`` label, and overrides the private
    ``_ensure_ready`` to build the DSPy signature's docstring from
    ``self.instructions`` instead of the module-level ``AGENT_INSTRUCTIONS``,
    while keeping the LM config, ``dspy.Predict``, and the diskcache wiring
    equivalent to the base.

    A distinct ``cache_dir`` per variant is REQUIRED (Requirement 4.4): the base
    ``_cache_key`` keys only on the module-level ``PROMPT_VERSION``, so variants
    sharing a cache would alias each other. The variant never writes the base
    ``CACHE_DIR``; prior versions are preserved by their distinct cache dirs.
    """

    def __init__(
        self,
        *,
        instructions: str,
        prompt_version: str,
        cache_dir: str | Path,
        **kwargs: object,
    ) -> None:
        """Build a variant with custom ``instructions`` and an isolated cache.

        Parameters
        ----------
        instructions:
            The variant's agent instructions; replaces the module-level
            ``AGENT_INSTRUCTIONS`` as the DSPy signature docstring (Requirement
            4.1).
        prompt_version:
            A safe version label (e.g. ``"v2"``) stored for later artifact naming
            (e.g. ``prompt_refinement_<version>_scores.json``; Requirement 4.4).
        cache_dir:
            A DISTINCT diskcache directory for this variant (Requirement 4.4).
            Must not be the base ``CACHE_DIR``; distinct versions must use
            distinct dirs so cached responses never alias across variants.
        **kwargs:
            Forwarded to ``LlmMacroAgent.__init__`` (``asset_map``, ``model``,
            ``api_base``, ``api_key_env``, ``temperature``, ``max_tokens``), so
            the base wiring is reused unchanged.
        """
        self.instructions = instructions
        self.prompt_version = prompt_version
        super().__init__(cache_dir=cache_dir, **kwargs)

    def _ensure_ready(self) -> None:
        """Lazy DSPy/diskcache wiring using the variant's ``instructions`` (R4.1).

        Mirrors ``LlmMacroAgent._ensure_ready`` line-for-line — the same lazy
        guard, ``dotenv`` load, credential check, ``dspy.LM`` config,
        ``dspy.Predict`` build, and ``diskcache.Cache(self.cache_dir)`` — but
        sets ``MacroViewSignature.__doc__`` from ``self.instructions`` instead of
        the module-level ``AGENT_INSTRUCTIONS``. Constructing ``dspy.LM`` /
        ``dspy.Predict`` does NOT call the network (only ``views_for_state`` →
        ``self._predict`` would), so this is offline-safe to run.

        The base ``llm_agent.py`` is NOT imported here for its instructions; only
        the variant's own ``self.instructions`` drives the prompt. The diskcache
        is created at ``self.cache_dir`` (the per-variant dir), never the base
        ``CACHE_DIR`` (Requirement 4.4).
        """
        if self._predict is not None:
            return
        import diskcache
        import dspy
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} not set in environment / .env")

        self._lm = dspy.LM(
            model=self.model,
            api_key=api_key,
            api_base=self.api_base,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        dspy.settings.configure(lm=self._lm)

        class MacroViewSignature(dspy.Signature):
            __doc__ = self.instructions  # variant instructions, NOT AGENT_INSTRUCTIONS
            macro_state: dict = dspy.InputField(
                desc="z-scored macro state: cpi_yoy_z, t10y2y_z, hy_oas_z (float each)"
            )
            assets:      list = dspy.InputField(
                desc="anonymized assets [{id, category, trailing_12m_return, trailing_vol_ann}]"
            )
            reasoning:   str  = dspy.OutputField(
                desc="2-4 sentence causal analysis of the macro state"
            )
            views_json:  str  = dspy.OutputField(
                desc='JSON array: [{"asset_long":"Asset_X","asset_short":"Asset_Y or null",'
                     '"expected_excess_annualized":float,"confidence":0..1,"rationale":str}]'
            )

        self._predict = dspy.Predict(MacroViewSignature)
        self._cache = diskcache.Cache(str(self.cache_dir))


# ---------------------------------------------------------------------------
# Task 2.7 — Score distribution reporting (score_distribution_report).
#
# A PURE summary over a list of recall_guard ``GuardedScore`` objects, used by
# nb12 (per prompt version) and nb11/eval (per variant) to report the measured
# contamination distribution alongside the head-to-head metrics (Requirements
# 4.2, 5.2). It makes NO NIM/network calls and NO I/O — it only reads dataclass
# fields off the already-computed scores.
#
# What it computes:
#   * The ``p_memorized`` distribution (mean / median / p90) over the parse-OK
#     scores ONLY — failure records (``parse_ok`` False / ``p_memorized`` None)
#     are excluded from the distribution so a failed score never drags the mean
#     toward 0.
#   * ``parse_fail_rate`` = fraction of scores with ``parse_ok`` False (i.e.
#     ``p_memorized`` None) over the total.
#   * ``n_scored`` = total number of scores (OK + failed).
#   * ``memguard_confidence_mean`` over the OK scores — REPORT-ONLY (design Data
#     Models note): it is surfaced for diagnostics but is NEVER a steering input,
#     because it carries the scorer's ``raw_confidence`` and using it as a factor
#     would break the non-predictive guarantee.
#
# What it deliberately does NOT do:
#   * Read the directional ``signal`` or ``raw_confidence`` (non-predictive
#     boundary — design Non-Goals); no ``signal_mean`` / ``raw_confidence_mean``.
#   * Invent ``holdout_auc`` from the scores: ``holdout_auc`` is a *scorer*
#     property, not a ``GuardedScore`` field, so it is accepted only as an
#     optional keyword arg and echoed into the dict when provided (omitted
#     otherwise).
#
# Graceful edges: an empty list or an all-failed list yields well-defined values
# (no ZeroDivision, no NaN). Distribution stats fall back to a documented sentinel
# (``_UNDEFINED_DISTRIBUTION_STAT`` = 0.0) when there is no parse-OK score to
# summarize; ``n_scored`` / ``parse_fail_rate`` make that case interpretable.
# ---------------------------------------------------------------------------

# Documented sentinel for a distribution statistic that is undefined because no
# parse-OK score is available to summarize (empty / all-failed input). Chosen as
# a finite 0.0 so the report stays a plain ``dict[str, float]`` (JSON-friendly)
# and never emits NaN; ``n_scored`` and ``parse_fail_rate`` disambiguate it.
_UNDEFINED_DISTRIBUTION_STAT = 0.0


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolated percentile of an already-sorted, non-empty list.

    ``pct`` is in ``[0, 1]`` (e.g. ``0.9`` for p90). Uses the standard
    "linear interpolation between closest ranks" rule, matching ``numpy``'s
    default; a single-element list returns that element. Pure, no I/O.
    """
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = pct * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def score_distribution_report(
    scores: Sequence[GuardedScore],
    *,
    holdout_auc: float | None = None,
) -> dict[str, float]:
    """Summarize the measured ``p_memorized`` distribution over a score log.

    Pure aggregation over a sequence of ``recall_guard.GuardedScore`` objects for
    evaluation reporting (Requirements 4.2, 5.2). Makes NO NIM/network calls and
    NO I/O — it only reads dataclass fields off the already-computed scores.

    The distribution statistics (``p_mem_mean``, ``p_mem_median``, ``p_mem_p90``)
    are computed over the **parse-OK** scores only (``parse_ok`` True with a
    non-``None`` ``p_memorized``); failure records are excluded so a failed score
    never pulls the distribution toward 0. ``parse_fail_rate`` is the fraction of
    scores that failed to parse (``parse_ok`` False / ``p_memorized`` None) over
    the total, and ``n_scored`` is the total count (OK + failed).

    ``memguard_confidence_mean`` is reported over the OK scores for diagnostics
    only — it is **never** a steering input (design Data Models note), because it
    carries the scorer's ``raw_confidence`` and using it as a factor would break
    the non-predictive guarantee. The directional ``signal`` / ``raw_confidence``
    are deliberately never read (non-predictive boundary).

    ``holdout_auc`` is a *scorer* property, not a ``GuardedScore`` field, so it is
    not invented from the scores: pass it explicitly to surface it in the report;
    when omitted it is absent from the returned dict.

    Empty-list and all-failed inputs are handled gracefully (no ZeroDivision, no
    NaN): the distribution statistics fall back to the documented sentinel
    ``0.0`` (``_UNDEFINED_DISTRIBUTION_STAT``) while ``n_scored`` /
    ``parse_fail_rate`` keep the result interpretable.

    Parameters
    ----------
    scores:
        The per-rebalance / per-prompt-version ``GuardedScore`` log (typically
        from ``ScoringAdapter.score_rebalances``).
    holdout_auc:
        Optional calibrator held-out AUC (a scorer property). Included in the
        report only when provided.

    Returns
    -------
    dict[str, float]
        ``{p_mem_mean, p_mem_median, p_mem_p90, memguard_confidence_mean,
        parse_fail_rate, n_scored}`` (plus ``holdout_auc`` when supplied). All
        values are plain ``float``.
    """
    n_scored = len(scores)

    ok_p_memorized: list[float] = []
    ok_memguard: list[float] = []
    n_failed = 0
    for score in scores:
        # A score is usable for the distribution only when it parsed AND carries
        # a p_memorized; otherwise it counts as a parse failure.
        if score.parse_ok and score.p_memorized is not None:
            ok_p_memorized.append(float(score.p_memorized))
            if score.memguard_confidence is not None:
                ok_memguard.append(float(score.memguard_confidence))
        else:
            n_failed += 1

    parse_fail_rate = (n_failed / n_scored) if n_scored else 0.0

    if ok_p_memorized:
        ordered = sorted(ok_p_memorized)
        p_mem_mean = sum(ordered) / len(ordered)
        p_mem_median = _percentile(ordered, 0.5)
        p_mem_p90 = _percentile(ordered, 0.9)
    else:
        p_mem_mean = _UNDEFINED_DISTRIBUTION_STAT
        p_mem_median = _UNDEFINED_DISTRIBUTION_STAT
        p_mem_p90 = _UNDEFINED_DISTRIBUTION_STAT

    memguard_confidence_mean = (
        sum(ok_memguard) / len(ok_memguard)
        if ok_memguard
        else _UNDEFINED_DISTRIBUTION_STAT
    )

    report: dict[str, float] = {
        "p_mem_mean": float(p_mem_mean),
        "p_mem_median": float(p_mem_median),
        "p_mem_p90": float(p_mem_p90),
        "memguard_confidence_mean": float(memguard_confidence_mean),
        "parse_fail_rate": float(parse_fail_rate),
        "n_scored": float(n_scored),
    }
    if holdout_auc is not None:
        report["holdout_auc"] = float(holdout_auc)
    return report
