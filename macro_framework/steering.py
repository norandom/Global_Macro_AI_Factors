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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from recall_guard import MemoryGuardedScorer
from recall_guard.dataset.fmp_corpora import build_calibration

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from recall_guard import GuardedScore

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
