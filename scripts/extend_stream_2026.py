"""Task 8.1 — extend the factor stream to 2026: the post-cutoff natural experiment (data-v3).

Extends the nb13/nb14 walk-forward (PIT deployable + non-PIT diagnostic + naive
directional eval) beyond 2024-12 with the SAME renderer, calibrator
(``openai/gpt-oss-20b`` @ cutoff 2024-06-01), recall guard and 0.7/0.3 HRP+BL
blend, as far as the macro panel allows. The 2019-2024 segment REPLAYS the
persisted v1 loadings/scores (zero NIM calls; the nb11/nb13 pre-scored replay
pattern); live NIM calls happen ONLY for the new 2025+ monthly rebalances.
The cheap comparison lines (nb07 baseline, nb08 track B) are re-run over the
same extended window (no LLM), and the contrast / luck-vs-skill / tear-sheet
artifacts are re-cut over the extended span under NEW ``*_ext2026`` filenames
(published v1/v2 artifacts are never overwritten).

Falsifiable prediction (reported either way in the split table): the
PIT-vs-non-PIT p_memorized premium (+0.528 in-training) collapses toward zero
post-cutoff, while return behavior stays comparable.

Data sources: FRED live via ``mf.build_macro_panel()`` with a patched
web loader (falls back to the committed ``data/macro_panel_monthly.parquet``
when FRED is unreachable; the source is recorded in the run header);
prices via yfinance (documented DB substitution, mirrors nb11/nb13/nb14).

Reproducible: ``uv run python scripts/extend_stream_2026.py`` (needs
``NVIDIA_API_KEY`` in ``.env`` for the ~90-150 live NIM calls).
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterable, Literal, Mapping, get_args

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

import macro_framework as mf
from macro_framework import factor_scoring as fs
from macro_framework import macro as macro_module
from macro_framework import steering
from macro_framework.evaluation import metric_block
from macro_framework.factor_scoring import _median, _paired_cohens_d
from macro_framework.ssr import ssr_inference

warnings.filterwarnings("ignore")

# --- constants (mirror nb13/nb14 exactly) ---------------------------------- #
DATA = REPO / "data"          # INPUTS always come from the canonical data dir
#: Independent simulation streams write to their OWN directory so two runs with different
#: SIM_STARTs cannot overwrite each other. Artifact NAMES are unchanged inside it, so every
#: downstream consumer works by pointing at the directory rather than learning new stems.
#: Defaults reproduce the published behaviour exactly (writes into data/, 2019 start).
OUT = Path(os.environ.get("STREAM_OUT_DIR") or DATA)
CSV_OUT = OUT / "csv_mirrors"
TEAR_OUT = OUT / "tear_sheet"
INIT_CASH = 10_000.0
SIM_START = os.environ.get("STREAM_SIM_START") or "2019-01-01"
SIM_END_EXT = "2026-06-30"  # task 8.1: extend to 2026-06 (~18 new monthly rebalances)
PRICE_FETCH_END = "2026-07-01"
LOOKBACK_DAYS = 756
TILT = 0.30  # nb09 final blend = 0.7*HRP + 0.3*BL

# task 4.2: OFF by default -> byte-identical published behavior. When set to a
# dict (e.g. {"min_scale": 0.20}) the combine seam swaps the constant BIL 0.25
# pin for the regime-steered pin from macro_framework.derisk_cash_pin.
REGIME_OVERLAY: dict | None = None


def _regime_cash_pin(returns_hist, overlay, base_cash_pin: float = 0.25) -> float:
    """BIL cash pin for the HRP-CVaR base.

    ``overlay is None`` -> exactly ``base_cash_pin`` (the published constant;
    byte-identical). Otherwise return the correlation-de-risk pin, which rises
    in high-correlation windows and equals ``base_cash_pin`` in calm ones. The
    allocation math is untouched: the result still flows through
    ``hrp_cvar_weights_with_fixed({"BIL": pin})``.
    """
    if overlay is None:
        return base_cash_pin
    risky = overlay.get("base_risky_symbols") or tuple(
        c for c in returns_hist.columns if c != "BIL")
    return mf.derisk_cash_pin(
        returns_hist, base_risky_symbols=tuple(risky),
        base_cash_pin=base_cash_pin, min_scale=overlay.get("min_scale", 0.20))

NIM_MODEL = "openai/gpt-oss-20b"
CUTOFF = date(2024, 6, 1)
SLUG = NIM_MODEL.replace("/", "_")
CAL_DIR = DATA / f"factor_calibrator_{SLUG}"
TIMEOUT_S = 120.0  # reasoning model; NvidiaLM's 15 s default is too tight
MAX_WORKERS = 6

PANEL_Z_COLS = ["cpi_yoy_z", "t10y2y_z", "hy_oas_z"]
PANEL_RAW_COLS = ["cpi_yoy", "t10y2y", "hy_oas"]

# nb08 track B parameters (replicated verbatim).
TRACK_B = {"horizon": 3, "n_paths": 10_000, "block_size": 3, "bootstrap_window_months": 12}

# NOTE: the published premium is READ from factor_contrast_summary_v1.json at run time
# (see the S8 reproduction check). The frozen constant here read 0.5282818618139323 and
# went stale when nb14 was re-run on 2026-07-27 (-> 0.3995801783).
PREMIUM_REPRO_TOL = 0.02
# Replayed 2019-2024 PIT equity vs the published factor_equity_v1.parquet.
# ponytail: 2e-3 relative, not exact — a fresh yfinance pull can carry new
# dividend adjustments/revisions; the replayed weights themselves are exact.
EQUITY_REL_TOL = 2e-3

# Post-cutoff premium counts as "collapsed" below this fraction of in-training.
COLLAPSE_FRACTION = 0.25

_DIR_RE = re.compile(r"Direction[\s\*_:]*(-?1|0)")
_CONF_RE = re.compile(r"Confidence[\s\*_:]*([01](?:\.\d+)?|\.\d+)")


# --------------------------------------------------------------------------- #
# Pure helpers (offline-tested in tests/test_extend_stream_2026.py)            #
# --------------------------------------------------------------------------- #


def synth_loadings_reply(loadings: dict[str, float] | None) -> str:
    """Reply text that ``parse_loadings`` round-trips to exactly these loadings.

    The v1 replies were never persisted, but the parsed loadings were
    (``factor_loadings_v1.parquet`` stores the clipped per-axis values), so a
    JSON re-render replays them exactly. ``None`` (a v1 parse failure) yields
    ``""`` which stays unparsed — the same base-allocation fallback as v1.

    Args:
        loadings: axis -> loading for a parsed v1 row, or ``None``.

    Returns:
        A JSON reply string, or ``""`` for the not-parsed case.
    """
    if loadings is None:
        return ""
    return json.dumps({axis: float(loadings[axis]) for axis in fs.MACRO_AXES})


def completed_months_only(panel: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Keep only rows of completed months (drop the current, incomplete month).

    The panel is month-end stamped (``ME`` resample); a row stamped inside the
    current month would mix a partial month into the PIT stream.

    Args:
        panel: the month-end-indexed macro panel.
        today: reference date (defaults to now; injectable for tests).

    Returns:
        The panel restricted to rows strictly before the current month start.
    """
    now = pd.Timestamp(today) if today is not None else pd.Timestamp.now()
    month_start = now.normalize().replace(day=1)
    return panel.loc[panel.index < month_start]


def classify_premium_outcome(in_training_mean: float, post_cutoff_mean: float) -> str:
    """State the falsifiable-prediction outcome either way (task 8.1).

    Prediction: the contamination premium collapses toward zero post-cutoff
    (the model cannot recall unseen dates). "Collapsed" = the post-cutoff mean
    delta fell below ``COLLAPSE_FRACTION`` of the in-training mean delta.

    Args:
        in_training_mean: mean non-PIT − PIT p_memorized delta, dates <= cutoff.
        post_cutoff_mean: mean delta, dates > cutoff.

    Returns:
        A one-sentence outcome statement carrying both numbers.
    """
    numbers = f"(in-training {in_training_mean:+.4f} -> post-cutoff {post_cutoff_mean:+.4f})"
    if abs(post_cutoff_mean) < COLLAPSE_FRACTION * abs(in_training_mean):
        return (
            "PREDICTION CONFIRMED: the p_memorized premium collapsed toward zero "
            f"post-cutoff {numbers} — the model cannot recall unseen dates."
        )
    return (
        "PREDICTION FALSIFIED: the p_memorized premium did NOT collapse post-cutoff "
        f"{numbers} — reported either way per task 8.1."
    )


def split_contrast_table(contrast_df: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    """The in-training vs post-cutoff split of the PIT-vs-non-PIT premium.

    Pairs with a NaN on either side are dropped (a failed score carries no
    premium), mirroring ``run_pit_vs_nonpit_contrast``'s pair-dropping rule.

    Args:
        contrast_df: per-date frame with ``pit_p`` / ``nonpit_p`` columns,
            indexed by rebalance date.
        cutoff: the model's training cutoff (in-training = index <= cutoff).

    Returns:
        ``{"in_training": {...}, "post_cutoff": {...}, "full_stream": {...},
        "prediction_outcome": str}`` where each segment carries
        ``n_pairs`` / ``mean_delta`` / ``median_delta`` / ``paired_d``.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    valid = contrast_df.dropna(subset=["pit_p", "nonpit_p"])

    def _segment(mask: pd.Series) -> dict:
        deltas = (valid.loc[mask, "nonpit_p"] - valid.loc[mask, "pit_p"]).astype(float).tolist()
        return {
            "n_pairs": len(deltas),
            "mean_delta": (sum(deltas) / len(deltas)) if deltas else 0.0,
            "median_delta": _median(deltas),
            "paired_d": _paired_cohens_d(deltas),
        }

    table = {
        "in_training": _segment(valid.index <= cutoff_ts),
        "post_cutoff": _segment(valid.index > cutoff_ts),
        "full_stream": _segment(pd.Series(True, index=valid.index)),
    }
    table["prediction_outcome"] = classify_premium_outcome(
        table["in_training"]["mean_delta"], table["post_cutoff"]["mean_delta"]
    )
    return table


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a binomial proportion (the nb13 S2 convention)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


# --------------------------------------------------------------------------- #
# Data acquisition                                                             #
# --------------------------------------------------------------------------- #


def _fetch_fred_series_web(series_id: str) -> pd.Series:
    """FRED series via the public fredgraph.csv endpoint (no key, no DB)."""
    import requests

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    frame = pd.read_csv(io.StringIO(resp.text))
    date_col = frame.columns[0]
    out = pd.Series(
        pd.to_numeric(frame[series_id], errors="coerce").values,
        index=pd.DatetimeIndex(pd.to_datetime(frame[date_col])),
        name=series_id,
    ).dropna()
    if out.empty:
        raise ValueError(f"empty FRED web series {series_id!r}")
    return out


def build_panel() -> tuple[pd.DataFrame, str]:
    """FRED-live macro panel via ``mf.build_macro_panel``; committed fallback.

    Patches the module-global ``load_fred_series`` (DB-backed; the Postgres DB
    is absent here) with the fredgraph.csv web loader so the UNCHANGED
    ``build_macro_panel`` assembly/z-scoring runs against live FRED. When FRED
    is unreachable the committed ``data/macro_panel_monthly.parquet`` is used
    (the exact panel nb13/nb14 consumed). Either way only completed months
    are kept and the source is returned for the run header.
    """
    original = macro_module.load_fred_series
    macro_module.load_fred_series = _fetch_fred_series_web
    try:
        panel = mf.build_macro_panel()
        source = "fred_live (fredgraph.csv via patched load_fred_series)"
    except Exception as exc:  # noqa: BLE001 -- FRED unreachable -> committed panel
        panel = pd.read_parquet(DATA / "macro_panel_monthly.parquet")
        source = (
            "committed data/macro_panel_monthly.parquet "
            f"(FRED live rebuild unavailable: {type(exc).__name__})"
        )
    finally:
        macro_module.load_fred_series = original

    panel.index = pd.DatetimeIndex(panel.index)
    return completed_months_only(panel), source


def fetch_prices(symbols: list[str]) -> pd.DataFrame:
    """Daily adjusted closes via yfinance (nb13's fetch, extended to mid-2026)."""
    import yfinance as yf

    want = symbols + ["SPY"]
    last_exc: Exception | None = None
    for _attempt in range(6):
        try:
            raw = yf.download(want, start="2014-01-01", end=PRICE_FETCH_END,
                              auto_adjust=True, progress=False, threads=False)
            close = raw["Close"] if ("Close" in raw.columns.get_level_values(0)) else raw
            close = close[want].copy()
            close.index = pd.DatetimeIndex(close.index)
            if close[symbols].dropna(how="all").shape[0] > 1000:
                return close
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
        time.sleep(8)
    raise RuntimeError(f"price fetch failed after retries: {last_exc!r}")


# --------------------------------------------------------------------------- #
# Live NIM helpers (the nb13/nb14 truncation-resilience patterns)               #
# --------------------------------------------------------------------------- #


def _generate_big(lm, prompts: list[str], max_tokens: int = 2048) -> list:
    """Parallel generate at a larger completion budget.

    gpt-oss-20b is a reasoning model: it can burn the 512-token default on its
    reasoning chain before emitting the requested JSON, so generation runs at
    2048 up front and retries at 4096 (nb13/nb14 live-measured precedent).
    Failures are returned as exceptions (recorded rows, never crashes).
    """

    def _one(prompt: str):
        try:
            return lm.generate(prompt, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001 -- keep the failure as a failed row
            return exc

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        return list(ex.map(_one, prompts))


def _reply_text(reply) -> str:
    return "" if isinstance(reply, BaseException) else reply.content


def _generate_and_parse(lm, prompts: list[str], dates: list[pd.Timestamp], label: str):
    """Generate loadings replies (2048 up front) + ONE batched 4096 format-retry.

    Returns ``(texts, parsed)`` where ``parsed[i]`` is a ``RegimeLoadings`` or
    ``None`` (a repeat failure stays a failed row — the module fallbacks handle
    it downstream per R4.3).
    """
    replies = _generate_big(lm, prompts, max_tokens=2048)
    texts = [_reply_text(r) for r in replies]
    parsed = [fs.parse_loadings(t, rb) for t, rb in zip(texts, dates)]
    bad = [i for i, rl in enumerate(parsed) if rl is None]
    if bad:
        for i, r in zip(bad, _generate_big(lm, [prompts[i] for i in bad], max_tokens=4096)):
            texts[i] = _reply_text(r)
            parsed[i] = fs.parse_loadings(texts[i], dates[i])
        still = sum(1 for i in bad if parsed[i] is None)
        print(f"  [{label}] format-retried {len(bad)} replies at max_tokens=4096; still unparsed: {still}")
    return texts, parsed


def _score_with_retry(scorer, prompts: list[str]) -> list:
    """score_many + one individual retry per failed score (nb13/nb14 pattern)."""
    scores = list(scorer.score_many(prompts, max_workers=MAX_WORKERS))
    for i, sc in enumerate(scores):
        if sc.p_memorized is None:
            scores[i] = scorer.score(prompts[i])
    return scores


# --------------------------------------------------------------------------- #
# Replay assembly                                                              #
# --------------------------------------------------------------------------- #


def factor_score_from_row(p_memorized, fail_reason) -> fs.FactorScore:
    """Rebuild a ``FactorScore`` from a persisted scores-parquet row."""
    if pd.isna(p_memorized):
        reason = fail_reason if isinstance(fail_reason, str) and fail_reason else "replayed_nan"
        return fs.FactorScore(p_memorized=None, parse_ok=False, fail_reason=reason)
    return fs.FactorScore(p_memorized=float(p_memorized), parse_ok=True, fail_reason=None)


def loadings_dict_from_row(row: pd.Series) -> dict[str, float] | None:
    """Per-axis loadings dict from a persisted loadings-parquet row (None if unparsed)."""
    if not bool(row["parse_ok"]):
        return None
    return {axis: float(row[axis]) for axis in fs.MACRO_AXES}


# --------------------------------------------------------------------------- #
# Dated Factor evidence (task 6.1 — design.md 'Dated Factor Evidence')         #
# --------------------------------------------------------------------------- #

Variant = Literal["pit", "nonpit_diagnostic"]
ResponseOrigin = Literal[
    "raw_nim",
    "reconstructed_from_v1_loadings",
    "generation_failed",
]
EvidenceKey = tuple[Variant, date]

_VARIANTS = get_args(Variant)
_RESPONSE_ORIGINS = get_args(ResponseOrigin)
_LOADING_FIELDS = tuple(f"loading_{axis}" for axis in fs.MACRO_AXES)


class ReplayValidationError(ValueError):
    """Replay evidence integrity failure — blocks the Factor bundle (never warn-and-continue)."""


@dataclass(frozen=True)
class DatedFactorEvidence:
    """One immutable replay-evidence record, naturally keyed by ``(variant, rebalance_date)``.

    Identical ``pit_prompt_text`` on multiple rebalance dates is legitimate
    (macro inputs are rounded for prompt rendering); the DATE is part of the
    replay identity, never injected into the anonymized prompt (R6.1/R6.2).
    """

    variant: Variant
    rebalance_date: date
    segment: str
    pit_prompt_text: str
    pit_prompt_sha256: str
    source_prompt_text: str
    source_prompt_sha256: str
    response_text: str
    response_sha256: str
    response_origin: ResponseOrigin
    score_p_memorized: float | None
    score_parse_ok: bool
    score_fail_reason: str | None
    score_origin: str
    loading_inflation: float | None
    loading_growth: float | None
    loading_credit_stress: float | None
    loading_policy: float | None
    loading_risk_appetite: float | None
    loadings_parse_ok: bool
    source_artifact: str | None
    source_artifact_sha256: str | None
    evidence_id: str


def sha256_text(text: str) -> str:
    """Canonical hex digest of a text field (UTF-8 bytes)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_evidence_id(record: DatedFactorEvidence) -> str:
    """SHA-256 over canonical JSON of every identity-bearing field except ``evidence_id``.

    Same canonicalization idiom as ``macro_framework.llm_agent._cache_key``:
    ``json.dumps(payload, sort_keys=True)``; the date is ISO-rendered.
    """
    payload = dataclasses.asdict(record)
    del payload["evidence_id"]
    payload["rebalance_date"] = record.rebalance_date.isoformat()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def with_evidence_id(record: DatedFactorEvidence) -> DatedFactorEvidence:
    """Return the record with its deterministic ``evidence_id`` filled in."""
    return dataclasses.replace(record, evidence_id=compute_evidence_id(record))


def validate_evidence_records(
    records: Iterable[DatedFactorEvidence],
    expected_keys: Iterable[EvidenceKey],
) -> Mapping[EvidenceKey, DatedFactorEvidence]:
    """Validate dated replay evidence and freeze it into an immutable mapping.

    Every integrity violation raises :class:`ReplayValidationError` (a
    ``ValueError``) naming the offending variant/date/field — never a warning
    (R6.3). Duplicate prompt TEXT across dates is valid; a duplicate
    ``(variant, rebalance_date)`` key is not. Coverage must match
    ``expected_keys`` exactly: missing and unexpected keys both raise.
    """
    out: dict[EvidenceKey, DatedFactorEvidence] = {}
    last_date: dict[str, date] = {}
    for rec in records:
        if rec.variant not in _VARIANTS:
            raise ReplayValidationError(
                f"unsupported variant {rec.variant!r} at {rec.rebalance_date}; "
                f"expected one of {_VARIANTS}")
        if not isinstance(rec.rebalance_date, date) or isinstance(rec.rebalance_date, datetime):
            raise ReplayValidationError(
                f"rebalance_date must be a datetime.date for variant {rec.variant!r}, "
                f"got {type(rec.rebalance_date).__name__}")
        where = f"({rec.variant!r}, {rec.rebalance_date.isoformat()})"
        if rec.response_origin not in _RESPONSE_ORIGINS:
            raise ReplayValidationError(
                f"unsupported response_origin {rec.response_origin!r} for {where}; "
                f"expected one of {_RESPONSE_ORIGINS}")
        if not (isinstance(rec.segment, str) and rec.segment.strip()):
            raise ReplayValidationError(f"blank segment for {where}")
        if not (isinstance(rec.score_origin, str) and rec.score_origin.strip()):
            raise ReplayValidationError(f"blank score_origin for {where}")
        if rec.score_p_memorized is not None and not math.isfinite(rec.score_p_memorized):
            raise ReplayValidationError(f"non-finite score_p_memorized for {where}")
        for field in _LOADING_FIELDS:
            val = getattr(rec, field)
            if val is not None and not math.isfinite(val):
                raise ReplayValidationError(f"non-finite {field} for {where}")
        if rec.score_parse_ok:
            if rec.score_p_memorized is None or rec.score_fail_reason is not None:
                raise ReplayValidationError(
                    f"score parse-state inconsistent for {where}: score_parse_ok=True "
                    f"requires a p_memorized value and score_fail_reason=None")
        elif rec.score_p_memorized is not None or not rec.score_fail_reason:
            raise ReplayValidationError(
                f"score parse-state inconsistent for {where}: score_parse_ok=False "
                f"requires score_p_memorized=None and a non-empty score_fail_reason")
        loadings = [getattr(rec, field) for field in _LOADING_FIELDS]
        if rec.loadings_parse_ok:
            if any(v is None for v in loadings):
                raise ReplayValidationError(
                    f"loadings parse-state inconsistent for {where}: loadings_parse_ok=True "
                    f"requires all five loadings present")
        elif any(v is not None for v in loadings):
            raise ReplayValidationError(
                f"loadings parse-state inconsistent for {where}: loadings_parse_ok=False "
                f"requires all five loadings None")
        for stem in ("pit_prompt", "source_prompt", "response"):
            if getattr(rec, f"{stem}_sha256") != sha256_text(getattr(rec, f"{stem}_text")):
                raise ReplayValidationError(f"{stem}_sha256 mismatch for {where}")
        if (rec.source_artifact is None) != (rec.source_artifact_sha256 is None):
            raise ReplayValidationError(
                f"source_artifact_sha256 must be present iff source_artifact is present for {where}")
        if rec.evidence_id != compute_evidence_id(rec):
            raise ReplayValidationError(f"evidence_id mismatch for {where}")
        key: EvidenceKey = (rec.variant, rec.rebalance_date)
        if key in out:
            raise ReplayValidationError(
                f"duplicate evidence key {where}: dated evidence must be unique per "
                f"variant and rebalance date (duplicate prompt text alone is valid)")
        prev = last_date.get(rec.variant)
        if prev is not None and rec.rebalance_date <= prev:
            raise ReplayValidationError(
                f"rebalance dates not strictly increasing for variant {rec.variant!r}: "
                f"{prev.isoformat()} then {rec.rebalance_date.isoformat()}")
        last_date[rec.variant] = rec.rebalance_date
        out[key] = rec

    expected = set(expected_keys)
    missing = sorted(expected - set(out))
    if missing:
        raise ReplayValidationError(f"missing expected evidence key(s): {missing}")
    extra = sorted(set(out) - expected)
    if extra:
        raise ReplayValidationError(
            f"unexpected evidence key(s) beyond expected coverage: {extra}")
    return MappingProxyType(out)


# --------------------------------------------------------------------------- #
# Dated Factor evidence persistence (task 6.2)                                 #
# --------------------------------------------------------------------------- #

EVIDENCE_TABLE_NAME = "factor_evidence_ext2026.parquet"


def sha256_file(path: Path | str) -> str:
    """Hex digest of a file's bytes (source-artifact identity, R7.3/R7.6)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def response_origin_for(segment: str, response_text: str) -> ResponseOrigin:
    """Origin label for a response (R7.3): never claim raw provenance for replays.

    v1 raw replies were never persisted, so every ``replayed_v1`` row is a
    deterministic reconstruction from the persisted loadings — including the
    ``""`` v1-parse-failure case. On the live segment ``""`` marks the
    generation-exception path (``_reply_text``); a returned-but-unparseable
    reply stays ``raw_nim`` with ``loadings_parse_ok=False``.
    """
    if segment == "replayed_v1":
        return "reconstructed_from_v1_loadings"
    return "raw_nim" if response_text else "generation_failed"


def build_dated_evidence(
    *,
    variant: Variant,
    rebalance_date: date | datetime,
    segment: str,
    pit_prompt: str,
    source_prompt: str,
    response_text: str,
    score: fs.FactorScore,
    loadings: dict[str, float] | None,
    source_artifact: str | None = None,
    source_artifact_sha256: str | None = None,
) -> DatedFactorEvidence:
    """One dated evidence record with hashes, origins, and ``evidence_id`` filled in.

    Prompt and response text pass through byte-for-byte — never mutated, never
    date-stamped (the DATE lives in the key, R6.1/R6.2). Failures are RETAINED
    as records: a generation exception becomes ``response_origin=
    "generation_failed"`` and a failed score keeps an explicit
    ``score_fail_reason`` (R6.3). Reconstructed ``replayed_v1`` rows must name
    their source artifact and its hash (R7.3).
    """
    rb = rebalance_date.date() if isinstance(rebalance_date, datetime) else rebalance_date
    origin = response_origin_for(segment, response_text)
    if origin == "reconstructed_from_v1_loadings" and not (
            source_artifact and source_artifact_sha256):
        raise ReplayValidationError(
            f"reconstructed evidence for ({variant!r}, {rb.isoformat()}) requires "
            f"source_artifact and source_artifact_sha256 — never claim raw provenance (R7.3)")
    p = None if score.p_memorized is None else float(score.p_memorized)
    # ponytail: score_parse_ok collapses to "p present" — FactorScore pairs them by contract.
    return with_evidence_id(DatedFactorEvidence(
        variant=variant,
        rebalance_date=rb,
        segment=segment,
        pit_prompt_text=pit_prompt,
        pit_prompt_sha256=sha256_text(pit_prompt),
        source_prompt_text=source_prompt,
        source_prompt_sha256=sha256_text(source_prompt),
        response_text=response_text,
        response_sha256=sha256_text(response_text),
        response_origin=origin,
        score_p_memorized=p,
        score_parse_ok=p is not None,
        score_fail_reason=None if p is not None else (score.fail_reason or "scoring_failed"),
        score_origin="replayed_v1_scores" if segment == "replayed_v1" else "live_nim",
        loadings_parse_ok=loadings is not None,
        source_artifact=source_artifact,
        source_artifact_sha256=source_artifact_sha256,
        evidence_id="",
        **{f"loading_{axis}": (float(loadings[axis]) if loadings is not None else None)
           for axis in fs.MACRO_AXES},
    ))


def write_evidence_table(
    records: Iterable[DatedFactorEvidence],
    expected_keys: Iterable[EvidenceKey],
    run_dir: Path | str,
) -> Path:
    """Validate then persist dated evidence into a NEW empty run staging directory.

    Validation (duplicate/missing keys, hash, origin, and parse-state
    integrity) runs BEFORE any byte is written (R6.3). The destination must be
    a fresh empty directory: historical v1..v3 artifacts are read-only inputs
    and are never edited in place (R7.6). One flat scalar Parquet row per
    ``(variant, rebalance_date)`` — design.md 'Factor Evidence Table'.
    """
    validated = validate_evidence_records(records, expected_keys)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise ReplayValidationError(
            f"evidence run staging directory is not empty: {run_dir} — dated evidence "
            f"is written only into a new empty run directory (R7.6)")
    df = pd.DataFrame([dataclasses.asdict(rec) for rec in validated.values()])
    path = run_dir / EVIDENCE_TABLE_NAME
    df.to_parquet(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Dated Factor replay (task 6.3 — exact (variant, date) resolution)            #
# --------------------------------------------------------------------------- #


def resolve_dated_evidence(
    evidence: Mapping[EvidenceKey, DatedFactorEvidence],
    variant: Variant,
    rebalance_date: date | datetime,
) -> DatedFactorEvidence:
    """The immutable evidence record for exactly ``(variant, rebalance_date)``.

    Replay identity is the DATE plus variant, never the prompt text (R6.2). A
    missing key raises :class:`ReplayValidationError` immediately — never a
    ``not_pre_scored`` degrade or an empty-reply fallback (R6.3).
    """
    rb = rebalance_date.date() if isinstance(rebalance_date, datetime) else rebalance_date
    rec = evidence.get((variant, rb))
    if rec is None:
        raise ReplayValidationError(
            f"missing dated evidence for ({variant!r}, {rb.isoformat()}): replay "
            f"resolves by exact variant and rebalance date, never by prompt text")
    return rec


def _require_prompt_match(rec: DatedFactorEvidence, prompt: str, stage: str) -> None:
    """The simulation-rendered anonymized prompt must EQUAL the dated evidence.

    Compared against ``pit_prompt_text``/``pit_prompt_sha256`` for BOTH
    variants: ``factor_rebalance`` always re-renders the anonymized PIT prompt;
    the identifying non-PIT source prompt is never re-rendered inside the sim.
    Identical text on OTHER dates is legitimate; a mismatch on THIS date is
    fatal (R6.2/R6.3).
    """
    if prompt != rec.pit_prompt_text or sha256_text(prompt) != rec.pit_prompt_sha256:
        raise ReplayValidationError(
            f"simulation-rendered prompt does not match dated evidence for "
            f"({rec.variant!r}, {rec.rebalance_date.isoformat()}) at {stage}: "
            f"sha256 {sha256_text(prompt)} != {rec.pit_prompt_sha256}")


class _ReplayScorer:
    """Per-date replay scorer closed over exactly ONE dated ``FactorScore``.

    Exposes the surface ``factor_rebalance`` uses (``is_weak`` + ``score`` +
    ``score_many``) unchanged — no scorer API change (design L586). Zero NIM
    calls: the score was measured when the evidence was produced; the prompt is
    only checked for equality against the dated record (a mismatch raises,
    never degrades to an unscored result).
    """

    is_weak = False  # the loaded calibrator is strong (asserted in main)

    def __init__(self, record: DatedFactorEvidence) -> None:
        self._record = record
        self._score = fs.FactorScore(
            p_memorized=record.score_p_memorized,
            parse_ok=record.score_parse_ok,
            fail_reason=record.score_fail_reason)

    def score(self, prompt: str) -> fs.FactorScore:
        _require_prompt_match(self._record, prompt, "score")
        return self._score

    def score_many(self, prompts, *, max_workers: int = 8):
        return [self.score(p) for p in prompts]


def dated_replay_closures(
    rec: DatedFactorEvidence,
) -> tuple[Callable[[str], str], _ReplayScorer]:
    """``(generate_loadings, scorer)`` closed over ONE dated evidence record."""

    def gen(prompt: str) -> str:
        _require_prompt_match(rec, prompt, "generate_loadings")
        return rec.response_text

    return gen, _ReplayScorer(rec)


def make_dated_replay_weight_fn(
    *,
    variant: Variant,
    evidence: Mapping[EvidenceKey, DatedFactorEvidence],
    agent: object,
    build_inputs: Callable,
    combine: Callable,
    failures: list[ReplayValidationError] | None = None,
    consumed: dict[EvidenceKey, dict[str, object]] | None = None,
) -> Callable[[dict], pd.Series]:
    """A walk-forward ``weight_fn`` that replays immutable DATED evidence.

    Per rebalance it resolves the evidence by exact ``(variant,
    ctx["rebalance_date"])`` BEFORE invoking the existing sequential rebalance
    path, closes the selected response and score over that one call via
    :func:`dated_replay_closures`, and delegates to the UNCHANGED
    ``fs.make_factor_weight_fn`` — no scorer API or PIT prompt renderer change
    (R6.2, design 'Corrected Factor Replay').

    ``mf.build_walk_forward_targets`` swallows weight_fn exceptions ("holding
    previous"), so every :class:`ReplayValidationError` is ALSO appended to the
    run-local ``failures`` list; the caller re-raises after the sim returns.
    Never warn-and-continue (R6.3).
    """

    def weight_fn(ctx: dict) -> pd.Series:
        try:
            rec = resolve_dated_evidence(evidence, variant, ctx["rebalance_date"])
            if consumed is not None:
                # task 6.4: fingerprint this consumption under the SIMULATION's
                # own (variant, date) key — the audit later proves it equals
                # the immutable source evidence at exactly that key.
                rb = ctx["rebalance_date"]
                record_consumption(
                    consumed,
                    (variant, rb.date() if isinstance(rb, datetime) else rb),
                    rec)
            gen, scorer = dated_replay_closures(rec)
            inner = fs.make_factor_weight_fn(
                generate_loadings=gen, scorer=scorer, agent=agent,
                build_inputs=build_inputs, combine=combine)
            return inner(ctx)
        except ReplayValidationError as exc:
            if failures is not None:
                failures.append(exc)
            raise

    return weight_fn


# --------------------------------------------------------------------------- #
# Source-to-consumption replay audit (task 6.4)                                 #
# --------------------------------------------------------------------------- #

REPLAY_AUDIT_NAME = "factor_ext2026_replay_audit.json"

#: The identity-bearing fields one consumption delivers: prompt, response,
#: score, and loadings identities plus ``evidence_id`` (the single cheapest
#: equality token — design 'Factor Evidence Table').
_FINGERPRINT_FIELDS = (
    "evidence_id", "pit_prompt_sha256", "response_sha256",
    "score_p_memorized", "score_parse_ok", "score_fail_reason",
    "loadings_parse_ok", *_LOADING_FIELDS,
)


def consumption_fingerprint(rec: DatedFactorEvidence) -> dict[str, object]:
    """Run-local fingerprint of the values a simulation consumption delivers."""
    return {f: getattr(rec, f) for f in _FINGERPRINT_FIELDS}


def record_consumption(
    consumed: dict[EvidenceKey, dict[str, object]],
    key: EvidenceKey,
    rec: DatedFactorEvidence,
) -> None:
    """Record one consumption fingerprint under the SIMULATION's ``key``.

    Each ``(variant, date)`` is legitimately consumed twice per run — the
    walk-forward weight_fn pass and the decision-log pass — so a REPEATED
    IDENTICAL consumption is valid; two consumptions of the same key with
    different fingerprints are an inconsistent duplicate and raise (R6.3).
    The run is sequential: one plain run-local dict is sufficient (design).
    """
    variant, rb = key
    fp = consumption_fingerprint(rec)
    entry = consumed.get(key)
    if entry is None:
        consumed[key] = {"evidence": fp}
    elif entry["evidence"] != fp:
        diff = [f for f in _FINGERPRINT_FIELDS if entry["evidence"][f] != fp[f]]
        raise ReplayValidationError(
            f"inconsistently duplicated consumption for ({variant!r}, {rb.isoformat()}): "
            f"fingerprint fields differ across consumptions: {diff}")


def record_decision_identity(
    consumed: dict[EvidenceKey, dict[str, object]],
    key: EvidenceKey,
    rec: DatedFactorEvidence,
    decision: object,
) -> None:
    """Prove the RESULTING decision identity derives from the consumed evidence.

    A parsed decision must carry the record's own score and loadings; any
    mismatch means a cross-associated or altered value reached the decision
    and raises before the decision log is published (R6.4).
    """
    variant, rb = key
    where = f"({variant!r}, {rb.isoformat()})"
    entry = consumed.get(key)
    if entry is None or entry["evidence"] != consumption_fingerprint(rec):
        raise ReplayValidationError(
            f"decision identity recorded without a matching consumption "
            f"fingerprint for {where}")
    if bool(decision.parse_ok) != bool(rec.loadings_parse_ok):
        raise ReplayValidationError(
            f"decision parse_ok does not match consumed evidence for {where}: "
            f"decision {decision.parse_ok} vs evidence {rec.loadings_parse_ok}")
    dec_loadings = dict(decision.loadings.loadings) if decision.loadings is not None else None
    src_loadings = ({axis: getattr(rec, f"loading_{axis}") for axis in fs.MACRO_AXES}
                    if rec.loadings_parse_ok else None)
    if dec_loadings != src_loadings:
        raise ReplayValidationError(
            f"decision loadings do not match consumed evidence for {where}: "
            f"{dec_loadings} vs {src_loadings}")
    if decision.parse_ok and decision.p_memorized != rec.score_p_memorized:
        raise ReplayValidationError(
            f"decision p_memorized does not match consumed evidence for {where}: "
            f"{decision.p_memorized} vs {rec.score_p_memorized}")
    entry["decision"] = {"p_memorized": decision.p_memorized,
                        "parse_ok": bool(decision.parse_ok),
                        "steered": bool(decision.steered),
                        "loadings": dec_loadings}


def validate_source_to_consumption(
    evidence: Mapping[EvidenceKey, DatedFactorEvidence],
    consumed: Mapping[EvidenceKey, Mapping[str, object]],
    expected_keys: Iterable[EvidenceKey] | None = None,
) -> None:
    """Prove the dated SOURCE evidence equals the values the simulation CONSUMED.

    Runs after both variant lines and BEFORE any publishable portfolio output
    (targets, equity, decision logs, metrics, completion state). An absent,
    cross-associated, or altered key/value raises :class:`ReplayValidationError`
    naming the variant, date, and field(s) — never warn-and-continue (R6.3,
    R6.4). Inconsistent duplicates already raised at recording time.
    """
    expected = set(expected_keys) if expected_keys is not None else set(evidence)
    missing_src = sorted(expected - set(evidence))
    if missing_src:
        raise ReplayValidationError(
            f"expected key(s) absent from source evidence: {missing_src}")
    unconsumed = sorted(expected - set(consumed))
    if unconsumed:
        raise ReplayValidationError(
            f"expected evidence key(s) never consumed by the simulation: {unconsumed}")
    stray = sorted(set(consumed) - expected)
    if stray:
        raise ReplayValidationError(
            f"consumption recorded for key(s) outside the expected set: {stray}")
    for key in sorted(expected):
        variant, rb = key
        where = f"({variant!r}, {rb.isoformat()})"
        src = consumption_fingerprint(evidence[key])
        got = consumed[key]["evidence"]
        if got != src:
            diff = [f for f in _FINGERPRINT_FIELDS if got.get(f) != src[f]]
            raise ReplayValidationError(
                f"consumed values do not equal source evidence for {where}: "
                f"mismatched field(s) {diff} — cross-associated or altered evidence")
        dec = consumed[key].get("decision")
        if dec is not None:
            rec = evidence[key]
            src_loadings = ({axis: getattr(rec, f"loading_{axis}") for axis in fs.MACRO_AXES}
                            if rec.loadings_parse_ok else None)
            if (bool(dec["parse_ok"]) != bool(rec.loadings_parse_ok)
                    or dec["loadings"] != src_loadings
                    or (dec["parse_ok"] and dec["p_memorized"] != rec.score_p_memorized)):
                raise ReplayValidationError(
                    f"resulting decision identity does not equal source evidence "
                    f"for {where}: cross-associated or altered decision inputs")


def write_replay_audit_summary(
    evidence: Mapping[EvidenceKey, DatedFactorEvidence],
    consumed: Mapping[EvidenceKey, Mapping[str, object]],
    expected_keys: Iterable[EvidenceKey],
    out_dir: Path | str,
) -> Path:
    """Validate source==consumption, then persist the PASSING audit summary.

    The summary (run window, variants, counts — the task-6.4 slice of R7.4)
    is written ONLY under the run output directory; immutable source evidence
    is never re-written or re-hashed. On any audit failure the raise happens
    BEFORE this file exists, so no publishable output can claim a passed audit.
    """
    expected = sorted(set(expected_keys))
    validate_source_to_consumption(evidence, consumed, expected)
    dates = sorted({rb for _, rb in expected})
    summary = {
        "audit": "source_to_consumption_replay",
        "result": "pass",
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "variants": sorted({v for v, _ in expected}),
        "window": {"first_rebalance": dates[0].isoformat(),
                   "last_rebalance": dates[-1].isoformat()},
        "counts": {"expected_keys": len(expected),
                   "consumed_keys": len(consumed),
                   "source_records": len(evidence),
                   "decision_identities": sum(
                       1 for e in consumed.values() if "decision" in e)},
        "checked_fields": list(_FINGERPRINT_FIELDS),
    }
    out = Path(out_dir) / REPLAY_AUDIT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return out


# --------------------------------------------------------------------------- #
# Run-local Factor metric records (task 6.6)                                   #
# --------------------------------------------------------------------------- #

FACTOR_METRIC_RECORDS_NAME = "factor_metric_records_ext2026.json"
FACTOR_METRIC_RECORDS_SCHEMA = "factor_run.metric_records.v1"
_FACTOR_RECORD_SHA256S_FIELD = "record_sha256s"
_FACTOR_CONTENT_SHA256_FIELD = "content_sha256"
_FACTOR_CONTENT_FIELDS = (
    "schema",
    "market_snapshot",
    "ssr_settings",
    "source_streams",
    "records",
    _FACTOR_RECORD_SHA256S_FIELD,
)
_FACTOR_BUNDLE_FIELDS = frozenset((*_FACTOR_CONTENT_FIELDS, _FACTOR_CONTENT_SHA256_FIELD))

_FACTOR_PRIMARY_PORTFOLIOS = (
    "factor_pit_ext2026",
    "factor_nonpit_diagnostic_ext2026",
)
_FACTOR_DIFFERENTIAL_PORTFOLIO = "factor_nonpit_minus_pit_ext2026"
_FACTOR_STREAM_ARTIFACTS = {
    "factor_pit_ext2026": "factor_equity_ext2026.parquet",
    "factor_nonpit_diagnostic_ext2026": "factor_nonpit_diagnostic_equity_ext2026.parquet",
}
_FACTOR_RECORD_KEYS = (
    ("factor_pit_ext2026", mf.READER_SCHEMA),
    ("factor_pit_ext2026", mf.LEGACY_SCHEMA),
    ("factor_nonpit_diagnostic_ext2026", mf.READER_SCHEMA),
    ("factor_nonpit_diagnostic_ext2026", mf.LEGACY_SCHEMA),
    (_FACTOR_DIFFERENTIAL_PORTFOLIO, mf.DIFFERENTIAL_SCHEMA),
    ("factor_pit_ext2026", mf.ATTRIBUTION_SCHEMA),
    ("factor_nonpit_diagnostic_ext2026", mf.ATTRIBUTION_SCHEMA),
    ("factor_pit_ext2026", mf.CRISIS_SCHEMA),
    ("factor_nonpit_diagnostic_ext2026", mf.CRISIS_SCHEMA),
)


def _active_portfolio_value(value: pd.Series) -> pd.Series:
    """Trim a flat pre-start stub while retaining the anchor before first movement."""
    if not isinstance(value, pd.Series) or value.empty:
        raise ValueError("portfolio value must be a non-empty pandas Series")
    moving = value[value.ne(value.iloc[0])]
    if moving.empty:
        return value
    first_move = moving.index[0]
    prior = value.index[value.index < first_move]
    start = prior[-1] if len(prior) else first_move
    return value.loc[start:]


def load_completed_snapshot_bil_returns(
    snapshot_dir: Path | str,
    return_index: pd.DatetimeIndex,
    *,
    anchor: pd.Timestamp,
) -> tuple[pd.Series, dict[str, object]]:
    """Load exact-interval BIL total returns from a completed, hash-valid snapshot.

    The snapshot validator owns completion, inventory, and byte-integrity checks.
    The shared ``factor_returns_on`` contract requires the actual preceding
    portfolio value anchor plus every portfolio return session before calculating
    returns. Missing or non-finite required BIL observations fail rather than
    widening an interval, being filled, or being set to zero (R2.1, R7.4).
    """
    from scripts import build_basket_long as snapshot_producer

    snapshot_dir = Path(snapshot_dir)
    snapshot_producer.validate_market_snapshot(snapshot_dir)
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("completed") is not True:
        raise ValueError(f"{snapshot_dir}: snapshot manifest is not completed")
    if manifest.get("cash_symbol") != "BIL":
        raise ValueError(
            f"{snapshot_dir}: Factor SSR requires cash_symbol 'BIL', got "
            f"{manifest.get('cash_symbol')!r}"
        )
    if manifest.get("total_return_field") != snapshot_producer.TOTAL_RETURN_FIELD:
        raise ValueError(
            f"{snapshot_dir}: BIL cash input is not the approved adjusted total-return field"
        )

    cash_path = snapshot_dir / "cash_market_total_return.parquet"
    levels = pd.read_parquet(cash_path)
    if "BIL" not in levels:
        raise ValueError(f"{cash_path}: required BIL total-return level is absent")
    bil = levels["BIL"].rename("BIL")
    cash_returns = mf.factor_returns_on(bil, return_index, anchor=anchor).rename("BIL")
    inventory = manifest["files"][cash_path.name]
    return cash_returns, {
        "schema": manifest["schema"],
        "snapshot_id": manifest["snapshot_id"],
        "manifest_sha256": sha256_file(manifest_path),
        "cash_benchmark_id": "BIL",
        "cash_semantics": "adjusted_total_return",
        "cash_file": cash_path.name,
        "cash_file_sha256": inventory["sha256"],
        "cash_anchor": anchor,
        "cash_start": cash_returns.index[0],
        "cash_end": cash_returns.index[-1],
        "cash_n_obs": len(cash_returns),
    }


def load_completed_snapshot_market_returns(
    snapshot_dir: Path | str,
    return_index: pd.DatetimeIndex,
    *,
    value_index: pd.DatetimeIndex,
) -> tuple[pd.Series, dict[str, object]]:
    """Load the longest valid SPY suffix ending on the performance endpoint.

    Initial unavailable benchmark history may shorten attribution explicitly. Once
    SPY coverage begins, every portfolio anchor and return date through the common
    performance endpoint is required by ``factor_returns_on``; internal gaps,
    non-finite selected values, and absent endpoint coverage fail rather than being
    intersected or dropped (R3.3, R3.4, R3.7).
    """
    from scripts import build_basket_long as snapshot_producer

    snapshot_dir = Path(snapshot_dir)
    snapshot_producer.validate_market_snapshot(snapshot_dir)
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    benchmark_id = manifest.get("benchmark_symbol")
    if benchmark_id != "SPY":
        raise ValueError(
            f"{snapshot_dir}: Factor attribution requires benchmark_symbol 'SPY', got "
            f"{benchmark_id!r}"
        )
    if manifest.get("total_return_field") != snapshot_producer.TOTAL_RETURN_FIELD:
        raise ValueError(
            f"{snapshot_dir}: SPY benchmark is not the approved adjusted total-return field"
        )
    if not isinstance(return_index, pd.DatetimeIndex) or return_index.empty:
        raise ValueError("performance return_index must be a non-empty DatetimeIndex")
    if not isinstance(value_index, pd.DatetimeIndex):
        raise ValueError("portfolio value_index must be a DatetimeIndex")
    if len(value_index) != len(return_index) + 1 or not value_index[1:].equals(return_index):
        raise ValueError(
            "portfolio value_index must contain exactly one preceding anchor plus return_index"
        )

    market_path = snapshot_dir / "cash_market_total_return.parquet"
    levels = pd.read_parquet(market_path)
    if benchmark_id not in levels:
        raise ValueError(f"{market_path}: required SPY total-return level is absent")
    observed = levels[benchmark_id].dropna().rename(benchmark_id)
    if observed.empty:
        raise ValueError(f"{market_path}: SPY has no observed total-return levels")
    performance_end = return_index[-1]
    if performance_end not in observed.index:
        raise ValueError(
            f"{market_path}: SPY benchmark is missing required performance end "
            f"{performance_end}"
        )

    anchors = value_index[:-1]
    eligible = np.flatnonzero(anchors >= observed.index[0])
    if not len(eligible):
        raise ValueError(
            f"{market_path}: SPY coverage begins after the performance window"
        )
    start_pos = int(eligible[0])
    attribution_index = return_index[start_pos:]
    anchor = anchors[start_pos]
    market_returns = mf.factor_returns_on(
        observed, attribution_index, anchor=anchor
    ).rename(benchmark_id)
    inventory = manifest["files"][market_path.name]
    return market_returns, {
        "benchmark_id": benchmark_id,
        "benchmark_semantics": "adjusted_total_return",
        "benchmark_file": market_path.name,
        "benchmark_file_sha256": inventory["sha256"],
        "benchmark_observed_start": observed.index[0],
        "benchmark_observed_end": observed.index[-1],
        "attribution_anchor": anchor,
        "attribution_start": market_returns.index[0],
        "attribution_end": market_returns.index[-1],
        "attribution_n_obs": len(market_returns),
    }


def _window_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{pd.Timestamp(start).date()}..{pd.Timestamp(end).date()}"


def _metric_meta(
    portfolio_id: str,
    label: str,
    returns: pd.Series,
    *,
    total_return_basis: str,
    cash_benchmark_id: str,
) -> mf.LineMetadata:
    return mf.LineMetadata(
        portfolio_id=portfolio_id,
        label=label,
        window_label=_window_label(returns.index[0], returns.index[-1]),
        currency_basis="legacy_mixed_local_quotes",
        total_return_basis=total_return_basis,
        cash_benchmark_id=cash_benchmark_id,
    )


def _require_factor_record_fields(
    row: Mapping[str, object], fields: Iterable[str]
) -> None:
    missing = sorted(set(fields) - set(row))
    if missing:
        raise ValueError(
            f"{row['schema']} record is missing required field(s): {', '.join(missing)}"
        )


def _require_factor_record_binding(
    row: Mapping[str, object], top_level: str, typed_result: str
) -> None:
    if row[top_level] != row[typed_result]:
        raise ValueError(
            f"{row['schema']} record has contradictory {top_level}/{typed_result}: "
            f"{row[top_level]!r} != {row[typed_result]!r}"
        )


def _validate_factor_record_window(row: Mapping[str, object]) -> dict[str, object]:
    """Require one exact producer-owned financial window on each run-local record."""
    checked = mf.validate_report_row(row)
    expected = _window_label(
        pd.Timestamp(checked["start"]), pd.Timestamp(checked["end"])
    )
    if checked["window_label"] != expected:
        raise ValueError(
            f"run-local record window_label {checked['window_label']!r} does not match "
            f"its explicit financial window {expected!r}"
        )

    if checked["schema"] == mf.ATTRIBUTION_SCHEMA:
        fields = tuple(
            f"raw_market_model_{field.name}"
            for field in dataclasses.fields(mf.MarketAttribution)
        )
        _require_factor_record_fields(checked, fields)
        if checked["raw_market_model_kind"] != "raw_market_model":
            raise ValueError(
                "attribution record raw_market_model_kind must be 'raw_market_model'"
            )
        for top_level, typed_result in (
            ("start", "raw_market_model_start"),
            ("end", "raw_market_model_end"),
            ("n_obs", "raw_market_model_n_obs"),
            ("periods_per_year", "raw_market_model_periods_per_year"),
        ):
            _require_factor_record_binding(checked, top_level, typed_result)
    elif checked["schema"] == mf.CRISIS_SCHEMA:
        fields = tuple(field.name for field in dataclasses.fields(mf.CrisisMetrics))
        _require_factor_record_fields(checked, fields)
        for top_level, typed_result in (
            ("start", "anchor"),
            ("end", "actual_end"),
            ("n_obs", "n_returns"),
        ):
            _require_factor_record_binding(checked, top_level, typed_result)
    return checked


def _factor_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _factor_exact_fields(
    value: Mapping[str, object], expected: Iterable[str], name: str
) -> None:
    expected_set = set(expected)
    missing = sorted(expected_set - set(value))
    extra = sorted(set(value) - expected_set)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"extra {extra}")
        raise ValueError(f"{name} has an invalid field catalog: {'; '.join(details)}")


def _factor_timestamp(value: object, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a valid timestamp") from exc
    if pd.isna(timestamp) or timestamp.tz is not None:
        raise ValueError(f"{name} must be a timezone-naive non-NaT timestamp")
    return timestamp


def _factor_integer(
    value: object, name: str, *, minimum: int = 0, strictly_greater: bool = False
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    invalid = result <= minimum if strictly_greater else result < minimum
    if invalid:
        relation = ">" if strictly_greater else ">="
        raise ValueError(f"{name} must be {relation} {minimum}")
    return result


def _factor_finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _factor_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _factor_values_equal(left: object, right: object) -> bool:
    if isinstance(left, Real) and isinstance(right, Real):
        if math.isnan(float(left)) and math.isnan(float(right)):
            return True
    return bool(left == right)


def _require_factor_typed_projection(
    row: Mapping[str, object],
    typed_result: object,
    *,
    name: str,
    prefix: str = "",
) -> None:
    """Require a report record to remain a verbatim shared-result projection."""
    for field in dataclasses.fields(typed_result):
        key = f"{prefix}{field.name}"
        if key not in row or not _factor_values_equal(row[key], getattr(typed_result, field.name)):
            raise ValueError(
                f"{name} diverges from shared {type(typed_result).__name__} field {key}"
            )


def _factor_canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        _json_record_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256_text(encoded)


def _factor_record_sha256s(records: Iterable[Mapping[str, object]]) -> list[str]:
    return [_factor_canonical_sha256(row) for row in records]


def _factor_typed_result_sha256(result: object) -> str:
    if not dataclasses.is_dataclass(result) or isinstance(result, type):
        raise ValueError("shared financial result must be a dataclass instance")
    return _factor_canonical_sha256(dataclasses.asdict(result))


def _factor_content_sha256(bundle: Mapping[str, object]) -> str:
    return _factor_canonical_sha256(
        {field: bundle[field] for field in _FACTOR_CONTENT_FIELDS}
    )


def _trusted_factor_metric_sources(
    pit_value: pd.Series,
    nonpit_value: pd.Series,
    *,
    snapshot_dir: Path | str,
    crisis_start: str | pd.Timestamp,
    crisis_end: str | pd.Timestamp,
) -> dict[str, object]:
    """Recompute Task 6.7 lineage from build-local observations and source bytes.

    Nothing in the candidate bundle participates in this reconstruction. Portfolio
    windows come from the two in-memory equity series, market lineage comes from the
    validated snapshot manifest and Parquet bytes, and typed attribution/crisis
    results come directly from the shared finance contracts.
    """
    pit_value = _active_portfolio_value(pit_value)
    nonpit_value = _active_portfolio_value(nonpit_value)
    pit_returns = metric_block(pit_value)["returns"]
    nonpit_returns = metric_block(nonpit_value)["returns"]
    if not pit_returns.index.equals(nonpit_returns.index):
        raise ValueError("PIT and non-PIT performance returns must have identical indexes")
    if pit_value.index[0] != nonpit_value.index[0]:
        raise ValueError("PIT and non-PIT performance returns must have the same value anchor")

    _cash_returns, snapshot = load_completed_snapshot_bil_returns(
        snapshot_dir, pit_returns.index, anchor=pit_value.index[0]
    )
    market_returns, market_lineage = load_completed_snapshot_market_returns(
        snapshot_dir, pit_returns.index, value_index=pit_value.index
    )
    snapshot.update(market_lineage)
    attribution_index = market_returns.index
    attributions = {
        "factor_pit_ext2026": mf.raw_market_model_attribution(
            pit_returns.loc[attribution_index], market_returns
        ),
        "factor_nonpit_diagnostic_ext2026": mf.raw_market_model_attribution(
            nonpit_returns.loc[attribution_index], market_returns
        ),
    }
    crises = {
        "factor_pit_ext2026": mf.crisis_metrics(
            pit_value, crisis_start, crisis_end
        ),
        "factor_nonpit_diagnostic_ext2026": mf.crisis_metrics(
            nonpit_value, crisis_start, crisis_end
        ),
    }
    if any(result is None for result in crises.values()):
        raise ValueError(
            "required crisis window has no preceding anchor or no portfolio observations"
        )

    source_streams: dict[str, object] = {}
    returns_by_portfolio = {
        "factor_pit_ext2026": pit_returns,
        "factor_nonpit_diagnostic_ext2026": nonpit_returns,
    }
    for portfolio_id, returns in returns_by_portfolio.items():
        source_streams[portfolio_id] = {
            "artifact": _FACTOR_STREAM_ARTIFACTS[portfolio_id],
            "start": returns.index[0],
            "end": returns.index[-1],
            "n_obs": len(returns),
            "attribution_result_sha256": _factor_typed_result_sha256(
                attributions[portfolio_id]
            ),
            "crisis_result_sha256": _factor_typed_result_sha256(
                crises[portfolio_id]
            ),
        }
    source_streams[_FACTOR_DIFFERENTIAL_PORTFOLIO] = {
        "comparison": _FACTOR_PRIMARY_PORTFOLIOS[1],
        "reference": _FACTOR_PRIMARY_PORTFOLIOS[0],
        "start": pit_returns.index[0],
        "end": pit_returns.index[-1],
        "n_obs": len(pit_returns),
    }
    return {
        "market_snapshot": snapshot,
        "source_streams": source_streams,
        "attributions": attributions,
        "crises": crises,
    }


def _require_factor_projection_fields(
    row: Mapping[str, object],
    expected: Mapping[str, object],
    fields: Iterable[str],
    *,
    name: str,
    prefix: str = "",
) -> None:
    """Require selected record fields to equal one normalized trusted projection."""
    for field in fields:
        key = f"{prefix}{field}"
        if key not in row or not _factor_values_equal(row[key], expected[field]):
            raise ValueError(f"{name} diverges from trusted shared field {key}")


def _require_trusted_factor_metric_sources(
    bundle: Mapping[str, object], trusted: Mapping[str, object]
) -> None:
    """Bind a structurally valid candidate to independently rebuilt Task 6.7 inputs."""
    bundle = _factor_mapping(_json_record_value(bundle), "Factor metric-record bundle")
    normalized_trusted = {
        **trusted,
        "attributions": {
            key: dataclasses.asdict(value)
            for key, value in trusted["attributions"].items()
        },
        "crises": {
            key: dataclasses.asdict(value)
            for key, value in trusted["crises"].items()
        },
    }
    trusted = _factor_mapping(
        _json_record_value(normalized_trusted), "trusted Factor metric inputs"
    )
    for field in ("market_snapshot", "source_streams"):
        if bundle[field] != trusted[field]:
            raise ValueError(
                f"Factor metric-record {field} diverges from trusted build-local inputs"
            )

    records = {
        (row["portfolio_id"], row["schema"]): row
        for row in bundle["records"]
    }
    attributions = _factor_mapping(trusted["attributions"], "trusted.attributions")
    crises = _factor_mapping(trusted["crises"], "trusted.crises")
    attribution_fields = tuple(
        field.name for field in dataclasses.fields(mf.MarketAttribution)
    )
    crisis_fields = tuple(field.name for field in dataclasses.fields(mf.CrisisMetrics))
    for portfolio_id in _FACTOR_PRIMARY_PORTFOLIOS:
        _require_factor_projection_fields(
            records[(portfolio_id, mf.ATTRIBUTION_SCHEMA)],
            _factor_mapping(
                attributions[portfolio_id], f"trusted.attributions.{portfolio_id}"
            ),
            attribution_fields,
            name=f"{portfolio_id} attribution",
            prefix="raw_market_model_",
        )
        _require_factor_projection_fields(
            records[(portfolio_id, mf.CRISIS_SCHEMA)],
            _factor_mapping(crises[portfolio_id], f"trusted.crises.{portfolio_id}"),
            crisis_fields,
            name=f"{portfolio_id} crisis",
        )


def _factor_attribution_from_record(
    row: Mapping[str, object], *, name: str
) -> mf.MarketAttribution:
    prefix = "raw_market_model_"
    kind = row[f"{prefix}kind"]
    if kind != "raw_market_model":
        raise ValueError(f"{name} kind must be 'raw_market_model'")
    numeric = {
        field: _factor_finite_real(row[f"{prefix}{field}"], f"{name}.{field}")
        for field in (
            "intercept_native_period",
            "intercept_ann_arithmetic",
            "intercept_se_hac",
            "intercept_t_hac",
            "beta",
            "r2",
        )
    }
    n_obs = _factor_integer(
        row[f"{prefix}n_obs"], f"{name}.n_obs", minimum=0, strictly_greater=True
    )
    start = _factor_timestamp(row[f"{prefix}start"], f"{name}.start")
    end = _factor_timestamp(row[f"{prefix}end"], f"{name}.end")
    periods_per_year = _factor_integer(
        row[f"{prefix}periods_per_year"],
        f"{name}.periods_per_year",
        minimum=0,
        strictly_greater=True,
    )
    hac_maxlags = _factor_integer(
        row[f"{prefix}hac_maxlags"], f"{name}.hac_maxlags", minimum=0
    )
    if start > end:
        raise ValueError(f"{name} start must be on or before end")
    if n_obs <= 2:
        raise ValueError(f"{name} requires more than two observations")
    if hac_maxlags >= n_obs:
        raise ValueError(f"{name}.hac_maxlags must be smaller than n_obs")
    if numeric["intercept_se_hac"] < 0.0:
        raise ValueError(f"{name}.intercept_se_hac must be non-negative")
    if not -1e-12 <= numeric["r2"] <= 1.0 + 1e-12:
        raise ValueError(f"{name}.r2 must lie in [0, 1]")
    expected_ann = numeric["intercept_native_period"] * periods_per_year
    if not math.isclose(
        numeric["intercept_ann_arithmetic"],
        expected_ann,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ValueError(
            f"{name}.intercept_ann_arithmetic does not equal native intercept "
            "times periods_per_year"
        )
    if numeric["intercept_se_hac"] == 0.0:
        if numeric["intercept_t_hac"] != 0.0:
            raise ValueError(f"{name}.intercept_t_hac is invalid for zero HAC standard error")
    else:
        expected_t = numeric["intercept_native_period"] / numeric["intercept_se_hac"]
        if not math.isclose(
            numeric["intercept_t_hac"], expected_t, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError(
                f"{name}.intercept_t_hac does not equal intercept divided by HAC standard error"
            )
    return mf.MarketAttribution(
        kind="raw_market_model",
        intercept_native_period=numeric["intercept_native_period"],
        intercept_ann_arithmetic=numeric["intercept_ann_arithmetic"],
        intercept_se_hac=numeric["intercept_se_hac"],
        intercept_t_hac=numeric["intercept_t_hac"],
        beta=numeric["beta"],
        r2=numeric["r2"],
        n_obs=n_obs,
        start=start,
        end=end,
        periods_per_year=periods_per_year,
        hac_maxlags=hac_maxlags,
    )


def _factor_crisis_from_record(
    row: Mapping[str, object], *, name: str
) -> mf.CrisisMetrics:
    requested_start = _factor_timestamp(row["requested_start"], f"{name}.requested_start")
    requested_end = _factor_timestamp(row["requested_end"], f"{name}.requested_end")
    anchor = _factor_timestamp(row["anchor"], f"{name}.anchor")
    first_return_date = _factor_timestamp(
        row["first_return_date"], f"{name}.first_return_date"
    )
    actual_end = _factor_timestamp(row["actual_end"], f"{name}.actual_end")
    episode_return = _factor_finite_real(row["episode_return"], f"{name}.episode_return")
    max_drawdown = _factor_finite_real(
        row["boundary_anchored_max_drawdown"],
        f"{name}.boundary_anchored_max_drawdown",
    )
    n_returns = _factor_integer(
        row["n_returns"], f"{name}.n_returns", minimum=0, strictly_greater=True
    )
    periods_per_year = _factor_integer(
        row["periods_per_year"],
        f"{name}.periods_per_year",
        minimum=0,
        strictly_greater=True,
    )
    volatility = row["volatility_ann"]
    if n_returns == 1:
        if volatility is None:
            # Canonical JSON uses ``null`` for the shared contract's undefined
            # one-return sample volatility. Reconstruct NaN only in the typed view.
            volatility_ann = float("nan")
        elif (
            isinstance(volatility, bool)
            or not isinstance(volatility, Real)
            or not math.isnan(float(volatility))
        ):
            raise ValueError(
                f"{name}.volatility_ann must be NaN in memory or null in canonical JSON "
                "for one return"
            )
        else:
            volatility_ann = float(volatility)
    else:
        volatility_ann = _factor_finite_real(volatility, f"{name}.volatility_ann")
        if volatility_ann < 0.0:
            raise ValueError(f"{name}.volatility_ann must be non-negative")

    if requested_start > requested_end:
        raise ValueError(f"{name} requested_start must be on or before requested_end")
    if not anchor < requested_start:
        raise ValueError(f"{name}.anchor must be strictly before requested_start")
    if not requested_start <= first_return_date <= actual_end <= requested_end:
        raise ValueError(
            f"{name} must satisfy requested_start <= first_return_date <= "
            "actual_end <= requested_end"
        )
    if (n_returns == 1) != (first_return_date == actual_end):
        raise ValueError(
            f"{name} first_return_date/actual_end are inconsistent with n_returns"
        )
    if not -1.0 <= max_drawdown <= 0.0:
        raise ValueError(
            f"{name}.boundary_anchored_max_drawdown must lie in [-1, 0]"
        )
    if episode_return < max_drawdown - 1e-12:
        raise ValueError(
            f"{name}.episode_return cannot be below its boundary-anchored max drawdown"
        )
    return mf.CrisisMetrics(
        requested_start=requested_start,
        requested_end=requested_end,
        anchor=anchor,
        first_return_date=first_return_date,
        actual_end=actual_end,
        episode_return=episode_return,
        boundary_anchored_max_drawdown=max_drawdown,
        volatility_ann=volatility_ann,
        n_returns=n_returns,
        periods_per_year=periods_per_year,
    )


def _validate_factor_metric_record_bundle(bundle: Mapping[str, object]) -> None:
    """Validate the complete run-local record set before any file is created."""
    bundle = _factor_mapping(bundle, "Factor metric-record bundle")
    _factor_exact_fields(bundle, _FACTOR_BUNDLE_FIELDS, "Factor metric-record bundle")
    if bundle["schema"] != FACTOR_METRIC_RECORDS_SCHEMA:
        raise ValueError(f"unknown Factor metric-record schema {bundle['schema']!r}")

    raw_records = bundle["records"]
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("Factor metric-record bundle must contain a non-empty records list")
    records = [
        _validate_factor_record_window(_factor_mapping(row, f"records[{position}]"))
        for position, row in enumerate(raw_records)
    ]
    actual_keys = [(row["portfolio_id"], row["schema"]) for row in records]
    expected_keys = list(_FACTOR_RECORD_KEYS)
    if actual_keys != expected_keys:
        missing = [key for key in expected_keys if actual_keys.count(key) == 0]
        duplicate = sorted({key for key in actual_keys if actual_keys.count(key) > 1})
        extra = [key for key in actual_keys if key not in expected_keys]
        raise ValueError(
            "Factor metric-record catalog must contain each approved portfolio/schema "
            f"exactly once in deterministic order; missing={missing}, "
            f"duplicate={duplicate}, extra={extra}, actual={actual_keys}"
        )
    by_key = dict(zip(actual_keys, records, strict=True))

    snapshot = _factor_mapping(bundle["market_snapshot"], "market_snapshot")
    snapshot_fields = (
        "schema",
        "snapshot_id",
        "manifest_sha256",
        "cash_benchmark_id",
        "cash_semantics",
        "cash_file",
        "cash_file_sha256",
        "cash_anchor",
        "cash_start",
        "cash_end",
        "cash_n_obs",
        "benchmark_id",
        "benchmark_semantics",
        "benchmark_file",
        "benchmark_file_sha256",
        "benchmark_observed_start",
        "benchmark_observed_end",
        "attribution_anchor",
        "attribution_start",
        "attribution_end",
        "attribution_n_obs",
    )
    _factor_exact_fields(snapshot, snapshot_fields, "market_snapshot")
    if snapshot["schema"] != "market_snapshot.v1":
        raise ValueError("market_snapshot.schema must be 'market_snapshot.v1'")
    snapshot_id = snapshot["snapshot_id"]
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise ValueError("market_snapshot.snapshot_id must be a non-empty string")
    _factor_sha256(snapshot["manifest_sha256"], "market_snapshot.manifest_sha256")
    if snapshot["cash_benchmark_id"] != "BIL":
        raise ValueError("market_snapshot.cash_benchmark_id must be 'BIL'")
    if snapshot["cash_semantics"] != "adjusted_total_return":
        raise ValueError("market_snapshot.cash_semantics must be 'adjusted_total_return'")
    if snapshot["cash_file"] != "cash_market_total_return.parquet":
        raise ValueError("market_snapshot.cash_file has an unexpected artifact identity")
    cash_file_sha256 = _factor_sha256(
        snapshot["cash_file_sha256"], "market_snapshot.cash_file_sha256"
    )
    if snapshot["benchmark_id"] != "SPY":
        raise ValueError("market_snapshot.benchmark_id must be 'SPY'")
    if snapshot["benchmark_semantics"] != "adjusted_total_return":
        raise ValueError("market_snapshot.benchmark_semantics must be 'adjusted_total_return'")
    if snapshot["benchmark_file"] != "cash_market_total_return.parquet":
        raise ValueError("market_snapshot.benchmark_file has an unexpected artifact identity")
    benchmark_file_sha256 = _factor_sha256(
        snapshot["benchmark_file_sha256"], "market_snapshot.benchmark_file_sha256"
    )
    if cash_file_sha256 != benchmark_file_sha256:
        raise ValueError("BIL and SPY lineage from one snapshot artifact must share its hash")

    cash_anchor = _factor_timestamp(snapshot["cash_anchor"], "market_snapshot.cash_anchor")
    cash_start = _factor_timestamp(snapshot["cash_start"], "market_snapshot.cash_start")
    cash_end = _factor_timestamp(snapshot["cash_end"], "market_snapshot.cash_end")
    cash_n_obs = _factor_integer(
        snapshot["cash_n_obs"],
        "market_snapshot.cash_n_obs",
        minimum=0,
        strictly_greater=True,
    )
    attribution_anchor = _factor_timestamp(
        snapshot["attribution_anchor"], "market_snapshot.attribution_anchor"
    )
    attribution_start = _factor_timestamp(
        snapshot["attribution_start"], "market_snapshot.attribution_start"
    )
    attribution_end = _factor_timestamp(
        snapshot["attribution_end"], "market_snapshot.attribution_end"
    )
    attribution_n_obs = _factor_integer(
        snapshot["attribution_n_obs"],
        "market_snapshot.attribution_n_obs",
        minimum=0,
        strictly_greater=True,
    )
    benchmark_observed_start = _factor_timestamp(
        snapshot["benchmark_observed_start"], "market_snapshot.benchmark_observed_start"
    )
    benchmark_observed_end = _factor_timestamp(
        snapshot["benchmark_observed_end"], "market_snapshot.benchmark_observed_end"
    )
    if not cash_anchor < cash_start <= cash_end:
        raise ValueError("market_snapshot cash lineage must satisfy anchor < start <= end")
    if not attribution_anchor < attribution_start <= attribution_end:
        raise ValueError(
            "market_snapshot attribution lineage must satisfy anchor < start <= end"
        )
    if not benchmark_observed_start <= attribution_anchor:
        raise ValueError("SPY observed coverage does not contain the attribution anchor")
    if benchmark_observed_end < attribution_end:
        raise ValueError("SPY observed coverage does not reach the attribution endpoint")

    source_streams = _factor_mapping(bundle["source_streams"], "source_streams")
    expected_streams = (*_FACTOR_PRIMARY_PORTFOLIOS, _FACTOR_DIFFERENTIAL_PORTFOLIO)
    _factor_exact_fields(source_streams, expected_streams, "source_streams")
    normalized_streams: dict[str, dict[str, object]] = {}
    for portfolio_id in _FACTOR_PRIMARY_PORTFOLIOS:
        stream = _factor_mapping(source_streams[portfolio_id], f"source_streams.{portfolio_id}")
        _factor_exact_fields(
            stream,
            (
                "artifact",
                "start",
                "end",
                "n_obs",
                "attribution_result_sha256",
                "crisis_result_sha256",
            ),
            f"source_streams.{portfolio_id}",
        )
        if stream["artifact"] != _FACTOR_STREAM_ARTIFACTS[portfolio_id]:
            raise ValueError(f"source_streams.{portfolio_id}.artifact is not the approved source")
        normalized_streams[portfolio_id] = {
            "artifact": stream["artifact"],
            "start": _factor_timestamp(stream["start"], f"source_streams.{portfolio_id}.start"),
            "end": _factor_timestamp(stream["end"], f"source_streams.{portfolio_id}.end"),
            "n_obs": _factor_integer(
                stream["n_obs"],
                f"source_streams.{portfolio_id}.n_obs",
                minimum=0,
                strictly_greater=True,
            ),
            "attribution_result_sha256": _factor_sha256(
                stream["attribution_result_sha256"],
                f"source_streams.{portfolio_id}.attribution_result_sha256",
            ),
            "crisis_result_sha256": _factor_sha256(
                stream["crisis_result_sha256"],
                f"source_streams.{portfolio_id}.crisis_result_sha256",
            ),
        }
    pit_stream = normalized_streams[_FACTOR_PRIMARY_PORTFOLIOS[0]]
    nonpit_stream = normalized_streams[_FACTOR_PRIMARY_PORTFOLIOS[1]]
    for field in ("start", "end", "n_obs"):
        if not _factor_values_equal(pit_stream[field], nonpit_stream[field]):
            raise ValueError(f"PIT and non-PIT source streams disagree on {field}")
    if (pit_stream["start"], pit_stream["end"], pit_stream["n_obs"]) != (
        cash_start,
        cash_end,
        cash_n_obs,
    ):
        raise ValueError("source-stream performance window does not equal BIL cash lineage")
    if attribution_end != cash_end or attribution_start < cash_start:
        raise ValueError(
            "attribution must be a full or shortened suffix ending on the performance endpoint"
        )
    if attribution_n_obs > cash_n_obs:
        raise ValueError("attribution_n_obs cannot exceed the performance observation count")

    differential_stream = _factor_mapping(
        source_streams[_FACTOR_DIFFERENTIAL_PORTFOLIO],
        f"source_streams.{_FACTOR_DIFFERENTIAL_PORTFOLIO}",
    )
    _factor_exact_fields(
        differential_stream,
        ("comparison", "reference", "start", "end", "n_obs"),
        f"source_streams.{_FACTOR_DIFFERENTIAL_PORTFOLIO}",
    )
    if differential_stream["comparison"] != _FACTOR_PRIMARY_PORTFOLIOS[1]:
        raise ValueError("differential source comparison must be the non-PIT portfolio")
    if differential_stream["reference"] != _FACTOR_PRIMARY_PORTFOLIOS[0]:
        raise ValueError("differential source reference must be the PIT portfolio")
    differential_window = (
        _factor_timestamp(
            differential_stream["start"],
            f"source_streams.{_FACTOR_DIFFERENTIAL_PORTFOLIO}.start",
        ),
        _factor_timestamp(
            differential_stream["end"],
            f"source_streams.{_FACTOR_DIFFERENTIAL_PORTFOLIO}.end",
        ),
        _factor_integer(
            differential_stream["n_obs"],
            f"source_streams.{_FACTOR_DIFFERENTIAL_PORTFOLIO}.n_obs",
            minimum=0,
            strictly_greater=True,
        ),
    )
    if differential_window != (pit_stream["start"], pit_stream["end"], pit_stream["n_obs"]):
        raise ValueError("differential source stream must use the common performance window")

    settings = _factor_mapping(bundle["ssr_settings"], "ssr_settings")
    _factor_exact_fields(
        settings,
        ("alpha", "n_boot", "periods_per_year", "seed", "sr_star", "window"),
        "ssr_settings",
    )
    alpha = _factor_finite_real(settings["alpha"], "ssr_settings.alpha")
    if not 0.0 < alpha < 1.0:
        raise ValueError("ssr_settings.alpha must lie strictly between zero and one")
    n_boot = _factor_integer(
        settings["n_boot"], "ssr_settings.n_boot", minimum=0, strictly_greater=True
    )
    periods_per_year = _factor_integer(
        settings["periods_per_year"],
        "ssr_settings.periods_per_year",
        minimum=0,
        strictly_greater=True,
    )
    if periods_per_year != 252:
        raise ValueError("Factor run-local records require 252 periods_per_year")
    seed = _factor_integer(settings["seed"], "ssr_settings.seed", minimum=0)
    sr_star = _factor_finite_real(settings["sr_star"], "ssr_settings.sr_star")
    window = _factor_integer(
        settings["window"], "ssr_settings.window", minimum=1, strictly_greater=True
    )
    expected_ssr = {
        "ssr_alpha": alpha,
        "ssr_n_boot": n_boot,
        "ssr_periods_per_year": periods_per_year,
        "ssr_seed": seed,
        "ssr_sr_star": sr_star,
        "ssr_window": window,
    }

    source_prefix = "scripts/extend_stream_2026.py"
    cash_source = (
        f"market_snapshot:{snapshot_id}/{snapshot['cash_file']}"
        f"#BIL@{cash_file_sha256}"
    )
    market_source = (
        f"market_snapshot:{snapshot_id}/{snapshot['benchmark_file']}"
        f"#SPY@{benchmark_file_sha256}"
    )
    crisis_results: dict[str, mf.CrisisMetrics] = {}
    row_kinds: set[object] = set()
    for portfolio_id in _FACTOR_PRIMARY_PORTFOLIOS:
        stream = normalized_streams[portfolio_id]
        performance_window = (stream["start"], stream["end"], stream["n_obs"])
        artifact_source = f"{source_prefix}:{stream['artifact']}"
        reader = by_key[(portfolio_id, mf.READER_SCHEMA)]
        legacy = by_key[(portfolio_id, mf.LEGACY_SCHEMA)]
        attribution = by_key[(portfolio_id, mf.ATTRIBUTION_SCHEMA)]
        crisis = by_key[(portfolio_id, mf.CRISIS_SCHEMA)]

        for name, row in (("reader", reader), ("legacy", legacy)):
            if (
                _factor_timestamp(row["start"], f"{portfolio_id}.{name}.start"),
                _factor_timestamp(row["end"], f"{portfolio_id}.{name}.end"),
                _factor_integer(
                    row["n_obs"],
                    f"{portfolio_id}.{name}.n_obs",
                    minimum=0,
                    strictly_greater=True,
                ),
            ) != performance_window:
                raise ValueError(f"{portfolio_id} {name} row diverges from its source stream")
            if row["return_basis"] != "adjusted_total_return_equity":
                raise ValueError(f"{portfolio_id} {name} row has the wrong return basis")
            if row["cash_benchmark_id"] != f"BIL@{snapshot_id}":
                raise ValueError(f"{portfolio_id} {name} row has the wrong cash benchmark")
            if row["currency_basis"] != "legacy_mixed_local_quotes":
                raise ValueError(f"{portfolio_id} {name} row has the wrong currency basis")
        if reader["periods_per_year"] != 252 or legacy["periods_per_year"] != 365:
            raise ValueError(f"{portfolio_id} reader/legacy annualization is invalid")
        if reader["source"] != f"{artifact_source}|{cash_source}|{market_source}":
            raise ValueError(f"{portfolio_id} reader source lineage is invalid")
        if legacy["source"] != artifact_source:
            raise ValueError(f"{portfolio_id} legacy source lineage is invalid")
        for field, expected in expected_ssr.items():
            if not _factor_values_equal(reader[field], expected):
                raise ValueError(f"{portfolio_id} reader {field} diverges from ssr_settings")

        typed_attribution = _factor_attribution_from_record(
            attribution, name=f"{portfolio_id}.attribution"
        )
        if (
            typed_attribution.start,
            typed_attribution.end,
            typed_attribution.n_obs,
        ) != (attribution_start, attribution_end, attribution_n_obs):
            raise ValueError(
                f"{portfolio_id} attribution row diverges from market-snapshot coverage"
            )
        if typed_attribution.periods_per_year != 252:
            raise ValueError(f"{portfolio_id} attribution must annualize on 252 periods")
        if attribution["return_basis"] != "adjusted_total_return_equity":
            raise ValueError(f"{portfolio_id} attribution has the wrong return basis")
        if attribution["cash_benchmark_id"] != f"BIL@{snapshot_id}":
            raise ValueError(f"{portfolio_id} attribution has the wrong cash benchmark")
        if attribution["currency_basis"] != "legacy_mixed_local_quotes":
            raise ValueError(f"{portfolio_id} attribution has the wrong currency basis")
        if attribution["source"] != f"{artifact_source}|{market_source}":
            raise ValueError(f"{portfolio_id} attribution source lineage is invalid")
        if (
            _factor_typed_result_sha256(typed_attribution)
            != stream["attribution_result_sha256"]
        ):
            raise ValueError(
                f"{portfolio_id} attribution diverges from shared-result lineage"
            )

        row_kind = reader["row_kind"]
        row_kinds.add(row_kind)
        expected_kind = "full" if (
            typed_attribution.start,
            typed_attribution.end,
            typed_attribution.n_obs,
        ) == performance_window else "performance_only"
        if row_kind != expected_kind:
            raise ValueError(
                f"{portfolio_id} reader row_kind {row_kind!r} contradicts attribution coverage"
            )
        if row_kind == "full":
            for field in dataclasses.fields(mf.MarketAttribution):
                key = f"raw_market_model_{field.name}"
                if not _factor_values_equal(reader[key], attribution[key]):
                    raise ValueError(
                        f"{portfolio_id} full reader attribution diverges from standalone {key}"
                    )
        elif not (
            typed_attribution.start > performance_window[0]
            and typed_attribution.end == performance_window[1]
            and typed_attribution.n_obs < performance_window[2]
        ):
            raise ValueError(
                f"{portfolio_id} performance_only reader lacks a separately disclosed "
                "shortened attribution suffix"
            )

        typed_crisis = _factor_crisis_from_record(crisis, name=f"{portfolio_id}.crisis")
        crisis_results[portfolio_id] = typed_crisis
        if typed_crisis.periods_per_year != 252:
            raise ValueError(f"{portfolio_id} crisis must annualize on 252 periods")
        if not cash_anchor <= typed_crisis.anchor:
            raise ValueError(f"{portfolio_id} crisis anchor precedes the portfolio source")
        if typed_crisis.actual_end > performance_window[1]:
            raise ValueError(f"{portfolio_id} crisis extends beyond the portfolio source")
        if crisis["return_basis"] != "adjusted_total_return_equity":
            raise ValueError(f"{portfolio_id} crisis has the wrong return basis")
        if crisis["cash_benchmark_id"] != f"BIL@{snapshot_id}":
            raise ValueError(f"{portfolio_id} crisis has the wrong cash benchmark")
        if crisis["currency_basis"] != "legacy_mixed_local_quotes":
            raise ValueError(f"{portfolio_id} crisis has the wrong currency basis")
        if crisis["source"] != artifact_source:
            raise ValueError(f"{portfolio_id} crisis source lineage is invalid")
        if _factor_typed_result_sha256(typed_crisis) != stream["crisis_result_sha256"]:
            raise ValueError(f"{portfolio_id} crisis diverges from shared-result lineage")

    if len(row_kinds) != 1:
        raise ValueError("PIT and non-PIT readers must disclose the same attribution coverage mode")
    pit_crisis = crisis_results[_FACTOR_PRIMARY_PORTFOLIOS[0]]
    nonpit_crisis = crisis_results[_FACTOR_PRIMARY_PORTFOLIOS[1]]
    for field in (
        "requested_start",
        "requested_end",
        "anchor",
        "first_return_date",
        "actual_end",
        "n_returns",
        "periods_per_year",
    ):
        if not _factor_values_equal(getattr(pit_crisis, field), getattr(nonpit_crisis, field)):
            raise ValueError(f"PIT and non-PIT crisis records disagree on {field}")

    differential = by_key[(_FACTOR_DIFFERENTIAL_PORTFOLIO, mf.DIFFERENTIAL_SCHEMA)]
    if (
        _factor_timestamp(differential["start"], "differential.start"),
        _factor_timestamp(differential["end"], "differential.end"),
        _factor_integer(
            differential["n_obs"],
            "differential.n_obs",
            minimum=0,
            strictly_greater=True,
        ),
    ) != differential_window:
        raise ValueError("differential row diverges from its source stream")
    if differential["return_basis"] != "direct_daily_return_spread_nonpit_minus_pit":
        raise ValueError("differential row has the wrong return basis")
    if differential["cash_benchmark_id"] != "not_applicable_direct_daily_spread":
        raise ValueError("differential row has the wrong cash-benchmark identity")
    if differential["currency_basis"] != "legacy_mixed_local_quotes":
        raise ValueError("differential row has the wrong currency basis")
    nonpit_source = f"{source_prefix}:{_FACTOR_STREAM_ARTIFACTS[_FACTOR_PRIMARY_PORTFOLIOS[1]]}"
    pit_source = f"{source_prefix}:{_FACTOR_STREAM_ARTIFACTS[_FACTOR_PRIMARY_PORTFOLIOS[0]]}"
    if differential["source"] != f"{nonpit_source}-{pit_source}":
        raise ValueError("differential row source lineage is invalid")
    for field, expected in expected_ssr.items():
        if not _factor_values_equal(differential[field], expected):
            raise ValueError(f"differential {field} diverges from ssr_settings")

    record_sha256s = bundle[_FACTOR_RECORD_SHA256S_FIELD]
    if not isinstance(record_sha256s, list) or len(record_sha256s) != len(records):
        raise ValueError(
            "Factor metric-record bundle must carry one record_sha256s entry per record"
        )
    for position, digest in enumerate(record_sha256s):
        _factor_sha256(digest, f"record_sha256s[{position}]")
    if record_sha256s != _factor_record_sha256s(records):
        raise ValueError("Factor metric-record record_sha256s do not match the record payloads")

    recorded_sha256 = _factor_sha256(
        bundle[_FACTOR_CONTENT_SHA256_FIELD],
        f"Factor metric-record bundle.{_FACTOR_CONTENT_SHA256_FIELD}",
    )
    expected_sha256 = _factor_content_sha256(bundle)
    if recorded_sha256 != expected_sha256:
        raise ValueError(
            "Factor metric-record content_sha256 does not match records and source lineage"
        )


def build_factor_metric_records(
    pit_value: pd.Series,
    nonpit_value: pd.Series,
    *,
    snapshot_dir: Path | str,
    window: int = 252,
    sr_star: float = 0.0,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    crisis_start: str | pd.Timestamp = "2022-01-01",
    crisis_end: str | pd.Timestamp = "2022-12-31",
) -> dict[str, object]:
    """Build immutable run-local finance records from strict shared contracts.

    Reader SSR is calculated once from portfolio returns minus completed-snapshot
    BIL total returns. Differential SSR receives the direct non-PIT-minus-PIT
    daily spread, with no second cash subtraction. SPY attribution and crisis
    records are projections of the shared typed results; this producer selects
    explicit windows and lineage but owns no duplicate financial formulas.
    """
    pit_value = _active_portfolio_value(pit_value)
    nonpit_value = _active_portfolio_value(nonpit_value)
    pit_metrics = metric_block(pit_value)
    nonpit_metrics = metric_block(nonpit_value)
    pit_returns = pit_metrics["returns"]
    nonpit_returns = nonpit_metrics["returns"]
    if not pit_returns.index.equals(nonpit_returns.index):
        raise ValueError("PIT and non-PIT performance returns must have identical indexes")
    pit_anchor = pit_value.index[0]
    nonpit_anchor = nonpit_value.index[0]
    if pit_anchor != nonpit_anchor:
        raise ValueError("PIT and non-PIT performance returns must have the same value anchor")

    cash_returns, snapshot = load_completed_snapshot_bil_returns(
        snapshot_dir, pit_returns.index, anchor=pit_anchor
    )
    market_returns, market_lineage = load_completed_snapshot_market_returns(
        snapshot_dir, pit_returns.index, value_index=pit_value.index
    )
    snapshot.update(market_lineage)
    attribution_index = market_returns.index
    pit_attribution = mf.raw_market_model_attribution(
        pit_returns.loc[attribution_index], market_returns
    )
    nonpit_attribution = mf.raw_market_model_attribution(
        nonpit_returns.loc[attribution_index], market_returns
    )
    pit_crisis = mf.crisis_metrics(pit_value, crisis_start, crisis_end)
    nonpit_crisis = mf.crisis_metrics(nonpit_value, crisis_start, crisis_end)
    if pit_crisis is None or nonpit_crisis is None:
        raise ValueError(
            "required crisis window has no preceding anchor or no portfolio observations"
        )

    cash_id = f"BIL@{snapshot['snapshot_id']}"
    settings = {
        "window": window,
        "sr_star": sr_star,
        "n_boot": n_boot,
        "seed": seed,
        "alpha": alpha,
    }
    pit_excess = mf.portfolio_excess_returns(pit_returns, cash_returns)
    nonpit_excess = mf.portfolio_excess_returns(nonpit_returns, cash_returns)
    spread = mf.differential_returns(nonpit_returns, pit_returns)
    pit_ssr = ssr_inference(pit_excess, **settings)
    nonpit_ssr = ssr_inference(nonpit_excess, **settings)
    differential_ssr = ssr_inference(spread, **settings)

    pit_meta = _metric_meta(
        "factor_pit_ext2026",
        "PIT recall-guarded factor (deployable, ext2026)",
        pit_returns,
        total_return_basis="adjusted_total_return_equity",
        cash_benchmark_id=cash_id,
    )
    nonpit_meta = _metric_meta(
        "factor_nonpit_diagnostic_ext2026",
        "Non-PIT recall-enabled factor (diagnostic, ext2026)",
        nonpit_returns,
        total_return_basis="adjusted_total_return_equity",
        cash_benchmark_id=cash_id,
    )
    differential_meta = _metric_meta(
        "factor_nonpit_minus_pit_ext2026",
        "Non-PIT minus PIT daily differential (ext2026)",
        spread,
        total_return_basis="direct_daily_return_spread_nonpit_minus_pit",
        cash_benchmark_id="not_applicable_direct_daily_spread",
    )
    source = "scripts/extend_stream_2026.py"
    pit_artifact = f"{source}:factor_equity_ext2026.parquet"
    nonpit_artifact = f"{source}:factor_nonpit_diagnostic_equity_ext2026.parquet"
    cash_source = (
        f"market_snapshot:{snapshot['snapshot_id']}/{snapshot['cash_file']}"
        f"#BIL@{snapshot['cash_file_sha256']}"
    )
    market_source = (
        f"market_snapshot:{snapshot['snapshot_id']}/{snapshot['benchmark_file']}"
        f"#SPY@{snapshot['benchmark_file_sha256']}"
    )
    pit_reader_source = f"{pit_artifact}|{cash_source}|{market_source}"
    nonpit_reader_source = f"{nonpit_artifact}|{cash_source}|{market_source}"

    records = [
        mf.build_reader_metric_row(
            pit_meta,
            pit_metrics,
            cash_returns,
            pit_ssr,
            source=pit_reader_source,
            attribution=pit_attribution,
        ),
        mf.build_legacy_metric_row(pit_meta, pit_metrics, source=pit_artifact),
        mf.build_reader_metric_row(
            nonpit_meta,
            nonpit_metrics,
            cash_returns,
            nonpit_ssr,
            source=nonpit_reader_source,
            attribution=nonpit_attribution,
        ),
        mf.build_legacy_metric_row(nonpit_meta, nonpit_metrics, source=nonpit_artifact),
        mf.build_differential_metric_row(
            differential_meta,
            nonpit_returns,
            pit_returns,
            differential_ssr,
            source=f"{nonpit_artifact}-{pit_artifact}",
        ),
    ]
    pit_attribution_meta = dataclasses.replace(
        pit_meta,
        window_label=_window_label(pit_attribution.start, pit_attribution.end),
    )
    nonpit_attribution_meta = dataclasses.replace(
        nonpit_meta,
        window_label=_window_label(nonpit_attribution.start, nonpit_attribution.end),
    )
    pit_crisis_meta = dataclasses.replace(
        pit_meta, window_label=_window_label(pit_crisis.anchor, pit_crisis.actual_end)
    )
    nonpit_crisis_meta = dataclasses.replace(
        nonpit_meta,
        window_label=_window_label(nonpit_crisis.anchor, nonpit_crisis.actual_end),
    )
    pit_attribution_record = mf.build_attribution_record(
        pit_attribution_meta,
        pit_attribution,
        source=f"{pit_artifact}|{market_source}",
    )
    nonpit_attribution_record = mf.build_attribution_record(
        nonpit_attribution_meta,
        nonpit_attribution,
        source=f"{nonpit_artifact}|{market_source}",
    )
    pit_crisis_record = mf.build_crisis_record(
        pit_crisis_meta, pit_crisis, source=pit_artifact
    )
    nonpit_crisis_record = mf.build_crisis_record(
        nonpit_crisis_meta, nonpit_crisis, source=nonpit_artifact
    )
    for portfolio_id, record, typed_result, prefix, shared_name in (
        (
            "factor_pit_ext2026",
            pit_attribution_record,
            pit_attribution,
            "raw_market_model_",
            "shared attribution",
        ),
        (
            "factor_nonpit_diagnostic_ext2026",
            nonpit_attribution_record,
            nonpit_attribution,
            "raw_market_model_",
            "shared attribution",
        ),
        (
            "factor_pit_ext2026",
            pit_crisis_record,
            pit_crisis,
            "",
            "shared crisis",
        ),
        (
            "factor_nonpit_diagnostic_ext2026",
            nonpit_crisis_record,
            nonpit_crisis,
            "",
            "shared crisis",
        ),
    ):
        _require_factor_typed_projection(
            record,
            typed_result,
            name=f"{portfolio_id} {shared_name}",
            prefix=prefix,
        )
    records.extend(
        [
            pit_attribution_record,
            nonpit_attribution_record,
            pit_crisis_record,
            nonpit_crisis_record,
        ]
    )
    records = [_validate_factor_record_window(row) for row in records]
    final_by_key = {
        (record["portfolio_id"], record["schema"]): record for record in records
    }
    for portfolio_id, typed_result, schema, prefix, shared_name in (
        (
            "factor_pit_ext2026",
            pit_attribution,
            mf.ATTRIBUTION_SCHEMA,
            "raw_market_model_",
            "shared attribution",
        ),
        (
            "factor_nonpit_diagnostic_ext2026",
            nonpit_attribution,
            mf.ATTRIBUTION_SCHEMA,
            "raw_market_model_",
            "shared attribution",
        ),
        (
            "factor_pit_ext2026",
            pit_crisis,
            mf.CRISIS_SCHEMA,
            "",
            "shared crisis",
        ),
        (
            "factor_nonpit_diagnostic_ext2026",
            nonpit_crisis,
            mf.CRISIS_SCHEMA,
            "",
            "shared crisis",
        ),
    ):
        _require_factor_typed_projection(
            final_by_key[(portfolio_id, schema)],
            typed_result,
            name=f"{portfolio_id} {shared_name}",
            prefix=prefix,
        )
    bundle = {
        "schema": FACTOR_METRIC_RECORDS_SCHEMA,
        "market_snapshot": snapshot,
        "ssr_settings": {
            "alpha": alpha,
            "n_boot": n_boot,
            "periods_per_year": 252,
            "seed": seed,
            "sr_star": sr_star,
            "window": window,
        },
        "source_streams": {
            "factor_pit_ext2026": {
                "artifact": "factor_equity_ext2026.parquet",
                "start": pit_returns.index[0],
                "end": pit_returns.index[-1],
                "n_obs": len(pit_returns),
                "attribution_result_sha256": _factor_typed_result_sha256(
                    pit_attribution
                ),
                "crisis_result_sha256": _factor_typed_result_sha256(pit_crisis),
            },
            "factor_nonpit_diagnostic_ext2026": {
                "artifact": "factor_nonpit_diagnostic_equity_ext2026.parquet",
                "start": nonpit_returns.index[0],
                "end": nonpit_returns.index[-1],
                "n_obs": len(nonpit_returns),
                "attribution_result_sha256": _factor_typed_result_sha256(
                    nonpit_attribution
                ),
                "crisis_result_sha256": _factor_typed_result_sha256(nonpit_crisis),
            },
            "factor_nonpit_minus_pit_ext2026": {
                "comparison": "factor_nonpit_diagnostic_ext2026",
                "reference": "factor_pit_ext2026",
                "start": spread.index[0],
                "end": spread.index[-1],
                "n_obs": len(spread),
            },
        },
        "records": records,
        _FACTOR_RECORD_SHA256S_FIELD: _factor_record_sha256s(records),
    }
    bundle[_FACTOR_CONTENT_SHA256_FIELD] = _factor_content_sha256(bundle)
    _validate_factor_metric_record_bundle(bundle)
    return bundle


def _json_record_value(value):
    if isinstance(value, Mapping):
        return {str(k): _json_record_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_record_value(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_factor_metric_records(
    bundle: Mapping[str, object],
    out_dir: Path | str,
    *,
    pit_value: pd.Series,
    nonpit_value: pd.Series,
    snapshot_dir: Path | str,
    crisis_start: str | pd.Timestamp = "2022-01-01",
    crisis_end: str | pd.Timestamp = "2022-12-31",
) -> Path:
    """Validate and persist records against independently trusted run-local inputs."""
    trusted = _trusted_factor_metric_sources(
        pit_value,
        nonpit_value,
        snapshot_dir=snapshot_dir,
        crisis_start=crisis_start,
        crisis_end=crisis_end,
    )
    _validate_factor_metric_record_bundle(bundle)
    _require_trusted_factor_metric_sources(bundle, trusted)
    serialized = (
        json.dumps(_json_record_value(bundle), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    # Validate the exact canonical JSON form, including timestamps and nulls,
    # before even creating the destination directory. The persisted run-local
    # source must therefore self-validate after a normal json.loads round-trip.
    serialized_bundle = json.loads(serialized)
    _validate_factor_metric_record_bundle(serialized_bundle)
    _require_trusted_factor_metric_sources(serialized_bundle, trusted)
    out = Path(out_dir) / FACTOR_METRIC_RECORDS_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(serialized)
    return out


# --------------------------------------------------------------------------- #
# Immutable Factor run manifest and bundle (task 6.9)                           #
# --------------------------------------------------------------------------- #

FACTOR_RUN_MANIFEST_SCHEMA = "factor_run.v1"
FACTOR_RUN_PROMPT_RENDERER_ID = (
    "macro_framework.factor_scoring.render_regime_loadings_prompt"
)

#: The complete artifact catalog of one immutable Factor run bundle: dated
#: evidence, full-stream scores and loadings, per-variant targets, equity and
#: decision logs, the PIT/non-PIT contrast pair, the PASSING replay audit, and
#: the run-local metric/report records (task 6.9; R7.1-R7.5). The manifest
#: inventories exactly these roles — nothing else may live in a run bundle.
FACTOR_RUN_ARTIFACTS: Mapping[str, Mapping[str, str]] = {
    "evidence": {
        "file": EVIDENCE_TABLE_NAME,
        "kind": "parquet",
        "lineage": "task 6.2 dated (variant, rebalance_date) replay evidence",
    },
    "loadings_pit": {
        "file": "factor_loadings_ext2026.parquet",
        "kind": "parquet",
        "lineage": "full-stream PIT loadings (v1 replay + live ext2026)",
    },
    "loadings_nonpit": {
        "file": "factor_nonpit_diagnostic_loadings_ext2026.parquet",
        "kind": "parquet",
        "lineage": "full-stream non-PIT diagnostic loadings (v1 replay + live ext2026)",
    },
    "scores_pit": {
        "file": "factor_scores_ext2026.parquet",
        "kind": "parquet",
        "lineage": "full-stream PIT recall-guard scores (v1 replay + live ext2026)",
    },
    "scores_nonpit": {
        "file": "factor_nonpit_diagnostic_scores_ext2026.parquet",
        "kind": "parquet",
        "lineage": "full-stream non-PIT diagnostic scores (v1 replay + live ext2026)",
    },
    "targets_pit": {
        "file": "factor_targets_ext2026.parquet",
        "kind": "parquet",
        "lineage": "walk-forward monthly target weights (PIT deployable line)",
    },
    "targets_nonpit": {
        "file": "factor_nonpit_diagnostic_targets_ext2026.parquet",
        "kind": "parquet",
        "lineage": "walk-forward monthly target weights (non-PIT diagnostic line)",
    },
    "equity_pit": {
        "file": "factor_equity_ext2026.parquet",
        "kind": "parquet",
        "lineage": "rebalance-simulation equity curve (PIT deployable line)",
    },
    "equity_nonpit": {
        "file": "factor_nonpit_diagnostic_equity_ext2026.parquet",
        "kind": "parquet",
        "lineage": "rebalance-simulation equity curve (non-PIT diagnostic line)",
    },
    "decision_log_pit": {
        "file": "factor_decision_log_ext2026.json",
        "kind": "json",
        "lineage": "per-rebalance decision log from the dated-evidence replay (PIT)",
    },
    "decision_log_nonpit": {
        "file": "factor_nonpit_diagnostic_decision_log_ext2026.json",
        "kind": "json",
        "lineage": "per-rebalance decision log from the dated-evidence replay (non-PIT)",
    },
    "contrast": {
        "file": "factor_contrast_ext2026.parquet",
        "kind": "parquet",
        "lineage": "per-date PIT vs non-PIT p_memorized contrast",
    },
    "contrast_split": {
        "file": "factor_contrast_split_ext2026.json",
        "kind": "json",
        "lineage": "in-training vs post-cutoff contrast split table",
    },
    "replay_audit": {
        "file": REPLAY_AUDIT_NAME,
        "kind": "json",
        "lineage": "task 6.4 passing source-to-consumption replay audit",
    },
    "metric_records": {
        "file": FACTOR_METRIC_RECORDS_NAME,
        "kind": "json",
        "lineage": "task 6.6/6.7 run-local metric and report records",
    },
}

_FACTOR_RUN_DATED_FRAMES = (
    "loadings_pit",
    "loadings_nonpit",
    "scores_pit",
    "scores_nonpit",
    "targets_pit",
    "targets_nonpit",
    "equity_pit",
    "equity_nonpit",
    "contrast",
)
_FACTOR_RUN_FILE_ENTRY_FIELDS = (
    "file",
    "sha256",
    "size",
    "rows",
    "start",
    "end",
    "schema_id",
    "lineage",
)
_FACTOR_RUN_MANIFEST_FIELDS = (
    "schema",
    "run_id",
    "build_time",
    "config",
    "source_commit",
    "prompt_renderer",
    "model",
    "input_manifests",
    "expected_evidence",
    "replay_audit",
    "files",
    "completed",
)


def prompt_renderer_identity() -> dict[str, str]:
    """The recorded identity of the PIT prompt renderer this run replayed with."""
    import inspect

    return {
        "id": FACTOR_RUN_PROMPT_RENDERER_ID,
        "source_sha256": sha256_text(
            inspect.getsource(fs.render_regime_loadings_prompt)
        ),
    }


def _git_source_commit() -> str:
    """The producing source commit recorded in the run manifest (env-overridable)."""
    commit = (os.environ.get("FACTOR_RUN_SOURCE_COMMIT") or "").strip()
    if commit:
        return commit
    import subprocess

    return subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _normalize_expected_evidence(expected: Mapping[str, object]) -> dict[str, object]:
    """Canonical expected-evidence declaration; a lying count always raises."""
    expected = _factor_mapping(expected, "expected_evidence")
    raw_variants = expected.get("variants")
    if not isinstance(raw_variants, (list, tuple)) or not raw_variants:
        raise ValueError("expected_evidence.variants must be a non-empty list")
    variants = sorted(str(v) for v in raw_variants)
    if len(set(variants)) != len(variants):
        raise ValueError(f"expected_evidence.variants contains duplicates: {variants}")
    for variant in variants:
        if variant not in _VARIANTS:
            raise ValueError(
                f"expected_evidence variant {variant!r} is not one of {_VARIANTS}"
            )
    raw_dates = expected.get("dates")
    if not isinstance(raw_dates, (list, tuple)) or not raw_dates:
        raise ValueError("expected_evidence.dates must be a non-empty list")
    parsed: list[date] = []
    for value in raw_dates:
        try:
            parsed.append(date.fromisoformat(str(value)))
        except ValueError as exc:
            raise ValueError(
                f"expected_evidence date {value!r} is not a valid ISO date"
            ) from exc
    if len(set(parsed)) != len(parsed):
        raise ValueError("expected_evidence.dates contains duplicates")
    parsed.sort()
    normalized = {
        "variants": variants,
        "dates": [d.isoformat() for d in parsed],
        "n_dates": len(parsed),
        "n_keys": len(variants) * len(parsed),
    }
    for key in ("n_dates", "n_keys"):
        if key in expected and expected[key] != normalized[key]:
            raise ValueError(
                f"expected_evidence {key} does not equal its declared variants and dates"
            )
    return normalized


def _factor_run_artifact_payload(path: Path, kind: str):
    if kind == "parquet":
        return pd.read_parquet(path)
    return json.loads(path.read_text())


def _factor_run_artifact_profile(
    role: str, payload: object, *, name: str
) -> dict[str, object]:
    """Recompute rows/start/end from artifact CONTENT (never trusted metadata)."""
    if role == "evidence":
        if (
            not isinstance(payload, pd.DataFrame)
            or payload.empty
            or "rebalance_date" not in payload.columns
        ):
            raise ValueError(f"{name} must be a non-empty dated evidence table")
        dates = list(payload["rebalance_date"])
        return {
            "rows": int(len(payload)),
            "start": min(dates).isoformat(),
            "end": max(dates).isoformat(),
        }
    if role in _FACTOR_RUN_DATED_FRAMES:
        if (
            not isinstance(payload, pd.DataFrame)
            or payload.empty
            or not isinstance(payload.index, pd.DatetimeIndex)
        ):
            raise ValueError(f"{name} must be a non-empty date-indexed table")
        return {
            "rows": int(len(payload)),
            "start": payload.index.min().date().isoformat(),
            "end": payload.index.max().date().isoformat(),
        }
    if role in ("decision_log_pit", "decision_log_nonpit"):
        parse_ok = _factor_mapping(
            _factor_mapping(payload, name).get("parse_ok"), f"{name}.parse_ok"
        )
        stamps = sorted(pd.Timestamp(key) for key in parse_ok)
        return {
            "rows": int(len(parse_ok)),
            "start": stamps[0].date().isoformat() if stamps else None,
            "end": stamps[-1].date().isoformat() if stamps else None,
        }
    if role == "contrast_split":
        payload = _factor_mapping(payload, name)
        segments = ("in_training", "post_cutoff", "full_stream")
        missing = [segment for segment in segments if segment not in payload]
        if missing:
            raise ValueError(f"{name} is missing split segment(s) {missing}")
        return {"rows": len(segments), "start": None, "end": None}
    if role == "replay_audit":
        payload = _factor_mapping(payload, name)
        counts = _factor_mapping(payload.get("counts"), f"{name}.counts")
        window = _factor_mapping(payload.get("window"), f"{name}.window")
        return {
            "rows": _factor_integer(
                counts.get("expected_keys"),
                f"{name}.counts.expected_keys",
                minimum=0,
                strictly_greater=True,
            ),
            "start": str(window.get("first_rebalance")),
            "end": str(window.get("last_rebalance")),
        }
    if role == "metric_records":
        records = _factor_mapping(payload, name).get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"{name} must carry a non-empty records list")
        starts = [_factor_timestamp(row["start"], f"{name}.start") for row in records]
        ends = [_factor_timestamp(row["end"], f"{name}.end") for row in records]
        return {
            "rows": len(records),
            "start": min(starts).date().isoformat(),
            "end": max(ends).date().isoformat(),
        }
    raise ValueError(f"unknown factor run artifact role {role!r}")


def _factor_evidence_records_from_frame(
    frame: pd.DataFrame,
) -> list[DatedFactorEvidence]:
    """Rebuild typed evidence records from the persisted flat scalar table."""
    field_names = [f.name for f in dataclasses.fields(DatedFactorEvidence)]
    missing = sorted(set(field_names) - set(frame.columns))
    if missing:
        raise ValueError(f"evidence table is missing column(s) {missing}")
    records: list[DatedFactorEvidence] = []
    for row in frame[field_names].to_dict("records"):
        clean: dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and math.isnan(value):
                value = None
            clean[key] = value
        records.append(DatedFactorEvidence(**clean))
    return records


def validate_factor_run_bundle(
    run_dir: Path | str, *, require_completed: bool = True
) -> dict[str, object]:
    """Validate one Factor run bundle from its manifest ALONE (task 6.9).

    Every check reads only the run directory: manifest structure, completion
    marker (which must carry the manifest's own sha256 — a rewritten manifest
    with a stale marker is rejected), byte-exact file inventory, recomputed
    row/window profiles, record-level dated-evidence revalidation against the
    manifest's expected keys, replay-audit binding, and the self-validating
    metric-record bundle with its market-snapshot lineage.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{run_dir}: manifest.json is missing")
    manifest = _factor_mapping(
        json.loads(manifest_path.read_text()), "factor run manifest"
    )
    if manifest.get("schema") != FACTOR_RUN_MANIFEST_SCHEMA:
        raise ValueError(
            f"{run_dir}: unknown factor run manifest schema {manifest.get('schema')!r}"
        )
    _factor_exact_fields(manifest, _FACTOR_RUN_MANIFEST_FIELDS, "factor run manifest")
    if manifest["completed"] is not True:
        raise ValueError(f"{run_dir}: factor run manifest must declare completed=true")
    run_id = manifest["run_id"]
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f"{run_dir}: run_id must be a non-empty string")
    if run_dir.name != run_id:
        raise ValueError(
            f"{run_dir}: bundle directory name does not equal run_id {run_id!r}"
        )
    try:
        built = pd.Timestamp(str(manifest["build_time"]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{run_dir}: build_time is not a valid timestamp") from exc
    if built.tz is None:
        raise ValueError(f"{run_dir}: build_time must be timezone-aware")
    for field in ("config", "model"):
        if not _factor_mapping(manifest[field], f"factor run manifest.{field}"):
            raise ValueError(f"{run_dir}: {field} must be a non-empty mapping")
    source_commit = manifest["source_commit"]
    if not isinstance(source_commit, str) or re.fullmatch(
        r"[0-9a-f]{40}", source_commit
    ) is None:
        raise ValueError(f"{run_dir}: source_commit must be a 40-hex Git commit id")
    renderer = _factor_mapping(manifest["prompt_renderer"], "prompt_renderer")
    _factor_exact_fields(renderer, ("id", "source_sha256"), "prompt_renderer")
    if renderer["id"] != FACTOR_RUN_PROMPT_RENDERER_ID:
        raise ValueError(
            f"{run_dir}: prompt_renderer.id must be {FACTOR_RUN_PROMPT_RENDERER_ID!r}"
        )
    _factor_sha256(renderer["source_sha256"], "prompt_renderer.source_sha256")

    input_manifests = _factor_mapping(manifest["input_manifests"], "input_manifests")
    snapshot_input = _factor_mapping(
        input_manifests.get("market_snapshot"), "input_manifests.market_snapshot"
    )
    _factor_exact_fields(
        snapshot_input,
        ("snapshot_id", "manifest_sha256"),
        "input_manifests.market_snapshot",
    )
    if (
        not isinstance(snapshot_input["snapshot_id"], str)
        or not snapshot_input["snapshot_id"].strip()
    ):
        raise ValueError(
            f"{run_dir}: input_manifests.market_snapshot.snapshot_id must be non-empty"
        )
    _factor_sha256(
        snapshot_input["manifest_sha256"],
        "input_manifests.market_snapshot.manifest_sha256",
    )
    for name, entry in input_manifests.items():
        if name == "market_snapshot":
            continue
        entry = _factor_mapping(entry, f"input_manifests.{name}")
        _factor_sha256(
            entry.get("sha256", entry.get("manifest_sha256")),
            f"input_manifests.{name}.sha256",
        )

    expected = _factor_mapping(manifest["expected_evidence"], "expected_evidence")
    _factor_exact_fields(
        expected, ("variants", "dates", "n_dates", "n_keys"), "expected_evidence"
    )
    normalized = _normalize_expected_evidence(expected)
    if dict(expected) != normalized:
        raise ValueError(
            f"{run_dir}: expected_evidence is not in canonical sorted form"
        )
    variants = list(normalized["variants"])
    iso_dates = list(normalized["dates"])
    n_keys = int(normalized["n_keys"])

    audit_summary = _factor_mapping(manifest["replay_audit"], "replay_audit")
    _factor_exact_fields(audit_summary, ("result", "counts"), "replay_audit")
    if audit_summary["result"] != "pass":
        raise ValueError(f"{run_dir}: recorded replay audit result must be 'pass'")

    if require_completed:
        marker_path = run_dir / "COMPLETED"
        if not marker_path.is_file():
            raise ValueError(
                f"{run_dir}: COMPLETED marker is absent; factor run is incomplete"
            )
        marker_match = re.search(
            r"manifest_sha256=([0-9a-f]{64})", marker_path.read_text()
        )
        if marker_match is None or marker_match.group(1) != sha256_file(manifest_path):
            raise ValueError(
                f"{run_dir}: COMPLETED marker does not match manifest bytes "
                "(stale or tampered completion)"
            )

    files = _factor_mapping(manifest["files"], "factor run manifest files")
    _factor_exact_fields(
        files, tuple(FACTOR_RUN_ARTIFACTS), "factor run manifest files"
    )
    payloads: dict[str, object] = {}
    for role, spec in FACTOR_RUN_ARTIFACTS.items():
        entry = _factor_mapping(files[role], f"files.{role}")
        _factor_exact_fields(entry, _FACTOR_RUN_FILE_ENTRY_FIELDS, f"files.{role}")
        if entry["file"] != spec["file"]:
            raise ValueError(
                f"{run_dir}: files.{role} names {entry['file']!r}, not the approved "
                f"artifact {spec['file']!r}"
            )
        if entry["schema_id"] != f"{FACTOR_RUN_MANIFEST_SCHEMA}/{role}":
            raise ValueError(f"{run_dir}: files.{role} has an invalid schema_id")
        if entry["lineage"] != spec["lineage"]:
            raise ValueError(f"{run_dir}: files.{role} lineage diverges from the catalog")
        path = run_dir / spec["file"]
        if not path.is_file():
            raise ValueError(f"{run_dir}: {spec['file']} is missing from disk")
        actual_sha = sha256_file(path)
        if actual_sha != entry["sha256"]:
            raise ValueError(
                f"{run_dir}: {spec['file']} bytes were mutated after inventory "
                f"(sha256 {actual_sha[:12]}... != recorded {str(entry['sha256'])[:12]}...)"
            )
        if int(path.stat().st_size) != entry["size"]:
            raise ValueError(f"{run_dir}: {spec['file']} size changed after inventory")
        payload = _factor_run_artifact_payload(path, spec["kind"])
        profile = _factor_run_artifact_profile(role, payload, name=spec["file"])
        if profile["rows"] != entry["rows"]:
            raise ValueError(
                f"{run_dir}: {spec['file']} rows {profile['rows']} do not match "
                f"inventoried rows {entry['rows']}"
            )
        if (profile["start"], profile["end"]) != (entry["start"], entry["end"]):
            raise ValueError(
                f"{run_dir}: {spec['file']} window {profile['start']}..{profile['end']} "
                "does not match the inventoried window"
            )
        payloads[role] = payload

    allowed = {
        "manifest.json",
        "COMPLETED",
        *(spec["file"] for spec in FACTOR_RUN_ARTIFACTS.values()),
    }
    stray = sorted(p.name for p in run_dir.iterdir() if p.name not in allowed)
    if stray:
        raise ValueError(f"{run_dir}: unmanifested file(s) present: {stray}")

    # dated evidence: full RECORD-level revalidation against the manifest's own
    # expected key set — byte inventory alone would accept a coherently
    # re-signed content forgery (R6.3, R6.4, R8.7).
    expected_dates = [date.fromisoformat(d) for d in iso_dates]
    expected_keys = [(variant, d) for variant in variants for d in expected_dates]
    validate_evidence_records(
        _factor_evidence_records_from_frame(payloads["evidence"]), expected_keys
    )

    expected_set = set(expected_dates)
    for role in ("loadings_pit", "loadings_nonpit", "scores_pit", "scores_nonpit",
                 "contrast"):
        got = {stamp.date() for stamp in payloads[role].index}
        if got != expected_set:
            raise ValueError(
                f"{run_dir}: {role} rebalance dates do not equal the expected "
                "evidence dates"
            )
    for role in ("targets_pit", "targets_nonpit"):
        got = {stamp.date() for stamp in payloads[role].index}
        if not expected_set <= got:
            raise ValueError(
                f"{run_dir}: {role} does not cover every expected rebalance date"
            )
    for role in ("equity_pit", "equity_nonpit"):
        idx = payloads[role].index
        if idx.min().date() > expected_dates[0] or idx.max().date() < expected_dates[-1]:
            raise ValueError(
                f"{run_dir}: {role} window does not contain the expected rebalance dates"
            )
    for role in ("decision_log_pit", "decision_log_nonpit"):
        keys = {pd.Timestamp(k).date() for k in payloads[role]["parse_ok"]}
        if not keys <= expected_set:
            raise ValueError(
                f"{run_dir}: {role} contains decision dates outside the expected "
                "evidence dates"
            )

    audit = _factor_mapping(payloads["replay_audit"], "replay audit file")
    if (
        audit.get("audit") != "source_to_consumption_replay"
        or audit.get("result") != "pass"
    ):
        raise ValueError(
            f"{run_dir}: replay audit file does not record a passing "
            "source-to-consumption audit"
        )
    counts = _factor_mapping(audit.get("counts"), "replay audit counts")
    if dict(audit_summary["counts"]) != dict(counts):
        raise ValueError(
            f"{run_dir}: recorded replay audit counts diverge from the audit file"
        )
    for field in ("expected_keys", "consumed_keys", "source_records"):
        value = _factor_integer(
            counts.get(field), f"replay audit counts.{field}", minimum=0,
            strictly_greater=True,
        )
        if value != n_keys:
            raise ValueError(
                f"{run_dir}: replay audit {field} does not equal expected_evidence n_keys"
            )
    if sorted(audit.get("variants", [])) != variants:
        raise ValueError(
            f"{run_dir}: replay audit variants diverge from expected_evidence"
        )
    window = _factor_mapping(audit.get("window"), "replay audit window")
    if (
        window.get("first_rebalance"),
        window.get("last_rebalance"),
    ) != (iso_dates[0], iso_dates[-1]):
        raise ValueError(
            f"{run_dir}: replay audit window diverges from expected_evidence dates"
        )

    metric = _factor_mapping(payloads["metric_records"], "metric records")
    _validate_factor_metric_record_bundle(metric)
    metric_snapshot = _factor_mapping(
        metric["market_snapshot"], "metric records market_snapshot"
    )
    if (
        metric_snapshot["snapshot_id"] != snapshot_input["snapshot_id"]
        or metric_snapshot["manifest_sha256"] != snapshot_input["manifest_sha256"]
    ):
        raise ValueError(
            f"{run_dir}: metric-record market snapshot lineage diverges from "
            "input_manifests"
        )
    for portfolio_id, role in (
        ("factor_pit_ext2026", "equity_pit"),
        ("factor_nonpit_diagnostic_ext2026", "equity_nonpit"),
    ):
        stream_end = pd.Timestamp(
            metric["source_streams"][portfolio_id]["end"]
        ).date()
        if stream_end != payloads[role].index.max().date():
            raise ValueError(
                f"{run_dir}: {role} does not end on the {portfolio_id} metric-record "
                "performance endpoint"
            )

    return {
        "run_id": run_id,
        "schema": manifest["schema"],
        "files": {
            role: {"rows": files[role]["rows"], "sha256": files[role]["sha256"]}
            for role in FACTOR_RUN_ARTIFACTS
        },
        "completed": (run_dir / "COMPLETED").is_file(),
    }


def load_completed_factor_run(run_dir: Path | str) -> dict[str, object]:
    """Downstream entry point: full validation plus completion, then the manifest."""
    validate_factor_run_bundle(run_dir, require_completed=True)
    return json.loads((Path(run_dir) / "manifest.json").read_text())


def build_factor_run_bundle(
    *,
    run_id: str,
    output_root: Path | str,
    artifacts: Mapping[str, Path | str],
    config: Mapping[str, object],
    source_commit: str,
    model: Mapping[str, object],
    input_manifests: Mapping[str, Mapping[str, object]],
    expected_evidence: Mapping[str, object],
    prompt_renderer: Mapping[str, str] | None = None,
    build_time: str | None = None,
) -> Path:
    """Assemble ONE immutable Factor run bundle; COMPLETED is written LAST.

    Mirrors the market-snapshot staging conventions (tasks 5.3/5.4): the
    destination must be a new empty staging directory, a COMPLETED destination
    is immutable and never overwritten, every inventory and audit validation
    must pass before the completion marker exists, and the marker carries the
    manifest sha256. A failed build leaves the staging directory dirty WITHOUT
    ``COMPLETED`` for diagnosis; recovery is delete-and-rebuild.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    expected = _normalize_expected_evidence(expected_evidence)
    _factor_exact_fields(
        _factor_mapping(artifacts, "artifacts"), tuple(FACTOR_RUN_ARTIFACTS), "artifacts"
    )
    sources: dict[str, Path] = {}
    for role in FACTOR_RUN_ARTIFACTS:
        src = Path(artifacts[role])
        if not src.is_file():
            raise ValueError(f"artifacts[{role!r}] source file is absent: {src}")
        sources[role] = src

    run_dir = Path(output_root) / run_id
    if run_dir.exists():
        if (run_dir / "COMPLETED").exists():
            raise ValueError(
                f"factor run {run_id!r} is COMPLETED and immutable; runs are append-only"
            )
        if any(run_dir.iterdir()):
            raise ValueError(
                f"refusing to write into non-empty staging directory {run_dir}"
            )
    if build_time is None:
        build_time = pd.Timestamp.now("UTC").isoformat()
    else:
        try:
            parsed = pd.Timestamp(build_time)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"build_time must be an ISO-8601 timestamp, got {build_time!r}"
            ) from exc
        if parsed.tz is None:
            raise ValueError(f"build_time must be timezone-aware, got {build_time!r}")
        build_time = parsed.tz_convert("UTC").isoformat()
    run_dir.mkdir(parents=True, exist_ok=True)

    for role, spec in FACTOR_RUN_ARTIFACTS.items():
        shutil.copyfile(sources[role], run_dir / spec["file"])

    files: dict[str, dict[str, object]] = {}
    for role, spec in FACTOR_RUN_ARTIFACTS.items():
        dest = run_dir / spec["file"]
        payload = _factor_run_artifact_payload(dest, spec["kind"])
        profile = _factor_run_artifact_profile(role, payload, name=spec["file"])
        files[role] = {
            "file": spec["file"],
            "sha256": sha256_file(dest),
            "size": int(dest.stat().st_size),
            "schema_id": f"{FACTOR_RUN_MANIFEST_SCHEMA}/{role}",
            "lineage": spec["lineage"],
            **profile,
        }
    audit_payload = _factor_run_artifact_payload(run_dir / REPLAY_AUDIT_NAME, "json")

    manifest = {
        "schema": FACTOR_RUN_MANIFEST_SCHEMA,
        "run_id": run_id,
        "build_time": build_time,
        "config": dict(config),
        "source_commit": source_commit,
        "prompt_renderer": (
            dict(prompt_renderer)
            if prompt_renderer is not None
            else prompt_renderer_identity()
        ),
        "model": dict(model),
        "input_manifests": {
            name: dict(entry) for name, entry in dict(input_manifests).items()
        },
        "expected_evidence": expected,
        "replay_audit": {
            "result": _factor_mapping(audit_payload, "replay audit file").get("result"),
            "counts": _factor_mapping(audit_payload, "replay audit file").get("counts"),
        },
        "files": files,
        "completed": True,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(_json_record_value(manifest), indent=2, sort_keys=True) + "\n"
    )

    # every inventory and audit validation must pass BEFORE the marker exists
    validate_factor_run_bundle(run_dir, require_completed=False)
    manifest_sha = sha256_file(run_dir / "manifest.json")
    (run_dir / "COMPLETED").write_text(
        f"{build_time}\nmanifest_sha256={manifest_sha}\n"
    )
    return run_dir


# --------------------------------------------------------------------------- #
# CSV mirrors (the export_csv_mirrors _write pattern, ext2026 additions only)   #
# --------------------------------------------------------------------------- #


def _write_csv(df: pd.DataFrame, name: str, index: bool = True) -> None:
    """US + _de (semicolon/comma) locale variants — scripts/export_csv_mirrors._write."""
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT / f"{name}.csv", float_format="%.8f", index=index)
    df.to_csv(CSV_OUT / f"{name}_de.csv", sep=";", decimal=",", float_format="%.8f", index=index)


def _flatten_decision_log(payload: dict) -> pd.DataFrame:
    per_date = {k: payload[k] for k in ("p_memorized", "parse_ok", "steered", "conviction")}
    df = pd.DataFrame(per_date)
    df.index.name = "date"
    return df.sort_index()


# --------------------------------------------------------------------------- #
# Main pipeline                                                                #
# --------------------------------------------------------------------------- #


def main() -> None:  # noqa: PLR0915 -- one linear, printed, stage-by-stage run
    from dotenv import load_dotenv

    from recall_guard import NvidiaLM

    load_dotenv(REPO / ".env")
    nvidia_key = (os.environ.get("NVIDIA_API_KEY") or "").strip()
    if not nvidia_key:
        raise RuntimeError("NVIDIA_API_KEY not set in .env — required for the live 2025+ NIM calls")

    print("=== Task 8.1: extend the stream to 2026 (post-cutoff natural experiment) ===")

    # --- S0: panel + prices + rebalance stream --------------------------------- #
    panel, panel_source = build_panel()
    print(f"macro panel: {panel.shape} | {panel.index.min().date()} -> {panel.index.max().date()}")
    print(f"panel source: {panel_source}")

    spec = pd.read_parquet(DATA / "portfolio_ssr_top_per_category.parquet")
    symbols = spec["symbol"].tolist()
    asset_map = mf.AssetMap.default()
    factor_snapshot = [
        {"id": pseudo, "category": cat} for pseudo, cat in sorted(asset_map.categories.items())
    ]

    prices = fetch_prices(symbols)
    all_returns = prices[symbols].pct_change()
    rebalance_dates = mf.monthly_rebalance_dates(prices[symbols], start=SIM_START, end=SIM_END_EXT)
    print(f"prices: {prices.shape} | {prices.index.min().date()} -> {prices.index.max().date()}")
    print(f"{len(rebalance_dates)} monthly rebalances  "
          f"{rebalance_dates[0].date()} -> {rebalance_dates[-1].date()}")

    # --- S1: v1 artifacts (the replayed 2019-2024 segment) --------------------- #
    loadings_v1 = pd.read_parquet(DATA / "factor_loadings_v1.parquet")
    scores_v1 = pd.read_parquet(DATA / "factor_scores_v1.parquet")
    np_loadings_v1 = pd.read_parquet(DATA / "factor_nonpit_diagnostic_loadings_v1.parquet")
    np_scores_v1 = pd.read_parquet(DATA / "factor_nonpit_diagnostic_scores_v1.parquet")
    targets_v1 = pd.read_parquet(DATA / "factor_targets_v1.parquet")
    equity_v1 = pd.read_parquet(DATA / "factor_equity_v1.parquet")["value"]
    naive_v1 = pd.read_parquet(DATA / f"naive_directional_eval_{SLUG}.parquet")

    v1_dates = pd.DatetimeIndex(loadings_v1.index)
    missing = v1_dates.difference(rebalance_dates)
    assert missing.empty, f"v1 rebalance dates missing from the extended stream: {list(missing)}"

    def _panel_row_asof(rb: pd.Timestamp):
        avail = panel.dropna(subset=PANEL_Z_COLS)
        asof = avail[avail.index < rb]
        if asof.empty:
            return None, None
        row = asof.iloc[-1]
        return row[PANEL_Z_COLS].to_dict(), {c: float(row[c]) for c in PANEL_RAW_COLS if c in row}

    # (rb, macro_state, raw_levels, pit_prompt, nonpit_prompt) for the FULL stream.
    factor_meta = []
    for rb in rebalance_dates:
        macro_state, raw_levels = _panel_row_asof(rb)
        if macro_state is None:
            continue
        pit = fs.render_regime_loadings_prompt(macro_state, factor_snapshot)
        nonpit = fs.render_regime_loadings_prompt(
            macro_state, factor_snapshot, identifying=True,
            as_of=rb.date().isoformat(), raw_levels=raw_levels)
        assert nonpit.startswith(pit), "R7.6 violated: non-PIT must be PIT + additions only"
        factor_meta.append((rb, macro_state, raw_levels, pit, nonpit))

    # task 6.3: duplicate PIT prompt text across dates is VALID — replay identity
    # is (variant, rebalance_date), so there is no later-date-wins collision.
    meta_dates = [m[0] for m in factor_meta]

    new_meta = [m for m in factor_meta if m[0] not in v1_dates]
    print(f"stream: {len(factor_meta)} prompts total | replayed v1: "
          f"{len(factor_meta) - len(new_meta)} | live new (2025+): {len(new_meta)}")

    # --- S2: calibrator + live NIM work for the NEW dates only ----------------- #
    def lm_factory(key: str, model: str) -> NvidiaLM:
        return NvidiaLM(api_key=key, model=model, timeout_s=TIMEOUT_S)

    lm = NvidiaLM(api_key=nvidia_key, model=NIM_MODEL, timeout_s=TIMEOUT_S)
    scorer = fs.FactorScorer.load(CAL_DIR, api_key=nvidia_key, lm_factory=lm_factory)
    print(f"FactorScorer loaded from {CAL_DIR.name}: holdout_auc={scorer.holdout_auc:.4f} "
          f"is_weak={scorer.is_weak}")
    assert scorer.is_weak is False, "calibrator weak -> guard would pass through (R4.3)"

    new_dates = [m[0] for m in new_meta]
    new_pit = [m[3] for m in new_meta]
    new_nonpit = [m[4] for m in new_meta]

    print(f"[live] PIT loadings generation for {len(new_pit)} new dates ...")
    new_pit_texts, new_pit_parsed = _generate_and_parse(lm, new_pit, new_dates, "PIT")
    print(f"[live] PIT scoring ...")
    new_pit_scores = _score_with_retry(scorer, new_pit)
    print(f"[live] non-PIT loadings generation (identifying, 2048 up front) ...")
    new_np_texts, new_np_parsed = _generate_and_parse(lm, new_nonpit, new_dates, "non-PIT")
    print(f"[live] non-PIT scoring ...")
    new_np_scores = _score_with_retry(scorer, new_nonpit)

    # --- S3: full-stream loadings/scores artifacts + replay maps ---------------- #
    def _assemble(loadings_old: pd.DataFrame, scores_old: pd.DataFrame,
                  new_parsed: list, new_texts: list, new_scores: list, variant: str,
                  src_artifact: str, src_artifact_sha256: str):
        """Full-stream loadings/scores frames + one DatedFactorEvidence per
        rebalance date (task 6.2; dated replay consumes the evidence, task 6.3)."""
        new_by_date = {rb: (rl, txt, sc) for rb, rl, txt, sc
                       in zip(new_dates, new_parsed, new_texts, new_scores)}
        load_rows, score_rows = [], []
        evidence: list[DatedFactorEvidence] = []
        for (rb, _, _, pit, nonpit) in factor_meta:
            if rb in v1_dates:
                ld = loadings_dict_from_row(loadings_old.loc[rb])
                reply = synth_loadings_reply(ld)
                fail = scores_old.loc[rb, "fail_reason"]
                sc = factor_score_from_row(scores_old.loc[rb, "p_memorized"], fail)
                seg, src, src_sha = "replayed_v1", src_artifact, src_artifact_sha256
            else:
                rl, reply, sc = new_by_date[rb]
                ld = dict(rl.loadings) if rl is not None else None
                seg, src, src_sha = "live_ext2026", None, None
            evidence.append(build_dated_evidence(
                variant=variant, rebalance_date=rb, segment=seg,
                pit_prompt=pit, source_prompt=pit if variant == "pit" else nonpit,
                response_text=reply, score=sc, loadings=ld,
                source_artifact=src, source_artifact_sha256=src_sha))
            lrow = {"date": rb, "parse_ok": ld is not None, "segment": seg, "variant": variant}
            for axis in fs.MACRO_AXES:
                lrow[axis] = ld[axis] if ld is not None else float("nan")
            load_rows.append(lrow)
            score_rows.append({"date": rb, "p_memorized": sc.p_memorized,
                               "fail_reason": sc.fail_reason, "segment": seg, "variant": variant})
        return (pd.DataFrame(load_rows).set_index("date"),
                pd.DataFrame(score_rows).set_index("date"),
                evidence)

    loadings_ext, scores_ext, evidence_pit = _assemble(
        loadings_v1, scores_v1, new_pit_parsed, new_pit_texts, new_pit_scores, "pit",
        "data/factor_loadings_v1.parquet", sha256_file(DATA / "factor_loadings_v1.parquet"))
    np_loadings_ext, np_scores_ext, evidence_np = _assemble(
        np_loadings_v1, np_scores_v1, new_np_parsed, new_np_texts, new_np_scores,
        "nonpit_diagnostic",
        "data/factor_nonpit_diagnostic_loadings_v1.parquet",
        sha256_file(DATA / "factor_nonpit_diagnostic_loadings_v1.parquet"))

    # task 6.2: validate + persist the dated evidence into a NEW empty run
    # staging directory BEFORE any portfolio artifact is written (promotion,
    # manifest and COMPLETED marker are task 6.9).
    evidence_records = evidence_pit + evidence_np
    expected_evidence_keys = [(v, rb.date()) for v in _VARIANTS for rb in meta_dates]
    evidence_run_dir = OUT / "evidence_staging" / datetime.now(timezone.utc).strftime(
        "run_%Y%m%dT%H%M%SZ")
    evidence_path = write_evidence_table(
        evidence_records, expected_evidence_keys, evidence_run_dir)
    print(f"dated evidence: {len(evidence_records)} records "
          f"({len(meta_dates)} dates x {len(_VARIANTS)} variants) -> {evidence_path}")
    # task 6.3: the immutable (variant, date)-keyed mapping BOTH sim consumption
    # paths resolve from — validated up front, so the full expected key set is
    # known good before any rebalance runs.
    evidence_map = validate_evidence_records(evidence_records, expected_evidence_keys)

    loadings_ext.to_parquet(OUT / "factor_loadings_ext2026.parquet")
    scores_ext.to_parquet(OUT / "factor_scores_ext2026.parquet")
    np_loadings_ext.to_parquet(OUT / "factor_nonpit_diagnostic_loadings_ext2026.parquet")
    np_scores_ext.to_parquet(OUT / "factor_nonpit_diagnostic_scores_ext2026.parquet")
    print(f"loadings parsed: PIT {int(loadings_ext['parse_ok'].sum())}/{len(loadings_ext)} | "
          f"non-PIT {int(np_loadings_ext['parse_ok'].sum())}/{len(np_loadings_ext)}")

    # --- S4: naive directional eval over the full stream ------------------------ #
    swda = prices["SWDA.L"].ffill()
    next_rb = {rb: (rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else swda.index.max())
               for i, rb in enumerate(rebalance_dates)}

    def _realized_dir(rb: pd.Timestamp) -> int:
        p0, p1 = float(swda.asof(rb)), float(swda.asof(next_rb[rb]))
        return int(np.sign(p1 / p0 - 1.0))

    def _t12m(col: pd.Series) -> float:
        p = col.dropna()
        return float(p.iloc[-1] / p.iloc[-253] - 1.0) if len(p) >= 253 else float("nan")

    def _vol(col: pd.Series) -> float:
        tail = col.dropna().tail(252)
        return float(tail.std(ddof=1) * np.sqrt(252)) if len(tail) >= 30 else float("nan")

    def _asset_snapshot_stats(rb: pd.Timestamp) -> list[dict]:
        price_hist = prices[symbols].loc[prices.index < rb].tail(LOOKBACK_DAYS)
        ret_hist = all_returns.loc[all_returns.index < rb].tail(LOOKBACK_DAYS).dropna(how="any")
        snap = []
        for real, pseudo in asset_map.real_to_pseudo.items():
            snap.append({"id": pseudo, "category": asset_map.categories[pseudo],
                         "trailing_12m_return": _t12m(price_hist[real]),
                         "trailing_vol_ann": _vol(ret_hist[real])})
        return snap

    naive_new_prompts = [steering.render_directional(m[1], _asset_snapshot_stats(m[0]))
                         for m in new_meta]
    print(f"[live] naive directional generation for {len(naive_new_prompts)} new dates ...")
    naive_replies = _generate_big(lm, naive_new_prompts, max_tokens=2048)
    fmt_bad = [i for i, r in enumerate(naive_replies)
               if isinstance(r, BaseException) or not _DIR_RE.search(r.content)]
    if fmt_bad:
        for i, r in zip(fmt_bad, _generate_big(lm, [naive_new_prompts[i] for i in fmt_bad],
                                               max_tokens=4096)):
            naive_replies[i] = r
        print(f"  [naive] format-retried {len(fmt_bad)} replies at max_tokens=4096")

    naive_new_rows = []
    for (rb, *_), prompt, reply in zip(new_meta, naive_new_prompts, naive_replies):
        realized = _realized_dir(rb)
        if isinstance(reply, BaseException):
            naive_new_rows.append({"date": rb, "prompt": prompt,
                                   "reply": f"<generate failed: {type(reply).__name__}>",
                                   "predicted_direction": None, "confidence": None,
                                   "realized_direction": realized, "correct": None})
            continue
        text = reply.content
        dm, cm = _DIR_RE.findall(text), _CONF_RE.findall(text)
        pred = int(dm[-1]) if dm else None
        naive_new_rows.append({"date": rb, "prompt": prompt, "reply": text,
                               "predicted_direction": pred,
                               "confidence": float(cm[-1]) if cm else None,
                               "realized_direction": realized,
                               "correct": (pred == realized) if pred is not None else None})

    naive_ext = pd.concat([naive_v1, pd.DataFrame(naive_new_rows)], ignore_index=True)
    naive_ext.to_parquet(OUT / "naive_directional_eval_ext2026.parquet")
    directional = naive_ext[naive_ext["predicted_direction"].isin([-1, 1])]
    n_dir = len(directional)
    n_correct = int((directional["predicted_direction"] == directional["realized_direction"]).sum())
    acc = n_correct / n_dir if n_dir else float("nan")
    ci_lo, ci_hi = wilson_ci(n_correct, n_dir)
    print(f"naive directional accuracy (full stream, n={n_dir}): {acc:.3f} "
          f"Wilson 95% CI [{ci_lo:.3f}, {ci_hi:.3f}] | 0.5 inside: {ci_lo <= 0.5 <= ci_hi}")

    # --- S5: walk-forward lines (PIT deployable + non-PIT diagnostic) ----------- #
    bl_agent = mf.LlmMacroAgent(asset_map=asset_map)

    def build_inputs(ctx):
        mz = ctx["macro_panel"][PANEL_Z_COLS].dropna()
        return mz.iloc[-1].to_dict(), factor_snapshot, ctx["rebalance_date"], None

    def combine(ctx, P, Q):
        """nb09's allocation UNCHANGED: HRP-CVaR base (BIL 25%) + BL, 0.7/0.3 blend."""
        returns_hist = ctx["returns"]
        w_hrp = mf.hrp_cvar_weights_with_fixed(
            returns_hist, {"BIL": _regime_cash_pin(returns_hist, REGIME_OVERLAY)})
        if P is None:
            return w_hrp
        try:
            w_bl = mf.bl_mv_weights(returns_hist, prior_weights=w_hrp, P=P, Q=Q, obj="Utility")
        except Exception:  # noqa: BLE001 -- BL can fail on degenerate inputs
            return w_hrp
        w = (1.0 - TILT) * w_hrp + TILT * w_bl
        return w / w.sum()

    # task 6.4: ONE run-local consumption dict for both variants and both
    # passes — every fingerprint is audited against the source evidence before
    # any publishable output is written.
    replay_consumption: dict[EvidenceKey, dict[str, object]] = {}

    def run_variant_line(name: str, variant: str,
                         evidence: Mapping[EvidenceKey, DatedFactorEvidence]):
        """One variant through the SAME pipeline; dated (variant, date) replay,
        zero NIM calls (task 6.3)."""
        failures: list[ReplayValidationError] = []
        weight_fn = make_dated_replay_weight_fn(
            variant=variant, evidence=evidence, agent=bl_agent,
            build_inputs=build_inputs, combine=combine, failures=failures,
            consumed=replay_consumption)
        targets = mf.build_walk_forward_targets(
            prices[symbols], rebalance_dates=rebalance_dates,
            weight_fns={name: weight_fn}, macro_panel=panel,
            lookback_days=LOOKBACK_DAYS)[name]
        # build_walk_forward_targets swallows weight_fn exceptions ("holding
        # previous") — replay integrity failures are re-raised here, never
        # warned past (R6.3).
        if failures:
            raise failures[0]
        pf = mf.run_rebalance_sim(prices[symbols], targets, init_cash=INIT_CASH)

        dlog = {"p_memorized": {}, "parse_ok": {}, "steered": {}, "conviction": {},
                "loadings": {}, "views": {}}
        for rb in rebalance_dates:
            try:
                macro_hist = panel.loc[panel.index < rb]
                price_hist = prices[symbols].loc[prices.index < rb].tail(LOOKBACK_DAYS)
                ret_hist = all_returns.loc[all_returns.index < rb].tail(LOOKBACK_DAYS).dropna(how="any")
                if macro_hist.empty or price_hist.shape[0] < 60 or ret_hist.shape[0] < 60:
                    continue
                ctx = {"rebalance_date": rb, "prices": price_hist, "returns": ret_hist,
                       "macro_panel": macro_hist}
                macro_state, snap, as_of, _raw = build_inputs(ctx)
                # task 6.3: the decision log consumes the SAME dated evidence —
                # exact (variant, date) resolution, not prompt-keyed maps.
                rec_d = resolve_dated_evidence(evidence, variant, rb)
                # task 6.4: second legitimate consumption of this key — must
                # fingerprint identically to the weight_fn pass.
                record_consumption(replay_consumption, (variant, rb.date()), rec_d)
                gen_d, replay_d = dated_replay_closures(rec_d)
                dec = fs.factor_rebalance(
                    generate_loadings=gen_d, scorer=replay_d, agent=bl_agent,
                    macro_state=macro_state, asset_snapshot=snap,
                    real_symbols=symbols, as_of=as_of)
                # task 6.4: the resulting decision identity must derive from
                # exactly the consumed evidence (R6.4).
                record_decision_identity(
                    replay_consumption, (variant, rb.date()), rec_d, dec)
                dlog["p_memorized"][rb] = dec.p_memorized
                dlog["parse_ok"][rb] = bool(dec.parse_ok)
                dlog["steered"][rb] = bool(dec.steered)
                dlog["conviction"][rb] = float(dec.views[0].confidence) if dec.views else None
                dlog["loadings"][rb] = dict(dec.loadings.loadings) if dec.loadings is not None else None
                dlog["views"][rb] = [v.to_dict() for v in dec.views]
            except ReplayValidationError:
                raise  # replay integrity is fatal — never a warned-past log row (R6.3)
            except Exception as exc:  # noqa: BLE001 -- per-date resilience
                dlog["parse_ok"][rb] = False
                dlog["steered"][rb] = False
                dlog["views"][rb] = [f"<decision failed: {type(exc).__name__}>"]
        return targets, pf, dlog

    print("[sim] extended PIT deployable line ...")
    targets_ext, pf_ext, dlog_ext = run_variant_line(
        "factor_ext2026", "pit", evidence_map)
    print("[sim] extended non-PIT diagnostic line ...")
    targets_np_ext, pf_np_ext, dlog_np_ext = run_variant_line(
        "factor_nonpit_ext2026", "nonpit_diagnostic", evidence_map)

    # task 6.4: source-to-consumption replay audit. Prove the dated source
    # evidence equals every value BOTH passes consumed for every expected
    # (variant, date), then persist the passing summary — BEFORE any targets,
    # equity, decision-log, metric, or completion-state output is written. A
    # cross-date swap, absent, or altered key raises ReplayValidationError here
    # and blocks every publishable portfolio output downstream.
    audit_path = write_replay_audit_summary(
        evidence_map, replay_consumption, expected_evidence_keys, OUT)
    print(f"replay audit: source==consumption for {len(expected_evidence_keys)} "
          f"(variant, date) keys -> {audit_path}")

    equity_ext = pf_ext.value()
    equity_np_ext = pf_np_ext.value()

    # task 6.6: immutable run-local finance records. The completed snapshot is a
    # mandatory upstream input; no live-price fallback or zero cash substitution.
    from scripts import build_basket_long as snapshot_producer

    snapshot_dir = Path(
        os.environ.get("MARKET_SNAPSHOT_DIR")
        or DATA / "market_snapshots" / snapshot_producer.SNAPSHOT_ID
    )
    metric_bundle = build_factor_metric_records(
        equity_ext, equity_np_ext, snapshot_dir=snapshot_dir
    )
    metric_records_path = write_factor_metric_records(
        metric_bundle,
        OUT,
        pit_value=equity_ext,
        nonpit_value=equity_np_ext,
        snapshot_dir=snapshot_dir,
    )
    metric_rows = {
        (row["portfolio_id"], row["schema"]): row
        for row in metric_bundle["records"]
    }
    pit_reader = metric_rows[("factor_pit_ext2026", mf.READER_SCHEMA)]
    nonpit_reader = metric_rows[("factor_nonpit_diagnostic_ext2026", mf.READER_SCHEMA)]
    differential_reader = metric_rows[
        ("factor_nonpit_minus_pit_ext2026", mf.DIFFERENTIAL_SCHEMA)
    ]
    print(
        f"metric records: {len(metric_bundle['records'])} "
        f"reader/legacy/differential/attribution/crisis records -> {metric_records_path}"
    )

    n_guarded = sum(1 for v in dlog_ext["steered"].values() if v)
    n_guarded_np = sum(1 for v in dlog_np_ext["steered"].values() if v)
    print(f"recall-guarded decisions: PIT {n_guarded}/{len(dlog_ext['steered'])} | "
          f"non-PIT {n_guarded_np}/{len(dlog_np_ext['steered'])}")

    # --- S6: consistency check — replayed 2019-2024 segment vs published v1 ----- #
    # equity_v1 carries a FLAT pre-start stub (1238 rows back to 2014-01-02) before its
    # first rebalance. Its index is therefore not the comparable span: over the stub v1 sits
    # at INIT_CASH while an earlier-starting extended line is already trading. Compare only
    # where v1 is actually running.
    _v1_moves = equity_v1[equity_v1.ne(equity_v1.iloc[0])]
    _v1_start = _v1_moves.index.min() if len(_v1_moves) else equity_v1.index.min()
    common = equity_ext.index.intersection(equity_v1.index)
    common = common[common >= _v1_start]
    # RE-BASE before comparing. With SIM_START earlier than v1's, the extended line has
    # already been compounding when `common` begins, so its LEVEL differs from v1 by
    # construction (~30% on a 2016 start) while the replayed segment is still identical.
    # The invariant that must hold is the PATH over the overlap, not the level. When
    # SIM_START == v1's start both series equal INIT_CASH on common[0] and this reduces
    # to the original ratio exactly, so the published check is unchanged.
    ext_rebased = equity_ext.loc[common] / equity_ext.loc[common].iloc[0]
    v1_rebased = equity_v1.loc[common] / equity_v1.loc[common].iloc[0]
    equity_rel_diff = float((ext_rebased / v1_rebased - 1.0).abs().max())
    t_common = targets_ext.loc[v1_dates].dropna(how="all")
    t_v1 = targets_v1.loc[t_common.index, t_common.columns]
    targets_max_diff = float((t_common - t_v1).abs().max().max())
    print(f"consistency: replayed-vs-v1 equity max rel diff = {equity_rel_diff:.2e} "
          f"(tol {EQUITY_REL_TOL:.0e}) | targets max abs diff = {targets_max_diff:.2e}")
    assert equity_rel_diff <= EQUITY_REL_TOL, (
        f"replayed 2019-2024 PIT equity drifted from factor_equity_v1.parquet: "
        f"max rel diff {equity_rel_diff:.4e} > {EQUITY_REL_TOL:.0e}")

    # --- S7: persist the extended factor lines ---------------------------------- #
    def _dump_line(prefix: str, targets: pd.DataFrame, equity: pd.Series, dlog: dict,
                   n_grd: int, line_desc: str) -> None:
        targets.to_parquet(OUT / f"{prefix}_targets_ext2026.parquet")
        equity.to_frame("value").to_parquet(OUT / f"{prefix}_equity_ext2026.parquet")
        payload = {
            "meta": {"nim_model": NIM_MODEL, "cutoff_date": CUTOFF.isoformat(),
                     "holdout_auc": float(scorer.holdout_auc), "is_weak": bool(scorer.is_weak),
                     "n_rebalances": len(dlog["steered"]), "n_recall_guarded": int(n_grd),
                     "window": f"{SIM_START}..{SIM_END_EXT}", "line": line_desc},
            **{k: {str(d): v for d, v in dd.items()} for k, dd in dlog.items()},
        }
        (OUT / f"{prefix}_decision_log_ext2026.json").write_text(
            json.dumps(payload, indent=2, default=str))

    _dump_line("factor", targets_ext, equity_ext, dlog_ext, n_guarded,
               "PIT anonymized deployable (recall-guarded), extended 2019-01..2026-06; "
               "2019-2024 replayed from v1 artifacts, 2025+ live")
    _dump_line("factor_nonpit_diagnostic", targets_np_ext, equity_np_ext, dlog_np_ext,
               n_guarded_np,
               "NON-PIT DIAGNOSTIC CONTROL extended 2019-01..2026-06 — NEVER deployable (R7.4)")

    # --- S8: baseline (nb07) + track B (nb08) over the extended window ---------- #
    print("[sim] extended baseline (HRP+momentum, nb07 logic) ...")
    baseline_targets = mf.build_walk_forward_targets(
        prices[symbols], rebalance_dates=rebalance_dates,
        weight_fns={"baseline": lambda ctx: mf.hrp_momentum_weights(ctx["returns"], ctx["prices"])},
        lookback_days=LOOKBACK_DAYS)["baseline"]
    pf_baseline = mf.run_rebalance_sim(prices[symbols], baseline_targets, init_cash=INIT_CASH)
    baseline_targets.to_parquet(OUT / "baseline_targets_ext2026.parquet")
    pf_baseline.value().to_frame("value").to_parquet(OUT / "baseline_equity_ext2026.parquet")

    print("[sim] extended track B (MC-Nash, nb08 logic; slow MC step) ...")
    rng = np.random.default_rng(42)

    def track_b_fn(ctx):
        panel_z_hist = ctx["macro_panel"][PANEL_Z_COLS].dropna()
        etf_w, _probs, _payoff, _mix = mf.mc_nash_asset_weights(
            panel_z_hist, ctx["returns"], ctx["macro_panel"], symbols=symbols,
            rng=rng, **TRACK_B)
        return etf_w

    track_b_targets = mf.build_walk_forward_targets(
        prices[symbols], rebalance_dates=rebalance_dates,
        weight_fns={"track_b": track_b_fn}, macro_panel=panel,
        lookback_days=LOOKBACK_DAYS)["track_b"]
    pf_track_b = mf.run_rebalance_sim(prices[symbols], track_b_targets, init_cash=INIT_CASH)
    track_b_targets.to_parquet(OUT / "track_b_targets_ext2026.parquet")
    pf_track_b.value().to_frame("value").to_parquet(OUT / "track_b_equity_ext2026.parquet")

    # --- S9: contrast + split table ---------------------------------------------- #
    # Cross-strategy report-table assembly is owned by the later canonical report
    # producer; this Factor run persists only its run-local records (task 6.6).
    contrast_df = pd.DataFrame({
        "pit_p": scores_ext["p_memorized"].astype(float),
        "nonpit_p": np_scores_ext["p_memorized"].astype(float),
    })
    contrast_df["delta"] = contrast_df["nonpit_p"] - contrast_df["pit_p"]
    contrast_df["segment"] = np.where(contrast_df.index <= pd.Timestamp(CUTOFF),
                                      "in_training", "post_cutoff")
    contrast_df.to_parquet(OUT / "factor_contrast_ext2026.parquet")

    split = split_contrast_table(contrast_df, cutoff=pd.Timestamp(CUTOFF))
    in_premium = split["in_training"]["mean_delta"]

    # Reproduce the published premium ON THE SAME DATES it was computed over — v1's own
    # rebalances. With SIM_START before 2019 the extended in-training segment also spans
    # months v1 never saw, so comparing the two means compares different samples, not two
    # measurements of one thing. And read the reference from the artifact rather than a
    # frozen constant: the published value moved 0.5283 -> 0.3996 when nb14 was re-run on
    # 2026-07-27 and the hardcoded copy silently went stale.
    published_premium = float(
        json.loads((DATA / "factor_contrast_summary_v1.json").read_text())
        ["contamination_premium"]["p_memorized_mean_delta"])
    repro_dates = contrast_df.index.intersection(pd.DatetimeIndex(v1_dates))
    premium_on_v1_dates = float(contrast_df.loc[repro_dates, "delta"].mean())
    premium_repro_diff = abs(premium_on_v1_dates - published_premium)
    assert premium_repro_diff <= PREMIUM_REPRO_TOL, (
        f"premium on v1's own {len(repro_dates)} dates {premium_on_v1_dates:+.4f} does not "
        f"reproduce the published {published_premium:+.4f} within {PREMIUM_REPRO_TOL}")

    split_payload = {
        **split,
        "cutoff_date": CUTOFF.isoformat(),
        "published_v1_full_stream_premium": published_premium,
        "premium_on_v1_dates": premium_on_v1_dates,
        "in_training_reproduction_abs_diff": premium_repro_diff,
        "collapse_rule": f"post-cutoff |mean_delta| < {COLLAPSE_FRACTION} * in-training |mean_delta|",
        "nim_model": NIM_MODEL,
    }
    (OUT / "factor_contrast_split_ext2026.json").write_text(
        json.dumps(split_payload, indent=2, sort_keys=True))

    print("\n=== SPLIT TABLE: PIT-vs-non-PIT p_memorized premium ===")
    split_tbl = pd.DataFrame({k: split[k] for k in ("in_training", "post_cutoff", "full_stream")}).T
    print(split_tbl.round(4).to_string())
    print(f"\n{split['prediction_outcome']}")
    print(f"(on v1's own dates reproduces published +{published_premium:.3f} "
          f"within {premium_repro_diff:.4f})")

    # --- S10: CSV mirrors (US + _de) for non-report run artifacts --------------- #
    # Reader/legacy/differential records stay in their manifest-owned JSON source;
    # canonical table and locale projection are owned by the report producer.
    print("[csv] writing ext2026 mirrors (US + _de) ...")
    plain = {
        "factor_loadings_ext2026": loadings_ext,
        "factor_scores_ext2026": scores_ext,
        "factor_nonpit_diagnostic_loadings_ext2026": np_loadings_ext,
        "factor_nonpit_diagnostic_scores_ext2026": np_scores_ext,
        "factor_targets_ext2026": targets_ext,
        "factor_nonpit_diagnostic_targets_ext2026": targets_np_ext,
        "baseline_targets_ext2026": baseline_targets,
        "track_b_targets_ext2026": track_b_targets,
        "factor_contrast_ext2026": contrast_df,
        "naive_directional_eval_ext2026": naive_ext,
    }
    for name, frame in plain.items():
        _write_csv(frame, name, index=(name != "naive_directional_eval_ext2026"))
    for name, equity in {"factor_equity_ext2026": equity_ext,
                         "factor_nonpit_diagnostic_equity_ext2026": equity_np_ext,
                         "baseline_equity_ext2026": pf_baseline.value(),
                         "track_b_equity_ext2026": pf_track_b.value()}.items():
        eq = equity.to_frame("value")
        eq["daily_return"] = eq["value"].pct_change()
        eq["drawdown"] = eq["value"] / eq["value"].cummax() - 1
        _write_csv(eq, name)
    for name in ("factor_decision_log_ext2026", "factor_nonpit_diagnostic_decision_log_ext2026"):
        _write_csv(_flatten_decision_log(json.loads((OUT / f"{name}.json").read_text())), name)
    _write_csv(pd.DataFrame({k: split[k] for k in ("in_training", "post_cutoff", "full_stream")}).T
               .rename_axis("segment"), "factor_contrast_split_ext2026")

    # --- S11: immutable Factor run manifest and bundle (task 6.9) ---------------- #
    # Stable run identity; the bundle validates from its manifest alone and the
    # COMPLETED marker (carrying the manifest sha256) is written LAST.
    factor_run_id = f"factor_ext2026_{SIM_START}_{SIM_END_EXT}_v1"
    bundle_artifacts = {
        role: (evidence_path if role == "evidence" else OUT / spec["file"])
        for role, spec in FACTOR_RUN_ARTIFACTS.items()
    }
    v1_input_names = (
        "factor_loadings_v1",
        "factor_scores_v1",
        "factor_nonpit_diagnostic_loadings_v1",
        "factor_nonpit_diagnostic_scores_v1",
        "factor_targets_v1",
        "factor_equity_v1",
    )
    factor_run_dir = build_factor_run_bundle(
        run_id=factor_run_id,
        output_root=OUT / "factor_runs",
        artifacts=bundle_artifacts,
        config={
            "sim_start": SIM_START,
            "sim_end": SIM_END_EXT,
            "price_fetch_end": PRICE_FETCH_END,
            "lookback_days": LOOKBACK_DAYS,
            "tilt": TILT,
            "init_cash": INIT_CASH,
            "panel_source": panel_source,
        },
        source_commit=_git_source_commit(),
        model={
            "nim_model": NIM_MODEL,
            "cutoff_date": CUTOFF.isoformat(),
            "calibrator_dir": CAL_DIR.name,
            "holdout_auc": float(scorer.holdout_auc),
            "is_weak": bool(scorer.is_weak),
        },
        input_manifests={
            "market_snapshot": {
                "snapshot_id": metric_bundle["market_snapshot"]["snapshot_id"],
                "manifest_sha256": metric_bundle["market_snapshot"]["manifest_sha256"],
            },
            **{
                name: {
                    "path": f"data/{name}.parquet",
                    "sha256": sha256_file(DATA / f"{name}.parquet"),
                }
                for name in v1_input_names
            },
        },
        expected_evidence={
            "variants": sorted(_VARIANTS),
            "dates": [rb.date().isoformat() for rb in meta_dates],
        },
    )
    print(f"factor run bundle COMPLETED: {factor_run_dir}")

    # --- S13: run header ---------------------------------------------------------- #
    header = {
        "task": "8.1 post-cutoff extension (data-v3)",
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "nim_model": NIM_MODEL,
        "cutoff_date": CUTOFF.isoformat(),
        "calibrator": {"dir": CAL_DIR.name, "holdout_auc": float(scorer.holdout_auc),
                       "is_weak": bool(scorer.is_weak)},
        "window": {"sim_start": SIM_START, "sim_end": SIM_END_EXT,
                   "first_rebalance": str(rebalance_dates[0].date()),
                   "last_rebalance": str(rebalance_dates[-1].date())},
        "n_rebalances": {"total": len(meta_dates), "replayed_v1": len(meta_dates) - len(new_meta),
                         "live_new": len(new_meta)},
        "panel": {"source": panel_source, "rows": int(len(panel)),
                  "span": f"{panel.index.min().date()}..{panel.index.max().date()}"},
        "prices": {"source": "yfinance (documented DB substitution)",
                   "span": f"{prices.index.min().date()}..{prices.index.max().date()}"},
        "parse": {"pit_parsed": int(loadings_ext["parse_ok"].sum()),
                  "nonpit_parsed": int(np_loadings_ext["parse_ok"].sum()),
                  "n_rows": len(loadings_ext)},
        # task 6.2: dated evidence table provenance (full run manifest is task 6.9)
        "dated_evidence": {"path": str(evidence_path),
                           "n_records": len(evidence_records),
                           "n_expected_keys": len(expected_evidence_keys)},
        # task 6.9: the immutable, manifest-validated Factor run bundle
        "factor_run_bundle": {"run_id": factor_run_id, "dir": str(factor_run_dir)},
        "consistency": {
            "replay_equity_max_rel_diff_vs_v1": equity_rel_diff,
            "replay_equity_tolerance": EQUITY_REL_TOL,
            "replay_targets_max_abs_diff_vs_v1": targets_max_diff,
            "in_training_premium": in_premium,
            "published_v1_premium": published_premium,
            "in_training_reproduction_abs_diff": premium_repro_diff,
        },
        "split_table": {k: split[k] for k in ("in_training", "post_cutoff", "full_stream")},
        "prediction_outcome": split["prediction_outcome"],
        "naive_eval": {"n_directional": n_dir, "accuracy": acc,
                       "wilson_ci": [ci_lo, ci_hi], "half_inside_ci": bool(ci_lo <= 0.5 <= ci_hi)},
        "metric_records": {
            "path": str(metric_records_path),
            "schema": metric_bundle["schema"],
            "n_records": len(metric_bundle["records"]),
            "schema_ids": sorted({row["schema"] for row in metric_bundle["records"]}),
            "market_snapshot": metric_bundle["market_snapshot"],
            "ssr_settings": metric_bundle["ssr_settings"],
        },
        "immutability": "all outputs under NEW *_ext2026 names; v1/v2 artifacts untouched (data-v2 immutable)",
    }
    (OUT / "factor_ext2026_run_header.json").write_text(
        json.dumps(header, indent=2, sort_keys=True, default=str))

    print("\n=== headline numbers ===")
    print(f"stream: {len(meta_dates)} rebalances {rebalance_dates[0].date()} -> "
          f"{rebalance_dates[-1].date()} ({len(new_meta)} new live)")
    print(f"premium in-training: mean {split['in_training']['mean_delta']:+.4f} "
          f"(d={split['in_training']['paired_d']:.2f}, n={split['in_training']['n_pairs']}) | "
          f"post-cutoff: mean {split['post_cutoff']['mean_delta']:+.4f} "
          f"(d={split['post_cutoff']['paired_d']:.2f}, n={split['post_cutoff']['n_pairs']})")
    print(f"PIT total_return {float(pit_reader['total_return']):+.4f} vs non-PIT "
          f"{float(nonpit_reader['total_return']):+.4f} | differential SSR "
          f"{float(differential_reader['ssr_ssr']):+.2f}")
    print("[done] all *_ext2026 artifacts written under data/ (+ csv_mirrors, tear_sheet)")


if __name__ == "__main__":
    main()
