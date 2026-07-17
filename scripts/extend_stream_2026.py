"""Task 8.1 — extend the factor stream to 2026: the post-cutoff natural experiment (data-v3).

Extends the nb13/nb14 walk-forward (PIT deployable + non-PIT diagnostic + naive
directional eval) beyond 2024-12 with the SAME renderer, calibrator
(``openai/gpt-oss-20b`` @ cutoff 2024-06-01), recall guard and 0.7/0.3 HRP+BL
blend, as far as the macro panel allows. The 2019-2024 segment REPLAYS the
persisted v1 loadings/scores (zero NIM calls — the nb11/nb13 pre-scored replay
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
when FRED is unreachable — the source is recorded in the run header);
prices via yfinance (documented DB substitution, mirrors nb11/nb13/nb14).

Reproducible: ``uv run python scripts/extend_stream_2026.py`` (needs
``NVIDIA_API_KEY`` in ``.env`` for the ~90-150 live NIM calls).
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

import macro_framework as mf
from macro_framework import factor_scoring as fs
from macro_framework import macro as macro_module
from macro_framework import steering
from macro_framework.factor_scoring import _median, _paired_cohens_d
from macro_framework.ssr import compute_ssr

warnings.filterwarnings("ignore")

# --- constants (mirror nb13/nb14 exactly) ---------------------------------- #
DATA = REPO / "data"
CSV_OUT = DATA / "csv_mirrors"
TEAR_OUT = DATA / "tear_sheet"
INIT_CASH = 10_000.0
SIM_START = "2019-01-01"
SIM_END_EXT = "2026-06-30"  # task 8.1: extend to 2026-06 (~18 new monthly rebalances)
PRICE_FETCH_END = "2026-07-01"
LOOKBACK_DAYS = 756
TILT = 0.30  # nb09 final blend = 0.7*HRP + 0.3*BL

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

# Published in-training reference (factor_contrast_summary_v1.json,
# contamination_premium.p_memorized_mean_delta over the full 72-pair v1 stream).
PUBLISHED_V1_PREMIUM = 0.5282818618139323
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


class _ReplayScorer:
    """Replay scorer exposing the surface factor_rebalance uses (is_weak + score).

    Backed by pre-computed ``FactorScore``s keyed on the prompt string, so no
    NIM call happens during the walk-forward; an unseen prompt degrades to an
    unscored result and the module's R4.3 passthrough handles it.
    """

    is_weak = False  # the loaded calibrator is strong (asserted in main)

    def __init__(self, by_prompt: dict) -> None:
        self._by = dict(by_prompt)

    def score(self, prompt: str) -> fs.FactorScore:
        sc = self._by.get(prompt)
        if sc is None:
            return fs.FactorScore(p_memorized=None, parse_ok=False, fail_reason="not_pre_scored")
        return sc

    def score_many(self, prompts, *, max_workers: int = 8):
        return [self.score(p) for p in prompts]


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

    meta_dates = [m[0] for m in factor_meta]
    pit_prompts = [m[3] for m in factor_meta]
    if len(set(pit_prompts)) != len(pit_prompts):
        print("WARNING: duplicate PIT prompts across dates (identical 2dp macro states) — "
              "replay maps key by prompt; the later date wins for the duplicated key")

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
                  new_parsed: list, new_texts: list, new_scores: list, variant: str):
        """Full-stream loadings/scores frames + prompt-keyed reply/score maps."""
        reply_by_prompt: dict[str, str] = {}
        score_by_prompt: dict[str, fs.FactorScore] = {}
        new_by_date = {rb: (rl, txt, sc) for rb, rl, txt, sc
                       in zip(new_dates, new_parsed, new_texts, new_scores)}
        load_rows, score_rows = [], []
        for (rb, _, _, pit, _) in factor_meta:
            if rb in v1_dates:
                ld = loadings_dict_from_row(loadings_old.loc[rb])
                reply = synth_loadings_reply(ld)
                fail = scores_old.loc[rb, "fail_reason"]
                sc = factor_score_from_row(scores_old.loc[rb, "p_memorized"], fail)
                seg = "replayed_v1"
            else:
                rl, reply, sc = new_by_date[rb]
                ld = dict(rl.loadings) if rl is not None else None
                seg = "live_ext2026"
            reply_by_prompt[pit] = reply
            score_by_prompt[pit] = sc
            lrow = {"date": rb, "parse_ok": ld is not None, "segment": seg, "variant": variant}
            for axis in fs.MACRO_AXES:
                lrow[axis] = ld[axis] if ld is not None else float("nan")
            load_rows.append(lrow)
            score_rows.append({"date": rb, "p_memorized": sc.p_memorized,
                               "fail_reason": sc.fail_reason, "segment": seg, "variant": variant})
        return (pd.DataFrame(load_rows).set_index("date"),
                pd.DataFrame(score_rows).set_index("date"),
                reply_by_prompt, score_by_prompt)

    loadings_ext, scores_ext, reply_by_prompt, score_by_prompt = _assemble(
        loadings_v1, scores_v1, new_pit_parsed, new_pit_texts, new_pit_scores, "pit")
    np_loadings_ext, np_scores_ext, np_reply_by_prompt, np_score_by_prompt = _assemble(
        np_loadings_v1, np_scores_v1, new_np_parsed, new_np_texts, new_np_scores,
        "nonpit_diagnostic")

    loadings_ext.to_parquet(DATA / "factor_loadings_ext2026.parquet")
    scores_ext.to_parquet(DATA / "factor_scores_ext2026.parquet")
    np_loadings_ext.to_parquet(DATA / "factor_nonpit_diagnostic_loadings_ext2026.parquet")
    np_scores_ext.to_parquet(DATA / "factor_nonpit_diagnostic_scores_ext2026.parquet")
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
    naive_ext.to_parquet(DATA / "naive_directional_eval_ext2026.parquet")
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
        w_hrp = mf.hrp_cvar_weights_with_fixed(returns_hist, {"BIL": 0.25})
        if P is None:
            return w_hrp
        try:
            w_bl = mf.bl_mv_weights(returns_hist, prior_weights=w_hrp, P=P, Q=Q, obj="Utility")
        except Exception:  # noqa: BLE001 -- BL can fail on degenerate inputs
            return w_hrp
        w = (1.0 - TILT) * w_hrp + TILT * w_bl
        return w / w.sum()

    def run_variant_line(name: str, replies: dict, scores: dict):
        """One variant through the SAME pipeline (nb14 pattern); zero NIM calls."""
        replay = _ReplayScorer(scores)

        def gen(prompt: str) -> str:
            return replies.get(prompt, "")

        weight_fn = fs.make_factor_weight_fn(
            generate_loadings=gen, scorer=replay, agent=bl_agent,
            build_inputs=build_inputs, combine=combine)
        targets = mf.build_walk_forward_targets(
            prices[symbols], rebalance_dates=rebalance_dates,
            weight_fns={name: weight_fn}, macro_panel=panel,
            lookback_days=LOOKBACK_DAYS)[name]
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
                dec = fs.factor_rebalance(
                    generate_loadings=gen, scorer=replay, agent=bl_agent,
                    macro_state=macro_state, asset_snapshot=snap,
                    real_symbols=symbols, as_of=as_of)
                dlog["p_memorized"][rb] = dec.p_memorized
                dlog["parse_ok"][rb] = bool(dec.parse_ok)
                dlog["steered"][rb] = bool(dec.steered)
                dlog["conviction"][rb] = float(dec.views[0].confidence) if dec.views else None
                dlog["loadings"][rb] = dict(dec.loadings.loadings) if dec.loadings is not None else None
                dlog["views"][rb] = [v.to_dict() for v in dec.views]
            except Exception as exc:  # noqa: BLE001 -- per-date resilience
                dlog["parse_ok"][rb] = False
                dlog["steered"][rb] = False
                dlog["views"][rb] = [f"<decision failed: {type(exc).__name__}>"]
        return targets, pf, dlog

    print("[sim] extended PIT deployable line ...")
    targets_ext, pf_ext, dlog_ext = run_variant_line(
        "factor_ext2026", reply_by_prompt, score_by_prompt)
    print("[sim] extended non-PIT diagnostic line ...")
    targets_np_ext, pf_np_ext, dlog_np_ext = run_variant_line(
        "factor_nonpit_ext2026", np_reply_by_prompt, np_score_by_prompt)

    equity_ext = pf_ext.value()
    equity_np_ext = pf_np_ext.value()
    n_guarded = sum(1 for v in dlog_ext["steered"].values() if v)
    n_guarded_np = sum(1 for v in dlog_np_ext["steered"].values() if v)
    print(f"recall-guarded decisions: PIT {n_guarded}/{len(dlog_ext['steered'])} | "
          f"non-PIT {n_guarded_np}/{len(dlog_np_ext['steered'])}")

    # --- S6: consistency check — replayed 2019-2024 segment vs published v1 ----- #
    common = equity_ext.index.intersection(equity_v1.index)
    equity_rel_diff = float((equity_ext.loc[common] / equity_v1.loc[common] - 1.0).abs().max())
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
        targets.to_parquet(DATA / f"{prefix}_targets_ext2026.parquet")
        equity.to_frame("value").to_parquet(DATA / f"{prefix}_equity_ext2026.parquet")
        payload = {
            "meta": {"nim_model": NIM_MODEL, "cutoff_date": CUTOFF.isoformat(),
                     "holdout_auc": float(scorer.holdout_auc), "is_weak": bool(scorer.is_weak),
                     "n_rebalances": len(dlog["steered"]), "n_recall_guarded": int(n_grd),
                     "window": f"{SIM_START}..{SIM_END_EXT}", "line": line_desc},
            **{k: {str(d): v for d, v in dd.items()} for k, dd in dlog.items()},
        }
        (DATA / f"{prefix}_decision_log_ext2026.json").write_text(
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
    baseline_targets.to_parquet(DATA / "baseline_targets_ext2026.parquet")
    pf_baseline.value().to_frame("value").to_parquet(DATA / "baseline_equity_ext2026.parquet")

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
    track_b_targets.to_parquet(DATA / "track_b_targets_ext2026.parquet")
    pf_track_b.value().to_frame("value").to_parquet(DATA / "track_b_equity_ext2026.parquet")

    # --- S9: head-to-head + contrast + split table ------------------------------- #
    pfs = {"Baseline (ext2026)": pf_baseline, "Track B (ext2026)": pf_track_b,
           "Factor PIT (ext2026)": pf_ext, "Non-PIT DIAGNOSTIC (ext2026)": pf_np_ext}
    tmap = {"Baseline (ext2026)": baseline_targets, "Track B (ext2026)": track_b_targets,
            "Factor PIT (ext2026)": targets_ext, "Non-PIT DIAGNOSTIC (ext2026)": targets_np_ext}
    report = mf.head_to_head_report(pfs, tmap, crisis_start="2022-01-01", crisis_end="2022-12-31")
    print("\n=== head-to-head (extended 2019-01..2026-06) ===")
    print(report.round(4).to_string())

    contrast_df = pd.DataFrame({
        "pit_p": scores_ext["p_memorized"].astype(float),
        "nonpit_p": np_scores_ext["p_memorized"].astype(float),
    })
    contrast_df["delta"] = contrast_df["nonpit_p"] - contrast_df["pit_p"]
    contrast_df["segment"] = np.where(contrast_df.index <= pd.Timestamp(CUTOFF),
                                      "in_training", "post_cutoff")
    contrast_df.to_parquet(DATA / "factor_contrast_ext2026.parquet")

    split = split_contrast_table(contrast_df, cutoff=pd.Timestamp(CUTOFF))
    in_premium = split["in_training"]["mean_delta"]
    premium_repro_diff = abs(in_premium - PUBLISHED_V1_PREMIUM)
    assert premium_repro_diff <= PREMIUM_REPRO_TOL, (
        f"in-training premium {in_premium:+.4f} does not reproduce the published "
        f"{PUBLISHED_V1_PREMIUM:+.4f} within {PREMIUM_REPRO_TOL}")

    split_payload = {
        **split,
        "cutoff_date": CUTOFF.isoformat(),
        "published_v1_full_stream_premium": PUBLISHED_V1_PREMIUM,
        "in_training_reproduction_abs_diff": premium_repro_diff,
        "collapse_rule": f"post-cutoff |mean_delta| < {COLLAPSE_FRACTION} * in-training |mean_delta|",
        "nim_model": NIM_MODEL,
    }
    (DATA / "factor_contrast_split_ext2026.json").write_text(
        json.dumps(split_payload, indent=2, sort_keys=True))

    print("\n=== SPLIT TABLE: PIT-vs-non-PIT p_memorized premium ===")
    split_tbl = pd.DataFrame({k: split[k] for k in ("in_training", "post_cutoff", "full_stream")}).T
    print(split_tbl.round(4).to_string())
    print(f"\n{split['prediction_outcome']}")
    print(f"(in-training reproduces published +{PUBLISHED_V1_PREMIUM:.3f} "
          f"within {premium_repro_diff:.4f})")

    # --- S10: luck-vs-skill (compute_ssr, NW-HAC) over the extended lines -------- #
    sim_start_ts = rebalance_dates[0]
    ret_pit = equity_ext.loc[sim_start_ts:].pct_change().dropna()
    ret_np = equity_np_ext.loc[sim_start_ts:].pct_change().dropna()
    common_r = ret_pit.index.intersection(ret_np.index)
    ret_diff = (ret_np.loc[common_r] - ret_pit.loc[common_r]).rename("recall_minus_norecall")
    ssr_pit, ssr_np, ssr_diff = compute_ssr(ret_pit), compute_ssr(ret_np), compute_ssr(ret_diff)

    pit_row = report.loc["Factor PIT (ext2026)"]
    np_row = report.loc["Non-PIT DIAGNOSTIC (ext2026)"]
    z_ref = 1.96

    def _verdict(res, differential: bool) -> str:
        if not np.isfinite(res.ssr):
            return "insufficient rolling observations for HAC inference"
        if differential:
            if abs(res.ssr) < z_ref:
                return (f"|SSR|={abs(res.ssr):.2f} < {z_ref}: differential indistinguishable from "
                        f"zero under NW-HAC — the premium is LUCK-COMPATIBLE, not skill")
            return (f"|SSR|={abs(res.ssr):.2f} >= {z_ref}: QUANTIFIED LOOKAHEAD/RECALL BIAS, "
                    f"never attainable skill (R7.5)")
        if res.ssr >= z_ref:
            return (f"SSR={res.ssr:.2f} >= {z_ref}: rolling Sharpe stably above zero under NW-HAC "
                    f"(no skill claim — non-predictive framing, R6.4)")
        return f"SSR={res.ssr:.2f} < {z_ref}: NOT stably distinguishable from zero — luck-compatible"

    def _luck_row(name: str, res, total_return: float, sharpe: float, differential: bool) -> dict:
        return {"line": name, "n_obs": res.n_obs, "n_rolling": res.n_rolling,
                "total_return": total_return, "sharpe": sharpe,
                "mean_rolling_sr": res.mean_rolling_sr, "ssr": res.ssr,
                "nw_long_run_var": float(res.sigma_hac ** 2) if np.isfinite(res.sigma_hac) else float("nan"),
                "nw_sigma_hac": res.sigma_hac, "nw_bandwidth_L": res.L_hac,
                "verdict": _verdict(res, differential)}

    luck_df = pd.DataFrame([
        _luck_row("PIT recall-guarded (deployable, ext2026)", ssr_pit,
                  float(pit_row["total_return"]), ssr_pit.sr_full, False),
        _luck_row("Non-PIT recall-enabled (DIAGNOSTIC, ext2026)", ssr_np,
                  float(np_row["total_return"]), ssr_np.sr_full, False),
        _luck_row("Differential (non-PIT minus PIT, ext2026)", ssr_diff,
                  float(np_row["total_return"] - pit_row["total_return"]),
                  ssr_diff.sr_full, True),
    ]).set_index("line")
    luck_df.to_parquet(DATA / "factor_luck_vs_skill_ext2026.parquet")
    print("\n=== luck-vs-skill (extended span, NW-HAC/Andrews) ===")
    for line, r in luck_df.iterrows():
        print(f"- {line}: {r['verdict']}")

    # --- S11: tear-sheet re-cut over the extended span (build_tear_sheet math) --- #
    print("\n[tear sheet] re-cutting over the extended span ...")
    sys.path.insert(0, str(REPO / "scripts"))
    import build_tear_sheet as bts
    from factor_workbook.rederive import equity_metrics

    TEAR_OUT.mkdir(parents=True, exist_ok=True)
    ext_lines = {
        "factor_pit_ext2026": (equity_ext, "PIT recall-guarded factor, extended to 2026-06"),
        "factor_nonpit_diag_ext2026": (equity_np_ext, "Non-PIT DIAGNOSTIC, extended to 2026-06"),
        "baseline_ext2026": (pf_baseline.value(), "Baseline HRP+momentum, extended to 2026-06"),
        "track_b_ext2026": (pf_track_b.value(), "Track B (MC/Nash), extended to 2026-06"),
    }
    factor_returns = prices[bts.BASKET + ["SPY"]].pct_change().dropna(how="all")
    tear_rows, risk_rows = [], []
    for key, (value_raw, label) in ext_lines.items():
        value = bts._active_value(value_raw)
        r = value.pct_change().dropna()
        m = equity_metrics(value)
        ssr = compute_ssr(r)
        dd = value / value.cummax() - 1
        dd_end = dd.idxmin()
        dd_start = value.loc[:dd_end].idxmax()
        recovery = dd.loc[dd_end:][dd.loc[dd_end:] >= -1e-12]
        monthly = (1 + r).resample("ME").prod() - 1
        tear_rows.append({
            "line": key, "label": label,
            "start": r.index.min().date(), "end": r.index.max().date(), "n_days": len(r),
            "total_return": m.total_return, "cagr": m.annualized_return,
            "ann_vol": m.annualized_vol, "sharpe": m.sharpe, "sortino": m.sortino,
            "calmar": m.calmar, "max_drawdown": m.max_drawdown,
            "max_dd_peak": dd_start.date(), "max_dd_trough": dd_end.date(),
            "max_dd_recovered": recovery.index.min().date() if len(recovery) else None,
            "skew": float(r.skew()), "excess_kurtosis": float(r.kurtosis()),
            "var_95_daily": float(r.quantile(0.05)),
            "cvar_95_daily": float(r[r <= r.quantile(0.05)].mean()),
            "best_day": float(r.max()), "worst_day": float(r.min()),
            "positive_day_rate": float((r > 0).mean()),
            "best_month": float(monthly.max()), "worst_month": float(monthly.min()),
            "crisis_2022_return": m.crisis_return, "crisis_2022_max_dd": m.crisis_max_drawdown,
            "ssr": ssr.ssr, "mean_rolling_sharpe": ssr.mean_rolling_sr,
            "nw_sigma_hac": ssr.sigma_hac, "nw_bandwidth_L": ssr.L_hac,
            "ssr_verdict": ("stably > 0" if abs(ssr.ssr) >= 1.96 else
                            "NOT distinguishable from zero under HAC — luck-compatible"),
        })
        spy = factor_returns["SPY"].reindex(r.index).dropna()
        y = r.reindex(spy.index)
        coef, r2_capm, resid = bts._ols(y, spy.to_frame())
        basket = factor_returns[bts.BASKET].reindex(r.index).dropna()
        yb = r.reindex(basket.index)
        _, r2_basket, resid_b = bts._ols(yb, basket)
        risk_rows.append({
            "line": key, "label": label, "beta_spy": float(coef[1]),
            "alpha_ann_vs_spy": float(coef[0] * bts.ANNUAL), "r2_capm": r2_capm,
            "corr_spy": float(y.corr(spy)), "systematic_share_capm": r2_capm,
            "idio_vol_ann_capm": float(resid.std(ddof=1) * np.sqrt(bts.ANNUAL)),
            "r2_basket_4etf": r2_basket,
            "residual_vol_ann_basket": float(resid_b.std(ddof=1) * np.sqrt(bts.ANNUAL)),
            "note": "extended-span re-cut; conventions identical to tear_sheet.csv",
        })
        if key == "factor_pit_ext2026":
            mt = monthly.to_frame("ret")
            mt["year"], mt["month"] = mt.index.year, mt.index.month
            (mt.pivot_table(index="year", columns="month", values="ret")
               .to_csv(TEAR_OUT / f"monthly_returns_{key}.csv", float_format="%.6f"))

    tear_ext = pd.DataFrame(tear_rows)
    risk_ext = pd.DataFrame(risk_rows)
    tear_ext.to_csv(TEAR_OUT / "tear_sheet_ext2026.csv", index=False, float_format="%.8f")
    tear_ext.to_csv(TEAR_OUT / "tear_sheet_ext2026_de.csv", sep=";", decimal=",",
                    index=False, float_format="%.8f")
    risk_ext.to_csv(TEAR_OUT / "risk_decomposition_ext2026.csv", index=False, float_format="%.8f")
    risk_ext.to_csv(TEAR_OUT / "risk_decomposition_ext2026_de.csv", sep=";", decimal=",",
                    index=False, float_format="%.8f")

    # --- S12: CSV mirrors (US + _de) for every new tabular artifact -------------- #
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
        "factor_luck_vs_skill_ext2026": luck_df,
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
        _write_csv(_flatten_decision_log(json.loads((DATA / f"{name}.json").read_text())), name)
    _write_csv(pd.DataFrame({k: split[k] for k in ("in_training", "post_cutoff", "full_stream")}).T
               .rename_axis("segment"), "factor_contrast_split_ext2026")

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
        "consistency": {
            "replay_equity_max_rel_diff_vs_v1": equity_rel_diff,
            "replay_equity_tolerance": EQUITY_REL_TOL,
            "replay_targets_max_abs_diff_vs_v1": targets_max_diff,
            "in_training_premium": in_premium,
            "published_v1_premium": PUBLISHED_V1_PREMIUM,
            "in_training_reproduction_abs_diff": premium_repro_diff,
        },
        "split_table": {k: split[k] for k in ("in_training", "post_cutoff", "full_stream")},
        "prediction_outcome": split["prediction_outcome"],
        "naive_eval": {"n_directional": n_dir, "accuracy": acc,
                       "wilson_ci": [ci_lo, ci_hi], "half_inside_ci": bool(ci_lo <= 0.5 <= ci_hi)},
        "luck_vs_skill_ssr": {"pit": float(ssr_pit.ssr), "nonpit": float(ssr_np.ssr),
                              "differential": float(ssr_diff.ssr)},
        # defensive_lead_date is a date column — stringify anything non-numeric.
        "head_to_head": {name: {c: (float(v) if isinstance(v, (int, float, np.integer, np.floating))
                                    else str(v))
                                for c, v in report.loc[name].items()}
                         for name in report.index},
        "immutability": "all outputs under NEW *_ext2026 names; v1/v2 artifacts untouched (data-v2 immutable)",
    }
    (DATA / "factor_ext2026_run_header.json").write_text(
        json.dumps(header, indent=2, sort_keys=True, default=str))

    print("\n=== headline numbers ===")
    print(f"stream: {len(meta_dates)} rebalances {rebalance_dates[0].date()} -> "
          f"{rebalance_dates[-1].date()} ({len(new_meta)} new live)")
    print(f"premium in-training: mean {split['in_training']['mean_delta']:+.4f} "
          f"(d={split['in_training']['paired_d']:.2f}, n={split['in_training']['n_pairs']}) | "
          f"post-cutoff: mean {split['post_cutoff']['mean_delta']:+.4f} "
          f"(d={split['post_cutoff']['paired_d']:.2f}, n={split['post_cutoff']['n_pairs']})")
    print(f"PIT total_return {float(pit_row['total_return']):+.4f} vs non-PIT "
          f"{float(np_row['total_return']):+.4f} | differential SSR {float(ssr_diff.ssr):+.2f}")
    print("[done] all *_ext2026 artifacts written under data/ (+ csv_mirrors, tear_sheet)")


if __name__ == "__main__":
    main()
