"""Sample the recall guard's ``p_memorized`` N times for ONE rebalance prompt.

The companion to ``run_factor_dispersion_study.py``. That script measures how
stable the model's regime *reading* is; this one measures how stable the guard's
*verdict on that reading* is, because the guard is scored by its own model call
and therefore has a distribution of its own.

``p_memorized`` decides how much of the raw tilt survives into the portfolio:
``guarded_tilt = raw_tilt * (1 - p_memorized)``. A guard that swung wildly would
make the deployed exposure unpredictable even when the regime read is stable.

Usage::

    NVIDIA_API_KEY=... uv run python scripts/run_factor_guard_dispersion.py \
        --draws 60 --workers 12
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

os.environ.setdefault(
    "MARKET_SNAPSHOT_DIR",
    str(
        REPO
        / "data/provisional_remediation/market_snapshots"
        / "provisional_market_total_return_fx_2026-06-30_v1"
    ),
)

from macro_framework import factor_scoring as fs  # noqa: E402
from recall_guard import NvidiaLM  # noqa: E402
from scripts import extend_stream_2026 as ext  # noqa: E402
from scripts.run_factor_dispersion_study import DEFAULT_REBALANCE, build_prompt  # noqa: E402

FIELDS = ("draw", "rebalance_date", "parse_ok", "p_memorized", "fail_reason", "latency_s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=60)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--rebalance-date", default=DEFAULT_REBALANCE)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--out", default="data/appendix_i_factor_dispersion/factor_dispersion_guard.csv"
    )
    args = parser.parse_args()

    load_dotenv(REPO / ".env")
    api_key = (os.environ.get("NVIDIA_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is required for the live guard sample")

    rebalance = pd.Timestamp(args.rebalance_date)
    prompt, macro_date, _macro_state = build_prompt(rebalance)
    out_path = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def lm_factory(key: str, model: str) -> NvidiaLM:
        return NvidiaLM(api_key=key, model=model, timeout_s=args.timeout_s, max_retries=1)

    calibrator_dir = REPO / "data" / f"factor_calibrator_{ext.SLUG}"

    def load_scorer() -> fs.FactorScorer:
        return fs.FactorScorer.load(calibrator_dir, api_key=api_key, lm_factory=lm_factory)

    scorer = load_scorer()
    print(f"calibrator holdout_auc={scorer.holdout_auc:.4f} is_weak={scorer.is_weak}")
    if scorer.is_weak:
        raise RuntimeError("calibrator is weak — the guard would pass everything through (R4.3)")
    print(f"rebalance {rebalance.date()} | macro source {macro_date.date()} | draws {args.draws}")

    # A FactorScorer owns ONE NvidiaLM, and that client serialises every request
    # through a per-instance pacing lock, so score_many() on a single scorer runs
    # effectively sequentially regardless of max_workers. Give each worker thread
    # its own scorer (hence its own client) to score the repeats concurrently.
    local = threading.local()

    def thread_scorer() -> fs.FactorScorer:
        existing = getattr(local, "scorer", None)
        if existing is None:
            existing = load_scorer()
            local.scorer = existing
        return existing

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        scores = list(pool.map(lambda _: thread_scorer().score(prompt), range(args.draws)))
    elapsed = time.monotonic() - started

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index, score in enumerate(scores):
            writer.writerow({
                "draw": index,
                "rebalance_date": rebalance.date().isoformat(),
                "parse_ok": bool(score.parse_ok),
                "p_memorized": "" if score.p_memorized is None else float(score.p_memorized),
                "fail_reason": score.fail_reason or "",
                "latency_s": "",
            })

    values = [float(s.p_memorized) for s in scores if s.p_memorized is not None]
    print(f"scored {len(values)}/{args.draws} in {elapsed / 60:.1f} min")
    if values:
        series = pd.Series(values)
        print(f"p_memorized mean {series.mean():.4f} sd {series.std(ddof=1):.4f} "
              f"min {series.min():.4f} max {series.max():.4f}")
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
