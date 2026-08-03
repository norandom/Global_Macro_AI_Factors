"""Sample the PIT factor decision N times at ONE critical rebalance date.

Appendix I asks a narrow empirical question: at a crisis onset, is the macro
factor's answer a stable, economically coherent reading of the regime, or is it
noise that happened to land well once?  The only way to answer it is to re-run
the SAME anonymized PIT prompt many times and look at the spread.

This script performs the live generation ONCE and streams every draw to CSV.
The notebook never calls a provider: it reads the CSV.  Nothing here touches a
completed run bundle, and the canonical artifacts are read-only inputs.

Each draw records the raw reply text, the parsed five-axis loading vector, the
sampling temperature, and the latency, so the stored evidence can be re-parsed
or audited later without re-querying the model.

Usage::

    NVIDIA_API_KEY=... uv run python scripts/run_factor_dispersion_study.py \
        --draws 1000 --workers 24

Resumable: an existing output CSV is read first and only the missing draws run.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

#: The market snapshot supplying the Factor run's prices; set before importing
#: the stream module so its snapshot resolution does not fall back to a
#: non-existent release directory.
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

#: COVID-crash onset. The 2020-03-02 rebalance is the first decision after the
#: 2020-02-19 equity peak and sits at the start of the Factor line's deepest
#: drawdown episode (peak 2020-02-21, trough 2020-03-19): a date where getting
#: the defensive call right is what protects the drawdown.
DEFAULT_REBALANCE = "2020-03-02"

#: Production generation settings (scripts/extend_stream_2026.py::_generate_big).
PRODUCTION_TEMPERATURE = 0.0
PRODUCTION_MAX_TOKENS = 2048

FIELDS = (
    "draw",
    "rebalance_date",
    "macro_source_date",
    "temperature",
    "max_tokens",
    "parse_ok",
    "inflation",
    "growth",
    "credit_stress",
    "policy",
    "risk_appetite",
    "latency_s",
    "error",
    "reply_text",
)


def build_prompt(rebalance_date: pd.Timestamp) -> tuple[str, pd.Timestamp, dict[str, float]]:
    """The exact anonymized PIT prompt the production stream renders for this date.

    Uses the committed macro panel and the same point-in-time selection rule as
    the walk-forward run: the latest COMPLETED month strictly before the
    rebalance date. The date itself never enters the prompt (R6.1/R6.2).
    """
    panel, _source = ext.build_panel(force_committed_fallback=True)
    available = panel.dropna(subset=ext.PANEL_Z_COLS)
    prior = available[available.index < rebalance_date]
    if prior.empty:
        raise ValueError(f"no completed macro month precedes {rebalance_date.date()}")
    row = prior.iloc[-1]
    macro_state = row[ext.PANEL_Z_COLS].to_dict()
    asset_map = ext.mf.AssetMap.default()
    snapshot = [
        {"id": pseudo, "category": category}
        for pseudo, category in sorted(asset_map.categories.items())
    ]
    prompt = fs.render_regime_loadings_prompt(macro_state, snapshot)
    return prompt, row.name, {k: float(v) for k, v in macro_state.items()}


def completed_draws(path: Path) -> set[int]:
    """Draw indices already answered, after dropping transport failures.

    A draw whose reply did not parse is a real outcome and is kept. A draw that
    errored (timeout, HTTP failure) carries no model answer, so it is removed
    from the file and re-run on the next pass rather than silently counting as
    a completed observation.
    """
    if not path.is_file():
        return set()
    frame = pd.read_csv(path)
    if frame.empty:
        return set()
    answered = frame[frame["error"].isna() | (frame["error"].astype(str).str.strip() == "")]
    answered = answered.drop_duplicates(subset="draw", keep="first")
    if len(answered) != len(frame):
        answered.to_csv(path, index=False)
    return set(answered["draw"].astype(int).tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=PRODUCTION_TEMPERATURE)
    parser.add_argument("--rebalance-date", default=DEFAULT_REBALANCE)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument(
        "--out",
        default="data/appendix_i_factor_dispersion/factor_dispersion_draws.csv",
    )
    args = parser.parse_args()

    load_dotenv(REPO / ".env")
    api_key = (os.environ.get("NVIDIA_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is required for the live dispersion study")

    rebalance = pd.Timestamp(args.rebalance_date)
    prompt, macro_date, macro_state = build_prompt(rebalance)
    out_path = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = completed_draws(out_path)
    todo = [i for i in range(args.draws) if i not in done]
    print(f"rebalance {rebalance.date()} | macro source {macro_date.date()} "
          f"{ {k: round(v, 3) for k, v in macro_state.items()} }")
    print(f"model {ext.NIM_MODEL} | T={args.temperature} | max_tokens={PRODUCTION_MAX_TOKENS}")
    print(f"draws requested {args.draws} | already stored {len(done)} | to run {len(todo)}")
    if not todo:
        print("nothing to do")
        return

    # One client PER THREAD. NvidiaLM serialises every request through a
    # per-instance pacing lock (recall_guard.core.nvidia_lm holds it across the
    # whole HTTP POST), so a single shared client would make any worker count
    # behave like one. Independent clients give real concurrency.
    local = threading.local()

    def client() -> NvidiaLM:
        existing = getattr(local, "lm", None)
        if existing is None:
            existing = NvidiaLM(
                api_key=api_key,
                model=ext.NIM_MODEL,
                timeout_s=args.timeout_s,
                max_retries=1,
            )
            local.lm = existing
        return existing

    write_lock = threading.Lock()
    new_file = not out_path.is_file()
    handle = out_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()
        handle.flush()

    started = time.monotonic()
    counter = {"done": 0, "ok": 0}

    def one_draw(index: int) -> dict[str, object]:
        call_start = time.monotonic()
        record: dict[str, object] = {
            "draw": index,
            "rebalance_date": rebalance.date().isoformat(),
            "macro_source_date": macro_date.date().isoformat(),
            "temperature": args.temperature,
            "max_tokens": PRODUCTION_MAX_TOKENS,
            "parse_ok": False,
            "error": "",
            "reply_text": "",
        }
        for axis in fs.MACRO_AXES:
            record[axis] = ""
        try:
            result = client().generate(
                prompt,
                temperature=args.temperature,
                max_tokens=PRODUCTION_MAX_TOKENS,
            )
            record["reply_text"] = result.content
            loadings = fs.parse_loadings(result.content, rebalance)
            if loadings is not None:
                record["parse_ok"] = True
                for axis in fs.MACRO_AXES:
                    record[axis] = float(loadings.loadings[axis])
        except Exception as exc:  # noqa: BLE001 — a failed draw is data, not a crash
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["latency_s"] = round(time.monotonic() - call_start, 2)
        return record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one_draw, index) for index in todo]
        for future in as_completed(futures):
            record = future.result()
            with write_lock:
                writer.writerow(record)
                handle.flush()
                counter["done"] += 1
                counter["ok"] += int(bool(record["parse_ok"]))
                if counter["done"] % 25 == 0 or counter["done"] == len(todo):
                    elapsed = time.monotonic() - started
                    rate = counter["done"] / elapsed * 60.0
                    remaining = (len(todo) - counter["done"]) / max(rate, 1e-9)
                    print(
                        f"  {counter['done']}/{len(todo)} draws "
                        f"({counter['ok']} parsed) | {rate:.1f}/min | "
                        f"~{remaining:.0f} min left",
                        flush=True,
                    )
    handle.close()
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
