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

``--ensemble`` switches from the raw dump to ``recall_guard.generate_ensemble``
(0.3.0+), which performs the same N draws and additionally returns the reduced
answer: a robust per-axis location, decision agreement with a Wilson interval,
any axis whose draws form separated clusters, and a hash over the draw set. The
raw dump remains the default because it is what the Appendix I evidence CSV was
built from and re-running it must keep reproducing that file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Mapping
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

#: The grid the model emits loadings on. Drives the scale floor and the
#: separated-cluster test, both of which are meaningless without it.
LOADING_GRID = 0.1

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


#: Categories whose exposure tilt the factor line treats as risk-seeking and as
#: protective. The decision an ensemble is asked to agree on is the sign of
#: (defensive tilt - risk tilt), i.e. the posture, not the loading values.
RISK_CATEGORIES = ("world_equity", "tech_sector")
DEFENSIVE_CATEGORIES = ("gold_commodity", "short_treasury_cash")


def defensive_spread(loadings: Mapping[str, float]) -> float:
    """(gold + cash) - (equity + tech) exposure tilt for one loading vector.

    An axis absent from ``loadings`` contributes nothing. That is the case when
    the ensemble refuses to reduce a separated component to one location: the
    honest response is to stop tilting on that axis rather than to substitute a
    centre the draws do not support, so the refusal propagates into a smaller
    tilt instead of a fabricated one.
    """
    total = 0.0
    for category in DEFENSIVE_CATEGORIES:
        exposure = fs.REGIME_ASSET_EXPOSURE[category]
        total += sum(loadings[axis] * exposure[axis] for axis in fs.MACRO_AXES if axis in loadings)
    for category in RISK_CATEGORIES:
        exposure = fs.REGIME_ASSET_EXPOSURE[category]
        total -= sum(loadings[axis] * exposure[axis] for axis in fs.MACRO_AXES if axis in loadings)
    return total


def run_ensemble(
    lm: NvidiaLM,
    prompt: str,
    rebalance: pd.Timestamp,
    *,
    draws: int,
    workers: int,
    temperature: float,
) -> dict[str, object]:
    """Reduce N draws through the library's consensus contract.

    ``decide`` returns the posture, so agreement is measured on the decision the
    portfolio acts on rather than on the loading values (which almost never
    repeat). ``components`` exposes the five axes so location and the
    separated-cluster test run per axis. A reply that does not parse raises out
    of ``decide`` and is counted as a projection failure rather than being
    silently treated as a vote.
    """
    from recall_guard import EnsembleSpec, generate_ensemble

    def parse(result) -> Mapping[str, float]:
        loadings = fs.parse_loadings(result.content, rebalance)
        if loadings is None:
            raise ValueError("reply did not yield a full five-axis vector")
        return {axis: float(loadings.loadings[axis]) for axis in fs.MACRO_AXES}

    # recall-guard 0.4.0 forwards these to every draw. Before it, the ensemble
    # sampled at the client default of 512 tokens while production generated at
    # 2048, which truncated this reasoning model's reply before the JSON and cut
    # the parse rate from 95% to 48%.
    spec = EnsembleSpec(
        draws=draws,
        max_workers=workers,
        min_parsed=max(8, draws // 4),
        grid=LOADING_GRID,
        retain_draws=True,
        max_tokens=PRODUCTION_MAX_TOKENS,
        temperature=temperature,
    )
    result = generate_ensemble(
        lm,
        prompt,
        spec,
        decide=lambda r: "defensive" if defensive_spread(parse(r)) > 0 else "risk_on",
        components=parse,
    )
    return {
        "consensus_reply": result.consensus.content,
        "location": dict(result.location),
        "location_snapped": dict(result.location_snapped or {}),
        "consensus_spread": defensive_spread(result.location),
        "abstained_axes": [a for a in fs.MACRO_AXES if a not in result.location],
        "agreement": result.agreement,
        "agreement_ci": list(result.agreement_ci or ()),
        "multimodal_axes": list(result.multimodal),
        "n_requested": result.n_requested,
        "n_parsed": result.n_parsed,
        "fail_counts": dict(result.fail_counts),
        "draws_sha256": result.draws_sha256,
        "max_tokens": result.max_tokens,
        "temperature": result.temperature,
        "component_verdicts": {
            axis: (None if verdict is None else {
                "separated": bool(verdict.separated),
                "lower_mass": verdict.lower_mass,
                "upper_mass": verdict.upper_mass,
                "trough_mass": verdict.trough_mass,
                "gap": list(verdict.gap) if verdict.gap else None,
            })
            for axis, verdict in result.component_verdicts
        },
    }


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
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="reduce the draws through recall_guard.generate_ensemble and write "
             "a consensus JSON instead of the raw per-draw CSV",
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

    print(f"rebalance {rebalance.date()} | macro source {macro_date.date()} "
          f"{ {k: round(v, 3) for k, v in macro_state.items()} }")
    print(f"model {ext.NIM_MODEL} | T={args.temperature} | max_tokens={PRODUCTION_MAX_TOKENS}")

    # One shared client is correct from recall-guard 0.3.0 on: the pacing lock
    # now covers only the send-slot bookkeeping, not the HTTP round trip, so
    # concurrent calls through one client actually run concurrently. (Under
    # 0.2.0 the lock spanned the POST and any worker count behaved like one,
    # which is why this file used to build a client per thread.)
    lm = NvidiaLM(
        api_key=api_key,
        model=ext.NIM_MODEL,
        timeout_s=args.timeout_s,
        max_retries=1,
    )

    if args.ensemble:
        consensus_path = out_path.with_name(
            out_path.stem.replace("_draws", "") + "_consensus.json"
        )
        print(f"ensemble mode | draws {args.draws} | workers {args.workers}")
        started = time.monotonic()
        summary = run_ensemble(
            lm, prompt, rebalance, draws=args.draws, workers=args.workers,
            temperature=args.temperature,
        )
        summary["rebalance_date"] = rebalance.date().isoformat()
        summary["macro_source_date"] = macro_date.date().isoformat()
        summary["model"] = ext.NIM_MODEL
        summary["temperature"] = args.temperature
        summary["elapsed_s"] = round(time.monotonic() - started, 1)
        consensus_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"  parsed {summary['n_parsed']}/{summary['n_requested']} | "
              f"agreement {summary['agreement']:.4f} {tuple(summary['agreement_ci'])}")
        print(f"  location {summary['location_snapped'] or summary['location']}")
        print(f"  consensus spread {summary['consensus_spread']:+.3f} | "
              f"multimodal axes {summary['multimodal_axes'] or 'none'} | "
              f"abstained {summary['abstained_axes'] or 'none'}")
        print(f"  draws_sha256 {summary['draws_sha256']}")
        print(f"[done] {consensus_path}")
        return

    done = completed_draws(out_path)
    todo = [i for i in range(args.draws) if i not in done]
    print(f"draws requested {args.draws} | already stored {len(done)} | to run {len(todo)}")
    if not todo:
        print("nothing to do")
        return

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
            result = lm.generate(
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
