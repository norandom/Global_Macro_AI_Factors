"""Task 3.3 live half — R8 certified no-recall model screen across NIM candidates.

Runs the certification screen (controlled number-native separation + certainty
statistics + prose-confounded positive control + parse-rate gate) for every
logprob-bearing candidate at a CONSERVATIVE COMMON CUTOFF, so the pre-cutoff
states are trained-on for every candidate and the AUCs are comparable.

Reproducible: `uv run python scripts/screen_norecall_models.py`. Reads
NVIDIA_API_KEY from .env. Additive: writes only data/norecall_screen/.
No credential is ever persisted. Candidate pool = the 2026-07-03 live probe
(logprobs confirmed; mistral-large-2 404s, gemma-4/gpt-oss-120b time out).
"""
from __future__ import annotations

import json
import sys
import os
import time
import traceback
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
load_dotenv(REPO / ".env")

import pandas as pd  # noqa: E402
from recall_guard.core.nvidia_lm import NvidiaLM  # noqa: E402

from macro_framework.anonymize import AssetMap  # noqa: E402
from macro_framework.factor_scoring import screen_candidate  # noqa: E402

CANDIDATES = [
    "meta/llama-4-maverick-17b-128e-instruct",
    "meta/llama-3.3-70b-instruct",
    "microsoft/phi-4-mini-instruct",
    "meta/llama-3.1-8b-instruct",
    "openai/gpt-oss-20b",
]
# Common conservative cutoff: every candidate's training window covers < 2023-12
# (llama-3.1/3.3 cutoff 2023-12, phi-4 ~2024-06, gpt-oss ~2024-06, maverick ~2024-08),
# so the identifying pre-cutoff states are recall-enabled for ALL of them and the
# controlled AUCs are comparable across candidates.
CUTOFF = date(2023, 12, 1)
# SCREEN_N_PER_CLASS env overrides (capped by available pre-cutoff panel rows);
# used to resolve an "inconclusive" verdict with maximum data.
N_PER_CLASS = int(os.environ.get("SCREEN_N_PER_CLASS") or 120)
PARSE_SAMPLE = 20
OUT_DIR = REPO / "data" / "norecall_screen"
PANEL = REPO / "data" / "macro_panel_monthly.parquet"


def main() -> None:
    api_key = (os.environ.get("NVIDIA_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("NVIDIA_API_KEY must be set in .env")
    panel = pd.read_parquet(PANEL)
    asset_map = AssetMap.default()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Retry support: `uv run python scripts/screen_norecall_models.py <model> [...]`
    # screens only those models and MERGES into the existing results.json
    # (replace-by-model). NIM_TIMEOUT_S env overrides the client timeout for
    # slow models (e.g. 70B: NIM_TIMEOUT_S=60; the module default is 15s).
    models = sys.argv[1:] or CANDIDATES
    timeout_s = float(os.environ.get("NIM_TIMEOUT_S") or 0) or None
    lm_factory = (
        (lambda key, m: NvidiaLM(api_key=key, model=m, timeout_s=timeout_s))
        if timeout_s
        else None
    )

    results: list[dict] = []
    prior_by_model: dict[str, dict] = {}
    out_path = OUT_DIR / "results.json"
    if out_path.exists():
        prior = json.loads(out_path.read_text())
        prior_by_model = {r.get("model"): r for r in prior.get("results", [])}
        results = [r for r in prior.get("results", []) if r.get("model") not in models]

    for i, model in enumerate(models, 1):
        print(f"[{i}/{len(models)}] screening {model} ...", flush=True)
        t0 = time.time()
        try:
            res = screen_candidate(
                nim_model=model,
                cutoff_date=CUTOFF,
                macro_panel=panel,
                asset_map=asset_map,
                api_key=api_key,
                n_per_class=N_PER_CLASS,
                parse_sample=PARSE_SAMPLE,
                lm_factory=lm_factory,
                # Raw per-prompt audit trail (R8.6 / PyXLL data contract):
                # evidence.parquet + baseline.json + summary.json per model.
                # Gitignored (MB-scale) — distributed via the GH data release.
                evidence_dir=OUT_DIR / "evidence" / model.replace("/", "_"),
            )
            row = res.to_dict()
            row["elapsed_s"] = round(time.time() - t0, 1)
            print(
                f"    verdict={res.verdict}  controlled_auc={res.controlled_auc:.3f} "
                f"CI[{res.controlled_ci_low:.3f},{res.controlled_ci_high:.3f}] "
                f"perm_p={res.controlled_perm_p:.4f}  "
                f"pc_auc={res.positive_control_auc}  pc_p={res.positive_control_perm_p}  "
                f"parse_rate={res.parse_rate}  ({row['elapsed_s']}s)",
                flush=True,
            )
        except Exception as exc:  # one failing candidate must not kill the screen
            prior_row = prior_by_model.get(model)
            if prior_row and prior_row.get("verdict") not in (None, "screen_failed"):
                # NEVER clobber prior successful evidence with a failed retry.
                row = prior_row
                print(f"    FAILED ({type(exc).__name__}) — keeping prior "
                      f"'{prior_row['verdict']}' row for {model}", flush=True)
            else:
                row = {
                    "model": model,
                    "verdict": "screen_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_s": round(time.time() - t0, 1),
                }
                print(f"    FAILED {row['error']}", flush=True)
            traceback.print_exc()
        results.append(row)
        # Incremental write so a crash preserves completed candidates.
        out_path.write_text(json.dumps({
            "screen": "R8 certified no-recall model selection (task 3.3 live half)",
            "cutoff_date": CUTOFF.isoformat(),
            "n_per_class": N_PER_CLASS,
            "parse_sample": PARSE_SAMPLE,
            "candidates": sorted({r["model"] for r in results}),
            "built_at": datetime.now(timezone.utc).isoformat(),
            "results": results,
        }, indent=2))

    print("[done] ->", out_path.relative_to(REPO), flush=True)


if __name__ == "__main__":
    main()
