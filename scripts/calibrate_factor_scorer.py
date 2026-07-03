"""Task 3.2 runner — one-time NUMBER-NATIVE calibration of the FactorScorer + persist.

Builds the number-native calibrator on the factor task from the FRED macro panel
(identifying IS vs anonymized OOS), records held-out AUC / is_weak, smoke-scores an
anonymized factor prompt, and persists the calibrator (joblib + JSON, no credential)
so nb13/nb14 reuse it without the ~135-call rebuild.

Reproducible: `uv run python scripts/calibrate_factor_scorer.py`. Reads NVIDIA_API_KEY
from .env. Additive — writes only a new directory under data/. No FMP, no news.

Model + cutoff (research.md 2026-06-26): meta/llama-4-maverick-17b-128e-instruct @
2024-08-01 — logprob-bearing, validated number-native holdout_auc ~0.96.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
load_dotenv(REPO / ".env")

import pandas as pd  # noqa: E402

from macro_framework.anonymize import AssetMap  # noqa: E402
from macro_framework.factor_scoring import (  # noqa: E402
    FactorScorer,
    render_regime_loadings_prompt,
)

# Defaults = the original task-3.2 run (maverick @ 2024-08-01). The R8
# selection re-runs this for the CERTIFIED model: pass `<model> <cutoff>` as
# argv; NIM_TIMEOUT_S / CAL_N_PER_CLASS envs override the client timeout and
# corpus size (slow-serving models need >15 s).
NIM_MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta/llama-4-maverick-17b-128e-instruct"
CUTOFF = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2024, 8, 1)
N_PER_CLASS = int(os.environ.get("CAL_N_PER_CLASS") or 60)
MIN_AUC = 0.6
TIMEOUT_S = float(os.environ.get("NIM_TIMEOUT_S") or 0) or None
SLUG = NIM_MODEL.replace("/", "_")
OUT_DIR = REPO / "data" / f"factor_calibrator_{SLUG}"
PANEL = REPO / "data" / "macro_panel_monthly.parquet"


def main() -> None:
    api_key = (os.environ.get("NVIDIA_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("NVIDIA_API_KEY must be set in .env")

    panel = pd.read_parquet(PANEL)
    print(f"[1/3] Calibrating {NIM_MODEL} @ cutoff {CUTOFF} "
          f"(number-native: identifying IS vs anonymized OOS, {N_PER_CLASS}/class) ...")
    lm_factory = None
    if TIMEOUT_S:
        from recall_guard.core.nvidia_lm import NvidiaLM

        def lm_factory(key: str, model: str) -> NvidiaLM:  # noqa: E731
            return NvidiaLM(api_key=key, model=model, timeout_s=TIMEOUT_S)

    scorer = FactorScorer.calibrate(
        nim_model=NIM_MODEL,
        cutoff_date=CUTOFF,
        macro_panel=panel,
        asset_map=AssetMap.default(),
        api_key=api_key,
        n_per_class=N_PER_CLASS,
        min_auc=MIN_AUC,
        lm_factory=lm_factory,
    )
    print(f"      holdout_auc={scorer.holdout_auc:.4f}  is_weak={scorer.is_weak}")

    print("[2/3] Persisting calibrator (no credential) ...")
    scorer.save(OUT_DIR)
    print(f"      saved -> {OUT_DIR.relative_to(REPO)}")

    print("[3/3] Smoke: score an anonymized factor prompt ...")
    macro_state = {"cpi_yoy_z": 0.55, "t10y2y_z": -1.59, "hy_oas_z": 0.65}
    asset_snapshot = [
        {"id": pid, "category": cat}
        for pid, cat in sorted(AssetMap.default().categories.items())
    ]
    prompt = render_regime_loadings_prompt(macro_state, asset_snapshot)
    fs = scorer.score(prompt)
    print(f"      parse_ok={fs.parse_ok} p_memorized={fs.p_memorized} fail_reason={fs.fail_reason}")

    header = {
        "nim_model": NIM_MODEL,
        "cutoff_date": CUTOFF.isoformat(),
        "calibration": "number-native (identifying IS vs anonymized OOS on the factor task; no news/FMP)",
        "holdout_auc": float(scorer.holdout_auc),
        "is_weak": bool(scorer.is_weak),
        "n_per_class": N_PER_CLASS,
        "smoke_anon_p_memorized": fs.p_memorized,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUT_DIR / "run_header.json").write_text(json.dumps(header, indent=2))
    print(f"[done] header -> {(OUT_DIR / 'run_header.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
