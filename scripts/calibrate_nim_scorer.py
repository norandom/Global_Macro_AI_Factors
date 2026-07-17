"""Task 3.2 runner — one-time live calibration of the NIM scorer + directional smoke.

Builds the dated FMP IS/OOS corpora, calibrates a recall_guard MemoryGuardedScorer
for the chosen NIM model, records held-out AUC / is_weak, smoke-tests that the
directional prompt parses, and persists a calibration header under a new filename.

Reproducible: `uv run python scripts/calibrate_nim_scorer.py`. Reads keys from .env
(NVIDIA_API_KEY, FMP_API_KEY). Additive: writes only new files under data/.

Model + cutoff (see .kiro/specs/track-a-macro-steering/research.md 2026-06-26):
  meta/llama-4-maverick-17b-128e-instruct, cutoff 2024-08-01; logprobs-capable,
  emits a clean Direction/Confidence answer, and its ~2024-08 cutoff puts both the
  IS (pre-cutoff) and OOS (post-cutoff) corpora inside FMP's dense news window.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # make the local macro_framework package importable
load_dotenv(REPO / ".env")

from recall_guard import MemoryGuardedScorer  # noqa: E402
from recall_guard.dataset.fmp_corpora import build_calibration  # noqa: E402

from macro_framework.anonymize import AssetMap  # noqa: E402
from macro_framework.steering import (  # noqa: E402
    _read_corpus_jsonl,
    render_directional,
)

NIM_MODEL = "meta/llama-4-maverick-17b-128e-instruct"
CUTOFF = date(2024, 8, 1)
TARGET_PER_CORPUS = 100
MIN_AUC = 0.6
OUT_DIR = REPO / "data" / "calibration"
SLUG = NIM_MODEL.replace("/", "_")
HEADER_PATH = REPO / "data" / f"track_a_scores_{SLUG}.json"


def _count(path: Path) -> int:
    return sum(1 for ln in path.read_text().splitlines() if ln.strip())


def main() -> None:
    nvidia = (os.environ.get("NVIDIA_API_KEY") or "").strip()
    fmp = (os.environ.get("FMP_API_KEY") or "").strip()
    if not nvidia or not fmp:
        raise SystemExit("NVIDIA_API_KEY and FMP_API_KEY must be set in .env")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    is_path = OUT_DIR / "is_memorized.jsonl"
    oos_path = OUT_DIR / "oos_control.jsonl"
    if is_path.exists() and oos_path.exists() and _count(is_path) and _count(oos_path):
        print(f"[1/4] Reusing existing FMP corpora in {OUT_DIR.relative_to(REPO)} (skip fetch) ...")
    else:
        print(f"[1/4] Building FMP corpora (cutoff={CUTOFF}, target={TARGET_PER_CORPUS}/side) ...")
        is_path, oos_path = build_calibration(
            OUT_DIR,
            cutoffs={NIM_MODEL: CUTOFF},
            target_per_corpus=TARGET_PER_CORPUS,
            api_key=fmp,
            today=datetime.now(timezone.utc).date(),
        )
    n_is, n_oos = _count(is_path), _count(oos_path)
    print(f"      IS={n_is}  OOS={n_oos}  ({is_path.name}, {oos_path.name})")

    # Dogfood the 2.2 module-level read-back helper, then the real calibrate
    # (avoids a second FMP fetch that calibrate_from_fmp would do).
    is_mem = _read_corpus_jsonl(is_path)
    oos = _read_corpus_jsonl(oos_path)

    print(f"[2/4] Calibrating {NIM_MODEL} over IS={len(is_mem)} / OOS={len(oos)} (NIM calls) ...")
    try:
        scorer = MemoryGuardedScorer.calibrate(
            api_key=nvidia,
            model=NIM_MODEL,
            is_memorized=is_mem,
            oos_control=oos,
            reference_model=None,
            min_auc=MIN_AUC,
        )
    except Exception as exc:  # noqa: BLE001  -- surface the precise failure
        print(f"      CALIBRATION FAILED: {type(exc).__name__}: {exc}")
        HEADER_PATH.write_text(json.dumps({
            "nim_model": NIM_MODEL, "cutoff_date": CUTOFF.isoformat(),
            "calibrated": False, "error": f"{type(exc).__name__}: {exc}",
            "n_is": n_is, "n_oos": n_oos,
            "built_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
        print(f"      header -> {HEADER_PATH.relative_to(REPO)}")
        raise

    print(f"      holdout_auc={scorer.holdout_auc:.4f}  is_weak={scorer.is_weak}")

    # [3/4] Directional smoke: render a PIT macro prompt and score it.
    print("[3/4] Directional smoke test ...")
    macro_state = {"cpi_yoy_z": 0.55, "t10y2y_z": -1.59, "hy_oas_z": 0.65}
    asset_snapshot = [
        {"id": pid, "category": cat, "trailing_12m_return": 0.05, "trailing_vol_ann": 0.15}
        for pid, cat in sorted(AssetMap.default().categories.items())
    ]
    prompt = render_directional(macro_state, asset_snapshot)
    g = scorer.score(prompt)
    print(f"      parse_ok={g.parse_ok} signal={g.signal} "
          f"p_memorized={g.p_memorized} fail_reason={g.fail_reason}")

    # [4/4] Persist calibration header.
    header = {
        "nim_model": NIM_MODEL,
        "cutoff_date": CUTOFF.isoformat(),
        "calibrated": True,
        "holdout_auc": float(scorer.holdout_auc),
        "is_weak": bool(scorer.is_weak),
        "min_auc": MIN_AUC,
        "reference_model": None,
        "n_is": n_is,
        "n_oos": n_oos,
        "corpora": {"is": str(is_path.relative_to(REPO)), "oos": str(oos_path.relative_to(REPO))},
        "smoke": {
            "macro_state": macro_state,
            "parse_ok": g.parse_ok,
            "signal": g.signal,
            "p_memorized": g.p_memorized,
            "fail_reason": g.fail_reason,
        },
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    HEADER_PATH.write_text(json.dumps(header, indent=2))
    print(f"[done] header -> {HEADER_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
