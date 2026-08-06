"""Paired test: do today's loadings differ systematically from the recorded v1-era ones?

The 2016-2026 factor line's signal was generated at three different times. This asks
whether the oldest segment (2019-2024, sourced from data/factor_loadings_v1.parquet)
still matches what the model says now, with the prompt held fixed.

Per-date design: the recorded side is ONE draw (production's convention); today's side
is the median of a small ensemble, which cuts this side's sampling noise without
pretending the recorded side had any. Per-draw noise inflates the paired variance but
does not bias it, so a signed-rank test over many dates still detects a systematic
shift.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats as st

REPO = Path("/home/mc/projects/Global_Macro_AI_Factors")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault(
    "MARKET_SNAPSHOT_DIR",
    str(REPO / "data/provisional_remediation/market_snapshots/provisional_market_total_return_fx_2026-06-30_v1"),
)
load_dotenv(REPO / ".env")

from macro_framework import factor_scoring as fs  # noqa: E402
from recall_guard import NvidiaLM  # noqa: E402
from scripts import extend_stream_2026 as ext  # noqa: E402
from scripts.run_factor_dispersion_study import (  # noqa: E402
    PRODUCTION_MAX_TOKENS, defensive_spread,
)

RUN = REPO / "data/provisional_remediation/factor_runs/factor_ext2026_2016-01-01_2026-06-30_v1"
AX = list(fs.MACRO_AXES)
N_DATES = 30
DRAWS_PER_DATE = 5

# recorded loadings for the v1-sourced segment
evidence = pd.read_parquet(RUN / "factor_evidence_ext2026.parquet")
v1 = evidence[
    (evidence.variant == "pit")
    & (evidence.segment == "replayed_v1")
    & (evidence.source_artifact == "data/factor_loadings_v1.parquet")
].copy()
v1["rebalance_date"] = pd.to_datetime(v1["rebalance_date"])
v1 = v1[v1.loadings_parse_ok].sort_values("rebalance_date")
rng = np.random.default_rng(0)
picked = v1.iloc[sorted(rng.choice(len(v1), size=min(N_DATES, len(v1)), replace=False))]
print(f"v1-sourced PIT dates available: {len(v1)} | sampling {len(picked)}")
print(f"span {picked.rebalance_date.min().date()} -> {picked.rebalance_date.max().date()}")

# rebuild each prompt exactly as the stream does
panel, _ = ext.build_panel(force_committed_fallback=True)
avail = panel.dropna(subset=ext.PANEL_Z_COLS)
asset_map = ext.mf.AssetMap.default()
snapshot = [{"id": p, "category": c} for p, c in sorted(asset_map.categories.items())]

jobs = []
for _, row in picked.iterrows():
    rb = pd.Timestamp(row["rebalance_date"])
    prior = avail[avail.index < rb]
    state = prior.iloc[-1][ext.PANEL_Z_COLS].to_dict()
    prompt = fs.render_regime_loadings_prompt(state, snapshot)
    # the stored prompt must match, or we are not asking the same question
    assert prompt == row["pit_prompt_text"], f"prompt drift at {rb.date()}"
    jobs.append((rb, prompt, {a: float(row[f"loading_{a}"]) for a in AX}))
print(f"all {len(jobs)} prompts reproduce the stored text byte-for-byte")

lm = NvidiaLM(api_key=os.environ["NVIDIA_API_KEY"].strip(), model=ext.NIM_MODEL,
              timeout_s=300, max_retries=1)


def one(args):
    rb, prompt = args
    try:
        r = lm.generate(prompt, temperature=0.0, max_tokens=PRODUCTION_MAX_TOKENS)
        l = fs.parse_loadings(r.content, rb)
        return None if l is None else {a: float(l.loadings[a]) for a in AX}
    except Exception:
        return None


t0 = time.monotonic()
tasks = [(rb, prompt) for rb, prompt, _ in jobs for _ in range(DRAWS_PER_DATE)]
with ThreadPoolExecutor(max_workers=25) as pool:
    out = list(pool.map(one, tasks))
print(f"{len(tasks)} draws in {(time.monotonic()-t0)/60:.1f} min "
      f"({sum(o is not None for o in out)} parsed)")

rows = []
for i, (rb, _prompt, recorded) in enumerate(jobs):
    got = [o for o in out[i*DRAWS_PER_DATE:(i+1)*DRAWS_PER_DATE] if o is not None]
    if len(got) < 3:
        continue
    today = {a: float(np.median([g[a] for g in got])) for a in AX}
    rows.append({"date": rb.date(), "n": len(got),
                 **{f"rec_{a}": recorded[a] for a in AX},
                 **{f"now_{a}": today[a] for a in AX},
                 "rec_spread": defensive_spread(recorded),
                 "now_spread": defensive_spread(today)})
df = pd.DataFrame(rows)
print(f"\nusable dates: {len(df)}\n")
print("per-axis paired comparison (today's ensemble median vs recorded v1 draw):")
print(f"{'axis':14s} {'recorded':>9s} {'today':>8s} {'delta':>8s} {'Wilcoxon p':>11s}")
for a in AX:
    d = df[f"now_{a}"] - df[f"rec_{a}"]
    p = st.wilcoxon(d).pvalue if d.abs().sum() > 0 else 1.0
    print(f"{a:14s} {df[f'rec_{a}'].mean():+9.3f} {df[f'now_{a}'].mean():+8.3f} "
          f"{d.mean():+8.3f} {p:11.2e}")
ds = df["now_spread"] - df["rec_spread"]
print(f"\ndefensive spread: recorded {df.rec_spread.mean():+.2f} -> today {df.now_spread.mean():+.2f} "
      f"(delta {ds.mean():+.2f}, Wilcoxon p={st.wilcoxon(ds).pvalue:.2e})")
print(f"sign of posture unchanged on {(np.sign(df.now_spread)==np.sign(df.rec_spread)).mean():.0%} of dates")
out = REPO / "data/appendix_i_factor_dispersion/seam_test_v1_vs_today.csv"
df.to_csv(out, index=False)
print("written:", out)
