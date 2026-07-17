"""CSV mirrors of the tabular release assets: Excel access without add-ins.

Converts every step-relevant tabular asset to CSV so any Excel (no Parquet
connector, no PyXLL) can load the storyboard data via Data -> From Web:

- parquet tables 1:1 (same basename, .csv extension),
- equity value series gain ``daily_return`` and ``drawdown`` columns,
- decision-log JSONs flatten to one row per rebalance date
  (p_memorized / parse_ok / steered / conviction),
- the screen results JSON flattens to one row per candidate,
- the per-model raw-evidence parquets (inside the evidence tarball) flatten to
  one CSV per model.

Outputs to ``data/csv_mirrors/`` (gitignored); upload with
``gh release upload data-v2 data/csv_mirrors/*.csv``. Derived 1:1 from the
released assets: mirrors add access, never meaning.

Reproducible: ``uv run python scripts/export_csv_mirrors.py``.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "csv_mirrors"

PLAIN_TABLES = [
    "factor_loadings_v1", "factor_loadings_v2",
    "factor_scores_v1", "factor_scores_v2",
    "factor_views_v1",
    "factor_targets_v1", "factor_targets_v2",
    "factor_nonpit_diagnostic_loadings_v1", "factor_nonpit_diagnostic_scores_v1",
    "factor_nonpit_diagnostic_targets_v1",
    "factor_contrast_v1", "factor_luck_vs_skill_v1",
    "naive_directional_eval_openai_gpt-oss-20b",
    "macro_panel_monthly",
]
EQUITY_TABLES = [
    "factor_equity_v1", "factor_equity_v2", "factor_nonpit_diagnostic_equity_v1",
]
DECISION_LOGS = [
    "factor_decision_log_v1", "factor_decision_log_v2",
    "factor_nonpit_diagnostic_decision_log_v1",
]


def flatten_decision_log(payload: dict) -> pd.DataFrame:
    per_date = {k: payload[k] for k in ("p_memorized", "parse_ok", "steered", "conviction")}
    df = pd.DataFrame(per_date)
    df.index.name = "date"
    return df.sort_index()


def _write(df: pd.DataFrame, name: str, index: bool = True) -> None:
    """Write both locale variants: US (dot decimals) and _de (semicolon+comma)."""
    df.to_csv(OUT / f"{name}.csv", float_format="%.8f", index=index)
    df.to_csv(OUT / f"{name}_de.csv", sep=";", decimal=",", float_format="%.8f", index=index)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0

    for name in PLAIN_TABLES:
        _write(pd.read_parquet(REPO / "data" / f"{name}.parquet"), name)
        n += 1

    for name in EQUITY_TABLES:
        eq = pd.read_parquet(REPO / "data" / f"{name}.parquet")
        eq["daily_return"] = eq["value"].pct_change()
        eq["drawdown"] = eq["value"] / eq["value"].cummax() - 1
        _write(eq, name)
        n += 1

    for name in DECISION_LOGS:
        payload = json.loads((REPO / "data" / f"{name}.json").read_text())
        _write(flatten_decision_log(payload), name)
        n += 1

    screen = json.loads((REPO / "data" / "norecall_screen" / "results.json").read_text())
    _write(pd.DataFrame(screen["results"]), "norecall_screen_results", index=False)
    n += 1

    evidence_dir = REPO / "data" / "norecall_screen" / "evidence"
    for model_dir in sorted(evidence_dir.iterdir()) if evidence_dir.exists() else []:
        parquet = model_dir / "evidence.parquet"
        if parquet.exists():
            _write(pd.read_parquet(parquet), f"norecall_evidence_{model_dir.name}", index=False)
            n += 1

    print(f"[done] {n} CSV mirrors -> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
