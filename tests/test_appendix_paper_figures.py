"""Static contracts for the presentation-only Appendix B and Appendix G notebooks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", ())) for cell in notebook["cells"])


def test_appendix_b_uses_verified_paired_recall_and_separate_pit_coinflip_check() -> None:
    source = _source(ROOT / "notebooks" / "appendix_b_directional_coinflip.ipynb")

    assert "factor_ext2026_2019-01-01_2026-06-30_v1" in source
    assert "factor_run.v1" in source
    assert "manifest_sha256" in source
    assert "COMPLETED" in source
    assert "factor_contrast_ext2026.parquet" in source
    assert "Identifying prompt: date and assets shown" in source
    assert "PIT prompt: date and assets hidden" in source
    assert "Both streams retain the recall guard" in source
    assert "not guard on versus guard off" in source
    assert "not a matched PIT-versus-non-PIT prediction comparison" in source
    assert "valid up/down calls" in source
    assert "Appendix E" in source
    assert "Guard-disabled portfolio experiment" in source
    assert "before (no guard)" not in source
    assert "unguarded non-PIT" not in source.lower()
    assert "guard does not change directional accuracy" not in source


def test_appendix_g_exports_manifest_pinned_paper_figures_only() -> None:
    source = _source(ROOT / "notebooks" / "appendix_g_final_trio_paper_figures.ipynb")

    required = {
        "appendix_g_total_risk_map.png",
        "appendix_g_regression_adjusted_map.png",
        "appendix_g_ratio_ladder.png",
        "appendix_g_metric_profile.png",
        "appendix_g_final_trio_composite.png",
    }
    for filename in required:
        assert filename in source

    assert "canonical_reports_devstartfix_full_20260730T154855Z" in source
    assert "tear_sheet_trio_ext2026" in source
    assert "COMPLETED" in source
    assert "manifest_sha256" in source
    assert "validate_canonical_report_bundle" in source
    assert "sha256" in source
    assert "projection_of" in source
    assert "APPENDIX_G_REPORT_BUNDLE" in source
    assert "APPENDIX_G_OUTPUT_DIR" in source

    # A paper renderer may scale values for display but must not recreate financial metrics.
    forbidden_local_metric_definitions = (
        "def cagr",
        "def annualized_vol",
        "def annualized_volatility",
        "def sharpe",
        "def sortino",
        "def calmar",
        "def max_drawdown",
        "def raw_market_model",
        "def regression",
    )
    lower = source.lower()
    for definition in forbidden_local_metric_definitions:
        assert definition not in lower
