from __future__ import annotations

import ast
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "14_prompt_refinement_and_contrast.ipynb"
RUN_DIR = ROOT / "data" / "provisional_remediation" / "factor_runs" / "factor_ext2026_2019-01-01_2026-06-30_v1"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)


def _code_source(notebook: dict) -> str:
    return "\n\n".join(_source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code")


def _called_attributes(notebook: dict) -> set[str]:
    calls: set[str] = set()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse(_source(cell), filename=f"{NOTEBOOK.name}:cell-{index}")
        calls.update(
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        )
    return calls


def _exact_wilcoxon_greater(values: np.ndarray) -> tuple[int, float]:
    absolute_values = np.abs(values)
    assert np.all(values != 0.0)
    assert len(np.unique(absolute_values)) == len(values)
    ranks = stats.rankdata(absolute_values, method="ordinal").astype(int)
    w_plus = int(ranks[values > 0.0].sum())
    lower_tail_rank = len(values) * (len(values) + 1) // 2 - w_plus
    counts = [0] * (lower_tail_rank + 1)
    counts[0] = 1
    for rank in range(1, len(values) + 1):
        for subtotal in range(lower_tail_rank, rank - 1, -1):
            counts[subtotal] += counts[subtotal - rank]
    return w_plus, float(Fraction(sum(counts), 2 ** len(values)))


def _paired_summary(values: pd.Series) -> dict[str, float | int]:
    values = values.to_numpy(dtype=float)
    t_result = stats.ttest_1samp(values, popmean=0.0, alternative="greater")
    ci_low, ci_high = stats.t.interval(
        0.95,
        len(values) - 1,
        loc=float(values.mean()),
        scale=float(values.std(ddof=1) / np.sqrt(len(values))),
    )
    w_plus, wilcoxon_p = _exact_wilcoxon_greater(values)
    positives = int((values > 0.0).sum())
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "paired_d": float(values.mean() / values.std(ddof=0)),
        "t": float(t_result.statistic),
        "t_p": float(t_result.pvalue),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "w_plus": w_plus,
        "wilcoxon_p": wilcoxon_p,
        "positives": positives,
        "sign_p": float(binomtest(positives, len(values), 0.5, alternative="greater").pvalue),
    }


def test_notebook_is_clean_and_code_cells_compile() -> None:
    notebook = _load_notebook()
    assert notebook["nbformat"] == 4
    assert [cell["id"] for cell in notebook["cells"]] == [
        "765687cf",
        "13726906",
        "629ec1ce",
        "61e598bc",
        "98102f9d",
        "b2194349",
        "1d7dd22c",
        "192b655d",
        "385eab4d",
        "73bae373",
        "a0ef2cb6",
    ]
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(_source(cell), filename=f"{NOTEBOOK.name}:cell-{index}")
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_notebook_retains_immutable_paired_design_and_pm_conclusion() -> None:
    source = _code_source(_load_notebook())
    narrative = " ".join(" ".join(_source(cell).split()) for cell in _load_notebook()["cells"]).replace("**", "")

    for token in (
        "FINANCE_NOTEBOOK_SOURCE_ROOT",
        "FINANCE_NOTEBOOK_FACTOR_RUN_DIR",
        "COMPLETED",
        "sha256_file(path) == entry[\"sha256\"]",
        'replay_audit["result"] == "pass"',
        'expected_variants == ("nonpit_diagnostic", "pit")',
        'pit_rows["rebalance_date"].equals(nonpit_rows["rebalance_date"])',
        'pit_rows["pit_prompt_sha256"] == nonpit_rows["pit_prompt_sha256"]',
        'contrast["nonpit_p"] - contrast["pit_p"]',
        "exact_wilcoxon_greater",
        "stats.ttest_1samp",
        "binomtest",
        "stats.ttest_ind",
        "stats.mannwhitneyu",
        "build_factor_report_tables",
    ):
        assert token in source

    for phrase in (
        "Why a portfolio manager should care",
        "The statistical test",
        "What the result means in plain English",
        "Portfolio implication and supporting context",
        "invalid for deployment",
        "not a market-return forecast",
        "not a causal attribution",
        "not 90 strategy trials",
    ):
        assert phrase in narrative


def test_notebook_is_presentation_only_and_does_not_rederive_finance_metrics() -> None:
    notebook = _load_notebook()
    source = _code_source(notebook).lower()
    assert not any(
        module in source
        for module in ("requests", "yfinance", "httpx", "urllib", "socket", "sqlalchemy", "openai", "anthropic")
    )
    forbidden_calls = {
        "savefig",
        "write_text",
        "write_bytes",
        "to_csv",
        "to_parquet",
        "to_json",
        "to_pickle",
        "to_feather",
        "to_excel",
        "pct_change",
        "cumprod",
        "rolling",
        "resample",
    }
    assert not (_called_attributes(notebook) & forbidden_calls)
    assert "ssr_inference(" not in source
    assert "metric_block(" not in source


@pytest.mark.parametrize(
    ("segment", "expected"),
    [
        (
            "full_stream",
            {
                "n": 90,
                "mean": 0.3812553276932518,
                "median": 0.38261048464764547,
                "paired_d": 1.3804448545954078,
                "t": 13.0230907120977,
                "t_p": 1.258767272565715e-22,
                "ci_low": 0.3230858630134071,
                "ci_high": 0.4394247923730962,
                "w_plus": 4005,
                "wilcoxon_p": 1.728769513938181e-21,
                "positives": 84,
                "sign_p": 5.40608303118081e-19,
            },
        ),
        (
            "in_training",
            {
                "n": 65,
                "mean": 0.3969973072163796,
                "paired_d": 1.3484860590379284,
                "t": 10.787888472303427,
                "t_p": 2.441038674385798e-16,
                "w_plus": 2083,
                "wilcoxon_p": 3.454620487244875e-15,
                "positives": 60,
                "sign_p": 2.4347803504257137e-13,
            },
        ),
        (
            "post_cutoff",
            {
                "n": 25,
                "mean": 0.3403261809331194,
                "paired_d": 1.5709977355325155,
                "t": 7.696285678245,
                "t_p": 3.1088983620637954e-08,
                "w_plus": 324,
                "wilcoxon_p": 5.960464477539063e-08,
                "positives": 24,
                "sign_p": 7.748603820800781e-07,
            },
        ),
    ],
)
def test_immutable_paired_contamination_statistics(segment: str, expected: dict[str, float | int]) -> None:
    contrast = pd.read_parquet(RUN_DIR / "factor_contrast_ext2026.parquet")
    if segment == "full_stream":
        deltas = contrast["delta"]
    else:
        deltas = contrast.loc[contrast["segment"] == segment, "delta"]
    assert np.allclose(contrast["nonpit_p"] - contrast["pit_p"], contrast["delta"])

    actual = _paired_summary(deltas)
    for key, expected_value in expected.items():
        if isinstance(expected_value, int):
            assert actual[key] == expected_value
        else:
            assert actual[key] == pytest.approx(expected_value, rel=1e-12, abs=1e-15)


def test_cutoff_comparison_remains_non_significant() -> None:
    contrast = pd.read_parquet(RUN_DIR / "factor_contrast_ext2026.parquet")
    in_training = contrast.loc[contrast["segment"] == "in_training", "delta"]
    post_cutoff = contrast.loc[contrast["segment"] == "post_cutoff", "delta"]
    welch = stats.ttest_ind(in_training, post_cutoff, equal_var=False, alternative="two-sided")
    mann_whitney = stats.mannwhitneyu(in_training, post_cutoff, alternative="two-sided")
    assert welch.statistic == pytest.approx(0.98508171838584, abs=1e-12)
    assert welch.pvalue == pytest.approx(0.3286563996805718, abs=1e-12)
    assert mann_whitney.pvalue == pytest.approx(0.335, abs=1e-3)
