from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "appendix_e": ROOT / "notebooks" / "appendix_e_factor_guard_ablation.ipynb",
    "appendix_f": ROOT / "notebooks" / "appendix_f_macro_to_bl_tilt.ipynb",
}
REQUIRED_TABLES = {
    "tear_sheet_factor_guard_ablation_ext2026.parquet",
    "factor_guard_ablation_equity_ext2026.parquet",
    "factor_guard_ablation_panel_ext2026.parquet",
}
CONFIG_IDS = {
    "factor_pit_ext2026",
    "factor_pit_unguarded_diagnostic_ext2026",
    "factor_nonpit_diagnostic_ext2026",
    "factor_nonpit_unguarded_diagnostic_ext2026",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _code_source(path: Path) -> str:
    notebook = _load(path)
    return "\n\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else str(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def _all_source(path: Path) -> str:
    notebook = _load(path)
    return "\n\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else str(cell.get("source", ""))
        for cell in notebook["cells"]
    )


def _trees(path: Path) -> list[ast.AST]:
    notebook = _load(path)
    trees: list[ast.AST] = []
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else str(cell.get("source", ""))
        trees.append(ast.parse(source, filename=f"{path.name}:cell-{index}"))
    return trees


def _called_attributes(path: Path) -> set[str]:
    return {
        node.func.attr
        for tree in _trees(path)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def _defined_functions(path: Path) -> set[str]:
    return {
        node.name
        for tree in _trees(path)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for tree in _trees(path):
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
    return modules


def _write_calls(path: Path) -> list[ast.Call]:
    write_attributes = {
        "savefig",
        "write_text",
        "write_bytes",
        "to_csv",
        "to_parquet",
        "to_json",
        "to_pickle",
        "to_feather",
        "to_excel",
    }
    calls: list[ast.Call] = []
    for tree in _trees(path):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in write_attributes:
                calls.append(node)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and any(flag in str(node.args[1].value) for flag in "wax+"):
                    calls.append(node)
    return calls


@pytest.mark.parametrize("path", NOTEBOOKS.values(), ids=NOTEBOOKS.keys())
def test_notebook_json_and_code_cells_compile(path: Path) -> None:
    notebook = _load(path)
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    assert _trees(path)


@pytest.mark.parametrize("path", NOTEBOOKS.values(), ids=NOTEBOOKS.keys())
def test_completed_bundle_contract_is_hash_row_schema_and_lineage_gated(path: Path) -> None:
    source = _code_source(path)
    assert "canonical_guard_ablation_reports.v1" in source
    assert "manifest_sha256=" in source
    assert "completed_manifest_sha256(marker_path) != manifest_sha" in source
    assert "sha256_file(path)" in source
    assert 'meta.get("rows"' in source
    assert "EXPECTED_TABLE_SCHEMAS" in source
    assert 'meta.get("schema", meta.get("schema_id"))' in source
    assert "factor_guard_ablation_run.v1" in source
    assert "factor_guard_ablation_run" in source
    assert "validate_source_child_run_lineage" in source
    assert "source_artifacts" in source
    assert "parent_manifests" in source
    assert "assert_sources_unchanged" in source
    assert "relative_to(repo_root)" in source
    assert "relative_to(bundle_dir)" in source
    assert REQUIRED_TABLES <= {token for token in REQUIRED_TABLES if token in source}


@pytest.mark.parametrize("path", NOTEBOOKS.values(), ids=NOTEBOOKS.keys())
def test_paths_are_environment_configurable_and_repository_contained(path: Path) -> None:
    source = _code_source(path)
    assert "GUARD_ABLATION_BUNDLE_DIR" in source
    assert "GUARD_ABLATION_SOURCE_RUN_DIR" in source
    assert "resolve_repo_path" in source
    assert "must remain inside repository root" in source
    assert "output namespace must be separate from canonical input bundle" in source
    expected_output_env = "APPENDIX_E_OUTPUT_DIR" if path == NOTEBOOKS["appendix_e"] else "APPENDIX_F_OUTPUT_DIR"
    assert expected_output_env in source


@pytest.mark.parametrize("path", NOTEBOOKS.values(), ids=NOTEBOOKS.keys())
def test_notebooks_do_not_rederive_finance_metrics(path: Path) -> None:
    forbidden_calls = {
        "pct_change",
        "cumprod",
        "cov",
        "corr",
        "rolling",
        "resample",
        "ewm",
        "prod",
        "quantile",
    }
    forbidden_function_tokens = (
        "metric",
        "cagr",
        "sharpe",
        "sortino",
        "calmar",
        "drawdown",
        "return",
        "ssr",
        "bootstrap",
        "regress",
    )
    assert not (_called_attributes(path) & forbidden_calls)
    locally_defined = _defined_functions(path)
    assert not {
        name
        for name in locally_defined
        if any(token in name.lower() for token in forbidden_function_tokens)
    }
    source = _code_source(path)
    assert "ssr_inference(" not in source
    assert "metric_block(" not in source


@pytest.mark.parametrize("path", NOTEBOOKS.values(), ids=NOTEBOOKS.keys())
def test_notebooks_have_no_network_provider_or_canonical_table_writes(path: Path) -> None:
    source = _code_source(path).lower()
    forbidden_modules = {"requests", "yfinance", "httpx", "urllib", "socket", "sqlalchemy", "openai", "anthropic"}
    imported = _imported_modules(path)
    assert not {module for module in imported if module.split(".")[0] in forbidden_modules}
    assert not any(token in source for token in ("nvidialm", "generate_many", "read_sql", "urlopen", "download(", ".unlink(", "shutil."))
    writes = _write_calls(path)
    assert writes
    for call in writes:
        rendered = ast.unparse(call)
        assert "OUTPUT_DIR" in rendered or "path" in rendered, f"write escapes presentation namespace: {rendered}"
        assert not any(token in rendered for token in ("BUNDLE_DIR", "SOURCE_RUN_DIR", "tables/", "mirrors/"))
    assert 'path = output_dir / "presentation_manifest.json"' in _code_source(path)
    assert "presentation_manifest.json" in source
    assert "source_hashes_before" in source


def test_appendix_e_protocol_cutoff_labels_and_honest_reading() -> None:
    source = _all_source(NOTEBOOKS["appendix_e"])
    assert CONFIG_IDS <= {token for token in CONFIG_IDS if token in source}
    assert "2024-06-01" in source
    assert "PIT unguarded — guard-disabled diagnostic; non-deployable" in source
    assert "non-PIT unguarded — identifying guard-disabled diagnostic; non-deployable" in source
    assert "Only guarded PIT is" in source and "deployable" in source
    assert "not proof" in source.lower()
    assert "no significant change" in source.lower()
    assert "no visible change" in source.lower()
    assert "observations" in source
    assert "rebalances" in source
    assert "inference" in source


def test_appendix_e_keeps_controlled_and_combined_relative_wealth_distinct() -> None:
    source = _code_source(NOTEBOOKS["appendix_e"])
    assert "controlled_pit_unguarded_vs_guarded" in source
    assert "combined_nonpit_unguarded_vs_pit_guarded_stress" in source
    assert "Controlled PIT ablation" in source
    assert "Combined naïve stress" in source
    assert "PIT unguarded / PIT guarded" in source
    assert "non-PIT unguarded / PIT guarded" in source
    assert "Distinct estimands" in source


def test_appendix_e_mechanism_uses_persisted_panel_fields() -> None:
    source = _code_source(NOTEBOOKS["appendix_e"])
    assert "p_memorized" in source
    assert "raw_view_tilt" in source
    assert "applied_view_tilt" in source
    assert "persisted_observed_attenuation" in source
    assert "expected_attenuation" in source
    assert "relation_error" in source
    assert "no portfolio outcome is rederived" in source


def test_appendix_e_figure_and_manifest_inventory() -> None:
    source = _code_source(NOTEBOOKS["appendix_e"])
    for filename in (
        "appendix_e_equity_drawdown.png",
        "appendix_e_relative_wealth.png",
        "appendix_e_protocol_metrics.png",
        "appendix_e_mechanism_attenuation.png",
    ):
        assert filename in source
    assert "appendix_e_factor_guard_ablation.presentation.v1" in source
    assert "width_px" in source and "height_px" in source
    assert "source_tables" in source and "outputs" in source


def test_appendix_f_zero_centered_heatmaps_and_prespecified_pairs() -> None:
    source = _code_source(NOTEBOOKS["appendix_f"])
    assert 'MACRO_Z_LIMIT = 3.0' in source
    assert "vmin=-MACRO_Z_LIMIT" in source
    assert "vmax=MACRO_Z_LIMIT" in source
    assert "vmin=-target_delta_limit" in source
    assert "vmax=target_delta_limit" in source
    assert "DIVERGING_CMAP" in source
    assert "target_delta" in source
    for macro_axis, asset in (("cpi_yoy_z", "IAU"), ("t10y2y_z", "SWDA.L"), ("hy_oas_z", "BIL")):
        assert macro_axis in source
        assert asset in source
    assert "Table view (CVD/print-safe signed values)" in source


def test_appendix_f_descriptive_noncausal_association_only_when_supplied() -> None:
    source = _all_source(NOTEBOOKS["appendix_f"])
    assert "association_col is not None" in source
    assert "producer-supplied" in source.lower()
    assert "no association value supplied" in source.lower()
    assert "model-implied contemporaneous" in source.lower()
    assert "non-causal" in source.lower()
    assert ".corr(" not in source


def test_appendix_f_worked_dates_use_only_macro_state_norm() -> None:
    source = _code_source(NOTEBOOKS["appendix_f"])
    assert "macro_state_norm" in source
    assert "norm_by_date.idxmax()" in source
    assert "norm_by_date.idxmin()" in source
    assert "maximum macro-state norm" in source
    assert "nearest-neutral macro-state norm" in source
    assert "never returns" in source
    for field in ("raw_tilt", "applied_tilt", "bl_q", "target_delta"):
        assert field in source
    for loading in (
        "loading_inflation",
        "loading_growth",
        "loading_credit_stress",
        "loading_policy",
        "loading_risk_appetite",
    ):
        assert loading in source


def test_appendix_f_explains_transformation_chain() -> None:
    source = _all_source(NOTEBOOKS["appendix_f"]).lower()
    for phrase in ("exposure map", "conviction", "recall attenuation", "bl conversion", "hrp/bl blend"):
        assert phrase in source


def test_appendix_f_figure_and_manifest_inventory() -> None:
    source = _code_source(NOTEBOOKS["appendix_f"])
    for filename in (
        "appendix_f_macro_state_heatmap.png",
        "appendix_f_target_delta_heatmap.png",
        "appendix_f_macro_tilt_links.png",
        "appendix_f_worked_dates.png",
    ):
        assert filename in source
    assert "appendix_f_macro_to_bl_tilt.presentation.v1" in source
    assert "width_px" in source and "height_px" in source
    assert "source_tables" in source and "outputs" in source
