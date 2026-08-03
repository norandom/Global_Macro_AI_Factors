from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "appendix_h_deterministic_factor.ipynb"
SOURCE_DIR = ROOT / "data" / "streams" / "10y"
MACRO_PATH = ROOT / "data" / "macro_panel_monthly.parquet"
EXPECTED_HASHES = {
    "loadings": "97b3eaff5a004be0ce0bbe3b69264da9410cf0eb5230c1cebdeb854e05d1ea83",
    "decision_log": "d0f77d59f92854965bce3cc6c543b4738dfd7c4ccb9883599d63f45928db042c",
    "targets": "c5bb7dda289377ea0d3fc4938abf19a7eea14719478611bf0201c743310a00fc",
    "equity": "6fedeb27c327580ea83fc599adadc3a6ea8c3cff54293ea138600b267e764296",
    "header": "b42c855970619c4eb2f33a1e537faa6a3a1d4c9c27bb2d2c013f287d5a07c392",
    "macro_panel": "1444e80a3e1d9e375e581d829c507b84e406e504dfed421e072edaa6eb061cac",
}
SOURCE_PATHS = {
    "loadings": SOURCE_DIR / "factor_loadings_ext2026.parquet",
    "decision_log": SOURCE_DIR / "factor_decision_log_ext2026.json",
    "targets": SOURCE_DIR / "factor_targets_ext2026.parquet",
    "equity": SOURCE_DIR / "factor_equity_ext2026.parquet",
    "header": SOURCE_DIR / "factor_ext2026_run_header.json",
    "macro_panel": MACRO_PATH,
}
AXES = ("inflation", "growth", "credit_stress", "policy", "risk_appetite")
ASSETS = ("SWDA.L", "XLK", "IAU", "BIL")
EXPECTED_FIGURES = {
    "appendix_h_pit_macro_zscores.png",
    "appendix_h_factor_loading_zscores.png",
    "appendix_h_signed_tilts_bl_q.png",
    "appendix_h_persisted_target_weights.png",
    "appendix_h_persisted_equity_level.png",
    "appendix_h_worked_factor_trace.png",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _code_source() -> str:
    return "\n\n".join(_cell_source(cell) for cell in _notebook()["cells"] if cell.get("cell_type") == "code")


def _all_source() -> str:
    return "\n\n".join(_cell_source(cell) for cell in _notebook()["cells"])


def _trees() -> list[ast.AST]:
    trees: list[ast.AST] = []
    for index, cell in enumerate(_notebook()["cells"]):
        if cell.get("cell_type") == "code":
            trees.append(ast.parse(_cell_source(cell), filename=f"{NOTEBOOK.name}:cell-{index}"))
    return trees


def _called_attributes() -> set[str]:
    return {
        node.func.attr
        for tree in _trees()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def test_appendix_h_notebook_is_valid_and_clean_source() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    assert _trees()
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell.get("cell_type") == "code")
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell.get("cell_type") == "code")


def test_legacy_source_hashes_are_explicit_and_current() -> None:
    source = _code_source()
    for role, path in SOURCE_PATHS.items():
        assert path.is_file(), role
        assert _sha256(path) == EXPECTED_HASHES[role]
        assert EXPECTED_HASHES[role] in source
    assert "non-manifested legacy/reconstructed" in _all_source().lower()
    assert "factor_run.v1" in _all_source()
    assert "not a completed" in _all_source().lower()


def test_legacy_ten_year_inputs_have_expected_structure() -> None:
    loadings = pd.read_parquet(SOURCE_PATHS["loadings"])
    targets = pd.read_parquet(SOURCE_PATHS["targets"])
    equity = pd.read_parquet(SOURCE_PATHS["equity"])
    header = json.loads(SOURCE_PATHS["header"].read_text(encoding="utf-8"))
    decision_log = json.loads(SOURCE_PATHS["decision_log"].read_text(encoding="utf-8"))

    assert list(loadings.columns) == ["parse_ok", "segment", "variant", *AXES]
    assert len(loadings) == 126
    assert loadings.index.min() == pd.Timestamp("2016-01-04")
    assert loadings.index.max() == pd.Timestamp("2026-06-01")
    assert set(loadings["variant"]) == {"pit"}
    assert int(loadings["parse_ok"].sum()) == 125
    assert list(loadings.index[~loadings["parse_ok"]]) == [pd.Timestamp("2024-03-01")]
    assert loadings.loc[~loadings["parse_ok"], list(AXES)].isna().all(axis=None)
    assert np.isfinite(loadings.loc[loadings["parse_ok"], list(AXES)].to_numpy(dtype=float)).all()
    assert (np.abs(loadings.loc[loadings["parse_ok"], list(AXES)].to_numpy(dtype=float)) <= 1.0).all()

    for key in ("p_memorized", "parse_ok", "steered", "conviction", "loadings", "views"):
        assert set(pd.to_datetime(list(decision_log[key]))) == set(loadings.index), key
    assert list(targets.columns) == list(ASSETS)
    target_rows = targets.loc[loadings.index]
    assert target_rows.notna().all(axis=None)
    assert np.isfinite(target_rows.to_numpy(dtype=float)).all()
    assert np.allclose(target_rows.sum(axis=1), 1.0, atol=1e-10)
    assert list(equity.columns) == ["value"]
    assert np.isfinite(equity["value"].to_numpy(dtype=float)).all()
    assert (equity["value"] > 0).all()
    equity_positions = equity.index.searchsorted(loadings.index, side="left")
    assert (equity_positions < len(equity)).all()
    observed_equity_dates = equity.index[equity_positions]
    assert (observed_equity_dates >= loadings.index).all()
    assert ((observed_equity_dates - loadings.index).days <= 3).all()
    assert header["n_rebalances"]["total"] == 126
    assert header["window"]["first_rebalance"] == "2016-01-04"
    assert header["window"]["last_rebalance"] == "2026-06-01"


def test_macro_selection_is_strictly_prior_and_loading_log_mismatches_are_disclosed() -> None:
    loadings = pd.read_parquet(SOURCE_PATHS["loadings"])
    macro = pd.read_parquet(SOURCE_PATHS["macro_panel"])
    decision_log = json.loads(SOURCE_PATHS["decision_log"].read_text(encoding="utf-8"))

    assert list(macro.columns) == ["cpi_yoy", "t10y2y", "hy_oas", "cpi_yoy_z", "t10y2y_z", "hy_oas_z"]
    source_dates = []
    for date in loadings.index:
        candidates = macro.loc[macro.index < date]
        assert not candidates.empty
        complete = candidates.dropna(subset=["cpi_yoy_z", "t10y2y_z", "hy_oas_z"])
        assert not complete.empty
        selected = complete.index[-1]
        assert selected < date
        assert np.isfinite(complete.iloc[-1][["cpi_yoy_z", "t10y2y_z", "hy_oas_z"]].to_numpy(dtype=float)).all()
        source_dates.append(selected)
    assert max((date - source_date).days for date, source_date in zip(loadings.index, source_dates, strict=True)) >= 60

    mismatches: list[pd.Timestamp] = []
    for date, row in loadings.loc[loadings["parse_ok"]].iterrows():
        axes = [axis for axis in AXES if abs(float(row[axis]) - float(decision_log["loadings"][str(date)][axis])) > 1e-12]
        if axes:
            mismatches.append(date)
    assert mismatches == [pd.Timestamp("2025-10-01"), pd.Timestamp("2026-04-01"), pd.Timestamp("2026-05-01")]
    source = _all_source()
    for date in mismatches:
        assert date.strftime("%Y-%m-%d") in source
    assert "macro_source_date < rebalance_date" in _code_source()


def test_notebook_uses_expanding_prior_loading_zscores_and_masks_parse_failure() -> None:
    source = _code_source()
    assert "LOADING_Z_MIN_HISTORY = 12" in source
    assert "LOADING_Z_DDOF = 1" in source
    assert "expanding_prior_zscore" in source
    assert "len(history) >= LOADING_Z_MIN_HISTORY" in source
    assert "ddof=LOADING_Z_DDOF" in source
    assert "unavailable, not zero" in _all_source().lower()
    assert "2024-03-01" in source
    assert "hatch=\"///\"" in source


def test_notebook_is_presentation_only_and_restricts_writes_to_output_namespace() -> None:
    source = _code_source()
    forbidden_modules = {"requests", "yfinance", "httpx", "urllib", "socket", "sqlalchemy", "openai", "anthropic"}
    imported: set[str] = set()
    for tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
    assert not {module for module in imported if module.split(".")[0] in forbidden_modules}
    assert not (_called_attributes() & {"pct_change", "cumprod", "cov", "corr", "rolling", "resample", "ewm", "quantile", "optimization", "blacklitterman_stats"})
    assert "LlmMacroAgent" not in source
    assert "hrp_cvar_weights(" not in source
    assert "bl_mv_weights(" not in source
    assert "metric_block(" not in source
    assert "ssr_inference(" not in source
    assert "to_parquet(panel_path" in source
    assert "OUTPUT_DIR / PANEL_NAME" in source
    assert "manifest_path = OUTPUT_DIR / \"presentation_manifest.json\"" in source
    assert "OUTPUT_DIR / \"appendix_h_" in source
    assert "source_hashes_before" in source
    assert "assert_sources_unchanged" in source


def test_ridra_is_future_markdown_only_and_expected_outputs_are_declared() -> None:
    notebook = _notebook()
    code = _code_source().lower()
    markdown = "\n".join(_cell_source(cell) for cell in notebook["cells"] if cell.get("cell_type") == "markdown").lower()
    assert "ridra" not in code
    assert "future work — ridra" in markdown
    assert "not implemented" in markdown
    assert "adwin" in markdown
    for name in EXPECTED_FIGURES:
        assert name in _code_source()
    assert "appendix_h.deterministic_factor_panel.v1" in _code_source()
    assert "appendix_h_deterministic_factor.presentation.v1" in _code_source()


def test_notebook_executes_offline_into_an_isolated_output_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("nbclient")
    from nbclient import NotebookClient

    source_hashes_before = {name: _sha256(path) for name, path in SOURCE_PATHS.items()}
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    guard = nbformat.v4.new_code_cell(
        "import socket\n"
        "import urllib.request\n"
        "def _appendix_h_network_block(*args, **kwargs):\n"
        "    raise RuntimeError('network disabled for Appendix H execution')\n"
        "socket.socket.connect = _appendix_h_network_block\n"
        "urllib.request.urlopen = _appendix_h_network_block\n"
    )
    notebook.cells.insert(0, guard)
    output_dir = ROOT / ".pytest_appendix_h_outputs" / tmp_path.name
    shutil.rmtree(output_dir, ignore_errors=True)
    try:
        monkeypatch.setenv("APPENDIX_H_SOURCE_DIR", "data/streams/10y")
        monkeypatch.setenv("APPENDIX_H_MACRO_PANEL", "data/macro_panel_monthly.parquet")
        monkeypatch.setenv("APPENDIX_H_NOTEBOOK_PATH", str(NOTEBOOK))
        monkeypatch.setenv("APPENDIX_H_OUTPUT_DIR", str(output_dir))
        monkeypatch.setenv("MPLBACKEND", "Agg")
        client = NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
        client.execute(cwd=str(ROOT))

        assert {path.name for path in output_dir.iterdir()} == EXPECTED_FIGURES | {"appendix_h_deterministic_factor_panel.parquet", "presentation_manifest.json"}
        manifest = json.loads((output_dir / "presentation_manifest.json").read_text(encoding="utf-8"))
        assert manifest["schema"] == "appendix_h_deterministic_factor.presentation.v1"
        assert manifest["source_status"] == "non-manifested legacy/reconstructed"
        assert set(manifest["outputs"]) == EXPECTED_FIGURES
        assert manifest["panel"]["schema"] == "appendix_h.deterministic_factor_panel.v1"
        assert manifest["panel"]["rows"] == 504
        assert {name: _sha256(path) for name, path in SOURCE_PATHS.items()} == source_hashes_before
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.parent.rmdir() if output_dir.parent.exists() and not any(output_dir.parent.iterdir()) else None
