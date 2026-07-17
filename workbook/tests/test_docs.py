"""Deployment documentation and configuration example (R7.6, R1.3).

Asserts the README and ``pyxll.cfg.example`` exist in the package
directory, state the commercial runtime prerequisite and the
Excel-free verifiability of the data/computation layer, forbid tokens
in workbook artifacts, and enumerate the Excel-only manual checklist
behaviors excluded from the automated suite.
"""

from pathlib import Path

import pytest

_PKG_DIR = Path(__file__).resolve().parents[1]
_README = _PKG_DIR / "README.md"
_CFG = _PKG_DIR / "pyxll.cfg.example"


@pytest.fixture(scope="module")
def readme() -> str:
    assert _README.is_file(), "workbook/README.md is missing"
    return _README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cfg() -> str:
    assert _CFG.is_file(), "workbook/pyxll.cfg.example is missing"
    return _CFG.read_text(encoding="utf-8")


def test_readme_states_commercial_prerequisite(readme: str) -> None:
    """R7.6: the commercial license/trial prerequisite is explicit."""
    lower = readme.lower()
    assert "commercial" in lower
    assert "30-day trial" in lower
    assert "per-user" in lower and "annual" in lower
    assert "windows" in lower and "excel" in lower


def test_readme_states_excel_free_verifiability(readme: str) -> None:
    """R7.6: the data/computation layer is verifiable without Excel."""
    lower = readme.lower()
    assert "without excel" in lower
    assert "uv run pytest" in readme
    assert "PYTHONPATH=workbook" in readme


def test_readme_token_setup_never_stores_in_workbook(readme: str) -> None:
    """R1.3: keyring/env token path documented, never stored in artifacts."""
    assert 'keyring.set_password("factor-workbook", "github"' in readme
    assert "GITHUB_TOKEN" in readme
    lower = readme.lower()
    assert "never" in lower and "token" in lower
    # The prohibition must cover workbook files/cells explicitly.
    assert "workbook file" in lower or "workbook files" in lower


def test_readme_manual_checklist_covers_excel_only_behaviors(readme: str) -> None:
    """The checklist enumerates behaviors automated tests cannot cover."""
    lower = readme.lower()
    for phrase in (
        "manual excel verification checklist",
        "pyxll.cfg",
        "without freezing",
        "spill",
        "object-cache",
        "recalc",
        "#error",
        "provenance",
        "40-row",
        "ribbon",
    ):
        assert phrase in lower, f"checklist missing: {phrase}"


def test_readme_documents_install_and_usage(readme: str) -> None:
    lower = readme.lower()
    assert "pip install" in lower
    assert "[token]" in readme and "[stats]" in readme
    assert "pyxll install" in lower
    assert "build_workbook.py" in readme
    assert "Index!B1" in readme or "Index!$B$1" in readme
    assert "data-v2" in readme


def test_cfg_example_lists_addin_module_and_no_secrets(cfg: str) -> None:
    assert "[PYXLL]" in cfg and "[PYTHON]" in cfg
    assert "modules" in cfg and "factor_workbook.addin" in cfg
    assert "pythonpath" in cfg
    assert "<YOUR-" in cfg  # placeholder style, no real local paths
    lower = cfg.lower()
    for secret in ("ghp_", "github_pat_", "/home/", "c:\\users\\"):
        assert secret not in lower, f"cfg example leaks: {secret}"
