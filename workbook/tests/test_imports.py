"""Import-smoke test: every planned workbook module imports in the root env.

Proves the task 1.1 observable — the root test run can import the whole
``factor_workbook`` package with no heavy work at import time, and the
package ``__init__`` re-exports every planned module.
"""

import importlib

MODULES = (
    "release",
    "contract",
    "rederive",
    "vendored_ssr",
    "certification",
    "verify",
    "steps",
    "addin",
)


def test_package_imports() -> None:
    """The bare package imports without Excel or heavy dependencies."""
    pkg = importlib.import_module("factor_workbook")
    assert isinstance(pkg.__version__, str)


def test_every_module_importable() -> None:
    """Each planned module imports individually in the root environment."""
    for name in MODULES:
        importlib.import_module(f"factor_workbook.{name}")


def test_init_reexports_all_modules() -> None:
    """The package re-exports every module (module objects, not symbols)."""
    import factor_workbook

    assert set(factor_workbook.__all__) == set(MODULES)
    for name in MODULES:
        module = getattr(factor_workbook, name)
        assert module.__name__ == f"factor_workbook.{name}"


def test_excel_support_libs_in_root_env() -> None:
    """The dev group carries the pyxll stub and openpyxl for the root run."""
    importlib.import_module("pyxll")
    importlib.import_module("openpyxl")


def test_stats_extra_dep_available_in_root_env() -> None:
    """scikit-learn (the optional stats extra) is importable in the root env."""
    importlib.import_module("sklearn")
