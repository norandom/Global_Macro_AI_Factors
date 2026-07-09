"""Optional certification re-derivation behind the ``stats`` extra (task 3.3, R2.6).

``factor_workbook.certification`` vendors
``macro_framework/factor_scoring.py::certification_stats``. With scikit-learn
installed (root environment) the vendored copy must match the original exactly
and — on the REAL local evidence parquet — reproduce the published controlled
separation values. Without the extra the module must still import,
``available()`` must be ``False``, and calling must raise a clear RuntimeError
naming ``factor-workbook[stats]`` (never an ImportError at import time).

The published-values test needs the full local evidence data
(``data/norecall_screen/evidence/...``) — the shipped 6-row fixture subset is
a single arm and cannot reproduce the full-matrix statistics — so it skips
outside the root checkout.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from factor_workbook import certification

# In the root checkout the repo root is two levels above this file; pytest does
# not put it on sys.path, so add it only when the original module is present.
# The original is imported INSIDE the parity tests (never at collection time):
# the root suite's test_feature_path_does_not_use_directional_facade asserts
# `macro_framework.factor_scoring` is absent from sys.modules until used.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_HAS_ORIGINAL = (_REPO_ROOT / "macro_framework" / "factor_scoring.py").is_file()
if _HAS_ORIGINAL:
    sys.path.insert(0, str(_REPO_ROOT))


def _original_stats():
    """The source-of-truth function, imported lazily (root env only)."""
    from macro_framework.factor_scoring import certification_stats

    return certification_stats

_EVIDENCE_DIR = (
    _REPO_ROOT / "data" / "norecall_screen" / "evidence" / "openai_gpt-oss-20b"
)

_SKLEARN = certification.available()


def _gaussian_clusters(
    seed: int, n: int, dim: int, shift: float
) -> tuple[list[list[float]], list[list[float]]]:
    """Two seeded Gaussian feature clusters; ``shift`` separates the classes."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, size=(n, dim)) + shift
    b = rng.normal(0.0, 1.0, size=(n, dim))
    return a.tolist(), b.tolist()


# --------------------------------------------------------------------------- #
# Parity against the original module (root env only)                           #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not _HAS_ORIGINAL or not _SKLEARN,
    reason="macro_framework source or sklearn not present",
)
class TestParityWithOriginal:
    def test_parity_on_separable_synthetic(self):
        x_is, x_oos = _gaussian_clusters(seed=1, n=20, dim=3, shift=3.0)
        kwargs = dict(n_boot=20, n_perm=39, seed=0)
        assert certification.certification_stats(
            x_is, x_oos, **kwargs
        ) == _original_stats()(x_is, x_oos, **kwargs)

    def test_parity_on_noise_synthetic_nondefault_seed(self):
        x_is, x_oos = _gaussian_clusters(seed=3, n=12, dim=2, shift=0.0)
        kwargs = dict(n_boot=15, n_perm=19, seed=7)
        assert certification.certification_stats(
            x_is, x_oos, **kwargs
        ) == _original_stats()(x_is, x_oos, **kwargs)


# --------------------------------------------------------------------------- #
# Vendored behavior with the extra installed (mirrors the producer's tests)    #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _SKLEARN, reason="stats extra (sklearn) not installed")
class TestVendoredWithExtra:
    def test_deterministic_given_seed(self):
        x_is, x_oos = _gaussian_clusters(seed=3, n=12, dim=2, shift=1.0)
        first = certification.certification_stats(
            x_is, x_oos, n_boot=20, n_perm=39, seed=7
        )
        second = certification.certification_stats(
            x_is, x_oos, n_boot=20, n_perm=39, seed=7
        )
        assert first == second

    def test_separable_clusters_are_significant(self):
        x_is, x_oos = _gaussian_clusters(seed=1, n=30, dim=3, shift=3.0)
        auc, ci_low, ci_high, perm_p = certification.certification_stats(
            x_is, x_oos, n_boot=50, n_perm=99, seed=0
        )
        assert auc > 0.9
        assert perm_p < 0.05
        assert ci_low > 0.5
        assert ci_high >= ci_low

    def test_pure_noise_ci_contains_chance(self):
        # seed=3 is a comfortably null draw (same choice as the producer test).
        x_is, x_oos = _gaussian_clusters(seed=3, n=30, dim=3, shift=0.0)
        _auc, ci_low, ci_high, perm_p = certification.certification_stats(
            x_is, x_oos, n_boot=50, n_perm=99, seed=0
        )
        assert perm_p > 0.1
        assert ci_low <= 0.5 <= ci_high

    def test_degenerate_class_raises_value_error(self):
        tiny = [[0.1, 0.2], [0.3, 0.4]]  # 2 rows < n_splits=5
        _x_is, x_oos = _gaussian_clusters(seed=4, n=10, dim=2, shift=0.0)
        with pytest.raises(ValueError):
            certification.certification_stats(tiny, x_oos)


# --------------------------------------------------------------------------- #
# Published-values reproduction from the REAL local evidence (root env only)   #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not _SKLEARN or not os.path.exists(_EVIDENCE_DIR / "evidence.parquet"),
    reason="local full evidence data or sklearn not present",
)
def test_reproduces_published_controlled_separation_from_raw_evidence():
    """Full-matrix re-derivation matches summary.json's published values (R2.6).

    Builds x_is/x_oos from the identifying/anonymized arms' included rows
    (std_* feature columns in the baseline's feature order) and re-runs the
    vendored statistics with the production defaults. ~15 s, local data only.
    """
    import pandas as pd

    baseline = json.loads((_EVIDENCE_DIR / "baseline.json").read_text())
    order = [k for k, m in baseline["feature_means"].items() if m is not None]
    cols = [f"std_{k}" for k in order]

    df = pd.read_parquet(_EVIDENCE_DIR / "evidence.parquet")
    x_is = df[(df["arm"] == "identifying") & df["included"]][cols].to_numpy()
    x_oos = df[(df["arm"] == "anonymized") & df["included"]][cols].to_numpy()

    published = json.loads((_EVIDENCE_DIR / "summary.json").read_text())
    assert len(x_is) == len(x_oos) == published["n_per_class"]

    auc, ci_low, ci_high, perm_p = certification.certification_stats(
        x_is, x_oos, seed=0
    )
    tol = 1e-9
    assert auc == pytest.approx(published["controlled_auc"], abs=tol)
    assert ci_low == pytest.approx(published["controlled_ci_low"], abs=tol)
    assert ci_high == pytest.approx(published["controlled_ci_high"], abs=tol)
    assert perm_p == pytest.approx(published["controlled_perm_p"], abs=tol)


# --------------------------------------------------------------------------- #
# Lean surface: no sklearn -> importable, unavailable, clear error             #
# --------------------------------------------------------------------------- #


class TestLeanSurface:
    def test_unavailable_flag_raises_clear_runtime_error(self, monkeypatch):
        monkeypatch.setattr(certification, "available", lambda: False)
        with pytest.raises(RuntimeError, match=r"factor-workbook\[stats\]"):
            certification.certification_stats([[0.0] * 2] * 5, [[0.0] * 2] * 5)

    def test_module_imports_without_sklearn(self):
        """Fresh interpreter with sklearn blocked: import ok, available() False,
        calling raises the RuntimeError (never ImportError at import time)."""
        code = textwrap.dedent(
            """
            import sys
            sys.modules["sklearn"] = None  # blocks `import sklearn` everywhere
            from factor_workbook import certification
            assert certification.available() is False
            try:
                certification.certification_stats([[0.0]] * 5, [[0.0]] * 5)
            except RuntimeError as exc:
                assert "factor-workbook[stats]" in str(exc)
            else:
                raise AssertionError("expected RuntimeError")
            print("LEAN-OK")
            """
        )
        env = dict(os.environ, PYTHONPATH=str(Path(certification.__file__).parents[1]))
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env
        )
        assert proc.returncode == 0, proc.stderr
        assert "LEAN-OK" in proc.stdout


# --------------------------------------------------------------------------- #
# Provenance                                                                   #
# --------------------------------------------------------------------------- #


def test_provenance_header_names_source_and_commit():
    source = Path(certification.__file__).read_text()
    assert "macro_framework/factor_scoring.py" in source
    assert "c85c2ed73eb0aba52b6bb937e9ef99a71272b4a6" in source
