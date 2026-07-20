"""Foundation smoke test: the statsmodels HAC-regression path (Task 1).

Proves the Newey-West HAC t-stat dependency (statsmodels) is a declared direct
dependency and that its ``OLS(...).fit(cov_type="HAC")`` path — the same call the
nb05 own-basket attribution uses — imports and returns a finite t-statistic.
Requirements: 8.2 (reuse existing implementations; no divergent re-implementation).
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_statsmodels_is_declared_direct_dependency() -> None:
    """statsmodels is a declared direct dep, not merely transitively present (8.2)."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    deps = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    names = {d.split(">")[0].split("=")[0].split("[")[0].strip() for d in deps}
    assert "statsmodels" in names


def test_hac_ols_fit_returns_finite_tstat() -> None:
    """OLS(...).fit(cov_type="HAC") loads and yields a finite t-stat (8.2)."""
    import numpy as np
    import statsmodels.api as sm

    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    y = 1.0 + 0.5 * x + rng.standard_normal(200)  # known slope, positive alpha

    res = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    assert np.isfinite(res.tvalues[0])  # alpha (const) t-stat is finite
