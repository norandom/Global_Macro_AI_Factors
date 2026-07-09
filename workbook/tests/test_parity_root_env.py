"""Parity proof for the vendored Sharpe-stability computation (task 3.2, R6.2).

``factor_workbook.vendored_ssr`` is a verbatim copy of
``macro_framework/ssr.py``. In the root environment (pytest run from the repo
root, where ``macro_framework`` is importable) both implementations are run on
identical inputs and every ``SSRResult`` field must match exactly. Outside the
root environment the parity tests auto-skip, but the vendored module must
still import and compute standalone on numpy/pandas alone.
"""

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_workbook import vendored_ssr

# In the root checkout the repo root is two levels above this file; pytest does
# not put it on sys.path, so add it only when the original module is present.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if (_REPO_ROOT / "macro_framework" / "ssr.py").is_file():
    sys.path.insert(0, str(_REPO_ROOT))

try:  # root env only; the lean env has no macro_framework
    from macro_framework import ssr as original_ssr
except ImportError:  # pragma: no cover - lean env
    original_ssr = None

FIXTURES_EQUITY = "factor_equity_v1.parquet"


def _fixture_returns() -> pd.Series:
    value = pd.read_parquet(
        Path(__file__).parent / "fixtures" / FIXTURES_EQUITY
    )["value"]
    return value.pct_change().dropna()


def _synthetic_returns(n: int = 600) -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(
        rng.normal(0.0005, 0.01, n), index=pd.bdate_range("2015-01-01", periods=n)
    )


def _assert_results_equal(vendored, original) -> None:
    """Every SSRResult field exactly equal (NaN treated as equal to NaN)."""
    v = dataclasses.asdict(vendored)
    o = dataclasses.asdict(original)
    assert v.keys() == o.keys()
    for field, ov in o.items():
        vv = v[field]
        if isinstance(ov, float) and math.isnan(ov):
            assert math.isnan(vv), field
        else:
            assert vv == ov, field


# --------------------------------------------------------------------------- #
# Parity against the original module (root env only)                           #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(original_ssr is None, reason="macro_framework not importable")
class TestParityWithOriginal:
    def test_parity_on_released_equity_fixture(self):
        returns = _fixture_returns()
        _assert_results_equal(
            vendored_ssr.compute_ssr(returns), original_ssr.compute_ssr(returns)
        )

    def test_parity_on_long_synthetic_series(self):
        # 600 obs exercises the rolling-window + HAC path (n_rolling > 0),
        # which the 5-row fixture cannot reach.
        returns = _synthetic_returns()
        vendored = vendored_ssr.compute_ssr(returns)
        original = original_ssr.compute_ssr(returns)
        assert original.n_rolling > 0
        _assert_results_equal(vendored, original)

    def test_parity_of_helpers_on_synthetic(self):
        z = vendored_ssr.rolling_sharpe(_synthetic_returns()).to_numpy()
        assert vendored_ssr.andrews_bandwidth(z) == original_ssr.andrews_bandwidth(z)
        assert vendored_ssr.newey_west_var(z) == original_ssr.newey_west_var(z)


# --------------------------------------------------------------------------- #
# Standalone behavior of the vendored copy (runs in any env)                   #
# --------------------------------------------------------------------------- #


class TestVendoredStandalone:
    def test_synthetic_series_yields_finite_result(self):
        returns = _synthetic_returns()
        result = vendored_ssr.compute_ssr(returns)
        assert isinstance(result, vendored_ssr.SSRResult)
        assert result.n_obs == len(returns)
        assert result.n_rolling == len(returns) - vendored_ssr.TRADING_DAYS + 1
        assert result.L_hac >= 1
        for field in ("sr_full", "mean_rolling_sr", "sigma_hac", "ssr"):
            assert math.isfinite(getattr(result, field)), field

    def test_short_series_degrades_to_nan(self):
        result = vendored_ssr.compute_ssr(_fixture_returns())
        assert result.n_rolling == 0
        assert math.isnan(result.ssr)

    def test_provenance_header_names_source_and_commit(self):
        source = Path(vendored_ssr.__file__).read_text()
        assert "macro_framework/ssr.py" in source
        assert "c8b03e7efafc42a6567dbd23566a5531a3e1276d" in source
