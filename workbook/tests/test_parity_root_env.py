"""Parity proof for the vendored Sharpe-stability computation (task 3.2, R6.2).

``factor_workbook.vendored_ssr`` is a verbatim copy of
``macro_framework/ssr.py``. In the root environment (pytest run from the repo
root, where ``macro_framework`` is importable) the vendored file is compared
byte for byte against the source, and both implementations are run on
identical inputs so every ``SSRResult`` field matches exactly. Outside the root
environment the parity tests auto-skip, but the vendored module must still
import and compute standalone on numpy/pandas alone.

Parity covers the VERDICT path too, not just the point estimates: SSR is an
effect size, the one-sided moving-block-bootstrap p-value is what decides the
storyboard's headline claims, and ``SSRInference.verdict()`` is the single
rendering of that decision. Both the p-values (main and mirror tail) and the
verdict string must be identical to the root module's, or the workbook would
re-derive a different conclusion than the repo it mirrors.
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
    def test_vendored_body_is_byte_identical_to_the_source(self):
        """The header's "VERBATIM" claim, enforced rather than asserted.

        Behavioral parity tests only catch drift on the inputs they happen to
        exercise; this catches it on the first character. A failure here is not
        a bug in either module — it means macro_framework/ssr.py moved and the
        vendored copy must be regenerated from it (header + source, verbatim).
        """
        source = (_REPO_ROOT / "macro_framework" / "ssr.py").read_text(encoding="utf-8")
        vendored = Path(vendored_ssr.__file__).read_text(encoding="utf-8")
        header, marker, body = vendored.partition('"""')
        assert all(
            line.startswith("#") for line in header.splitlines() if line.strip()
        ), "the vendored provenance header must be comments only"
        assert marker + body == source, (
            "vendored_ssr.py has drifted from macro_framework/ssr.py — "
            "re-sync the vendored copy from the source module"
        )

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

    def test_parity_of_block_length_on_synthetic(self):
        r = _synthetic_returns().to_numpy()
        assert vendored_ssr.politis_white_block_length(
            r
        ) == original_ssr.politis_white_block_length(r)

    @pytest.mark.parametrize("differential", [False, True])
    def test_inference_parity_p_values_and_verdict(self, differential):
        """The verdict path, end to end: both MBB tails and the rendered
        verdict string. n_boot=200 keeps the test quick — the bootstrap is
        seeded, so parity holds draw for draw at any B."""
        returns = _synthetic_returns()
        vendored = vendored_ssr.ssr_inference(returns, n_boot=200)
        original = original_ssr.ssr_inference(returns, n_boot=200)
        _assert_results_equal(vendored.result, original.result)
        assert vendored.p_value == original.p_value
        assert vendored.p_value_lower == original.p_value_lower
        assert vendored.block_len == original.block_len
        assert vendored.stable is original.stable
        assert vendored.stably_below is original.stably_below
        assert vendored.verdict(differential=differential) == original.verdict(
            differential=differential
        )

    def test_inference_parity_on_degenerate_series(self):
        """The too-short path renders the same 'insufficient observations'
        verdict in both — the S0/S5 fixture-subset case."""
        returns = _fixture_returns()
        vendored = vendored_ssr.ssr_inference(returns, n_boot=50)
        original = original_ssr.ssr_inference(returns, n_boot=50)
        assert math.isnan(vendored.p_value) and math.isnan(original.p_value)
        assert vendored.verdict() == original.verdict()


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

    def test_inference_decides_the_verdict_standalone(self):
        """The vendored module ships the whole verdict authority, not just the
        effect size: p-value, mirror tail, block length, verdict string."""
        inference = vendored_ssr.ssr_inference(_synthetic_returns(), n_boot=200)
        assert isinstance(inference, vendored_ssr.SSRInference)
        assert 0.0 <= inference.p_value <= 1.0
        assert 0.0 <= inference.p_value_lower <= 1.0
        assert inference.block_len >= 1
        assert isinstance(inference.stable, bool)
        assert f"SSR={inference.result.ssr:.2f}" in inference.verdict()

    def test_provenance_header_names_source_and_resync_procedure(self):
        source = Path(vendored_ssr.__file__).read_text()
        assert "macro_framework/ssr.py" in source
        assert "VERBATIM" in source
        assert "re-sync" in source
