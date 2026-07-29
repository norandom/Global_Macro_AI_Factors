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
from factor_workbook.rederive import equity_metrics

# In the root checkout the repo root is two levels above this file; pytest does
# not put it on sys.path, so add it only when the original module is present.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if (_REPO_ROOT / "macro_framework" / "ssr.py").is_file():
    sys.path.insert(0, str(_REPO_ROOT))

try:  # root env only; the lean env has no macro_framework
    from macro_framework import ssr as original_ssr
    from macro_framework.evaluation import crisis_metrics as original_crisis_metrics
except ImportError:  # pragma: no cover - lean env
    original_ssr = None
    original_crisis_metrics = None

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


def _assert_inferences_equal(vendored, original) -> None:
    _assert_results_equal(vendored.result, original.result)
    for field in dataclasses.fields(original):
        if field.name == "result":
            continue
        vv = getattr(vendored, field.name)
        ov = getattr(original, field.name)
        if isinstance(ov, float) and math.isnan(ov):
            assert math.isnan(vv), field.name
        else:
            assert vv == ov, field.name


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
        _assert_inferences_equal(vendored, original)
        assert vendored.stable is original.stable
        assert vendored.stably_below is original.stably_below
        assert vendored.verdict(differential=differential) == original.verdict(
            differential=differential
        )

    def test_cash_excess_nondefault_settings_match_root_exactly(self):
        portfolio = _synthetic_returns()
        cash = pd.Series(0.0001, index=portfolio.index)
        settings = {
            "window": 126,
            "sr_star": 0.25,
            "n_boot": 40,
            "seed": 7,
            "alpha": 0.10,
        }
        vendored = vendored_ssr.ssr_inference(portfolio - cash, **settings)
        original = original_ssr.ssr_inference(portfolio - cash, **settings)
        _assert_inferences_equal(vendored, original)
        assert vendored.verdict() == original.verdict()

    def test_inference_parity_on_valid_short_series(self):
        """A valid window with fewer than ten rolling observations renders the
        same insufficient-inference result in both implementations."""
        returns = _synthetic_returns(n=vendored_ssr.TRADING_DAYS + 8)
        vendored = vendored_ssr.ssr_inference(returns, n_boot=50)
        original = original_ssr.ssr_inference(returns, n_boot=50)
        _assert_inferences_equal(vendored, original)
        assert math.isnan(vendored.p_value) and math.isnan(original.p_value)
        assert vendored.verdict() == original.verdict()

    def test_crisis_parity_includes_the_entry_return(self):
        value = pd.Series(
            [90.0, 100.0, 80.0, 88.0],
            index=pd.to_datetime(
                ["2021-12-29", "2021-12-31", "2022-01-03", "2022-01-04"]
            ),
        )
        workbook = equity_metrics(value, crisis=("2022-01-01", "2022-01-04"))
        original = original_crisis_metrics(value, "2022-01-01", "2022-01-04")
        assert original is not None
        assert workbook.crisis_return == original.episode_return
        assert workbook.crisis_max_drawdown == original.boundary_anchored_max_drawdown
        assert workbook.crisis_vol_ann == original.volatility_ann


# --------------------------------------------------------------------------- #
# data-v4 contract mirror parity (task 10.3; root env only)                    #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(original_ssr is None, reason="macro_framework not importable")
class TestDataV4ContractMirror:
    """The workbook's mirrored data-v4 vocabulary IS the root authority's.

    The lean workbook project cannot import ``macro_framework`` or ``scripts``,
    so ``factor_workbook.contract`` mirrors the canonical schema vocabulary as
    data. These tests pin that mirror, field for field, to the root reporting
    schemas, the report producer, and the frozen publication catalog — any
    drift in either direction fails here in the root environment.
    """

    def test_row_vocabularies_match_macro_framework_reporting(self):
        reporting = pytest.importorskip("macro_framework.reporting")
        from factor_workbook import contract

        assert tuple(contract.V4_PROVENANCE_COLUMNS) == reporting.REQUIRED_PROVENANCE
        specs = contract.DATA_V4_ASSET_SPECS
        fields = {sid: s.fields for sid, s in reporting.REPORT_SCHEMAS.items()}
        by_family = {
            "portfolio_metrics_reader_ext2026": reporting.READER_SCHEMA,
            "portfolio_metrics_vectorbt365_ext2026": reporting.LEGACY_SCHEMA,
            "portfolio_metrics_differential_ext2026": reporting.DIFFERENTIAL_SCHEMA,
            "attribution_raw_market_model_ext2026": reporting.ATTRIBUTION_SCHEMA,
            "crisis_metrics_ext2026": reporting.CRISIS_SCHEMA,
            "monthly_returns_ext2026": reporting.MONTHLY_SCHEMA,
        }
        for key, schema_id in by_family.items():
            assert set(specs[key].columns) == fields[schema_id], key
            assert contract.DATA_V4_TABLE_SCHEMAS[key] == schema_id
        # mixed tear sheets are exactly the unions of their row families
        assert set(specs["tear_sheet_ai_variants_ext2026"].columns) == (
            fields[reporting.READER_SCHEMA] | fields[reporting.DIFFERENTIAL_SCHEMA]
        )
        assert set(specs["tear_sheet_sjm_crowding_ext2026"].columns) == (
            fields[reporting.READER_SCHEMA]
            | fields[reporting.ATTRIBUTION_SCHEMA]
            | fields[reporting.CRISIS_SCHEMA]
        )
        assert (
            set(specs["tear_sheet_trio_ext2026"].columns)
            == fields[reporting.READER_SCHEMA]
        )

    def test_tables_settings_and_aliases_match_the_report_producer(self):
        bts = pytest.importorskip("scripts.build_tear_sheet")
        from factor_workbook import contract

        assert dict(contract.SSR_REPORT_DEFAULTS) == dict(bts.SSR_REPORT_DEFAULTS)
        assert {k: dict(v) for k, v in contract.REPORT_CSV_LOCALE_SPECS.items()} == {
            k: dict(v) for k, v in bts.REPORT_CSV_LOCALE_SPECS.items()
        }
        universe = bts.MARKOWITZ_ASSET_UNIVERSE
        assert contract.MARKOWITZ_ASSET_UNIVERSE == universe
        specs = contract.DATA_V4_ASSET_SPECS
        for window in ("10y", "max"):
            assert set(specs[f"markowitz_{window}_moments"].columns) == set(
                bts.markowitz_moments_columns(universe)
            )
            assert set(specs[f"markowitz_{window}_frontier"].columns) == set(
                bts.markowitz_frontier_columns(universe)
            )
        assert set(specs["risk_decomposition_ext2026"].columns) == set(
            bts.risk_decomposition_columns()
        )
        for mapping in (
            bts.FACTOR_REPORT_TABLE_SCHEMAS,
            bts.AUXILIARY_REPORT_TABLE_SCHEMAS,
            bts.SJM_REPORT_TABLE_SCHEMAS,
            bts.TRIO_REPORT_TABLE_SCHEMAS,
            bts.MARKOWITZ_REPORT_TABLE_SCHEMAS,
        ):
            for table, schema_id in mapping.items():
                if table in contract.DATA_V4_TABLE_SCHEMAS:
                    assert contract.DATA_V4_TABLE_SCHEMAS[table] == schema_id, table
        # the two explicitly frozen tear-sheet identities, pinned verbatim
        assert (
            contract.DATA_V4_TABLE_SCHEMAS["tear_sheet_sjm_crowding_ext2026"]
            == bts.SJM_REPORT_TABLE_SCHEMAS["tear_sheet_sjm_crowding_ext2026"]
            == "tear_sheet.sjm.v3"
        )
        assert (
            contract.DATA_V4_TABLE_SCHEMAS["tear_sheet_trio_ext2026"]
            == bts.TRIO_REPORT_TABLE_SCHEMAS["tear_sheet_trio_ext2026"]
            == "tear_sheet.trio.v4"
        )

    def test_catalog_identities_and_compatibility_match_the_frozen_publisher(self):
        publisher = pytest.importorskip("scripts.publish_finance_remediation")
        from factor_workbook import contract

        catalog = publisher.DATA_V4_CATALOG
        assert contract.DATA_V4_TAG == publisher.RELEASE_TAG
        assert (
            contract.PUBLICATION_MANIFEST_SCHEMA
            == publisher.PUBLICATION_MANIFEST_SCHEMA
        )
        by_basename = {asset.public_basename: asset for asset in catalog.assets}
        for key, spec in contract.DATA_V4_ASSET_SPECS.items():
            if key == "publication_manifest":
                assert spec.asset == catalog.publication_manifest
                continue
            asset = by_basename[spec.asset]
            assert asset.schema_id == contract.DATA_V4_TABLE_SCHEMAS[key], key
        aliases = {
            path: target
            for path, target in catalog.compatibility_paths
            if path != target
        }
        assert dict(contract.DATA_V4_COMPATIBILITY_ALIASES) == aliases


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

    def test_cash_excess_nondefault_metadata_and_verdict_are_deterministic(self):
        portfolio = _synthetic_returns()
        cash = pd.Series(0.0001, index=portfolio.index)
        excess = portfolio - cash
        settings = {
            "window": 126,
            "sr_star": 0.25,
            "n_boot": 40,
            "seed": 7,
            "alpha": 0.10,
        }

        first = vendored_ssr.ssr_inference(excess, **settings)
        second = vendored_ssr.ssr_inference(excess, **settings)

        assert first == second
        assert first.verdict() == second.verdict()
        assert first.window == 126
        assert first.periods_per_year == vendored_ssr.TRADING_DAYS
        assert first.sr_star == 0.25
        assert first.n_boot == 40
        assert first.seed == 7
        assert first.alpha == 0.10
        assert first.block_len >= 1
        expected_sr = excess.mean() / excess.std(ddof=1) * np.sqrt(vendored_ssr.TRADING_DAYS)
        assert first.result.sr_full == pytest.approx(expected_sr)

    def test_exactly_ten_rolling_observations_runs_inference(self):
        returns = _synthetic_returns(n=vendored_ssr.TRADING_DAYS + 9)
        inference = vendored_ssr.ssr_inference(returns, n_boot=10)
        assert inference.result.n_rolling == 10
        assert inference.block_len >= 1
        assert math.isfinite(inference.p_value)
        assert math.isfinite(inference.p_value_lower)

    @pytest.mark.parametrize(
        ("settings", "message"),
        [
            ({"alpha": 0.0}, "alpha"),
            ({"n_boot": True}, "n_boot"),
            ({"window": 1}, "window"),
            ({"window": 601}, "window"),
            ({"seed": True}, "seed"),
            ({"sr_star": np.inf}, "sr_star"),
        ],
    )
    def test_invalid_inference_settings_are_rejected(self, settings, message):
        kwargs = {"n_boot": 10, **settings}
        with pytest.raises(ValueError, match=message):
            vendored_ssr.ssr_inference(_synthetic_returns(), **kwargs)

    def test_provenance_header_names_source_and_resync_procedure(self):
        source = Path(vendored_ssr.__file__).read_text()
        assert "macro_framework/ssr.py" in source
        assert "VERBATIM" in source
        assert "re-sync" in source
