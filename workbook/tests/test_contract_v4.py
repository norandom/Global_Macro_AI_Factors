"""Tag-aware ``data-v4`` workbook schema registry (task 10.3).

The corrected canonical tables published under the immutable ``data-v4`` tag
get their own registry (``DATA_V4_ASSET_SPECS``) bound explicitly to that tag:
canonical reader, legacy, differential, attribution, crisis, Factor, SJM,
Markowitz, manifest, and compatibility schemas. Historical ``data-v1``-``v3``
contracts in ``ASSET_SPECS`` stay byte-unchanged, cross-tag substitution fails
with field-specific errors, and every fixture here is a compact schema-true
construction — never a copied release payload.
"""

import io
import json

import pandas as pd
import pytest

from factor_workbook import contract
from factor_workbook.contract import (
    ASSET_SPECS,
    DATA_V4_ASSET_SPECS,
    DATA_V4_COMPATIBILITY_ALIASES,
    DATA_V4_TABLE_SCHEMAS,
    SchemaError,
    load_frame,
    load_v4_frame,
    load_v4_json,
)
from factor_workbook.release import Provenance

MARKOWITZ_UNIVERSE = ("SWDA.L", "XLK", "IAU", "BIL")


class FakeV4Client:
    """ReleaseClient stand-in serving in-memory data-v4 payload bytes."""

    def __init__(self, payloads: dict[str, bytes], tag: str = "data-v4"):
        self.tag = tag
        self._payloads = payloads

    def fetch(self, asset: str) -> tuple[bytes, Provenance]:
        provenance = Provenance(
            tag=self.tag,
            asset=asset,
            url=f"fixture://{asset}",
            fetched_at="2026-07-29T00:00:00+00:00",
            sha256="0" * 64,
            from_cache=False,
        )
        return self._payloads[asset], provenance


def parquet_bytes(rows: list[dict]) -> bytes:
    buf = io.BytesIO()
    pd.DataFrame(rows).to_parquet(buf)
    return buf.getvalue()


def series_bytes(columns: dict[str, list[float]]) -> bytes:
    frame = pd.DataFrame(
        columns, index=pd.bdate_range("2026-01-05", periods=3, name="Date")
    )
    buf = io.BytesIO()
    frame.to_parquet(buf)
    return buf.getvalue()


# --- compact schema-true report rows ----------------------------------------


def provenance_row(schema: str, portfolio_id: str, *, ppy: int = 252) -> dict:
    return {
        "schema": schema,
        "portfolio_id": portfolio_id,
        "return_basis": "portfolio_value_curve",
        "window_label": "full 2026-01-05..2026-01-07",
        "start": pd.Timestamp("2026-01-05"),
        "end": pd.Timestamp("2026-01-07"),
        "n_obs": 3,
        "periods_per_year": ppy,
        "cash_benchmark_id": "BIL@snapshot_2026",
        "currency_basis": "legacy_mixed_local_quotes",
        "source": "factor_run:run_1/factor_equity_ext2026.parquet#" + "a" * 64,
    }


SSR_VALUES = {
    "ssr_n_obs": 3,
    "ssr_n_rolling": 1,
    "ssr_sr_full": 0.5,
    "ssr_mean_rolling_sr": 0.4,
    "ssr_sigma_hac": 0.2,
    "ssr_L_hac": 1,
    "ssr_ssr": 2.0,
    "ssr_sr_star": 0.0,
    "ssr_p_value": 0.2,
    "ssr_block_len": 2,
    "ssr_n_boot": 1000,
    "ssr_seed": 0,
    "ssr_alpha": 0.05,
    "ssr_p_value_lower": 0.8,
    "ssr_window": 252,
    "ssr_periods_per_year": 252,
}

ATTRIBUTION_VALUES = {
    "raw_market_model_kind": "raw_market_model",
    "raw_market_model_intercept_native_period": 0.0001,
    "raw_market_model_intercept_ann_arithmetic": 0.0252,
    "raw_market_model_intercept_se_hac": 0.001,
    "raw_market_model_intercept_t_hac": 0.1,
    "raw_market_model_beta": 0.9,
    "raw_market_model_r2": 0.8,
    "raw_market_model_n_obs": 3,
    "raw_market_model_start": pd.Timestamp("2026-01-05"),
    "raw_market_model_end": pd.Timestamp("2026-01-07"),
    "raw_market_model_periods_per_year": 252,
    "raw_market_model_hac_maxlags": 1,
}

READER_METRICS = {
    metric: 0.01
    for metric in (
        "total_return",
        "maxdd",
        "downside_rms",
        "cagr",
        "ann_vol",
        "sharpe",
        "sortino",
        "calmar",
    )
}


def reader_row(portfolio_id: str = "factor_pit_ext2026", **overrides) -> dict:
    row = provenance_row(contract.READER_SCHEMA, portfolio_id)
    row["row_kind"] = "full"
    row.update(READER_METRICS)
    row.update(SSR_VALUES)
    row.update(ATTRIBUTION_VALUES)
    row.update(overrides)
    return row


def legacy_row(portfolio_id: str = "factor_pit_ext2026", **overrides) -> dict:
    row = provenance_row(contract.LEGACY_SCHEMA, portfolio_id, ppy=365)
    row.update(
        {
            metric: 0.01
            for metric in (
                "total_return",
                "maxdd",
                "downside_rms",
                "cagr_rows",
                "ann_vol_cal",
                "sharpe_cal",
                "sortino_cal",
                "calmar_rows",
            )
        }
    )
    row.update(overrides)
    return row


def attribution_row(portfolio_id: str = "factor_pit_ext2026", **overrides) -> dict:
    row = provenance_row(contract.ATTRIBUTION_SCHEMA, portfolio_id)
    row.update(ATTRIBUTION_VALUES)
    row.update(overrides)
    return row


def crisis_row(portfolio_id: str = "factor_pit_ext2026", **overrides) -> dict:
    row = provenance_row(contract.CRISIS_SCHEMA, portfolio_id)
    row.update(
        {
            "requested_start": pd.Timestamp("2022-01-01"),
            "requested_end": pd.Timestamp("2022-12-31"),
            "anchor": pd.Timestamp("2021-12-31"),
            "first_return_date": pd.Timestamp("2022-01-03"),
            "actual_end": pd.Timestamp("2022-12-30"),
            "episode_return": -0.1,
            "boundary_anchored_max_drawdown": -0.2,
            "volatility_ann": 0.15,
            "n_returns": 250,
        }
    )
    row.update(overrides)
    return row


def markowitz_identity(window: str) -> dict:
    return {
        "window": window,
        "snapshot_id": "snapshot_2026",
        "base_currency": "USD",
        "valuation_rule": "friday_close_last_observation",
        "requested_start": pd.Timestamp("2016-01-01"),
        "requested_end": pd.Timestamp("2026-01-01"),
        "actual_start": pd.Timestamp("2016-01-08"),
        "actual_end": pd.Timestamp("2025-12-26"),
        "n_obs": 520,
        "periods_per_year": 365.2425 / 7,
        "source_dates_sha256": "b" * 64,
    }


def moments_row(window: str = "10y", **overrides) -> dict:
    row = markowitz_identity(window)
    row.update(
        {
            "asset": "SWDA.L",
            "quote_currency": "GBP",
            "quote_unit": "major",
            "mean_ann_arithmetic": 0.08,
            "vol_ann": 0.15,
        }
    )
    row.update({f"cov_{asset}": 0.01 for asset in MARKOWITZ_UNIVERSE})
    row.update(overrides)
    return row


def frontier_row(window: str = "10y", **overrides) -> dict:
    row = markowitz_identity(window)
    row.update(
        {
            "residual_tolerance": 1e-8,
            "target_return_ann": 0.06,
            "success": True,
            "status": 0,
            "message": "Optimization terminated successfully",
            "iterations": 7,
            "objective": 0.012,
            "budget_residual": 0.0,
            "target_residual": 0.0,
            "bound_violation": 0.0,
            "return_ann": 0.06,
            "volatility_ann": 0.11,
            "feasible": True,
        }
    )
    row.update({f"weight_{asset}": 0.25 for asset in MARKOWITZ_UNIVERSE})
    row.update(overrides)
    return row


def manifest_payload(**overrides) -> bytes:
    manifest = {
        "schema": "publication_manifest.v1",
        "schema_id": "publication_manifest.v1",
        "release_tag": "data-v4",
        "publication_id": "data-v4_pub_1",
        "build_time": "2026-07-29T00:00:00+00:00",
        "catalog_sha256": "c" * 64,
        "artifacts": [
            {"path": "factor_equity_ext2026.parquet", "sha256": "d" * 64, "size": 10}
        ],
        "assets": {"factor_equity_ext2026.parquet": {"sha256": "d" * 64}},
        "input_manifests": {"factor_run": {"schema": "factor_run.v1"}},
        "compatibility_paths": [
            {"public_path": "tear_sheet_ext2026.csv",
             "target": "portfolio_metrics_reader_ext2026.csv"}
        ],
        "completed": True,
    }
    manifest.update(overrides)
    return json.dumps(manifest).encode()


def reader_client(*rows) -> FakeV4Client:
    payload = parquet_bytes(list(rows) or [reader_row(), reader_row("factor_nonpit_diagnostic_ext2026")])
    return FakeV4Client({"portfolio_metrics_reader_ext2026.parquet": payload})


# --- registry identity -------------------------------------------------------


def test_v4_registry_mirrors_the_frozen_catalog_identities():
    """Every named family is registered under its exact producer schema id."""
    expected = {
        "portfolio_metrics_reader_ext2026": "portfolio_metrics.reader.v2",
        "portfolio_metrics_vectorbt365_ext2026": "portfolio_metrics.vectorbt365.v1",
        "portfolio_metrics_differential_ext2026": "portfolio_metrics.differential.v2",
        "attribution_raw_market_model_ext2026": "attribution.raw_market_model.v1",
        "crisis_metrics_ext2026": "crisis_metrics.boundary_anchored.v1",
        "monthly_returns_ext2026": "monthly_returns.reader.v1",
        "risk_decomposition_ext2026": "risk_decomposition.v1",
        "tear_sheet_ai_variants_ext2026": "tear_sheet.ai_variants.v1",
        "tear_sheet_sjm_crowding_ext2026": "tear_sheet.sjm.v3",
        "tear_sheet_trio_ext2026": "tear_sheet.trio.v4",
        "markowitz_10y_moments": "markowitz.moments.v1",
        "markowitz_10y_frontier": "markowitz.frontier.v1",
        "markowitz_max_moments": "markowitz.moments.v1",
        "markowitz_max_frontier": "markowitz.frontier.v1",
        "factor_equity_ext2026": "factor.equity.v1",
        "factor_targets_ext2026": "factor.targets.v1",
        "factor_nonpit_diagnostic_equity_ext2026": "factor.equity.v1",
        "factor_nonpit_diagnostic_targets_ext2026": "factor.targets.v1",
        "sjm_crowding_v3_total_return_bil_equity_ext2026": "sjm.equity.v3",
        "sjm_crowding_v3_total_return_bil_targets_ext2026": "sjm.targets.v3",
        "sjm_crowding_v3_total_return_bil_daily_returns_ext2026": "sjm.daily_returns.v3",
        "sjm_crowding_v3_total_return_bil_control_returns_ext2026": "sjm.control_returns.v3",
        "sjm_crowding_derisk_equity_ext2026": "sjm.equity.v3",
        "publication_manifest": "publication_manifest.v1",
    }
    assert DATA_V4_TABLE_SCHEMAS == expected
    assert set(DATA_V4_ASSET_SPECS) == set(expected)


def test_v4_keys_stay_disjoint_and_historical_contracts_stay_unchanged():
    assert not set(ASSET_SPECS) & set(DATA_V4_ASSET_SPECS)
    # spot-anchor the historical registry: the immutable data-v2 tolerances
    assert ASSET_SPECS["factor_luck_vs_skill_v1"].optional == {"mbb_p", "mbb_block"}
    assert ASSET_SPECS["factor_loadings_v1"].columns["inflation"] == "float64"


def test_locale_specs_mirror_the_report_producer():
    assert dict(contract.REPORT_CSV_LOCALE_SPECS["en-US"]) == {
        "sep": ",", "decimal": ".", "float_format": "%.8f", "encoding": "utf-8"
    }
    assert dict(contract.REPORT_CSV_LOCALE_SPECS["de-DE"]) == {
        "sep": ";", "decimal": ",", "float_format": "%.8f", "encoding": "utf-8"
    }


def test_compatibility_aliases_mirror_the_catalog_pairs():
    assert DATA_V4_COMPATIBILITY_ALIASES == {
        "nb16_ai_variants_tearsheet.csv": "tear_sheet_ai_variants_ext2026.csv",
        "nb16_ai_variants_tearsheet_de.csv": "tear_sheet_ai_variants_ext2026_de.csv",
        "nb17_sjm_crowding_tearsheet.csv": "tear_sheet_sjm_crowding_ext2026.csv",
        "nb17_sjm_crowding_tearsheet_de.csv": "tear_sheet_sjm_crowding_ext2026_de.csv",
        "sjm_crowding_derisk_equity_ext2026.parquet":
            "sjm_crowding_v3_total_return_bil_equity_ext2026.parquet",
        "sjm_crowding_derisk_equity_ext2026.csv":
            "sjm_crowding_v3_total_return_bil_equity_ext2026.csv",
        "sjm_crowding_derisk_equity_ext2026_de.csv":
            "sjm_crowding_v3_total_return_bil_equity_ext2026_de.csv",
        "tear_sheet_ext2026.csv": "portfolio_metrics_reader_ext2026.csv",
        "tear_sheet_ext2026_de.csv": "portfolio_metrics_reader_ext2026_de.csv",
    }
    alias = DATA_V4_ASSET_SPECS["sjm_crowding_derisk_equity_ext2026"]
    target = DATA_V4_ASSET_SPECS["sjm_crowding_v3_total_return_bil_equity_ext2026"]
    assert alias.asset == "sjm_crowding_derisk_equity_ext2026.parquet"
    assert alias.columns == target.columns and alias.index == target.index


# --- tag binding: cross-tag substitution fails --------------------------------


def test_v4_key_refuses_a_historical_client():
    client = reader_client()
    client.tag = "data-v2"
    with pytest.raises(SchemaError) as exc:
        load_v4_frame(client, "portfolio_metrics_reader_ext2026")
    message = str(exc.value)
    assert "portfolio_metrics_reader_ext2026" in message
    assert "data-v4" in message and "data-v2" in message


def test_historical_key_refuses_a_data_v4_client():
    class V4Tagged(FakeV4Client):
        pass

    with pytest.raises(SchemaError) as exc:
        load_frame(V4Tagged({}, tag="data-v4"), "factor_loadings_v1")
    message = str(exc.value)
    assert "factor_loadings_v1" in message and "data-v4" in message


def test_cross_tag_payload_substitution_names_the_missing_field():
    """A legacy-shaped table served under the reader key fails on the exact
    reader-only column, not with a generic error."""
    payload = parquet_bytes([legacy_row()])
    client = FakeV4Client({"portfolio_metrics_reader_ext2026.parquet": payload})
    with pytest.raises(SchemaError) as exc:
        load_v4_frame(client, "portfolio_metrics_reader_ext2026")
    message = str(exc.value)
    assert "portfolio_metrics_reader_ext2026.parquet" in message
    assert "row_kind" in message


def test_wrong_row_schema_identity_fails_field_specifically():
    rows = [reader_row(schema="portfolio_metrics.vectorbt365.v1")]
    client = reader_client(*rows)
    with pytest.raises(SchemaError) as exc:
        load_v4_frame(client, "portfolio_metrics_reader_ext2026")
    message = str(exc.value)
    assert "'schema'" in message
    assert "portfolio_metrics.vectorbt365.v1" in message


# --- canonical fixtures validate ---------------------------------------------


def test_v4_reader_table_validates_and_returns_provenance():
    df, provenance = load_v4_frame(reader_client(), "portfolio_metrics_reader_ext2026")
    assert len(df) == 2
    assert provenance.asset == "portfolio_metrics_reader_ext2026.parquet"
    assert set(df.columns) == set(
        DATA_V4_ASSET_SPECS["portfolio_metrics_reader_ext2026"].columns
    )


def test_v4_legacy_table_pins_the_365_convention():
    payload = parquet_bytes([legacy_row()])
    client = FakeV4Client({"portfolio_metrics_vectorbt365_ext2026.parquet": payload})
    df, _ = load_v4_frame(client, "portfolio_metrics_vectorbt365_ext2026")
    assert list(df["periods_per_year"]) == [365]

    bad = parquet_bytes([legacy_row(periods_per_year=252)])
    client = FakeV4Client({"portfolio_metrics_vectorbt365_ext2026.parquet": bad})
    with pytest.raises(SchemaError, match="periods_per_year"):
        load_v4_frame(client, "portfolio_metrics_vectorbt365_ext2026")


def test_sjm_tear_sheet_carries_reader_attribution_and_crisis_rows():
    payload = parquet_bytes(
        [reader_row("sjm_run_1"), attribution_row("sjm_run_1"), crisis_row("sjm_run_1")]
    )
    client = FakeV4Client({"tear_sheet_sjm_crowding_ext2026.parquet": payload})
    df, _ = load_v4_frame(client, "tear_sheet_sjm_crowding_ext2026")
    assert list(df["schema"]) == [
        "portfolio_metrics.reader.v2",
        "attribution.raw_market_model.v1",
        "crisis_metrics.boundary_anchored.v1",
    ]


def test_trio_v4_accepts_only_reader_rows():
    good = parquet_bytes([reader_row("factor_pit_ext2026"), reader_row("sjm_run_1")])
    client = FakeV4Client({"tear_sheet_trio_ext2026.parquet": good})
    load_v4_frame(client, "tear_sheet_trio_ext2026")

    # a structurally reader-shaped row claiming another identity is caught by
    # the row-schema gate; a shape-divergent smuggle (e.g. a real crisis row)
    # already fails the structural column contract with its own field error
    smuggled = parquet_bytes(
        [reader_row(), reader_row(schema="crisis_metrics.boundary_anchored.v1")]
    )
    client = FakeV4Client({"tear_sheet_trio_ext2026.parquet": smuggled})
    with pytest.raises(SchemaError) as exc:
        load_v4_frame(client, "tear_sheet_trio_ext2026")
    assert "crisis_metrics.boundary_anchored.v1" in str(exc.value)
    assert "'schema'" in str(exc.value)


# --- field-specific semantic validation ---------------------------------------


@pytest.mark.parametrize(
    ("override", "column"),
    [
        ({"periods_per_year": 365}, "periods_per_year"),
        ({"start": pd.Timestamp("2026-02-01")}, "start"),
        ({"n_obs": 0}, "n_obs"),
        ({"cash_benchmark_id": "  "}, "cash_benchmark_id"),
        ({"currency_basis": "GBP"}, "currency_basis"),
        ({"ssr_n_boot": 500}, "ssr_n_boot"),
        ({"ssr_window": 126}, "ssr_window"),
        ({"ssr_alpha": 0.10}, "ssr_alpha"),
    ],
)
def test_reader_semantics_fail_with_the_exact_field(override, column):
    client = reader_client(reader_row(**override))
    with pytest.raises(SchemaError) as exc:
        load_v4_frame(client, "portfolio_metrics_reader_ext2026")
    assert column in str(exc.value)


def test_markowitz_moments_validate_and_pin_window_and_currency():
    payload = parquet_bytes(
        [moments_row(asset=asset) for asset in MARKOWITZ_UNIVERSE]
    )
    client = FakeV4Client({"markowitz_10y_moments.parquet": payload})
    df, _ = load_v4_frame(client, "markowitz_10y_moments")
    assert list(df["asset"]) == list(MARKOWITZ_UNIVERSE)

    for override, column in (
        ({"window": "max"}, "window"),
        ({"base_currency": "GBP"}, "base_currency"),
        ({"periods_per_year": 52.0}, "periods_per_year"),
        ({"actual_start": pd.Timestamp("2015-12-01")}, "actual_start"),
    ):
        bad = parquet_bytes([moments_row(**override)])
        client = FakeV4Client({"markowitz_10y_moments.parquet": bad})
        with pytest.raises(SchemaError) as exc:
            load_v4_frame(client, "markowitz_10y_moments")
        assert column in str(exc.value)


def test_markowitz_frontier_validates_with_full_solver_diagnostics():
    payload = parquet_bytes([frontier_row("max")])
    client = FakeV4Client({"markowitz_max_frontier.parquet": payload})
    df, _ = load_v4_frame(client, "markowitz_max_frontier")
    assert {f"weight_{asset}" for asset in MARKOWITZ_UNIVERSE} <= set(df.columns)
    assert bool(df["feasible"].iloc[0]) is True


def test_sjm_strategy_and_factor_tables_validate_structurally():
    client = FakeV4Client(
        {
            "factor_equity_ext2026.parquet": series_bytes({"value": [1.0, 1.01, 1.02]}),
            "sjm_crowding_v3_total_return_bil_daily_returns_ext2026.parquet": series_bytes(
                {
                    "daily_return": [0.0, 0.001, -0.002],
                    "factor_return": [0.0, 0.002, -0.001],
                    "cash_return": [0.0001, 0.0001, 0.0001],
                }
            ),
        }
    )
    equity, _ = load_v4_frame(client, "factor_equity_ext2026")
    assert equity.index.name == "Date"
    returns, _ = load_v4_frame(
        client, "sjm_crowding_v3_total_return_bil_daily_returns_ext2026"
    )
    assert list(returns.columns) == ["daily_return", "factor_return", "cash_return"]


# --- manifest schema -----------------------------------------------------------


def test_publication_manifest_validates_and_pins_identity():
    client = FakeV4Client({"publication_manifest.json": manifest_payload()})
    manifest, provenance = load_v4_json(client, "publication_manifest")
    assert manifest["release_tag"] == "data-v4"
    assert provenance.asset == "publication_manifest.json"

    for override, column in (
        ({"completed": False}, "completed"),
        ({"release_tag": "data-v2"}, "release_tag"),
        ({"schema_id": "publication_manifest.v0"}, "schema_id"),
    ):
        client = FakeV4Client(
            {"publication_manifest.json": manifest_payload(**override)}
        )
        with pytest.raises(SchemaError) as exc:
            load_v4_json(client, "publication_manifest")
        assert column in str(exc.value)


def test_v4_loader_kind_discipline():
    with pytest.raises(ValueError):
        load_v4_frame(FakeV4Client({}), "publication_manifest")
    with pytest.raises(ValueError):
        load_v4_json(FakeV4Client({}), "factor_equity_ext2026")
    with pytest.raises(KeyError):
        load_v4_frame(FakeV4Client({}), "no_such_key")
