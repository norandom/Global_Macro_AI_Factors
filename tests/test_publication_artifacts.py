"""Artifact-boundary checks: published rows disclose their conventions (R7).

``macro_framework.reporting.validate_report_row`` is the producer-side gate;
these tests pin the artifact-facing consequences the coverage matrix assigns to
it: a deliberate legacy convention is identified on the emitted row itself
(7.3), and a raw-on-raw regression can never surface under a CAPM/Jensen
label (defect 7).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from macro_framework.reporting import LEGACY_SCHEMA, validate_report_row


def _legacy_artifact_row() -> dict:
    return {
        "schema": LEGACY_SCHEMA,
        "portfolio_id": "factor_v1_2016_2025",
        "return_basis": "vectorbt_total_return",
        "window_label": "2016-01-04..2025-12-31",
        "start": "2016-01-04",
        "end": "2025-12-31",
        "n_obs": 2515,
        "periods_per_year": 365,
        "cash_benchmark_id": "BIL",
        "currency_basis": "legacy_mixed_local_quotes",
        "source": "release data-v2 tear_sheet.csv",
        "cagr_rows": 0.123,
        "sharpe_cal": 1.31,
    }


def test_legacy_convention_is_identified_on_the_emitted_row():
    row = validate_report_row(_legacy_artifact_row())
    assert row["schema"] == "portfolio_metrics.vectorbt365.v1"  # names the convention
    assert row["periods_per_year"] == 365
    # stripping the identification blocks emission entirely
    for key in ("schema", "periods_per_year", "return_basis", "source"):
        broken = _legacy_artifact_row()
        del broken[key]
        with pytest.raises(ValueError):
            validate_report_row(broken)


def test_ac_7_3():
    test_legacy_convention_is_identified_on_the_emitted_row()


def test_reports_reject_capm_labels_for_raw_regression():
    with pytest.raises(ValueError, match="raw market-model"):
        validate_report_row(_legacy_artifact_row() | {"capm_alpha_ann": 0.02})


def test_reader_outputs_disclose_annualization_basis():
    # defect 2 downstream boundary, fixture class: stale_artifacts — business-daily
    # metrics annualized on 365 under generic fields must be unemittable
    from macro_framework.reporting import build_legacy_metric_row, build_reader_metric_row
    from tests.test_reporting import _line

    meta, _, metrics, cash, excess, ssr, attr, _ = _line()
    reader = build_reader_metric_row(meta, metrics, cash, ssr, source="t", attribution=attr)
    legacy = build_legacy_metric_row(meta, metrics, source="t")

    assert reader["schema"] == "portfolio_metrics.reader.v2"
    assert reader["periods_per_year"] == 252
    assert not any(key.endswith(("_cal", "_rows")) for key in reader)
    assert legacy["schema"] == "portfolio_metrics.vectorbt365.v1"  # names the basis
    assert legacy["periods_per_year"] == 365

    # the stale-artifact shape cannot be emitted: 365 basis under the reader schema
    with pytest.raises(ValueError, match="252"):
        validate_report_row(reader | {"periods_per_year": 365})
    with pytest.raises(ValueError, match="not part of"):
        validate_report_row(reader | {"sharpe_cal": legacy["sharpe_cal"]})


def test_differential_artifacts_separate_endpoint_gap():
    # defect 9 downstream boundary: the published differential row labels the
    # endpoint wealth gap distinctly and no single-portfolio schema can carry it
    from macro_framework.reporting import build_differential_metric_row
    from tests.test_reporting import _differential_line

    meta, comparison, reference, spread, ssr = _differential_line()
    row = build_differential_metric_row(meta, comparison, reference, ssr, source="t")

    assert row["schema"] == "portfolio_metrics.differential.v2"
    assert "endpoint_total_return_difference" in row
    assert row["endpoint_total_return_difference"] != pytest.approx(row["total_return"])
    # the legacy luck-table shape — endpoint gap published as the row's
    # total_return next to spread SSR — is structurally unemittable through the
    # single-portfolio schemas
    with pytest.raises(ValueError, match="not part of"):
        validate_report_row(
            _legacy_artifact_row() | {"endpoint_total_return_difference": 0.31}
        )


def test_crisis_exports_reproduce_the_shared_boundary_inclusive_result():
    # R4.5: exported crisis rows equal the shared typed calculation exactly
    import dataclasses as dc

    from macro_framework.evaluation import crisis_metrics
    from macro_framework.reporting import build_crisis_record
    from tests.test_reporting import _line

    meta, value, *_ = _line()
    crisis = crisis_metrics(value, "2016-06-01", "2016-08-31")
    record = build_crisis_record(meta, crisis, source="t")

    assert record["episode_return"] == crisis.episode_return
    assert record["boundary_anchored_max_drawdown"] == crisis.boundary_anchored_max_drawdown
    assert record["volatility_ann"] == crisis.volatility_ann
    assert all(record[f.name] == getattr(crisis, f.name) for f in dc.fields(crisis))


def test_ac_4_5():
    test_crisis_exports_reproduce_the_shared_boundary_inclusive_result()


def test_factor_source_to_consumption_equality_blocks_publish(tmp_path):
    # defect 1 downstream boundary, fixture classes: dated_prompt_collisions,
    # manifest_failures — R6.3/R6.4: a cross-date response/score swap between
    # two rebalance dates sharing an IDENTICAL rendered PIT prompt fails the
    # source-to-consumption audit BEFORE any publishable output (targets,
    # equity, decision logs, metrics, completion state) exists. Offline only.
    import dataclasses
    import inspect

    # tests.test_stream_ext2026 inserts scripts/ on sys.path; _mod() defers the
    # extend_stream_2026 import past collection (test_factor_scoring ordering).
    from tests.test_stream_ext2026 import _consume_both_passes, _mod, _replay_fixture

    ext = _mod()
    macro_state, snapshot, amap, prompt, d1, d2, evidence = _replay_fixture(ext)

    # deliberate cross-date swap consumed through BOTH production passes —
    # prompt matching cannot see it because the rendered prompt is identical
    swapped = {("pit", d1): evidence[("pit", d2)], ("pit", d2): evidence[("pit", d1)]}
    consumed = {}
    _consume_both_passes(ext, swapped, "pit", (d1, d2), macro_state, snapshot, amap,
                         consumed)
    expected = [("pit", d1), ("pit", d2)]
    with pytest.raises(ValueError, match="cross-associated or altered"):
        ext.validate_source_to_consumption(evidence, consumed, expected)

    # a score-only cross-date swap (response kept in place) blocks too
    tampered = ext.with_evidence_id(dataclasses.replace(
        evidence[("pit", d1)],
        score_p_memorized=evidence[("pit", d2)].score_p_memorized))
    consumed_score = {}
    ext.record_consumption(consumed_score, ("pit", d1), tampered)
    ext.record_consumption(consumed_score, ("pit", d2), evidence[("pit", d2)])
    with pytest.raises(ValueError, match="cross-associated or altered"):
        ext.validate_source_to_consumption(evidence, consumed_score, expected)

    # publication boundary (manifest_failures): the audit gate raises before
    # the summary file exists, so the run directory holds NO artifact at all —
    # nothing downstream can claim a passed audit or publish from this run
    out = tmp_path / "run_out"
    with pytest.raises(ValueError, match="cross-associated or altered"):
        ext.write_replay_audit_summary(evidence, consumed, expected, out)
    assert not out.exists() or not any(out.iterdir())

    # and in main() the audit sits after both variant lines and before the
    # first publication write (_dump_line: targets/equity/decision logs, then
    # every metric artifact and the run header)
    src = inspect.getsource(ext.main)
    audit_at = src.index("write_replay_audit_summary")
    assert src.index("run_variant_line") < audit_at < src.index('_dump_line("factor"')


def test_report_parity_rejects_mixed_windows():
    # defect 10 downstream boundary: a published full row whose attribution
    # window differs from performance coverage is unemittable
    from macro_framework.reporting import build_reader_metric_row
    from tests.test_reporting import _line

    meta, _, metrics, cash, excess, ssr, attr, _ = _line()
    full = build_reader_metric_row(meta, metrics, cash, ssr, source="t", attribution=attr)

    with pytest.raises(ValueError, match="mixed windows"):
        validate_report_row(full | {"raw_market_model_end": full["start"]})
    with pytest.raises(ValueError, match="row_kind 'full'"):
        validate_report_row(full | {"row_kind": "performance_only"})


# Task 10.1: the data-v4 release contract is a flat, deterministic public
# inventory.  These declarations are deliberately independent from the
# production catalog so a changed basename cannot make its own regression test
# pass merely by changing the implementation in the same place.
_FACTOR_PARQUET_STEMS = (
    "baseline_equity_ext2026",
    "baseline_targets_ext2026",
    "factor_contrast_ext2026",
    "factor_equity_ext2026",
    "factor_evidence_ext2026",
    "factor_loadings_ext2026",
    "factor_nonpit_diagnostic_equity_ext2026",
    "factor_nonpit_diagnostic_loadings_ext2026",
    "factor_nonpit_diagnostic_scores_ext2026",
    "factor_nonpit_diagnostic_targets_ext2026",
    "factor_scores_ext2026",
    "factor_targets_ext2026",
    "track_b_equity_ext2026",
    "track_b_targets_ext2026",
)
_SJM_PARQUET_STEMS = (
    "sjm_crowding_v3_total_return_bil_control_returns_ext2026",
    "sjm_crowding_v3_total_return_bil_daily_returns_ext2026",
    "sjm_crowding_v3_total_return_bil_equity_ext2026",
    "sjm_crowding_v3_total_return_bil_targets_ext2026",
)
_REPORT_PARQUET_STEMS = (
    "attribution_raw_market_model_ext2026",
    "crisis_metrics_ext2026",
    "markowitz_10y_frontier",
    "markowitz_10y_moments",
    "markowitz_max_frontier",
    "markowitz_max_moments",
    "monthly_returns_ext2026",
    "portfolio_metrics_differential_ext2026",
    "portfolio_metrics_reader_ext2026",
    "portfolio_metrics_vectorbt365_ext2026",
    "risk_decomposition_ext2026",
    "tear_sheet_ai_variants_ext2026",
    "tear_sheet_sjm_crowding_ext2026",
    "tear_sheet_static_bh_window_dashboard",
    "tear_sheet_static_bh_windows",
    "tear_sheet_trio_10y",
    "tear_sheet_trio_ext2026",
    "tear_sheet_trio_max",
)
_TABULAR_JSON_STEMS = (
    "factor_contrast_split_ext2026",
    "factor_decision_log_ext2026",
    "factor_nonpit_diagnostic_decision_log_ext2026",
    "sjm_crowding_v3_total_return_bil_ledger",
)
_NON_TABULAR_JSON = (
    "factor_replay_audit_ext2026.json",
    "sjm_crowding_v3_total_return_bil_protocol.json",
)
_COMPATIBILITY_ALIASES = (
    "nb16_ai_variants_tearsheet.csv",
    "nb16_ai_variants_tearsheet_de.csv",
    "nb17_sjm_crowding_tearsheet.csv",
    "nb17_sjm_crowding_tearsheet_de.csv",
    "sjm_crowding_derisk_equity_ext2026.csv",
    "sjm_crowding_derisk_equity_ext2026.parquet",
    "sjm_crowding_derisk_equity_ext2026_de.csv",
    "tear_sheet_ext2026.csv",
    "tear_sheet_ext2026_de.csv",
)
_FIGURES = (
    "nb15_3_long_window_equity.png",
    "nb15_3_sharpe_vs_window.png",
    "nb18_2_ratio_ladder.png",
    "nb18_2_risk_return_map.png",
    "nb18_3_markowitz_plane_10y.png",
    "nb18_3_panels_10y.png",
    "nb18_4_markowitz_plane_max.png",
    "nb18_4_panels_max.png",
    "nb18_metric_profile.png",
    "nb18_ratio_ladder.png",
    "nb18_risk_return_maps.png",
)
_FORMATTED_REPORTS = (
    "tear_sheet_paper.csv",
    "tear_sheet_paper.md",
    "tear_sheet_paper.tex",
    "tear_sheet_paper_de.csv",
    "tear_sheet_thesis.csv",
    "tear_sheet_thesis.md",
    "tear_sheet_thesis.tex",
    "tear_sheet_thesis_de.csv",
)
_EXPECTED_CATALOG_SHA256 = "5ed26c16b1dd3bba5746c834906d5abf8587803272f26274966594e46a467316"
_EXPECTED_PRODUCER_MANIFESTS = (
    ("canonical_reports", "canonical_reports.v1"),
    ("factor_run", "factor_run.v1"),
    ("market_snapshot", "market_snapshot.v1"),
    ("presentation_outputs", "presentation_outputs.v1"),
    ("sjm_run", "sjm_run.v3"),
)
_EXPECTED_COMPATIBILITY_PATHS = (
    ("baseline_equity_ext2026.parquet", "baseline_equity_ext2026.parquet"),
    ("baseline_targets_ext2026.parquet", "baseline_targets_ext2026.parquet"),
    ("factor_decision_log_ext2026.json", "factor_decision_log_ext2026.json"),
    ("factor_equity_ext2026.parquet", "factor_equity_ext2026.parquet"),
    (
        "factor_nonpit_diagnostic_decision_log_ext2026.json",
        "factor_nonpit_diagnostic_decision_log_ext2026.json",
    ),
    (
        "factor_nonpit_diagnostic_equity_ext2026.parquet",
        "factor_nonpit_diagnostic_equity_ext2026.parquet",
    ),
    (
        "factor_nonpit_diagnostic_targets_ext2026.parquet",
        "factor_nonpit_diagnostic_targets_ext2026.parquet",
    ),
    ("factor_targets_ext2026.parquet", "factor_targets_ext2026.parquet"),
    ("nb16_ai_variants_tearsheet.csv", "tear_sheet_ai_variants_ext2026.csv"),
    (
        "nb16_ai_variants_tearsheet_de.csv",
        "tear_sheet_ai_variants_ext2026_de.csv",
    ),
    ("nb17_sjm_crowding_tearsheet.csv", "tear_sheet_sjm_crowding_ext2026.csv"),
    (
        "nb17_sjm_crowding_tearsheet_de.csv",
        "tear_sheet_sjm_crowding_ext2026_de.csv",
    ),
    ("risk_decomposition_ext2026.csv", "risk_decomposition_ext2026.csv"),
    ("risk_decomposition_ext2026_de.csv", "risk_decomposition_ext2026_de.csv"),
    (
        "sjm_crowding_derisk_equity_ext2026.csv",
        "sjm_crowding_v3_total_return_bil_equity_ext2026.csv",
    ),
    (
        "sjm_crowding_derisk_equity_ext2026.parquet",
        "sjm_crowding_v3_total_return_bil_equity_ext2026.parquet",
    ),
    (
        "sjm_crowding_derisk_equity_ext2026_de.csv",
        "sjm_crowding_v3_total_return_bil_equity_ext2026_de.csv",
    ),
    ("tear_sheet_ext2026.csv", "portfolio_metrics_reader_ext2026.csv"),
    ("tear_sheet_ext2026_de.csv", "portfolio_metrics_reader_ext2026_de.csv"),
    (
        "tear_sheet_static_bh_window_dashboard.csv",
        "tear_sheet_static_bh_window_dashboard.csv",
    ),
    (
        "tear_sheet_static_bh_window_dashboard_de.csv",
        "tear_sheet_static_bh_window_dashboard_de.csv",
    ),
    ("tear_sheet_static_bh_windows.csv", "tear_sheet_static_bh_windows.csv"),
    (
        "tear_sheet_static_bh_windows_de.csv",
        "tear_sheet_static_bh_windows_de.csv",
    ),
    ("tear_sheet_trio_10y.csv", "tear_sheet_trio_10y.csv"),
    ("tear_sheet_trio_10y_de.csv", "tear_sheet_trio_10y_de.csv"),
    ("tear_sheet_trio_ext2026.csv", "tear_sheet_trio_ext2026.csv"),
    ("tear_sheet_trio_ext2026_de.csv", "tear_sheet_trio_ext2026_de.csv"),
    ("tear_sheet_trio_max.csv", "tear_sheet_trio_max.csv"),
    ("tear_sheet_trio_max_de.csv", "tear_sheet_trio_max_de.csv"),
    ("track_b_equity_ext2026.parquet", "track_b_equity_ext2026.parquet"),
    ("track_b_targets_ext2026.parquet", "track_b_targets_ext2026.parquet"),
)


def _mirror_names(stem: str, canonical_suffix: str) -> tuple[str, str, str]:
    return (f"{stem}{canonical_suffix}", f"{stem}.csv", f"{stem}_de.csv")


def _expected_data_v4_payloads() -> tuple[str, ...]:
    names: list[str] = []
    for stem in _FACTOR_PARQUET_STEMS + _SJM_PARQUET_STEMS + _REPORT_PARQUET_STEMS:
        names.extend(_mirror_names(stem, ".parquet"))
    for stem in _TABULAR_JSON_STEMS:
        names.extend(_mirror_names(stem, ".json"))
    names.extend(_NON_TABULAR_JSON)
    names.extend(_COMPATIBILITY_ALIASES)
    names.extend(_FIGURES)
    names.extend(_FORMATTED_REPORTS)
    return tuple(sorted(names))


def _publisher():
    import importlib

    return importlib.import_module("scripts.publish_finance_remediation")


def _catalog_inputs(pub, catalog):
    return tuple(
        pub.CatalogAssetInput(
            public_basename=asset.public_basename,
            source_artifact=asset.source_artifact,
            producer_manifest_role=asset.producer_manifest_role,
        )
        for asset in catalog.assets
    )


def _replace_catalog_asset(pub, catalog, public_basename: str, **changes):
    import dataclasses

    return dataclasses.replace(
        catalog,
        assets=tuple(
            dataclasses.replace(asset, **changes)
            if asset.public_basename == public_basename
            else asset
            for asset in catalog.assets
        ),
    )


def test_data_v4_catalog_freezes_the_exact_collision_free_payload_inventory():
    pub = _publisher()
    catalog = pub.DATA_V4_CATALOG

    assert catalog.release_tag == "data-v4"
    assert catalog.schema_id == "publication_asset_catalog.v1"
    assert catalog.payload_basenames == _expected_data_v4_payloads()
    assert len(catalog.payload_basenames) == len(
        {name.casefold() for name in catalog.payload_basenames}
    )
    assert all("/" not in name and "\\" not in name for name in catalog.payload_basenames)
    assert all("current" not in name.casefold() for name in catalog.payload_basenames)
    assert pub.build_data_v4_catalog() == catalog
    assert pub.catalog_sha256(pub.build_data_v4_catalog()) == _EXPECTED_CATALOG_SHA256
    assert pub.catalog_sha256(catalog) == _EXPECTED_CATALOG_SHA256


def test_catalog_pins_exact_producer_manifests_and_compatibility_path_order():
    pub = _publisher()
    catalog = pub.DATA_V4_CATALOG

    assert tuple(
        (item.role, item.schema_id) for item in catalog.producer_manifests
    ) == _EXPECTED_PRODUCER_MANIFESTS
    assert catalog.compatibility_paths == _EXPECTED_COMPATIBILITY_PATHS


def test_catalog_defines_schemas_locales_producers_lineage_and_exact_projections():
    pub = _publisher()
    catalog = pub.DATA_V4_CATALOG
    by_name = {asset.public_basename: asset for asset in catalog.assets}

    for asset in catalog.assets:
        assert asset.schema_id
        assert asset.media_type
        assert asset.producer
        assert asset.producer_manifest_role in catalog.producer_manifest_map
        assert asset.lineage[-1] == asset.producer_manifest_role
        assert len(asset.lineage) == len(set(asset.lineage))

    for stem in _FACTOR_PARQUET_STEMS + _SJM_PARQUET_STEMS + _REPORT_PARQUET_STEMS:
        canonical, us_csv, de_csv = _mirror_names(stem, ".parquet")
        assert by_name[canonical].locale == "und"
        assert by_name[canonical].required_projections == (us_csv, de_csv)
        assert by_name[us_csv].projection == "csv_us"
        assert by_name[us_csv].projection_of == canonical
        assert by_name[us_csv].locale == "en-US"
        assert by_name[de_csv].projection == "csv_de"
        assert by_name[de_csv].projection_of == canonical
        assert by_name[de_csv].locale == "de-DE"

    for stem in _TABULAR_JSON_STEMS:
        canonical, us_csv, de_csv = _mirror_names(stem, ".json")
        assert by_name[canonical].required_projections == (us_csv, de_csv)

    assert by_name["factor_evidence_ext2026.parquet"].producer == (
        "scripts/extend_stream_2026.py"
    )
    assert by_name["sjm_crowding_v3_total_return_bil_equity_ext2026.parquet"].lineage == (
        "market_snapshot",
        "factor_run",
        "sjm_run",
    )
    assert by_name["markowitz_10y_frontier.parquet"].lineage == (
        "market_snapshot",
        "canonical_reports",
    )
    assert by_name["nb18_3_markowitz_plane_10y.png"].projection_of == (
        "markowitz_10y_frontier.parquet"
    )
    assert by_name["tear_sheet_paper.md"].projection_of == (
        "tear_sheet_trio_ext2026.parquet"
    )


def test_catalog_rejects_a_self_consistent_missing_required_asset_family():
    import dataclasses

    pub = _publisher()
    catalog = pub.DATA_V4_CATALOG
    removed = {
        "factor_evidence_ext2026.parquet",
        "factor_evidence_ext2026.csv",
        "factor_evidence_ext2026_de.csv",
    }
    weakened = dataclasses.replace(
        catalog,
        assets=tuple(
            asset for asset in catalog.assets if asset.public_basename not in removed
        ),
    )

    with pytest.raises(ValueError, match="frozen asset contract"):
        pub.validate_asset_catalog(weakened)
    with pytest.raises(ValueError, match="frozen asset contract"):
        pub.validate_catalog_inputs(_catalog_inputs(pub, weakened), catalog=weakened)
    with pytest.raises(ValueError, match="frozen asset contract"):
        pub.catalog_sha256(weakened)


def test_catalog_rejects_a_self_consistent_undeclared_asset_addition():
    import dataclasses

    pub = _publisher()
    catalog = pub.DATA_V4_CATALOG
    source = next(
        asset
        for asset in catalog.assets
        if asset.public_basename == "factor_replay_audit_ext2026.json"
    )
    extra = dataclasses.replace(
        source,
        public_basename="factor_replay_audit_ext2026_copy.json",
        source_artifact="artifacts/factor_replay_audit_ext2026_copy.json",
    )
    weakened = dataclasses.replace(
        catalog,
        assets=tuple(sorted(catalog.assets + (extra,), key=lambda item: item.public_basename)),
    )

    with pytest.raises(ValueError, match="frozen asset contract"):
        pub.validate_asset_catalog(weakened)
    with pytest.raises(ValueError, match="frozen asset contract"):
        pub.validate_catalog_inputs(_catalog_inputs(pub, weakened), catalog=weakened)


@pytest.mark.parametrize(
    ("public_basename", "changes"),
    (
        (
            "factor_evidence_ext2026.parquet",
            {"source_artifact": "artifacts/factor_evidence_ext2026_v2.parquet"},
        ),
        ("factor_evidence_ext2026.parquet", {"asset_class": "projection"}),
        (
            "factor_evidence_ext2026.parquet",
            {"media_type": "application/octet-stream"},
        ),
        ("factor_evidence_ext2026.parquet", {"schema_id": "wrong.schema.v999"}),
        ("factor_evidence_ext2026.parquet", {"locale": "en-US"}),
        ("factor_evidence_ext2026.parquet", {"producer": "scripts/wrong.py"}),
        (
            "factor_replay_audit_ext2026.json",
            {
                "producer_manifest_role": "canonical_reports",
                "lineage": ("market_snapshot", "factor_run", "canonical_reports"),
            },
        ),
        ("factor_evidence_ext2026.parquet", {"lineage": ("factor_run",)}),
        ("factor_evidence_ext2026.csv", {"projection": "csv_unapproved"}),
        ("factor_evidence_ext2026.parquet", {"required_projections": ()}),
    ),
)
def test_catalog_rejects_every_exact_asset_metadata_dimension(
    public_basename, changes
):
    pub = _publisher()
    weakened = _replace_catalog_asset(
        pub,
        pub.DATA_V4_CATALOG,
        public_basename,
        **changes,
    )

    with pytest.raises(ValueError, match="frozen asset contract"):
        pub.validate_asset_catalog(weakened)
    with pytest.raises(ValueError, match="frozen asset contract"):
        pub.catalog_sha256(weakened)


def test_catalog_input_gate_cannot_accept_corrupt_schema_or_producer_metadata():
    pub = _publisher()
    catalog = pub.DATA_V4_CATALOG
    for changes in (
        {"schema_id": "wrong.schema.v999"},
        {"producer": "scripts/wrong.py"},
    ):
        weakened = _replace_catalog_asset(
            pub,
            catalog,
            "factor_evidence_ext2026.parquet",
            **changes,
        )
        with pytest.raises(ValueError, match="frozen asset contract"):
            pub.validate_catalog_inputs(_catalog_inputs(pub, weakened), catalog=weakened)


def test_catalog_rejects_self_consistent_projection_retargeting():
    import dataclasses

    pub = _publisher()
    catalog = pub.DATA_V4_CATALOG
    projection_name = "factor_evidence_ext2026.csv"
    old_target = "factor_evidence_ext2026.parquet"
    new_target = "factor_loadings_ext2026.parquet"
    assets = []
    for asset in catalog.assets:
        if asset.public_basename == projection_name:
            asset = dataclasses.replace(asset, projection_of=new_target)
        elif asset.public_basename == old_target:
            asset = dataclasses.replace(
                asset,
                allowed_projections=tuple(
                    name for name in asset.allowed_projections if name != projection_name
                ),
                required_projections=tuple(
                    name for name in asset.required_projections if name != projection_name
                ),
            )
        elif asset.public_basename == new_target:
            asset = dataclasses.replace(
                asset,
                allowed_projections=tuple(
                    sorted(asset.allowed_projections + (projection_name,))
                ),
            )
        assets.append(asset)
    weakened = dataclasses.replace(catalog, assets=tuple(assets))

    with pytest.raises(ValueError, match="frozen asset contract"):
        pub.validate_asset_catalog(weakened)


def test_catalog_rejects_duplicate_and_reordered_producer_manifest_declarations():
    import dataclasses

    pub = _publisher()
    catalog = pub.DATA_V4_CATALOG
    variants = (
        dataclasses.replace(
            catalog,
            producer_manifests=catalog.producer_manifests
            + (catalog.producer_manifests[0],),
        ),
        dataclasses.replace(
            catalog,
            producer_manifests=tuple(reversed(catalog.producer_manifests)),
        ),
    )

    for weakened in variants:
        with pytest.raises(ValueError, match="frozen producer-manifest contract"):
            pub.validate_asset_catalog(weakened)
        with pytest.raises(ValueError, match="frozen producer-manifest contract"):
            pub.catalog_sha256(weakened)
        with pytest.raises(ValueError, match="frozen producer-manifest contract"):
            pub.validate_producer_manifest_inputs(
                dict(_EXPECTED_PRODUCER_MANIFESTS), catalog=weakened
            )


def test_catalog_rejects_missing_and_reordered_compatibility_path_declarations():
    import dataclasses

    pub = _publisher()
    catalog = pub.DATA_V4_CATALOG
    without_stable_path = dataclasses.replace(
        catalog,
        compatibility_paths=tuple(
            pair
            for pair in catalog.compatibility_paths
            if pair
            != ("factor_equity_ext2026.parquet", "factor_equity_ext2026.parquet")
        ),
    )
    reordered = dataclasses.replace(
        catalog,
        compatibility_paths=tuple(reversed(catalog.compatibility_paths)),
    )

    for weakened in (without_stable_path, reordered):
        with pytest.raises(ValueError, match="frozen compatibility-path contract"):
            pub.validate_asset_catalog(weakened)
        with pytest.raises(ValueError, match="frozen compatibility-path contract"):
            pub.catalog_sha256(weakened)


def test_catalog_rejects_duplicate_public_names_and_mutable_current_paths():
    import dataclasses

    pub = _publisher()
    catalog = pub.DATA_V4_CATALOG

    duplicate = dataclasses.replace(catalog, assets=catalog.assets + (catalog.assets[0],))
    with pytest.raises(ValueError, match="duplicate public basename"):
        pub.validate_asset_catalog(duplicate)

    first = catalog.assets[0]
    mutable_pointer = dataclasses.replace(first, source_artifact="data/current.json")
    changed = dataclasses.replace(catalog, assets=(mutable_pointer,) + catalog.assets[1:])
    with pytest.raises(ValueError, match="mutable current-release pointer"):
        pub.validate_asset_catalog(changed)

    reserved = dataclasses.replace(first, public_basename="SHA256SUMS")
    changed = dataclasses.replace(catalog, assets=(reserved,) + catalog.assets[1:])
    with pytest.raises(ValueError, match="reserved release filename"):
        pub.validate_asset_catalog(changed)

    completed_payload = dataclasses.replace(first, public_basename="completed")
    changed = dataclasses.replace(catalog, assets=(completed_payload,) + catalog.assets[1:])
    with pytest.raises(ValueError, match="reserved release filename"):
        pub.validate_asset_catalog(changed)


def test_catalog_input_gate_rejects_missing_assets_aliases_and_wrong_manifests():
    import dataclasses

    pub = _publisher()
    catalog = pub.DATA_V4_CATALOG
    inputs = tuple(
        pub.CatalogAssetInput(
            public_basename=asset.public_basename,
            source_artifact=asset.source_artifact,
            producer_manifest_role=asset.producer_manifest_role,
        )
        for asset in catalog.assets
    )
    assert pub.validate_catalog_inputs(inputs) == catalog.assets

    with pytest.raises(ValueError, match="missing required asset"):
        pub.validate_catalog_inputs(inputs[1:])

    undeclared_alias = dataclasses.replace(
        inputs[0], public_basename="latest.csv", source_artifact="latest.csv"
    )
    with pytest.raises(ValueError, match="undeclared alias or extra asset"):
        pub.validate_catalog_inputs(inputs + (undeclared_alias,))

    wrong_manifest = dataclasses.replace(inputs[0], producer_manifest_role="scratch_run")
    with pytest.raises(ValueError, match="unexpected producer manifest"):
        pub.validate_catalog_inputs((wrong_manifest,) + inputs[1:])

    case_collision = dataclasses.replace(
        inputs[0], public_basename=inputs[1].public_basename.swapcase()
    )
    with pytest.raises(ValueError, match="duplicate public basename"):
        pub.validate_catalog_inputs((case_collision,) + inputs[1:])

    expected_manifests = {
        "canonical_reports": "canonical_reports.v1",
        "factor_run": "factor_run.v1",
        "market_snapshot": "market_snapshot.v1",
        "presentation_outputs": "presentation_outputs.v1",
        "sjm_run": "sjm_run.v3",
    }
    assert pub.validate_producer_manifest_inputs(expected_manifests) == tuple(
        sorted(expected_manifests.items())
    )
    with pytest.raises(ValueError, match="unexpected producer manifest"):
        pub.validate_producer_manifest_inputs(expected_manifests | {"current": "pointer.v1"})
    missing = dict(expected_manifests)
    missing.pop("factor_run")
    with pytest.raises(ValueError, match="missing producer manifest"):
        pub.validate_producer_manifest_inputs(missing)


def test_catalog_input_gate_rejects_wrong_source_artifact_with_frozen_catalog():
    import dataclasses

    pub = _publisher()
    catalog = pub.DATA_V4_CATALOG
    inputs = _catalog_inputs(pub, catalog)
    target_name = "factor_evidence_ext2026.parquet"
    changed = tuple(
        dataclasses.replace(
            item,
            source_artifact="artifacts/factor_evidence_ext2026_copy.parquet",
        )
        if item.public_basename == target_name
        else item
        for item in inputs
    )

    assert pub.validate_asset_catalog(catalog) == catalog
    with pytest.raises(ValueError, match="unexpected source artifact"):
        pub.validate_catalog_inputs(changed)


def test_catalog_input_gate_rejects_declared_but_wrong_producer_role():
    import dataclasses

    pub = _publisher()
    catalog = pub.DATA_V4_CATALOG
    inputs = _catalog_inputs(pub, catalog)
    target_name = "factor_evidence_ext2026.parquet"
    wrong_role = "canonical_reports"
    expected_role = next(
        asset.producer_manifest_role
        for asset in catalog.assets
        if asset.public_basename == target_name
    )
    changed = tuple(
        dataclasses.replace(item, producer_manifest_role=wrong_role)
        if item.public_basename == target_name
        else item
        for item in inputs
    )

    assert wrong_role in catalog.producer_manifest_map
    assert wrong_role != expected_role
    with pytest.raises(ValueError, match=f"expected {expected_role!r}"):
        pub.validate_catalog_inputs(changed)


def test_producer_manifest_gate_rejects_wrong_schema_for_expected_role():
    pub = _publisher()
    catalog = pub.DATA_V4_CATALOG
    manifests = catalog.producer_manifest_map
    expected_schema = manifests["factor_run"]
    wrong_schema = manifests["canonical_reports"]
    changed = manifests | {"factor_run": wrong_schema}

    assert wrong_schema != expected_schema
    with pytest.raises(ValueError, match=f"expected {expected_schema!r}"):
        pub.validate_producer_manifest_inputs(changed)


def test_checksum_contract_is_non_recursive_and_manifest_is_the_only_completion_state():
    pub = _publisher()
    catalog = pub.DATA_V4_CATALOG

    assert catalog.manifest_inventory_basenames == catalog.payload_basenames
    assert catalog.checksum_basenames == tuple(
        sorted(catalog.payload_basenames + ("publication_manifest.json",))
    )
    assert "SHA256SUMS" not in catalog.checksum_basenames
    assert catalog.final_inventory_basenames == tuple(
        sorted(catalog.checksum_basenames + ("SHA256SUMS",))
    )
    assert "COMPLETED" not in catalog.final_inventory_basenames
    assert catalog.completion_authority == "publication_manifest.json"
    assert catalog.forbidden_release_files == ("COMPLETED",)


def test_compatibility_paths_are_explicit_catalog_owned_aliases_only():
    pub = _publisher()
    catalog = pub.DATA_V4_CATALOG
    by_name = {asset.public_basename: asset for asset in catalog.assets}
    compatibility = dict(catalog.compatibility_paths)

    assert compatibility["tear_sheet_ext2026.csv"] == (
        "portfolio_metrics_reader_ext2026.csv"
    )
    assert compatibility["sjm_crowding_derisk_equity_ext2026.parquet"] == (
        "sjm_crowding_v3_total_return_bil_equity_ext2026.parquet"
    )
    assert compatibility["factor_equity_ext2026.parquet"] == (
        "factor_equity_ext2026.parquet"
    )
    for public_path, target in catalog.compatibility_paths:
        assert public_path in by_name
        assert target in by_name
        if public_path != target:
            alias = by_name[public_path]
            target_asset = by_name[target]
            assert alias.projection == "compatibility_alias"
            assert alias.projection_of == target
            assert alias.source_artifact == target_asset.source_artifact
            assert alias.producer_manifest_role == target_asset.producer_manifest_role
            assert alias.lineage == target_asset.lineage
            assert alias.schema_id == target_asset.schema_id
            assert alias.media_type == target_asset.media_type
            assert alias.locale == target_asset.locale


def test_local_data_v4_release_assets_are_ignored_without_hiding_contract_sources():
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    def is_ignored(path: str) -> bool:
        return subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", path],
            cwd=root,
            check=False,
        ).returncode == 0

    assert is_ignored("release_assets/data-v4/payload.parquet")
    assert not is_ignored("scripts/publish_finance_remediation.py")
    assert not is_ignored("tests/test_publication_artifacts.py")
    assert not is_ignored(
        ".kiro/specs/finance-metric-integrity-remediation/design.md"
    )


# --------------------------------------------------------------------------- #
# Tasks 10.8/10.9: incomplete candidate staging, direct validation,           #
# finalization, checksums, read-only verification, and atomic promotion.      #
# Fixture-driven only: producer trees below are synthetic byte payloads with  #
# completed producer manifests — no real canonical artifacts are consumed.    #
# --------------------------------------------------------------------------- #

_PARQUET = "application/vnd.apache.parquet"
_BUILD_TIME = "2026-07-29T12:00:00+00:00"


def _write_producer_manifest(role_dir, manifest) -> None:
    """Producer convention (tasks 5.3/6.9): COMPLETED carries the manifest sha."""
    role_dir.mkdir(parents=True, exist_ok=True)
    path = role_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (role_dir / "COMPLETED").write_text(
        "2026-07-29T00:00:00+00:00\n"
        f"manifest_sha256={hashlib.sha256(path.read_bytes()).hexdigest()}\n"
    )


def _mutate_producer_manifest(role_dir, mutate) -> None:
    manifest = json.loads((role_dir / "manifest.json").read_text())
    mutate(manifest)
    _write_producer_manifest(role_dir, manifest)


def _producer_dirs(root, pub, payload=None) -> dict:
    """Synthetic completed producer trees covering the full frozen catalog.

    ``payload`` maps a catalog asset to its fixture bytes; the default emits
    plain text bytes (cheap, sufficient for staging/finalization tests).
    """
    catalog = pub.DATA_V4_CATALOG
    entries: dict[str, dict[str, dict]] = {
        role: {} for role in catalog.producer_manifest_map
    }
    for asset in catalog.assets:
        role_entries = entries[asset.producer_manifest_role]
        if asset.source_artifact in role_entries:
            continue  # aliases share their target's source bytes
        src = root / asset.producer_manifest_role / asset.source_artifact
        src.parent.mkdir(parents=True, exist_ok=True)
        if payload is None:
            data = (
                f"fixture:{asset.producer_manifest_role}:{asset.source_artifact}\n"
            ).encode()
        else:
            data = payload(asset)
        src.write_bytes(data)
        entry: dict = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
            "schema_id": asset.schema_id,
        }
        if asset.asset_class == "canonical_payload" and asset.media_type == _PARQUET:
            entry |= {"rows": 5, "start": "2016-01-04", "end": "2026-03-31"}
        role_entries[asset.source_artifact] = entry
    for role, schema_id in catalog.producer_manifest_map.items():
        _write_producer_manifest(
            root / role,
            {"schema": schema_id, "completed": True, "assets": entries[role]},
        )
    return {role: root / role for role in catalog.producer_manifest_map}


def _stage(pub, producers, destination, publication_id="pub-fixture-0001"):
    return pub.stage_publication_candidate(
        destination=destination,
        producers=producers,
        publication_id=publication_id,
        build_time=_BUILD_TIME,
    )


def _candidate_manifest(candidate) -> dict:
    return json.loads((candidate / "publication_manifest.json").read_text())


def _rewrite_candidate_manifest(candidate, manifest) -> None:
    (candidate / "publication_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def test_staging_builds_an_incomplete_candidate_with_exact_inventory(tmp_path):
    pub = _publisher()
    catalog = pub.DATA_V4_CATALOG
    producers = _producer_dirs(tmp_path / "producers", pub)

    candidate = _stage(pub, producers, tmp_path / "candidate")

    names = sorted(p.name for p in candidate.iterdir())
    assert names == sorted(catalog.payload_basenames + ("publication_manifest.json",))
    assert not (candidate / "SHA256SUMS").exists()
    assert not (candidate / "COMPLETED").exists()

    manifest = _candidate_manifest(candidate)
    assert manifest["schema"] == "publication_manifest.v1"
    assert manifest["release_tag"] == "data-v4"
    assert manifest["completed"] is False
    assert manifest["catalog_sha256"] == _EXPECTED_CATALOG_SHA256
    assert sorted(manifest["assets"]) == sorted(catalog.payload_basenames)
    for name, entry in manifest["assets"].items():
        data = (candidate / name).read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
        assert entry["size"] == len(data)
    for role, recorded in manifest["input_manifests"].items():
        assert recorded["manifest_sha256"] == hashlib.sha256(
            (producers[role] / "manifest.json").read_bytes()
        ).hexdigest()
    # aliases stage their target's exact bytes
    assert (candidate / "tear_sheet_ext2026.csv").read_bytes() == (
        candidate / "portfolio_metrics_reader_ext2026.csv"
    ).read_bytes()

    # the direct validators accept the incomplete candidate, with and without
    # the producer manifests supplied for source-hash cross-checks
    pub.validate_staged_candidate(candidate)
    pub.validate_staged_candidate(candidate, producers=producers)

    # deterministic: an identical second staging produces identical manifests
    second = _stage(pub, producers, tmp_path / "candidate_2")
    assert (second / "publication_manifest.json").read_bytes() == (
        candidate / "publication_manifest.json"
    ).read_bytes()


def test_staging_refuses_existing_or_completed_candidates(tmp_path):
    pub = _publisher()
    producers = _producer_dirs(tmp_path / "producers", pub)

    # a pre-created EMPTY destination is a valid new candidate location
    empty = tmp_path / "empty"
    empty.mkdir()
    _stage(pub, producers, empty)

    # an existing candidate is never overwritten
    with pytest.raises(ValueError, match="non-empty"):
        _stage(pub, producers, empty)

    # a candidate claiming completion is immutable
    done = _stage(pub, producers, tmp_path / "done")
    manifest = _candidate_manifest(done)
    manifest["completed"] = True
    _rewrite_candidate_manifest(done, manifest)
    with pytest.raises(ValueError, match="immutable"):
        _stage(pub, producers, done)


def test_every_source_fault_blocks_staging_and_leaves_no_manifest(tmp_path):
    pub = _publisher()
    target = next(
        asset
        for asset in pub.DATA_V4_CATALOG.assets
        if asset.public_basename == "factor_evidence_ext2026.parquet"
    )

    def fault_missing(producers):
        (producers["factor_run"] / target.source_artifact).unlink()

    def fault_corrupt(producers):
        (producers["factor_run"] / target.source_artifact).write_bytes(b"tampered")

    def fault_unowned(producers):
        _mutate_producer_manifest(
            producers["factor_run"],
            lambda m: m["assets"].pop(target.source_artifact),
        )

    def fault_incomplete_producer(producers):
        _mutate_producer_manifest(
            producers["factor_run"], lambda m: m.__setitem__("completed", False)
        )

    def fault_stale_completion(producers):
        # rewrite manifest bytes WITHOUT refreshing the completion marker
        path = producers["factor_run"] / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["assets"][target.source_artifact]["rows"] = 6
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    def fault_wrong_producer_schema(producers):
        _mutate_producer_manifest(
            producers["factor_run"],
            lambda m: m.__setitem__("schema", "factor_run.v999"),
        )

    def fault_missing_role(producers):
        producers.pop("market_snapshot")

    def fault_extra_role(producers):
        producers["scratch_run"] = producers["factor_run"]

    def fault_wrong_schema_id(producers):
        _mutate_producer_manifest(
            producers["factor_run"],
            lambda m: m["assets"][target.source_artifact].__setitem__(
                "schema_id", "wrong.schema.v999"
            ),
        )

    def fault_incoherent_window(producers):
        _mutate_producer_manifest(
            producers["factor_run"],
            lambda m: m["assets"][target.source_artifact].__setitem__(
                "start", "2027-01-01"
            ),
        )

    faults = (
        ("missing", fault_missing, "absent"),
        ("corrupt", fault_corrupt, "stale or corrupt"),
        ("unowned", fault_unowned, "not inventoried"),
        ("incomplete_producer", fault_incomplete_producer, "completed=true"),
        ("stale_completion", fault_stale_completion, "COMPLETED marker"),
        ("wrong_producer_schema", fault_wrong_producer_schema, "has schema"),
        ("missing_role", fault_missing_role, "missing producer manifest"),
        ("extra_role", fault_extra_role, "unexpected producer manifest"),
        ("wrong_schema_id", fault_wrong_schema_id, "catalog schema"),
        ("bad_window", fault_incoherent_window, "window"),
    )
    for name, fault, pattern in faults:
        producers = _producer_dirs(tmp_path / f"producers_{name}", pub)
        fault(producers)
        destination = tmp_path / f"candidate_{name}"
        with pytest.raises(ValueError, match=pattern):
            _stage(pub, producers, destination)
        # the failure is reported but the candidate never becomes complete
        assert not (destination / "publication_manifest.json").exists()
        assert not (destination / "SHA256SUMS").exists()
        assert not (destination / "COMPLETED").exists()


def test_direct_staging_validators_reject_each_integrity_dimension(tmp_path):
    pub = _publisher()
    producers = _producer_dirs(tmp_path / "producers", pub)
    payload = "factor_evidence_ext2026.parquet"

    def fresh(name):
        return _stage(pub, producers, tmp_path / name)

    cand = fresh("extra")
    (cand / "stray.txt").write_text("x")
    with pytest.raises(ValueError, match="extra file"):
        pub.validate_staged_candidate(cand)

    cand = fresh("sums")
    (cand / "SHA256SUMS").write_text("")
    with pytest.raises(ValueError, match="SHA256SUMS"):
        pub.validate_staged_candidate(cand)

    cand = fresh("marker")
    (cand / "COMPLETED").write_text("x")
    with pytest.raises(ValueError, match="COMPLETED"):
        pub.validate_staged_candidate(cand)

    cand = fresh("claim")
    manifest = _candidate_manifest(cand)
    manifest["completed"] = True
    _rewrite_candidate_manifest(cand, manifest)
    with pytest.raises(ValueError, match="completed=false"):
        pub.validate_staged_candidate(cand)

    cand = fresh("values")
    (cand / payload).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="mutated after staging"):
        pub.validate_staged_candidate(cand)

    cand = fresh("inventory")
    (cand / payload).unlink()
    with pytest.raises(ValueError, match="missing from disk"):
        pub.validate_staged_candidate(cand)

    cand = fresh("duplicate")
    (cand / payload.upper()).write_bytes(b"twin")
    with pytest.raises(ValueError, match="duplicate public basename"):
        pub.validate_staged_candidate(cand)

    cand = fresh("lineage")
    manifest = _candidate_manifest(cand)
    manifest["assets"][payload]["lineage"] = ["factor_run"]
    _rewrite_candidate_manifest(cand, manifest)
    with pytest.raises(ValueError, match="lineage"):
        pub.validate_staged_candidate(cand)

    cand = fresh("window")
    manifest = _candidate_manifest(cand)
    manifest["assets"][payload]["start"] = "2027-01-01"
    _rewrite_candidate_manifest(cand, manifest)
    with pytest.raises(ValueError, match="window"):
        pub.validate_staged_candidate(cand)

    # source-hash cross-check: the producer's bytes drift AFTER staging
    cand = fresh("source_hash")
    target = next(
        asset
        for asset in pub.DATA_V4_CATALOG.assets
        if asset.public_basename == payload
    )
    (producers["factor_run"] / target.source_artifact).write_bytes(b"drifted")
    pub.validate_staged_candidate(cand)  # without producers the drift is unseen
    with pytest.raises(ValueError, match="stale or corrupt"):
        pub.validate_staged_candidate(cand, producers=producers)


def test_finalization_completes_the_manifest_once_and_checksums_exclude_itself(
    tmp_path,
):
    pub = _publisher()
    catalog = pub.DATA_V4_CATALOG
    producers = _producer_dirs(tmp_path / "producers", pub)
    cand = _stage(pub, producers, tmp_path / "candidate")

    finalized = pub.finalize_publication_candidate(cand, producers=producers)
    assert finalized == cand

    manifest = _candidate_manifest(cand)
    assert manifest["completed"] is True

    parsed: dict[str, str] = {}
    for line in (cand / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        parsed[name] = digest
    assert tuple(sorted(parsed)) == catalog.checksum_basenames
    assert "SHA256SUMS" not in parsed
    assert "publication_manifest.json" in parsed
    for name, digest in parsed.items():
        assert digest == hashlib.sha256((cand / name).read_bytes()).hexdigest()

    # finalization happens exactly once
    with pytest.raises(ValueError, match="already finalized"):
        pub.finalize_publication_candidate(cand, producers=producers)


def test_finalization_failure_reports_and_leaves_candidate_incomplete(tmp_path):
    pub = _publisher()
    producers = _producer_dirs(tmp_path / "producers", pub)
    cand = _stage(pub, producers, tmp_path / "candidate")
    (cand / "factor_evidence_ext2026.parquet").write_bytes(b"mutated")

    with pytest.raises(ValueError, match="mutated after staging"):
        pub.finalize_publication_candidate(cand, producers=producers)

    # the fault is reported and the candidate remains diagnosable but incomplete
    assert _candidate_manifest(cand)["completed"] is False
    assert not (cand / "SHA256SUMS").exists()
    with pytest.raises(ValueError):
        pub.verify_finalized_candidate(cand)


def test_finalized_candidate_verifies_read_only_and_promotes_atomically(tmp_path):
    pub = _publisher()
    producers = _producer_dirs(tmp_path / "producers", pub)
    cand = _stage(pub, producers, tmp_path / "candidate")
    pub.finalize_publication_candidate(cand, producers=producers)

    before = {
        p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in cand.iterdir()
    }
    for p in cand.iterdir():
        p.chmod(0o444)
    cand.chmod(0o555)
    try:
        pub.verify_finalized_candidate(cand)  # must succeed WITHOUT mutation
    finally:
        cand.chmod(0o755)
        for p in cand.iterdir():
            p.chmod(0o644)
    after = {
        p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in cand.iterdir()
    }
    assert after == before

    final = tmp_path / "release_assets" / "data-v4"
    promoted = pub.promote_finalized_candidate(cand, final)
    assert promoted == final
    assert final.is_dir()
    assert not cand.exists()
    pub.verify_finalized_candidate(final)


def test_promotion_refuses_overwrite_and_faults_leave_final_destination_absent(
    tmp_path,
):
    pub = _publisher()
    producers = _producer_dirs(tmp_path / "producers", pub)
    final = tmp_path / "final" / "data-v4"

    # an unfinalized provisional candidate can never promote
    provisional = _stage(pub, producers, tmp_path / "provisional")
    with pytest.raises(ValueError, match="completed"):
        pub.promote_finalized_candidate(provisional, final)
    assert not final.exists()

    # a finalized candidate corrupted before promotion fails verification
    corrupt = _stage(pub, producers, tmp_path / "corrupt")
    pub.finalize_publication_candidate(corrupt, producers=producers)
    (corrupt / "factor_evidence_ext2026.parquet").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="mutated after staging"):
        pub.promote_finalized_candidate(corrupt, final)
    assert not final.exists()

    # an existing final destination is never overwritten — even an empty one
    good = _stage(pub, producers, tmp_path / "good")
    pub.finalize_publication_candidate(good, producers=producers)
    final.mkdir(parents=True)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        pub.promote_finalized_candidate(good, final)
    assert good.exists()  # the candidate itself is untouched
    assert not any(final.iterdir())  # nothing partially promoted


def test_ac_7_5(tmp_path):
    # R7.5: every staging, finalization, or promotion fault is reported and the
    # publication stays incomplete — no completed manifest, no SHA256SUMS, and
    # no final destination.
    test_every_source_fault_blocks_staging_and_leaves_no_manifest(
        tmp_path / "staging"
    )
    test_finalization_failure_reports_and_leaves_candidate_incomplete(
        tmp_path / "finalize"
    )
    test_promotion_refuses_overwrite_and_faults_leave_final_destination_absent(
        tmp_path / "promote"
    )



# --------------------------------------------------------------------------- #
# Task 10.10: offline clean-room upload-set smoke tooling.                     #
# Task 10.11: read-only public-release smoke-test hooks.                       #
# A finalized fixture candidate (media-valid bytes) is served over a           #
# temporary localhost endpoint shaped like the public release endpoint; the    #
# REAL workbook release client, checksum verification, schema loaders, and    #
# representative S0-S5 builds all run from a clean temporary directory.        #
# --------------------------------------------------------------------------- #

_FIXTURE_DATES = (
    "2016-01-04",
    "2016-01-05",
    "2016-01-06",
    "2026-03-30",
    "2026-03-31",
)


def _png_fixture_bytes() -> bytes:
    import struct
    import zlib

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00\x00"))
        + chunk(b"IEND", b"")
    )


def _media_payload(asset) -> bytes:
    """Media-valid fixture bytes for one catalog asset (5-row tables)."""
    import io

    values = [round(1.0 + i * 0.25, 2) for i in range(len(_FIXTURE_DATES))]
    if asset.media_type == "application/vnd.apache.parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        sink = io.BytesIO()
        pq.write_table(
            pa.table(
                {
                    "date": list(_FIXTURE_DATES),
                    "value": values,
                    "series": [asset.source_artifact] * len(_FIXTURE_DATES),
                }
            ),
            sink,
        )
        return sink.getvalue()
    if asset.media_type.startswith("text/csv"):
        if asset.locale == "de-DE":
            rows = ["date;value;series"] + [
                f"{d};{str(v).replace('.', ',')};{asset.source_artifact}"
                for d, v in zip(_FIXTURE_DATES, values)
            ]
        else:
            rows = ["date,value,series"] + [
                f"{d},{v},{asset.source_artifact}"
                for d, v in zip(_FIXTURE_DATES, values)
            ]
        return ("\n".join(rows) + "\n").encode()
    if asset.media_type == "application/json":
        return (
            json.dumps(
                {
                    "source_artifact": asset.source_artifact,
                    "records": [
                        {"date": d, "value": v}
                        for d, v in zip(_FIXTURE_DATES, values)
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
    if asset.media_type == "image/png":
        return _png_fixture_bytes()
    return f"# fixture {asset.source_artifact}\ncontent body\n".encode()


def _finalized_media_candidate(pub, producers, destination):
    cand = _stage(pub, producers, destination)
    pub.finalize_publication_candidate(cand, producers=producers)
    return cand


def test_publication_manifest_speaks_the_release_client_contract(tmp_path):
    # Task 10.10 seam: the finalized manifest must satisfy the READ-ONLY
    # workbook release-client contract (schema_id + artifacts list) while the
    # richer assets inventory stays authoritative — and the two may not drift.
    pub = _publisher()
    producers = _producer_dirs(tmp_path / "producers", pub)
    cand = _stage(pub, producers, tmp_path / "candidate")

    manifest = _candidate_manifest(cand)
    assert manifest["schema_id"] == "publication_manifest.v1"
    artifacts = manifest["artifacts"]
    assert [row["path"] for row in artifacts] == sorted(manifest["assets"])
    for row in artifacts:
        entry = manifest["assets"][row["path"]]
        assert row["sha256"] == entry["sha256"]
        assert row["size"] == entry["size"]

    drifted = _candidate_manifest(cand)
    drifted["artifacts"][0]["sha256"] = "0" * 64
    _rewrite_candidate_manifest(cand, drifted)
    with pytest.raises(ValueError, match="artifacts"):
        pub.validate_staged_candidate(cand)


def test_offline_clean_room_smoke_passes_a_finalized_fixture_upload_set(tmp_path):
    pub = _publisher()
    catalog = pub.DATA_V4_CATALOG
    producers = _producer_dirs(tmp_path / "producers", pub, payload=_media_payload)
    cand = _finalized_media_candidate(pub, producers, tmp_path / "candidate")
    clean_room = tmp_path / "clean_room"

    with pub.serve_publication_candidate(cand) as served:
        report = pub.run_offline_release_smoke(
            base_url=served.base_url, tag="data-v4", clean_room=clean_room
        )

    assert report["state"] == "verified"
    assert report["tag"] == "data-v4"
    assert report["assets_verified"] == len(catalog.payload_basenames)
    assert report["schemas_validated"] == len(catalog.payload_basenames)
    # representative S0-S5 builds all loaded real assets via the real client
    assert set(report["step_builds"]) == {"S0", "S1", "S2", "S3", "S4", "S5"}
    assert all(report["step_builds"][step] for step in report["step_builds"])
    # the endpoint saw only read requests
    assert served.requests and all(m == "GET" for m, _ in served.requests)
    # the clean room holds the client cache: checksums verified with no
    # repository data path in reach
    assert (clean_room / "cache" / "data-v4").is_dir()


def test_offline_smoke_blocks_repository_paths_and_mutable_pointers(tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    pub = _publisher()
    producers = _producer_dirs(tmp_path / "producers", pub, payload=_media_payload)
    cand = _finalized_media_candidate(pub, producers, tmp_path / "candidate")

    # a clean room inside the repository is refused outright
    repo_room = Path(pub.__file__).resolve().parents[1] / "data" / "_smoke_room"
    with pub.serve_publication_candidate(cand) as served:
        with pytest.raises(ValueError, match="repository"):
            pub.run_offline_release_smoke(
                base_url=served.base_url, tag="data-v4", clean_room=repo_room
            )
        assert not repo_room.exists()

        # a non-empty clean room (stale cache) is refused
        dirty = tmp_path / "dirty"
        dirty.mkdir()
        (dirty / "stale").write_text("x")
        with pytest.raises(ValueError, match="empty"):
            pub.run_offline_release_smoke(
                base_url=served.base_url, tag="data-v4", clean_room=dirty
            )

        # only an explicit immutable tag is accepted
        with pytest.raises(ValueError, match="immutable"):
            pub.run_offline_release_smoke(
                base_url=served.base_url, tag="latest", clean_room=tmp_path / "r1"
            )

    # the offline smoke never talks to a non-local endpoint
    with pytest.raises(ValueError, match="localhost"):
        pub.run_offline_release_smoke(
            base_url="https://github.com/norandom/Global_Macro_AI_Factors",
            tag="data-v4",
            clean_room=tmp_path / "r2",
        )

    # the server itself refuses to impersonate a mutable pointer tag
    with pytest.raises(ValueError, match="immutable"):
        with pub.serve_publication_candidate(cand, tag="latest"):
            pass

    # an endpoint that DOES expose a mutable current-release pointer fails
    class _PointerHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib API name
            body = b"{}"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _PointerHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(ValueError, match="mutable current-release pointer"):
            pub.run_offline_release_smoke(
                base_url=f"http://127.0.0.1:{server.server_port}",
                tag="data-v4",
                clean_room=tmp_path / "r3",
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_offline_smoke_fails_each_faulted_upload_set(tmp_path):
    pub = _publisher()
    producers = _producer_dirs(tmp_path / "producers", pub, payload=_media_payload)

    def fresh(name):
        return _finalized_media_candidate(pub, producers, tmp_path / name)

    def smoke(cand, room):
        with pub.serve_publication_candidate(cand) as served:
            return pub.run_offline_release_smoke(
                base_url=served.base_url, tag="data-v4", clean_room=tmp_path / room
            )

    payload = "factor_evidence_ext2026.parquet"

    # missing: a manifest-owned asset absent from the upload set
    cand = fresh("missing")
    (cand / payload).unlink()
    with pytest.raises(ValueError, match="missing"):
        smoke(cand, "room_missing")

    # extra: an uncataloged file in the upload set
    cand = fresh("extra")
    (cand / "stray.bin").write_bytes(b"stray")
    with pytest.raises(ValueError, match="extra"):
        smoke(cand, "room_extra")

    # corrupt: asset bytes diverge from the manifest hash — the REAL release
    # client refuses them before any load
    cand = fresh("corrupt")
    (cand / payload).write_bytes(b"tampered payload")
    with pytest.raises(ValueError, match="integrity"):
        smoke(cand, "room_corrupt")

    # stale: SHA256SUMS no longer matches the finalized bytes
    cand = fresh("stale")
    sums = (cand / "SHA256SUMS").read_text().splitlines()
    sums[0] = ("f" * 64) + sums[0][64:]
    (cand / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    with pytest.raises(ValueError, match="SHA256SUMS"):
        smoke(cand, "room_stale")

    # incomplete: a provisional (never finalized) candidate is not releasable
    provisional = _stage(pub, producers, tmp_path / "provisional")
    with pytest.raises(ValueError, match="SHA256SUMS|completed"):
        smoke(provisional, "room_provisional")

    # schema-invalid: manifest-consistent bytes that do not parse as the
    # declared media/schema fail the schema loaders
    bad_producers = _producer_dirs(
        tmp_path / "producers_schema", pub, payload=_media_payload
    )
    target = next(
        asset
        for asset in pub.DATA_V4_CATALOG.assets
        if asset.public_basename == payload
    )
    bad_bytes = b"not a parquet table\n"
    (bad_producers["factor_run"] / target.source_artifact).write_bytes(bad_bytes)
    _mutate_producer_manifest(
        bad_producers["factor_run"],
        lambda m: m["assets"][target.source_artifact].update(
            sha256=hashlib.sha256(bad_bytes).hexdigest(), size=len(bad_bytes)
        ),
    )
    cand = _finalized_media_candidate(
        pub, bad_producers, tmp_path / "schema_invalid"
    )
    with pytest.raises(ValueError, match="schema-invalid"):
        smoke(cand, "room_schema")


def test_public_smoke_verifies_release_and_frozen_history_read_only(tmp_path):
    pub = _publisher()
    producers = _producer_dirs(tmp_path / "producers", pub, payload=_media_payload)
    cand = _finalized_media_candidate(pub, producers, tmp_path / "candidate")

    hist = tmp_path / "hist-v2"
    hist.mkdir()
    (hist / "factor_loadings_v1.parquet").write_bytes(b"frozen-a")
    (hist / "static_bh_stats.json").write_bytes(b"frozen-b")
    pins = {
        "data-v2": {
            "factor_loadings_v1.parquet": hashlib.sha256(b"frozen-a").hexdigest(),
            "static_bh_stats.json": hashlib.sha256(b"frozen-b").hexdigest(),
        }
    }

    with pub.serve_publication_candidate(
        cand, historical={"data-v2": hist}
    ) as served:
        report = pub.run_public_release_smoke(
            tag="data-v4",
            base_url=served.base_url,
            clean_room=tmp_path / "room",
            historical_pins=pins,
        )
        # only an explicit immutable tag is accepted — never a mutable pointer
        with pytest.raises(ValueError, match="immutable"):
            pub.run_public_release_smoke(
                tag="latest", base_url=served.base_url, clean_room=tmp_path / "r1"
            )
        # a clean cache is mandatory
        dirty = tmp_path / "dirty"
        dirty.mkdir()
        (dirty / "stale").write_text("x")
        with pytest.raises(ValueError, match="empty"):
            pub.run_public_release_smoke(
                tag="data-v4", base_url=served.base_url, clean_room=dirty
            )

    assert report["state"] == "verified"
    assert report["historical"] == {"data-v2": {"assets_verified": 2}}
    # strictly read-only: the endpoint saw GET requests only, nothing
    # upload- or release-creation-shaped
    assert served.requests and all(m == "GET" for m, _ in served.requests)
    assert not any("upload" in path.lower() for _, path in served.requests)
    # and the publisher module exposes no release-creation/upload capability
    assert not [
        n
        for n in dir(pub)
        if "upload" in n.lower() or "create_release" in n.lower()
    ]


def test_public_smoke_reports_not_yet_published_and_fails_public_faults(tmp_path):
    pub = _publisher()
    producers = _producer_dirs(tmp_path / "producers", pub, payload=_media_payload)

    # not-yet-published: the public endpoint has no data-v4 release at all —
    # a clear state, not a verification failure
    hist = tmp_path / "hist"
    hist.mkdir()
    (hist / "asset.bin").write_bytes(b"old")
    with pub.serve_publication_candidate(hist, tag="data-v3") as served:
        report = pub.run_public_release_smoke(
            tag="data-v4", base_url=served.base_url, clean_room=tmp_path / "r0"
        )
    assert report["state"] == "not_yet_published"
    assert "data-v4" in report["detail"]

    def fresh(name):
        return _finalized_media_candidate(pub, producers, tmp_path / name)

    def smoke(cand, room, **kwargs):
        with pub.serve_publication_candidate(cand, **kwargs) as served:
            return pub.run_public_release_smoke(
                tag="data-v4",
                base_url=served.base_url,
                clean_room=tmp_path / room,
            )

    # missing asset in the public release
    cand = fresh("missing")
    (cand / "factor_evidence_ext2026.parquet").unlink()
    with pytest.raises(ValueError, match="missing"):
        smoke(cand, "r_missing")

    # duplicated asset names in the public listing
    cand = fresh("duplicate")
    names = sorted(p.name for p in cand.iterdir())
    with pytest.raises(ValueError, match="duplicate"):
        smoke(cand, "r_duplicate", listing_names=names + [names[0]])

    # public bytes that no longer match the publication manifest hash
    cand = fresh("mismatch")
    (cand / "factor_evidence_ext2026.parquet").write_bytes(b"drifted bytes")
    with pytest.raises(ValueError, match="integrity"):
        smoke(cand, "r_mismatch")

    # historical stability: a frozen historical hash that changed is an error
    cand = fresh("history")
    hist2 = tmp_path / "hist2"
    hist2.mkdir()
    (hist2 / "asset.bin").write_bytes(b"rewritten history")
    bad_pins = {"data-v2": {"asset.bin": hashlib.sha256(b"original").hexdigest()}}
    with pub.serve_publication_candidate(
        cand, historical={"data-v2": hist2}
    ) as served:
        with pytest.raises(ValueError, match="frozen hash"):
            pub.run_public_release_smoke(
                tag="data-v4",
                base_url=served.base_url,
                clean_room=tmp_path / "r_history",
                historical_pins=bad_pins,
            )


# --------------------------------------------------------------------------- #
# Task 9.1: manifest-aware canonical report input loading.                     #
# Task 9.2: canonical Factor and AI-variant publication tables.                #
# Fixture-driven only: completed Factor/SJM/snapshot bundles are built in      #
# temporary directories with the SAME producer code paths the real pipeline    #
# uses (tests/test_stream_ext2026.py and tests/test_sjm_crowding.py            #
# conventions); no tracked file under data/ or reports/ is read or written.    #
# --------------------------------------------------------------------------- #


def _reports_producer():
    import importlib

    return importlib.import_module("scripts.build_tear_sheet")


@pytest.fixture(scope="module")
def canonical_case(tmp_path_factory):
    """One completed Factor run, its completed market snapshot, and one
    completed SJM v3 run derived from them, plus the pinned identities a
    canonical report build would carry."""
    from tests.test_sjm_crowding import _sjm_build_context
    from tests.test_stream_ext2026 import _completed_factor_run

    tmp = tmp_path_factory.mktemp("canonical_reports_case")
    _, _, run_dir, run_kwargs = _completed_factor_run(tmp)
    snapshot_dir = tmp / run_kwargs["input_manifests"]["market_snapshot"]["snapshot_id"]
    factor_sha = hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest()
    snapshot_sha = hashlib.sha256(
        (snapshot_dir / "manifest.json").read_bytes()
    ).hexdigest()
    sjm_expected = {
        "factor_run_id": run_kwargs["run_id"],
        "factor_manifest_sha256": factor_sha,
        "market_snapshot_id": snapshot_dir.name,
        "market_snapshot_sha256": snapshot_sha,
    }
    _, _, build = _sjm_build_context((run_dir, snapshot_dir, sjm_expected), tmp / "sjm")
    sjm_run_dir = build("sjm_crowding_v3_canonical_reports")
    return {
        "factor_run_dir": run_dir,
        "factor_run_id": run_kwargs["run_id"],
        "factor_sha": factor_sha,
        "snapshot_dir": snapshot_dir,
        "snapshot_id": snapshot_dir.name,
        "snapshot_sha": snapshot_sha,
        "sjm_run_dir": sjm_run_dir,
        "sjm_run_id": "sjm_crowding_v3_canonical_reports",
        "sjm_sha": hashlib.sha256(
            (sjm_run_dir / "manifest.json").read_bytes()
        ).hexdigest(),
    }


def _load_factor(bts, case, run_dir=None, **overrides):
    kwargs = {
        "run_id": case["factor_run_id"],
        "manifest_sha256": case["factor_sha"],
    } | overrides
    return bts.load_factor_report_input(run_dir or case["factor_run_dir"], **kwargs)


def test_canonical_report_inputs_load_only_through_completed_manifests(
    canonical_case,
):
    bts = _reports_producer()
    case = canonical_case

    assert bts.REPORT_TABLE_OWNER == "scripts/build_tear_sheet.py"

    factor = _load_factor(bts, case)
    assert (factor.run_id, factor.manifest_sha256) == (
        case["factor_run_id"],
        case["factor_sha"],
    )
    assert factor.manifest["schema"] == "factor_run.v1"
    assert factor.metric_records["schema"] == "factor_run.metric_records.v1"
    assert len(factor.metric_records["records"]) == 9

    sjm_input = bts.load_sjm_report_input(
        case["sjm_run_dir"],
        run_id=case["sjm_run_id"],
        manifest_sha256=case["sjm_sha"],
    )
    assert (sjm_input.family, sjm_input.identity) == ("sjm_run", case["sjm_run_id"])
    assert sjm_input.manifest_sha256 == case["sjm_sha"]
    assert sjm_input.manifest["schema"] == "sjm_run.v3"

    market = bts.load_market_report_input(
        case["snapshot_dir"],
        snapshot_id=case["snapshot_id"],
        manifest_sha256=case["snapshot_sha"],
    )
    assert (market.family, market.identity) == ("market_snapshot", case["snapshot_id"])
    assert market.manifest["schema"] == "market_snapshot.v1"

    markowitz = bts.load_markowitz_report_input(
        case["snapshot_dir"],
        snapshot_id=case["snapshot_id"],
        manifest_sha256=case["snapshot_sha"],
    )
    assert (markowitz.family, markowitz.identity) == (
        "markowitz_inputs",
        case["snapshot_id"],
    )


def test_canonical_report_input_gate_rejects_loose_incomplete_stale_or_incompatible(
    canonical_case, tmp_path
):
    import shutil

    bts = _reports_producer()
    case = canonical_case

    # loose artifacts are never canonical inputs, in any family
    with pytest.raises(ValueError, match="loose"):
        _load_factor(
            bts, case, run_dir=case["factor_run_dir"] / "factor_equity_ext2026.parquet"
        )
    with pytest.raises(ValueError, match="loose"):
        bts.load_market_report_input(
            case["snapshot_dir"] / "cash_market_total_return.parquet",
            snapshot_id=case["snapshot_id"],
            manifest_sha256=case["snapshot_sha"],
        )
    with pytest.raises(ValueError, match="loose"):
        bts.load_sjm_report_input(
            case["sjm_run_dir"] / "sjm_equity.parquet",
            run_id=case["sjm_run_id"],
            manifest_sha256=case["sjm_sha"],
        )

    # stale or wrong pins: identity and manifest digest must both hold
    with pytest.raises(ValueError, match="identity"):
        _load_factor(bts, case, run_id="factor_ext2026_someone_else_v9")
    with pytest.raises(ValueError, match="sha256"):
        _load_factor(bts, case, manifest_sha256="0" * 64)
    with pytest.raises(ValueError, match="identity"):
        bts.load_sjm_report_input(
            case["sjm_run_dir"],
            run_id="sjm_crowding_v3_other",
            manifest_sha256=case["sjm_sha"],
        )
    with pytest.raises(ValueError, match="sha256"):
        bts.load_markowitz_report_input(
            case["snapshot_dir"],
            snapshot_id=case["snapshot_id"],
            manifest_sha256="f" * 64,
        )
    with pytest.raises(ValueError, match="sha256"):
        _load_factor(bts, case, manifest_sha256="not-a-digest")

    # incomplete: a run whose COMPLETED marker is gone never loads
    incomplete = tmp_path / case["factor_run_dir"].name
    shutil.copytree(case["factor_run_dir"], incomplete)
    (incomplete / "COMPLETED").unlink()
    with pytest.raises(ValueError, match="COMPLETED"):
        _load_factor(bts, case, run_dir=incomplete)

    # schema-incompatible: an unknown manifest schema fails validation
    snapshot_copy = tmp_path / case["snapshot_dir"].name
    shutil.copytree(case["snapshot_dir"], snapshot_copy)
    manifest = json.loads((snapshot_copy / "manifest.json").read_text())
    manifest["schema"] = "market_snapshot.v999"
    (snapshot_copy / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(ValueError, match="schema"):
        bts.load_market_report_input(
            snapshot_copy,
            snapshot_id=case["snapshot_id"],
            manifest_sha256=hashlib.sha256(
                (snapshot_copy / "manifest.json").read_bytes()
            ).hexdigest(),
        )


def test_artifact_edit_cannot_stand_in_for_the_owning_producer(
    canonical_case, tmp_path
):
    # R7.6: a published value is never "corrected" by editing the artifact while
    # the owning producer remains unchanged — every escalation of an artifact-only
    # edit fails the canonical input gate BEFORE any table assembly.
    import shutil

    bts = _reports_producer()
    case = canonical_case
    run = tmp_path / case["factor_run_dir"].name
    shutil.copytree(case["factor_run_dir"], run)

    metric_path = run / "factor_metric_records_ext2026.json"
    payload = json.loads(metric_path.read_text())
    reader = next(
        row
        for row in payload["records"]
        if row["schema"] == "portfolio_metrics.reader.v2"
    )
    reader["total_return"] = float(reader["total_return"]) + 0.05  # a hand "fix"
    metric_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    # (a) the artifact edit alone: byte inventory refuses the mutated file
    with pytest.raises(ValueError, match="mutated after inventory"):
        _load_factor(bts, case, run_dir=run)

    # (b) editing the manifest inventory too leaves a stale completion marker
    manifest = json.loads((run / "manifest.json").read_text())
    entry = manifest["files"]["metric_records"]
    entry["sha256"] = hashlib.sha256(metric_path.read_bytes()).hexdigest()
    entry["size"] = metric_path.stat().st_size
    (run / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    resigned_sha = hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="COMPLETED marker"):
        _load_factor(bts, case, run_dir=run, manifest_sha256=resigned_sha)

    # (c) re-signing the completion marker as well: the edited record no longer
    # matches its own immutable record digests inside the metric payload
    marker_first_line = (run / "COMPLETED").read_text().splitlines()[0]
    (run / "COMPLETED").write_text(
        f"{marker_first_line}\nmanifest_sha256={resigned_sha}\n"
    )
    with pytest.raises(ValueError, match="record_sha256s|content_sha256"):
        _load_factor(bts, case, run_dir=run, manifest_sha256=resigned_sha)

    # the untouched producer bundle still loads: only a producer re-run (a new
    # completed bundle) can legitimately change a published value
    _load_factor(bts, case)


def test_ac_7_6(canonical_case, tmp_path):
    test_artifact_edit_cannot_stand_in_for_the_owning_producer(
        canonical_case, tmp_path
    )


def _published_row_equals_record(row, record) -> None:
    import math

    for key, wanted in record.items():
        actual = row[key]
        if wanted is None:
            assert actual is None or (
                isinstance(actual, float) and math.isnan(actual)
            ), key
        else:
            assert actual == wanted, key


def test_factor_tables_project_validated_run_local_records_with_lineage(
    canonical_case,
):
    bts = _reports_producer()
    pub = _publisher()
    case = canonical_case
    factor = _load_factor(bts, case)

    result = bts.build_factor_report_tables(factor)

    assert result.owner == bts.REPORT_TABLE_OWNER
    assert sorted(result.tables) == sorted(bts.FACTOR_REPORT_TABLE_SCHEMAS)
    # the table catalog and schema identities equal the frozen data-v4 contract
    by_name = {asset.public_basename: asset for asset in pub.DATA_V4_CATALOG.assets}
    for stem, schema_id in bts.FACTOR_REPORT_TABLE_SCHEMAS.items():
        assert by_name[f"{stem}.parquet"].schema_id == schema_id
        assert by_name[f"{stem}.parquet"].producer == bts.REPORT_TABLE_OWNER

    records = factor.metric_records["records"]
    by_key = {(row["portfolio_id"], row["schema"]): row for row in records}

    reader = result.tables["portfolio_metrics_reader_ext2026"]
    assert list(reader["portfolio_id"]) == [
        "factor_pit_ext2026",
        "factor_nonpit_diagnostic_ext2026",
    ]
    # representative published rows equal the validated run-local records —
    # nothing is recalculated into a second Factor row family
    _published_row_equals_record(
        reader.iloc[0].to_dict(),
        by_key[("factor_pit_ext2026", "portfolio_metrics.reader.v2")],
    )
    legacy = result.tables["portfolio_metrics_vectorbt365_ext2026"]
    _published_row_equals_record(
        legacy.iloc[1].to_dict(),
        by_key[("factor_nonpit_diagnostic_ext2026", "portfolio_metrics.vectorbt365.v1")],
    )
    differential = result.tables["portfolio_metrics_differential_ext2026"]
    _published_row_equals_record(
        differential.iloc[0].to_dict(),
        by_key[("factor_nonpit_minus_pit_ext2026", "portfolio_metrics.differential.v2")],
    )
    crisis = result.tables["crisis_metrics_ext2026"]
    _published_row_equals_record(
        crisis.iloc[0].to_dict(),
        by_key[("factor_pit_ext2026", "crisis_metrics.boundary_anchored.v1")],
    )

    ai_variants = result.tables["tear_sheet_ai_variants_ext2026"]
    assert len(ai_variants) == 3  # PIT + non-PIT readers and their differential
    assert set(ai_variants["schema"]) == {
        "portfolio_metrics.reader.v2",
        "portfolio_metrics.differential.v2",
    }

    # every canonical row traces to verified producer lineage
    assert result.lineage["factor_run"]["run_id"] == case["factor_run_id"]
    assert result.lineage["factor_run"]["manifest_sha256"] == case["factor_sha"]
    assert result.lineage["market_snapshot"] == dict(
        factor.manifest["input_manifests"]["market_snapshot"]
    )
    assert all(
        source.startswith("scripts/extend_stream_2026.py:")
        for source in legacy["source"]
    )


def test_factor_tables_preserve_performance_only_rows_and_shortened_attribution(
    tmp_path,
):
    from tests.test_stream_ext2026 import _factor_metric_trusted_writer_case

    _, bundle, _, _, _ = _factor_metric_trusted_writer_case(tmp_path, shortened=True)
    bts = _reports_producer()
    lineage = {
        "factor_run": {"run_id": "factor_fixture_run", "manifest_sha256": "a" * 64},
        "market_snapshot": {
            "snapshot_id": bundle["market_snapshot"]["snapshot_id"],
            "manifest_sha256": bundle["market_snapshot"]["manifest_sha256"],
        },
    }

    result = bts.assemble_factor_report_tables(bundle, lineage=lineage)

    reader = result.tables["portfolio_metrics_reader_ext2026"]
    assert set(reader["row_kind"]) == {"performance_only"}
    assert not any(col.startswith("raw_market_model_") for col in reader.columns)
    by_key = {(row["portfolio_id"], row["schema"]): row for row in bundle["records"]}
    _published_row_equals_record(
        reader.iloc[0].to_dict(),
        by_key[("factor_pit_ext2026", "portfolio_metrics.reader.v2")],
    )

    # the shortened attribution survives as its OWN records with actual windows
    attribution = result.tables["attribution_raw_market_model_ext2026"]
    assert len(attribution) == 2
    assert (attribution["n_obs"] < reader["n_obs"].iloc[0]).all()
    assert (attribution["start"] > reader["start"].iloc[0]).all()
    assert (attribution["end"] == reader["end"].iloc[0]).all()


def test_factor_table_assembly_rejects_stream_divergence_and_extra_row_families(
    tmp_path,
):
    import copy

    import pandas as pd

    from tests.test_stream_ext2026 import _factor_metric_trusted_writer_case

    _, bundle, _, _, _ = _factor_metric_trusted_writer_case(tmp_path)
    bts = _reports_producer()
    lineage = {
        "factor_run": {"run_id": "factor_fixture_run", "manifest_sha256": "a" * 64},
        "market_snapshot": {
            "snapshot_id": bundle["market_snapshot"]["snapshot_id"],
            "manifest_sha256": bundle["market_snapshot"]["manifest_sha256"],
        },
    }
    bts.assemble_factor_report_tables(bundle, lineage=lineage)  # sanity: valid

    def assemble(tampered, tampered_lineage=None):
        return bts.assemble_factor_report_tables(
            tampered, lineage=tampered_lineage or lineage
        )

    # a record window diverging from its declared source stream
    tampered = copy.deepcopy(bundle)
    reader = tampered["records"][0]
    reader["start"] = pd.Timestamp(reader["start"]) + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="declared source stream"):
        assemble(tampered)

    # an extra, independently produced Factor row family is not a canonical input
    tampered = copy.deepcopy(bundle)
    tampered["records"].append(dict(tampered["records"][0]))
    with pytest.raises(ValueError, match="exactly once"):
        assemble(tampered)

    # differential drift against its declared spread stream
    tampered = copy.deepcopy(bundle)
    differential = next(
        row
        for row in tampered["records"]
        if row["schema"] == "portfolio_metrics.differential.v2"
    )
    differential["n_obs"] = int(differential["n_obs"]) - 1
    with pytest.raises(ValueError, match="declared source stream"):
        assemble(tampered)

    # attribution claiming more coverage than its declared stream
    tampered = copy.deepcopy(bundle)
    attribution = next(
        row
        for row in tampered["records"]
        if row["schema"] == "attribution.raw_market_model.v1"
    )
    attribution["n_obs"] = int(attribution["n_obs"]) + 1
    with pytest.raises(ValueError, match="attribution"):
        assemble(tampered)

    # a full reader may not hide a shortened attribution: claiming full over a
    # shorter attribution stream fails coherence
    shortened_dir = tmp_path / "shortened"
    shortened_dir.mkdir()
    _, shortened, _, _, _ = _factor_metric_trusted_writer_case(
        shortened_dir, shortened=True
    )
    tampered = copy.deepcopy(shortened)
    for row in tampered["records"]:
        if row["schema"] == "portfolio_metrics.reader.v2":
            row["row_kind"] = "full"
    with pytest.raises(ValueError, match="full|attribution"):
        assemble(
            tampered,
            tampered_lineage={
                "factor_run": lineage["factor_run"],
                "market_snapshot": {
                    "snapshot_id": shortened["market_snapshot"]["snapshot_id"],
                    "manifest_sha256": shortened["market_snapshot"]["manifest_sha256"],
                },
            },
        )

    # lineage divergence: rows may not claim a snapshot the records were not built on
    with pytest.raises(ValueError, match="lineage"):
        assemble(
            bundle,
            tampered_lineage={
                "factor_run": lineage["factor_run"],
                "market_snapshot": {
                    "snapshot_id": bundle["market_snapshot"]["snapshot_id"],
                    "manifest_sha256": "b" * 64,
                },
            },
        )


# --------------------------------------------------------------------------- #
# Task 9.3: canonical SJM report tables from the completed SJM v3 run.         #
# Task 9.4: canonical trio, static-window, and window-dashboard tables.        #
# Task 9.5: canonical ten-year / maximum-window Markowitz tables.              #
# All fixture-driven; no tracked file under data/ or reports/ is touched.      #
# --------------------------------------------------------------------------- #


_SJM_TEST_SSR = {"n_boot": 25, "seed": 17}


def _load_market(bts, case):
    return bts.load_market_report_input(
        case["snapshot_dir"],
        snapshot_id=case["snapshot_id"],
        manifest_sha256=case["snapshot_sha"],
    )


def _sjm_reports(bts, case, **overrides):
    sjm_input = bts.load_sjm_report_input(
        case["sjm_run_dir"],
        run_id=case["sjm_run_id"],
        manifest_sha256=case["sjm_sha"],
    )
    kwargs = {"ssr_settings": dict(_SJM_TEST_SSR)} | overrides
    return bts.build_sjm_report_tables(sjm_input, _load_market(bts, case), **kwargs)


def test_sjm_report_tables_project_the_completed_run_with_full_provenance(
    canonical_case,
):
    import numpy as np
    import pandas as pd

    from macro_framework.evaluation import crisis_metrics
    from macro_framework.skill_metric import raw_market_model_attribution
    from macro_framework.ssr import ssr_inference
    from tests.test_stream_ext2026 import _mod

    bts = _reports_producer()
    pub = _publisher()
    case = canonical_case
    result = _sjm_reports(bts, case)

    assert result.owner == bts.REPORT_TABLE_OWNER
    assert sorted(result.tables) == sorted(bts.SJM_REPORT_TABLE_SCHEMAS)
    by_name = {asset.public_basename: asset for asset in pub.DATA_V4_CATALOG.assets}
    for stem, schema_id in bts.SJM_REPORT_TABLE_SCHEMAS.items():
        assert by_name[f"{stem}.parquet"].schema_id == schema_id
        assert by_name[f"{stem}.parquet"].producer == bts.REPORT_TABLE_OWNER

    manifest = json.loads((case["sjm_run_dir"] / "manifest.json").read_text())
    # selected-configuration hash, protocol identity, cash benchmark, input
    # manifests, dates, counts, annualization (task 9.3 provenance bullet)
    assert result.protocol == manifest["protocol"]
    assert result.selected_config_sha256 == hashlib.sha256(
        json.dumps(
            manifest["selected_config"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert result.cash_benchmark == manifest["cash_benchmark"]
    assert result.coverage == manifest["coverage"]
    assert result.lineage["sjm_run"] == {
        "run_id": case["sjm_run_id"],
        "manifest_sha256": case["sjm_sha"],
    }
    assert result.lineage["factor_run"] == manifest["input_manifests"]["factor_run"]
    assert result.lineage["market_snapshot"] == (
        manifest["input_manifests"]["market_snapshot"]
    )
    assert result.inference_settings["n_boot"] == 25
    assert result.inference_settings["seed"] == 17
    assert result.inference_settings["window"] == 252
    assert result.inference_settings["periods_per_year"] == 252

    # every displayed SJM field reproduces from the persisted run + shared
    # macro_framework calculators alone — no notebook-local finance formulas
    table = result.tables["tear_sheet_sjm_crowding_ext2026"]
    overlay_id = result.portfolios["overlay"]
    control_id = result.portfolios["control"]
    assert overlay_id == case["sjm_run_id"]
    value = pd.read_parquet(case["sjm_run_dir"] / "sjm_equity.parquet")["value"]
    returns = pd.read_parquet(case["sjm_run_dir"] / "sjm_daily_returns.parquet")
    r = value.pct_change().dropna()
    cash = returns["cash_return"]
    excess = r - cash
    expected_sharpe = float(excess.mean() / excess.std(ddof=1)) * float(np.sqrt(252))
    expected_ssr = ssr_inference(excess, n_boot=25, seed=17)

    readers = table[table["schema"] == "portfolio_metrics.reader.v2"]
    assert set(readers["portfolio_id"]) == {overlay_id, control_id}
    overlay_row = readers[readers["portfolio_id"] == overlay_id].iloc[0]
    assert overlay_row["sharpe"] == pytest.approx(expected_sharpe)
    assert overlay_row["ssr_ssr"] == pytest.approx(expected_ssr.result.ssr)
    assert (overlay_row["start"], overlay_row["end"]) == (r.index[0], r.index[-1])
    assert overlay_row["n_obs"] == len(r)
    assert overlay_row["periods_per_year"] == 252
    assert overlay_row["cash_benchmark_id"] == f"BIL@{case['snapshot_id']}"
    assert overlay_row["currency_basis"] == "legacy_mixed_local_quotes"
    # tail block: the schema's downside/drawdown vocabulary rides on the row
    assert overlay_row["downside_rms"] == pytest.approx(
        float(np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2)))
    )
    assert overlay_row["maxdd"] == pytest.approx(float((value / value.cummax() - 1).min()))

    # crisis records reproduce the shared boundary-inclusive result verbatim
    crisis = crisis_metrics(value, "2022-01-01", "2022-12-31")
    crisis_rows = table[table["schema"] == "crisis_metrics.boundary_anchored.v1"]
    assert set(crisis_rows["portfolio_id"]) == {overlay_id, control_id}
    overlay_crisis = crisis_rows[crisis_rows["portfolio_id"] == overlay_id].iloc[0]
    assert overlay_crisis["episode_return"] == crisis.episode_return
    assert (
        overlay_crisis["boundary_anchored_max_drawdown"]
        == crisis.boundary_anchored_max_drawdown
    )

    # raw market-model rows come from the shared attribution on snapshot SPY
    ext = _mod()
    market_returns, _ = ext.load_completed_snapshot_market_returns(
        case["snapshot_dir"], r.index, value_index=value.index
    )
    expected_attribution = raw_market_model_attribution(
        r.loc[market_returns.index], market_returns
    )
    attribution_rows = table[table["schema"] == "attribution.raw_market_model.v1"]
    assert set(attribution_rows["portfolio_id"]) == {overlay_id, control_id}
    overlay_attribution = attribution_rows[
        attribution_rows["portfolio_id"] == overlay_id
    ].iloc[0]
    assert overlay_attribution["raw_market_model_beta"] == pytest.approx(
        expected_attribution.beta
    )
    assert overlay_attribution["n_obs"] == expected_attribution.n_obs
    expected_kind = "full" if len(market_returns) == len(r) else "performance_only"
    assert overlay_row["row_kind"] == expected_kind

    # window disclosure: this run ends inside development, so holdout is
    # labeled absent — never fabricated (performance/holdout row families)
    assert result.windows["development"] == {"coincides_with": "full"}
    assert result.windows["holdout"]["available"] is False
    assert result.windows["full"]["n_obs"] == len(r)


def test_sjm_report_windows_split_and_input_gates(canonical_case, tmp_path):
    import shutil

    import pandas as pd

    bts = _reports_producer()
    case = canonical_case
    dev_end = pd.Timestamp("2024-06-30")

    spanning = bts.sjm_report_windows(
        pd.Timestamp("2021-01-05"), pd.Timestamp("2026-06-30"), dev_end=dev_end
    )
    assert spanning["full"]["start"] == pd.Timestamp("2021-01-05")
    assert spanning["full"]["end"] == pd.Timestamp("2026-06-30")
    assert spanning["development"] == {
        "start": pd.Timestamp("2021-01-05"),
        "end": dev_end,
    }
    assert spanning["holdout"] == {
        "start": pd.Timestamp("2024-07-01"),
        "end": pd.Timestamp("2026-06-30"),
    }

    dev_only = bts.sjm_report_windows(
        pd.Timestamp("2021-01-05"), pd.Timestamp("2023-06-30"), dev_end=dev_end
    )
    assert dev_only["development"] == {"coincides_with": "full"}
    assert dev_only["holdout"]["available"] is False

    holdout_only = bts.sjm_report_windows(
        pd.Timestamp("2025-01-02"), pd.Timestamp("2026-06-30"), dev_end=dev_end
    )
    assert holdout_only["development"]["available"] is False
    assert holdout_only["holdout"] == {"coincides_with": "full"}

    sjm_input = bts.load_sjm_report_input(
        case["sjm_run_dir"], run_id=case["sjm_run_id"], manifest_sha256=case["sjm_sha"]
    )
    market = _load_market(bts, case)

    # wrong input family in either slot fails before any table exists
    with pytest.raises(ValueError, match="family"):
        bts.build_sjm_report_tables(market, market, ssr_settings=_SJM_TEST_SSR)
    with pytest.raises(ValueError, match="family"):
        bts.build_sjm_report_tables(sjm_input, sjm_input, ssr_settings=_SJM_TEST_SSR)

    # a completed snapshot that is NOT the run's recorded lineage is refused
    divergent = tmp_path / case["snapshot_dir"].name
    shutil.copytree(case["snapshot_dir"], divergent)
    manifest = json.loads((divergent / "manifest.json").read_text())
    manifest["overlap_revisions"]["note"] = "resigned divergent copy"
    (divergent / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    divergent_market = bts.load_market_report_input(
        divergent,
        snapshot_id=case["snapshot_id"],
        manifest_sha256=hashlib.sha256(
            (divergent / "manifest.json").read_bytes()
        ).hexdigest(),
    )
    with pytest.raises(ValueError, match="lineage"):
        bts.build_sjm_report_tables(
            sjm_input, divergent_market, ssr_settings=_SJM_TEST_SSR
        )

    # a persisted stream mutated AFTER the gated load can never reach a table
    tampered_dir = tmp_path / "tampered_run"
    shutil.copytree(case["sjm_run_dir"], tampered_dir)
    tampered_input = bts.load_sjm_report_input(
        tampered_dir,
        run_id=case["sjm_run_id"],
        manifest_sha256=hashlib.sha256(
            (tampered_dir / "manifest.json").read_bytes()
        ).hexdigest(),
    )
    returns_path = tampered_dir / "sjm_daily_returns.parquet"
    returns_path.write_bytes(returns_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="mutated after inventory"):
        bts.build_sjm_report_tables(
            tampered_input, market, ssr_settings=_SJM_TEST_SSR
        )


def _static_specs(bts, case):
    """Two fresh-buy ladder rungs inside the fixture snapshot's coverage."""
    import pandas as pd

    manifest = json.loads((case["sjm_run_dir"] / "manifest.json").read_text())
    start = pd.Timestamp(manifest["coverage"]["anchor"])
    end = pd.Timestamp(manifest["coverage"]["end"])
    return (
        bts.StaticWindowSpec("full fixture window", start, end),
        bts.StaticWindowSpec("late fresh buy", pd.Timestamp("2022-01-10"), end),
    )


def test_trio_static_and_dashboard_tables_share_one_validated_identity(
    canonical_case,
):
    import numpy as np
    import pandas as pd

    bts = _reports_producer()
    pub = _publisher()
    case = canonical_case
    factor = _load_factor(bts, case)
    sjm_reports = _sjm_reports(bts, case)
    market = _load_market(bts, case)
    rung_full, rung_late = _static_specs(bts, case)

    result = bts.build_trio_report_tables(
        factor,
        sjm_reports,
        market,
        static_windows=(rung_full, rung_late),
        trio_static_window=rung_full,
        ssr_settings=_SJM_TEST_SSR,
    )

    assert result.owner == bts.REPORT_TABLE_OWNER
    assert sorted(result.tables) == sorted(bts.TRIO_REPORT_TABLE_SCHEMAS)
    by_name = {asset.public_basename: asset for asset in pub.DATA_V4_CATALOG.assets}
    for stem, schema_id in bts.TRIO_REPORT_TABLE_SCHEMAS.items():
        assert by_name[f"{stem}.parquet"].schema_id == schema_id
        assert by_name[f"{stem}.parquet"].producer == bts.REPORT_TABLE_OWNER

    # the trio table carries exactly the three canonical lines, and each row IS
    # its validated source row — never a second recalculated family
    trio = result.tables["tear_sheet_trio_ext2026"]
    assert len(trio) == 3
    records = factor.metric_records["records"]
    by_key = {(row["portfolio_id"], row["schema"]): row for row in records}
    factor_row = trio[trio["portfolio_id"] == "factor_pit_ext2026"].iloc[0]
    _published_row_equals_record(
        factor_row.to_dict(),
        by_key[("factor_pit_ext2026", "portfolio_metrics.reader.v2")],
    )
    overlay_id = sjm_reports.portfolios["overlay"]
    sjm_table = sjm_reports.tables["tear_sheet_sjm_crowding_ext2026"]
    sjm_full = sjm_table[
        (sjm_table["portfolio_id"] == overlay_id)
        & (sjm_table["schema"] == "portfolio_metrics.reader.v2")
    ].iloc[0]
    trio_sjm = trio[trio["portfolio_id"] == overlay_id].iloc[0]
    assert trio_sjm["total_return"] == sjm_full["total_return"]
    assert trio_sjm["window_label"] == sjm_full["window_label"]
    # one cash benchmark and one currency basis across the whole trio
    assert set(trio["cash_benchmark_id"]) == {f"BIL@{case['snapshot_id']}"}
    assert set(trio["currency_basis"]) == {"legacy_mixed_local_quotes"}

    # static rungs: exact fresh-buy window identity, reproducible from the
    # snapshot alone; Sharpe is CASH-EXCESS based (the corrected convention)
    static = result.tables["tear_sheet_static_bh_windows"]
    assert list(static["window_label"]) == [rung_full.label, rung_late.label]
    basket = pd.read_parquet(
        case["snapshot_dir"] / "basket_adjusted_close_local.parquet"
    )
    cash_market = pd.read_parquet(
        case["snapshot_dir"] / "cash_market_total_return.parquet"
    )
    common = pd.concat([basket, cash_market[["BIL"]]], axis=1).dropna()
    sliced = common.loc[rung_late.start : rung_late.end]
    value = 0.25 * (sliced / sliced.iloc[0]).sum(axis=1)
    late_row = static.iloc[1]
    assert late_row["start"] == value.index[1]
    assert late_row["end"] == value.index[-1]
    assert late_row["n_obs"] == len(value) - 1
    assert late_row["total_return"] == pytest.approx(
        float(value.iloc[-1] / value.iloc[0] - 1)
    )
    r = value.pct_change().dropna()
    bil = sliced["BIL"].pct_change().dropna()
    excess = r - bil
    raw_sharpe = float(r.mean() / r.std(ddof=1)) * float(np.sqrt(252))
    excess_sharpe = float(excess.mean() / excess.std(ddof=1)) * float(np.sqrt(252))
    assert late_row["sharpe"] == pytest.approx(excess_sharpe)
    assert late_row["sharpe"] != pytest.approx(raw_sharpe)

    # the dashboard table reuses the SAME reader rows plus per-window raw
    # market-model records — repeated portfolio/window rows agree exactly
    dashboard = result.tables["tear_sheet_static_bh_window_dashboard"]
    dash_readers = dashboard[dashboard["schema"] == "portfolio_metrics.reader.v2"]
    shared = [c for c in static.columns if c in dash_readers.columns]
    pd.testing.assert_frame_equal(
        dash_readers[shared].reset_index(drop=True),
        static[shared].reset_index(drop=True),
        check_dtype=False,  # NaN-padding from attribution records widens dtypes
    )
    dash_attribution = dashboard[
        dashboard["schema"] == "attribution.raw_market_model.v1"
    ]
    assert len(dash_attribution) == 2
    assert set(dash_attribution["window_label"]) == {
        rung_full.label,
        rung_late.label,
    }


def test_trio_tables_reject_lineage_divergence_and_tampered_component_rows(
    canonical_case, tmp_path
):
    import dataclasses
    import shutil

    import pandas as pd

    bts = _reports_producer()
    case = canonical_case
    factor = _load_factor(bts, case)
    sjm_reports = _sjm_reports(bts, case)
    market = _load_market(bts, case)
    rung_full, rung_late = _static_specs(bts, case)
    build_kwargs = dict(
        static_windows=(rung_full, rung_late),
        trio_static_window=rung_full,
        ssr_settings=_SJM_TEST_SSR,
    )

    # (a) a market snapshot that is not the recorded lineage of BOTH producers
    divergent = tmp_path / case["snapshot_dir"].name
    shutil.copytree(case["snapshot_dir"], divergent)
    manifest = json.loads((divergent / "manifest.json").read_text())
    manifest["overlap_revisions"]["note"] = "resigned divergent copy"
    (divergent / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    divergent_market = bts.load_market_report_input(
        divergent,
        snapshot_id=case["snapshot_id"],
        manifest_sha256=hashlib.sha256(
            (divergent / "manifest.json").read_bytes()
        ).hexdigest(),
    )
    with pytest.raises(ValueError, match="lineage"):
        bts.build_trio_report_tables(
            factor, sjm_reports, divergent_market, **build_kwargs
        )

    # (b) a doctored SJM component whose rows disagree with its own tables
    tampered_rows = []
    for row in sjm_reports.rows:
        row = dict(row)
        if (
            row["schema"] == "portfolio_metrics.reader.v2"
            and row["portfolio_id"] == sjm_reports.portfolios["overlay"]
        ):
            row["total_return"] = float(row["total_return"]) + 0.05
        tampered_rows.append(row)
    doctored = dataclasses.replace(sjm_reports, rows=tuple(tampered_rows))
    with pytest.raises(ValueError, match="match|disagree"):
        bts.build_trio_report_tables(factor, doctored, market, **build_kwargs)

    # (c) a doctored SJM lineage claiming a different factor run
    wrong_factor = dict(sjm_reports.lineage)
    wrong_factor["factor_run"] = {
        "run_id": "factor_ext2026_other_v9",
        "manifest_sha256": "c" * 64,
    }
    relabeled = dataclasses.replace(sjm_reports, lineage=wrong_factor)
    with pytest.raises(ValueError, match="lineage"):
        bts.build_trio_report_tables(factor, relabeled, market, **build_kwargs)

    # (d) a static window with no eligible common trading days fails loudly
    dead_window = bts.StaticWindowSpec(
        "outside coverage", pd.Timestamp("1999-01-04"), pd.Timestamp("1999-12-31")
    )
    with pytest.raises(ValueError, match="common|coverage|eligible"):
        bts.build_trio_report_tables(
            factor,
            sjm_reports,
            market,
            static_windows=(dead_window,),
            trio_static_window=dead_window,
            ssr_settings=_SJM_TEST_SSR,
        )


@pytest.fixture(scope="module")
def markowitz_case(tmp_path_factory):
    """One REAL completed market snapshot (producer-built, quote metadata and
    marker-carried manifest sha) with a late-listing SWDA.L so the maximum
    supported window is governed by the youngest asset."""
    import dataclasses
    import importlib

    import numpy as np
    import pandas as pd

    bbl = importlib.import_module("scripts.build_basket_long")
    tmp = tmp_path_factory.mktemp("markowitz_reports_case")
    digest = hashlib.sha256(b"markowitz-offline-fixture").hexdigest()
    contract = dataclasses.replace(
        bbl.make_snapshot_contract(vintage_date="2026-07-03"),
        etf_raw_response_sha256=digest,
        fx_raw_response_sha256=digest,
    )
    index = pd.bdate_range("2024-01-01", "2024-07-19")
    rng = np.random.default_rng(99)

    def levels(start, scale):
        return start * np.cumprod(1.0 + rng.normal(0.0004, scale, len(index)))

    swda = pd.Series(levels(8_000.0, 0.009), index=index)
    swda.iloc[:15] = np.nan  # lists late: governs the max supported window
    basket = pd.DataFrame(
        {
            "SWDA.L": swda,
            "XLK": levels(200.0, 0.011),
            "IAU": levels(40.0, 0.008),
        },
        index=index,
    )
    cash_market = pd.DataFrame(
        {
            "BIL": 100.0 * np.cumprod(np.full(len(index), 1.0 + 2e-4)),
            "SPY": levels(500.0, 0.010),
        },
        index=index,
    )
    fx = pd.DataFrame(
        {"USD_per_GBP": 1.25 + 0.03 * np.sin(np.arange(len(index)) / 9.0)},
        index=index,
    )
    data = bbl.NormalizedSnapshotData(
        basket_local=basket, cash_market=cash_market, fx=fx, coverage={}
    )
    snapshot_dir = bbl.build_market_snapshot(
        snapshot_id=contract.snapshot_id,
        requested_start=contract.requested_start,
        requested_end=contract.requested_end,
        output_root=tmp,
        contract=contract,
        data=data,
        build_time="2026-07-28T12:00:00+00:00",
    )
    return {
        "snapshot_dir": snapshot_dir,
        "snapshot_id": contract.snapshot_id,
        "snapshot_sha": hashlib.sha256(
            (snapshot_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "first_swda_date": pd.Timestamp(swda.dropna().index[0]),
    }


def _markowitz_inputs(bts, case):
    markowitz_input = bts.load_markowitz_report_input(
        case["snapshot_dir"],
        snapshot_id=case["snapshot_id"],
        manifest_sha256=case["snapshot_sha"],
    )
    market_input = bts.load_market_report_input(
        case["snapshot_dir"],
        snapshot_id=case["snapshot_id"],
        manifest_sha256=case["snapshot_sha"],
    )
    return markowitz_input, market_input


def _synthetic_reader_row(portfolio_id, start, end, n_obs, **overrides):
    from macro_framework.reporting import READER_SCHEMA, validate_report_row

    row = {
        "schema": READER_SCHEMA,
        "portfolio_id": portfolio_id,
        "return_basis": "adjusted_total_return_equity",
        "window_label": f"{start}..{end}",
        "start": start,
        "end": end,
        "n_obs": n_obs,
        "periods_per_year": 252,
        "cash_benchmark_id": "BIL@market_total_return_fx_2026-06-30_v1",
        "currency_basis": "legacy_mixed_local_quotes",
        "source": "fixture:ai_line",
        "row_kind": "performance_only",
        "total_return": 0.05,
        "cagr": 0.04,
        "ann_vol": 0.10,
        "sharpe": 0.5,
        "sortino": 0.7,
        "calmar": 0.6,
        "maxdd": -0.08,
        "downside_rms": 0.005,
    }
    row.update(overrides)
    return validate_report_row(row)


_STATIC_TEST_SSR = {"window": 60, "n_boot": 25, "seed": 7}


def test_markowitz_tables_disclose_windows_moments_weights_and_diagnostics(
    markowitz_case,
):
    import numpy as np
    import pandas as pd

    from macro_framework.markowitz import (
        WEEKLY_PERIODS_PER_YEAR,
        annualized_moments,
        weekly_usd_valuations,
    )

    bts = _reports_producer()
    pub = _publisher()
    case = markowitz_case
    markowitz_input, market_input = _markowitz_inputs(bts, case)

    requested_end = pd.Timestamp("2024-07-19")
    max_start = bts.markowitz_max_supported_start(
        markowitz_input, requested_end=requested_end
    )
    # the youngest asset (SWDA.L, first observed 2024-01-22) governs
    assert max_start == pd.Timestamp("2024-01-26")
    ten = (pd.Timestamp("2024-03-01"), requested_end)

    static_10y, _ = bts.build_static_bh_rows(
        market_input,
        bts.StaticWindowSpec("10y panel", *ten),
        ssr_settings=_STATIC_TEST_SSR,
    )
    static_max, _ = bts.build_static_bh_rows(
        market_input,
        bts.StaticWindowSpec("max panel", max_start, requested_end),
        ssr_settings=_STATIC_TEST_SSR,
    )
    shorter_ai = _synthetic_reader_row(
        "factor_pit_ext2026",
        pd.Timestamp("2024-04-01"),
        pd.Timestamp("2024-07-19"),
        79,
    )
    result = bts.build_markowitz_report_tables(
        markowitz_input,
        requested_windows={"10y": ten, "max": (max_start, requested_end)},
        trio_rows={
            "10y": [static_10y, shorter_ai],
            "max": [static_max, shorter_ai],
        },
        n_points=7,
    )

    assert result.owner == bts.REPORT_TABLE_OWNER
    assert sorted(result.tables) == sorted(bts.MARKOWITZ_REPORT_TABLE_SCHEMAS)
    by_name = {asset.public_basename: asset for asset in pub.DATA_V4_CATALOG.assets}
    for stem, schema_id in bts.MARKOWITZ_REPORT_TABLE_SCHEMAS.items():
        assert by_name[f"{stem}.parquet"].schema_id == schema_id
        assert by_name[f"{stem}.parquet"].producer == bts.REPORT_TABLE_OWNER
    assert result.base_currency == "USD"
    assert result.lineage["market_snapshot"] == {
        "snapshot_id": case["snapshot_id"],
        "manifest_sha256": case["snapshot_sha"],
    }

    # moments reproduce macro_framework.markowitz exactly — identity, windows,
    # weekly count, annualization, and source-date hashes all disclosed on rows
    universe = tuple(bts.markowitz_quote_specs())
    moments = annualized_moments(
        weekly_usd_valuations(
            case["snapshot_dir"],
            quote_specs=bts.markowitz_quote_specs(),
            requested_start=ten[0],
            requested_end=ten[1],
        )
    )
    table = result.tables["markowitz_10y_moments"]
    assert list(table.columns) == list(bts.markowitz_moments_columns(universe))
    assert list(table["asset"]) == list(universe)
    for _, row in table.iterrows():
        assert row["snapshot_id"] == case["snapshot_id"]
        assert row["base_currency"] == "USD"
        assert row["requested_start"] == ten[0]
        assert row["requested_end"] == ten[1]
        assert row["actual_start"] == moments.start
        assert row["actual_end"] == moments.end
        assert row["n_obs"] == moments.n_obs
        assert row["periods_per_year"] == WEEKLY_PERIODS_PER_YEAR
        assert len(row["source_dates_sha256"]) == 64
        assert row["mean_ann_arithmetic"] == pytest.approx(
            float(moments.mean_ann_arithmetic[row["asset"]])
        )
        assert row["vol_ann"] == pytest.approx(
            float(np.sqrt(moments.covariance_ann.loc[row["asset"], row["asset"]]))
        )
    windows_meta = result.windows["10y"]
    assert windows_meta["requested_start"] == ten[0]
    assert windows_meta["actual_start"] == moments.start
    assert windows_meta["n_obs"] == moments.n_obs

    # frontier rows: every attempted target retains solver diagnostics and its
    # complete weight vector; feasible rows respect the residual tolerance
    frontier = result.tables["markowitz_10y_frontier"]
    assert list(frontier.columns) == list(bts.markowitz_frontier_columns(universe))
    assert len(frontier) == 7
    assert "portfolio_id" not in frontier.columns
    weight_columns = [f"weight_{asset}" for asset in universe]
    feasible = frontier[frontier["feasible"]]
    assert len(feasible) >= 2
    assert np.allclose(feasible[weight_columns].sum(axis=1), 1.0, atol=1e-6)
    assert (feasible["budget_residual"].abs() <= 1e-8).all()
    assert set(frontier["status"].map(int)) is not None
    assert frontier["message"].map(str).notna().all()

    # trio panels: validated mixed-local strategy rows, shorter coverage LABELED
    trio_10y = result.tables["tear_sheet_trio_10y"]
    assert set(trio_10y["portfolio_id"]) == {
        static_10y["portfolio_id"],
        "factor_pit_ext2026",
    }
    coverage = result.windows["10y"]["trio_coverage"]
    assert coverage[static_10y["portfolio_id"]]["coverage"] == "spans_requested_start"
    assert coverage["factor_pit_ext2026"]["coverage"] == "shorter_than_requested"

    # the German-locale source schemas are producer-owned data, not notebook code
    assert bts.GERMAN_LOCALE_CSV_SPEC["sep"] == ";"
    assert bts.GERMAN_LOCALE_CSV_SPEC["decimal"] == ","
    assert bts.GERMAN_LOCALE_CSV_SPEC["float_format"] == "%.8f"
    assert bts.REPORT_CSV_LOCALE_SPECS["de-DE"] == bts.GERMAN_LOCALE_CSV_SPEC
    assert list(trio_10y.columns) == list(
        bts.report_row_table_columns(list(result.rows["tear_sheet_trio_10y"]))
    )


def test_markowitz_tables_reject_strategy_points_on_usd_frontiers_and_bad_windows(
    markowitz_case,
):
    import pandas as pd

    from macro_framework.markowitz import weekly_usd_valuations

    bts = _reports_producer()
    case = markowitz_case
    markowitz_input, market_input = _markowitz_inputs(bts, case)
    requested_end = pd.Timestamp("2024-07-19")
    ten = (pd.Timestamp("2024-03-01"), requested_end)
    static_10y, _ = bts.build_static_bh_rows(
        market_input,
        bts.StaticWindowSpec("10y panel", *ten),
        ssr_settings=_STATIC_TEST_SSR,
    )

    def build(**overrides):
        kwargs = dict(
            requested_windows={"10y": ten},
            trio_rows={"10y": [static_10y]},
            n_points=5,
        )
        kwargs.update(overrides)
        return bts.build_markowitz_report_tables(markowitz_input, **kwargs)

    build()  # sanity: the base case is valid

    # a USD-claiming strategy row can never join the mixed-local panels — the
    # USD frontier carries no strategy points (R5.5, task 9.5)
    usd_strategy = _synthetic_reader_row(
        "factor_pit_ext2026",
        pd.Timestamp("2024-04-01"),
        pd.Timestamp("2024-07-19"),
        79,
        currency_basis="USD",
    )
    with pytest.raises(ValueError, match="USD frontier|mixed-local"):
        build(trio_rows={"10y": [static_10y, usd_strategy]})

    # a trio row claiming coverage beyond the requested window is refused
    overreach = _synthetic_reader_row(
        "factor_pit_ext2026",
        pd.Timestamp("2024-04-01"),
        pd.Timestamp("2024-08-30"),
        100,
    )
    with pytest.raises(ValueError, match="requested window"):
        build(trio_rows={"10y": [static_10y, overreach]})

    # only the two canonical window names exist
    with pytest.raises(ValueError, match="10y|max"):
        build(requested_windows={"weekly": ten}, trio_rows={"weekly": [static_10y]})

    # a requested window the snapshot cannot support fails instead of shrinking
    with pytest.raises(ValueError):
        build(requested_windows={"10y": (pd.Timestamp("2010-01-04"), requested_end)})

    # wrong input family fails before any optimization
    with pytest.raises(ValueError, match="family"):
        bts.build_markowitz_report_tables(
            market_input,
            requested_windows={"10y": ten},
            trio_rows={"10y": [static_10y]},
            n_points=5,
        )

    # honesty of the maximum supported window: one week earlier has no eligible
    # SWDA.L observation, so the shared validator itself refuses it
    with pytest.raises(ValueError, match="eligible|stale"):
        weekly_usd_valuations(
            case["snapshot_dir"],
            quote_specs=bts.markowitz_quote_specs(),
            requested_start=pd.Timestamp("2024-01-19"),
            requested_end=requested_end,
        )


# --------------------------------------------------------------------------- #
# Task 9.6: canonical risk, attribution, crisis, and monthly-return tables.    #
# Task 9.7: deterministic US and German locale mirrors (export_csv_mirrors).   #
# Task 9.8: canonical report and mirror integration checks (AC 7.1, AC 8.6).   #
# Fixture-driven; no tracked file under data/ or reports/ is read or written.  #
# --------------------------------------------------------------------------- #


def _mirror_exporter():
    import importlib

    return importlib.import_module("scripts.export_csv_mirrors")


def test_auxiliary_tables_reconcile_to_canonical_portfolio_rows(canonical_case):
    import pandas as pd

    bts = _reports_producer()
    pub = _publisher()
    case = canonical_case
    factor = _load_factor(bts, case)

    result = bts.build_auxiliary_report_tables(factor)

    assert result.owner == bts.REPORT_TABLE_OWNER
    assert sorted(result.tables) == sorted(bts.AUXILIARY_REPORT_TABLE_SCHEMAS)
    by_name = {asset.public_basename: asset for asset in pub.DATA_V4_CATALOG.assets}
    for stem, schema_id in bts.AUXILIARY_REPORT_TABLE_SCHEMAS.items():
        assert by_name[f"{stem}.parquet"].schema_id == schema_id
        assert by_name[f"{stem}.parquet"].producer == bts.REPORT_TABLE_OWNER
    # one producer call carries the complete secondary family, reconciled
    assert sorted(result.factor_tables.tables) == sorted(
        bts.FACTOR_REPORT_TABLE_SCHEMAS
    )
    assert result.lineage == dict(result.factor_tables.lineage)
    assert result.lineage["factor_run"]["run_id"] == case["factor_run_id"]

    records = factor.metric_records["records"]
    by_key = {(row["portfolio_id"], row["schema"]): row for row in records}
    portfolios = ("factor_pit_ext2026", "factor_nonpit_diagnostic_ext2026")

    # monthly returns: SAME validated stream, portfolio/window identity on
    # every record, and exact recompounding to the canonical reader row
    monthly = result.tables["monthly_returns_ext2026"]
    assert set(monthly["schema"]) == {"monthly_returns.reader.v1"}
    assert set(monthly["periods_per_year"]) == {12}
    for portfolio_id in portfolios:
        reader = by_key[(portfolio_id, "portfolio_metrics.reader.v2")]
        months = monthly[monthly["portfolio_id"] == portfolio_id]
        assert set(months["window_label"]) == {reader["window_label"]}
        assert set(months["return_basis"]) == {reader["return_basis"]}
        assert set(months["cash_benchmark_id"]) == {reader["cash_benchmark_id"]}
        assert set(months["currency_basis"]) == {reader["currency_basis"]}
        assert months["start"].min() == pd.Timestamp(reader["start"])
        assert months["end"].max() == pd.Timestamp(reader["end"])
        assert int(months["n_obs"].sum()) == int(reader["n_obs"])
        compounded = float((1.0 + months["monthly_return"]).prod() - 1.0)
        assert compounded == pytest.approx(float(reader["total_return"]), rel=1e-9)
        # per-month provenance points into the manifest-inventoried stream
        assert all(
            source.startswith(f"factor_run:{case['factor_run_id']}/")
            for source in months["source"]
        )

    # risk decomposition: a pure raw market-model projection that reconciles
    # with the attribution records AND the reader rows' embedded attribution
    risk = result.tables["risk_decomposition_ext2026"]
    assert list(risk.columns) == list(bts.risk_decomposition_columns())
    assert set(risk["schema"]) == {bts.RISK_DECOMPOSITION_SCHEMA}
    assert set(risk["source_schema"]) == {"attribution.raw_market_model.v1"}
    assert not any(
        "capm" in column.lower() or "jensen" in column.lower()
        for column in risk.columns
    )
    for portfolio_id in portfolios:
        attribution = by_key[(portfolio_id, "attribution.raw_market_model.v1")]
        reader = by_key[(portfolio_id, "portfolio_metrics.reader.v2")]
        row = risk[risk["portfolio_id"] == portfolio_id].iloc[0].to_dict()
        for key in bts.RISK_PROJECTED_ATTRIBUTION_FIELDS:
            assert row[key] == attribution[key], key
        # shortened-or-exact attribution window identity kept on the record
        assert row["window_label"] == attribution["window_label"]
        assert row["start"] == pd.Timestamp(attribution["start"])
        assert row["end"] == pd.Timestamp(attribution["end"])
        assert int(row["n_obs"]) == int(attribution["n_obs"])
        assert row["source"] == attribution["source"]
        assert row["systematic_variance_share"] == attribution["raw_market_model_r2"]
        assert row["idiosyncratic_variance_share"] == pytest.approx(
            1.0 - float(attribution["raw_market_model_r2"])
        )
        if reader["row_kind"] == "full":
            assert row["raw_market_model_beta"] == reader["raw_market_model_beta"]

    # boundary-inclusive crisis values pass through the family untouched
    crisis_table = result.factor_tables.tables["crisis_metrics_ext2026"]
    _published_row_equals_record(
        crisis_table.iloc[0].to_dict(),
        by_key[("factor_pit_ext2026", "crisis_metrics.boundary_anchored.v1")],
    )


def test_auxiliary_tables_reject_divergent_streams_and_stale_reader_rows(
    canonical_case, tmp_path
):
    import copy
    import dataclasses
    import shutil

    import pandas as pd

    bts = _reports_producer()
    case = canonical_case
    factor = _load_factor(bts, case)
    bts.build_auxiliary_report_tables(factor)  # sanity: the untouched run builds

    def doctored(mutate):
        metric_records = copy.deepcopy(dict(factor.metric_records))
        mutate(metric_records)
        return dataclasses.replace(factor, metric_records=metric_records)

    def record(metric_records, portfolio_id, schema):
        return next(
            row
            for row in metric_records["records"]
            if (row["portfolio_id"], row["schema"]) == (portfolio_id, schema)
        )

    # (a) a stale published reader row its own persisted stream no longer
    # reproduces fails monthly reconciliation
    def stale_reader(metric_records):
        row = record(
            metric_records, "factor_pit_ext2026", "portfolio_metrics.reader.v2"
        )
        row["total_return"] = float(row["total_return"]) + 0.02

    with pytest.raises(ValueError, match="stale generated values"):
        bts.build_auxiliary_report_tables(doctored(stale_reader))

    # (b) an attribution record that no longer reconciles with the canonical
    # reader row's embedded attribution
    def divergent_attribution(metric_records):
        row = record(
            metric_records, "factor_pit_ext2026", "attribution.raw_market_model.v1"
        )
        row["raw_market_model_beta"] = float(row["raw_market_model_beta"]) + 0.25

    with pytest.raises(ValueError, match="reconcile"):
        bts.build_auxiliary_report_tables(doctored(divergent_attribution))

    # (c) a declared stream artifact outside the manifest inventory
    def uninventoried_stream(metric_records):
        metric_records["source_streams"]["factor_pit_ext2026"]["artifact"] = (
            "handmade_equity.parquet"
        )

    with pytest.raises(ValueError, match="inventory"):
        bts.build_auxiliary_report_tables(doctored(uninventoried_stream))

    # (d) the WRONG portfolio's stream (same window shape, different values)
    # cannot stand in for the declared source stream
    def swapped_stream(metric_records):
        metric_records["source_streams"]["factor_pit_ext2026"]["artifact"] = (
            "factor_nonpit_diagnostic_equity_ext2026.parquet"
        )

    with pytest.raises(ValueError, match="stale generated values"):
        bts.build_auxiliary_report_tables(doctored(swapped_stream))

    # (e) persisted stream bytes mutated after inventory fail under the hash
    run = tmp_path / case["factor_run_dir"].name
    shutil.copytree(case["factor_run_dir"], run)
    loaded = bts.load_factor_report_input(
        run, run_id=case["factor_run_id"], manifest_sha256=case["factor_sha"]
    )
    equity_path = run / "factor_equity_ext2026.parquet"
    frame = pd.read_parquet(equity_path)
    frame["value"] = frame["value"] * 1.01
    frame.to_parquet(equity_path)
    with pytest.raises(ValueError, match="mutated after inventory"):
        bts.build_auxiliary_report_tables(loaded)


def test_locale_mirrors_round_trip_deterministically_and_cover_the_catalog(
    canonical_case, tmp_path
):
    import inspect

    import pandas as pd

    bts = _reports_producer()
    mirrors = _mirror_exporter()
    case = canonical_case
    factor = _load_factor(bts, case)
    auxiliary = bts.build_auxiliary_report_tables(factor)
    tables = dict(auxiliary.factor_tables.tables) | dict(auxiliary.tables)
    # a mixed-schema table (reader + attribution + crisis rows) exercises the
    # object-column date and null projections
    tables |= dict(_sjm_reports(bts, case).tables)

    out = tmp_path / "mirrors"
    written = mirrors.write_locale_mirrors(tables, out)
    for stem in tables:
        assert written[stem]["en-US"] == out / f"{stem}.csv"
        assert written[stem]["de-DE"] == out / f"{stem}_de.csv"

    reader = tables["portfolio_metrics_reader_ext2026"]
    us_first = (out / "portfolio_metrics_reader_ext2026.csv").read_text().splitlines()[1]
    de_first = (
        (out / "portfolio_metrics_reader_ext2026_de.csv").read_text().splitlines()[1]
    )
    eight_decimals = f"{float(reader['total_return'].iloc[0]):.8f}"
    assert eight_decimals in us_first  # comma/dot US file, eight decimals
    assert eight_decimals.replace(".", ",") in de_first  # semicolon/comma German
    assert ";" in de_first
    assert str(pd.Timestamp(reader["start"].iloc[0]).date()) in us_first  # ISO dates
    assert tables["tear_sheet_ai_variants_ext2026"].isna().any().any()  # nulls exercised

    # deterministic: re-exporting reproduces byte-identical mirrors
    again = tmp_path / "mirrors_again"
    mirrors.write_locale_mirrors(tables, again)
    for stem in tables:
        for name in (f"{stem}.csv", f"{stem}_de.csv"):
            assert (out / name).read_bytes() == (again / name).read_bytes(), name

    # a locale parser reproduces the source values within 5e-9 in BOTH locales
    assert mirrors.ROUND_TRIP_TOLERANCE == 5e-9
    for stem, table in tables.items():
        mirrors.verify_mirror_round_trip(table, out / f"{stem}.csv", locale="en-US")
        mirrors.verify_mirror_round_trip(
            table, out / f"{stem}_de.csv", locale="de-DE"
        )

    # projections only: the exporter owns no financial derivation at all
    source = inspect.getsource(mirrors)
    assert not any(
        token in source
        for token in ("metric_block", "ssr_inference", "pct_change", "cummax")
    )

    # every cataloged canonical table requires exactly the matching basenames
    required = mirrors.catalog_required_mirrors()
    for canonical_name, mirror_names in required.items():
        stem = canonical_name.rsplit(".", 1)[0]
        assert tuple(mirror_names) == (f"{stem}.csv", f"{stem}_de.csv")
    produced = [name for names in required.values() for name in names]
    mirrors.require_catalog_mirror_coverage(produced)
    with pytest.raises(ValueError, match="monthly_returns_ext2026_de.csv"):
        mirrors.require_catalog_mirror_coverage(
            [name for name in produced if name != "monthly_returns_ext2026_de.csv"]
        )


def test_locale_mirrors_reject_tampered_wrong_locale_or_non_flat_outputs(
    canonical_case, tmp_path
):
    bts = _reports_producer()
    mirrors = _mirror_exporter()
    case = canonical_case
    factor = _load_factor(bts, case)
    table = bts.build_factor_report_tables(factor).tables[
        "portfolio_metrics_reader_ext2026"
    ]
    out = tmp_path / "mirrors"
    mirrors.write_locale_mirrors({"portfolio_metrics_reader_ext2026": table}, out)
    us_path = out / "portfolio_metrics_reader_ext2026.csv"
    de_path = out / "portfolio_metrics_reader_ext2026_de.csv"

    # (a) a German-claimed mirror actually rendered in the US locale
    de_path.write_bytes(mirrors.render_locale_csv(table, locale="en-US"))
    with pytest.raises(ValueError, match="column|reproduce"):
        mirrors.verify_mirror_round_trip(table, de_path, locale="de-DE")

    # (b) a hand-edited numeric cell beyond the 5e-9 contract
    pristine = us_path.read_text()
    wanted = f"{float(table['total_return'].iloc[0]):.8f}"
    assert wanted in pristine
    tampered = pristine.replace(
        wanted, f"{float(table['total_return'].iloc[0]) + 1e-4:.8f}", 1
    )
    assert tampered != pristine
    us_path.write_text(tampered)
    with pytest.raises(ValueError, match="reproduce"):
        mirrors.verify_mirror_round_trip(table, us_path, locale="en-US")
    us_path.write_text(pristine)
    mirrors.verify_mirror_round_trip(table, us_path, locale="en-US")  # restored

    # (c) a mirror of a superseded table shape (dropped column) is stale
    with pytest.raises(ValueError, match="column"):
        mirrors.verify_mirror_round_trip(
            table.drop(columns=["sharpe"]), us_path, locale="en-US"
        )

    # (d) non-flat, colliding, or index-bearing canonical stems never write
    with pytest.raises(ValueError, match="basename"):
        mirrors.write_locale_mirrors({"../evil": table}, tmp_path / "bad")
    with pytest.raises(ValueError, match="_de"):
        mirrors.write_locale_mirrors(
            {"portfolio_metrics_reader_ext2026_de": table}, tmp_path / "bad"
        )
    with pytest.raises(ValueError, match="flat"):
        mirrors.write_locale_mirrors(
            {"indexed": table.set_index("portfolio_id")}, tmp_path / "bad"
        )
    assert not (tmp_path / "bad").exists() or not any((tmp_path / "bad").iterdir())


def test_ac_7_1(canonical_case, tmp_path):
    # AC 7.1: when a shared financial calculation changes, every directly
    # affected user-visible output regenerates from the corrected producer —
    # a mirror generated from the superseded table is detected as stale and
    # only re-export from the regenerated canonical table brings it current.
    from macro_framework.reporting import report_table

    bts = _reports_producer()
    mirrors = _mirror_exporter()
    pub = _publisher()
    case = canonical_case
    market = _load_market(bts, case)
    _, rung_late = _static_specs(bts, case)

    row_before, _ = bts.build_static_bh_rows(
        market, rung_late, ssr_settings={"n_boot": 16, "alpha": 0.05}
    )
    row_after, _ = bts.build_static_bh_rows(
        market, rung_late, ssr_settings={"n_boot": 16, "alpha": 0.10}
    )
    # the changed shared inference setting is visible in the regenerated row
    assert row_before["ssr_alpha"] == 0.05
    assert row_after["ssr_alpha"] == 0.10

    table_before = report_table([row_before])
    table_after = report_table([row_after])
    out_before = tmp_path / "v_before"
    mirrors.write_locale_mirrors(
        {"tear_sheet_static_bh_windows": table_before}, out_before
    )

    # the superseded mirror cannot stand in for the regenerated table
    with pytest.raises(ValueError, match="reproduce"):
        mirrors.verify_mirror_round_trip(
            table_after,
            out_before / "tear_sheet_static_bh_windows.csv",
            locale="en-US",
        )

    # regenerating from the corrected producer output brings every mirror current
    out_after = tmp_path / "v_after"
    mirrors.write_locale_mirrors(
        {"tear_sheet_static_bh_windows": table_after}, out_after
    )
    mirrors.verify_mirror_round_trip(
        table_after, out_after / "tear_sheet_static_bh_windows.csv", locale="en-US"
    )
    mirrors.verify_mirror_round_trip(
        table_after,
        out_after / "tear_sheet_static_bh_windows_de.csv",
        locale="de-DE",
    )

    # the report producer and the locale exporter are the sole owners of
    # canonical tables and their mirrors in the frozen publication contract
    for asset in pub.DATA_V4_CATALOG.assets:
        if (
            asset.asset_class == "canonical_payload"
            and asset.producer_manifest_role == "canonical_reports"
        ):
            assert asset.producer == bts.REPORT_TABLE_OWNER
        if asset.projection in ("csv_us", "csv_de"):
            assert asset.producer == mirrors.MIRROR_PRODUCER


def test_ac_8_6(canonical_case):
    # AC 8.6: report parity validation detects mixed measurement windows,
    # mixed portfolio definitions, undisclosed annualization bases, and stale
    # generated values at the canonical report boundary.
    import copy
    import dataclasses

    import pandas as pd

    from macro_framework.reporting import report_table, validate_report_row

    bts = _reports_producer()
    case = canonical_case
    factor = _load_factor(bts, case)
    reader = validate_report_row(
        next(
            row
            for row in factor.metric_records["records"]
            if (row["portfolio_id"], row["schema"])
            == ("factor_pit_ext2026", "portfolio_metrics.reader.v2")
        )
    )

    # mixed measurement windows for one portfolio/window identity
    shifted = dict(reader)
    shifted["start"] = pd.Timestamp(reader["start"]) - pd.Timedelta(days=7)
    shifted["n_obs"] = int(reader["n_obs"]) + 5
    with pytest.raises(ValueError, match="mixed windows"):
        report_table([reader, shifted])

    def doctored(mutate):
        metric_records = copy.deepcopy(dict(factor.metric_records))
        mutate(metric_records)
        return dataclasses.replace(factor, metric_records=metric_records)

    # mixed portfolio definitions: the differential family cannot silently
    # swap its declared comparison/reference portfolios
    def flipped_differential(metric_records):
        stream = metric_records["source_streams"]["factor_nonpit_minus_pit_ext2026"]
        stream["comparison"], stream["reference"] = (
            stream["reference"],
            stream["comparison"],
        )

    with pytest.raises(ValueError, match="differential source stream"):
        bts.build_auxiliary_report_tables(doctored(flipped_differential))

    # undisclosed annualization bases: reader rows must disclose 252 and can
    # never smuggle the 365 legacy fields
    with pytest.raises(ValueError, match="252"):
        validate_report_row(dict(reader) | {"periods_per_year": 365})
    with pytest.raises(ValueError, match="not part of"):
        validate_report_row(dict(reader) | {"sharpe_cal": 1.31})

    # stale generated values: a published reader row its own validated stream
    # no longer recompounds to
    def stale_reader(metric_records):
        row = next(
            r
            for r in metric_records["records"]
            if (r["portfolio_id"], r["schema"])
            == ("factor_pit_ext2026", "portfolio_metrics.reader.v2")
        )
        row["total_return"] = float(row["total_return"]) + 0.02

    with pytest.raises(ValueError, match="stale generated values"):
        bts.build_auxiliary_report_tables(doctored(stale_reader))
