"""Artifact-boundary checks: published rows disclose their conventions (R7).

``macro_framework.reporting.validate_report_row`` is the producer-side gate;
these tests pin the artifact-facing consequences the coverage matrix assigns to
it: a deliberate legacy convention is identified on the emitted row itself
(7.3), and a raw-on-raw regression can never surface under a CAPM/Jensen
label (defect 7).
"""

from __future__ import annotations

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

