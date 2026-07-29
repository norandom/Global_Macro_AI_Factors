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
