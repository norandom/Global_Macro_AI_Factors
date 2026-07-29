"""Frozen local publication contract for the immutable ``data-v4`` release.

This module defines the flat public asset catalog only.  Later publication
steps stage and validate bytes against this contract; they must not discover
assets by walking directories or by consulting a mutable current-release
pointer.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal, Mapping

RELEASE_TAG = "data-v4"
CATALOG_SCHEMA = "publication_asset_catalog.v1"
PUBLICATION_MANIFEST = "publication_manifest.json"
CHECKSUM_FILE = "SHA256SUMS"
COMPLETION_AUTHORITY = PUBLICATION_MANIFEST
FORBIDDEN_RELEASE_FILES = ("COMPLETED",)

AssetClass = Literal[
    "canonical_payload",
    "projection",
    "compatibility_alias",
    "figure",
    "formatted_report",
]
Locale = Literal["und", "en", "en-US", "de-DE"]


@dataclass(frozen=True, slots=True)
class ProducerManifestSpec:
    """One required immutable producer-manifest boundary."""

    role: str
    schema_id: str


@dataclass(frozen=True, slots=True)
class CatalogAsset:
    """One exact public payload and its owning producer lineage."""

    public_basename: str
    source_artifact: str
    asset_class: AssetClass
    media_type: str
    schema_id: str
    locale: Locale
    producer: str
    producer_manifest_role: str
    lineage: tuple[str, ...]
    projection: str | None = None
    projection_of: str | None = None
    allowed_projections: tuple[str, ...] = ()
    required_projections: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CatalogAssetInput:
    """Manifest-owned source selected for one public catalog entry."""

    public_basename: str
    source_artifact: str
    producer_manifest_role: str


@dataclass(frozen=True, slots=True)
class PublicationAssetCatalog:
    """Complete deterministic contract for the local data-v4 upload set."""

    release_tag: str
    schema_id: str
    assets: tuple[CatalogAsset, ...]
    producer_manifests: tuple[ProducerManifestSpec, ...]
    compatibility_paths: tuple[tuple[str, str], ...]
    publication_manifest: str = PUBLICATION_MANIFEST
    checksum_file: str = CHECKSUM_FILE
    completion_authority: str = COMPLETION_AUTHORITY
    forbidden_release_files: tuple[str, ...] = FORBIDDEN_RELEASE_FILES

    @property
    def payload_basenames(self) -> tuple[str, ...]:
        return tuple(asset.public_basename for asset in self.assets)

    @property
    def producer_manifest_map(self) -> dict[str, str]:
        return {item.role: item.schema_id for item in self.producer_manifests}

    @property
    def manifest_inventory_basenames(self) -> tuple[str, ...]:
        """The final manifest inventories payloads, not release control files."""

        return self.payload_basenames

    @property
    def checksum_basenames(self) -> tuple[str, ...]:
        """SHA256SUMS covers payloads plus the final manifest, never itself."""

        return tuple(sorted(self.payload_basenames + (self.publication_manifest,)))

    @property
    def final_inventory_basenames(self) -> tuple[str, ...]:
        return tuple(sorted(self.checksum_basenames + (self.checksum_file,)))


_PRODUCER_MANIFESTS = (
    ProducerManifestSpec("canonical_reports", "canonical_reports.v1"),
    ProducerManifestSpec("factor_run", "factor_run.v1"),
    ProducerManifestSpec("market_snapshot", "market_snapshot.v1"),
    ProducerManifestSpec("presentation_outputs", "presentation_outputs.v1"),
    ProducerManifestSpec("sjm_run", "sjm_run.v3"),
)
_LINEAGE_ORDER = (
    "market_snapshot",
    "factor_run",
    "sjm_run",
    "canonical_reports",
    "presentation_outputs",
)
_MANIFEST_ORDER_INDEX = {role: index for index, role in enumerate(_LINEAGE_ORDER)}

_PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
_JSON_MEDIA_TYPE = "application/json"
_CSV_MEDIA_TYPE = "text/csv; charset=utf-8"
_PNG_MEDIA_TYPE = "image/png"
_MARKDOWN_MEDIA_TYPE = "text/markdown; charset=utf-8"
_TEX_MEDIA_TYPE = "application/x-tex; charset=utf-8"

_FACTOR_TABLES = (
    ("baseline_equity_ext2026", "portfolio_series.equity.v1"),
    ("baseline_targets_ext2026", "portfolio_series.targets.v1"),
    ("factor_contrast_ext2026", "factor.contrast.v1"),
    ("factor_equity_ext2026", "factor.equity.v1"),
    ("factor_evidence_ext2026", "factor.evidence.v1"),
    ("factor_loadings_ext2026", "factor.loadings.v1"),
    ("factor_nonpit_diagnostic_equity_ext2026", "factor.equity.v1"),
    ("factor_nonpit_diagnostic_loadings_ext2026", "factor.loadings.v1"),
    ("factor_nonpit_diagnostic_scores_ext2026", "factor.scores.v1"),
    ("factor_nonpit_diagnostic_targets_ext2026", "factor.targets.v1"),
    ("factor_scores_ext2026", "factor.scores.v1"),
    ("factor_targets_ext2026", "factor.targets.v1"),
    ("track_b_equity_ext2026", "portfolio_series.equity.v1"),
    ("track_b_targets_ext2026", "portfolio_series.targets.v1"),
)
_SJM_TABLES = (
    (
        "sjm_crowding_v3_total_return_bil_control_returns_ext2026",
        "sjm.control_returns.v3",
    ),
    (
        "sjm_crowding_v3_total_return_bil_daily_returns_ext2026",
        "sjm.daily_returns.v3",
    ),
    ("sjm_crowding_v3_total_return_bil_equity_ext2026", "sjm.equity.v3"),
    ("sjm_crowding_v3_total_return_bil_targets_ext2026", "sjm.targets.v3"),
)
_REPORT_TABLES = (
    ("attribution_raw_market_model_ext2026", "attribution.raw_market_model.v1"),
    ("crisis_metrics_ext2026", "crisis_metrics.boundary_anchored.v1"),
    ("markowitz_10y_frontier", "markowitz.frontier.v1"),
    ("markowitz_10y_moments", "markowitz.moments.v1"),
    ("markowitz_max_frontier", "markowitz.frontier.v1"),
    ("markowitz_max_moments", "markowitz.moments.v1"),
    ("monthly_returns_ext2026", "monthly_returns.reader.v1"),
    ("portfolio_metrics_differential_ext2026", "portfolio_metrics.differential.v2"),
    ("portfolio_metrics_reader_ext2026", "portfolio_metrics.reader.v2"),
    ("portfolio_metrics_vectorbt365_ext2026", "portfolio_metrics.vectorbt365.v1"),
    ("risk_decomposition_ext2026", "risk_decomposition.v1"),
    ("tear_sheet_ai_variants_ext2026", "tear_sheet.ai_variants.v1"),
    ("tear_sheet_sjm_crowding_ext2026", "tear_sheet.sjm.v3"),
    ("tear_sheet_static_bh_window_dashboard", "tear_sheet.window_dashboard.v1"),
    ("tear_sheet_static_bh_windows", "tear_sheet.static_windows.v1"),
    ("tear_sheet_trio_10y", "tear_sheet.trio_10y.v1"),
    ("tear_sheet_trio_ext2026", "tear_sheet.trio.v4"),
    ("tear_sheet_trio_max", "tear_sheet.trio_max.v1"),
)


def _projection_lineage(lineage: tuple[str, ...]) -> tuple[str, ...]:
    if lineage[-1] == "canonical_reports":
        return lineage
    return lineage + ("canonical_reports",)


def _asset(
    public_basename: str,
    *,
    source_directory: str,
    asset_class: AssetClass,
    media_type: str,
    schema_id: str,
    locale: Locale,
    producer: str,
    producer_manifest_role: str,
    lineage: tuple[str, ...],
    projection: str | None = None,
    projection_of: str | None = None,
) -> CatalogAsset:
    return CatalogAsset(
        public_basename=public_basename,
        source_artifact=f"{source_directory}/{public_basename}",
        asset_class=asset_class,
        media_type=media_type,
        schema_id=schema_id,
        locale=locale,
        producer=producer,
        producer_manifest_role=producer_manifest_role,
        lineage=lineage,
        projection=projection,
        projection_of=projection_of,
    )


def _add_tabular_family(
    assets: list[CatalogAsset],
    required: dict[str, tuple[str, ...]],
    *,
    stem: str,
    canonical_suffix: Literal[".parquet", ".json"],
    schema_id: str,
    producer: str,
    producer_manifest_role: str,
    lineage: tuple[str, ...],
) -> None:
    canonical_name = f"{stem}{canonical_suffix}"
    canonical_media = (
        _PARQUET_MEDIA_TYPE if canonical_suffix == ".parquet" else _JSON_MEDIA_TYPE
    )
    assets.append(
        _asset(
            canonical_name,
            source_directory="artifacts" if producer_manifest_role != "canonical_reports" else "tables",
            asset_class="canonical_payload",
            media_type=canonical_media,
            schema_id=schema_id,
            locale="und",
            producer=producer,
            producer_manifest_role=producer_manifest_role,
            lineage=lineage,
        )
    )
    us_name = f"{stem}.csv"
    de_name = f"{stem}_de.csv"
    mirror_lineage = _projection_lineage(lineage)
    for name, projection, locale in (
        (us_name, "csv_us", "en-US"),
        (de_name, "csv_de", "de-DE"),
    ):
        assets.append(
            _asset(
                name,
                source_directory="mirrors",
                asset_class="projection",
                media_type=_CSV_MEDIA_TYPE,
                schema_id=schema_id,
                locale=locale,
                producer="scripts/export_csv_mirrors.py",
                producer_manifest_role="canonical_reports",
                lineage=mirror_lineage,
                projection=projection,
                projection_of=canonical_name,
            )
        )
    required[canonical_name] = (us_name, de_name)


def _add_compatibility_alias(
    assets: list[CatalogAsset],
    compatibility: list[tuple[str, str]],
    *,
    public_basename: str,
    target: str,
) -> None:
    target_asset = next(
        (asset for asset in assets if asset.public_basename == target),
        None,
    )
    if target_asset is None:
        raise ValueError(f"compatibility alias targets unknown asset {target!r}")
    assets.append(
        dataclasses.replace(
            target_asset,
            public_basename=public_basename,
            asset_class="compatibility_alias",
            projection="compatibility_alias",
            projection_of=target,
            allowed_projections=(),
            required_projections=(),
        )
    )
    compatibility.append((public_basename, target))


def _add_figure(
    assets: list[CatalogAsset],
    *,
    public_basename: str,
    projection_of: str,
    producer: str,
    lineage: tuple[str, ...],
) -> None:
    assets.append(
        _asset(
            public_basename,
            source_directory="figures",
            asset_class="figure",
            media_type=_PNG_MEDIA_TYPE,
            schema_id="figure.png.v1",
            locale="und",
            producer=producer,
            producer_manifest_role="presentation_outputs",
            lineage=lineage + ("presentation_outputs",),
            projection="figure_png",
            projection_of=projection_of,
        )
    )


def _add_formatted_report(
    assets: list[CatalogAsset],
    *,
    public_basename: str,
    projection: str,
    media_type: str,
    locale: Locale,
) -> None:
    assets.append(
        _asset(
            public_basename,
            source_directory="formatted",
            asset_class="formatted_report",
            media_type=media_type,
            schema_id="tear_sheet.formatted.v1",
            locale=locale,
            producer="notebooks/15_4_tear_sheet_paper_and_thesis.ipynb",
            producer_manifest_role="presentation_outputs",
            lineage=(
                "market_snapshot",
                "factor_run",
                "sjm_run",
                "canonical_reports",
                "presentation_outputs",
            ),
            projection=projection,
            projection_of="tear_sheet_trio_ext2026.parquet",
        )
    )


def build_data_v4_catalog() -> PublicationAssetCatalog:
    """Build the exact data-v4 contract without filesystem discovery."""

    assets: list[CatalogAsset] = []
    required: dict[str, tuple[str, ...]] = {}
    compatibility: list[tuple[str, str]] = []

    factor_lineage = ("market_snapshot", "factor_run")
    for stem, schema_id in _FACTOR_TABLES:
        _add_tabular_family(
            assets,
            required,
            stem=stem,
            canonical_suffix=".parquet",
            schema_id=schema_id,
            producer="scripts/extend_stream_2026.py",
            producer_manifest_role="factor_run",
            lineage=factor_lineage,
        )
    for stem, schema_id in (
        ("factor_contrast_split_ext2026", "factor.contrast_split.v1"),
        ("factor_decision_log_ext2026", "factor.decision_log.v1"),
        (
            "factor_nonpit_diagnostic_decision_log_ext2026",
            "factor.decision_log.v1",
        ),
    ):
        _add_tabular_family(
            assets,
            required,
            stem=stem,
            canonical_suffix=".json",
            schema_id=schema_id,
            producer="scripts/extend_stream_2026.py",
            producer_manifest_role="factor_run",
            lineage=factor_lineage,
        )
    assets.append(
        _asset(
            "factor_replay_audit_ext2026.json",
            source_directory="artifacts",
            asset_class="canonical_payload",
            media_type=_JSON_MEDIA_TYPE,
            schema_id="factor.replay_audit.v1",
            locale="und",
            producer="scripts/extend_stream_2026.py",
            producer_manifest_role="factor_run",
            lineage=factor_lineage,
        )
    )

    sjm_lineage = ("market_snapshot", "factor_run", "sjm_run")
    for stem, schema_id in _SJM_TABLES:
        _add_tabular_family(
            assets,
            required,
            stem=stem,
            canonical_suffix=".parquet",
            schema_id=schema_id,
            producer="scripts/build_sjm_crowding.py",
            producer_manifest_role="sjm_run",
            lineage=sjm_lineage,
        )
    _add_tabular_family(
        assets,
        required,
        stem="sjm_crowding_v3_total_return_bil_ledger",
        canonical_suffix=".json",
        schema_id="sjm.ledger.v3",
        producer="scripts/build_sjm_crowding.py",
        producer_manifest_role="sjm_run",
        lineage=sjm_lineage,
    )
    assets.append(
        _asset(
            "sjm_crowding_v3_total_return_bil_protocol.json",
            source_directory="artifacts",
            asset_class="canonical_payload",
            media_type=_JSON_MEDIA_TYPE,
            schema_id="sjm_selection_protocol.v2",
            locale="und",
            producer="scripts/build_sjm_crowding.py",
            producer_manifest_role="sjm_run",
            lineage=sjm_lineage,
        )
    )

    factor_report_stems = {
        "attribution_raw_market_model_ext2026",
        "crisis_metrics_ext2026",
        "monthly_returns_ext2026",
        "portfolio_metrics_differential_ext2026",
        "portfolio_metrics_reader_ext2026",
        "portfolio_metrics_vectorbt365_ext2026",
        "risk_decomposition_ext2026",
        "tear_sheet_ai_variants_ext2026",
    }
    markowitz_stems = {
        "markowitz_10y_frontier",
        "markowitz_10y_moments",
        "markowitz_max_frontier",
        "markowitz_max_moments",
    }
    for stem, schema_id in _REPORT_TABLES:
        if stem in markowitz_stems:
            lineage = ("market_snapshot", "canonical_reports")
        elif stem in factor_report_stems:
            lineage = ("market_snapshot", "factor_run", "canonical_reports")
        else:
            lineage = (
                "market_snapshot",
                "factor_run",
                "sjm_run",
                "canonical_reports",
            )
        _add_tabular_family(
            assets,
            required,
            stem=stem,
            canonical_suffix=".parquet",
            schema_id=schema_id,
            producer="scripts/build_tear_sheet.py",
            producer_manifest_role="canonical_reports",
            lineage=lineage,
        )

    alias_specs = (
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
        (
            "sjm_crowding_derisk_equity_ext2026.parquet",
            "sjm_crowding_v3_total_return_bil_equity_ext2026.parquet",
        ),
        (
            "sjm_crowding_derisk_equity_ext2026.csv",
            "sjm_crowding_v3_total_return_bil_equity_ext2026.csv",
        ),
        (
            "sjm_crowding_derisk_equity_ext2026_de.csv",
            "sjm_crowding_v3_total_return_bil_equity_ext2026_de.csv",
        ),
        ("tear_sheet_ext2026.csv", "portfolio_metrics_reader_ext2026.csv"),
        ("tear_sheet_ext2026_de.csv", "portfolio_metrics_reader_ext2026_de.csv"),
    )
    for public_basename, target in alias_specs:
        _add_compatibility_alias(
            assets,
            compatibility,
            public_basename=public_basename,
            target=target,
        )

    figures = (
        (
            "nb15_3_long_window_equity.png",
            "tear_sheet_static_bh_windows.parquet",
            "notebooks/15_3_extended_timeframe_static_bh.ipynb",
            ("market_snapshot", "factor_run", "sjm_run", "canonical_reports"),
        ),
        (
            "nb15_3_sharpe_vs_window.png",
            "tear_sheet_static_bh_windows.parquet",
            "notebooks/15_3_extended_timeframe_static_bh.ipynb",
            ("market_snapshot", "factor_run", "sjm_run", "canonical_reports"),
        ),
        (
            "nb18_2_ratio_ladder.png",
            "tear_sheet_static_bh_window_dashboard.parquet",
            "notebooks/18_2_window_dashboard.ipynb",
            ("market_snapshot", "factor_run", "sjm_run", "canonical_reports"),
        ),
        (
            "nb18_2_risk_return_map.png",
            "tear_sheet_static_bh_window_dashboard.parquet",
            "notebooks/18_2_window_dashboard.ipynb",
            ("market_snapshot", "factor_run", "sjm_run", "canonical_reports"),
        ),
        (
            "nb18_3_markowitz_plane_10y.png",
            "markowitz_10y_frontier.parquet",
            "notebooks/18_3_trio_10y.ipynb",
            ("market_snapshot", "canonical_reports"),
        ),
        (
            "nb18_3_panels_10y.png",
            "tear_sheet_trio_10y.parquet",
            "notebooks/18_3_trio_10y.ipynb",
            ("market_snapshot", "factor_run", "sjm_run", "canonical_reports"),
        ),
        (
            "nb18_4_markowitz_plane_max.png",
            "markowitz_max_frontier.parquet",
            "notebooks/18_4_trio_max_timeframe.ipynb",
            ("market_snapshot", "canonical_reports"),
        ),
        (
            "nb18_4_panels_max.png",
            "tear_sheet_trio_max.parquet",
            "notebooks/18_4_trio_max_timeframe.ipynb",
            ("market_snapshot", "factor_run", "sjm_run", "canonical_reports"),
        ),
        (
            "nb18_metric_profile.png",
            "tear_sheet_trio_ext2026.parquet",
            "notebooks/18_final_trio_dashboard.ipynb",
            ("market_snapshot", "factor_run", "sjm_run", "canonical_reports"),
        ),
        (
            "nb18_ratio_ladder.png",
            "tear_sheet_trio_ext2026.parquet",
            "notebooks/18_final_trio_dashboard.ipynb",
            ("market_snapshot", "factor_run", "sjm_run", "canonical_reports"),
        ),
        (
            "nb18_risk_return_maps.png",
            "tear_sheet_trio_ext2026.parquet",
            "notebooks/18_final_trio_dashboard.ipynb",
            ("market_snapshot", "factor_run", "sjm_run", "canonical_reports"),
        ),
    )
    for public_basename, projection_of, producer, lineage in figures:
        _add_figure(
            assets,
            public_basename=public_basename,
            projection_of=projection_of,
            producer=producer,
            lineage=lineage,
        )

    for name, projection, media_type, locale in (
        ("tear_sheet_paper.csv", "report_csv_us", _CSV_MEDIA_TYPE, "en-US"),
        ("tear_sheet_paper_de.csv", "report_csv_de", _CSV_MEDIA_TYPE, "de-DE"),
        ("tear_sheet_paper.md", "report_markdown", _MARKDOWN_MEDIA_TYPE, "en"),
        ("tear_sheet_paper.tex", "report_tex", _TEX_MEDIA_TYPE, "en"),
        ("tear_sheet_thesis.csv", "report_csv_us", _CSV_MEDIA_TYPE, "en-US"),
        ("tear_sheet_thesis_de.csv", "report_csv_de", _CSV_MEDIA_TYPE, "de-DE"),
        ("tear_sheet_thesis.md", "report_markdown", _MARKDOWN_MEDIA_TYPE, "en"),
        ("tear_sheet_thesis.tex", "report_tex", _TEX_MEDIA_TYPE, "en"),
    ):
        _add_formatted_report(
            assets,
            public_basename=name,
            projection=projection,
            media_type=media_type,
            locale=locale,
        )

    stable_compatibility_names = (
        "baseline_equity_ext2026.parquet",
        "baseline_targets_ext2026.parquet",
        "factor_decision_log_ext2026.json",
        "factor_equity_ext2026.parquet",
        "factor_nonpit_diagnostic_decision_log_ext2026.json",
        "factor_nonpit_diagnostic_equity_ext2026.parquet",
        "factor_nonpit_diagnostic_targets_ext2026.parquet",
        "factor_targets_ext2026.parquet",
        "risk_decomposition_ext2026.csv",
        "risk_decomposition_ext2026_de.csv",
        "tear_sheet_static_bh_window_dashboard.csv",
        "tear_sheet_static_bh_window_dashboard_de.csv",
        "tear_sheet_static_bh_windows.csv",
        "tear_sheet_static_bh_windows_de.csv",
        "tear_sheet_trio_10y.csv",
        "tear_sheet_trio_10y_de.csv",
        "tear_sheet_trio_ext2026.csv",
        "tear_sheet_trio_ext2026_de.csv",
        "tear_sheet_trio_max.csv",
        "tear_sheet_trio_max_de.csv",
        "track_b_equity_ext2026.parquet",
        "track_b_targets_ext2026.parquet",
    )
    compatibility.extend((name, name) for name in stable_compatibility_names)

    projections_by_target: dict[str, list[str]] = {}
    for asset in assets:
        if asset.projection_of is not None:
            projections_by_target.setdefault(asset.projection_of, []).append(
                asset.public_basename
            )
    assets = [
        dataclasses.replace(
            asset,
            allowed_projections=tuple(
                sorted(projections_by_target.get(asset.public_basename, ()))
            ),
            required_projections=required.get(asset.public_basename, ()),
        )
        for asset in assets
    ]

    catalog = PublicationAssetCatalog(
        release_tag=RELEASE_TAG,
        schema_id=CATALOG_SCHEMA,
        assets=tuple(sorted(assets, key=lambda item: item.public_basename)),
        producer_manifests=_PRODUCER_MANIFESTS,
        compatibility_paths=tuple(sorted(compatibility)),
    )
    return _validate_asset_catalog_structure(catalog)


def _validate_flat_basename(name: str, *, label: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{label} must be a non-blank string")
    path = PurePosixPath(name)
    if path.name != name or path.is_absolute() or "\\" in name:
        raise ValueError(f"{label} must be a flat public basename: {name!r}")


def _uses_mutable_pointer(path: str) -> bool:
    parts = tuple(part.casefold() for part in PurePosixPath(path).parts)
    mutable_names = {"current", "current.json", "latest", "latest.json"}
    return any(part in mutable_names for part in parts)


def _validate_asset_catalog_structure(
    catalog: PublicationAssetCatalog,
) -> PublicationAssetCatalog:
    """Validate catalog structure without reading producer directories."""

    if catalog.release_tag != RELEASE_TAG:
        raise ValueError(f"release tag must be {RELEASE_TAG!r}")
    if catalog.schema_id != CATALOG_SCHEMA:
        raise ValueError(f"catalog schema must be {CATALOG_SCHEMA!r}")
    if catalog.publication_manifest != PUBLICATION_MANIFEST:
        raise ValueError("publication manifest basename is not canonical")
    if catalog.checksum_file != CHECKSUM_FILE:
        raise ValueError("checksum basename is not canonical")
    if catalog.completion_authority != PUBLICATION_MANIFEST:
        raise ValueError("the final publication manifest is the only completion authority")
    if catalog.forbidden_release_files != FORBIDDEN_RELEASE_FILES:
        raise ValueError("release-directory COMPLETED marker must remain forbidden")

    role_map = catalog.producer_manifest_map
    if tuple(sorted(role_map.items())) != tuple(
        sorted((item.role, item.schema_id) for item in _PRODUCER_MANIFESTS)
    ):
        raise ValueError("producer manifest contract differs from the frozen data-v4 set")

    seen_names: dict[str, str] = {}
    by_name: dict[str, CatalogAsset] = {}
    reserved = {PUBLICATION_MANIFEST.casefold(), CHECKSUM_FILE.casefold(), "completed"}
    allowed_locales = {"und", "en", "en-US", "de-DE"}
    for asset in catalog.assets:
        _validate_flat_basename(asset.public_basename, label="public basename")
        folded = asset.public_basename.casefold()
        if folded in seen_names:
            raise ValueError(
                "duplicate public basename "
                f"{asset.public_basename!r} collides with {seen_names[folded]!r}"
            )
        if folded in reserved:
            raise ValueError(f"payload uses reserved release filename {asset.public_basename!r}")
        seen_names[folded] = asset.public_basename
        by_name[asset.public_basename] = asset

        if not asset.source_artifact or PurePosixPath(asset.source_artifact).is_absolute():
            raise ValueError(f"source artifact must be a relative manifest path: {asset.source_artifact!r}")
        if ".." in PurePosixPath(asset.source_artifact).parts or "\\" in asset.source_artifact:
            raise ValueError(f"source artifact escapes its producer manifest: {asset.source_artifact!r}")
        if _uses_mutable_pointer(asset.source_artifact):
            raise ValueError(
                f"source artifact uses a mutable current-release pointer: {asset.source_artifact!r}"
            )
        if not asset.media_type or not asset.schema_id or not asset.producer:
            raise ValueError(f"asset {asset.public_basename!r} has incomplete metadata")
        if asset.locale not in allowed_locales:
            raise ValueError(f"asset {asset.public_basename!r} has unsupported locale")
        if asset.producer_manifest_role not in role_map:
            raise ValueError(
                "unexpected producer manifest "
                f"{asset.producer_manifest_role!r} for {asset.public_basename!r}"
            )
        if not asset.lineage or asset.lineage[-1] != asset.producer_manifest_role:
            raise ValueError(f"asset {asset.public_basename!r} has incomplete producer lineage")
        if len(asset.lineage) != len(set(asset.lineage)):
            raise ValueError(f"asset {asset.public_basename!r} repeats a lineage manifest")
        if any(role not in role_map for role in asset.lineage):
            raise ValueError(f"asset {asset.public_basename!r} has unexpected lineage manifest")
        order = [_MANIFEST_ORDER_INDEX[role] for role in asset.lineage]
        if order != sorted(order):
            raise ValueError(f"asset {asset.public_basename!r} has out-of-order lineage")

    if tuple(by_name) != tuple(sorted(by_name)):
        raise ValueError("catalog assets must be sorted by public basename")

    projection_names_by_target: dict[str, list[str]] = {}
    for asset in catalog.assets:
        if asset.projection is None:
            if asset.projection_of is not None:
                raise ValueError(f"asset {asset.public_basename!r} has a target but no projection")
        else:
            if asset.projection_of not in by_name:
                raise ValueError(
                    f"projection {asset.public_basename!r} targets missing asset {asset.projection_of!r}"
                )
            projection_names_by_target.setdefault(asset.projection_of, []).append(
                asset.public_basename
            )
            target = by_name[asset.projection_of]
            if asset.public_basename not in target.allowed_projections:
                raise ValueError(
                    f"undeclared projection {asset.public_basename!r} for {asset.projection_of!r}"
                )
            if asset.projection == "compatibility_alias":
                if (
                    asset.source_artifact != target.source_artifact
                    or asset.producer != target.producer
                    or asset.producer_manifest_role != target.producer_manifest_role
                    or asset.lineage != target.lineage
                    or asset.schema_id != target.schema_id
                    or asset.media_type != target.media_type
                    or asset.locale != target.locale
                ):
                    raise ValueError(
                        f"compatibility alias {asset.public_basename!r} changes target ownership or semantics"
                    )

        if not set(asset.required_projections).issubset(asset.allowed_projections):
            raise ValueError(f"asset {asset.public_basename!r} has impossible required projections")

    for asset in catalog.assets:
        actual = tuple(projection_names_by_target.get(asset.public_basename, ()))
        if actual != asset.allowed_projections:
            raise ValueError(f"asset {asset.public_basename!r} projection declaration is not exact")
        missing = set(asset.required_projections) - set(actual)
        if missing:
            raise ValueError(
                f"asset {asset.public_basename!r} is missing required projection(s) {sorted(missing)}"
            )

    compatibility_seen: set[str] = set()
    alias_paths: set[str] = set()
    for public_path, target in catalog.compatibility_paths:
        _validate_flat_basename(public_path, label="compatibility path")
        if _uses_mutable_pointer(public_path):
            raise ValueError("compatibility path uses a mutable current-release pointer")
        folded = public_path.casefold()
        if folded in compatibility_seen:
            raise ValueError(f"duplicate compatibility path {public_path!r}")
        compatibility_seen.add(folded)
        if public_path not in by_name or target not in by_name:
            raise ValueError(f"compatibility path {public_path!r} is not catalog-owned")
        if public_path != target:
            alias = by_name[public_path]
            if alias.projection != "compatibility_alias" or alias.projection_of != target:
                raise ValueError(f"undeclared compatibility alias {public_path!r}")
            alias_paths.add(public_path)
    declared_aliases = {
        asset.public_basename
        for asset in catalog.assets
        if asset.projection == "compatibility_alias"
    }
    if declared_aliases != alias_paths:
        raise ValueError("compatibility aliases and compatibility-path map differ")

    if "COMPLETED" in catalog.final_inventory_basenames:
        raise ValueError("release-directory COMPLETED marker is forbidden")
    if CHECKSUM_FILE in catalog.checksum_basenames:
        raise ValueError("SHA256SUMS must never hash itself")
    return catalog


def _asset_contract_signature(
    assets: tuple[CatalogAsset, ...],
) -> tuple[tuple[object, ...], ...]:
    """Capture every CatalogAsset field in declared tuple order."""

    return tuple(dataclasses.astuple(asset) for asset in assets)


def _producer_manifest_contract_signature(
    manifests: tuple[ProducerManifestSpec, ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(dataclasses.astuple(manifest) for manifest in manifests)


def _validate_frozen_catalog_contract(
    catalog: PublicationAssetCatalog,
) -> PublicationAssetCatalog:
    producer_signature = _producer_manifest_contract_signature(
        catalog.producer_manifests
    )
    if producer_signature != _FROZEN_PRODUCER_MANIFEST_SIGNATURE:
        raise ValueError(
            "frozen producer-manifest contract differs in declaration, duplicate, "
            "or order"
        )

    asset_signature = _asset_contract_signature(catalog.assets)
    if asset_signature != _FROZEN_ASSET_SIGNATURE:
        if len(asset_signature) != len(_FROZEN_ASSET_SIGNATURE):
            detail = (
                f"expected {len(_FROZEN_ASSET_SIGNATURE)} assets, "
                f"received {len(asset_signature)}"
            )
        else:
            difference = next(
                index
                for index, (actual, expected) in enumerate(
                    zip(asset_signature, _FROZEN_ASSET_SIGNATURE, strict=True)
                )
                if actual != expected
            )
            field_names = tuple(field.name for field in dataclasses.fields(CatalogAsset))
            changed_fields = tuple(
                name
                for name, actual, expected in zip(
                    field_names,
                    asset_signature[difference],
                    _FROZEN_ASSET_SIGNATURE[difference],
                    strict=True,
                )
                if actual != expected
            )
            detail = (
                f"asset index {difference} differs in field(s) {changed_fields!r}"
            )
        raise ValueError(f"frozen asset contract differs: {detail}")

    if catalog.compatibility_paths != _FROZEN_COMPATIBILITY_PATHS:
        raise ValueError(
            "frozen compatibility-path contract differs in declaration, duplicate, "
            "or order"
        )
    return catalog


def validate_asset_catalog(
    catalog: PublicationAssetCatalog,
) -> PublicationAssetCatalog:
    """Require the exact frozen data-v4 contract after structural validation."""

    validated = _validate_asset_catalog_structure(catalog)
    return _validate_frozen_catalog_contract(validated)


def validate_catalog_inputs(
    inputs: tuple[CatalogAssetInput, ...] | list[CatalogAssetInput],
    *,
    catalog: PublicationAssetCatalog | None = None,
) -> tuple[CatalogAsset, ...]:
    """Require one exact manifest-owned source for every catalog payload."""

    active = DATA_V4_CATALOG if catalog is None else validate_asset_catalog(catalog)
    supplied = tuple(inputs)
    by_name: dict[str, CatalogAssetInput] = {}
    seen_folded: set[str] = set()
    for item in supplied:
        folded = item.public_basename.casefold()
        if folded in seen_folded:
            raise ValueError(f"duplicate public basename in catalog inputs: {item.public_basename!r}")
        seen_folded.add(folded)
        by_name[item.public_basename] = item

    expected = set(active.payload_basenames)
    actual = set(by_name)
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"missing required asset(s): {missing}")
    extra = sorted(actual - expected)
    if extra:
        raise ValueError(f"undeclared alias or extra asset(s): {extra}")

    roles = active.producer_manifest_map
    for asset in active.assets:
        item = by_name[asset.public_basename]
        if item.producer_manifest_role not in roles:
            raise ValueError(
                "unexpected producer manifest "
                f"{item.producer_manifest_role!r} for {item.public_basename!r}"
            )
        if item.producer_manifest_role != asset.producer_manifest_role:
            raise ValueError(
                "unexpected producer manifest "
                f"{item.producer_manifest_role!r} for {item.public_basename!r}; "
                f"expected {asset.producer_manifest_role!r}"
            )
        if item.source_artifact != asset.source_artifact:
            raise ValueError(
                f"unexpected source artifact for {item.public_basename!r}: "
                f"{item.source_artifact!r}"
            )
    return active.assets


def validate_producer_manifest_inputs(
    manifests: Mapping[str, str],
    *,
    catalog: PublicationAssetCatalog | None = None,
) -> tuple[tuple[str, str], ...]:
    """Reject missing, extra, or schema-incompatible producer manifests."""

    active = DATA_V4_CATALOG if catalog is None else validate_asset_catalog(catalog)
    expected = active.producer_manifest_map
    actual = dict(manifests)
    unexpected = sorted(set(actual) - set(expected))
    if unexpected:
        raise ValueError(f"unexpected producer manifest(s): {unexpected}")
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise ValueError(f"missing producer manifest(s): {missing}")
    for role, schema_id in expected.items():
        if actual[role] != schema_id:
            raise ValueError(
                f"producer manifest {role!r} has schema {actual[role]!r}; expected {schema_id!r}"
            )
    return tuple(sorted(actual.items()))


def _catalog_dict(catalog: PublicationAssetCatalog) -> dict[str, object]:
    return {
        "schema_id": catalog.schema_id,
        "release_tag": catalog.release_tag,
        "producer_manifests": [dataclasses.asdict(item) for item in catalog.producer_manifests],
        "payloads": [dataclasses.asdict(asset) for asset in catalog.assets],
        "compatibility_paths": [
            {"public_path": path, "target": target}
            for path, target in catalog.compatibility_paths
        ],
        "checksum_rule": {
            "manifest_inventories": "payloads_only",
            "sha256sums_covers": "payloads_plus_final_manifest",
            "sha256sums_excludes": CHECKSUM_FILE,
        },
        "completion_authority": catalog.completion_authority,
        "forbidden_release_files": list(catalog.forbidden_release_files),
    }


def catalog_sha256(catalog: PublicationAssetCatalog) -> str:
    """Hash the canonical, timestamp-free catalog contract."""

    validated = validate_asset_catalog(catalog)
    encoded = json.dumps(
        _catalog_dict(validated),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


DATA_V4_CATALOG = build_data_v4_catalog()
_FROZEN_PRODUCER_MANIFEST_SIGNATURE = _producer_manifest_contract_signature(
    DATA_V4_CATALOG.producer_manifests
)
_FROZEN_ASSET_SIGNATURE = _asset_contract_signature(DATA_V4_CATALOG.assets)
_FROZEN_COMPATIBILITY_PATHS = DATA_V4_CATALOG.compatibility_paths
