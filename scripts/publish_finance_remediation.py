"""Frozen local publication contract for the immutable ``data-v4`` release.

This module defines the flat public asset catalog only.  Later publication
steps stage and validate bytes against this contract; they must not discover
assets by walking directories or by consulting a mutable current-release
pointer.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
import os
import re
import shutil
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Iterator, Literal, Mapping

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


# --------------------------------------------------------------------------- #
# Task 10.8: incomplete candidate staging and direct validation.               #
# Task 10.9: finalization, checksums, read-only verification, promotion.       #
#                                                                              #
# The machinery is producer-manifest-driven: every input arrives by path and   #
# manifest (fixture or real), never by discovering repository artifacts. It    #
# mirrors the snapshot/factor staging conventions (tasks 5.3/5.4, 6.9): a new  #
# empty destination, full validation before any completion state exists, and   #
# diagnosable-but-incomplete failures. Completion here is represented ONLY by  #
# the final publication manifest — never by a release-directory COMPLETED      #
# marker (task 10.1).                                                          #
# --------------------------------------------------------------------------- #

PUBLICATION_MANIFEST_SCHEMA = "publication_manifest.v1"
PRODUCER_MANIFEST_NAME = "manifest.json"
PRODUCER_COMPLETION_MARKER = "COMPLETED"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Mapping[str, object]) -> str:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _normalize_build_time(build_time: str | None) -> str:
    if build_time is None:
        return datetime.now(timezone.utc).isoformat()
    try:
        parsed = datetime.fromisoformat(str(build_time))
    except ValueError as exc:
        raise ValueError(
            f"build_time must be an ISO-8601 timestamp, got {build_time!r}"
        ) from exc
    if parsed.tzinfo is None:
        raise ValueError(f"build_time must be timezone-aware, got {build_time!r}")
    return parsed.astimezone(timezone.utc).isoformat()


def _active_catalog(catalog: PublicationAssetCatalog | None) -> PublicationAssetCatalog:
    return DATA_V4_CATALOG if catalog is None else validate_asset_catalog(catalog)


def _validate_window(entry: Mapping[str, object], *, label: str) -> None:
    declared = [key for key in ("rows", "start", "end") if key in entry]
    if not declared:
        return
    if set(declared) != {"rows", "start", "end"}:
        raise ValueError(
            f"{label}: window declaration must carry rows, start, and end together"
        )
    rows = entry["rows"]
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ValueError(f"{label}: window rows must be a positive integer")
    try:
        start = date.fromisoformat(str(entry["start"]))
        end = date.fromisoformat(str(entry["end"]))
    except ValueError as exc:
        raise ValueError(f"{label}: window start/end must be ISO dates") from exc
    if start > end:
        raise ValueError(
            f"{label}: window start {entry['start']} is after end {entry['end']}"
        )


def _load_completed_producer_manifest(
    role: str, directory: Path
) -> Mapping[str, object]:
    """One completed producer boundary: manifest + marker carrying its sha256."""

    manifest_path = directory / PRODUCER_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"producer manifest {role!r} is absent: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"producer manifest {role!r} must be a JSON object")
    if manifest.get("completed") is not True:
        raise ValueError(
            f"producer manifest {role!r} must declare completed=true; "
            "refusing an incomplete producer run"
        )
    marker = directory / PRODUCER_COMPLETION_MARKER
    if not marker.is_file():
        raise ValueError(
            f"producer {role!r} COMPLETED marker is absent; the run is incomplete"
        )
    match = re.search(r"manifest_sha256=([0-9a-f]{64})", marker.read_text())
    if match is None or match.group(1) != _sha256_file(manifest_path):
        raise ValueError(
            f"producer {role!r} COMPLETED marker does not match manifest bytes "
            "(stale or tampered completion)"
        )
    if not isinstance(manifest.get("assets"), dict):
        raise ValueError(f"producer manifest {role!r} must inventory its assets")
    return manifest


def _load_producer_boundary(
    producers: Mapping[str, Path | str],
    active: PublicationAssetCatalog,
) -> tuple[dict[str, Path], dict[str, Mapping[str, object]]]:
    """Exact completed producer-manifest set for the frozen role map."""

    producer_dirs = {str(role): Path(path) for role, path in dict(producers).items()}
    unexpected = sorted(set(producer_dirs) - set(active.producer_manifest_map))
    if unexpected:
        raise ValueError(f"unexpected producer manifest(s): {unexpected}")
    missing = sorted(set(active.producer_manifest_map) - set(producer_dirs))
    if missing:
        raise ValueError(f"missing producer manifest(s): {missing}")
    manifests = {
        role: _load_completed_producer_manifest(role, directory)
        for role, directory in producer_dirs.items()
    }
    validate_producer_manifest_inputs(
        {role: str(manifest.get("schema")) for role, manifest in manifests.items()},
        catalog=active,
    )
    return producer_dirs, manifests


def _producer_source_entry(
    asset: CatalogAsset, manifests: Mapping[str, Mapping[str, object]]
) -> Mapping[str, object]:
    entries = manifests[asset.producer_manifest_role]["assets"]
    entry = entries.get(asset.source_artifact)
    if not isinstance(entry, dict):
        raise ValueError(
            f"source {asset.source_artifact!r} for {asset.public_basename!r} is "
            f"not inventoried by its producer manifest "
            f"{asset.producer_manifest_role!r} (unowned asset)"
        )
    label = f"{asset.producer_manifest_role}:{asset.source_artifact}"
    if entry.get("schema_id") != asset.schema_id:
        raise ValueError(
            f"{label}: producer schema_id {entry.get('schema_id')!r} does not "
            f"match the catalog schema {asset.schema_id!r}"
        )
    digest = entry.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{label}: producer inventory sha256 is not a hex digest")
    _validate_window(entry, label=label)
    return entry


def _verify_source_bytes(
    asset: CatalogAsset,
    entry: Mapping[str, object],
    producer_dirs: Mapping[str, Path],
) -> Path:
    label = f"{asset.producer_manifest_role}:{asset.source_artifact}"
    src = producer_dirs[asset.producer_manifest_role] / asset.source_artifact
    if not src.is_file():
        raise ValueError(f"{label}: source file is absent: {src}")
    actual = _sha256_file(src)
    if actual != entry["sha256"] or int(src.stat().st_size) != entry.get("size"):
        raise ValueError(
            f"{label}: source bytes are stale or corrupt against the completed "
            f"producer manifest (sha256 {actual[:12]}... != inventoried "
            f"{str(entry['sha256'])[:12]}...)"
        )
    return src


def _refuse_candidate_overwrite(destination: Path) -> None:
    if not destination.exists():
        return
    if not destination.is_dir():
        raise ValueError(f"candidate destination is not a directory: {destination}")
    completed_claim = False
    manifest_path = destination / PUBLICATION_MANIFEST
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text())
        except ValueError:
            loaded = None
        completed_claim = isinstance(loaded, dict) and loaded.get("completed") is True
    if completed_claim or (destination / CHECKSUM_FILE).exists():
        raise ValueError(
            f"candidate {destination} is a completed publication candidate and "
            "immutable; refusing overwrite"
        )
    if any(destination.iterdir()):
        raise ValueError(
            f"refusing to stage into non-empty candidate destination {destination}"
        )


def stage_publication_candidate(
    *,
    destination: Path | str,
    producers: Mapping[str, Path | str],
    publication_id: str,
    catalog: PublicationAssetCatalog | None = None,
    build_time: str | None = None,
) -> Path:
    """Stage one INCOMPLETE data-v4 candidate from completed producer manifests.

    Stages only cataloged assets into a new empty destination, verifies every
    source against its completed producer manifest first, and writes a
    provisional ``publication_manifest.json`` with ``completed=false`` LAST.
    ``SHA256SUMS`` and every completion claim are prohibited at this stage.
    A failed staging leaves the destination diagnosable but incomplete; recovery
    is delete-and-rebuild.
    """

    active = _active_catalog(catalog)
    if not isinstance(publication_id, str) or not publication_id.strip():
        raise ValueError("publication_id must be a non-empty string")
    build_time = _normalize_build_time(build_time)

    destination = Path(destination)
    _refuse_candidate_overwrite(destination)

    producer_dirs, manifests = _load_producer_boundary(producers, active)

    # verify EVERY source against its completed producer manifest before any
    # byte is staged: ownership, schema, window, then exact source bytes
    source_entries: dict[str, Mapping[str, object]] = {}
    for asset in active.assets:
        entry = _producer_source_entry(asset, manifests)
        _verify_source_bytes(asset, entry, producer_dirs)
        source_entries[asset.public_basename] = entry

    destination.mkdir(parents=True, exist_ok=True)
    inventory: dict[str, dict[str, object]] = {}
    for asset in active.assets:
        src = producer_dirs[asset.producer_manifest_role] / asset.source_artifact
        staged = destination / asset.public_basename
        shutil.copyfile(src, staged)
        entry = source_entries[asset.public_basename]
        staged_sha = _sha256_file(staged)
        if staged_sha != entry["sha256"]:
            raise ValueError(
                f"staged bytes for {asset.public_basename!r} do not match the "
                "verified producer source"
            )
        record: dict[str, object] = {
            "source_artifact": asset.source_artifact,
            "sha256": staged_sha,
            "size": int(staged.stat().st_size),
            "asset_class": asset.asset_class,
            "media_type": asset.media_type,
            "schema_id": asset.schema_id,
            "locale": asset.locale,
            "producer": asset.producer,
            "producer_manifest_role": asset.producer_manifest_role,
            "lineage": list(asset.lineage),
            "projection": asset.projection,
            "projection_of": asset.projection_of,
        }
        for key in ("rows", "start", "end"):
            if key in entry:
                record[key] = entry[key]
        inventory[asset.public_basename] = record

    manifest = {
        "schema": PUBLICATION_MANIFEST_SCHEMA,
        # the read-only workbook release client addresses the manifest through
        # ``schema_id`` plus an ``artifacts`` list; both representations are
        # written together and validated for exact parity so they cannot drift
        "schema_id": PUBLICATION_MANIFEST_SCHEMA,
        "artifacts": [
            {
                "path": name,
                "sha256": inventory[name]["sha256"],
                "size": inventory[name]["size"],
            }
            for name in sorted(inventory)
        ],
        "release_tag": active.release_tag,
        "publication_id": publication_id,
        "build_time": build_time,
        "catalog_sha256": catalog_sha256(active),
        "input_manifests": {
            role: {
                "schema": str(manifests[role].get("schema")),
                "manifest_sha256": _sha256_file(
                    producer_dirs[role] / PRODUCER_MANIFEST_NAME
                ),
            }
            for role in sorted(producer_dirs)
        },
        "assets": inventory,
        "compatibility_paths": [
            {"public_path": path, "target": target}
            for path, target in active.compatibility_paths
        ],
        "completed": False,
    }
    (destination / PUBLICATION_MANIFEST).write_text(_canonical_json(manifest))
    validate_staged_candidate(destination, producers=producer_dirs, catalog=catalog)
    return destination


def _load_candidate_manifest(
    candidate: Path, active: PublicationAssetCatalog
) -> dict[str, object]:
    if not candidate.is_dir():
        raise ValueError(f"candidate directory is absent: {candidate}")
    manifest_path = candidate / PUBLICATION_MANIFEST
    if not manifest_path.is_file():
        raise ValueError(
            f"{candidate}: {PUBLICATION_MANIFEST} is missing; the candidate is "
            "incomplete"
        )
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"{candidate}: publication manifest must be a JSON object")
    if manifest.get("schema") != PUBLICATION_MANIFEST_SCHEMA:
        raise ValueError(
            f"{candidate}: unknown publication manifest schema "
            f"{manifest.get('schema')!r}"
        )
    if manifest.get("release_tag") != active.release_tag:
        raise ValueError(
            f"{candidate}: publication release tag must be {active.release_tag!r}"
        )
    if manifest.get("catalog_sha256") != catalog_sha256(active):
        raise ValueError(
            f"{candidate}: catalog_sha256 does not match the frozen asset catalog"
        )
    publication_id = manifest.get("publication_id")
    if not isinstance(publication_id, str) or not publication_id.strip():
        raise ValueError(f"{candidate}: publication_id must be a non-empty string")
    _normalize_build_time(str(manifest.get("build_time")))
    if not isinstance(manifest.get("completed"), bool):
        raise ValueError(f"{candidate}: completed must be a JSON boolean")
    if not isinstance(manifest.get("assets"), dict):
        raise ValueError(f"{candidate}: manifest must inventory its assets")
    if not isinstance(manifest.get("input_manifests"), dict):
        raise ValueError(f"{candidate}: manifest must record its input manifests")
    if manifest.get("schema_id") != PUBLICATION_MANIFEST_SCHEMA:
        raise ValueError(
            f"{candidate}: manifest schema_id must be "
            f"{PUBLICATION_MANIFEST_SCHEMA!r} (release-client contract)"
        )
    artifacts = manifest.get("artifacts")
    assets = manifest["assets"]
    if not isinstance(artifacts, list):
        raise ValueError(
            f"{candidate}: manifest must carry the release-client artifacts list"
        )
    listed = [
        row.get("path") if isinstance(row, dict) else None for row in artifacts
    ]
    if listed != sorted(assets):
        raise ValueError(
            f"{candidate}: artifacts list diverges from the asset inventory"
        )
    for row in artifacts:
        entry = assets[row["path"]]
        if (
            not isinstance(entry, dict)
            or row.get("sha256") != entry.get("sha256")
            or row.get("size") != entry.get("size")
        ):
            raise ValueError(
                f"{candidate}: artifacts entry {row.get('path')!r} diverges "
                "from the asset inventory"
            )
    expected_paths = [
        {"public_path": path, "target": target}
        for path, target in active.compatibility_paths
    ]
    if manifest.get("compatibility_paths") != expected_paths:
        raise ValueError(
            f"{candidate}: compatibility paths diverge from the frozen catalog"
        )
    return manifest


def _check_candidate_files(
    candidate: Path,
    active: PublicationAssetCatalog,
    manifest: Mapping[str, object],
    *,
    finalized: bool,
) -> None:
    """Direct validators: inventory, metadata, windows, duplicates, extras, bytes."""

    assets = manifest["assets"]
    expected = set(active.payload_basenames)
    missing_inventory = sorted(expected - set(assets))
    if missing_inventory:
        raise ValueError(
            f"{candidate}: manifest inventory is missing asset(s): "
            f"{missing_inventory[:5]}"
        )
    extra_inventory = sorted(set(assets) - expected)
    if extra_inventory:
        raise ValueError(
            f"{candidate}: manifest inventories uncataloged asset(s): "
            f"{extra_inventory[:5]}"
        )

    by_name = {asset.public_basename: asset for asset in active.assets}
    for name, entry in assets.items():
        asset = by_name[name]
        if not isinstance(entry, dict):
            raise ValueError(f"{candidate}: {name} inventory entry must be an object")
        if (
            entry.get("schema_id") != asset.schema_id
            or entry.get("media_type") != asset.media_type
            or entry.get("locale") != asset.locale
        ):
            raise ValueError(
                f"{candidate}: {name} schema/media/locale metadata diverges from "
                "the catalog"
            )
        if (
            entry.get("producer") != asset.producer
            or entry.get("producer_manifest_role") != asset.producer_manifest_role
            or entry.get("source_artifact") != asset.source_artifact
        ):
            raise ValueError(
                f"{candidate}: {name} producer metadata diverges from the catalog"
            )
        if tuple(entry.get("lineage") or ()) != asset.lineage:
            raise ValueError(
                f"{candidate}: {name} lineage diverges from the catalog"
            )
        if (
            entry.get("asset_class") != asset.asset_class
            or entry.get("projection") != asset.projection
            or entry.get("projection_of") != asset.projection_of
        ):
            raise ValueError(
                f"{candidate}: {name} projection metadata diverges from the catalog"
            )
        _validate_window(entry, label=f"{candidate}: {name}")

    allowed = expected | {active.publication_manifest}
    if finalized:
        allowed.add(active.checksum_file)
    names = sorted(entry.name for entry in candidate.iterdir())
    folded: dict[str, str] = {}
    for name in names:
        key = name.casefold()
        if key in folded:
            raise ValueError(
                f"{candidate}: duplicate public basename {name!r} collides with "
                f"{folded[key]!r}"
            )
        folded[key] = name
    forbidden = {item.casefold() for item in active.forbidden_release_files}
    for name in names:
        if name.casefold() in forbidden:
            raise ValueError(
                f"{candidate}: forbidden release file {name!r} — completion is "
                "represented only by the final publication manifest, never a "
                "COMPLETED marker"
            )
    allowed_folded = {name.casefold(): name for name in allowed}
    for name in names:
        if name not in allowed and name.casefold() in allowed_folded:
            raise ValueError(
                f"{candidate}: duplicate public basename {name!r} collides with "
                f"{allowed_folded[name.casefold()]!r}"
            )
    extra_files = sorted(set(names) - allowed)
    if extra_files:
        raise ValueError(f"{candidate}: extra file(s) present: {extra_files[:5]}")
    missing_files = sorted(expected - set(names))
    if missing_files:
        raise ValueError(
            f"{candidate}: staged asset(s) missing from disk: {missing_files[:5]}"
        )

    for name, entry in assets.items():
        path = candidate / name
        actual = _sha256_file(path)
        if actual != entry.get("sha256") or int(path.stat().st_size) != entry.get(
            "size"
        ):
            raise ValueError(
                f"{candidate}: {name} bytes were mutated after staging "
                f"(sha256 {actual[:12]}... != inventoried "
                f"{str(entry.get('sha256'))[:12]}...)"
            )


def _check_candidate_sources(
    candidate: Path,
    active: PublicationAssetCatalog,
    manifest: Mapping[str, object],
    producers: Mapping[str, Path | str],
) -> None:
    """Source-hash validators against the completed producer manifests."""

    producer_dirs, producer_manifests = _load_producer_boundary(producers, active)
    recorded = manifest["input_manifests"]
    for role, directory in producer_dirs.items():
        entry = recorded.get(role)
        if not isinstance(entry, dict):
            raise ValueError(
                f"{candidate}: input manifest for {role!r} is not recorded"
            )
        if entry.get("manifest_sha256") != _sha256_file(
            directory / PRODUCER_MANIFEST_NAME
        ) or entry.get("schema") != producer_manifests[role].get("schema"):
            raise ValueError(
                f"{candidate}: recorded input manifest for {role!r} diverges from "
                "the supplied producer manifest"
            )
    assets = manifest["assets"]
    for asset in active.assets:
        entry = _producer_source_entry(asset, producer_manifests)
        staged = assets[asset.public_basename]
        if staged.get("sha256") != entry["sha256"]:
            raise ValueError(
                f"{candidate}: {asset.public_basename} source hash diverges from "
                "its completed producer manifest"
            )
        for key in ("rows", "start", "end"):
            if staged.get(key) != entry.get(key):
                raise ValueError(
                    f"{candidate}: {asset.public_basename} window diverges from "
                    "its completed producer manifest"
                )
        _verify_source_bytes(asset, entry, producer_dirs)


def validate_staged_candidate(
    candidate: Path | str,
    *,
    producers: Mapping[str, Path | str] | None = None,
    catalog: PublicationAssetCatalog | None = None,
) -> dict[str, object]:
    """Directly validate one INCOMPLETE staged candidate (task 10.8).

    The provisional manifest must declare ``completed=false``, ``SHA256SUMS``
    must not exist, and every schema/values/windows/inventory/lineage/
    duplicate/extra-file check must pass. When ``producers`` is supplied, every
    staged hash is additionally verified against its completed producer
    manifest and the current source bytes.
    """

    active = _active_catalog(catalog)
    candidate = Path(candidate)
    manifest = _load_candidate_manifest(candidate, active)
    if manifest["completed"] is not False:
        raise ValueError(
            f"{candidate}: provisional manifest must declare completed=false; "
            "completion claims are prohibited before finalization"
        )
    if (candidate / active.checksum_file).exists():
        raise ValueError(
            f"{candidate}: {active.checksum_file} is prohibited before finalization"
        )
    _check_candidate_files(candidate, active, manifest, finalized=False)
    if producers is not None:
        _check_candidate_sources(candidate, active, manifest, producers)
    return manifest


def finalize_publication_candidate(
    candidate: Path | str,
    *,
    producers: Mapping[str, Path | str] | None = None,
    catalog: PublicationAssetCatalog | None = None,
) -> Path:
    """Finalize a validated provisional candidate EXACTLY once (task 10.9).

    Replaces the provisional manifest with canonical ``completed=true``
    metadata, then generates ``SHA256SUMS`` LAST — covering payloads plus the
    final manifest and never itself — and verifies the finalized bytes through
    the read-only readers. Any fault leaves the candidate diagnosable but
    incomplete.
    """

    active = _active_catalog(catalog)
    candidate = Path(candidate)
    if (candidate / active.checksum_file).exists():
        raise ValueError(
            f"{candidate}: already finalized ({active.checksum_file} present); "
            "finalization happens exactly once"
        )
    manifest_path = candidate / PUBLICATION_MANIFEST
    if manifest_path.is_file():
        try:
            claimed = json.loads(manifest_path.read_text())
        except ValueError:
            claimed = None
        if isinstance(claimed, dict) and claimed.get("completed") is True:
            raise ValueError(
                f"{candidate}: already finalized (completed=true); finalization "
                "happens exactly once"
            )

    provisional = validate_staged_candidate(
        candidate, producers=producers, catalog=catalog
    )
    manifest_path.write_text(_canonical_json({**provisional, "completed": True}))

    # SHA256SUMS is generated LAST: payloads plus the final manifest, excluding
    # itself, over the exact finalized bytes.
    (candidate / active.checksum_file).write_text(
        "".join(
            f"{_sha256_file(candidate / name)}  {name}\n"
            for name in active.checksum_basenames
        )
    )
    verify_finalized_candidate(candidate, catalog=catalog)
    return candidate


def verify_finalized_candidate(
    candidate: Path | str,
    *,
    catalog: PublicationAssetCatalog | None = None,
) -> dict[str, object]:
    """Read-only verification of one finalized candidate (task 10.9).

    Reads the final manifest and ``SHA256SUMS`` and re-hashes every finalized
    byte without mutating anything. The manifest must declare
    ``completed=true`` and the checksum file must cover exactly the payloads
    plus the final manifest, never itself.
    """

    active = _active_catalog(catalog)
    candidate = Path(candidate)
    manifest = _load_candidate_manifest(candidate, active)
    if manifest["completed"] is not True:
        raise ValueError(
            f"{candidate}: finalized manifest must declare completed=true; the "
            "candidate is not finalized"
        )
    _check_candidate_files(candidate, active, manifest, finalized=True)

    sums_path = candidate / active.checksum_file
    if not sums_path.is_file():
        raise ValueError(f"{candidate}: {active.checksum_file} is missing")
    parsed: dict[str, str] = {}
    for lineno, line in enumerate(sums_path.read_text().splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (\S.*)", line)
        if match is None:
            raise ValueError(
                f"{candidate}: {active.checksum_file} line {lineno} is malformed"
            )
        digest, name = match.groups()
        if name in parsed:
            raise ValueError(
                f"{candidate}: {active.checksum_file} lists {name!r} twice"
            )
        parsed[name] = digest
    if active.checksum_file in parsed:
        raise ValueError(f"{candidate}: {active.checksum_file} must never hash itself")
    if tuple(sorted(parsed)) != active.checksum_basenames:
        raise ValueError(
            f"{candidate}: {active.checksum_file} coverage differs from payloads "
            "plus the final manifest"
        )
    for name, digest in parsed.items():
        if _sha256_file(candidate / name) != digest:
            raise ValueError(
                f"{candidate}: {name} bytes do not match {active.checksum_file}"
            )
    return manifest


def promote_finalized_candidate(
    candidate: Path | str,
    final_destination: Path | str,
    *,
    catalog: PublicationAssetCatalog | None = None,
) -> Path:
    """Atomically promote a verified finalized candidate (task 10.9).

    The final local destination must be ABSENT; an existing destination is
    never overwritten. Verification runs first and any fault — verification or
    the rename itself — leaves the final destination absent.
    """

    candidate = Path(candidate)
    final = Path(final_destination)
    if final.exists():
        raise ValueError(
            f"refusing to overwrite existing final destination {final}; "
            "promotion requires an absent destination"
        )
    verify_finalized_candidate(candidate, catalog=catalog)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.rename(candidate, final)
    return final


# --------------------------------------------------------------------------- #
# Task 10.10: offline clean-room upload-set smoke tooling.                     #
# Task 10.11: read-only public-release smoke-test hooks.                       #
#                                                                              #
# The smoke path is strictly read-only: it enumerates, downloads, and          #
# verifies.  There is NO release-creation, tagging, or upload capability       #
# anywhere in this module.  Verification runs from a clean temporary           #
# directory through the REAL workbook release client so no repository data     #
# path or mutable current-release pointer can compensate for a broken          #
# upload set.                                                                  #
# --------------------------------------------------------------------------- #

_MUTABLE_TAG_NAMES = frozenset({"latest", "current", "data-current"})
_IMMUTABLE_TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_HTTP_TIMEOUT = 30.0
_CLIENT_VERIFICATION = "publication_manifest_sha256"
_MUTABLE_POINTER_PROBES = (
    "/releases/download/latest/{manifest}",
    "/releases/download/current/{manifest}",
    "/data/current.json",
)
# Representative S0-S5 walkthrough consumption over the data-v4 payloads:
# each step family loads its canonical tables through the real release client
# plus the schema loaders, from the clean room.
_REPRESENTATIVE_STEP_BUILDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "S0",
        (
            "tear_sheet_static_bh_windows.parquet",
            "tear_sheet_static_bh_window_dashboard.parquet",
        ),
    ),
    ("S1", ("factor_replay_audit_ext2026.json", "factor_evidence_ext2026.parquet")),
    ("S2", ("factor_contrast_ext2026.parquet", "factor_contrast_split_ext2026.json")),
    (
        "S3",
        (
            "factor_scores_ext2026.parquet",
            "factor_loadings_ext2026.parquet",
            "factor_decision_log_ext2026.json",
        ),
    ),
    (
        "S4",
        (
            "factor_equity_ext2026.parquet",
            "factor_targets_ext2026.parquet",
            "portfolio_metrics_reader_ext2026.parquet",
        ),
    ),
    (
        "S5",
        (
            "sjm_crowding_v3_total_return_bil_equity_ext2026.parquet",
            "tear_sheet_trio_ext2026.parquet",
            "tear_sheet_sjm_crowding_ext2026.parquet",
        ),
    ),
)


class TagNotPublishedError(ValueError):
    """The requested release tag does not exist at the public endpoint."""


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _require_immutable_tag(tag: object) -> str:
    if (
        not isinstance(tag, str)
        or not _IMMUTABLE_TAG_PATTERN.fullmatch(tag)
        or tag.casefold() in _MUTABLE_TAG_NAMES
    ):
        raise ValueError(
            f"tag must be an explicit immutable release tag, got {tag!r}"
        )
    return tag


def _prepare_clean_room(clean_room: Path | str) -> Path:
    """An absent or empty directory OUTSIDE the repository tree (clean cache)."""

    room = Path(clean_room)
    resolved = room.resolve()
    repo = _repository_root()
    if resolved == repo or repo in resolved.parents:
        raise ValueError(
            f"clean room {room} is inside the repository; the smoke run must "
            "not have any repository data path available"
        )
    if room.exists() and (not room.is_dir() or any(room.iterdir())):
        raise ValueError(
            f"clean room {room} must be an absent or empty directory so the "
            "release cache starts clean"
        )
    room.mkdir(parents=True, exist_ok=True)
    return room


def _workbook_release():
    """Import the REAL workbook release client module (read-only dependency)."""

    workbook_dir = _repository_root() / "workbook"
    if str(workbook_dir) not in sys.path:
        sys.path.insert(0, str(workbook_dir))
    import factor_workbook.release as release_module

    return release_module


@contextmanager
def _release_client_endpoint(release_module, download_template: str):
    """Point the release client's download template at one explicit endpoint."""

    previous = release_module._DOWNLOAD_URL
    release_module._DOWNLOAD_URL = download_template
    try:
        yield
    finally:
        release_module._DOWNLOAD_URL = previous


def _http_get(url: str) -> tuple[int, bytes]:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), b""
    except urllib.error.URLError as exc:
        raise ValueError(f"release endpoint unreachable: {url}: {exc.reason}") from exc


@dataclass
class ServedRelease:
    """Handle for one temporary localhost release endpoint.

    Attributes:
        base_url: ``http://127.0.0.1:<port>`` root of the endpoint.
        requests: ``(method, path)`` log of every request the endpoint saw.
    """

    base_url: str
    requests: list[tuple[str, str]] = field(default_factory=list)


@contextmanager
def serve_publication_candidate(
    candidate: Path | str,
    *,
    tag: str | None = None,
    historical: Mapping[str, Path | str] | None = None,
    listing_names: list[str] | None = None,
) -> Iterator[ServedRelease]:
    """Serve one local candidate at GitHub-release-shaped localhost paths.

    Exposes ``/releases/download/<tag>/<asset>`` for byte downloads and
    ``/api/releases/tags/<tag>`` for the asset listing — the same shapes the
    public release endpoint has.  Strictly read-only: non-GET methods get 405.
    No mutable current-release pointer exists: unknown tags (including
    ``latest``/``current``) are 404.  ``historical`` maps additional immutable
    tags to directories for frozen-history verification. ``listing_names``
    overrides the primary tag's listing (fault injection for smoke tests).
    """

    candidate = Path(candidate)
    if tag is None:
        manifest_path = candidate / PUBLICATION_MANIFEST
        if not manifest_path.is_file():
            raise ValueError(
                f"{candidate}: cannot infer the release tag without "
                f"{PUBLICATION_MANIFEST}; pass tag= explicitly"
            )
        tag = str(json.loads(manifest_path.read_text()).get("release_tag"))
    primary_tag = _require_immutable_tag(tag)
    trees: dict[str, Path] = {primary_tag: candidate}
    for extra_tag, extra_dir in dict(historical or {}).items():
        trees[_require_immutable_tag(extra_tag)] = Path(extra_dir)

    served = ServedRelease(base_url="")

    def _tag_listing(request_tag: str) -> dict[str, object]:
        tree = trees[request_tag]
        if listing_names is not None and request_tag == primary_tag:
            names = list(listing_names)
        else:
            names = sorted(
                entry.name for entry in tree.iterdir() if entry.is_file()
            )
        return {
            "tag_name": request_tag,
            "draft": False,
            "prerelease": False,
            "assets": [
                {
                    "name": name,
                    "size": (
                        int((tree / name).stat().st_size)
                        if (tree / name).is_file()
                        else 0
                    ),
                }
                for name in names
            ],
        }

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # pragma: no cover - silence stdlib log
            pass

        def _reply(self, status: int, body: bytes = b"") -> None:
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _refuse_write(self) -> None:
            served.requests.append((self.command, self.path))
            self._reply(405)

        do_POST = do_PUT = do_PATCH = do_DELETE = _refuse_write

        def do_GET(self):  # noqa: N802 - stdlib API name
            served.requests.append(("GET", self.path))
            parts = self.path.split("?", 1)[0].strip("/").split("/")
            if parts[:3] == ["api", "releases", "tags"] and len(parts) == 4:
                if parts[3] in trees:
                    body = json.dumps(_tag_listing(parts[3])).encode()
                    self._reply(200, body)
                else:
                    self._reply(404)
                return
            if parts[:2] == ["releases", "download"] and len(parts) == 4:
                tree = trees.get(parts[2])
                name = parts[3]
                if tree is not None and PurePosixPath(name).name == name:
                    path = tree / name
                    if path.is_file():
                        self._reply(200, path.read_bytes())
                        return
            self._reply(404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    served.base_url = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield served
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _load_asset_payload(
    asset: CatalogAsset, data: bytes, entry: Mapping[str, object] | None
) -> int:
    """Load one asset's bytes as its declared media/schema; return its extent."""

    if asset.media_type == _PARQUET_MEDIA_TYPE:
        import pyarrow.parquet as pq  # heavy import stays local to the loader

        try:
            table = pq.read_table(io.BytesIO(data))
        except (ValueError, OSError) as exc:
            raise ValueError(f"not a readable parquet table: {exc}") from exc
        if table.num_columns == 0 or table.num_rows == 0:
            raise ValueError("parquet table is empty")
        if entry is not None and "rows" in entry:
            if int(table.num_rows) != entry["rows"]:
                raise ValueError(
                    f"parquet row count {table.num_rows} does not match the "
                    f"manifest window rows {entry['rows']}"
                )
        return int(table.num_rows)
    if asset.media_type == _CSV_MEDIA_TYPE:
        delimiter = ";" if asset.locale == "de-DE" else ","
        try:
            rows = list(
                csv.reader(io.StringIO(data.decode("utf-8")), delimiter=delimiter)
            )
        except csv.Error as exc:
            raise ValueError(f"not parseable CSV: {exc}") from exc
        if len(rows) < 2:
            raise ValueError("CSV must carry a header row plus data rows")
        widths = {len(row) for row in rows}
        if len(widths) != 1 or widths == {1}:
            raise ValueError(
                f"CSV columns are inconsistent for delimiter {delimiter!r}"
            )
        return len(rows) - 1
    if asset.media_type == _JSON_MEDIA_TYPE:
        payload = json.loads(data.decode("utf-8"))
        if not isinstance(payload, (dict, list)) or not payload:
            raise ValueError("expected a non-empty JSON object or array")
        return len(payload)
    if asset.media_type == _PNG_MEDIA_TYPE:
        if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
            raise ValueError("not a PNG stream")
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        if width < 1 or height < 1:
            raise ValueError("PNG dimensions are empty")
        return width * height
    if asset.media_type in (_MARKDOWN_MEDIA_TYPE, _TEX_MEDIA_TYPE):
        text = data.decode("utf-8")
        if not text.strip():
            raise ValueError("formatted report is empty")
        return len(text)
    raise ValueError(f"unknown media type {asset.media_type!r}")


def validate_catalog_asset_bytes(
    asset: CatalogAsset, data: bytes, entry: Mapping[str, object] | None = None
) -> int:
    """Schema loader: bytes must parse as the asset's declared media/schema."""

    try:
        return _load_asset_payload(asset, data, entry)
    except ValueError as exc:
        raise ValueError(
            f"schema-invalid asset {asset.public_basename!r} "
            f"({asset.schema_id}): {exc}"
        ) from exc


def _enumerate_release_listing(api_base: str, tag: str) -> list[str]:
    """Enumerate public asset names for one tag; verify release metadata."""

    url = f"{api_base.rstrip('/')}/releases/tags/{tag}"
    status, body = _http_get(url)
    if status == 404:
        raise TagNotPublishedError(
            f"release tag {tag!r} is not yet published at {api_base}"
        )
    if status != 200:
        raise ValueError(f"release listing failed: HTTP {status} at {url}")
    try:
        listing = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"release listing at {url} is not valid JSON") from exc
    if not isinstance(listing, dict) or listing.get("tag_name") != tag:
        raise ValueError(
            f"release metadata tag_name diverges from the requested tag {tag!r}"
        )
    if listing.get("draft") or listing.get("prerelease"):
        raise ValueError(
            f"release {tag!r} must be a final published release, not a draft "
            "or prerelease"
        )
    rows = listing.get("assets")
    if not isinstance(rows, list):
        raise ValueError(f"release listing at {url} carries no asset list")
    names: list[str] = []
    folded: dict[str, str] = {}
    for row in rows:
        name = row.get("name") if isinstance(row, dict) else None
        if not isinstance(name, str) or PurePosixPath(name).name != name:
            raise ValueError("release listing contains a non-basename asset name")
        key = name.casefold()
        if key in folded:
            raise ValueError(
                f"release listing contains duplicate asset {name!r} "
                f"(collides with {folded[key]!r})"
            )
        folded[key] = name
        names.append(name)
    return names


def _run_representative_step_builds(
    client,
    by_name: Mapping[str, CatalogAsset],
    manifest_assets: Mapping[str, object],
) -> dict[str, dict[str, int]]:
    builds: dict[str, dict[str, int]] = {}
    for step, names in _REPRESENTATIVE_STEP_BUILDS:
        loaded: dict[str, int] = {}
        for name in names:
            data, provenance = client.fetch(name)
            if not provenance.verified:
                raise ValueError(
                    f"step build {step}: {name!r} was not manifest-verified"
                )
            loaded[name] = validate_catalog_asset_bytes(
                by_name[name], data, manifest_assets.get(name)
            )
        builds[step] = loaded
    return builds


def _run_release_smoke_core(
    *,
    base_url: str,
    api_base: str,
    tag: str,
    clean_room: Path | str,
    active: PublicationAssetCatalog,
) -> dict[str, object]:
    _require_immutable_tag(tag)
    if tag != active.release_tag:
        raise ValueError(
            f"smoke tag {tag!r} is not the cataloged publication tag "
            f"{active.release_tag!r}"
        )
    room = _prepare_clean_room(clean_room)
    base = base_url.rstrip("/")

    # no mutable current-release pointer may be reachable
    for probe in _MUTABLE_POINTER_PROBES:
        status, _ = _http_get(
            base + probe.format(manifest=active.publication_manifest)
        )
        if status == 200:
            raise ValueError(
                f"a mutable current-release pointer is reachable at "
                f"{base + probe.format(manifest=active.publication_manifest)}; "
                "the release endpoint must serve immutable tags only"
            )

    served_names = _enumerate_release_listing(api_base, tag)

    mirror = room / "mirror"
    mirror.mkdir()
    for control in (active.publication_manifest, active.checksum_file):
        status, payload = _http_get(f"{base}/releases/download/{tag}/{control}")
        if status != 200:
            raise ValueError(
                f"{control} is missing from the {tag} release (HTTP {status})"
            )
        (mirror / control).write_bytes(payload)
    manifest = _load_candidate_manifest(mirror, active)
    if manifest["completed"] is not True:
        raise ValueError(
            f"the published {active.publication_manifest} must declare "
            "completed=true"
        )

    expected = set(manifest["assets"]) | {
        active.publication_manifest,
        active.checksum_file,
    }
    served = set(served_names)
    missing = sorted(expected - served)
    if missing:
        raise ValueError(
            f"asset(s) missing from the {tag} release listing: {missing[:5]}"
        )
    extra = sorted(served - expected)
    if extra:
        raise ValueError(
            f"extra unmanifested asset(s) in the {tag} release: {extra[:5]}"
        )

    release_module = _workbook_release()
    client = release_module.ReleaseClient(
        tag, cache_dir=room / "cache", token_provider=lambda: None
    )
    by_name = {asset.public_basename: asset for asset in active.assets}
    template = f"{base}/releases/download/{{tag}}/{{asset}}"
    with _release_client_endpoint(release_module, template):
        for name in active.payload_basenames:
            try:
                data, provenance = client.fetch(name)
            except release_module.ReleaseError as exc:
                raise ValueError(f"clean-room fetch failed: {exc}") from exc
            if (
                not provenance.verified
                or provenance.verification != _CLIENT_VERIFICATION
            ):
                raise ValueError(
                    f"the release client did not manifest-verify {name!r}"
                )
            (mirror / name).write_bytes(data)
        # full read-only re-verification of the mirrored release: manifest
        # metadata, exact inventory, and SHA256SUMS over the mirrored bytes
        verify_finalized_candidate(mirror, catalog=active)
        # schema loaders over every payload
        for name in active.payload_basenames:
            validate_catalog_asset_bytes(
                by_name[name],
                (mirror / name).read_bytes(),
                manifest["assets"].get(name),
            )
        step_builds = _run_representative_step_builds(
            client, by_name, manifest["assets"]
        )

    provenance_rows = client.provenance_table()
    if not provenance_rows or not all(row.verified for row in provenance_rows):
        raise ValueError("release client provenance shows unverified fetches")

    return {
        "state": "verified",
        "tag": tag,
        "publication_id": manifest["publication_id"],
        "assets_verified": len(active.payload_basenames),
        "checksum_file_verified": True,
        "schemas_validated": len(active.payload_basenames),
        "step_builds": step_builds,
    }


def run_offline_release_smoke(
    *,
    base_url: str,
    tag: str,
    clean_room: Path | str,
    catalog: PublicationAssetCatalog | None = None,
) -> dict[str, object]:
    """Offline clean-room smoke of one served upload set (task 10.10).

    Runs the REAL explicit-tag workbook release client, checksum
    verification, the schema loaders, and representative S0-S5 builds from a
    clean temporary directory against a temporary localhost endpoint.  The
    endpoint must be local (offline), the clean room must be empty and
    outside the repository, and no mutable current-release pointer may be
    reachable.
    """

    active = _active_catalog(catalog)
    host = urllib.parse.urlsplit(base_url).hostname
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError(
            f"offline clean-room smoke requires a localhost endpoint, got "
            f"{base_url!r}"
        )
    report = _run_release_smoke_core(
        base_url=base_url,
        api_base=base_url.rstrip("/") + "/api",
        tag=tag,
        clean_room=clean_room,
        active=active,
    )
    report["mode"] = "offline_clean_room"
    return report


def _verify_frozen_release_pins(
    base_url: str,
    historical_pins: Mapping[str, Mapping[str, str]],
    room: Path,
) -> dict[str, dict[str, int]]:
    """Frozen-hash verification of immutable historical tags (read-only)."""

    release_module = _workbook_release()
    base = base_url.rstrip("/")
    template = f"{base}/releases/download/{{tag}}/{{asset}}"
    results: dict[str, dict[str, int]] = {}
    with _release_client_endpoint(release_module, template):
        for pin_tag in sorted(historical_pins):
            _require_immutable_tag(pin_tag)
            expected_assets = historical_pins[pin_tag]
            client = release_module.ReleaseClient(
                pin_tag, cache_dir=room / "cache", token_provider=lambda: None
            )
            for asset_name in sorted(expected_assets):
                expected_sha = str(expected_assets[asset_name])
                try:
                    data, _ = client.fetch(asset_name)
                except release_module.ReleaseError as exc:
                    raise ValueError(
                        f"historical tag {pin_tag!r} asset {asset_name!r} is "
                        f"unavailable: {exc}"
                    ) from exc
                if hashlib.sha256(data).hexdigest() != expected_sha:
                    raise ValueError(
                        f"historical tag {pin_tag!r} asset {asset_name!r} no "
                        "longer matches its frozen hash; immutable history "
                        "has changed"
                    )
            results[pin_tag] = {"assets_verified": len(expected_assets)}
    return results


def run_public_release_smoke(
    *,
    tag: str,
    base_url: str,
    clean_room: Path | str,
    api_base_url: str | None = None,
    catalog: PublicationAssetCatalog | None = None,
    historical_pins: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, object]:
    """Read-only public-release smoke hooks (task 10.11).

    Accepts only an explicit immutable ``tag`` and a public ``base_url``;
    the entire code path can only enumerate, download, and verify — it has no
    release-creation or upload capability.  Public assets are compared with
    the publication manifest, checksums and schemas are verified with a clean
    cache, and ``historical_pins`` (``{tag: {asset: sha256}}``) verify that
    immutable historical tags still match their frozen hashes.  An absent
    public release reports ``state="not_yet_published"``.
    """

    active = _active_catalog(catalog)
    _require_immutable_tag(tag)
    split = urllib.parse.urlsplit(base_url) if isinstance(base_url, str) else None
    if split is None or split.scheme not in ("http", "https") or not split.hostname:
        raise ValueError(
            f"base_url must be an explicit http(s) release address, got "
            f"{base_url!r}"
        )
    api_base = (
        api_base_url if api_base_url is not None else base_url.rstrip("/") + "/api"
    )
    try:
        report = _run_release_smoke_core(
            base_url=base_url,
            api_base=api_base,
            tag=tag,
            clean_room=clean_room,
            active=active,
        )
    except TagNotPublishedError as exc:
        report = {"state": "not_yet_published", "tag": tag, "detail": str(exc)}
    report["mode"] = "public_read_only"
    if historical_pins:
        report["historical"] = _verify_frozen_release_pins(
            base_url, historical_pins, Path(clean_room)
        )
    return report
