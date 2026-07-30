#!/usr/bin/env python3
"""Validate the frozen remediation baseline, locks, and evidence matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SPEC = Path(__file__).resolve().parent
BASELINE = SPEC / "baseline_inventory.json"
COVERAGE = SPEC / "coverage_matrix.json"
DEPENDENCIES = SPEC / "dependency_evidence.json"
PLACEHOLDERS = ("placeholder", "todo", "tbd", "fill-me", "example_test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _metadata(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            schema = parquet.schema_arrow
            result: dict[str, Any] = {
                "format": "parquet",
                "rows": parquet.metadata.num_rows,
                "columns": [{"name": field.name, "type": str(field.type)} for field in schema],
            }
            candidates = [
                name
                for name in schema.names
                if name.lower() in {"date", "datetime", "index", "timestamp"}
                or "date" in name.lower()
            ]
            for column in candidates[:2]:
                try:
                    values = [
                        value
                        for value in pq.read_table(path, columns=[column]).column(0).to_pylist()
                        if value is not None
                    ]
                    if values:
                        result["window"] = {
                            "field": column,
                            "start": str(min(values)),
                            "end": str(max(values)),
                        }
                        break
                except Exception:
                    pass
        elif suffix == ".csv":
            with path.open(newline="", encoding="utf-8-sig", errors="replace") as stream:
                sample = stream.read(8192)
                stream.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                except csv.Error:
                    dialect = csv.excel
                reader = csv.reader(stream, dialect)
                header = next(reader, [])
                rows = 0
                dates: list[str] = []
                date_index = next(
                    (
                        index
                        for index, name in enumerate(header)
                        if name.lower() in {"date", "datetime", "timestamp"}
                        or "date" in name.lower()
                    ),
                    None,
                )
                for row in reader:
                    rows += 1
                    if date_index is not None and date_index < len(row) and row[date_index]:
                        dates.append(row[date_index])
                result = {
                    "format": "csv",
                    "rows": rows,
                    "columns": header,
                    "delimiter": dialect.delimiter,
                }
                if dates:
                    result["window"] = {
                        "field": header[date_index],
                        "start": min(dates),
                        "end": max(dates),
                    }
        elif suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            result = {"format": "json", "top_level_type": type(value).__name__}
            if isinstance(value, dict):
                result["top_level_keys"] = sorted(value)
                result["rows"] = len(value)
            elif isinstance(value, list):
                result["rows"] = len(value)
        elif suffix == ".png":
            header = path.read_bytes()[:24]
            result = {"format": "png"}
            if header[:8] == b"\x89PNG\r\n\x1a\n":
                result["dimensions"] = {
                    "width": struct.unpack(">I", header[16:20])[0],
                    "height": struct.unpack(">I", header[20:24])[0],
                }
        elif suffix == ".ipynb":
            value = json.loads(path.read_text(encoding="utf-8"))
            cells = value.get("cells", [])
            result = {
                "format": "ipynb",
                "cells": len(cells),
                "code_cells": sum(cell.get("cell_type") == "code" for cell in cells),
                "markdown_cells": sum(cell.get("cell_type") == "markdown" for cell in cells),
            }
        else:
            result = {"format": suffix.lstrip(".") or "binary"}
    except Exception as exc:
        result = {
            "format": suffix.lstrip(".") or "unknown",
            "metadata_error": f"{type(exc).__name__}: {exc}",
        }
    result["schema_fingerprint"] = _fingerprint(result)
    return result


def _validate_pytest_reference(
    evidence: dict[str, Any], *, subject: str, errors: list[str]
) -> None:
    """Require a matrix reference to identify one collected pytest node exactly."""
    check_id = str(evidence.get("check_id", "")).strip()
    command = str(evidence.get("command", "")).strip()
    for field, value in (("check_id", check_id), ("command", command)):
        if not value or any(token in value.lower() for token in PLACEHOLDERS):
            errors.append(f"{subject}: invalid {field}")
    if "::" not in check_id:
        errors.append(f"{subject}: check_id is not a pytest node")
        return
    relative_path, node_name = check_id.split("::", 1)
    path = ROOT / relative_path
    if not node_name.startswith("test_") or not path.is_file():
        errors.append(f"{subject}: check_id is not collected")
    elif f"def {node_name}(" not in path.read_text(encoding="utf-8"):
        errors.append(f"{subject}: check_id is not collected")
    expected_command = f"uv run --frozen pytest -q {check_id}"
    if command != expected_command:
        errors.append(f"{subject}: command does not target check_id exactly")


def _validate_coverage(errors: list[str]) -> None:
    matrix = json.loads(COVERAGE.read_text())
    criteria = matrix.get("acceptance_criteria", [])
    defects = matrix.get("confirmed_defects", [])
    expected_criteria = {f"{requirement}.{criterion}" for requirement, count in {
        1: 6, 2: 8, 3: 9, 4: 5, 5: 6, 6: 4, 7: 6, 8: 8
    }.items() for criterion in range(1, count + 1)}
    actual_criteria = {entry.get("criterion") for entry in criteria}
    if len(criteria) != 52 or actual_criteria != expected_criteria:
        errors.append("coverage matrix must contain each of the 52 acceptance criteria exactly once")
    if len(defects) != 15 or {entry.get("defect_id") for entry in defects} != set(range(1, 16)):
        errors.append("coverage matrix must contain defects 1 through 15 exactly once")
    for entry in criteria:
        subject = f"criterion {entry.get('criterion')}"
        for field in ("owner_task", "final_validation_command"):
            value = str(entry.get(field, "")).strip()
            if not value or any(token in value.lower() for token in PLACEHOLDERS):
                errors.append(f"{subject}: invalid {field}")
        _validate_pytest_reference(entry, subject=subject, errors=errors)
    fixture_classes: set[str] = set()
    for entry in defects:
        for boundary in ("shared_boundary", "downstream_boundary"):
            _validate_pytest_reference(
                entry.get(boundary, {}),
                subject=f"defect {entry.get('defect_id')} {boundary}",
                errors=errors,
            )
        fixture_classes.update(entry.get("fixtures", []))
    required_fixtures = set(matrix.get("required_fixture_classes", []))
    if not required_fixtures.issubset(fixture_classes):
        errors.append(f"coverage matrix misses fixtures: {sorted(required_fixtures - fixture_classes)}")


def _validate_dependencies(errors: list[str]) -> None:
    evidence = json.loads(DEPENDENCIES.read_text())
    for section in ("root", "workbook"):
        record = evidence[section]
        project = ROOT / record["project"]
        lock = ROOT / record["lock"]
        if not project.exists() or not lock.exists():
            errors.append(f"{section}: missing project manifest or lock")
            continue
        if _sha256(project) != record["project_sha256"]:
            errors.append(f"{section}: project manifest changed after dependency evidence capture")
        if _sha256(lock) != record["lock_sha256"]:
            errors.append(f"{section}: lock changed after dependency evidence capture")
        declared = {
            dependency.split(">", 1)[0].split("=", 1)[0].split("[", 1)[0].strip()
            for dependency in tomllib.loads(project.read_text())["project"]["dependencies"]
        }
        missing = {
            item["package"] for item in record["direct_import_drift_reconciled"]
        } - declared
        if missing:
            errors.append(f"{section}: undeclared direct imports remain: {sorted(missing)}")


def _validate_artifacts(errors: list[str]) -> None:
    baseline = json.loads(BASELINE.read_text())
    releases = baseline.get("immutable_releases", [])
    if {release.get("tag") for release in releases} != {"data-v1", "data-v2", "data-v3"}:
        errors.append("baseline must freeze data-v1, data-v2, and data-v3")
    for release in releases:
        assets = release.get("assets", [])
        if len(assets) != release.get("asset_count") or len({a.get("name") for a in assets}) != len(assets):
            errors.append(f"{release.get('tag')}: invalid asset inventory")
        for asset in assets:
            digest = str(asset.get("digest", ""))
            if not digest.startswith("sha256:") or len(digest) != 71:
                errors.append(f"{release.get('tag')}/{asset.get('name')}: missing SHA-256 digest")
    for artifact in baseline.get("affected_artifacts", []):
        path = ROOT / artifact["path"]
        if not path.is_file():
            errors.append(f"missing baseline artifact: {artifact['path']}")
            continue
        if path.stat().st_size != artifact["size"]:
            errors.append(f"size changed: {artifact['path']}")
        if _sha256(path) != artifact["sha256"]:
            errors.append(f"content changed or stale baseline: {artifact['path']}")
        if _metadata(path)["schema_fingerprint"] != artifact["metadata"]["schema_fingerprint"]:
            errors.append(f"schema/window changed: {artifact['path']}")
        for producer in artifact.get("producers", []):
            producer_path = ROOT / producer["path"]
            if producer["exists"] and (
                not producer_path.is_file() or _sha256(producer_path) != producer["sha256"]
            ):
                errors.append(
                    f"producer lineage changed; artifact evidence may be stale: {artifact['path']} <- {producer['path']}"
                )


def _validate_live_releases(errors: list[str]) -> None:
    baseline = json.loads(BASELINE.read_text())
    for release in baseline["immutable_releases"]:
        tag = release["tag"]
        try:
            current = json.loads(subprocess.check_output(
                ["gh", "api", f"repos/norandom/Global_Macro_AI_Factors/releases/tags/{tag}"],
                text=True,
            ))
            ref = json.loads(subprocess.check_output(
                ["gh", "api", f"repos/norandom/Global_Macro_AI_Factors/git/ref/tags/{tag}"],
                text=True,
            ))
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            errors.append(f"{tag}: unable to verify live release: {exc}")
            continue
        assets = [
            {key: asset.get(key) for key in (
                "id", "name", "size", "content_type", "created_at", "updated_at",
                "digest", "browser_download_url",
            )}
            for asset in sorted(current["assets"], key=lambda value: value["name"])
        ]
        if current["id"] != release["release_id"]:
            errors.append(f"{tag}: release identity changed")
        if ref["object"]["sha"] != release["tag_commit"]:
            errors.append(f"{tag}: tag commit changed")
        if _fingerprint(assets) != release["asset_inventory_sha256"]:
            errors.append(f"{tag}: immutable asset inventory changed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-releases", action="store_true", help="compare frozen releases with GitHub")
    parser.add_argument("--skip-artifacts", action="store_true", help="skip current artifact/hash checks")
    args = parser.parse_args()
    errors: list[str] = []
    _validate_coverage(errors)
    _validate_dependencies(errors)
    if not args.skip_artifacts:
        _validate_artifacts(errors)
    if args.check_releases:
        _validate_live_releases(errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("foundation validation passed: 52 criteria, 15 defects, locks, baseline artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
