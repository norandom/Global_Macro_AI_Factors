"""Offline contracts for the standalone four-cell guard-ablation report bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macro_framework.evaluation import metric_block
from macro_framework.reporting import (
    LineMetadata,
    build_differential_metric_row,
    build_reader_metric_row,
)
from macro_framework.ssr import ssr_inference


CONFIGS = (
    "factor_pit_ext2026",
    "factor_pit_unguarded_diagnostic_ext2026",
    "factor_nonpit_diagnostic_ext2026",
    "factor_nonpit_unguarded_diagnostic_ext2026",
)
WINDOWS = ("full", "pre_cutoff", "post_cutoff")
COMPARISONS = (
    "pit_unguarded_minus_guarded",
    "nonpit_unguarded_minus_guarded",
    "nonpit_unguarded_minus_pit_guarded_combined_stress",
)


def _report_module():
    import importlib

    return importlib.import_module("scripts.build_tear_sheet")


def _values(configuration: str) -> pd.Series:
    dates = pd.bdate_range("2022-01-03", periods=61)
    offset = CONFIGS.index(configuration)
    returns = 0.001 + offset * 0.00013 + 0.00035 * np.sin(np.arange(60) / 5.0)
    # Keep every differential volatility finite but not near zero: the
    # deterministic fixture is exercising report ownership, not huge Sharpe.
    returns = returns + offset * 0.0001 * np.sin(np.arange(60) / 3.0)
    return pd.Series(np.r_[1.0, np.cumprod(1.0 + returns)], index=dates, name="value")


def _window_value(value: pd.Series, window: str) -> pd.Series:
    if window == "full":
        return value
    if window == "pre_cutoff":
        return value.iloc[:31]
    return value.iloc[30:]


def _reader_row(configuration: str, window: str) -> dict:
    value = _window_value(_values(configuration), window)
    metrics = metric_block(value)
    returns = metrics["returns"]
    cash = pd.Series(0.0001, index=returns.index)
    ssr = ssr_inference(returns - cash, window=20, n_boot=8, seed=11)
    meta = LineMetadata(
        portfolio_id=configuration,
        label=configuration,
        window_label=f"{window} {returns.index[0].date()}..{returns.index[-1].date()}",
        currency_basis="legacy_mixed_local_quotes",
        total_return_basis="factor_guard_ablation_equity",
        cash_benchmark_id="BIL@fixture_snapshot",
    )
    return build_reader_metric_row(meta, metrics, cash, ssr, source="fixture:guard_ablation")


def _differential_row(comparison_id: str, window: str) -> dict:
    comparison, reference = {
        "pit_unguarded_minus_guarded": (CONFIGS[1], CONFIGS[0]),
        "nonpit_unguarded_minus_guarded": (CONFIGS[3], CONFIGS[2]),
        "nonpit_unguarded_minus_pit_guarded_combined_stress": (CONFIGS[3], CONFIGS[0]),
    }[comparison_id]
    comparison_returns = metric_block(_window_value(_values(comparison), window))["returns"]
    reference_returns = metric_block(_window_value(_values(reference), window))["returns"]
    spread = comparison_returns - reference_returns
    ssr = ssr_inference(spread, window=20, n_boot=8, seed=13)
    meta = LineMetadata(
        portfolio_id=f"{comparison_id}_ext2026",
        label=comparison_id,
        window_label=f"{window} {spread.index[0].date()}..{spread.index[-1].date()}",
        currency_basis="legacy_mixed_local_quotes",
        total_return_basis="daily_comparison_minus_reference_return",
        cash_benchmark_id="BIL@fixture_snapshot",
    )
    return build_differential_metric_row(
        meta,
        comparison_returns,
        reference_returns,
        ssr,
        source="fixture:guard_ablation",
    )


def _add_file(run_dir: Path, files: dict, role: str, relative: str, data: bytes, **entry) -> None:
    path = run_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    files[role] = {
        "file": relative,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        **entry,
    }


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    import io

    output = io.BytesIO()
    frame.to_parquet(output, index=True)
    return output.getvalue()


def _curve_table() -> pd.DataFrame:
    normalized = {configuration: _values(configuration) / _values(configuration).iloc[0] for configuration in CONFIGS}
    controlled = normalized[CONFIGS[1]] / normalized[CONFIGS[0]]
    stress = normalized[CONFIGS[3]] / normalized[CONFIGS[0]]
    rows = []
    for configuration in CONFIGS:
        curve = normalized[configuration]
        kind, relative = {
            CONFIGS[0]: (None, None),
            CONFIGS[1]: ("controlled_pit_unguarded_vs_guarded", controlled),
            CONFIGS[2]: (None, None),
            CONFIGS[3]: ("combined_nonpit_unguarded_vs_pit_guarded_stress", stress),
        }[configuration]
        drawdown = curve / curve.cummax() - 1.0
        for date, wealth in curve.items():
            rows.append(
                {
                    "date": date,
                    "configuration": configuration,
                    "normalized_wealth": float(wealth),
                    "drawdown": float(drawdown.loc[date]),
                    "relative_wealth": None if relative is None else float(relative.loc[date]),
                    "relative_wealth_kind": kind,
                }
            )
    return pd.DataFrame(rows)


def _completed_guard_ablation_run(tmp_path: Path) -> tuple[Path, dict]:
    run_dir = tmp_path / "factor_guard_ablation_fixture"
    run_dir.mkdir(parents=True)
    files: dict = {}
    records = []
    for configuration in CONFIGS:
        for window in WINDOWS:
            records.append(
                {
                    "record_kind": "reader",
                    "configuration": configuration,
                    "window": window,
                    "record": _reader_row(configuration, window),
                }
            )
    for comparison in COMPARISONS:
        for window in WINDOWS:
            records.append(
                {
                    "record_kind": "differential",
                    "comparison_id": comparison,
                    "window": window,
                    "record": _differential_row(comparison, window),
                }
            )
    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    metric_records = {
        "schema": "factor_guard_ablation.metric_records.v1",
        "records": json_safe(records),
    }
    _add_file(
        run_dir,
        files,
        "metric_records",
        "factor_guard_ablation_metric_records_ext2026.json",
        (json.dumps(metric_records, sort_keys=True, allow_nan=False, default=str) + "\n").encode(),
        schema="factor_guard_ablation.metric_records.v1",
    )

    rebalance_dates = pd.date_range("2019-01-01", periods=90, freq="MS")
    panel_rows = []
    for configuration in CONFIGS:
        guarded = configuration in (CONFIGS[0], CONFIGS[2])
        for date in rebalance_dates:
            panel_rows.append(
                {
                    "rebalance_date": date,
                    "configuration": configuration,
                    "evidence_id": f"{configuration}:{date.date()}",
                    "prompt_mode": "pit" if "nonpit" not in configuration else "nonpit_diagnostic",
                    "guard_enabled": guarded,
                    "p_memorized": 0.25,
                    "raw_view_tilt": 0.08,
                    "applied_view_tilt": 0.06 if guarded else 0.08,
                }
            )
    panel = pd.DataFrame(panel_rows)
    _add_file(
        run_dir,
        files,
        "panel",
        "factor_guard_ablation_panel_ext2026.parquet",
        _parquet_bytes(panel),
        schema="factor_guard_ablation.panel.v1",
        columns=list(panel.columns),
    )
    curve = _curve_table()
    _add_file(
        run_dir,
        files,
        "curves",
        "factor_guard_ablation_curves_ext2026.parquet",
        _parquet_bytes(curve),
        schema="factor_guard_ablation.equity.v1",
    )

    for configuration in (CONFIGS[1], CONFIGS[3]):
        stem = configuration.replace("_ext2026", "")
        _add_file(
            run_dir,
            files,
            f"{stem}_equity",
            f"{stem}_equity_ext2026.parquet",
            _parquet_bytes(_values(configuration).to_frame()),
            schema="factor_guard_ablation.equity_input.v1",
        )
        _add_file(
            run_dir,
            files,
            f"{stem}_targets",
            f"{stem}_targets_ext2026.parquet",
            _parquet_bytes(pd.DataFrame({"target": [0.25]}, index=[pd.Timestamp("2022-01-03")])),
            schema="factor_guard_ablation.targets.v1",
        )
        _add_file(
            run_dir,
            files,
            f"{stem}_decision_log",
            f"{stem}_decision_log_ext2026.json",
            b'{"schema":"factor_guard_ablation.decision_log.v1","records":[]}\n',
            schema="factor_guard_ablation.decision_log.v1",
        )

    manifest = {
        "schema": "factor_guard_ablation_run.v1",
        "completed": True,
        "run_id": run_dir.name,
        "source_commit": "a" * 40,
        "input_manifests": {
            "factor_run": {"run_id": "factor_parent_fixture", "manifest_sha256": "b" * 64},
            "market_snapshot": {"snapshot_id": "market_fixture", "manifest_sha256": "c" * 64},
        },
        "files": files,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (run_dir / "COMPLETED").write_text(f"manifest_sha256={manifest_sha}\n")
    return run_dir, {"run_id": run_dir.name, "manifest_sha256": manifest_sha, "metric_records": metric_records, "panel": panel}


def test_guard_ablation_reports_project_the_exact_four_cell_matrix_and_bundle(tmp_path):
    bts = _report_module()
    run_dir, case = _completed_guard_ablation_run(tmp_path)
    verified = bts.load_completed_guard_ablation_run(run_dir, **{k: case[k] for k in ("run_id", "manifest_sha256")})

    reports = bts.build_guard_ablation_report_tables(verified)
    tear = reports.tables["tear_sheet_factor_guard_ablation_ext2026"]
    assert len(tear) == 21
    assert list(tear["configuration"].iloc[:12]) == [
        configuration for configuration in CONFIGS for _ in WINDOWS
    ]
    assert list(tear["window"].iloc[:12]) == list(WINDOWS) * 4
    assert list(tear["comparison_id"].iloc[12:]) == [
        comparison for comparison in COMPARISONS for _ in WINDOWS
    ]
    assert (tear["record_kind"].iloc[:12] == "reader").all()
    assert (tear["record_kind"].iloc[12:] == "differential").all()
    assert "endpoint_gap" in tear["metric_semantics"].iloc[12]
    assert tear["total_return"].iloc[0] == pytest.approx(case["metric_records"]["records"][0]["record"]["total_return"])

    panel = reports.tables["factor_guard_ablation_panel_ext2026"]
    pd.testing.assert_frame_equal(panel, case["panel"])
    equity = reports.tables["factor_guard_ablation_equity_ext2026"]
    assert list(equity.columns) == list(bts._GUARD_ABLATION_EQUITY_COLUMNS)
    assert len(equity) == 4 * len(_values(CONFIGS[0]))
    assert set(equity[equity["relative_wealth_kind"].notna()]["relative_wealth_kind"]) == {
        "controlled_pit_unguarded_vs_guarded",
        "combined_nonpit_unguarded_vs_pit_guarded_stress",
    }

    destination = tmp_path / "canonical_guard_ablation_reports"
    bundle = bts.materialize_canonical_guard_ablation_report_bundle(verified, destination=destination)
    assert bundle.manifest["schema"] == "canonical_guard_ablation_reports.v1"
    assert set(bundle.tables) == set(bts.GUARD_ABLATION_REPORT_TABLE_SCHEMAS)
    assert not any("data-v4" in str(path) for path in destination.rglob("*"))
    assert (destination / "COMPLETED").read_text().splitlines() == [
        f"manifest_sha256={hashlib.sha256((destination / 'manifest.json').read_bytes()).hexdigest()}"
    ]
    loaded = bts.load_completed_canonical_guard_ablation_report_bundle(destination, guard_input=verified)
    pd.testing.assert_frame_equal(
        loaded.tables["tear_sheet_factor_guard_ablation_ext2026"],
        bts.parquet_safe_report_table(tear),
        check_dtype=False,
    )

    with pytest.raises(ValueError, match="non-empty"):
        bts.materialize_canonical_guard_ablation_report_bundle(verified, destination=destination)


def test_guard_ablation_loader_rejects_tampering_and_out_of_order_records(tmp_path):
    bts = _report_module()
    run_dir, case = _completed_guard_ablation_run(tmp_path)
    kwargs = {k: case[k] for k in ("run_id", "manifest_sha256")}
    verified = bts.load_guard_ablation_report_input(run_dir, **kwargs)

    panel_path = run_dir / "factor_guard_ablation_panel_ext2026.parquet"
    panel_path.write_bytes(panel_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="mutated"):
        bts.load_completed_guard_ablation_run(run_dir, **kwargs)

    # A new signed fixture with the two reader cells swapped still fails the
    # canonical ordering gate before any report table can be projected.
    other_dir, other = _completed_guard_ablation_run(tmp_path / "second")
    metric_path = other_dir / "factor_guard_ablation_metric_records_ext2026.json"
    payload = json.loads(metric_path.read_text())
    payload["records"][0], payload["records"][1] = payload["records"][1], payload["records"][0]
    metric_path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
    manifest_path = other_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["metric_records"]["sha256"] = hashlib.sha256(metric_path.read_bytes()).hexdigest()
    manifest["files"]["metric_records"]["size"] = metric_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    resigned = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (other_dir / "COMPLETED").write_text(f"manifest_sha256={resigned}\n")
    with pytest.raises(ValueError, match="exact ordered"):
        bts.load_completed_guard_ablation_run(other_dir, run_id=other["run_id"], manifest_sha256=resigned)

    # The first verified object remains immutable evidence; no local table build
    # can consume it after its on-disk manifest-owned panel was tampered with.
    with pytest.raises(ValueError, match="mutated"):
        bts.materialize_canonical_guard_ablation_report_bundle(verified, destination=tmp_path / "after_tamper")
