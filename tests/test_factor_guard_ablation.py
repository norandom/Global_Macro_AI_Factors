"""Fast contracts for the offline recall-guard ablation producer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _producer():
    from scripts import build_factor_guard_ablation as producer

    return producer


def _panel(producer, *, null_parse_cell: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2019-01-01", periods=90, freq="MS")
    rows: list[dict] = []
    for configuration in producer.CONFIGS:
        guarded = configuration in {
            "factor_pit_ext2026",
            "factor_nonpit_diagnostic_ext2026",
        }
        prompt_mode = "pit" if "nonpit" not in configuration else "nonpit_diagnostic"
        for position, rebalance_date in enumerate(dates):
            parse_ok = not (null_parse_cell and configuration == producer.CONFIGS[0] and position == 0)
            p_memorized = None if not parse_ok else 0.25
            raw = 0.8 if parse_ok else 0.0
            applied = raw * (1.0 - p_memorized) if guarded and p_memorized is not None else raw
            row = {
                "rebalance_date": rebalance_date,
                "macro_source_date": rebalance_date - pd.Timedelta(days=1),
                "configuration": configuration,
                "evidence_id": f"{configuration}-{position}",
                "prompt_mode": prompt_mode,
                "guard_enabled": guarded,
                "p_memorized": p_memorized,
                "parse_ok": parse_ok,
                "steered": parse_ok,
                "conviction": 0.5 if parse_ok else None,
                "cpi_yoy_z": 0.1,
                "t10y2y_z": -0.2,
                "hy_oas_z": 0.3,
                "macro_state_norm": float(np.sqrt(0.14)),
                "loading_inflation": 0.1 if parse_ok else None,
                "loading_growth": -0.2 if parse_ok else None,
                "loading_credit_stress": 0.3 if parse_ok else None,
                "loading_policy": 0.4 if parse_ok else None,
                "loading_risk_appetite": -0.5 if parse_ok else None,
                "raw_view_tilt": raw,
                "applied_view_tilt": applied,
                "expected_attenuation": (1.0 - p_memorized) if guarded and p_memorized is not None else (1.0 if parse_ok else None),
                "observed_attenuation": applied / raw if raw else None,
                "relation_error": applied - raw * (1.0 - p_memorized) if guarded and p_memorized is not None else 0.0 if parse_ok else None,
                "allocation_status": "bl_blend_applied" if parse_ok else "fallback",
                "bl_fallback_reason": None if parse_ok else "no_valid_views",
                "target_effective_date": rebalance_date,
                "target_reconstruction_error": 0.0,
            }
            per_asset_raw = raw / 4.0
            per_asset_applied = applied / 4.0
            for asset_key in producer.PANEL_ASSET_KEYS.values():
                row[f"raw_tilt_{asset_key}"] = per_asset_raw
                row[f"applied_tilt_{asset_key}"] = per_asset_applied
                row[f"bl_q_{asset_key}"] = per_asset_applied * 0.5 / 252.0
                row[f"hrp_base_weight_{asset_key}"] = 0.25
                row[f"bl_weight_{asset_key}"] = 0.25
                row[f"target_weight_{asset_key}"] = 0.25
                row[f"target_delta_{asset_key}"] = 0.0
            rows.append(row)
    return pd.DataFrame(rows, columns=list(producer.PANEL_COLUMNS))


def test_mechanism_panel_is_exact_four_by_ninety_and_preserves_parse_null() -> None:
    producer = _producer()
    panel = _panel(producer, null_parse_cell=True)

    producer.validate_mechanism_panel(panel)

    assert len(panel) == 360
    first = panel.iloc[0]
    assert not bool(first["parse_ok"])
    assert pd.isna(first["p_memorized"])
    assert panel.groupby("configuration", sort=False).size().to_dict() == {
        configuration: 90 for configuration in producer.CONFIGS
    }


def test_mechanism_panel_rejects_noncanonical_configuration_order() -> None:
    producer = _producer()
    panel = _panel(producer)
    panel.iloc[:90, panel.columns.get_loc("configuration")] = producer.CONFIGS[1]

    with pytest.raises(ValueError, match="canonical order"):
        producer.validate_mechanism_panel(panel)


def test_curve_table_keeps_controlled_and_combined_estimands_distinct() -> None:
    producer = _producer()
    dates = pd.bdate_range("2024-01-02", periods=8)
    values = {
        configuration: pd.Series(
            100.0 * np.cumprod(np.r_[1.0, np.repeat(1.001 + index * 0.0002, len(dates) - 1)]),
            index=dates,
            name="value",
        )
        for index, configuration in enumerate(producer.CONFIGS)
    }

    curves = producer.build_curve_table(values)

    assert tuple(curves.columns) == producer.CURVE_COLUMNS
    assert len(curves) == 4 * len(dates)
    assert set(curves.loc[curves["relative_wealth_kind"].notna(), "relative_wealth_kind"]) == {
        "controlled_pit_unguarded_vs_guarded",
        "combined_nonpit_unguarded_vs_pit_guarded_stress",
    }
    assert (curves.groupby("configuration", sort=False)["normalized_wealth"].first() == 1.0).all()


def test_cli_defaults_are_tracked_offline_inputs() -> None:
    producer = _producer()
    args = producer._parser().parse_args([])

    assert args.output_dir == producer.OUTPUT_DEFAULT
    assert args.parent_run_dir == producer.PARENT_RUN_DEFAULT
    assert args.snapshot_dir == producer.SNAPSHOT_DEFAULT
    assert args.macro_panel == producer.MACRO_PANEL_DEFAULT
    assert "factor_guard_ablation_runs" in str(args.output_dir)


def test_tracked_parent_input_loads_without_network_when_available() -> None:
    producer = _producer()
    if not producer.PARENT_RUN_DEFAULT.exists() or not producer.SNAPSHOT_DEFAULT.exists():
        pytest.skip("tracked provisional Factor parent/snapshot are unavailable")

    parent = producer.load_parent_inputs()

    assert parent.manifest["run_id"] == "factor_ext2026_2019-01-01_2026-06-30_v1"
    assert len(parent.evidence) == 180
    assert len(parent.rebalance_dates) == 90
    assert parent.snapshot_manifest["snapshot_id"] == "provisional_market_total_return_fx_2026-06-30_v1"


def test_tracked_run_loads_through_current_guard_ablation_report_loader_when_available() -> None:
    producer = _producer()
    run_dir = producer.OUTPUT_DEFAULT
    if not (run_dir / "COMPLETED").is_file():
        pytest.skip("tracked guard-ablation production run has not been materialized")

    import hashlib
    from scripts import build_tear_sheet as report_producer

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_commit"] == manifest["producer_source"]["git_head"]
    assert manifest["parent_source_commit"]
    assert manifest["config"]["macro_panel"] == "data/macro_panel_monthly.parquet"
    assert "/home/" not in (run_dir / "manifest.json").read_text(encoding="utf-8")

    loaded = report_producer.load_completed_guard_ablation_run(
        run_dir,
        run_id=run_dir.name,
        manifest_sha256=hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest(),
    )
    tables = report_producer.build_guard_ablation_report_tables(loaded)

    assert tables.tables["tear_sheet_factor_guard_ablation_ext2026"].shape[0] == 21
    assert tables.tables["factor_guard_ablation_panel_ext2026"].shape[0] == 360
