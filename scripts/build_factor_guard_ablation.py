"""Build the offline four-cell recall-guard ablation diagnostic.

The producer consumes a completed Factor replay and completed market snapshot.  It
never calls a model provider: dated parent evidence is replayed through the same
walk-forward architecture as ``extend_stream_2026`` with only
``RecallGuardedConfig(enabled=False)`` changed for the two diagnostic lines.

The output is deliberately a standalone, append-only producer bundle consumed by
``scripts.build_tear_sheet``'s isolated guard-ablation report loader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

import macro_framework as mf
from macro_framework import factor_scoring as fs
from macro_framework.evaluation import metric_block
from macro_framework.returns import daily_returns
from macro_framework.ssr import ssr_inference
from scripts import extend_stream_2026 as ext


RUN_SCHEMA = "factor_guard_ablation_run.v1"
METRIC_SCHEMA = "factor_guard_ablation.metric_records.v1"
PANEL_SCHEMA = "factor_guard_ablation.panel.v1"
EQUITY_SCHEMA = "factor_guard_ablation.equity.v1"
DECISION_SCHEMA = "factor_guard_ablation.decision_log.v1"
AUDIT_SCHEMA = "factor_guard_ablation.replay_audit.v1"

RUN_ID = "factor_guard_ablation_ext2026_2019-01-01_2026-06-30_v2"
PARENT_RUN_DEFAULT = (
    REPO
    / "data/provisional_remediation/factor_runs/factor_ext2026_2019-01-01_2026-06-30_v1"
)
SNAPSHOT_DEFAULT = (
    REPO
    / "data/provisional_remediation/market_snapshots/"
    "provisional_market_total_return_fx_2026-06-30_v1"
)
MACRO_PANEL_DEFAULT = REPO / "data/macro_panel_monthly.parquet"
OUTPUT_DEFAULT = REPO / "data/provisional_remediation/factor_guard_ablation_runs" / RUN_ID

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
VARIANTS = ("pit", "nonpit_diagnostic")
CUTOFF = pd.Timestamp("2024-06-01")
PANEL_ASSET_KEYS = {
    "SWDA.L": "SWDA_L",
    "XLK": "XLK",
    "IAU": "IAU",
    "BIL": "BIL",
}
PANEL_ASSET_FIELDS = (
    "raw_tilt",
    "applied_tilt",
    "bl_q",
    "hrp_base_weight",
    "bl_weight",
    "target_weight",
    "target_delta",
)
PANEL_COLUMNS = (
    "rebalance_date",
    "macro_source_date",
    "configuration",
    "evidence_id",
    "prompt_mode",
    "guard_enabled",
    "p_memorized",
    "parse_ok",
    "steered",
    "conviction",
    "cpi_yoy_z",
    "t10y2y_z",
    "hy_oas_z",
    "macro_state_norm",
    "loading_inflation",
    "loading_growth",
    "loading_credit_stress",
    "loading_policy",
    "loading_risk_appetite",
    "raw_view_tilt",
    "applied_view_tilt",
    "expected_attenuation",
    "observed_attenuation",
    "relation_error",
    "allocation_status",
    "bl_fallback_reason",
    "target_effective_date",
    "target_reconstruction_error",
    *tuple(
        f"{field}_{asset_key}"
        for field in PANEL_ASSET_FIELDS
        for asset_key in PANEL_ASSET_KEYS.values()
    ),
)
CURVE_COLUMNS = (
    "date",
    "configuration",
    "normalized_wealth",
    "drawdown",
    "relative_wealth",
    "relative_wealth_kind",
)

_UNGUARDED_ARTIFACTS = {
    "targets_pit_unguarded": "factor_pit_unguarded_diagnostic_targets_ext2026.parquet",
    "equity_pit_unguarded": "factor_pit_unguarded_diagnostic_equity_ext2026.parquet",
    "decision_log_pit_unguarded": "factor_pit_unguarded_diagnostic_decision_log_ext2026.json",
    "targets_nonpit_unguarded": "factor_nonpit_unguarded_diagnostic_targets_ext2026.parquet",
    "equity_nonpit_unguarded": "factor_nonpit_unguarded_diagnostic_equity_ext2026.parquet",
    "decision_log_nonpit_unguarded": "factor_nonpit_unguarded_diagnostic_decision_log_ext2026.json",
}


@dataclass(frozen=True)
class ParentInputs:
    """Verified, completed parent Factor and snapshot inputs."""

    run_dir: Path
    manifest: Mapping[str, object]
    manifest_sha256: str
    snapshot_dir: Path
    snapshot_manifest: Mapping[str, object]
    snapshot_sha256: str
    evidence: Mapping[ext.EvidenceKey, ext.DatedFactorEvidence]
    evidence_frame: pd.DataFrame
    symbols: tuple[str, ...]
    prices: pd.DataFrame
    price_input: Mapping[str, object]
    panel: pd.DataFrame
    rebalance_dates: pd.DatetimeIndex


@dataclass(frozen=True)
class VariantOutput:
    """One unguarded replay line and its per-date decisions."""

    configuration: str
    variant: str
    targets: pd.DataFrame
    value: pd.Series
    decision_log: Mapping[str, object]
    decisions: Mapping[pd.Timestamp, fs.FactorDecision]


def sha256_file(path: Path | str) -> str:
    """Return the byte SHA-256 for one file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _repo_relative_path(path: Path | str, *, label: str) -> str:
    """Return a portable repository-relative path for manifest lineage."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} must remain inside repository root {REPO}: {resolved}") from exc


def _json_value(value: Any) -> Any:
    """Convert pandas/numpy values into strict deterministic JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_json(payload: object) -> str:
    return json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"


def _require_new_destination(destination: Path) -> None:
    if destination.exists():
        if (destination / "COMPLETED").exists():
            raise ValueError(f"guard-ablation run {destination.name!r} is COMPLETED and immutable")
        if any(destination.iterdir()):
            raise ValueError(f"refusing to write into non-empty staging directory {destination}")


def _load_snapshot_manifest(snapshot_dir: Path) -> tuple[dict[str, object], str]:
    """Validate completed snapshot and return its immutable identity."""
    # This calls the established snapshot completion/hash validator; it is pure
    # local I/O and has no provider path.
    ext._load_completed_snapshot_tables(snapshot_dir)
    manifest_path = snapshot_dir / "manifest.json"
    marker_path = snapshot_dir / "COMPLETED"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha256 = sha256_file(manifest_path)
    if manifest.get("completed") is not True:
        raise ValueError(f"{snapshot_dir}: snapshot manifest is not completed")
    marker_lines = marker_path.read_text().splitlines()
    if not marker_lines or marker_lines[-1] != f"manifest_sha256={manifest_sha256}":
        raise ValueError(f"{snapshot_dir}: snapshot COMPLETED marker is stale")
    return manifest, manifest_sha256


def _load_macro_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_parquet(path).copy()
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise ValueError(f"{path}: macro panel must have a DatetimeIndex")
    panel.index = pd.DatetimeIndex(panel.index)
    if panel.empty or panel.index.tz is not None or panel.index.has_duplicates or not panel.index.is_monotonic_increasing:
        raise ValueError(f"{path}: macro panel index must be non-empty, unique, ordered, and timezone-naive")
    required = set(ext.PANEL_Z_COLS)
    if not required.issubset(panel.columns):
        raise ValueError(f"{path}: macro panel is missing required z-score columns {sorted(required - set(panel.columns))}")
    return panel


def load_parent_inputs(
    *,
    parent_run_dir: Path | str = PARENT_RUN_DEFAULT,
    snapshot_dir: Path | str = SNAPSHOT_DEFAULT,
    macro_panel_path: Path | str = MACRO_PANEL_DEFAULT,
) -> ParentInputs:
    """Load validated parent artifacts and reconstruct the dated evidence map.

    Parent validation is intentionally delegated to the existing Factor producer.
    The evidence map is reconstructed from its manifest-inventoried Parquet rather
    than from a loose export, so no network/model provider is reachable here.
    """
    parent_run_dir = Path(parent_run_dir)
    snapshot_dir = Path(snapshot_dir)
    macro_panel_path = Path(macro_panel_path)
    ext.load_completed_factor_run(parent_run_dir)
    manifest = json.loads((parent_run_dir / "manifest.json").read_text())
    parent_sha = sha256_file(parent_run_dir / "manifest.json")
    parent_snapshot = manifest["input_manifests"]["market_snapshot"]
    snapshot_manifest, snapshot_sha = _load_snapshot_manifest(snapshot_dir)
    if (
        parent_snapshot.get("snapshot_id"),
        parent_snapshot.get("manifest_sha256"),
    ) != (snapshot_manifest.get("snapshot_id"), snapshot_sha):
        raise ValueError("completed snapshot identity does not match the parent Factor run")

    evidence_entry = manifest["files"]["evidence"]
    evidence_path = parent_run_dir / str(evidence_entry["file"])
    if sha256_file(evidence_path) != evidence_entry["sha256"]:
        raise ValueError("parent factor evidence bytes were mutated after inventory")
    evidence_frame = pd.read_parquet(evidence_path)
    records = ext._factor_evidence_records_from_frame(evidence_frame)
    expected_dates = [date.fromisoformat(value) for value in manifest["expected_evidence"]["dates"]]
    expected_keys = [
        (variant, rebalance_date)
        for variant in manifest["expected_evidence"]["variants"]
        for rebalance_date in expected_dates
    ]
    evidence = ext.validate_evidence_records(records, expected_keys)

    symbols = tuple(pd.read_parquet(REPO / "data/portfolio_ssr_top_per_category.parquet")["symbol"].tolist())
    if symbols != tuple(PANEL_ASSET_KEYS):
        raise ValueError(f"Factor universe must equal the declared ordered panel assets: {tuple(PANEL_ASSET_KEYS)}")
    prices, price_input = ext.load_completed_snapshot_price_frame(
        snapshot_dir,
        list(symbols),
        start=manifest["config"]["price_window_start"],
        end=manifest["config"]["price_window_end"],
    )
    if (
        price_input["snapshot_id"],
        price_input["manifest_sha256"],
    ) != (snapshot_manifest["snapshot_id"], snapshot_sha):
        raise ValueError("rebuilt price input does not bind the supplied completed snapshot")
    expected_price_columns = [*PANEL_ASSET_KEYS, "SPY"]
    if list(prices.columns) != expected_price_columns:
        raise ValueError(f"snapshot price columns must equal {expected_price_columns}")
    expected_price_hash = "e72d89a52ae305b8f171c3e20abc21ebbcc90ed9c08f40c9461f1e213818776f"
    if price_input.get("price_frame", {}).get("content_sha256") != expected_price_hash:
        raise ValueError("snapshot price frame does not match the parent-approved ext2026 input")
    panel = _load_macro_panel(macro_panel_path)
    rebalance_dates = pd.DatetimeIndex(sorted({pd.Timestamp(key[1]) for key in evidence}))
    if len(rebalance_dates) != len(expected_dates):
        raise ValueError("parent evidence rebalance calendar is malformed")
    if rebalance_dates.tz is not None or rebalance_dates.has_duplicates or not rebalance_dates.is_monotonic_increasing:
        raise ValueError("parent evidence rebalance calendar must be unique, ordered, and timezone-naive")
    return ParentInputs(
        run_dir=parent_run_dir,
        manifest=manifest,
        manifest_sha256=parent_sha,
        snapshot_dir=snapshot_dir,
        snapshot_manifest=snapshot_manifest,
        snapshot_sha256=snapshot_sha,
        evidence=evidence,
        evidence_frame=evidence_frame,
        symbols=symbols,
        prices=prices,
        price_input=price_input,
        panel=panel,
        rebalance_dates=rebalance_dates,
    )


def _factor_meta(parent: ParentInputs) -> dict[pd.Timestamp, tuple[dict[str, float], dict[str, float], list[dict[str, object]]]]:
    """Recreate the parent point-in-time macro inputs for every evidence date."""
    asset_map = mf.AssetMap.default()
    asset_snapshot = [
        {"id": pseudo, "category": category}
        for pseudo, category in sorted(asset_map.categories.items())
    ]
    available = parent.panel.dropna(subset=ext.PANEL_Z_COLS)
    meta: dict[pd.Timestamp, tuple[dict[str, float], dict[str, float], list[dict[str, object]]]] = {}
    for rebalance_date in parent.rebalance_dates:
        history = available.loc[available.index < rebalance_date]
        if history.empty:
            raise ValueError(f"no point-in-time macro row exists before {rebalance_date.date()}")
        row = history.iloc[-1]
        macro_state = {column: float(row[column]) for column in ext.PANEL_Z_COLS}
        raw_levels = {
            column: float(row[column])
            for column in ext.PANEL_RAW_COLS
            if column in row and pd.notna(row[column])
        }
        meta[rebalance_date] = (macro_state, raw_levels, asset_snapshot)
    return meta


def _combine_factory(tilt: float):
    """Return the parent HRP-CVaR + BL blend, with no local finance formulas."""
    def combine(ctx: dict, P: pd.DataFrame | None, Q: pd.DataFrame | None) -> pd.Series:
        returns_hist = ctx["returns"]
        w_hrp = mf.hrp_cvar_weights_with_fixed(
            returns_hist,
            {"BIL": ext._regime_cash_pin(returns_hist, None)},
        )
        if P is None:
            return w_hrp
        try:
            w_bl = mf.bl_mv_weights(returns_hist, prior_weights=w_hrp, P=P, Q=Q, obj="Utility")
        except Exception:  # BL may be unavailable for a degenerate lookback.
            return w_hrp
        blended = (1.0 - tilt) * w_hrp + tilt * w_bl
        return blended / blended.sum()
    return combine


def _decision_for(
    *,
    evidence: Mapping[ext.EvidenceKey, ext.DatedFactorEvidence],
    variant: str,
    rebalance_date: pd.Timestamp,
    meta: tuple[dict[str, float], dict[str, float], list[dict[str, object]]],
    agent: mf.LlmMacroAgent,
    symbols: Sequence[str],
    guard_enabled: bool,
) -> fs.FactorDecision:
    record = ext.resolve_dated_evidence(evidence, variant, rebalance_date)
    generate, scorer = ext.dated_replay_closures(record)
    macro_state, raw_levels, asset_snapshot = meta
    return fs.factor_rebalance(
        generate_loadings=generate,
        scorer=scorer,
        agent=agent,
        macro_state=macro_state,
        asset_snapshot=asset_snapshot,
        real_symbols=list(symbols),
        as_of=rebalance_date,
        raw_levels=raw_levels,
        recall_guarded_config=fs.RecallGuardedConfig(enabled=guard_enabled),
    )


def _decision_payload(decisions: Mapping[pd.Timestamp, fs.FactorDecision], *, configuration: str, parent: ParentInputs) -> dict[str, object]:
    """Persist an inspectable unguarded decision log without changing p-null semantics."""
    log: dict[str, dict[str, object]] = {
        "p_memorized": {},
        "parse_ok": {},
        "steered": {},
        "loadings": {},
        "views": {},
        "raw_view_tilt": {},
        "applied_view_tilt": {},
    }
    for stamp, decision in decisions.items():
        key = stamp.isoformat()
        raw_tilt = float(sum(abs(view.expected_excess_annualized) for view in decision.views))
        log["p_memorized"][key] = decision.p_memorized
        log["parse_ok"][key] = bool(decision.parse_ok)
        log["steered"][key] = bool(decision.steered)
        log["loadings"][key] = (
            dict(decision.loadings.loadings) if decision.loadings is not None else None
        )
        log["views"][key] = [view.to_dict() for view in decision.views]
        log["raw_view_tilt"][key] = raw_tilt
        log["applied_view_tilt"][key] = raw_tilt
    return {
        "schema": DECISION_SCHEMA,
        "meta": {
            "configuration": configuration,
            "guard_enabled": False,
            "parent_factor_run": parent.manifest["run_id"],
            "n_rebalances": len(decisions),
            "window": f"{parent.manifest['config']['sim_start']}..{parent.manifest['config']['sim_end']}",
            "model": dict(parent.manifest["model"]),
        },
        **log,
    }


def replay_unguarded_variant(
    parent: ParentInputs,
    *,
    variant: str,
    configuration: str,
    n_boot: int | None = None,
) -> VariantOutput:
    """Replay one parent evidence stream with the recall attenuation disabled."""
    if variant not in VARIANTS:
        raise ValueError(f"unsupported factor variant {variant!r}")
    if configuration not in CONFIGS:
        raise ValueError(f"unsupported guard-ablation configuration {configuration!r}")
    config = parent.manifest["config"]
    meta_by_date = _factor_meta(parent)
    agent = mf.LlmMacroAgent(asset_map=mf.AssetMap.default())
    consumption: dict[ext.EvidenceKey, dict[str, object]] = {}
    failures: list[ext.ReplayValidationError] = []
    combine = _combine_factory(float(config["tilt"]))

    def build_inputs(ctx: dict):
        rebalance_date = pd.Timestamp(ctx["rebalance_date"])
        macro_state, raw_levels, asset_snapshot = meta_by_date[rebalance_date]
        return macro_state, asset_snapshot, rebalance_date, raw_levels

    weight_fn = ext.make_dated_replay_weight_fn(
        variant=variant,
        evidence=parent.evidence,
        agent=agent,
        build_inputs=build_inputs,
        combine=combine,
        failures=failures,
        consumed=consumption,
        recall_guarded_config=fs.RecallGuardedConfig(enabled=False),
    )
    targets = mf.build_walk_forward_targets(
        parent.prices[list(parent.symbols)],
        rebalance_dates=parent.rebalance_dates,
        weight_fns={configuration: weight_fn},
        macro_panel=parent.panel,
        lookback_days=int(config["lookback_days"]),
    )[configuration]
    if failures:
        raise failures[0]
    portfolio = mf.run_rebalance_sim(
        parent.prices[list(parent.symbols)], targets, init_cash=float(config["init_cash"])
    )
    value = portfolio.value().rename("value")
    targets = targets.reindex(value.index)

    decisions: dict[pd.Timestamp, fs.FactorDecision] = {}
    for rebalance_date in parent.rebalance_dates:
        record = ext.resolve_dated_evidence(parent.evidence, variant, rebalance_date)
        ext.record_consumption(consumption, (variant, rebalance_date.date()), record)
        decision = _decision_for(
            evidence=parent.evidence,
            variant=variant,
            rebalance_date=rebalance_date,
            meta=meta_by_date[rebalance_date],
            agent=agent,
            symbols=parent.symbols,
            guard_enabled=False,
        )
        ext.record_decision_identity(consumption, (variant, rebalance_date.date()), record, decision)
        decisions[rebalance_date] = decision
    expected = [(variant, stamp.date()) for stamp in parent.rebalance_dates]
    ext.validate_source_to_consumption(parent.evidence, consumption, expected)
    return VariantOutput(
        configuration=configuration,
        variant=variant,
        targets=targets,
        value=value,
        decision_log=_decision_payload(decisions, configuration=configuration, parent=parent),
        decisions=decisions,
    )


def _parent_equity(parent: ParentInputs, role: str) -> pd.Series:
    entry = parent.manifest["files"][role]
    path = parent.run_dir / entry["file"]
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(f"parent Factor {role} equity was mutated after inventory")
    frame = pd.read_parquet(path)
    if "value" not in frame:
        raise ValueError(f"parent Factor {role} equity has no value column")
    value = frame["value"].rename("value")
    value.index = pd.DatetimeIndex(value.index)
    return value


def _performance_value(parent: ParentInputs, value: pd.Series) -> pd.Series:
    """Project a value stream onto the completed parent's declared performance calendar.

    A floating-point no-op in a fresh rebalance simulation can make a raw
    ``value.ne(value.iloc[0])`` trim one session earlier than the completed
    parent.  The ablation must instead share the parent's manifest-owned return
    calendar exactly, so every cell uses its declared first return and preceding
    anchor rather than inferring a new start from numerical noise.
    """
    stream = parent.manifest["files"]["metric_records"]
    metric_path = parent.run_dir / str(stream["file"])
    metric_bundle = json.loads(metric_path.read_text())
    declared = metric_bundle["source_streams"]["factor_pit_ext2026"]
    first_return = pd.Timestamp(declared["start"])
    last_return = pd.Timestamp(declared["end"])
    value = value.copy()
    value.index = pd.DatetimeIndex(value.index)
    if first_return not in value.index or last_return not in value.index:
        raise ValueError("ablation equity stream does not cover the parent performance window")
    first_position = value.index.get_loc(first_return)
    last_position = value.index.get_loc(last_return)
    if not isinstance(first_position, int) or not isinstance(last_position, int) or first_position < 1:
        raise ValueError("parent performance window lacks a preceding anchor")
    return value.iloc[first_position - 1 : last_position + 1]


def _window_value(parent: ParentInputs, value: pd.Series, window: str) -> pd.Series:
    """Select one explicit shared performance window while retaining its anchor."""
    performance = _performance_value(parent, value)
    returns = metric_block(performance)["returns"]
    if window == "full":
        selected = returns
    elif window == "pre_cutoff":
        selected = returns.loc[returns.index <= CUTOFF]
    elif window == "post_cutoff":
        selected = returns.loc[returns.index > CUTOFF]
    else:
        raise ValueError(f"unknown guard-ablation reporting window {window!r}")
    if selected.empty:
        raise ValueError(f"{window}: no portfolio returns in the requested reporting window")
    first_position = performance.index.get_loc(selected.index[0])
    last_position = performance.index.get_loc(selected.index[-1])
    if not isinstance(first_position, int) or not isinstance(last_position, int) or first_position < 1:
        raise ValueError("selected return window lacks an explicit prior value anchor")
    return performance.iloc[first_position - 1 : last_position + 1]


def _line_metadata(configuration: str, returns: pd.Series, *, snapshot_id: str) -> mf.LineMetadata:
    return mf.LineMetadata(
        portfolio_id=configuration,
        label=configuration,
        window_label=f"{returns.index[0].date()}..{returns.index[-1].date()}",
        currency_basis="legacy_mixed_local_quotes",
        total_return_basis="factor_guard_ablation_equity",
        cash_benchmark_id=f"BIL@{snapshot_id}",
    )


def _reader_record(
    *,
    parent: ParentInputs,
    configuration: str,
    value: pd.Series,
    window: str,
    ssr_settings: Mapping[str, object],
) -> dict[str, object]:
    selected = _window_value(parent, value, window)
    metrics = metric_block(selected)
    returns = metrics["returns"]
    cash, _cash_lineage = ext.load_completed_snapshot_bil_returns(
        parent.snapshot_dir, returns.index, anchor=selected.index[0]
    )
    ssr = ssr_inference(returns - cash, **ssr_settings)
    return mf.build_reader_metric_row(
        _line_metadata(configuration, returns, snapshot_id=str(parent.snapshot_manifest["snapshot_id"])),
        metrics,
        cash,
        ssr,
        source=f"scripts/build_factor_guard_ablation.py:{configuration}:{window}",
    )


def _differential_record(
    *,
    parent: ParentInputs,
    comparison_id: str,
    comparison_value: pd.Series,
    reference_value: pd.Series,
    window: str,
    ssr_settings: Mapping[str, object],
) -> dict[str, object]:
    comparison_returns = metric_block(_window_value(parent, comparison_value, window))["returns"]
    reference_returns = metric_block(_window_value(parent, reference_value, window))["returns"]
    if not comparison_returns.index.equals(reference_returns.index):
        raise ValueError(f"{comparison_id}/{window}: comparison and reference returns have different calendars")
    spread = mf.differential_returns(comparison_returns, reference_returns)
    ssr = ssr_inference(spread, **ssr_settings)
    metadata = mf.LineMetadata(
        portfolio_id=f"{comparison_id}_ext2026",
        label=comparison_id,
        window_label=f"{spread.index[0].date()}..{spread.index[-1].date()}",
        currency_basis="legacy_mixed_local_quotes",
        total_return_basis="daily_comparison_minus_reference_return",
        cash_benchmark_id="not_applicable_direct_daily_spread",
    )
    return mf.build_differential_metric_row(
        metadata,
        comparison_returns,
        reference_returns,
        ssr,
        source=f"scripts/build_factor_guard_ablation.py:{comparison_id}:{window}",
    )


def build_metric_records(
    parent: ParentInputs,
    *,
    values: Mapping[str, pd.Series],
    n_boot: int = 1000,
) -> dict[str, object]:
    """Build the exact 12 reader + 9 differential report-record matrix."""
    if set(values) != set(CONFIGS):
        raise ValueError("metric records require exactly the four configured equity streams")
    ssr_settings: dict[str, object] = {
        "window": 252,
        "sr_star": 0.0,
        "n_boot": int(n_boot),
        "seed": 0,
        "alpha": 0.05,
    }
    records: list[dict[str, object]] = []
    for configuration in CONFIGS:
        for window in WINDOWS:
            records.append(
                {
                    "record_kind": "reader",
                    "configuration": configuration,
                    "window": window,
                    "record": _reader_record(
                        parent=parent,
                        configuration=configuration,
                        value=values[configuration],
                        window=window,
                        ssr_settings=ssr_settings,
                    ),
                }
            )
    pairs = {
        "pit_unguarded_minus_guarded": (
            "factor_pit_unguarded_diagnostic_ext2026",
            "factor_pit_ext2026",
        ),
        "nonpit_unguarded_minus_guarded": (
            "factor_nonpit_unguarded_diagnostic_ext2026",
            "factor_nonpit_diagnostic_ext2026",
        ),
        "nonpit_unguarded_minus_pit_guarded_combined_stress": (
            "factor_nonpit_unguarded_diagnostic_ext2026",
            "factor_pit_ext2026",
        ),
    }
    for comparison_id in COMPARISONS:
        comparison, reference = pairs[comparison_id]
        for window in WINDOWS:
            records.append(
                {
                    "record_kind": "differential",
                    "comparison_id": comparison_id,
                    "window": window,
                    "record": _differential_record(
                        parent=parent,
                        comparison_id=comparison_id,
                        comparison_value=values[comparison],
                        reference_value=values[reference],
                        window=window,
                        ssr_settings=ssr_settings,
                    ),
                }
            )
    return {
        "schema": METRIC_SCHEMA,
        "ssr_settings": ssr_settings,
        "records": records,
    }


def _tilt_magnitude(decision: fs.FactorDecision) -> float:
    return float(sum(abs(view.expected_excess_annualized) for view in decision.views))


def _parent_targets(parent: ParentInputs, role: str) -> pd.DataFrame:
    entry = parent.manifest["files"][role]
    path = parent.run_dir / str(entry["file"])
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(f"parent Factor {role} targets were mutated after inventory")
    targets = pd.read_parquet(path).copy()
    targets.index = pd.DatetimeIndex(targets.index)
    if list(targets.columns) != list(parent.symbols):
        raise ValueError(f"parent Factor {role} target columns diverge from the asset universe")
    return targets


def _view_values(
    decision: fs.FactorDecision,
    *,
    asset_map: mf.AssetMap,
) -> tuple[dict[str, float], dict[str, float]]:
    tilts = {symbol: 0.0 for symbol in PANEL_ASSET_KEYS}
    bl_q = {symbol: 0.0 for symbol in PANEL_ASSET_KEYS}
    for position, view in enumerate(decision.views):
        symbol = asset_map.pseudo_to_real.get(view.asset_long)
        if symbol not in tilts:
            continue
        tilts[symbol] = float(view.expected_excess_annualized)
        if decision.Q is not None and position < len(decision.Q):
            bl_q[symbol] = float(decision.Q.iloc[position, 0])
    return tilts, bl_q


def _hrp_base_by_date(parent: ParentInputs) -> dict[pd.Timestamp, pd.Series]:
    all_returns = daily_returns(parent.prices[list(parent.symbols)])
    lookback_days = int(parent.manifest["config"]["lookback_days"])
    base_by_date: dict[pd.Timestamp, pd.Series] = {}
    for rebalance_date in parent.rebalance_dates:
        returns_hist = all_returns.loc[all_returns.index < rebalance_date]
        returns_hist = returns_hist.tail(lookback_days).dropna(how="any")
        if len(returns_hist) < 60:
            raise ValueError(f"{rebalance_date.date()}: insufficient return history for allocation diagnostics")
        base_by_date[rebalance_date] = mf.hrp_cvar_weights_with_fixed(
            returns_hist,
            {"BIL": ext._regime_cash_pin(returns_hist, None)},
        ).reindex(parent.symbols)
    return base_by_date


def _allocation_diagnostics(
    parent: ParentInputs,
    *,
    rebalance_date: pd.Timestamp,
    decision: fs.FactorDecision,
    base: pd.Series,
    target: pd.Series,
) -> dict[str, object]:
    target = pd.to_numeric(target.reindex(parent.symbols), errors="raise")
    if target.isna().any():
        raise ValueError(f"{rebalance_date.date()}: target row is incomplete")
    tilt = float(parent.manifest["config"]["tilt"])
    fallback_reason: str | None = None
    if decision.P is None or decision.Q is None:
        bl_weight = base.copy()
        final = base.copy()
        fallback_reason = "no_valid_views"
    else:
        # The target was already produced through the unchanged BL optimizer.  The
        # blend identity therefore gives the exact producer-owned BL allocation
        # without rerunning an expensive numerical optimizer during diagnostics.
        bl_weight = (target - (1.0 - tilt) * base) / tilt
        final = (1.0 - tilt) * base + tilt * bl_weight
    reconstruction_error = float((final - target).abs().max())
    if reconstruction_error > 1e-10:
        raise ValueError(
            f"{rebalance_date.date()}: persisted target diverges from reconstructed decision path "
            f"by {reconstruction_error:.3g}"
        )
    return {
        "base": base,
        "bl_weight": bl_weight,
        "target": target,
        "target_delta": target - base,
        "allocation_status": "fallback" if fallback_reason else "bl_blend_applied",
        "bl_fallback_reason": fallback_reason,
        "target_reconstruction_error": reconstruction_error,
    }


def build_mechanism_panel(
    parent: ParentInputs,
    *,
    pit_unguarded: VariantOutput,
    nonpit_unguarded: VariantOutput,
) -> pd.DataFrame:
    """Build the sealed 360-row wide mechanism panel in report-loader order.

    Each rebalance/configuration row preserves the evidence and guard identity,
    point-in-time macro source row, all five parsed loadings, per-asset raw and
    applied views, BL ``Q``, HRP base and BL weights, final targets, and target
    deltas.  The wide shape keeps the exact 4 × 90 protocol matrix while allowing
    Appendix F to project an asset-long display without recreating allocations.
    """
    meta = _factor_meta(parent)
    asset_map = mf.AssetMap.default()
    agent = mf.LlmMacroAgent(asset_map=asset_map)
    available_macro = parent.panel.dropna(subset=ext.PANEL_Z_COLS)
    unguarded_by_variant = {
        "pit": pit_unguarded,
        "nonpit_diagnostic": nonpit_unguarded,
    }
    base_by_date = _hrp_base_by_date(parent)
    targets_by_configuration = {
        "factor_pit_ext2026": _parent_targets(parent, "targets_pit"),
        "factor_pit_unguarded_diagnostic_ext2026": pit_unguarded.targets,
        "factor_nonpit_diagnostic_ext2026": _parent_targets(parent, "targets_nonpit"),
        "factor_nonpit_unguarded_diagnostic_ext2026": nonpit_unguarded.targets,
    }
    configuration_pairs = (
        ("factor_pit_ext2026", "pit", True),
        ("factor_pit_unguarded_diagnostic_ext2026", "pit", False),
        ("factor_nonpit_diagnostic_ext2026", "nonpit_diagnostic", True),
        ("factor_nonpit_unguarded_diagnostic_ext2026", "nonpit_diagnostic", False),
    )
    rows: list[dict[str, object]] = []
    guarded_cache: dict[tuple[str, pd.Timestamp], fs.FactorDecision] = {}
    for configuration, variant, guard_enabled in configuration_pairs:
        target_table = targets_by_configuration[configuration]
        last_target: pd.Series | None = None
        last_target_date: pd.Timestamp | None = None
        for rebalance_date in parent.rebalance_dates:
            record = ext.resolve_dated_evidence(parent.evidence, variant, rebalance_date)
            if guard_enabled:
                cache_key = (variant, rebalance_date)
                decision = guarded_cache.get(cache_key)
                if decision is None:
                    decision = _decision_for(
                        evidence=parent.evidence,
                        variant=variant,
                        rebalance_date=rebalance_date,
                        meta=meta[rebalance_date],
                        agent=agent,
                        symbols=parent.symbols,
                        guard_enabled=True,
                    )
                    guarded_cache[cache_key] = decision
                raw_decision = unguarded_by_variant[variant].decisions[rebalance_date]
            else:
                decision = unguarded_by_variant[variant].decisions[rebalance_date]
                raw_decision = decision

            macro_history = available_macro.loc[available_macro.index < rebalance_date]
            if macro_history.empty:
                raise ValueError(f"no point-in-time macro state exists before {rebalance_date.date()}")
            macro_source_date = pd.Timestamp(macro_history.index[-1])
            macro_state = {column: float(macro_history.iloc[-1][column]) for column in ext.PANEL_Z_COLS}
            executed_target_date: pd.Timestamp | None = None
            if rebalance_date in target_table.index and target_table.loc[rebalance_date].notna().any():
                last_target = target_table.loc[rebalance_date]
                last_target_date = rebalance_date
                executed_target_date = rebalance_date
            if last_target is None:
                raise ValueError(f"{rebalance_date.date()}: no current or prior target exists")
            raw_by_asset, _raw_q = _view_values(raw_decision, asset_map=asset_map)
            applied_by_asset, q_by_asset = _view_values(decision, asset_map=asset_map)
            allocation = _allocation_diagnostics(
                parent,
                rebalance_date=rebalance_date,
                decision=decision,
                base=base_by_date[rebalance_date],
                target=last_target,
            )
            if executed_target_date is None:
                allocation["allocation_status"] = "non_trading_rebalance_date"
                allocation["bl_fallback_reason"] = "rebalance_date_not_in_price_calendar"
                allocation["target_reconstruction_error"] = None
                allocation["bl_weight"] = pd.Series(np.nan, index=parent.symbols)
            raw_tilt = _tilt_magnitude(raw_decision)
            applied_tilt = _tilt_magnitude(decision)
            p_memorized = record.score_p_memorized
            expected = (
                1.0 - float(p_memorized)
                if guard_enabled and decision.parse_ok and p_memorized is not None
                else 1.0
                if decision.parse_ok
                else None
            )
            observed = (applied_tilt / raw_tilt) if raw_tilt > 0.0 else None
            relation_error = applied_tilt - raw_tilt * expected if expected is not None else None
            loadings = raw_decision.loadings.loadings if raw_decision.loadings is not None else {}
            conviction = float(raw_decision.views[0].confidence) if raw_decision.views else None
            row: dict[str, object] = {
                "rebalance_date": rebalance_date,
                "macro_source_date": macro_source_date,
                "configuration": configuration,
                "evidence_id": record.evidence_id,
                "prompt_mode": variant,
                "guard_enabled": guard_enabled,
                "p_memorized": p_memorized,
                "parse_ok": bool(decision.parse_ok),
                "steered": bool(decision.steered),
                "conviction": conviction,
                **macro_state,
                "macro_state_norm": float(np.linalg.norm(list(macro_state.values()))),
                **{f"loading_{axis}": loadings.get(axis) for axis in fs.MACRO_AXES},
                "raw_view_tilt": raw_tilt,
                "applied_view_tilt": applied_tilt,
                "expected_attenuation": expected,
                "observed_attenuation": observed,
                "relation_error": relation_error,
                "allocation_status": allocation["allocation_status"],
                "bl_fallback_reason": allocation["bl_fallback_reason"],
                "target_effective_date": last_target_date,
                "target_reconstruction_error": allocation["target_reconstruction_error"],
            }
            for symbol, asset_key in PANEL_ASSET_KEYS.items():
                row[f"raw_tilt_{asset_key}"] = raw_by_asset[symbol]
                row[f"applied_tilt_{asset_key}"] = applied_by_asset[symbol]
                row[f"bl_q_{asset_key}"] = q_by_asset[symbol]
                row[f"hrp_base_weight_{asset_key}"] = float(allocation["base"].loc[symbol])
                row[f"bl_weight_{asset_key}"] = float(allocation["bl_weight"].loc[symbol])
                row[f"target_weight_{asset_key}"] = float(allocation["target"].loc[symbol])
                row[f"target_delta_{asset_key}"] = float(allocation["target_delta"].loc[symbol])
            rows.append(row)
    panel = pd.DataFrame(rows, columns=list(PANEL_COLUMNS))
    validate_mechanism_panel(panel, expected_dates=parent.rebalance_dates)
    return panel


def build_curve_table(values: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Normalize four common-window producer equity curves for presentation."""
    if set(values) != set(CONFIGS):
        raise ValueError("curve table requires exactly the four configured equity streams")
    common = values[CONFIGS[0]].index
    for value in values.values():
        common = common.intersection(value.index)
    common = common.sort_values()
    if len(common) < 2:
        raise ValueError("guard-ablation curves have no viable common window")
    normalized = {
        configuration: values[configuration].loc[common] / float(values[configuration].loc[common].iloc[0])
        for configuration in CONFIGS
    }
    controlled = normalized[CONFIGS[1]] / normalized[CONFIGS[0]]
    stress = normalized[CONFIGS[3]] / normalized[CONFIGS[0]]
    rows: list[dict[str, object]] = []
    for configuration in CONFIGS:
        curve = normalized[configuration]
        relative = controlled if configuration == CONFIGS[1] else stress if configuration == CONFIGS[3] else None
        kind = (
            "controlled_pit_unguarded_vs_guarded"
            if configuration == CONFIGS[1]
            else "combined_nonpit_unguarded_vs_pit_guarded_stress"
            if configuration == CONFIGS[3]
            else None
        )
        drawdown = curve / curve.cummax() - 1.0
        for stamp, wealth in curve.items():
            rows.append(
                {
                    "date": pd.Timestamp(stamp),
                    "configuration": configuration,
                    "normalized_wealth": float(wealth),
                    "drawdown": float(drawdown.loc[stamp]),
                    "relative_wealth": None if relative is None else float(relative.loc[stamp]),
                    "relative_wealth_kind": kind,
                }
            )
    return pd.DataFrame(rows, columns=list(CURVE_COLUMNS))


def validate_mechanism_panel(panel: pd.DataFrame, *, expected_dates: pd.DatetimeIndex | None = None) -> None:
    """Validate the strict flat 4 × 90 panel without coercing score nulls."""
    if not isinstance(panel, pd.DataFrame) or not panel.index.equals(pd.RangeIndex(len(panel))):
        raise ValueError("mechanism panel must be a flat DataFrame")
    if tuple(panel.columns) != PANEL_COLUMNS:
        raise ValueError("mechanism panel columns diverge from the producer contract")
    if len(panel) != 360:
        raise ValueError("mechanism panel must contain exactly 360 rebalance/configuration rows")
    parsed_dates = pd.to_datetime(panel["rebalance_date"], errors="raise")
    if parsed_dates.dt.tz is not None:
        raise ValueError("mechanism panel dates must be timezone-naive")
    calendar: tuple[pd.Timestamp, ...] | None = None
    for offset, configuration in enumerate(CONFIGS):
        block = panel.iloc[offset * 90 : (offset + 1) * 90]
        dates = parsed_dates.iloc[offset * 90 : (offset + 1) * 90]
        if tuple(block["configuration"]) != (configuration,) * 90:
            raise ValueError("mechanism panel configuration blocks are out of canonical order")
        if dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError("mechanism panel rebalance dates must be unique and increasing")
        expected_guard = configuration in {CONFIGS[0], CONFIGS[2]}
        if set(bool(item) for item in block["guard_enabled"]) != {expected_guard}:
            raise ValueError("mechanism panel guard flags do not agree with the configuration")
        p_values = pd.to_numeric(block.loc[block["p_memorized"].notna(), "p_memorized"], errors="raise")
        if not np.isfinite(p_values.to_numpy(dtype=float)).all() or not ((p_values >= 0.0) & (p_values <= 1.0)).all():
            raise ValueError("mechanism panel non-null p_memorized values must be finite probabilities")
        if block["evidence_id"].isna().any() or block["prompt_mode"].isna().any():
            raise ValueError("mechanism panel retains all evidence and prompt identities")
        macro_source_dates = pd.to_datetime(block["macro_source_date"], errors="raise")
        if not np.all(macro_source_dates.to_numpy() < dates.to_numpy()):
            raise ValueError("mechanism panel macro source dates must precede each rebalance")
        always_finite_columns = [
            *ext.PANEL_Z_COLS,
            "macro_state_norm",
            "raw_view_tilt",
            "applied_view_tilt",
            *(
                f"{field}_{asset_key}"
                for field in ("raw_tilt", "applied_tilt", "bl_q", "hrp_base_weight", "target_weight", "target_delta")
                for asset_key in PANEL_ASSET_KEYS.values()
            ),
        ]
        finite = block[always_finite_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(finite).all():
            raise ValueError("mechanism panel allocation and macro diagnostics must be finite")
        executed = block["allocation_status"] != "non_trading_rebalance_date"
        executed_error = pd.to_numeric(block.loc[executed, "target_reconstruction_error"], errors="raise")
        if not np.isfinite(executed_error.to_numpy(dtype=float)).all() or (executed_error > 1e-10).any():
            raise ValueError("mechanism panel target reconstruction exceeds tolerance")
        if block.loc[~executed, "target_reconstruction_error"].notna().any():
            raise ValueError("non-trading panel rows must not claim a target reconstruction")
        for asset_key in PANEL_ASSET_KEYS.values():
            bl_weights = pd.to_numeric(block.loc[executed, f"bl_weight_{asset_key}"], errors="raise")
            if not np.isfinite(bl_weights.to_numpy(dtype=float)).all():
                raise ValueError("executed mechanism rows require finite reconstructed BL weights")
            if block.loc[~executed, f"bl_weight_{asset_key}"].notna().any():
                raise ValueError("non-trading mechanism rows must not claim reconstructed BL weights")
        if not set(block["allocation_status"]).issubset({"bl_blend_applied", "fallback", "non_trading_rebalance_date"}):
            raise ValueError("mechanism panel allocation status is invalid")
        for symbol, asset_key in PANEL_ASSET_KEYS.items():
            raw = pd.to_numeric(block[f"raw_tilt_{asset_key}"], errors="raise")
            applied = pd.to_numeric(block[f"applied_tilt_{asset_key}"], errors="raise")
            if expected_guard:
                probabilities = pd.to_numeric(block["p_memorized"], errors="raise")
                expected_asset = raw * (1.0 - probabilities.fillna(0.0))
            else:
                expected_asset = raw
            if not np.allclose(applied, expected_asset, atol=1e-10, rtol=0.0):
                raise ValueError(f"mechanism panel guard relation failed for {configuration}/{symbol}")
        block_calendar = tuple(pd.Timestamp(value) for value in dates)
        if calendar is None:
            calendar = block_calendar
        elif block_calendar != calendar:
            raise ValueError("mechanism panel configurations must share the parent rebalance calendar")
    if expected_dates is not None and calendar != tuple(pd.Timestamp(value) for value in expected_dates):
        raise ValueError("mechanism panel calendar diverges from parent evidence")


def _artifact_entry(path: Path, *, schema: str, rows: int | None = None, columns: Sequence[str] | None = None) -> dict[str, object]:
    entry: dict[str, object] = {
        "file": path.name,
        "sha256": sha256_file(path),
        "size": int(path.stat().st_size),
        "schema": schema,
    }
    if rows is not None:
        entry["rows"] = int(rows)
    if columns is not None:
        entry["columns"] = list(columns)
    return entry


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def _write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(payload))


def _audit_payload(parent: ParentInputs, *, n_rebalances: int) -> dict[str, object]:
    return {
        "schema": AUDIT_SCHEMA,
        "result": "pass",
        "guard_enabled": False,
        "source_to_consumption": "validated for each unguarded variant/date through parent dated evidence",
        "parent_evidence": {
            "file": str(parent.manifest["files"]["evidence"]["file"]),
            "sha256": str(parent.manifest["files"]["evidence"]["sha256"]),
            "rows": int(len(parent.evidence_frame)),
        },
        "counts": {
            "rebalances_per_variant": n_rebalances,
            "unguarded_variants": len(VARIANTS),
            "consumed_keys": n_rebalances * len(VARIANTS),
        },
    }


def _validate_local_inventory(
    run_dir: Path, manifest: Mapping[str, object], *, include_completion_marker: bool = True
) -> None:
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("guard-ablation manifest must inventory its artifacts")
    names: set[str] = set()
    for role, entry in files.items():
        if not isinstance(role, str) or not isinstance(entry, Mapping):
            raise ValueError("guard-ablation manifest inventory is malformed")
        name = entry.get("file")
        if not isinstance(name, str) or Path(name).name != name or name in names:
            raise ValueError("guard-ablation manifest inventory file name is unsafe or duplicate")
        names.add(name)
        path = run_dir / name
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"guard-ablation artifact is missing or mutated: {name}")
        if entry.get("size") != path.stat().st_size:
            raise ValueError(f"guard-ablation artifact size diverges: {name}")
    actual = {path.name for path in run_dir.iterdir() if path.is_file()}
    expected = names | {"manifest.json"}
    if include_completion_marker:
        expected.add("COMPLETED")
    if actual != expected:
        raise ValueError("guard-ablation run contains unmanifested output")


def validate_guard_ablation_run(run_dir: Path | str) -> Mapping[str, object]:
    """Validate bundle identity, inventory and compatibility-critical shapes."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    marker_path = run_dir / "COMPLETED"
    if not manifest_path.is_file() or not marker_path.is_file():
        raise ValueError("guard-ablation run is incomplete: manifest.json and COMPLETED are required")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != RUN_SCHEMA or manifest.get("completed") is not True:
        raise ValueError("guard-ablation manifest has an incompatible completion schema")
    if manifest.get("run_id") != run_dir.name:
        raise ValueError("guard-ablation directory name must equal manifest run_id")
    marker = marker_path.read_text().splitlines()
    if marker != [f"manifest_sha256={sha256_file(manifest_path)}"]:
        raise ValueError("guard-ablation COMPLETED marker does not bind final manifest bytes")
    inputs = manifest.get("input_manifests")
    if not isinstance(inputs, Mapping) or set(("factor_run", "market_snapshot")) - set(inputs):
        raise ValueError("guard-ablation manifest must pin parent Factor and market snapshots")
    _validate_local_inventory(run_dir, manifest)
    files = manifest["files"]
    for filename in _UNGUARDED_ARTIFACTS.values():
        if not any(entry["file"] == filename for entry in files.values()):
            raise ValueError(f"guard-ablation manifest lacks required unguarded artifact {filename}")
    panel_entry = files["panel"]
    panel = pd.read_parquet(run_dir / panel_entry["file"])
    validate_mechanism_panel(panel)
    metric_records = json.loads((run_dir / files["metric_records"]["file"]).read_text())
    if metric_records.get("schema") != METRIC_SCHEMA or len(metric_records.get("records", [])) != 21:
        raise ValueError("guard-ablation metric records lack the exact 21-row matrix")
    curves = pd.read_parquet(run_dir / files["curves"]["file"])
    if tuple(curves.columns) != CURVE_COLUMNS or curves.empty:
        raise ValueError("guard-ablation curve table has an incompatible schema")
    return manifest


def build_guard_ablation_run(
    *,
    output_dir: Path | str = OUTPUT_DEFAULT,
    parent_run_dir: Path | str = PARENT_RUN_DEFAULT,
    snapshot_dir: Path | str = SNAPSHOT_DEFAULT,
    macro_panel_path: Path | str = MACRO_PANEL_DEFAULT,
    n_boot: int = 1000,
) -> Path:
    """Produce an immutable offline guard-ablation run, writing ``COMPLETED`` last."""
    destination = Path(output_dir)
    _require_new_destination(destination)
    parent = load_parent_inputs(
        parent_run_dir=parent_run_dir,
        snapshot_dir=snapshot_dir,
        macro_panel_path=macro_panel_path,
    )
    destination.mkdir(parents=True, exist_ok=True)

    pit_unguarded = replay_unguarded_variant(
        parent,
        variant="pit",
        configuration="factor_pit_unguarded_diagnostic_ext2026",
    )
    nonpit_unguarded = replay_unguarded_variant(
        parent,
        variant="nonpit_diagnostic",
        configuration="factor_nonpit_unguarded_diagnostic_ext2026",
    )
    values: dict[str, pd.Series] = {
        "factor_pit_ext2026": _parent_equity(parent, "equity_pit"),
        "factor_pit_unguarded_diagnostic_ext2026": pit_unguarded.value,
        "factor_nonpit_diagnostic_ext2026": _parent_equity(parent, "equity_nonpit"),
        "factor_nonpit_unguarded_diagnostic_ext2026": nonpit_unguarded.value,
    }
    panel = build_mechanism_panel(
        parent,
        pit_unguarded=pit_unguarded,
        nonpit_unguarded=nonpit_unguarded,
    )
    metric_records = build_metric_records(parent, values=values, n_boot=n_boot)
    curves = build_curve_table(values)

    files: dict[str, dict[str, object]] = {}
    # Retaining the two guarded streams makes the child independently drawable;
    # their bytes remain explicitly identified as projections of the parent run.
    for role, configuration in (
        ("equity_pit_guarded", "factor_pit_ext2026"),
        ("equity_nonpit_guarded", "factor_nonpit_diagnostic_ext2026"),
    ):
        name = "factor_equity_ext2026.parquet" if configuration == CONFIGS[0] else "factor_nonpit_diagnostic_equity_ext2026.parquet"
        path = destination / name
        _write_parquet(values[configuration].to_frame("value"), path)
        files[role] = _artifact_entry(path, schema="factor_guard_ablation.equity_input.v1", rows=len(values[configuration]))
    for role, output in (
        ("targets_pit_unguarded", pit_unguarded.targets),
        ("equity_pit_unguarded", pit_unguarded.value.to_frame("value")),
        ("targets_nonpit_unguarded", nonpit_unguarded.targets),
        ("equity_nonpit_unguarded", nonpit_unguarded.value.to_frame("value")),
    ):
        path = destination / _UNGUARDED_ARTIFACTS[role]
        _write_parquet(output, path)
        schema = "factor_guard_ablation.targets.v1" if role.startswith("targets") else "factor_guard_ablation.equity_input.v1"
        files[role] = _artifact_entry(path, schema=schema, rows=len(output))
    for role, payload in (
        ("decision_log_pit_unguarded", pit_unguarded.decision_log),
        ("decision_log_nonpit_unguarded", nonpit_unguarded.decision_log),
    ):
        path = destination / _UNGUARDED_ARTIFACTS[role]
        _write_json(payload, path)
        files[role] = _artifact_entry(path, schema=DECISION_SCHEMA, rows=len(parent.rebalance_dates))

    metric_path = destination / "factor_guard_ablation_metric_records_ext2026.json"
    _write_json(metric_records, metric_path)
    files["metric_records"] = _artifact_entry(metric_path, schema=METRIC_SCHEMA, rows=len(metric_records["records"]))
    panel_path = destination / "factor_guard_ablation_panel_ext2026.parquet"
    _write_parquet(panel, panel_path)
    files["panel"] = _artifact_entry(panel_path, schema=PANEL_SCHEMA, rows=len(panel), columns=PANEL_COLUMNS)
    curves_path = destination / "factor_guard_ablation_curves_ext2026.parquet"
    _write_parquet(curves, curves_path)
    files["curves"] = _artifact_entry(curves_path, schema=EQUITY_SCHEMA, rows=len(curves), columns=CURVE_COLUMNS)
    audit_path = destination / "factor_guard_ablation_replay_audit_ext2026.json"
    _write_json(_audit_payload(parent, n_rebalances=len(parent.rebalance_dates)), audit_path)
    files["replay_audit"] = _artifact_entry(audit_path, schema=AUDIT_SCHEMA, rows=2 * len(parent.rebalance_dates))

    manifest = {
        "schema": RUN_SCHEMA,
        "completed": True,
        "run_id": destination.name,
        "build_time": datetime.now(timezone.utc).isoformat(),
        "source_commit": ext._git_source_commit(),
        "parent_source_commit": parent.manifest["source_commit"],
        "producer_source": {
            "file": "scripts/build_factor_guard_ablation.py",
            "sha256": sha256_file(__file__),
            "git_head": ext._git_source_commit(),
        },
        "config": {
            "guard_enabled": False,
            "n_boot": int(n_boot),
            "parent_factor_config": dict(parent.manifest["config"]),
            "macro_panel": _repo_relative_path(macro_panel_path, label="macro_panel_path"),
            "macro_panel_sha256": sha256_file(macro_panel_path),
        },
        "input_manifests": {
            "factor_run": {
                "run_id": parent.manifest["run_id"],
                "manifest_sha256": parent.manifest_sha256,
            },
            "market_snapshot": {
                "snapshot_id": parent.snapshot_manifest["snapshot_id"],
                "manifest_sha256": parent.snapshot_sha256,
            },
        },
        "panel_schema": {"columns": list(PANEL_COLUMNS)},
        "files": files,
    }
    manifest_path = destination / "manifest.json"
    _write_json(manifest, manifest_path)
    # Validate every inventory entry before the marker exists.  Completion is the
    # final mutation and no failed staging output can be mistaken for a run.
    _validate_local_inventory(destination, manifest, include_completion_marker=False)
    (destination / "COMPLETED").write_text(f"manifest_sha256={sha256_file(manifest_path)}\n")
    validate_guard_ablation_run(destination)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--parent-run-dir", type=Path, default=PARENT_RUN_DEFAULT)
    parser.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_DEFAULT)
    parser.add_argument("--macro-panel", type=Path, default=MACRO_PANEL_DEFAULT)
    parser.add_argument("--n-boot", type=int, default=1000, help="SSR bootstrap replications (default: 1000)")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.n_boot <= 0:
        raise ValueError("--n-boot must be positive")
    run_dir = build_guard_ablation_run(
        output_dir=args.output_dir,
        parent_run_dir=args.parent_run_dir,
        snapshot_dir=args.snapshot_dir,
        macro_panel_path=args.macro_panel,
        n_boot=args.n_boot,
    )
    print(f"guard-ablation run COMPLETED: {run_dir}")


if __name__ == "__main__":
    main()
