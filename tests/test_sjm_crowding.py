"""SJM producer boundary: frozen protocol, input gate, cash alignment, equations.

Remediation tasks 7.2 (protocol), 7.1 (manifest-gated inputs), 7.3 (exact
factor-calendar BIL total returns), 7.4 (exact overlay/control equations),
7.5 (ordered deterministic mutation-registry replay), 7.6 (development-only
gates and corrected winner selection), 7.7 (immutable SJM v3 run assembly and
its one-validator proof), 7.8 (immutable staging and command behavior), and
7.9 (the serialized SJM producer test boundary). These tests are offline.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_sjm_crowding as sjm  # noqa: E402


LIMITS = {
    "bull": {0: 1.0, 1: 0.9, 2: 0.8},
    "neutral": {0: 1.0, 1: 0.9, 2: 0.8},
    "bear": {0: 1.0, 1: 0.8, 2: 0.5},
}
MUTATIONS = (
    ("lam", 20.0),
    ("lam", 100.0),
    ("window", 126),
    ("window", 504),
    ("signal", "turbulence"),
    ("signal", "absorption"),
    ("scale", 0.9),
    ("scale", 1.1),
    ("scale", 1.2),
    ("scale", 1.3),
    ("floor", 0.5),
    ("floor", 0.3),
    ("arm", None),
    ("arm", -0.02),
    ("arm", -0.03),
    ("arm", -0.04),
    ("arm", -0.05),
)


class _EqualityMaskingStr(str):
    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False

    __hash__ = str.__hash__


class _EqualityMaskingMutation(sjm.SJMMutation):
    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False


def _forged_mutation() -> _EqualityMaskingMutation:
    approved = sjm.approved_mutation_registry()[0]
    forged = object.__new__(_EqualityMaskingMutation)
    forged.__dict__.update(approved.__dict__)
    object.__setattr__(forged, "value", 999.0)
    return forged


def _sha(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plain_limits(protocol: sjm.SJMSelectionProtocol) -> dict[str, dict[int, float]]:
    return {
        regime: {bucket: value for bucket, value in row.items()}
        for regime, row in protocol.seed_config.limits.items()
    }


def test_protocol_pins_the_approved_dates_objective_gates_and_search() -> None:
    protocol = sjm.make_sjm_selection_protocol()

    assert protocol.protocol_id == "sjm_selection_v2_calmar_frozen"
    assert protocol.dev_start == pd.Timestamp("2019-01-03")
    assert protocol.dev_end == pd.Timestamp("2024-06-30")
    assert protocol.holdout_start == pd.Timestamp("2024-07-01")
    assert protocol.objective == "dev_calmar"
    assert protocol.cagr_budget == 0.035
    assert protocol.control_rule == "derisk_cash_pin"
    assert protocol.require_maxdd_not_worse_than_control is True
    assert (protocol.seed, protocol.dry_rounds, protocol.max_iters) == (42, 12, 60)
    assert dict(protocol.signal_step) == {"absorption": 5, "turbulence": 1}
    assert protocol.seed_config.lam == 50.0
    assert protocol.seed_config.signal == "absorption"
    assert protocol.seed_config.window == 252
    assert protocol.seed_config.scale == 1.0
    assert protocol.seed_config.floor == 0.4
    assert protocol.seed_config.arm is None
    assert _plain_limits(protocol) == LIMITS
    assert protocol.lambda_candidates == (20.0, 100.0)
    assert protocol.window_candidates == (126, 504)
    assert protocol.signal_candidates == ("turbulence", "absorption")
    assert protocol.scale_candidates == (0.9, 1.1, 1.2, 1.3)
    assert protocol.floor_candidates == (0.5, 0.3)
    assert protocol.arm_candidates == (None, -0.02, -0.03, -0.04, -0.05)
    assert tuple((item.parameter, item.value) for item in protocol.mutation_registry) == MUTATIONS


def test_protocol_hashes_canonical_fields_limit_table_and_full_mutation_order() -> None:
    protocol = sjm.make_sjm_selection_protocol()
    limit_payload = {
        regime: {str(bucket): value for bucket, value in row.items()}
        for regime, row in LIMITS.items()
    }
    mutation_payload = [
        {"parameter": parameter, "value": value}
        for parameter, value in MUTATIONS
    ]
    protocol_payload = {
        "protocol_id": "sjm_selection_v2_calmar_frozen",
        "dev_start": "2019-01-03",
        "dev_end": "2024-06-30",
        "holdout_start": "2024-07-01",
        "objective": "dev_calmar",
        "cagr_budget": 0.035,
        "control_rule": "derisk_cash_pin",
        "require_maxdd_not_worse_than_control": True,
        "seed": 42,
        "dry_rounds": 12,
        "max_iters": 60,
        "signal_step": {"absorption": 5, "turbulence": 1},
        "seed_config": {
            "lam": 50.0,
            "signal": "absorption",
            "window": 252,
            "scale": 1.0,
            "floor": 0.4,
            "arm": None,
            "limits": limit_payload,
        },
        "lambda_candidates": [20.0, 100.0],
        "window_candidates": [126, 504],
        "signal_candidates": ["turbulence", "absorption"],
        "scale_candidates": [0.9, 1.1, 1.2, 1.3],
        "floor_candidates": [0.5, 0.3],
        "arm_candidates": [None, -0.02, -0.03, -0.04, -0.05],
        "limit_table": limit_payload,
        "mutation_registry": mutation_payload,
    }

    assert protocol.limit_table_sha256 == _sha(limit_payload)
    assert protocol.mutation_registry_sha256 == _sha(mutation_payload)
    assert protocol.protocol_sha256 == _sha(protocol_payload)
    assert protocol.fingerprint() == protocol.protocol_sha256


def test_identical_constructions_hash_identically_despite_mapping_insertion_order() -> None:
    reordered = {
        "bear": {2: 0.5, 1: 0.8, 0: 1.0},
        "neutral": {2: 0.8, 1: 0.9, 0: 1.0},
        "bull": {2: 0.8, 1: 0.9, 0: 1.0},
    }
    first = sjm.make_sjm_selection_protocol()
    second = sjm.make_sjm_selection_protocol(limit_table=reordered)

    assert second == first
    assert second.protocol_sha256 == first.protocol_sha256
    assert second.limit_table_sha256 == first.limit_table_sha256
    assert second.mutation_registry_sha256 == first.mutation_registry_sha256


def test_limit_table_and_signal_cadence_are_defensive_read_only_copies() -> None:
    supplied = {regime: dict(row) for regime, row in LIMITS.items()}
    protocol = sjm.make_sjm_selection_protocol(limit_table=supplied)
    supplied["bear"][2] = 0.99

    assert protocol.seed_config.limits["bear"][2] == 0.5
    with pytest.raises(TypeError):
        protocol.seed_config.limits["bear"][2] = 0.99  # type: ignore[index]
    with pytest.raises(TypeError):
        protocol.seed_config.limits["bear"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        protocol.signal_step["absorption"] = 1  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        protocol.seed = 7  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("dev_start", pd.Timestamp("2019-01-04"), "dev_start"),
        ("dev_end", pd.Timestamp("2024-06-29"), "dev_end"),
        ("holdout_start", pd.Timestamp("2024-07-02"), "holdout_start"),
        ("objective", "dev_maxdd", "objective"),
        ("cagr_budget", 0.04, "cagr_budget"),
        ("seed", 43, "seed"),
        ("dry_rounds", 11, "dry_rounds"),
        ("max_iters", 61, "max_iters"),
        ("lambda_candidates", (100.0, 20.0), "lambda_candidates"),
        ("signal_candidates", ("absorption", "turbulence"), "signal_candidates"),
        ("arm_candidates", (None, -0.03, -0.02, -0.04, -0.05), "arm_candidates"),
    ],
)
def test_any_frozen_protocol_field_or_candidate_order_change_is_rejected(
    field: str, value: object, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(sjm.make_sjm_selection_protocol(), **{field: value})


def test_limit_value_or_mutation_registry_order_change_invalidates_protocol() -> None:
    changed_limit = {regime: dict(row) for regime, row in LIMITS.items()}
    changed_limit["bear"][2] = 0.65
    with pytest.raises(ValueError, match="limit table"):
        sjm.make_sjm_selection_protocol(limit_table=changed_limit)

    registry = list(sjm.approved_mutation_registry())
    registry[0], registry[1] = registry[1], registry[0]
    with pytest.raises(ValueError, match="mutation_registry"):
        sjm.make_sjm_selection_protocol(mutation_registry=registry)


def test_constructor_rejects_str_subclasses_that_mask_changed_values() -> None:
    protocol = sjm.make_sjm_selection_protocol()

    with pytest.raises(ValueError, match="protocol_id must be a plain str"):
        dataclasses.replace(
            protocol,
            protocol_id=_EqualityMaskingStr("tampered_protocol"),
        )
    with pytest.raises(ValueError, match="signal_candidates.*plain str"):
        dataclasses.replace(
            protocol,
            signal_candidates=(
                _EqualityMaskingStr("tampered_signal"),
                "absorption",
            ),
        )


def test_constructor_rejects_mutation_subclasses_that_mask_changed_values() -> None:
    registry = list(sjm.approved_mutation_registry())
    forged = _forged_mutation()
    registry[0] = forged

    with pytest.raises(ValueError, match="exact SJMMutation"):
        sjm.make_sjm_selection_protocol(mutation_registry=registry)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "protocol_id",
            _EqualityMaskingStr("tampered_protocol"),
            "protocol_id must be a plain str",
        ),
        (
            "signal_candidates",
            (_EqualityMaskingStr("tampered_signal"), "absorption"),
            "signal_candidates.*plain str",
        ),
    ],
)
def test_validator_rejects_str_subclass_tampering_before_hash_comparison(
    field: str, value: object, match: str
) -> None:
    protocol = sjm.make_sjm_selection_protocol()
    object.__setattr__(protocol, field, value)

    with pytest.raises(ValueError, match=match):
        sjm.validate_sjm_selection_protocol(protocol)


def test_validator_rejects_mutation_subclass_tampering_before_hash_comparison() -> None:
    protocol = sjm.make_sjm_selection_protocol()
    registry = list(protocol.mutation_registry)
    forged = _forged_mutation()
    registry[0] = forged
    object.__setattr__(protocol, "mutation_registry", tuple(registry))

    with pytest.raises(ValueError, match="exact SJMMutation"):
        sjm.validate_sjm_selection_protocol(protocol)


def test_validator_requires_the_exact_protocol_runtime_type() -> None:
    class ProtocolSubclass(sjm.SJMSelectionProtocol):
        pass

    protocol = sjm.make_sjm_selection_protocol()
    forged = object.__new__(ProtocolSubclass)
    forged.__dict__.update(protocol.__dict__)

    with pytest.raises(ValueError, match="exact SJMSelectionProtocol"):
        sjm.validate_sjm_selection_protocol(forged)


def test_derisk_cash_pin_is_the_only_authoritative_control_identity() -> None:
    protocol = sjm.make_sjm_selection_protocol(control_alias="derisk_cash_pin")
    sjm.validate_sjm_selection_protocol(protocol, control_alias="derisk_cash_pin")

    for alias in ("corr_overlay", "correlation-overlay control", "derisk-cash-pin"):
        with pytest.raises(ValueError, match="derisk_cash_pin"):
            sjm.make_sjm_selection_protocol(control_alias=alias)
        with pytest.raises(ValueError, match="derisk_cash_pin"):
            sjm.validate_sjm_selection_protocol(protocol, control_alias=alias)

    with pytest.raises(ValueError, match="derisk_cash_pin"):
        dataclasses.replace(protocol, control_rule="corr_overlay")


# --------------------------------------------------------------------------- #
# Input gate, exact cash alignment, and portfolio equations (7.1 / 7.3 / 7.4)  #
# --------------------------------------------------------------------------- #


def _write_completed_snapshot(
    root: Path,
    index: pd.DatetimeIndex,
    *,
    bil: pd.Series | None = None,
    name: str = "market_total_return_fx_2026-06-30_v9",
) -> Path:
    """Hash-valid completed market snapshot with an SJM-controllable BIL column."""
    snapshot = root / name
    snapshot.mkdir(parents=True)
    n = len(index)
    if bil is None:
        bil = pd.Series(100.0 * np.cumprod(np.full(n, 1.0 + 2e-4)), index=index)
    tables = {
        "basket_adjusted_close_local.parquet": pd.DataFrame(
            {"SWDA.L": np.linspace(100.0, 110.0, n)}, index=index
        ),
        "cash_market_total_return.parquet": pd.DataFrame(
            {
                "BIL": bil,
                "SPY": pd.Series(
                    100.0 * np.cumprod(np.full(n, 1.0 + 3e-4)), index=index
                ),
            },
            index=index,
        ),
        "fx_usd_per_gbp.parquet": pd.DataFrame(
            {"USD_per_GBP": np.linspace(1.2, 1.3, n)}, index=index
        ),
    }
    files = {}
    for fname, frame in tables.items():
        path = snapshot / fname
        frame.to_parquet(path)
        files[fname] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(frame),
        }
    manifest = {
        "schema": "market_snapshot.v1",
        "snapshot_id": name,
        "cash_symbol": "BIL",
        "benchmark_symbol": "SPY",
        "total_return_field": (
            "yfinance auto_adjust=True Close "
            "(adjusted total-return level, dividends reinvested)"
        ),
        "requested_coverage": {
            "start": index.min().date().isoformat(),
            "end": index.max().date().isoformat(),
        },
        "files": files,
        "overlap_revisions": {"preceding_snapshot": None},
        "completed": True,
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    (snapshot / "COMPLETED").write_text("complete\n")
    return snapshot


@pytest.fixture(scope="module")
def sjm_completed_case(tmp_path_factory):
    """One completed Factor run bundle plus the completed snapshot it was built on."""
    from tests.test_stream_ext2026 import _completed_factor_run

    tmp = tmp_path_factory.mktemp("sjm_inputs_case")
    _, _, run_dir, run_kwargs = _completed_factor_run(tmp)
    snapshot_dir = tmp / run_kwargs["input_manifests"]["market_snapshot"]["snapshot_id"]
    expected = {
        "factor_run_id": run_kwargs["run_id"],
        "factor_manifest_sha256": hashlib.sha256(
            (run_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "market_snapshot_id": snapshot_dir.name,
        "market_snapshot_sha256": hashlib.sha256(
            (snapshot_dir / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    return run_dir, snapshot_dir, expected


def test_sjm_input_gate_loads_one_typed_input_set_from_completed_manifests(
    sjm_completed_case,
) -> None:
    run_dir, snapshot_dir, expected = sjm_completed_case

    inputs = sjm.load_sjm_inputs(run_dir, snapshot_dir, **expected)

    assert type(inputs) is sjm.SJMInputs
    assert (inputs.factor_run_id, inputs.factor_manifest_sha256) == (
        expected["factor_run_id"],
        expected["factor_manifest_sha256"],
    )
    assert (inputs.market_snapshot_id, inputs.market_snapshot_sha256) == (
        expected["market_snapshot_id"],
        expected["market_snapshot_sha256"],
    )

    equity = pd.read_parquet(run_dir / "factor_equity_ext2026.parquet")["value"]
    assert inputs.factor_value.index.equals(equity.index)
    assert inputs.factor_returns.index.equals(equity.index[1:])
    assert np.allclose(
        inputs.factor_returns.to_numpy(),
        equity.pct_change(fill_method=None).iloc[1:].to_numpy(),
        rtol=1e-12,
        atol=0.0,
    )
    assert inputs.cash_returns.index.equals(inputs.factor_returns.index)
    assert inputs.cash_record["cash_benchmark_id"] == "BIL"
    assert inputs.cash_record["snapshot_id"] == expected["market_snapshot_id"]
    assert inputs.cash_record["cash_anchor"] == equity.index[0]
    assert inputs.cash_record["cash_n_obs"] == len(inputs.cash_returns)

    targets = pd.read_parquet(run_dir / "factor_targets_ext2026.parquet")
    assert inputs.rebalance_dates.equals(targets.index)

    assert {r: dict(row) for r, row in inputs.limit_table.items()} == LIMITS
    with pytest.raises(TypeError):
        inputs.limit_table["bear"][2] = 0.9  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        inputs.factor_run_id = "tampered"  # type: ignore[misc]
    assert inputs.protocol.fingerprint() == sjm.make_sjm_selection_protocol().fingerprint()


def test_sjm_input_gate_rejects_loose_incomplete_or_mismatched_inputs(
    sjm_completed_case, tmp_path
) -> None:
    run_dir, snapshot_dir, expected = sjm_completed_case

    def load(factor=run_dir, snapshot=snapshot_dir, **overrides):
        return sjm.load_sjm_inputs(factor, snapshot, **{**expected, **overrides})

    with pytest.raises(ValueError, match="factor run identity"):
        load(factor_run_id="factor_ext2026_someone_else_v9")
    with pytest.raises(ValueError, match="factor run manifest sha256"):
        load(factor_manifest_sha256="0" * 64)
    with pytest.raises(ValueError, match="snapshot identity"):
        load(market_snapshot_id="market_total_return_fx_1999-01-01_v0")
    with pytest.raises(ValueError, match="snapshot manifest sha256"):
        load(market_snapshot_sha256="f" * 64)
    with pytest.raises(ValueError, match="loose legacy artifact"):
        load(factor=run_dir / "factor_equity_ext2026.parquet")
    with pytest.raises(ValueError, match="loose legacy artifact"):
        load(snapshot=snapshot_dir / "cash_market_total_return.parquet")

    incomplete = tmp_path / run_dir.name
    shutil.copytree(run_dir, incomplete)
    (incomplete / "COMPLETED").unlink()
    with pytest.raises(ValueError, match="COMPLETED"):
        load(factor=incomplete)

    equity = pd.read_parquet(run_dir / "factor_equity_ext2026.parquet")
    other = _write_completed_snapshot(
        tmp_path, equity.index, name="market_total_return_fx_2026-06-30_v8"
    )
    with pytest.raises(ValueError, match="lineage"):
        load(
            snapshot=other,
            market_snapshot_id=other.name,
            market_snapshot_sha256=hashlib.sha256(
                (other / "manifest.json").read_bytes()
            ).hexdigest(),
        )


def test_snapshot_coverage_gate_requires_anchor_through_factor_endpoint() -> None:
    manifest = {"requested_coverage": {"start": "2021-01-04", "end": "2023-05-01"}}
    sjm.require_snapshot_coverage(
        manifest, anchor=pd.Timestamp("2021-01-04"), endpoint=pd.Timestamp("2023-05-01")
    )
    with pytest.raises(ValueError, match="factor endpoint"):
        sjm.require_snapshot_coverage(
            manifest,
            anchor=pd.Timestamp("2021-01-04"),
            endpoint=pd.Timestamp("2023-05-02"),
        )
    with pytest.raises(ValueError, match="factor endpoint"):
        sjm.require_snapshot_coverage(
            manifest,
            anchor=pd.Timestamp("2021-01-01"),
            endpoint=pd.Timestamp("2023-04-01"),
        )
    with pytest.raises(ValueError, match="requested_coverage"):
        sjm.require_snapshot_coverage(
            {}, anchor=pd.Timestamp("2021-01-04"), endpoint=pd.Timestamp("2023-05-01")
        )


def test_ac_4_2(tmp_path) -> None:
    """R4.2: unavailable cash for a required interval fails; zero is never substituted."""
    calendar = pd.bdate_range("2022-01-03", periods=40)
    anchor, ridx = calendar[0], calendar[1:]

    interior = _write_completed_snapshot(tmp_path / "interior", calendar.delete(20))
    with pytest.raises(ValueError, match="missing"):
        sjm.cash_returns_on_factor_calendar(interior, ridx, anchor=anchor)

    endpoint = _write_completed_snapshot(tmp_path / "endpoint", calendar[:-1])
    with pytest.raises(ValueError, match="missing"):
        sjm.cash_returns_on_factor_calendar(endpoint, ridx, anchor=anchor)

    no_anchor = _write_completed_snapshot(tmp_path / "no_anchor", calendar[1:])
    with pytest.raises(ValueError, match="missing"):
        sjm.cash_returns_on_factor_calendar(no_anchor, ridx, anchor=anchor)

    bil = pd.Series(
        100.0 * np.cumprod(np.full(len(calendar), 1.0 + 2e-4)), index=calendar
    )
    bil.iloc[10] = np.nan
    gapped = _write_completed_snapshot(tmp_path / "gapped", calendar, bil=bil)
    with pytest.raises(ValueError, match="non-finite"):
        sjm.cash_returns_on_factor_calendar(gapped, ridx, anchor=anchor)


def test_cash_returns_match_factor_calendar_and_record_identity(tmp_path) -> None:
    calendar = pd.bdate_range("2022-01-03", periods=30)
    anchor, ridx = calendar[0], calendar[1:]
    snapshot = _write_completed_snapshot(tmp_path, calendar)

    cash, record = sjm.cash_returns_on_factor_calendar(snapshot, ridx, anchor=anchor)
    levels = pd.read_parquet(snapshot / "cash_market_total_return.parquet")["BIL"]

    assert cash.index.equals(ridx)  # exactly the Factor return index...
    expected = levels.pct_change(fill_method=None).iloc[1:]
    assert np.allclose(cash.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=0.0)
    # ...including the FIRST Factor return, anchored one session before it
    assert cash.iloc[0] == pytest.approx(levels.iloc[1] / levels.iloc[0] - 1.0, rel=1e-12)

    assert record["cash_benchmark_id"] == "BIL"
    assert record["cash_semantics"] == "adjusted_total_return"
    assert record["snapshot_id"] == snapshot.name
    assert record["cash_anchor"] == anchor
    assert (record["cash_start"], record["cash_end"]) == (ridx[0], ridx[-1])
    assert record["cash_n_obs"] == len(ridx)
    with pytest.raises(TypeError):
        record["cash_n_obs"] = 0  # type: ignore[index]

    twice = calendar.append(pd.DatetimeIndex([calendar[5]])).sort_values()
    duplicated = _write_completed_snapshot(
        tmp_path / "dup", twice, name="market_total_return_fx_2026-06-30_v7"
    )
    with pytest.raises(ValueError, match="unique"):
        sjm.cash_returns_on_factor_calendar(duplicated, ridx, anchor=anchor)


def test_ac_4_1() -> None:
    """R4.1: the residual 1-exposure sleeve earns the cash benchmark total return."""
    idx = pd.bdate_range("2023-01-02", periods=6)
    base = pd.Series([0.01, -0.02, 0.005, 0.0, 0.03, -0.01], index=idx)
    cash = pd.Series([0.0002, 0.0002, 0.0003, 0.0001, 0.0002, 0.0003], index=idx)
    exposure = pd.Series([1.0, 0.8, 0.5, 0.0, 0.4, 1.0], index=idx)

    overlay = sjm.overlay_returns(base, exposure, cash)

    expected = exposure.to_numpy() * base.to_numpy() + (
        1.0 - exposure.to_numpy()
    ) * cash.to_numpy()
    assert overlay.index.equals(idx)
    assert np.array_equal(overlay.to_numpy(), expected)
    # fully de-risked day: the residual sleeve IS the cash total return, not zero
    assert overlay.loc[idx[3]] == cash.loc[idx[3]]
    # fully invested day: no cash contamination
    assert overlay.loc[idx[0]] == base.loc[idx[0]]


def test_ac_4_3() -> None:
    """R4.3: overlay and derisk_cash_pin control share one cash-return convention."""
    idx = pd.bdate_range("2023-01-02", periods=6)
    base = pd.Series([0.01, -0.02, 0.005, 0.004, 0.03, -0.01], index=idx)
    cash = pd.Series([0.0002, 0.0002, 0.0003, 0.0001, 0.0002, 0.0003], index=idx)
    overlay_exposure = pd.Series([1.0, 0.8, 0.5, 0.9, 0.4, 1.0], index=idx)
    control_exposure = pd.Series([1.0, 0.6, 0.6, 1.0, 0.7, 0.9], index=idx)

    overlay = sjm.overlay_returns(base, overlay_exposure, cash)
    control = sjm.build_control_returns(base, control_exposure, cash)

    for returns, exposure in ((overlay, overlay_exposure), (control, control_exposure)):
        residual = 1.0 - exposure.to_numpy()
        mask = residual > 0
        implied = (
            returns.to_numpy() - exposure.to_numpy() * base.to_numpy()
        )[mask] / residual[mask]
        assert np.allclose(implied, cash.to_numpy()[mask], rtol=1e-9, atol=1e-15)

    # identical exposures plus ONE shared cash vector => identical daily returns
    same = sjm.build_control_returns(base, overlay_exposure, cash)
    assert np.array_equal(same.to_numpy(), overlay.to_numpy())
    assert same.name == "derisk_cash_pin"

    for alias in ("corr_overlay", "derisk-cash-pin"):
        with pytest.raises(ValueError, match="derisk_cash_pin"):
            sjm.build_control_returns(base, control_exposure, cash, control_rule=alias)


def test_overlay_and_control_share_total_return_cash(tmp_path) -> None:
    """Defect 5 boundary: both legs price residual cash from the SAME snapshot
    BIL total-return series on the Factor calendar; neither leg zero-fills."""
    calendar = pd.bdate_range("2022-01-03", periods=60)
    anchor, ridx = calendar[0], calendar[1:]
    snapshot = _write_completed_snapshot(tmp_path, calendar)
    cash, record = sjm.cash_returns_on_factor_calendar(snapshot, ridx, anchor=anchor)

    rng = np.random.default_rng(7)
    base = pd.Series(rng.normal(4e-4, 0.008, len(ridx)), index=ridx)
    overlay_exposure = pd.Series(
        np.where(np.arange(len(ridx)) % 3 == 0, 0.5, 1.0), index=ridx
    )
    control_exposure = pd.Series(
        np.where(np.arange(len(ridx)) % 4 == 0, 0.8, 1.0), index=ridx
    )

    overlay = sjm.overlay_returns(base, overlay_exposure, cash)
    control = sjm.build_control_returns(base, control_exposure, cash)

    assert record["cash_semantics"] == "adjusted_total_return"
    assert (cash.to_numpy() > 0.0).all()  # a real total-return cash line, not zeros

    for returns, exposure in ((overlay, overlay_exposure), (control, control_exposure)):
        weights = exposure.to_numpy()
        expected = weights * base.to_numpy() + (1.0 - weights) * cash.to_numpy()
        assert np.array_equal(returns.to_numpy(), expected)
        mask = weights < 1.0
        implied = (
            returns.to_numpy()[mask] - weights[mask] * base.to_numpy()[mask]
        ) / (1.0 - weights[mask])
        assert np.allclose(implied, cash.to_numpy()[mask], rtol=1e-9, atol=1e-15)


def test_overlay_equation_rejects_unaligned_nonfinite_or_unbounded_inputs() -> None:
    idx = pd.bdate_range("2023-01-02", periods=5)
    base = pd.Series(0.001, index=idx)
    cash = pd.Series(0.0002, index=idx)
    exposure = pd.Series(0.8, index=idx)

    shifted = pd.Series(0.0002, index=idx + pd.Timedelta(days=1))
    with pytest.raises(ValueError, match="identical index"):
        sjm.overlay_returns(base, exposure, shifted)
    with pytest.raises(ValueError, match="identical index"):
        sjm.build_control_returns(base, exposure, shifted)

    with pytest.raises(ValueError, match="non-finite"):
        sjm.overlay_returns(base.mask(base.index == idx[2], np.nan), exposure, cash)
    with pytest.raises(ValueError, match="non-finite"):
        sjm.overlay_returns(base, exposure, cash.mask(cash.index == idx[4], np.inf))

    with pytest.raises(ValueError, match="exposure"):
        sjm.overlay_returns(base, exposure.mask(exposure.index == idx[1], 1.2), cash)
    with pytest.raises(ValueError, match="exposure"):
        sjm.overlay_returns(base, exposure.mask(exposure.index == idx[1], -0.1), cash)


def test_drawdown_arming_is_lagged_and_signal_cadence_frozen() -> None:
    ridx = pd.bdate_range("2023-01-02", periods=8)
    vidx = ridx.insert(0, pd.Timestamp("2022-12-30"))
    returns = np.array([0.0, 0.0, -0.05, 0.0, 0.0, 0.08, 0.0, 0.0])
    value = pd.Series(np.r_[1.0, np.cumprod(1.0 + returns)], index=vidx)
    caps = pd.Series(0.5, index=ridx)

    armed = sjm.drawdown_armed_exposure(caps, value, arm=-0.03)
    # the day-t cap is decided at the t-1 close: the crash day itself is uncapped,
    # caps engage the NEXT day and disengage only the day AFTER recovery
    assert list(armed.to_numpy()) == [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0]
    assert armed.index.equals(ridx)

    unarmed = sjm.drawdown_armed_exposure(caps, value, arm=None)
    assert np.array_equal(unarmed.to_numpy(), caps.to_numpy())

    with pytest.raises(ValueError, match="anchor"):
        sjm.drawdown_armed_exposure(caps, value.iloc[1:], arm=-0.03)
    with pytest.raises(ValueError, match="negative"):
        sjm.drawdown_armed_exposure(caps, value, arm=0.03)

    protocol = sjm.make_sjm_selection_protocol()
    assert {
        name: sjm.frozen_signal_cadence(name) for name in ("absorption", "turbulence")
    } == dict(protocol.signal_step)
    with pytest.raises(ValueError, match="approved"):
        sjm.frozen_signal_cadence("volatility")


# --------------------------------------------------------------------------- #
# Ordered deterministic registry replay + development-only gates (7.5 / 7.6)   #
# --------------------------------------------------------------------------- #


_SEL_INDEX = pd.bdate_range("2019-01-02", "2024-07-31")
_SEL_ANCHOR = pd.Timestamp("2019-01-01")


def _sel_path(dev_daily: float, dip: float, holdout_daily: float = 0.001) -> pd.Series:
    """Constant returns with one development dip and a holdout-only tail."""
    protocol = sjm.make_sjm_selection_protocol()
    dev_mask = (
        (_SEL_INDEX >= protocol.dev_start) & (_SEL_INDEX <= protocol.dev_end)
    )
    values = np.where(dev_mask, dev_daily, holdout_daily).astype(float)
    values[np.flatnonzero(dev_mask)[10]] = dip
    return pd.Series(values, index=_SEL_INDEX)


def _cfg_key(config) -> tuple:
    return (config.lam, config.signal, config.window, config.scale, config.floor, config.arm)


def _sel_evaluator(paths: dict, default: pd.Series):
    """Injected candidate pricer: full-calendar returns per configuration."""

    def candidate_returns(config):
        return paths.get(_cfg_key(config), default)

    return candidate_returns


def _registry_mutation(protocol, parameter: str, value):
    return next(
        m
        for m in protocol.mutation_registry
        if (m.parameter, m.value) == (parameter, value)
    )


def test_replay_visits_frozen_registry_order_and_repeats_identically() -> None:
    """7.5: candidates only from seed+registry in exact order; reruns over identical
    inputs give the same visited order and equivalent canonical ledgers."""
    protocol = sjm.make_sjm_selection_protocol()
    factor = _sel_path(0.004, -0.019)
    control = _sel_path(0.003, -0.04)
    paths = {_cfg_key(protocol.seed_config): _sel_path(0.004, -0.02)}
    evaluator = _sel_evaluator(paths, default=_sel_path(0.004, -0.03))

    kwargs = dict(factor_returns=factor, control_returns=control)
    first = sjm.select_sjm_config(protocol, evaluator, **kwargs)
    second = sjm.select_sjm_config(protocol, evaluator, **kwargs)

    # seed first (iteration 0), then the first dry_rounds=12 registry entries in
    # exact frozen order (no-ops vs the seed -- signal=absorption, arm=None --
    # excluded); every mutation group appears and the alternate-signal slot keeps
    # its registry position
    expected = [
        (m.parameter, m.value)
        for m in protocol.mutation_registry
        if getattr(protocol.seed_config, m.parameter) != m.value
    ][:12]
    assert [(e.mutation.parameter, e.mutation.value) for e in first.ledger[1:]] == expected
    assert expected[4] == ("signal", "turbulence")
    assert [e.iteration for e in first.ledger] == list(range(13))
    assert first.ledger[0].mutation is None and first.ledger[0].decision == "KEEP"
    for entry in first.ledger[1:]:
        # every candidate, mutation, metric, and keep-or-revert decision recorded
        assert entry.candidate == sjm.apply_sjm_mutation(protocol.seed_config, entry.mutation)
        assert {"dev_calmar", "dev_cagr", "dev_maxdd"} <= set(entry.metrics)
        assert {"cagr_budget_pass", "maxdd_vs_control_pass", "passed"} <= set(entry.gates)
        assert entry.decision == "REVERT"  # nothing improves the seed's dev Calmar
    assert first.selected_config == protocol.seed_config

    assert [(e.iteration, e.mutation, e.decision) for e in second.ledger] == [
        (e.iteration, e.mutation, e.decision) for e in first.ledger
    ]
    assert second.ledger_sha256 == first.ledger_sha256
    assert first.ledger_sha256 == _sha(sjm.canonical_ledger_payload(first.ledger))
    assert second.selected_config == first.selected_config


def test_dev_gates_enforce_cagr_budget_and_control_maxdd_before_adoption() -> None:
    """7.6: development Calmar objective; the CAGR budget and the maxDD-no-worse-
    than-derisk_cash_pin gates decide adoption; retained candidates pass both."""
    protocol = sjm.make_sjm_selection_protocol()
    seed = protocol.seed_config
    factor = _sel_path(0.004, -0.019)
    control = _sel_path(0.003, -0.04)
    lam20 = _registry_mutation(protocol, "lam", 20.0)
    lam100 = _registry_mutation(protocol, "lam", 100.0)
    w126 = _registry_mutation(protocol, "window", 126)
    paths = {
        _cfg_key(seed): _sel_path(0.004, -0.02),
        # better dev Calmar but dev CAGR far below the Factor budget -> reject
        _cfg_key(sjm.apply_sjm_mutation(seed, lam20)): _sel_path(0.002, -0.005),
        # dev CAGR inside the budget but drawdown deeper than the control -> reject
        _cfg_key(sjm.apply_sjm_mutation(seed, lam100)): _sel_path(0.0055, -0.06),
        # improves dev Calmar and passes both gates -> the corrected winner
        _cfg_key(sjm.apply_sjm_mutation(seed, w126)): _sel_path(0.004, -0.01),
    }
    result = sjm.select_sjm_config(
        protocol,
        _sel_evaluator(paths, default=_sel_path(0.004, -0.03)),
        factor_returns=factor,
        control_returns=control,
    )

    seed_row, budget_row, maxdd_row, keep_row = result.ledger[:4]
    assert budget_row.metrics["dev_calmar"] > seed_row.metrics["dev_calmar"]
    assert budget_row.gates["cagr_budget_pass"] is False
    assert budget_row.decision == "REVERT"
    assert maxdd_row.gates["cagr_budget_pass"] is True
    assert maxdd_row.gates["maxdd_vs_control_pass"] is False
    assert maxdd_row.decision == "REVERT"
    assert dict(keep_row.gates) == {
        "cagr_budget_pass": True,
        "maxdd_vs_control_pass": True,
        "passed": True,
    }
    assert keep_row.decision == "KEEP"
    assert result.selected_config == sjm.apply_sjm_mutation(seed, w126)
    assert all(
        dict(e.gates)["passed"]
        for e in result.ledger
        if e.decision == "KEEP" and e.iteration > 0
    )

    # after the adoption the frozen registry restarts, in order, over the new best
    expected_post = [
        (m.parameter, m.value)
        for m in protocol.mutation_registry
        if getattr(result.selected_config, m.parameter) != m.value
    ][:12]
    assert [(e.mutation.parameter, e.mutation.value) for e in result.ledger[4:]] == expected_post
    for entry in result.ledger[4:]:
        assert entry.candidate == sjm.apply_sjm_mutation(result.selected_config, entry.mutation)
        assert entry.decision == "REVERT"
    assert len(result.ledger) == 16  # dry stop after dry_rounds=12 consecutive reverts

    # gate baselines use only the selected dev window and its local session anchor.
    dev = factor.loc[
        (factor.index >= protocol.dev_start) & (factor.index <= protocol.dev_end)
    ]
    growth = float(np.prod(1.0 + dev.to_numpy()))
    local_anchor = factor.index[factor.index.get_loc(dev.index[0]) - 1]
    years = (dev.index[-1] - local_anchor).days / 365.25
    assert result.baselines["factor_dev_cagr"] == pytest.approx(
        growth ** (1 / years) - 1, rel=1e-12
    )
    assert result.baselines["control_dev_maxdd"] == pytest.approx(-0.04, rel=1e-9)
    assert result.dev_start == protocol.dev_start
    assert result.dev_end == protocol.dev_end
    assert result.protocol_sha256 == protocol.protocol_sha256


def test_selection_never_restores_the_previous_winner_and_ignores_holdout_glory() -> None:
    """7.5: the previous winning configuration is never forced or restored -- a
    dev-inferior candidate stays rejected however spectacular its holdout is."""
    protocol = sjm.make_sjm_selection_protocol()
    seed = protocol.seed_config
    w126 = _registry_mutation(protocol, "window", 126)
    v2_direction = sjm.apply_sjm_mutation(seed, w126)  # the old shipped winner's lever
    paths = {
        _cfg_key(seed): _sel_path(0.004, -0.02, holdout_daily=0.0),
        # slightly WORSE dev Calmar, spectacular holdout (+5%/day after 2024-06-30)
        _cfg_key(v2_direction): _sel_path(0.004, -0.021, holdout_daily=0.05),
    }
    result = sjm.select_sjm_config(
        protocol,
        _sel_evaluator(paths, default=_sel_path(0.004, -0.03, holdout_daily=0.0)),
        factor_returns=_sel_path(0.004, -0.019),
        control_returns=_sel_path(0.003, -0.04),
    )

    v2_row = next(e for e in result.ledger if e.mutation == w126)
    assert dict(v2_row.gates)["passed"] is True  # gates alone would admit it
    assert v2_row.metrics["dev_calmar"] < result.ledger[0].metrics["dev_calmar"]
    assert v2_row.decision == "REVERT"  # the dev objective, not holdout glory, decides
    assert result.selected_config == seed  # nothing forces the old winner back


def test_changing_holdout_only_values_cannot_change_selection() -> None:
    """7.6 proof: rewriting observations strictly after dev_end leaves the canonical
    ledger, visited order, winner, and gate baselines identical."""
    protocol = sjm.make_sjm_selection_protocol()
    seed = protocol.seed_config
    w126 = _registry_mutation(protocol, "window", 126)
    paths = {
        _cfg_key(seed): _sel_path(0.004, -0.02),
        _cfg_key(sjm.apply_sjm_mutation(seed, w126)): _sel_path(0.004, -0.01),
    }
    default = _sel_path(0.004, -0.03)
    factor = _sel_path(0.004, -0.019)
    control = _sel_path(0.003, -0.04)

    def poison(series: pd.Series) -> pd.Series:
        out = series.copy()
        out.loc[out.index > protocol.dev_end] = -0.35  # catastrophic holdout rewrite
        return out

    baseline = sjm.select_sjm_config(
        protocol,
        _sel_evaluator(paths, default),
        factor_returns=factor,
        control_returns=control,
    )
    poisoned = sjm.select_sjm_config(
        protocol,
        _sel_evaluator({key: poison(path) for key, path in paths.items()}, poison(default)),
        factor_returns=poison(factor),
        control_returns=poison(control),
    )

    assert poisoned.ledger_sha256 == baseline.ledger_sha256
    assert poisoned.selected_config == baseline.selected_config
    assert dict(poisoned.baselines) == dict(baseline.baselines)
    assert [e.decision for e in poisoned.ledger] == [e.decision for e in baseline.ledger]


def test_selection_is_invariant_to_predev_and_holdout_perturbations() -> None:
    """Only protocol.dev_start through protocol.dev_end may affect selection."""
    protocol = sjm.make_sjm_selection_protocol()
    seed = protocol.seed_config
    winner = sjm.apply_sjm_mutation(seed, _registry_mutation(protocol, "window", 126))
    paths = {
        _cfg_key(seed): _sel_path(0.004, -0.02),
        _cfg_key(winner): _sel_path(0.004, -0.01),
    }
    default = _sel_path(0.004, -0.03)
    factor = _sel_path(0.004, -0.019)
    control = _sel_path(0.003, -0.04)

    def perturb_outside_development(series: pd.Series) -> pd.Series:
        out = series.copy()
        outside = (out.index < protocol.dev_start) | (out.index > protocol.dev_end)
        out.loc[outside] = -0.35
        return out

    baseline = sjm.select_sjm_config(
        protocol,
        _sel_evaluator(paths, default),
        factor_returns=factor,
        control_returns=control,
    )
    perturbed = sjm.select_sjm_config(
        protocol,
        _sel_evaluator(
            {key: perturb_outside_development(value) for key, value in paths.items()},
            perturb_outside_development(default),
        ),
        factor_returns=perturb_outside_development(factor),
        control_returns=perturb_outside_development(control),
    )

    assert perturbed.ledger_sha256 == baseline.ledger_sha256
    assert perturbed.selected_config == baseline.selected_config
    assert dict(perturbed.baselines) == dict(baseline.baselines)


def test_development_metrics_use_the_immediately_preceding_return_session() -> None:
    """The local session before dev_start anchors CAGR; no such session is fatal."""
    index = pd.DatetimeIndex(["2019-01-02", "2019-01-03", "2019-01-07"])
    returns = pd.Series([0.75, 0.10, 0.10], index=index)

    metrics = sjm.development_metrics(
        returns,
        dev_start=pd.Timestamp("2019-01-03"),
        dev_end=pd.Timestamp("2019-01-07"),
        name="candidate",
    )
    expected_cagr = (1.10 * 1.10) ** (365.25 / 5) - 1.0
    assert metrics["dev_n_obs"] == 2
    assert metrics["dev_cagr"] == pytest.approx(expected_cagr, rel=1e-12)

    with pytest.raises(ValueError, match="no preceding return session"):
        sjm.development_metrics(
            returns.iloc[1:],
            dev_start=pd.Timestamp("2019-01-03"),
            dev_end=pd.Timestamp("2019-01-07"),
            name="candidate",
        )


# --------------------------------------------------------------------------- #
# Immutable SJM v3 run assembly, staging, and one-validator proof (7.7 / 7.8)   #
# --------------------------------------------------------------------------- #


_SJM_RUN_FILES = (
    "sjm_targets.parquet",
    "sjm_exposures.parquet",
    "sjm_daily_returns.parquet",
    "sjm_control_returns.parquet",
    "sjm_equity.parquet",
    "sjm_ledger.json",
    "sjm_protocol.json",
)


def _run_series(ridx: pd.DatetimeIndex, dip: float, daily: float = 4e-4) -> pd.Series:
    """Constant daily returns on the REAL factor calendar with one controlled dip."""
    values = np.full(len(ridx), daily)
    values[10] = dip
    return pd.Series(values, index=ridx)


def _sjm_build_context(sjm_completed_case, output_root: Path):
    """Gated inputs + one controlled replay whose corrected winner arms at -0.02,
    plus a ``build(run_id)`` closure staging into ``output_root``."""
    factor_dir, snapshot_dir, expected = sjm_completed_case
    inputs = sjm.load_sjm_inputs(factor_dir, snapshot_dir, **expected)
    protocol = inputs.protocol
    ridx = inputs.factor_returns.index
    selection_index = ridx.insert(0, protocol.dev_start - pd.Timedelta(days=1))
    winner = sjm.apply_sjm_mutation(
        protocol.seed_config, _registry_mutation(protocol, "arm", -0.02)
    )
    paths = {
        _cfg_key(protocol.seed_config): _run_series(selection_index, -0.02),
        _cfg_key(winner): _run_series(selection_index, -0.01),
    }
    selection = sjm.select_sjm_config(
        protocol,
        _sel_evaluator(paths, default=_run_series(selection_index, -0.03)),
        factor_returns=_run_series(selection_index, -0.019),
        control_returns=_run_series(selection_index, -0.04),
    )
    assert selection.selected_config == winner  # the drawdown-armed corrected winner
    n = len(ridx)
    targets = pd.Series(
        np.where(np.arange(n) % 7 == 0, 0.6, 1.0), index=ridx
    ).to_frame("target_exposure")
    control_exposure = pd.Series(np.where(np.arange(n) % 5 == 0, 0.8, 1.0), index=ridx)

    def build(run_id: str, **overrides):
        kwargs = dict(
            run_id=run_id,
            output_root=output_root,
            targets=targets,
            control_exposure=control_exposure,
            build_time="2026-07-29T12:00:00+00:00",
        )
        kwargs.update(overrides)
        return sjm.build_sjm_run(inputs, selection, **kwargs)

    return inputs, selection, build


def _resign_sjm_run(run_dir: Path, mutate) -> None:
    """Adversary able to rewrite the manifest AND completion marker coherently:
    file hashes and the marker sha are refreshed, semantic checks must catch it."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    mutate(manifest)
    for entry in manifest["files"].values():
        entry["sha256"] = hashlib.sha256(
            (run_dir / entry["file"]).read_bytes()
        ).hexdigest()
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    sha = hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest()
    first = (run_dir / "COMPLETED").read_text().splitlines()[0]
    (run_dir / "COMPLETED").write_text(f"{first}\nmanifest_sha256={sha}\n")


def test_sjm_run_assembly_persists_the_complete_immutable_run(
    sjm_completed_case, tmp_path
) -> None:
    """7.7: selected configuration, ledger, targets, exposures, overlay returns,
    control returns, anchored equity, protocol hashes, and input manifest lineage
    all persist; returns and equity reconstruct with max abs error < 1e-9."""
    inputs, selection, build = _sjm_build_context(sjm_completed_case, tmp_path / "runs")
    run_dir = build("sjm_crowding_v3_case_assembly")
    ridx = inputs.factor_returns.index

    for fname in (*_SJM_RUN_FILES, "manifest.json", "COMPLETED"):
        assert (run_dir / fname).is_file()

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["schema"] == "sjm_run.v3"
    assert manifest["run_id"] == "sjm_crowding_v3_case_assembly"
    assert manifest["completed"] is True
    assert manifest["selected_config"]["arm"] == -0.02
    assert manifest["protocol"]["protocol_sha256"] == inputs.protocol.protocol_sha256
    assert manifest["protocol"]["dev_start"] == inputs.protocol.dev_start.date().isoformat()
    assert manifest["protocol"]["dev_end"] == inputs.protocol.dev_end.date().isoformat()
    assert manifest["protocol"]["limit_table_sha256"] == inputs.protocol.limit_table_sha256
    assert (
        manifest["protocol"]["mutation_registry_sha256"]
        == inputs.protocol.mutation_registry_sha256
    )
    assert manifest["ledger_sha256"] == selection.ledger_sha256
    assert manifest["input_manifests"]["factor_run"] == {
        "run_id": inputs.factor_run_id,
        "manifest_sha256": inputs.factor_manifest_sha256,
    }
    assert manifest["input_manifests"]["market_snapshot"] == {
        "snapshot_id": inputs.market_snapshot_id,
        "manifest_sha256": inputs.market_snapshot_sha256,
    }
    assert manifest["cash_benchmark"] == {
        "cash_benchmark_id": "BIL",
        "cash_semantics": "adjusted_total_return",
        "snapshot_id": inputs.market_snapshot_id,
    }
    assert manifest["coverage"] == {
        "anchor": inputs.factor_value.index[0].date().isoformat(),
        "start": ridx[0].date().isoformat(),
        "end": ridx[-1].date().isoformat(),
        "n_obs": len(ridx),
    }
    assert manifest["gate_results"] == {
        "cagr_budget_pass": True,
        "maxdd_vs_control_pass": True,
        "passed": True,
    }

    # completion marker (written LAST) carries the manifest sha256 (repo convention)
    marker = (run_dir / "COMPLETED").read_text()
    manifest_sha = hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest()
    assert f"manifest_sha256={manifest_sha}" in marker

    # persisted ledger IS the selection's canonical ledger
    ledger_payload = json.loads((run_dir / "sjm_ledger.json").read_text())
    assert _sha(ledger_payload) == selection.ledger_sha256
    assert ledger_payload == sjm.canonical_ledger_payload(selection.ledger)

    # independent reconstruction: exact equations and the anchored equity
    returns = pd.read_parquet(run_dir / "sjm_daily_returns.parquet")
    exposures = pd.read_parquet(run_dir / "sjm_exposures.parquet")
    control = pd.read_parquet(run_dir / "sjm_control_returns.parquet")["control_return"]
    equity = pd.read_parquet(run_dir / "sjm_equity.parquet")["value"]
    caps = pd.read_parquet(run_dir / "sjm_targets.parquet")["target_exposure"]
    assert np.array_equal(returns["factor_return"].to_numpy(), inputs.factor_returns.to_numpy())
    assert np.array_equal(returns["cash_return"].to_numpy(), inputs.cash_returns.to_numpy())
    w = exposures["overlay_exposure"].to_numpy()
    recon = w * returns["factor_return"].to_numpy() + (1.0 - w) * returns["cash_return"].to_numpy()
    assert float(np.max(np.abs(returns["daily_return"].to_numpy() - recon))) < 1e-9
    wc = exposures["control_exposure"].to_numpy()
    recon_c = wc * returns["factor_return"].to_numpy() + (1.0 - wc) * returns["cash_return"].to_numpy()
    assert float(np.max(np.abs(control.to_numpy() - recon_c))) < 1e-9
    assert equity.index[0] == inputs.factor_value.index[0] and equity.iloc[0] == 1.0
    recon_eq = np.r_[1.0, np.cumprod(1.0 + returns["daily_return"].to_numpy())]
    assert float(np.max(np.abs(equity.to_numpy() - recon_eq))) < 1e-9
    # exposures derive from the persisted targets under the SELECTED arm
    curve = pd.Series(
        np.r_[1.0, np.cumprod(1.0 + inputs.factor_returns.to_numpy())],
        index=inputs.factor_value.index,
    )
    expected_exposure = sjm.drawdown_armed_exposure(caps, curve, arm=-0.02)
    assert np.array_equal(w, expected_exposure.to_numpy())

    # the ONE validator accepts the complete run
    report = sjm.validate_sjm_run(run_dir)
    assert report["run_id"] == "sjm_crowding_v3_case_assembly"
    assert report["completed"] is True
    assert report["selected_config"] == manifest["selected_config"]
    assert report["ledger_sha256"] == selection.ledger_sha256


def test_one_validator_proves_hash_protocol_config_equation_and_reconstruction(
    sjm_completed_case, tmp_path
) -> None:
    """7.7: hash, protocol, configuration, equation, and reconstruction equality
    are all proven by validate_sjm_run; each tamper class fails deterministically."""
    _, _, build = _sjm_build_context(sjm_completed_case, tmp_path / "runs")
    run_dir = build("sjm_crowding_v3_case_validator")
    sjm.validate_sjm_run(run_dir)

    def tampered(name: str) -> Path:
        dst = tmp_path / f"tampered_{name}"
        shutil.copytree(run_dir, dst)
        return dst

    # hash: artifact bytes differing from the manifest inventory
    case = tampered("hash")
    path = case / "sjm_equity.parquet"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="sha256"):
        sjm.validate_sjm_run(case)

    # hash: manifest rewritten without the completion marker following
    case = tampered("marker")
    manifest = json.loads((case / "manifest.json").read_text())
    manifest["run_id"] = "forged_run_id"
    (case / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="COMPLETED marker"):
        sjm.validate_sjm_run(case)

    # protocol: a coherently re-signed manifest cannot smuggle a foreign protocol
    case = tampered("protocol")
    _resign_sjm_run(case, lambda m: m["protocol"].__setitem__("protocol_sha256", "0" * 64))
    with pytest.raises(ValueError, match="frozen SJM selection protocol"):
        sjm.validate_sjm_run(case)

    # protocol: a re-signed manifest must retain the declared development start.
    case = tampered("protocol_dev_start")
    _resign_sjm_run(
        case,
        lambda m: m["protocol"].__setitem__("dev_start", "2019-01-04"),
    )
    with pytest.raises(ValueError, match="protocol dev_start"):
        sjm.validate_sjm_run(case)

    # configuration: manifest selected_config must equal the ledger's last KEEP
    case = tampered("config")
    _resign_sjm_run(case, lambda m: m["selected_config"].__setitem__("window", 126))
    with pytest.raises(ValueError, match="last kept candidate"):
        sjm.validate_sjm_run(case)

    # configuration: persisted exposures must derive from targets under the arm
    case = tampered("exposure")
    frame = pd.read_parquet(case / "sjm_exposures.parquet")
    frame.iloc[3, frame.columns.get_loc("overlay_exposure")] = 0.9
    frame.to_parquet(case / "sjm_exposures.parquet")
    _resign_sjm_run(case, lambda m: None)
    with pytest.raises(ValueError, match="selected configuration"):
        sjm.validate_sjm_run(case)

    # equation: overlay daily returns must satisfy the exact overlay equation
    case = tampered("equation")
    frame = pd.read_parquet(case / "sjm_daily_returns.parquet")
    frame.iloc[5, frame.columns.get_loc("daily_return")] += 1e-6
    frame.to_parquet(case / "sjm_daily_returns.parquet")
    _resign_sjm_run(case, lambda m: None)
    with pytest.raises(ValueError, match="overlay equation"):
        sjm.validate_sjm_run(case)

    # reconstruction: anchored equity must rebuild within 1e-9
    case = tampered("equity")
    frame = pd.read_parquet(case / "sjm_equity.parquet")
    frame.iloc[-1, frame.columns.get_loc("value")] += 1e-6
    frame.to_parquet(case / "sjm_equity.parquet")
    _resign_sjm_run(case, lambda m: None)
    with pytest.raises(ValueError, match="equity"):
        sjm.validate_sjm_run(case)


def test_sjm_staging_refuses_completed_overwrite_and_incomplete_reuse(
    sjm_completed_case, tmp_path
) -> None:
    """7.8: a COMPLETED run is immutable; files from an incomplete prior attempt
    are never reused; a fresh run-specific staging directory builds cleanly."""
    root = tmp_path / "runs"
    _, _, build = _sjm_build_context(sjm_completed_case, root)
    run_dir = build("sjm_crowding_v3_case_staging")
    before = (run_dir / "manifest.json").read_bytes()

    with pytest.raises(ValueError, match="COMPLETED and immutable"):
        build("sjm_crowding_v3_case_staging")
    assert (run_dir / "manifest.json").read_bytes() == before  # untouched

    stale = root / "sjm_crowding_v3_case_stale"
    stale.mkdir(parents=True)
    (stale / "sjm_targets.parquet").write_text("stale partial artifact")
    with pytest.raises(ValueError, match="non-empty staging"):
        build("sjm_crowding_v3_case_stale")
    assert (stale / "sjm_targets.parquet").read_text() == "stale partial artifact"

    fresh = build("sjm_crowding_v3_case_fresh")
    assert sjm.validate_sjm_run(fresh)["completed"] is True


def test_wide_signal_source_manifest_blocks_panel_drift(tmp_path) -> None:
    index = pd.bdate_range("2013-01-02", periods=630)
    panel = pd.DataFrame(
        {
            "SWDA.L": 100.0 * np.cumprod(np.full(len(index), 1.0002)),
            **{
                f"ETF_{column:03d}": 50.0 * np.cumprod(
                    np.full(len(index), 1.0 + (column + 1) * 1e-6)
                )
                for column in range(99)
            },
        },
        index=index,
    )
    panel_path = tmp_path / "etf_prices_wide_2013_2026.parquet"
    panel.to_parquet(panel_path)
    source_dir = tmp_path / "signal_source"

    manifest_path = sjm.write_sjm_signal_source_manifest(
        panel_path, output_dir=source_dir, build_time="2026-07-30T12:30:00+00:00"
    )
    loaded = sjm.load_sjm_signal_source(panel_path, manifest_path=manifest_path)
    assert loaded.panel_path == panel_path
    assert loaded.manifest["schema"] == "sjm_signal_source.v1"
    assert loaded.panel.shape == (630, 100)

    panel.iloc[-1, 0] *= 1.01
    panel.to_parquet(panel_path)
    with pytest.raises(ValueError, match="does not match the panel bytes"):
        sjm.load_sjm_signal_source(panel_path, manifest_path=manifest_path)


def test_injected_late_failure_cannot_be_consumed_as_a_valid_run(
    sjm_completed_case, tmp_path, monkeypatch
) -> None:
    """7.8: run data and manifest first, validation next, COMPLETED marker LAST;
    an injected late failure leaves no marker and the attempt is unconsumable."""
    root = tmp_path / "runs"
    _, _, build = _sjm_build_context(sjm_completed_case, root)
    seen = {}

    def boom(run_dir, **kwargs):
        seen["manifest_present"] = (Path(run_dir) / "manifest.json").is_file()
        seen["marker_present"] = (Path(run_dir) / "COMPLETED").exists()
        raise RuntimeError("injected late validation failure")

    with monkeypatch.context() as m:
        m.setattr(sjm, "validate_sjm_run", boom)
        with pytest.raises(RuntimeError, match="injected late validation failure"):
            build("sjm_crowding_v3_case_failed")

    run_dir = root / "sjm_crowding_v3_case_failed"
    # validation ran AFTER data+manifest and BEFORE any marker existed
    assert seen == {"manifest_present": True, "marker_present": False}
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "sjm_daily_returns.parquet").is_file()
    assert not (run_dir / "COMPLETED").exists()
    # the failed attempt cannot be consumed as a valid run
    with pytest.raises(ValueError, match="COMPLETED marker is absent"):
        sjm.validate_sjm_run(run_dir)
    # and its files are never silently reused by a retry
    with pytest.raises(ValueError, match="non-empty staging"):
        build("sjm_crowding_v3_case_failed")
