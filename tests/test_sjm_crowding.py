"""Frozen SJM selection protocol boundary (remediation task 7.2).

These tests are offline and intentionally stop before selection replay.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

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
