"""Canonical frozen protocol for the corrected deterministic SJM producer.

This module currently owns only remediation task 7.2: protocol identity, immutable
configuration, and canonical hashes.  It deliberately does not load inputs or replay
the selection loop; those are later SJM producer tasks.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

import pandas as pd

ProtocolId = Literal["sjm_selection_v2_calmar_frozen"]
Objective = Literal["dev_calmar"]
ControlRule = Literal["derisk_cash_pin"]
Signal = Literal["absorption", "turbulence"]
MutationParameter = Literal["lam", "window", "signal", "scale", "floor", "arm"]

PROTOCOL_ID: ProtocolId = "sjm_selection_v2_calmar_frozen"
CONTROL_RULE: ControlRule = "derisk_cash_pin"
DEV_START = pd.Timestamp("2019-01-03")
DEV_END = pd.Timestamp("2024-06-30")
HOLDOUT_START = pd.Timestamp("2024-07-01")
OBJECTIVE: Objective = "dev_calmar"
CAGR_BUDGET = 0.035
SEED = 42
DRY_ROUNDS = 12
MAX_ITERS = 60

_LAMBDA_CANDIDATES = (20.0, 100.0)
_WINDOW_CANDIDATES = (126, 504)
_SIGNAL_CANDIDATES: tuple[Signal, ...] = ("turbulence", "absorption")
_SCALE_CANDIDATES = (0.9, 1.1, 1.2, 1.3)
_FLOOR_CANDIDATES = (0.5, 0.3)
_ARM_CANDIDATES = (None, -0.02, -0.03, -0.04, -0.05)
_REGIMES = ("bull", "neutral", "bear")
_BUCKETS = (0, 1, 2)
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))

_APPROVED_LIMIT_ROWS = (
    ("bull", (1.0, 0.9, 0.8)),
    ("neutral", (1.0, 0.9, 0.8)),
    ("bear", (1.0, 0.8, 0.5)),
)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _date_field(name: str, value: object) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid date") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid date")
    if timestamp.tz is not None:
        raise ValueError(f"{name} must be timezone-naive")
    if timestamp != timestamp.normalize():
        raise ValueError(f"{name} must be a date at midnight")
    return timestamp


def _exact_float(name: str, value: object) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return value


def _exact_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _plain_str(name: str, value: object) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be a plain str")
    return value


def _exact_type(name: str, value: object, expected: type[object]) -> None:
    if type(value) is not expected:
        raise ValueError(f"{name} must be an exact {expected.__name__}")


def _freeze_limit_table(
    limits: Mapping[str, Mapping[int, float]],
) -> Mapping[str, Mapping[int, float]]:
    if not isinstance(limits, Mapping):
        raise ValueError("limit table must be a mapping")
    for regime in limits:
        _plain_str("limit table regime", regime)
    if set(limits) != set(_REGIMES):
        raise ValueError(
            "limit table must contain exactly bull, neutral, and bear rows"
        )

    frozen_rows: dict[str, Mapping[int, float]] = {}
    for regime in _REGIMES:
        row = limits[regime]
        if not isinstance(row, Mapping):
            raise ValueError(f"limit table row {regime!r} must be a mapping")
        for bucket in row:
            _exact_int(f"limit table row {regime!r} bucket", bucket)
        if set(row) != set(_BUCKETS):
            raise ValueError(
                f"limit table row {regime!r} must contain exactly buckets 0, 1, and 2"
            )
        normalized: dict[int, float] = {}
        for bucket in _BUCKETS:
            normalized[bucket] = _exact_float(
                f"limit table {regime}[{bucket}]",
                row[bucket],
            )
        frozen_rows[regime] = MappingProxyType(normalized)
    return MappingProxyType(frozen_rows)


def _limit_payload(
    limits: Mapping[str, Mapping[int, float]],
) -> dict[str, dict[str, float]]:
    return {
        regime: {str(bucket): float(limits[regime][bucket]) for bucket in _BUCKETS}
        for regime in _REGIMES
    }


def approved_limit_table() -> Mapping[str, Mapping[int, float]]:
    """Return a fresh immutable copy of the approved nine-cell cap table."""
    return _freeze_limit_table(
        {
            regime: {bucket: values[bucket] for bucket in _BUCKETS}
            for regime, values in _APPROVED_LIMIT_ROWS
        }
    )


@dataclass(frozen=True)
class SJMMutation:
    """One entry in the fully ordered SJM mutation registry."""

    parameter: MutationParameter
    value: float | int | Signal | None

    def __post_init__(self) -> None:
        parameter = _plain_str("mutation parameter", self.parameter)
        object.__setattr__(self, "parameter", parameter)
        if parameter not in {"lam", "window", "signal", "scale", "floor", "arm"}:
            raise ValueError(f"unsupported mutation parameter {parameter!r}")
        value = self.value
        if parameter == "window":
            object.__setattr__(self, "value", _exact_int("window mutation", value))
        elif parameter == "signal":
            signal = _plain_str("signal mutation", value)
            if signal not in ("absorption", "turbulence"):
                raise ValueError(
                    f"signal mutation must name an approved signal, got {signal!r}"
                )
            object.__setattr__(self, "value", signal)
        elif parameter == "arm" and value is None:
            return
        else:
            object.__setattr__(
                self,
                "value",
                _exact_float(f"{parameter} mutation", value),
            )


def approved_mutation_registry() -> tuple[SJMMutation, ...]:
    """Return the complete approved mutation registry in replay order."""
    return (
        *(SJMMutation("lam", value) for value in _LAMBDA_CANDIDATES),
        *(SJMMutation("window", value) for value in _WINDOW_CANDIDATES),
        *(SJMMutation("signal", value) for value in _SIGNAL_CANDIDATES),
        *(SJMMutation("scale", value) for value in _SCALE_CANDIDATES),
        *(SJMMutation("floor", value) for value in _FLOOR_CANDIDATES),
        *(SJMMutation("arm", value) for value in _ARM_CANDIDATES),
    )


def _mutation_payload(
    registry: Sequence[SJMMutation],
) -> list[dict[str, float | int | str | None]]:
    return [
        {"parameter": mutation.parameter, "value": mutation.value}
        for mutation in registry
    ]


@dataclass(frozen=True)
class SJMConfig:
    """Immutable SJM seed/configuration value used by the frozen protocol."""

    lam: float
    signal: Signal
    window: int
    scale: float
    floor: float
    arm: float | None
    limits: Mapping[str, Mapping[int, float]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "lam", _exact_float("lam", self.lam))
        signal = _plain_str("signal", self.signal)
        if signal not in ("absorption", "turbulence"):
            raise ValueError(f"signal must be absorption or turbulence, got {signal!r}")
        object.__setattr__(self, "signal", signal)
        object.__setattr__(self, "window", _exact_int("window", self.window))
        object.__setattr__(self, "scale", _exact_float("scale", self.scale))
        object.__setattr__(self, "floor", _exact_float("floor", self.floor))
        if self.arm is not None:
            object.__setattr__(self, "arm", _exact_float("arm", self.arm))
        object.__setattr__(self, "limits", _freeze_limit_table(self.limits))


def _approved_seed_config(
    limits: Mapping[str, Mapping[int, float]] | None = None,
) -> SJMConfig:
    return SJMConfig(
        lam=50.0,
        signal="absorption",
        window=252,
        scale=1.0,
        floor=0.4,
        arm=None,
        limits=approved_limit_table() if limits is None else limits,
    )


@dataclass(frozen=True)
class SJMSelectionProtocol:
    """Approved SJM v2 search contract plus its canonical integrity hashes."""

    protocol_id: ProtocolId
    dev_start: pd.Timestamp
    dev_end: pd.Timestamp
    holdout_start: pd.Timestamp
    objective: Objective
    cagr_budget: float
    control_rule: ControlRule
    require_maxdd_not_worse_than_control: Literal[True]
    seed: int
    dry_rounds: int
    max_iters: int
    signal_step: Mapping[str, int]
    seed_config: SJMConfig
    lambda_candidates: tuple[float, ...]
    window_candidates: tuple[int, ...]
    signal_candidates: tuple[Signal, ...]
    scale_candidates: tuple[float, ...]
    floor_candidates: tuple[float, ...]
    arm_candidates: tuple[float | None, ...]
    mutation_registry: tuple[SJMMutation, ...]
    limit_table_sha256: str = field(init=False)
    mutation_registry_sha256: str = field(init=False)
    protocol_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("protocol_id", "objective", "control_rule"):
            object.__setattr__(self, name, _plain_str(name, getattr(self, name)))
        if type(self.require_maxdd_not_worse_than_control) is not bool:
            raise ValueError("require_maxdd_not_worse_than_control must be a bool")
        _exact_type("seed_config", self.seed_config, SJMConfig)
        for name in ("dev_start", "dev_end", "holdout_start"):
            object.__setattr__(self, name, _date_field(name, getattr(self, name)))
        object.__setattr__(
            self,
            "cagr_budget",
            _exact_float("cagr_budget", self.cagr_budget),
        )
        for name in ("seed", "dry_rounds", "max_iters"):
            object.__setattr__(self, name, _exact_int(name, getattr(self, name)))

        if not isinstance(self.signal_step, Mapping):
            raise ValueError("signal_step must be a mapping")
        for name in self.signal_step:
            _plain_str("signal_step key", name)
        if set(self.signal_step) != {"absorption", "turbulence"}:
            raise ValueError("signal_step must contain absorption and turbulence")
        steps = {
            name: _exact_int(f"signal_step[{name!r}]", self.signal_step[name])
            for name in ("absorption", "turbulence")
        }
        object.__setattr__(self, "signal_step", MappingProxyType(steps))

        object.__setattr__(
            self,
            "lambda_candidates",
            tuple(_exact_float("lambda_candidates", value) for value in self.lambda_candidates),
        )
        object.__setattr__(
            self,
            "window_candidates",
            tuple(_exact_int("window_candidates", value) for value in self.window_candidates),
        )
        object.__setattr__(
            self,
            "signal_candidates",
            tuple(
                _plain_str("signal_candidates item", value)
                for value in self.signal_candidates
            ),
        )
        object.__setattr__(
            self,
            "scale_candidates",
            tuple(_exact_float("scale_candidates", value) for value in self.scale_candidates),
        )
        object.__setattr__(
            self,
            "floor_candidates",
            tuple(_exact_float("floor_candidates", value) for value in self.floor_candidates),
        )
        arms: list[float | None] = []
        for value in self.arm_candidates:
            arms.append(None if value is None else _exact_float("arm_candidates", value))
        object.__setattr__(self, "arm_candidates", tuple(arms))
        registry = tuple(self.mutation_registry)
        for index, mutation in enumerate(registry):
            _exact_type(f"mutation_registry[{index}]", mutation, SJMMutation)
        object.__setattr__(self, "mutation_registry", registry)

        _validate_approved_values(self)
        limit_hash = _canonical_sha256(_limit_payload(self.seed_config.limits))
        mutation_hash = _canonical_sha256(_mutation_payload(self.mutation_registry))
        object.__setattr__(self, "limit_table_sha256", limit_hash)
        object.__setattr__(self, "mutation_registry_sha256", mutation_hash)
        object.__setattr__(
            self,
            "protocol_sha256",
            _canonical_sha256(_protocol_payload(self)),
        )

    def fingerprint(self) -> str:
        """Canonical SHA-256 of all frozen fields, limits, and ordered mutations."""
        return self.protocol_sha256


def _validate_seed_config_structure(seed: SJMConfig) -> None:
    _exact_type("seed_config", seed, SJMConfig)
    if type(seed.lam) is not float:
        raise ValueError("seed_config.lam must be a finite float")
    _plain_str("seed_config.signal", seed.signal)
    if type(seed.window) is not int:
        raise ValueError("seed_config.window must be an integer")
    for name in ("scale", "floor"):
        if type(getattr(seed, name)) is not float:
            raise ValueError(f"seed_config.{name} must be a finite float")
    if seed.arm is not None and type(seed.arm) is not float:
        raise ValueError("seed_config.arm must be None or a finite float")
    if type(seed.limits) is not _MAPPING_PROXY_TYPE:
        raise ValueError("seed_config.limits must be the read-only limit table")
    limit_keys = []
    for regime, row in seed.limits.items():
        limit_keys.append(_plain_str("seed_config.limits regime", regime))
        if type(row) is not _MAPPING_PROXY_TYPE:
            raise ValueError("seed_config.limits rows must be read-only mappings")
        bucket_keys = []
        for bucket, value in row.items():
            bucket_keys.append(_exact_int("seed_config.limits bucket", bucket))
            if type(value) is not float:
                raise ValueError("seed_config.limits values must be floats")
        if set(bucket_keys) != set(_BUCKETS):
            raise ValueError("seed_config.limits rows must contain buckets 0, 1, and 2")
    if set(limit_keys) != set(_REGIMES):
        raise ValueError("seed_config.limits must contain the approved regimes")


def _validate_protocol_structure(protocol: SJMSelectionProtocol) -> None:
    _exact_type("protocol", protocol, SJMSelectionProtocol)
    for name in ("protocol_id", "objective", "control_rule"):
        _plain_str(name, getattr(protocol, name))
    for name in ("dev_start", "dev_end", "holdout_start"):
        if type(getattr(protocol, name)) is not pd.Timestamp:
            raise ValueError(f"{name} must be an exact pandas Timestamp")
    if type(protocol.cagr_budget) is not float:
        raise ValueError("cagr_budget must be a finite float")
    if type(protocol.require_maxdd_not_worse_than_control) is not bool:
        raise ValueError("require_maxdd_not_worse_than_control must be a bool")
    for name in ("seed", "dry_rounds", "max_iters"):
        if type(getattr(protocol, name)) is not int:
            raise ValueError(f"{name} must be an integer")
    if type(protocol.signal_step) is not _MAPPING_PROXY_TYPE:
        raise ValueError("signal_step must be a read-only mapping")
    step_keys = []
    for name, value in protocol.signal_step.items():
        step_keys.append(_plain_str("signal_step key", name))
        if type(value) is not int:
            raise ValueError(f"signal_step[{name!r}] must be an integer")
    if set(step_keys) != {"absorption", "turbulence"}:
        raise ValueError("signal_step must contain absorption and turbulence")

    _validate_seed_config_structure(protocol.seed_config)
    candidate_types = {
        "lambda_candidates": float,
        "window_candidates": int,
        "signal_candidates": str,
        "scale_candidates": float,
        "floor_candidates": float,
    }
    for name, item_type in candidate_types.items():
        candidates = getattr(protocol, name)
        if type(candidates) is not tuple:
            raise ValueError(f"{name} must be a tuple")
        for index, value in enumerate(candidates):
            if type(value) is not item_type:
                expected = "plain str" if item_type is str else item_type.__name__
                raise ValueError(f"{name}[{index}] must be a {expected}")
    if type(protocol.arm_candidates) is not tuple:
        raise ValueError("arm_candidates must be a tuple")
    for index, value in enumerate(protocol.arm_candidates):
        if value is not None and type(value) is not float:
            raise ValueError(f"arm_candidates[{index}] must be None or a float")
    if type(protocol.mutation_registry) is not tuple:
        raise ValueError("mutation_registry must be a tuple")
    for index, mutation in enumerate(protocol.mutation_registry):
        _exact_type(f"mutation_registry[{index}]", mutation, SJMMutation)
        _plain_str(f"mutation_registry[{index}].parameter", mutation.parameter)
        parameter = mutation.parameter
        value = mutation.value
        if parameter == "window":
            _exact_int(f"mutation_registry[{index}].value", value)
        elif parameter == "signal":
            _plain_str(f"mutation_registry[{index}].value", value)
        elif parameter == "arm" and value is None:
            continue
        else:
            _exact_float(f"mutation_registry[{index}].value", value)


def _validate_approved_values(protocol: SJMSelectionProtocol) -> None:
    _validate_protocol_structure(protocol)
    approved_scalars = {
        "protocol_id": PROTOCOL_ID,
        "dev_start": DEV_START,
        "dev_end": DEV_END,
        "holdout_start": HOLDOUT_START,
        "objective": OBJECTIVE,
        "cagr_budget": CAGR_BUDGET,
        "seed": SEED,
        "dry_rounds": DRY_ROUNDS,
        "max_iters": MAX_ITERS,
    }
    for name, expected in approved_scalars.items():
        actual = getattr(protocol, name)
        if actual != expected:
            raise ValueError(f"{name} must remain frozen at {expected!r}, got {actual!r}")

    if protocol.control_rule != CONTROL_RULE:
        raise ValueError(
            "derisk_cash_pin is the authoritative control identity; "
            f"conflicting control alias {protocol.control_rule!r} is not allowed"
        )
    if protocol.require_maxdd_not_worse_than_control is not True:
        raise ValueError("require_maxdd_not_worse_than_control must remain True")
    if dict(protocol.signal_step) != {"absorption": 5, "turbulence": 1}:
        raise ValueError("signal_step must remain absorption=5, turbulence=1")

    seed = protocol.seed_config
    expected_seed = (50.0, "absorption", 252, 1.0, 0.4, None)
    actual_seed = (seed.lam, seed.signal, seed.window, seed.scale, seed.floor, seed.arm)
    if actual_seed != expected_seed:
        raise ValueError(
            f"seed_config must remain frozen at {expected_seed!r}, got {actual_seed!r}"
        )
    if _limit_payload(seed.limits) != _limit_payload(approved_limit_table()):
        raise ValueError("limit table differs from the approved read-only SJM table")

    approved_candidates = {
        "lambda_candidates": _LAMBDA_CANDIDATES,
        "window_candidates": _WINDOW_CANDIDATES,
        "signal_candidates": _SIGNAL_CANDIDATES,
        "scale_candidates": _SCALE_CANDIDATES,
        "floor_candidates": _FLOOR_CANDIDATES,
        "arm_candidates": _ARM_CANDIDATES,
    }
    for name, expected in approved_candidates.items():
        actual = getattr(protocol, name)
        if actual != expected:
            raise ValueError(
                f"{name} must retain approved values and order {expected!r}, got {actual!r}"
            )

    expected_registry = approved_mutation_registry()
    if protocol.mutation_registry != expected_registry:
        raise ValueError(
            "mutation_registry must retain the complete approved candidate order"
        )


def _protocol_payload(protocol: SJMSelectionProtocol) -> dict[str, object]:
    limits = _limit_payload(protocol.seed_config.limits)
    return {
        "protocol_id": protocol.protocol_id,
        "dev_start": protocol.dev_start.date().isoformat(),
        "dev_end": protocol.dev_end.date().isoformat(),
        "holdout_start": protocol.holdout_start.date().isoformat(),
        "objective": protocol.objective,
        "cagr_budget": protocol.cagr_budget,
        "control_rule": protocol.control_rule,
        "require_maxdd_not_worse_than_control": (
            protocol.require_maxdd_not_worse_than_control
        ),
        "seed": protocol.seed,
        "dry_rounds": protocol.dry_rounds,
        "max_iters": protocol.max_iters,
        "signal_step": dict(protocol.signal_step),
        "seed_config": {
            "lam": protocol.seed_config.lam,
            "signal": protocol.seed_config.signal,
            "window": protocol.seed_config.window,
            "scale": protocol.seed_config.scale,
            "floor": protocol.seed_config.floor,
            "arm": protocol.seed_config.arm,
            "limits": limits,
        },
        "lambda_candidates": list(protocol.lambda_candidates),
        "window_candidates": list(protocol.window_candidates),
        "signal_candidates": list(protocol.signal_candidates),
        "scale_candidates": list(protocol.scale_candidates),
        "floor_candidates": list(protocol.floor_candidates),
        "arm_candidates": list(protocol.arm_candidates),
        "limit_table": limits,
        "mutation_registry": _mutation_payload(protocol.mutation_registry),
    }


def _validate_control_alias(control_alias: str | None) -> None:
    if control_alias is None:
        return
    alias = _plain_str("control_alias", control_alias)
    if alias != CONTROL_RULE:
        raise ValueError(
            "derisk_cash_pin is the authoritative control identity; "
            f"conflicting control alias {alias!r} is not allowed"
        )


def make_sjm_selection_protocol(
    limit_table: Mapping[str, Mapping[int, float]] | None = None,
    *,
    mutation_registry: Sequence[SJMMutation] | None = None,
    control_alias: str | None = None,
) -> SJMSelectionProtocol:
    """Construct the one approved frozen protocol.

    Mapping insertion order is irrelevant, but values and mutation order are part of
    the contract.  A conflicting control alias is rejected instead of being retained
    as a second identity for the maximum-drawdown gate.
    """
    _validate_control_alias(control_alias)
    limits = approved_limit_table() if limit_table is None else limit_table
    registry = (
        approved_mutation_registry()
        if mutation_registry is None
        else tuple(mutation_registry)
    )
    return SJMSelectionProtocol(
        protocol_id=PROTOCOL_ID,
        dev_start=DEV_START,
        dev_end=DEV_END,
        holdout_start=HOLDOUT_START,
        objective=OBJECTIVE,
        cagr_budget=CAGR_BUDGET,
        control_rule=CONTROL_RULE,
        require_maxdd_not_worse_than_control=True,
        seed=SEED,
        dry_rounds=DRY_ROUNDS,
        max_iters=MAX_ITERS,
        signal_step={"absorption": 5, "turbulence": 1},
        seed_config=_approved_seed_config(limits),
        lambda_candidates=_LAMBDA_CANDIDATES,
        window_candidates=_WINDOW_CANDIDATES,
        signal_candidates=_SIGNAL_CANDIDATES,
        scale_candidates=_SCALE_CANDIDATES,
        floor_candidates=_FLOOR_CANDIDATES,
        arm_candidates=_ARM_CANDIDATES,
        mutation_registry=registry,
    )


def validate_sjm_selection_protocol(
    protocol: SJMSelectionProtocol,
    *,
    control_alias: str | None = None,
) -> None:
    """Recompute every protocol digest and reject drift or alias conflicts."""
    _validate_control_alias(control_alias)
    _validate_approved_values(protocol)
    for name in (
        "limit_table_sha256",
        "mutation_registry_sha256",
        "protocol_sha256",
    ):
        digest = _plain_str(name, getattr(protocol, name))
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    expected_limit = _canonical_sha256(_limit_payload(protocol.seed_config.limits))
    if protocol.limit_table_sha256 != expected_limit:
        raise ValueError("limit_table_sha256 does not match the read-only limit table")
    expected_registry = _canonical_sha256(_mutation_payload(protocol.mutation_registry))
    if protocol.mutation_registry_sha256 != expected_registry:
        raise ValueError(
            "mutation_registry_sha256 does not match the fully ordered mutation registry"
        )
    expected_protocol = _canonical_sha256(_protocol_payload(protocol))
    if protocol.protocol_sha256 != expected_protocol:
        raise ValueError("protocol_sha256 does not match the canonical frozen protocol")


__all__ = [
    "SJMConfig",
    "SJMMutation",
    "SJMSelectionProtocol",
    "approved_limit_table",
    "approved_mutation_registry",
    "make_sjm_selection_protocol",
    "validate_sjm_selection_protocol",
]
