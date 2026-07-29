"""Canonical corrected deterministic SJM producer.

This module owns remediation tasks 7.2 (frozen protocol identity, immutable
configuration, canonical hashes), 7.1 (the manifest-gated SJM input set), 7.3
(exact factor-calendar BIL total returns), 7.4 (the exact overlay/control
return equations plus lagged drawdown arming and the frozen signal cadence),
7.5 (ordered deterministic mutation-registry replay with a canonical ledger),
7.6 (development-only gates and corrected winner selection), 7.7 (immutable
SJM v3 run assembly plus the one-validator proof of hash, protocol,
configuration, equation, and reconstruction equality), and 7.8 (immutable
staging and command behavior: fresh run-specific staging, no completed-run
overwrite, no reuse of incomplete attempts, COMPLETED marker written last).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

ProtocolId = Literal["sjm_selection_v2_calmar_frozen"]
Objective = Literal["dev_calmar"]
ControlRule = Literal["derisk_cash_pin"]
Signal = Literal["absorption", "turbulence"]
MutationParameter = Literal["lam", "window", "signal", "scale", "floor", "arm"]

PROTOCOL_ID: ProtocolId = "sjm_selection_v2_calmar_frozen"
CONTROL_RULE: ControlRule = "derisk_cash_pin"
CASH_BENCHMARK_ID = "BIL"
CASH_SEMANTICS = "adjusted_total_return"
#: Frozen trading-day refresh cadence of the approved crowding signals.  The
#: values are part of the hashed protocol contract (``signal_step``).
_SIGNAL_STEPS: Mapping[str, int] = MappingProxyType({"absorption": 5, "turbulence": 1})
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
        signal_step=dict(_SIGNAL_STEPS),
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


# --------------------------------------------------------------------------- #
# Task 7.1 / 7.3 / 7.4: input gate, exact cash alignment, portfolio equations   #
# --------------------------------------------------------------------------- #


def _extension_module():
    """scripts/extend_stream_2026 (Factor run + snapshot BIL contracts), lazily.

    Imported inside functions because that module pulls in
    ``macro_framework.factor_scoring``, which must stay out of ``sys.modules``
    at test collection time (the factor-scoring foundation test asserts it).
    """
    try:
        from scripts import extend_stream_2026 as ext
    except ImportError:  # scripts/ itself on sys.path (test convention)
        import extend_stream_2026 as ext
    return ext


def _snapshot_module():
    try:
        from scripts import build_basket_long as producer
    except ImportError:
        import build_basket_long as producer
    return producer


def _sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_index(index: object, name: str) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{name} must be a pandas DatetimeIndex")
    if index.empty:
        raise ValueError(f"{name} must not be empty")
    if index.tz is not None:
        raise ValueError(f"{name} must be timezone-naive")
    if index.hasnans:
        raise ValueError(f"{name} must not contain NaT labels")
    if not index.is_unique:
        raise ValueError(f"{name} must contain unique labels")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} must be strictly increasing")
    return index


def _validate_series(series: object, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise ValueError(f"{name} must be a pandas Series")
    _validate_index(series.index, f"{name}.index")
    values = series.to_numpy()
    try:
        finite = np.isfinite(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain only finite numeric values") from exc
    if not bool(finite.all()):
        label = series.index[int(np.flatnonzero(~finite)[0])]
        raise ValueError(f"{name} contains a non-finite value at {label}")
    return series


def _validate_exposure_bounds(exposure: pd.Series, name: str) -> np.ndarray:
    values = exposure.to_numpy(dtype=float)
    violations = (values < 0.0) | (values > 1.0)
    if bool(violations.any()):
        offender = exposure.index[int(np.flatnonzero(violations)[0])]
        raise ValueError(
            f"{name} must stay within the de-risk exposure bounds [0.0, 1.0]; "
            f"violation at {offender}"
        )
    return values


def frozen_signal_cadence(signal: str) -> int:
    """Trading-day refresh cadence of one approved crowding signal (frozen)."""
    name = _plain_str("signal", signal)
    if name not in _SIGNAL_STEPS:
        raise ValueError(
            f"signal must name an approved crowding signal "
            f"{tuple(_SIGNAL_STEPS)}, got {name!r}"
        )
    return _SIGNAL_STEPS[name]


def overlay_returns(
    base_returns: pd.Series,
    exposure: pd.Series,
    cash_returns: pd.Series,
) -> pd.Series:
    """Exact SJM overlay equation on one shared calendar (task 7.4, R4.1).

    ``overlay = exposure * base + (1 - exposure) * cash``: the residual
    ``1 - exposure`` sleeve earns the aligned cash TOTAL return.  Non-finite
    values, index differences, and exposure-bound violations are rejected
    before anything is compounded; nothing is filled or widened.
    """
    base = _validate_series(base_returns, "base_returns")
    exposed = _validate_series(exposure, "exposure")
    cash = _validate_series(cash_returns, "cash_returns")
    if not (base.index.equals(exposed.index) and base.index.equals(cash.index)):
        raise ValueError(
            "base_returns, exposure, and cash_returns must share one identical index"
        )
    weights = _validate_exposure_bounds(exposed, "exposure")
    values = weights * base.to_numpy(dtype=float) + (1.0 - weights) * cash.to_numpy(
        dtype=float
    )
    return pd.Series(values, index=base.index.copy(), name="sjm_overlay")


def build_control_returns(
    base_returns: pd.Series,
    exposure: pd.Series,
    cash_returns: pd.Series,
    *,
    control_rule: str = CONTROL_RULE,
) -> pd.Series:
    """``derisk_cash_pin`` control leg on the SAME cash series and timing (R4.3).

    The control is priced by the identical overlay equation, so supplying one
    shared cash vector guarantees both legs use one cash-return convention.
    A conflicting control alias is rejected instead of becoming a second
    identity for the maximum-drawdown gate.
    """
    alias = _plain_str("control_rule", control_rule)
    if alias != CONTROL_RULE:
        raise ValueError(
            "derisk_cash_pin is the authoritative control identity; "
            f"conflicting control alias {alias!r} is not allowed"
        )
    return overlay_returns(base_returns, exposure, cash_returns).rename(CONTROL_RULE)


def drawdown_armed_exposure(
    caps: pd.Series,
    portfolio_value: pd.Series,
    *,
    arm: float | None,
) -> pd.Series:
    """v2 lagged drawdown arming: the day-t cap is decided at the t-1 close.

    ``portfolio_value`` carries exactly one preceding anchor plus the caps
    calendar, so the first cap day is armed by the anchor's true (zero)
    drawdown instead of a filled placeholder.  ``arm=None`` leaves the caps
    always-on; otherwise caps engage only while the LAGGED drawdown of the
    base line is below the strictly negative ``arm`` threshold.
    """
    cap = _validate_series(caps, "caps")
    _validate_exposure_bounds(cap, "caps")
    value = _validate_series(portfolio_value, "portfolio_value")
    if len(value) != len(cap) + 1 or not value.index[1:].equals(cap.index):
        raise ValueError(
            "portfolio_value must contain exactly one preceding anchor plus the caps index"
        )
    levels = value.to_numpy(dtype=float)
    if bool((levels <= 0.0).any()):
        raise ValueError("portfolio_value must be strictly positive")
    if arm is None:
        return pd.Series(
            cap.to_numpy(dtype=float).copy(), index=cap.index.copy(), name="sjm_exposure"
        )
    threshold = _exact_float("arm", arm)
    if threshold >= 0.0:
        raise ValueError("arm must be a strictly negative drawdown threshold")
    drawdown = levels / np.maximum.accumulate(levels) - 1.0
    lagged = drawdown[:-1]  # the t-1 close decides the day-t cap
    exposed = np.where(lagged < threshold, cap.to_numpy(dtype=float), 1.0)
    return pd.Series(exposed, index=cap.index.copy(), name="sjm_exposure")


def cash_returns_on_factor_calendar(
    snapshot_dir: Path | str,
    return_index: pd.DatetimeIndex,
    *,
    anchor: pd.Timestamp,
) -> tuple[pd.Series, Mapping[str, object]]:
    """Exact factor-calendar BIL total returns plus their lineage record (7.3).

    Adjusted BIL levels are selected on the Factor calendar plus one explicit
    preceding anchor by the completed-snapshot loader; returns exist only after
    strict level alignment with filling disabled, and missing labels,
    non-finite values, duplicate dates, or an absent anchor fail instead of
    being substituted with zero (R4.2).  The returned read-only record pins the
    cash identity, snapshot identity, anchor, dates, and count.
    """
    ext = _extension_module()
    requested = _validate_index(return_index, "return_index")
    cash, record = ext.load_completed_snapshot_bil_returns(
        snapshot_dir, requested, anchor=anchor
    )
    if not cash.index.equals(requested):
        raise ValueError("cash returns must sit on exactly the Factor return index")
    _validate_series(cash, "cash_returns")
    expected_record = (
        ("cash_benchmark_id", CASH_BENCHMARK_ID),
        ("cash_semantics", CASH_SEMANTICS),
        ("cash_anchor", pd.Timestamp(anchor)),
        ("cash_start", requested[0]),
        ("cash_end", requested[-1]),
        ("cash_n_obs", len(requested)),
    )
    for key, expected in expected_record:
        if record.get(key) != expected:
            raise ValueError(
                f"cash record {key!r} must equal {expected!r}, got {record.get(key)!r}"
            )
    return cash, MappingProxyType(dict(record))


def require_snapshot_coverage(
    snapshot_manifest: Mapping[str, object],
    *,
    anchor: pd.Timestamp,
    endpoint: pd.Timestamp,
) -> None:
    """Declared snapshot coverage must span the anchor through the factor endpoint."""
    if not isinstance(snapshot_manifest, Mapping):
        raise ValueError("snapshot manifest must be a mapping")
    coverage = snapshot_manifest.get("requested_coverage")
    if not isinstance(coverage, Mapping) or not {"start", "end"} <= set(coverage):
        raise ValueError(
            "snapshot manifest must declare requested_coverage start and end"
        )
    start = _date_field("requested_coverage.start", coverage["start"])
    end = _date_field("requested_coverage.end", coverage["end"])
    anchor = pd.Timestamp(anchor)
    endpoint = pd.Timestamp(endpoint)
    if start > anchor or end < endpoint:
        raise ValueError(
            f"declared snapshot coverage {start.date()}..{end.date()} does not "
            f"include the required observations {anchor.date()}..{endpoint.date()} "
            f"through the factor endpoint"
        )


def _require_completed_run_directory(path: Path | str, name: str) -> Path:
    path = Path(path)
    if path.is_file():
        raise ValueError(
            f"{name} must be a completed immutable run directory, not a loose "
            f"legacy artifact: {path}"
        )
    if not (path / "manifest.json").is_file():
        raise ValueError(
            f"{name} has no manifest.json; loose legacy artifacts and unmanaged "
            f"directories are rejected: {path}"
        )
    return path


@dataclass(frozen=True, eq=False)
class SJMInputs:
    """One typed, validated SJM input set from completed immutable manifests."""

    factor_run_id: str
    factor_manifest_sha256: str
    market_snapshot_id: str
    market_snapshot_sha256: str
    factor_value: pd.Series
    factor_returns: pd.Series
    rebalance_dates: pd.DatetimeIndex
    cash_returns: pd.Series
    cash_record: Mapping[str, object]
    limit_table: Mapping[str, Mapping[int, float]]
    protocol: SJMSelectionProtocol

    def __post_init__(self) -> None:
        for name in (
            "factor_run_id",
            "factor_manifest_sha256",
            "market_snapshot_id",
            "market_snapshot_sha256",
        ):
            object.__setattr__(self, name, _plain_str(name, getattr(self, name)))
        for name in ("factor_manifest_sha256", "market_snapshot_sha256"):
            digest = getattr(self, name)
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        value = _validate_series(self.factor_value, "factor_value")
        if len(value) < 2:
            raise ValueError(
                "factor_value must contain an anchor plus at least one return session"
            )
        if bool((value.to_numpy(dtype=float) <= 0.0).any()):
            raise ValueError("factor_value must be strictly positive")
        returns = _validate_series(self.factor_returns, "factor_returns")
        if not returns.index.equals(value.index[1:]):
            raise ValueError(
                "factor_returns must sit on the factor_value index after its anchor"
            )
        cash = _validate_series(self.cash_returns, "cash_returns")
        if not cash.index.equals(returns.index):
            raise ValueError("cash_returns must sit on exactly the factor return index")
        rebalance = _validate_index(self.rebalance_dates, "rebalance_dates")
        if not rebalance.isin(value.index).all():
            raise ValueError("rebalance_dates must all lie on the factor calendar")
        if not isinstance(self.cash_record, Mapping):
            raise ValueError("cash_record must be a mapping")
        object.__setattr__(self, "cash_record", MappingProxyType(dict(self.cash_record)))
        object.__setattr__(self, "limit_table", _freeze_limit_table(self.limit_table))
        _exact_type("protocol", self.protocol, SJMSelectionProtocol)
        validate_sjm_selection_protocol(self.protocol)


def load_sjm_inputs(
    factor_run_dir: Path | str,
    market_snapshot_dir: Path | str,
    *,
    factor_run_id: str,
    factor_manifest_sha256: str,
    market_snapshot_id: str,
    market_snapshot_sha256: str,
) -> SJMInputs:
    """Gate every SJM input on completed immutable manifests (task 7.1).

    Selection never starts from loose legacy artifacts: both inputs must be
    completed, hash-valid run directories whose identities and manifest digests
    equal the caller's pinned expectations, and the Factor run's recorded
    market-snapshot lineage must bind to the very snapshot supplying cash.
    Every incomplete or mismatched input fails before any calculation.
    """
    for name, value in (
        ("factor_run_id", factor_run_id),
        ("factor_manifest_sha256", factor_manifest_sha256),
        ("market_snapshot_id", market_snapshot_id),
        ("market_snapshot_sha256", market_snapshot_sha256),
    ):
        _plain_str(name, value)
    run_dir = _require_completed_run_directory(factor_run_dir, "factor_run_dir")
    snapshot_dir = _require_completed_run_directory(
        market_snapshot_dir, "market_snapshot_dir"
    )

    ext = _extension_module()
    factor_manifest = ext.load_completed_factor_run(run_dir)
    if factor_manifest.get("run_id") != factor_run_id:
        raise ValueError(
            f"factor run identity mismatch: expected {factor_run_id!r}, "
            f"manifest declares {factor_manifest.get('run_id')!r}"
        )
    actual_factor_sha = _sha256_file(run_dir / "manifest.json")
    if actual_factor_sha != factor_manifest_sha256:
        raise ValueError(
            "factor run manifest sha256 mismatch: the completed bundle does not "
            "carry the pinned manifest digest"
        )

    _snapshot_module().validate_market_snapshot(snapshot_dir)
    snapshot_manifest = json.loads((snapshot_dir / "manifest.json").read_text())
    if snapshot_manifest.get("snapshot_id") != market_snapshot_id:
        raise ValueError(
            f"market snapshot identity mismatch: expected {market_snapshot_id!r}, "
            f"manifest declares {snapshot_manifest.get('snapshot_id')!r}"
        )
    actual_snapshot_sha = _sha256_file(snapshot_dir / "manifest.json")
    if actual_snapshot_sha != market_snapshot_sha256:
        raise ValueError(
            "market snapshot manifest sha256 mismatch: the completed snapshot "
            "does not carry the pinned manifest digest"
        )

    lineage = factor_manifest.get("input_manifests", {}).get("market_snapshot", {})
    if (lineage.get("snapshot_id"), lineage.get("manifest_sha256")) != (
        market_snapshot_id,
        market_snapshot_sha256,
    ):
        raise ValueError(
            "factor run market-snapshot lineage mismatch: the run was built on "
            f"{lineage.get('snapshot_id')!r}, refusing the unrelated snapshot "
            f"{market_snapshot_id!r}"
        )

    files = factor_manifest["files"]
    equity_entry = files["equity_pit"]
    equity_frame = pd.read_parquet(run_dir / equity_entry["file"])
    if "value" not in equity_frame.columns:
        raise ValueError("factor equity artifact must carry a 'value' column")
    factor_value = equity_frame["value"].rename("factor_pit")
    _validate_series(factor_value, "factor equity")
    if len(factor_value) < 2:
        raise ValueError(
            "factor equity must contain an anchor plus at least one return session"
        )
    declared = (
        int(equity_entry["rows"]),
        str(equity_entry["start"]),
        str(equity_entry["end"]),
    )
    actual = (
        len(factor_value),
        factor_value.index.min().date().isoformat(),
        factor_value.index.max().date().isoformat(),
    )
    if declared != actual:
        raise ValueError(
            f"factor equity declared coverage {declared!r} does not match its "
            f"content {actual!r}"
        )

    anchor = factor_value.index[0]
    return_index = factor_value.index[1:]
    factor_returns = factor_value.pct_change(fill_method=None).iloc[1:]

    require_snapshot_coverage(
        snapshot_manifest, anchor=anchor, endpoint=return_index[-1]
    )
    cash_returns, cash_record = cash_returns_on_factor_calendar(
        snapshot_dir, return_index, anchor=anchor
    )

    rebalance_dates = _validate_index(
        pd.read_parquet(run_dir / files["targets_pit"]["file"]).index,
        "factor targets index",
    )

    protocol = make_sjm_selection_protocol()
    return SJMInputs(
        factor_run_id=factor_run_id,
        factor_manifest_sha256=actual_factor_sha,
        market_snapshot_id=market_snapshot_id,
        market_snapshot_sha256=actual_snapshot_sha,
        factor_value=factor_value,
        factor_returns=factor_returns,
        rebalance_dates=rebalance_dates,
        cash_returns=cash_returns,
        cash_record=cash_record,
        limit_table=approved_limit_table(),
        protocol=protocol,
    )


# --------------------------------------------------------------------------- #
# Task 7.5 / 7.6: ordered deterministic registry replay + development gates     #
# --------------------------------------------------------------------------- #


def _evaluation_module():
    """``macro_framework.evaluation`` (elapsed-time CAGR / maxDD / Calmar), lazily.

    Lazy for the same reason as ``_extension_module``: importing the package
    must not happen at test collection time.
    """
    from macro_framework import evaluation

    return evaluation


def apply_sjm_mutation(config: SJMConfig, mutation: SJMMutation) -> SJMConfig:
    """Return a new configuration with EXACTLY one frozen-registry lever changed."""
    _exact_type("config", config, SJMConfig)
    _exact_type("mutation", mutation, SJMMutation)
    return replace(config, **{mutation.parameter: mutation.value})


def sjm_mutation_candidates(
    protocol: SJMSelectionProtocol, config: SJMConfig
) -> tuple[SJMMutation, ...]:
    """Frozen-registry candidates over ``config`` in EXACT registry order (7.5).

    The registry is never re-sorted, extended, or regrouped: the alternate-signal
    slot and every configured mutation group keep their frozen positions.  Only
    no-op entries (the value ``config`` already holds) are excluded, mirroring
    the shipped ``factor_loop.mutation_registry`` convention.
    """
    _exact_type("protocol", protocol, SJMSelectionProtocol)
    _exact_type("config", config, SJMConfig)
    return tuple(
        mutation
        for mutation in protocol.mutation_registry
        if getattr(config, mutation.parameter) != mutation.value
    )


def development_metrics(
    returns: pd.Series,
    *,
    anchor: pd.Timestamp,
    dev_end: pd.Timestamp,
    name: str,
) -> Mapping[str, float | int]:
    """Elapsed-time CAGR, max drawdown, and Calmar on the development window ONLY.

    Observations after ``dev_end`` are physically excluded before any
    compounding, so no holdout value can reach the objective or the gates
    (task 7.6).  The anchored value curve starts at 1.0 one session before the
    first development return.
    """
    series = _validate_series(returns, name)
    anchor_ts = _date_field("anchor", anchor)
    if anchor_ts >= series.index[0]:
        raise ValueError(f"anchor must precede the first {name} observation")
    end = _date_field("dev_end", dev_end)
    dev = series.loc[series.index <= end]
    if dev.empty:
        raise ValueError(
            f"{name} has no development observations at or before {end.date()}"
        )
    values = dev.to_numpy(dtype=float)
    if bool((values <= -1.0).any()):
        raise ValueError(f"{name} must stay above -100% per session")
    curve = pd.Series(
        np.r_[1.0, np.cumprod(1.0 + values)],
        index=dev.index.insert(0, anchor_ts),
    )
    evaluation = _evaluation_module()
    return MappingProxyType(
        {
            "dev_cagr": float(evaluation.cagr(curve)),
            "dev_maxdd": float(evaluation.max_drawdown(curve)),
            "dev_calmar": float(evaluation.calmar(curve)),
            "dev_n_obs": int(len(dev)),
        }
    )


def sjm_selection_gates(
    candidate_metrics: Mapping[str, float | int],
    *,
    factor_dev_cagr: float,
    control_dev_maxdd: float,
    protocol: SJMSelectionProtocol,
) -> Mapping[str, bool]:
    """Development-only hard gates of the frozen protocol (task 7.6, R4.3).

    ``cagr_budget_pass``: the candidate's development CAGR may trail the
    uncapped Factor line by at most the frozen ``cagr_budget``.
    ``maxdd_vs_control_pass``: its development max drawdown may not be deeper
    than the same-cash ``derisk_cash_pin`` control's.  A non-finite metric
    fails its gate instead of passing silently.
    """
    _exact_type("protocol", protocol, SJMSelectionProtocol)
    cagr_pass = bool(
        candidate_metrics["dev_cagr"] >= factor_dev_cagr - protocol.cagr_budget
    )
    maxdd_pass = bool(candidate_metrics["dev_maxdd"] >= control_dev_maxdd)
    return MappingProxyType(
        {
            "cagr_budget_pass": cagr_pass,
            "maxdd_vs_control_pass": maxdd_pass,
            "passed": cagr_pass and maxdd_pass,
        }
    )


@dataclass(frozen=True, eq=False)
class SJMLoopEntry:
    """One canonical ledger row: candidate, mutation, metrics, gates, decision."""

    iteration: int
    mutation: SJMMutation | None
    candidate: SJMConfig
    metrics: Mapping[str, float | int]
    gates: Mapping[str, bool]
    decision: Literal["KEEP", "REVERT"]

    def __post_init__(self) -> None:
        if _exact_int("iteration", self.iteration) < 0:
            raise ValueError("iteration must be >= 0")
        if self.mutation is not None:
            _exact_type("mutation", self.mutation, SJMMutation)
        _exact_type("candidate", self.candidate, SJMConfig)
        if not isinstance(self.metrics, Mapping) or not isinstance(self.gates, Mapping):
            raise ValueError("metrics and gates must be mappings")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "gates", MappingProxyType(dict(self.gates)))
        if _plain_str("decision", self.decision) not in ("KEEP", "REVERT"):
            raise ValueError("decision must be KEEP or REVERT")


def _json_metric(value: float | int) -> float | int | None:
    if type(value) is int:
        return value
    number = float(value)
    return number if math.isfinite(number) else None


def _sjm_config_payload(config: SJMConfig) -> dict[str, object]:
    return {
        "lam": config.lam,
        "signal": config.signal,
        "window": config.window,
        "scale": config.scale,
        "floor": config.floor,
        "arm": config.arm,
        "limits": _limit_payload(config.limits),
    }


def canonical_ledger_payload(
    ledger: Sequence[SJMLoopEntry],
) -> list[dict[str, object]]:
    """Canonical JSON-serializable ledger: identical inputs -> identical payload."""
    rows: list[dict[str, object]] = []
    for entry in ledger:
        _exact_type("ledger entry", entry, SJMLoopEntry)
        rows.append(
            {
                "iteration": entry.iteration,
                "mutation": (
                    None
                    if entry.mutation is None
                    else {
                        "parameter": entry.mutation.parameter,
                        "value": entry.mutation.value,
                    }
                ),
                "candidate": _sjm_config_payload(entry.candidate),
                "metrics": {
                    key: _json_metric(value) for key, value in entry.metrics.items()
                },
                "gates": {key: bool(value) for key, value in entry.gates.items()},
                "decision": entry.decision,
            }
        )
    return rows


@dataclass(frozen=True, eq=False)
class SJMSelectionResult:
    """Deterministic replay outcome: winner, canonical ledger, dev baselines."""

    selected_config: SJMConfig
    ledger: tuple[SJMLoopEntry, ...]
    ledger_sha256: str
    protocol_sha256: str
    baselines: Mapping[str, float]
    dev_end: pd.Timestamp

    def __post_init__(self) -> None:
        _exact_type("selected_config", self.selected_config, SJMConfig)
        ledger = tuple(self.ledger)
        if not ledger:
            raise ValueError("ledger must not be empty")
        for entry in ledger:
            _exact_type("ledger entry", entry, SJMLoopEntry)
        object.__setattr__(self, "ledger", ledger)
        last_keep = [entry for entry in ledger if entry.decision == "KEEP"][-1]
        if last_keep.candidate != self.selected_config:
            raise ValueError("selected_config must equal the last kept candidate")
        for name in ("ledger_sha256", "protocol_sha256"):
            digest = _plain_str(name, getattr(self, name))
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not isinstance(self.baselines, Mapping):
            raise ValueError("baselines must be a mapping")
        object.__setattr__(self, "baselines", MappingProxyType(dict(self.baselines)))
        object.__setattr__(self, "dev_end", _date_field("dev_end", self.dev_end))


def select_sjm_config(
    protocol: SJMSelectionProtocol,
    candidate_returns,  # Callable[[SJMConfig], pd.Series] on the factor calendar
    *,
    factor_returns: pd.Series,
    control_returns: pd.Series,
    anchor: pd.Timestamp,
) -> SJMSelectionResult:
    """Replay the frozen ordered mutation registry and pick the gated winner.

    Candidates arise ONLY from the frozen seed configuration plus the ordered
    mutation registry (task 7.5): the previous winning configuration is never
    forced, seeded, or restored.  Every candidate is priced on the full factor
    calendar but evaluated exclusively through ``protocol.dev_end`` on the
    development Calmar objective under the CAGR-budget and
    max-drawdown-no-worse-than-``derisk_cash_pin`` gates (task 7.6): holdout
    observations are physically excluded until the final configuration is
    fixed, so changing holdout-only values cannot change selection.  A mutated
    candidate replaces the incumbent iff it strictly improves the development
    Calmar AND passes both gates; otherwise the incumbent is kept (REVERT).
    After an adoption the registry restarts in exact frozen order over the new
    best (the shipped ``factor_loop.run_loop`` convention).  The search stops
    after ``dry_rounds`` consecutive reverts, on registry exhaustion, or at the
    ``max_iters`` backstop.  Identical inputs therefore produce the same
    visited candidate order and equivalent canonical ledgers.
    """
    validate_sjm_selection_protocol(protocol)
    if not callable(candidate_returns):
        raise ValueError("candidate_returns must be callable")
    factor = _validate_series(factor_returns, "factor_returns")
    control = _validate_series(control_returns, "control_returns")
    if not factor.index.equals(control.index):
        raise ValueError(
            "factor_returns and control_returns must share one identical index"
        )

    dev_end = protocol.dev_end
    factor_dev = development_metrics(
        factor, anchor=anchor, dev_end=dev_end, name="factor_returns"
    )
    control_dev = development_metrics(
        control, anchor=anchor, dev_end=dev_end, name="control_returns"
    )
    baselines = {
        "factor_dev_cagr": factor_dev["dev_cagr"],
        "control_dev_maxdd": control_dev["dev_maxdd"],
    }

    def _evaluate(config: SJMConfig):
        series = _validate_series(candidate_returns(config), "candidate returns")
        if not series.index.equals(factor.index):
            raise ValueError(
                "candidate returns must sit on exactly the factor return index"
            )
        metrics = development_metrics(
            series, anchor=anchor, dev_end=dev_end, name="candidate returns"
        )
        gates = sjm_selection_gates(
            metrics,
            factor_dev_cagr=baselines["factor_dev_cagr"],
            control_dev_maxdd=baselines["control_dev_maxdd"],
            protocol=protocol,
        )
        return metrics, gates

    best = protocol.seed_config
    metrics, gates = _evaluate(best)
    best_calmar = float(metrics["dev_calmar"])
    ledger = [SJMLoopEntry(0, None, best, metrics, gates, "KEEP")]

    pending = list(sjm_mutation_candidates(protocol, best))
    consecutive_reverts = 0
    iteration = 1
    while (
        pending
        and consecutive_reverts < protocol.dry_rounds
        and iteration <= protocol.max_iters
    ):
        mutation = pending.pop(0)
        candidate = apply_sjm_mutation(best, mutation)
        metrics, gates = _evaluate(candidate)
        candidate_calmar = float(metrics["dev_calmar"])
        improves = math.isfinite(candidate_calmar) and (
            not math.isfinite(best_calmar) or candidate_calmar > best_calmar
        )
        if improves and gates["passed"]:
            best = candidate
            best_calmar = candidate_calmar
            consecutive_reverts = 0
            pending = list(sjm_mutation_candidates(protocol, best))
            decision: Literal["KEEP", "REVERT"] = "KEEP"
        else:
            consecutive_reverts += 1
            decision = "REVERT"
        ledger.append(
            SJMLoopEntry(iteration, mutation, candidate, metrics, gates, decision)
        )
        iteration += 1

    entries = tuple(ledger)
    return SJMSelectionResult(
        selected_config=best,
        ledger=entries,
        ledger_sha256=_canonical_sha256(canonical_ledger_payload(entries)),
        protocol_sha256=protocol.protocol_sha256,
        baselines=baselines,
        dev_end=dev_end,
    )


# --------------------------------------------------------------------------- #
# Task 7.7 / 7.8: immutable SJM v3 run assembly, staging, and one validator     #
# --------------------------------------------------------------------------- #


SJM_RUN_MANIFEST_SCHEMA = "sjm_run.v3"
#: Task 7.7 reconstruction contract: persisted returns and equity must rebuild
#: from the persisted inputs with maximum absolute error strictly below this.
RECONSTRUCTION_TOLERANCE = 1e-9

_SJM_RUN_FILES: Mapping[str, str] = MappingProxyType(
    {
        "targets": "sjm_targets.parquet",
        "exposures": "sjm_exposures.parquet",
        "daily_returns": "sjm_daily_returns.parquet",
        "control_returns": "sjm_control_returns.parquet",
        "equity": "sjm_equity.parquet",
        "ledger": "sjm_ledger.json",
        "protocol": "sjm_protocol.json",
    }
)
_SJM_RUN_COLUMNS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "targets": ("target_exposure",),
        "exposures": ("overlay_exposure", "control_exposure"),
        "daily_returns": ("daily_return", "factor_return", "cash_return"),
        "control_returns": ("control_return",),
        "equity": ("value",),
    }
)


def _anchored_value_curve(returns: pd.Series, anchor: pd.Timestamp) -> pd.Series:
    """Value curve starting at exactly 1.0 on the anchor session."""
    return pd.Series(
        np.r_[1.0, np.cumprod(1.0 + returns.to_numpy(dtype=float))],
        index=returns.index.insert(0, anchor),
    )


def _config_from_payload(payload: object) -> SJMConfig:
    """Rebuild (and thereby re-validate) an SJMConfig from its JSON payload."""
    if not isinstance(payload, Mapping):
        raise ValueError("selected_config payload must be a mapping")
    limits_payload = payload.get("limits")
    if not isinstance(limits_payload, Mapping):
        raise ValueError("selected_config payload must carry the limit table")
    try:
        limits = {
            regime: {int(bucket): float(value) for bucket, value in row.items()}
            for regime, row in limits_payload.items()
        }
        arm = payload["arm"]
        return SJMConfig(
            lam=float(payload["lam"]),
            signal=payload["signal"],
            window=int(payload["window"]),
            scale=float(payload["scale"]),
            floor=float(payload["floor"]),
            arm=None if arm is None else float(arm),
            limits=limits,
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"selected_config payload is malformed: {exc}") from exc


def _normalize_build_time(build_time: str | None) -> str:
    if build_time is None:
        return pd.Timestamp.now("UTC").isoformat()
    try:
        parsed = pd.Timestamp(build_time)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"build_time must be an ISO-8601 timestamp, got {build_time!r}"
        ) from exc
    if parsed.tz is None:
        raise ValueError(f"build_time must be timezone-aware, got {build_time!r}")
    return parsed.tz_convert("UTC").isoformat()


def build_sjm_run(
    inputs: SJMInputs,
    selection: SJMSelectionResult,
    *,
    run_id: str,
    output_root: Path | str,
    targets: pd.DataFrame,
    control_exposure: pd.Series,
    build_time: str | None = None,
) -> Path:
    """Assemble ONE immutable SJM v3 run; COMPLETED is written LAST (7.7/7.8).

    Inputs are the exact manifest-gated ``SJMInputs`` (task 7.1: pinned Factor
    and market-snapshot identities/digests) and the deterministic replay
    outcome under the same frozen protocol.  Staging follows the repo
    convention (market snapshot tasks 5.3/5.4, factor bundle 6.9/6.10): the
    destination must be a NEW empty run-specific staging directory; a
    COMPLETED destination is immutable and never overwritten; files from an
    incomplete prior attempt are never reused; run data and the manifest are
    written first, the whole staged run is validated by ``validate_sjm_run``,
    and only then does the completion marker (carrying the manifest sha256)
    exist.  A failed build leaves staging dirty WITHOUT ``COMPLETED`` for
    diagnosis; recovery is delete-and-rebuild.
    """
    _exact_type("inputs", inputs, SJMInputs)
    _exact_type("selection", selection, SJMSelectionResult)
    if type(run_id) is not str or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    protocol = inputs.protocol
    if selection.protocol_sha256 != protocol.protocol_sha256:
        raise ValueError(
            "selection protocol_sha256 does not match the gated inputs' frozen protocol"
        )
    if selection.dev_end != protocol.dev_end:
        raise ValueError("selection dev_end does not match the frozen protocol")

    return_index = inputs.factor_returns.index
    anchor = inputs.factor_value.index[0]
    selected = selection.selected_config
    if not isinstance(targets, pd.DataFrame) or list(targets.columns) != [
        "target_exposure"
    ]:
        raise ValueError(
            "targets must be a DataFrame with exactly one target_exposure column"
        )
    caps = _validate_series(targets["target_exposure"], "targets.target_exposure")
    if not caps.index.equals(return_index):
        raise ValueError("targets must sit on exactly the factor return index")
    _validate_exposure_bounds(caps, "targets.target_exposure")
    if bool((caps.to_numpy(dtype=float) < selected.floor).any()):
        raise ValueError(
            "targets breach the selected configuration's floor "
            f"{selected.floor!r}"
        )
    control_caps = _validate_series(control_exposure, "control_exposure")
    if not control_caps.index.equals(return_index):
        raise ValueError("control_exposure must sit on exactly the factor return index")
    _validate_exposure_bounds(control_caps, "control_exposure")
    build_time = _normalize_build_time(build_time)

    run_dir = Path(output_root) / run_id
    if run_dir.exists():
        if (run_dir / "COMPLETED").exists():
            raise ValueError(
                f"SJM run {run_id!r} is COMPLETED and immutable; runs are append-only"
            )
        if any(run_dir.iterdir()):
            raise ValueError(
                f"refusing to write into non-empty staging directory {run_dir}; "
                "files from an incomplete prior attempt are never reused"
            )
    run_dir.mkdir(parents=True, exist_ok=True)

    value_curve = _anchored_value_curve(inputs.factor_returns, anchor)
    exposures = drawdown_armed_exposure(caps, value_curve, arm=selected.arm)
    overlay = overlay_returns(inputs.factor_returns, exposures, inputs.cash_returns)
    control = build_control_returns(
        inputs.factor_returns, control_caps, inputs.cash_returns
    )
    equity = _anchored_value_curve(overlay, anchor)

    frames: dict[str, pd.DataFrame] = {
        "targets": caps.to_frame("target_exposure"),
        "exposures": pd.DataFrame(
            {"overlay_exposure": exposures, "control_exposure": control_caps}
        ),
        "daily_returns": pd.DataFrame(
            {
                "daily_return": overlay,
                "factor_return": inputs.factor_returns,
                "cash_return": inputs.cash_returns,
            }
        ),
        "control_returns": control.to_frame("control_return"),
        "equity": equity.to_frame("value"),
    }
    for role, frame in frames.items():
        frame.to_parquet(run_dir / _SJM_RUN_FILES[role])

    ledger_payload = canonical_ledger_payload(selection.ledger)
    (run_dir / _SJM_RUN_FILES["ledger"]).write_text(
        json.dumps(ledger_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    protocol_payload = dict(_protocol_payload(protocol))
    protocol_payload["limit_table_sha256"] = protocol.limit_table_sha256
    protocol_payload["mutation_registry_sha256"] = protocol.mutation_registry_sha256
    protocol_payload["protocol_sha256"] = protocol.protocol_sha256
    (run_dir / _SJM_RUN_FILES["protocol"]).write_text(
        json.dumps(protocol_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    files: dict[str, dict[str, object]] = {}
    for role, fname in _SJM_RUN_FILES.items():
        dest = run_dir / fname
        entry: dict[str, object] = {
            "file": fname,
            "sha256": _sha256_file(dest),
            "size": int(dest.stat().st_size),
        }
        if role in frames:
            frame = frames[role]
            entry["rows"] = int(len(frame))
            entry["start"] = frame.index[0].date().isoformat()
            entry["end"] = frame.index[-1].date().isoformat()
        elif role == "ledger":
            entry["rows"] = int(len(ledger_payload))
        files[role] = entry

    last_keep = [entry for entry in selection.ledger if entry.decision == "KEEP"][-1]
    manifest = {
        "schema": SJM_RUN_MANIFEST_SCHEMA,
        "run_id": run_id,
        "build_time": build_time,
        "cash_benchmark": {
            "cash_benchmark_id": CASH_BENCHMARK_ID,
            "cash_semantics": CASH_SEMANTICS,
            "snapshot_id": inputs.market_snapshot_id,
        },
        "coverage": {
            "anchor": anchor.date().isoformat(),
            "start": return_index[0].date().isoformat(),
            "end": return_index[-1].date().isoformat(),
            "n_obs": int(len(return_index)),
        },
        "protocol": {
            "protocol_id": protocol.protocol_id,
            "objective": protocol.objective,
            "control_rule": protocol.control_rule,
            "seed": protocol.seed,
            "dev_end": protocol.dev_end.date().isoformat(),
            "protocol_sha256": protocol.protocol_sha256,
            "limit_table_sha256": protocol.limit_table_sha256,
            "mutation_registry_sha256": protocol.mutation_registry_sha256,
        },
        "selected_config": _sjm_config_payload(selected),
        "gate_results": {key: bool(value) for key, value in last_keep.gates.items()},
        "baselines": {key: float(value) for key, value in selection.baselines.items()},
        "ledger_sha256": selection.ledger_sha256,
        "input_manifests": {
            "factor_run": {
                "run_id": inputs.factor_run_id,
                "manifest_sha256": inputs.factor_manifest_sha256,
            },
            "market_snapshot": {
                "snapshot_id": inputs.market_snapshot_id,
                "manifest_sha256": inputs.market_snapshot_sha256,
            },
        },
        "files": files,
        "completed": True,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    # every hash, protocol, configuration, equation, and reconstruction check
    # must pass BEFORE the completion marker exists (marker is written LAST)
    validate_sjm_run(run_dir, require_completed=False)
    manifest_sha = _sha256_file(run_dir / "manifest.json")
    (run_dir / "COMPLETED").write_text(
        f"{build_time}\nmanifest_sha256={manifest_sha}\n"
    )
    return run_dir


def validate_sjm_run(
    run_dir: Path | str, *, require_completed: bool = True
) -> dict[str, object]:
    """ONE validator across the complete persisted SJM run (task 7.7).

    Proves, in one pass: completion-marker and per-file hash integrity, frozen
    protocol identity (all three canonical digests plus the persisted protocol
    payload), selected-configuration equality between the manifest and the
    persisted ledger's last kept candidate, the exact overlay/control
    equations, and reconstruction of the persisted exposures, returns, and
    anchored equity with maximum absolute error below ``1e-9``.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{run_dir}: manifest.json is missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SJM_RUN_MANIFEST_SCHEMA:
        raise ValueError(
            f"{run_dir}: unknown manifest schema {manifest.get('schema')!r}"
        )
    if manifest.get("completed") is not True:
        raise ValueError(f"{run_dir}: manifest must declare completed=true")

    marker_path = run_dir / "COMPLETED"
    if require_completed and not marker_path.is_file():
        raise ValueError(
            f"{run_dir}: COMPLETED marker is absent; SJM run is incomplete"
        )
    if marker_path.is_file():
        manifest_sha = _sha256_file(manifest_path)
        if f"manifest_sha256={manifest_sha}" not in marker_path.read_text():
            raise ValueError(
                f"{run_dir}: COMPLETED marker does not match manifest bytes"
            )

    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(_SJM_RUN_FILES):
        raise ValueError(
            f"{run_dir}: manifest files inventory must cover exactly "
            f"{sorted(_SJM_RUN_FILES)}"
        )
    for role, fname in _SJM_RUN_FILES.items():
        entry = files[role]
        if not isinstance(entry, Mapping) or entry.get("file") != fname:
            raise ValueError(f"{run_dir}: files[{role!r}] must record {fname!r}")
        path = run_dir / fname
        if not path.is_file():
            raise ValueError(f"{run_dir}: {fname} is missing from disk")
        if _sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"{run_dir}: {fname} sha256 mismatch against the manifest")

    # frozen protocol identity: canonical digests plus the persisted payload
    frozen = make_sjm_selection_protocol()
    declared = manifest.get("protocol")
    if not isinstance(declared, Mapping):
        raise ValueError(f"{run_dir}: manifest must declare the protocol section")
    for name, expected in (
        ("protocol_id", frozen.protocol_id),
        ("objective", frozen.objective),
        ("control_rule", frozen.control_rule),
        ("seed", frozen.seed),
        ("dev_end", frozen.dev_end.date().isoformat()),
        ("protocol_sha256", frozen.protocol_sha256),
        ("limit_table_sha256", frozen.limit_table_sha256),
        ("mutation_registry_sha256", frozen.mutation_registry_sha256),
    ):
        if declared.get(name) != expected:
            raise ValueError(
                f"{run_dir}: protocol {name} {declared.get(name)!r} does not "
                f"match the frozen SJM selection protocol {expected!r}"
            )
    persisted_protocol = json.loads((run_dir / _SJM_RUN_FILES["protocol"]).read_text())
    if not isinstance(persisted_protocol, dict):
        raise ValueError(f"{run_dir}: persisted protocol payload must be a mapping")
    digest_fields = {
        name: persisted_protocol.pop(name, None)
        for name in ("protocol_sha256", "limit_table_sha256", "mutation_registry_sha256")
    }
    if persisted_protocol != _protocol_payload(frozen) or digest_fields != {
        "protocol_sha256": frozen.protocol_sha256,
        "limit_table_sha256": frozen.limit_table_sha256,
        "mutation_registry_sha256": frozen.mutation_registry_sha256,
    }:
        raise ValueError(
            f"{run_dir}: persisted protocol payload does not match the frozen "
            "SJM selection protocol"
        )

    # ledger hash + selected-configuration provenance
    ledger_payload = json.loads((run_dir / _SJM_RUN_FILES["ledger"]).read_text())
    if not isinstance(ledger_payload, list) or not ledger_payload:
        raise ValueError(f"{run_dir}: persisted ledger must be a non-empty list")
    if _canonical_sha256(ledger_payload) != manifest.get("ledger_sha256"):
        raise ValueError(
            f"{run_dir}: ledger_sha256 does not match the persisted canonical ledger"
        )
    keeps = [
        row
        for row in ledger_payload
        if isinstance(row, Mapping) and row.get("decision") == "KEEP"
    ]
    if not keeps:
        raise ValueError(f"{run_dir}: persisted ledger has no kept candidate")
    last_keep = keeps[-1]
    if last_keep.get("candidate") != manifest.get("selected_config"):
        raise ValueError(
            f"{run_dir}: manifest selected_config does not equal the ledger's "
            "last kept candidate configuration"
        )
    gates = last_keep.get("gates")
    if not isinstance(gates, Mapping) or {
        key: bool(value) for key, value in gates.items()
    } != manifest.get("gate_results"):
        raise ValueError(
            f"{run_dir}: gate_results do not match the ledger's last kept candidate"
        )
    selected = _config_from_payload(manifest.get("selected_config"))
    if _canonical_sha256(_limit_payload(selected.limits)) != frozen.limit_table_sha256:
        raise ValueError(
            f"{run_dir}: selected configuration limit table does not match the "
            "frozen limit table"
        )

    # persisted data: exact columns, declared coverage, one shared calendar
    frames = {
        role: pd.read_parquet(run_dir / _SJM_RUN_FILES[role])
        for role in _SJM_RUN_COLUMNS
    }
    for role, frame in frames.items():
        expected_columns = list(_SJM_RUN_COLUMNS[role])
        if list(frame.columns) != expected_columns:
            raise ValueError(
                f"{run_dir}: {_SJM_RUN_FILES[role]} must carry exactly columns "
                f"{expected_columns}"
            )
        entry = files[role]
        actual = (
            len(frame),
            frame.index[0].date().isoformat(),
            frame.index[-1].date().isoformat(),
        )
        declared_cov = (entry.get("rows"), entry.get("start"), entry.get("end"))
        if declared_cov != actual:
            raise ValueError(
                f"{run_dir}: {_SJM_RUN_FILES[role]} declared coverage "
                f"{declared_cov!r} does not match its content {actual!r}"
            )
    if files["ledger"].get("rows") != len(ledger_payload):
        raise ValueError(
            f"{run_dir}: ledger declared rows do not match the persisted ledger"
        )

    returns = frames["daily_returns"]
    return_index = _validate_index(returns.index, "daily_returns index")
    for role in ("targets", "exposures", "control_returns"):
        if not frames[role].index.equals(return_index):
            raise ValueError(
                f"{run_dir}: {_SJM_RUN_FILES[role]} must sit on exactly the "
                "persisted return index"
            )
    equity = _validate_series(frames["equity"]["value"], "equity")
    if len(equity) != len(return_index) + 1 or not equity.index[1:].equals(return_index):
        raise ValueError(
            f"{run_dir}: equity must carry exactly one anchor plus the return index"
        )
    anchor = equity.index[0]
    coverage = manifest.get("coverage")
    actual_cov = {
        "anchor": anchor.date().isoformat(),
        "start": return_index[0].date().isoformat(),
        "end": return_index[-1].date().isoformat(),
        "n_obs": len(return_index),
    }
    if not isinstance(coverage, Mapping) or dict(coverage) != actual_cov:
        raise ValueError(
            f"{run_dir}: manifest coverage {coverage!r} does not match the run "
            f"content {actual_cov!r}"
        )

    # cash-benchmark identity and input manifest lineage
    cash_decl = manifest.get("cash_benchmark")
    lineage = manifest.get("input_manifests")
    if not isinstance(cash_decl, Mapping) or not isinstance(lineage, Mapping):
        raise ValueError(
            f"{run_dir}: manifest must declare cash_benchmark and input_manifests"
        )
    factor_lineage = lineage.get("factor_run")
    snapshot_lineage = lineage.get("market_snapshot")
    for lineage_name, entry, id_key in (
        ("factor_run", factor_lineage, "run_id"),
        ("market_snapshot", snapshot_lineage, "snapshot_id"),
    ):
        identity = entry.get(id_key) if isinstance(entry, Mapping) else None
        digest = entry.get("manifest_sha256") if isinstance(entry, Mapping) else None
        if type(identity) is not str or not identity:
            raise ValueError(
                f"{run_dir}: input manifest lineage {lineage_name} must pin {id_key}"
            )
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(
                f"{run_dir}: input manifest lineage {lineage_name} must pin a "
                "manifest sha256"
            )
    if (
        cash_decl.get("cash_benchmark_id") != CASH_BENCHMARK_ID
        or cash_decl.get("cash_semantics") != CASH_SEMANTICS
        or cash_decl.get("snapshot_id") != snapshot_lineage.get("snapshot_id")
    ):
        raise ValueError(
            f"{run_dir}: cash benchmark must be {CASH_BENCHMARK_ID} "
            f"{CASH_SEMANTICS} from the lineage market snapshot"
        )

    # equation + reconstruction equality (max abs error strictly below 1e-9)
    caps = _validate_series(
        frames["targets"]["target_exposure"], "targets.target_exposure"
    )
    _validate_exposure_bounds(caps, "targets.target_exposure")
    if bool((caps.to_numpy(dtype=float) < selected.floor).any()):
        raise ValueError(
            f"{run_dir}: persisted targets breach the selected configuration's floor"
        )
    factor = _validate_series(returns["factor_return"], "factor_return")
    cash = _validate_series(returns["cash_return"], "cash_return")
    overlay = _validate_series(returns["daily_return"], "daily_return")
    control = _validate_series(
        frames["control_returns"]["control_return"], "control_return"
    )
    overlay_exposure = _validate_series(
        frames["exposures"]["overlay_exposure"], "overlay_exposure"
    )
    control_exposure = _validate_series(
        frames["exposures"]["control_exposure"], "control_exposure"
    )
    _validate_exposure_bounds(overlay_exposure, "overlay_exposure")
    _validate_exposure_bounds(control_exposure, "control_exposure")

    value_curve = _anchored_value_curve(factor, anchor)
    err = float(
        np.max(
            np.abs(
                overlay_exposure.to_numpy()
                - drawdown_armed_exposure(caps, value_curve, arm=selected.arm).to_numpy()
            )
        )
    )
    if err >= RECONSTRUCTION_TOLERANCE:
        raise ValueError(
            f"{run_dir}: persisted exposures do not reconstruct from the "
            f"persisted targets under the selected configuration "
            f"(max abs error {err:.3e})"
        )
    err = float(
        np.max(
            np.abs(
                overlay.to_numpy()
                - overlay_returns(factor, overlay_exposure, cash).to_numpy()
            )
        )
    )
    if err >= RECONSTRUCTION_TOLERANCE:
        raise ValueError(
            f"{run_dir}: persisted daily returns break the exact overlay equation "
            f"(max abs error {err:.3e})"
        )
    err = float(
        np.max(
            np.abs(
                control.to_numpy()
                - build_control_returns(factor, control_exposure, cash).to_numpy()
            )
        )
    )
    if err >= RECONSTRUCTION_TOLERANCE:
        raise ValueError(
            f"{run_dir}: persisted control returns break the exact shared-cash "
            f"control equation (max abs error {err:.3e})"
        )
    if equity.iloc[0] != 1.0:
        raise ValueError(f"{run_dir}: anchored equity must start at exactly 1.0")
    err = float(
        np.max(np.abs(equity.to_numpy() - _anchored_value_curve(overlay, anchor).to_numpy()))
    )
    if err >= RECONSTRUCTION_TOLERANCE:
        raise ValueError(
            f"{run_dir}: persisted equity does not reconstruct from the anchored "
            f"overlay returns (max abs error {err:.3e})"
        )

    return {
        "run_id": manifest.get("run_id"),
        "schema": manifest["schema"],
        "selected_config": manifest["selected_config"],
        "protocol_sha256": frozen.protocol_sha256,
        "ledger_sha256": manifest["ledger_sha256"],
        "coverage": dict(coverage),
        "input_manifests": {
            "factor_run": dict(factor_lineage),
            "market_snapshot": dict(snapshot_lineage),
        },
        "completed": marker_path.is_file(),
    }


__all__ = [
    "CASH_BENCHMARK_ID",
    "CASH_SEMANTICS",
    "RECONSTRUCTION_TOLERANCE",
    "SJM_RUN_MANIFEST_SCHEMA",
    "SJMConfig",
    "SJMInputs",
    "SJMLoopEntry",
    "SJMMutation",
    "SJMSelectionProtocol",
    "SJMSelectionResult",
    "apply_sjm_mutation",
    "approved_limit_table",
    "approved_mutation_registry",
    "build_control_returns",
    "build_sjm_run",
    "canonical_ledger_payload",
    "cash_returns_on_factor_calendar",
    "development_metrics",
    "drawdown_armed_exposure",
    "frozen_signal_cadence",
    "load_sjm_inputs",
    "make_sjm_selection_protocol",
    "overlay_returns",
    "require_snapshot_coverage",
    "select_sjm_config",
    "sjm_mutation_candidates",
    "sjm_selection_gates",
    "validate_sjm_run",
    "validate_sjm_selection_protocol",
]
