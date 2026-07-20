"""factor_loop — /loop driver scaffolding: config, mutation registry, single-mutation apply.

Task 6.1 scope ONLY: the immutable factor-configuration object, the ``Mutation``
contract (design.md State Management), a deterministic mutation registry over the
located levers, and ``apply_mutation`` (exactly one lever changed per iteration,
view-affecting levers bounded within the HRP/BL blend — R7.4).

The verify step (6.2), keep/revert + ledger (6.3), and regime-view execution
(6.4) are SEPARATE tasks. ``LedgerEntry`` is declared here (data shape those tasks
need) but ``run_loop`` is intentionally left unimplemented. This module imports no
DB/NIM — ``import factor_loop`` is pure.

Levers (research.md §"AI-view pipeline + mutation levers") → one config field per
``Mutation.kind`` {blend, tau, conviction, exposure, prompt, axes, overlay,
regime_view}:
  - blend      : TILT blend weight (extend_stream_2026.TILT = 0.30)  [cache-reusing]
  - tau        : BL calibration tau (allocation.bl_mv_weights, 0.05) [cache-reusing]
  - conviction : dimensionless conviction scale in [0,1]             [cache-reusing]
  - exposure   : REGIME_ASSET_EXPOSURE table variant (post-parse)    [cache-reusing]
  - overlay    : regime de-risk overlay config (None = OFF)          [cache-reusing]
  - prompt     : prompt-text variant (changes what the LLM is asked) [re-scoring]
  - axes       : MACRO_AXES variant (changes what the LLM is asked)  [re-scoring]
  - regime_view: regime-conditioned AI view influence in the blend   [re-scoring]

Cache-reusing mutations (rescoring=False) act downstream of the LLM call, so a
later task can reuse persisted scores; re-scoring mutations (rescoring=True)
change the prompt/axes/view and require live NIM calls.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, replace
from typing import Literal

# --- lever bounds ------------------------------------------------------------
# R7.4: a view-affecting lever's influence is capped so it can never become an
# unconstrained directional bet — it stays inside the HRP/BL blend.
MAX_VIEW_INFLUENCE: float = 0.50

CACHE_REUSING_KINDS: frozenset[str] = frozenset(
    {"blend", "tau", "conviction", "exposure", "overlay"}
)
RESCORING_KINDS: frozenset[str] = frozenset({"prompt", "axes", "regime_view"})
KINDS: frozenset[str] = CACHE_REUSING_KINDS | RESCORING_KINDS

# View-affecting levers are clamped into [0, MAX_VIEW_INFLUENCE]; conviction into
# [0, 1]. Everything else is passed through as given.
_VIEW_PARAMS: frozenset[str] = frozenset({"blend", "regime_view"})


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


@dataclass(frozen=True)
class FactorConfig:
    """Current best factor configuration — one tunable field per ``Mutation.kind``.

    Defaults match the published ``*_ext2026`` pipeline (overlay OFF, blend 0.30,
    tau 0.05). Frozen + plain types → serializable for the later ledger.
    """

    blend: float = 0.30  # extend_stream_2026.TILT
    tau: float = 0.05  # allocation.bl_mv_weights default
    conviction: float = 1.0  # scale on _conviction_from_loadings (1.0 = no override)
    exposure: str = "default"  # REGIME_ASSET_EXPOSURE table variant
    prompt: str = "default"  # render_regime_loadings_prompt variant
    axes: tuple[str, ...] = ("growth", "inflation", "policy", "risk")  # MACRO_AXES variant
    overlay: object | None = None  # regime de-risk overlay config; None = OFF (published)
    regime_view: float = 0.0  # regime-conditioned AI view influence; 0.0 = disabled


@dataclass(frozen=True)
class Mutation:
    """One lever change (design.md State Management contract).

    ``rescoring`` marks whether adopting this mutation needs live NIM re-scoring
    (True) or may reuse persisted scores (False).
    """

    kind: str
    param: str
    value: object
    rescoring: bool


@dataclass(frozen=True)
class LedgerEntry:
    """Data shape the verify/keep-revert task (6.3) records per iteration.

    Declared here so downstream tasks share it; not populated in this task.
    ``verdict`` is left as ``object`` to avoid importing the skill_metric gate at
    scaffolding time.
    """

    iteration: int
    mutation: Mutation
    appraisal: float | None
    verdict: object
    decision: Literal["KEEP", "REVERT"]


def config_to_dict(config: FactorConfig) -> dict:
    """JSON-serializable view of a config (for the later ledger)."""
    d = dataclasses.asdict(config)
    d["axes"] = list(config.axes)  # tuple -> list for stable JSON
    return d


def apply_mutation(config: FactorConfig, mutation: Mutation) -> FactorConfig:
    """Return a NEW config with EXACTLY the mutation's lever changed.

    Pure/deterministic. View-affecting levers (blend, regime_view) are clamped
    into ``[0, MAX_VIEW_INFLUENCE]`` and conviction into ``[0, 1]`` (R7.4) so a
    mutation can never push the AI view into an unconstrained directional bet.
    """
    if mutation.kind not in KINDS:
        raise ValueError(f"unknown mutation kind: {mutation.kind!r}")
    if mutation.param not in {f.name for f in dataclasses.fields(config)}:
        raise ValueError(f"unknown lever: {mutation.param!r}")

    value = mutation.value
    if mutation.param in _VIEW_PARAMS:
        value = _clamp(float(value), 0.0, MAX_VIEW_INFLUENCE)
    elif mutation.param == "conviction":
        value = _clamp(float(value), 0.0, 1.0)
    return replace(config, **{mutation.param: value})


def _mut(kind: str, param: str, value: object) -> Mutation:
    return Mutation(kind=kind, param=param, value=value, rescoring=kind in RESCORING_KINDS)


# Candidate values per lever. Blend/regime_view stay <= MAX_VIEW_INFLUENCE (7.4).
_CANDIDATES: dict[str, tuple[object, ...]] = {
    "blend": (0.20, 0.25, 0.35, 0.40),
    "tau": (0.025, 0.10, 0.20),
    "conviction": (0.5, 0.75),
    "exposure": ("defensive",),
    "overlay": ({"kind": "correlation", "min_scale": 0.20},),
    "prompt": ("regime_aware",),
    "axes": (("growth", "inflation", "policy", "risk", "liquidity"),),
    "regime_view": (0.15, 0.30),
}

# Deterministic lever order: cache-reusing levers first (design cost-control note),
# then re-scoring levers.
_LEVER_ORDER: tuple[str, ...] = (
    "blend",
    "tau",
    "conviction",
    "exposure",
    "overlay",
    "prompt",
    "axes",
    "regime_view",
)


def mutation_registry(config: FactorConfig) -> list[Mutation]:
    """Deterministic candidate mutations over every lever of ``config``.

    Cache-reusing mutations are ordered first. No-op candidates (value equal to
    the current config) are excluded.
    """
    out: list[Mutation] = []
    for param in _LEVER_ORDER:
        current = getattr(config, param)
        for value in _CANDIDATES[param]:
            if value == current:
                continue
            out.append(_mut(param, param, value))
    return out


# --- Task 6.2: point-in-time verify step + look-ahead guards -----------------
# Pure/testable core. Reuses the released skill_metric + ssr; NO DB/NIM here (the
# loop wires the real PIT-vs-non-PIT contrast for `recall_premium` in task 6.3).

import pandas as pd

from macro_framework.skill_metric import (
    GateConfig,
    GateVerdict,
    basket_residual,
    evaluate_gates,
)
from macro_framework.ssr import compute_ssr


class ConfigurationError(RuntimeError):
    """Fail-fast config error (mirrors ``factor_scoring.ConfigurationError``).

    Defined locally so ``import factor_loop`` stays free of recall_guard/NIM.
    """


@dataclass(frozen=True)
class VerifyResult:
    """Appraisal + composite gate verdict + injected recall premium (OOS only)."""

    appraisal: float | None
    verdict: GateVerdict
    recall_premium: float


def assert_oos_disjoint(
    oos_window: tuple, tuning_window: tuple
) -> None:
    """Raise ``ConfigurationError`` if the OOS window overlaps the tuning/cutoff
    window (R3.1). Called BEFORE any evaluation — fail-fast."""
    oos_lo, oos_hi = (pd.Timestamp(x) for x in oos_window)
    tun_lo, tun_hi = (pd.Timestamp(x) for x in tuning_window)
    if oos_lo <= tun_hi and tun_lo <= oos_hi:
        raise ConfigurationError(
            f"OOS window {oos_window} overlaps tuning/cutoff window {tuning_window}; "
            "the objective metric must be evaluated on a disjoint out-of-sample window (R3.1)."
        )


def _marker(value: object, key: str):
    """Read ``key`` from a mapping (dict) or object attribute; None if absent."""
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def check_lookahead(mutation: Mutation) -> str | None:
    """Return a look-ahead reason if a mutation would need post-decision info (R3.3).

    Concrete, testable rules:
    - any component carrying a truthy ``requires_future`` marker, or
    - a ``regime_view``/``overlay`` fit explicitly flagged non-walk-forward
      (``walk_forward=False``) — a detector refit on full history is a look-ahead
      vector as real as an LLM recalling a ticker (requirements non-negotiable).
    """
    val = mutation.value
    if _marker(val, "requires_future"):
        return f"{mutation.kind}:{mutation.param} requires post-decision (future) data"
    if mutation.kind in {"regime_view", "overlay"} and _marker(val, "walk_forward") is False:
        return f"{mutation.kind} fitted on a non-walk-forward window (uses post-decision data)"
    return None


def _lookahead_verdict(reason: str) -> GateVerdict:
    """A FAIL verdict for a look-ahead-rejected mutation — never a silent pass (R3.3)."""
    return GateVerdict(
        passed=False,
        skill_pass=False,
        stability_pass=False,
        recall_pass=False,
        risk_shape_pass=False,
        first_failure=f"lookahead: {reason}",
        values={},
    )


def verify(
    config: FactorConfig,
    oos_strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
    *,
    recall_premium: float,
    baseline_calmar: float,
    baseline_maxdd: float,
    oos_calmar: float,
    oos_maxdd: float,
    gate_config: GateConfig = GateConfig(),
    lookahead_reason: str | None = None,
) -> VerifyResult:
    """Point-in-time verify of one candidate on the OOS window only (R3, R5.2).

    Evaluated strictly on the supplied OOS series — there is NO in-sample window
    input, so nothing here can rank on in-sample Sharpe/return (R3.5). The recall
    premium (PIT-vs-non-PIT memorization delta) is injected so this stays unit-
    testable without NIM; the loop supplies the real contrast (6.3).

    If ``lookahead_reason`` is set, short-circuits to a FAIL verdict with
    ``first_failure="lookahead: <reason>"`` rather than evaluating gates (R3.3).
    """
    if lookahead_reason is not None:
        return VerifyResult(
            appraisal=None,
            verdict=_lookahead_verdict(lookahead_reason),
            recall_premium=recall_premium,
        )
    residual = basket_residual(oos_strategy_returns, factor_returns)
    ssr = compute_ssr(oos_strategy_returns)
    verdict = evaluate_gates(
        residual,
        ssr,
        recall_premium,
        oos_calmar,
        baseline_calmar,
        oos_maxdd,
        baseline_maxdd,
        config=gate_config,
    )
    return VerifyResult(appraisal=residual.appraisal, verdict=verdict, recall_premium=recall_premium)


def run_loop(config: object) -> list[LedgerEntry]:  # pragma: no cover - task 6.3
    raise NotImplementedError("keep/revert loop + ledger is task 6.3")


if __name__ == "__main__":  # pragma: no cover - stub; heavy wiring is task 6.3
    base = FactorConfig()
    print(f"default config: {config_to_dict(base)}")
    print(f"{len(mutation_registry(base))} candidate mutations "
          f"(cache-reusing first, cap={MAX_VIEW_INFLUENCE})")
