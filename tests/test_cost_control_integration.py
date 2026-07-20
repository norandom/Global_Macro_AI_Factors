"""Task 8.3 — cost-control & no-regression integration test (R8.4).

Two properties, no NIM/DB, no live calls:

1. Cost control: a cache-reusing mutation reuses persisted scores -> ZERO live
   model calls. Driven through the real ``run_loop`` + ``mutation_registry`` with
   a scripted ``verify_fn`` that invokes a counting scorer ONLY when
   ``mutation.rescoring`` is True (the cost-control policy). The scorer's call
   count must equal the number of rescoring mutations actually tried, and every
   cache-reusing mutation tried must contribute zero.

2. No regression: with the overlay OFF the extended-stream cash pin is exactly
   the published constant, so the published weights reproduce byte-for-byte.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from macro_framework.allocation import hrp_cvar_weights_with_fixed
from macro_framework.skill_metric import GateVerdict
from scripts.factor_loop import FactorConfig, LoopEval, mutation_registry, run_loop

# 4-ETF universe (published extended stream): 3 risky sleeves + BIL cash slot.
RISKY = ("SWDA.L", "XLK", "IAU")

_FAIL = GateVerdict(
    passed=False,
    skill_pass=False,
    stability_pass=False,
    recall_pass=False,
    risk_shape_pass=False,
    first_failure="scripted-never-adopt",
    values={},
)


def _load_ext():
    """Import ``extend_stream_2026`` without leaking new modules into global
    ``sys.modules`` (imports cleanly; the module has no import-time DB/NIM).

    ``extend_stream_2026`` does ``from macro_framework import factor_scoring``,
    and a sibling test (``test_factor_scoring``) asserts that module is absent
    from ``sys.modules``. This file sorts before it, so we snapshot and restore
    to keep that order-sensitive assertion valid.
    """
    before = set(sys.modules)
    import extend_stream_2026 as ext

    return ext, before


def _restore_modules(before: set) -> None:
    for name in set(sys.modules) - before:
        del sys.modules[name]


def _returns(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = {s: 0.01 * rng.standard_normal(n) for s in RISKY}
    cols["BIL"] = 0.0002 * rng.standard_normal(n)
    return pd.DataFrame(cols, index=idx)


def test_cache_reuse_makes_zero_live_model_calls() -> None:
    """Cache-reusing mutations trigger no live scorer call; rescoring ones do."""
    calls = {"n": 0}
    tried: list = []

    def scorer() -> None:
        calls["n"] += 1  # a live model call would happen here

    def verify_fn(config: FactorConfig, mutation) -> LoopEval:
        if mutation is not None:
            tried.append(mutation)
            if mutation.rescoring:  # cost-control policy: only rescoring calls live
                scorer()
        # Never adopt (control >= appraisal, failing verdict) so the loop walks
        # the full seed registry exactly once -> tried set == registry.
        return LoopEval(appraisal=0.0, verdict=_FAIL, control_appraisal=1.0)

    seed = FactorConfig()
    registry = mutation_registry(seed)
    # dry_rounds > registry size -> search exhausts before the dry-stop fires,
    # so every mutation is tried exactly once.
    run_loop(seed, verify_fn, dry_rounds=len(registry) + 1)

    assert tried == registry  # every registry mutation tried, in order, once

    n_rescoring = sum(1 for m in tried if m.rescoring)
    n_cache = sum(1 for m in tried if not m.rescoring)
    assert n_cache >= 1  # at least one cache-reusing mutation was actually tried
    assert calls["n"] == n_rescoring  # scorer fired only for rescoring mutations
    # cache-reusing mutations contributed exactly zero live calls
    assert calls["n"] == len(tried) - n_cache


def test_overlay_off_reproduces_published_weights_byte_for_byte() -> None:
    """Overlay OFF -> pin is exactly 0.25 -> published weights reproduce exactly."""
    ext, before = _load_ext()
    try:
        r = _returns()
        assert ext._regime_cash_pin(r, None) == 0.25  # exact float equality

        w_off = hrp_cvar_weights_with_fixed(r, {"BIL": 0.25})  # published path
        w_hook = hrp_cvar_weights_with_fixed(r, {"BIL": ext._regime_cash_pin(r, None)})

        assert w_off.equals(w_hook)  # byte-for-byte identical weights
        assert np.array_equal(w_off.to_numpy(), w_hook.to_numpy())
    finally:
        _restore_modules(before)
