"""Tests for scripts/factor_loop — mutation registry + single-mutation application (task 6.1).

Pure/deterministic: NO DB, NO NIM. Only the config dataclasses, the mutation
registry, and apply_mutation are exercised here. Verify/keep-revert/ledger and
regime-view execution are separate tasks (6.2-6.4).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import factor_loop as fl  # noqa: E402

CACHE_REUSING = {"blend", "tau", "conviction", "exposure", "overlay"}
RESCORING = {"prompt", "axes", "regime_view"}


def test_default_config_matches_published_pipeline():
    c = fl.FactorConfig()
    assert c.blend == pytest.approx(0.30)  # nb09 final blend 0.7 HRP + 0.3 BL
    assert c.tau == pytest.approx(0.05)  # allocation.bl_mv_weights default
    assert c.overlay is None  # published *_ext2026 has overlay OFF
    assert 0.0 <= c.conviction <= 1.0


def test_config_is_frozen_and_serializable():
    c = fl.FactorConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.blend = 0.5  # type: ignore[misc]
    # serializable for the later ledger (plain JSON-able types)
    import json

    json.dumps(fl.config_to_dict(c))


def test_apply_mutation_changes_exactly_one_lever():
    base = fl.FactorConfig()
    m = fl.Mutation(kind="tau", param="tau", value=0.10, rescoring=False)
    out = fl.apply_mutation(base, m)
    assert out.tau == pytest.approx(0.10)
    # every OTHER field is identical
    for f in dataclasses.fields(base):
        if f.name == "tau":
            continue
        assert getattr(out, f.name) == getattr(base, f.name), f.name


def test_apply_mutation_is_pure_and_deterministic():
    base = fl.FactorConfig()
    m = fl.Mutation(kind="blend", param="blend", value=0.35, rescoring=False)
    a = fl.apply_mutation(base, m)
    b = fl.apply_mutation(base, m)
    assert a == b
    assert base.blend == pytest.approx(0.30)  # base untouched


def test_registry_rescoring_flags_partition_correctly():
    for m in fl.mutation_registry(fl.FactorConfig()):
        assert isinstance(m.rescoring, bool)
        if m.kind in CACHE_REUSING:
            assert m.rescoring is False, m
        elif m.kind in RESCORING:
            assert m.rescoring is True, m
        else:
            pytest.fail(f"unknown kind {m.kind}")


def test_registry_orders_cache_reusing_first():
    kinds = [m.kind for m in fl.mutation_registry(fl.FactorConfig())]
    rescoring_flags = [k in RESCORING for k in kinds]
    # once we hit the first re-scoring mutation, everything after is re-scoring
    first_rescore = rescoring_flags.index(True) if True in rescoring_flags else len(kinds)
    assert all(rescoring_flags[first_rescore:]), kinds
    assert not any(rescoring_flags[:first_rescore]), kinds


def test_registry_is_deterministic():
    a = fl.mutation_registry(fl.FactorConfig())
    b = fl.mutation_registry(fl.FactorConfig())
    assert a == b


def test_registry_excludes_noops():
    c = fl.FactorConfig()
    for m in fl.mutation_registry(c):
        assert getattr(c, m.param) != m.value, f"no-op mutation {m}"


def test_registry_param_equals_field_name():
    c = fl.FactorConfig()
    names = {f.name for f in dataclasses.fields(c)}
    for m in fl.mutation_registry(c):
        assert m.param in names, m


def test_blend_mutation_clamped_into_valid_range():
    base = fl.FactorConfig()
    hi = fl.apply_mutation(base, fl.Mutation("blend", "blend", 1.5, False))
    assert hi.blend == pytest.approx(fl.MAX_VIEW_INFLUENCE)  # cannot exceed configured max
    lo = fl.apply_mutation(base, fl.Mutation("blend", "blend", -0.4, False))
    assert lo.blend == pytest.approx(0.0)


def test_view_mutation_cannot_become_unconstrained_bet():
    base = fl.FactorConfig()
    out = fl.apply_mutation(base, fl.Mutation("regime_view", "regime_view", 0.99, True))
    assert out.regime_view <= fl.MAX_VIEW_INFLUENCE  # bounded within HRP/BL blend (7.4)


def test_conviction_clamped_to_unit_interval():
    base = fl.FactorConfig()
    assert fl.apply_mutation(base, fl.Mutation("conviction", "conviction", 5.0, False)).conviction == 1.0
    assert fl.apply_mutation(base, fl.Mutation("conviction", "conviction", -1.0, False)).conviction == 0.0


def test_apply_mutation_rejects_unknown_kind():
    with pytest.raises(ValueError):
        fl.apply_mutation(fl.FactorConfig(), fl.Mutation("bogus", "blend", 0.4, False))


# --- Task 6.2: point-in-time verify step + look-ahead guards ------------------
# Pure/synthetic, still NO DB/NIM. The recall premium is injected (the loop wires
# the real PIT-vs-non-PIT contrast later).

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _synth_oos(seed: int = 7, n: int = 400):
    """OOS strategy returns (stable positive drift) + own-4-ETF factor frame.

    Stable drift -> high SSR; tiny residual noise -> defined appraisal + huge HAC t.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    fac = pd.DataFrame(
        rng.normal(0, 0.01, (n, 4)), index=idx, columns=["SWDA", "XLK", "IAU", "BIL"]
    )
    strat = pd.Series(0.0005 + rng.normal(0, 0.0001, n), index=idx)
    return strat, fac


def test_assert_oos_disjoint_rejects_overlap():
    # OOS [2024-07..2025-12] overlaps tuning [2016-01..2024-12]
    with pytest.raises(fl.ConfigurationError):
        fl.assert_oos_disjoint(("2024-07-01", "2025-12-31"), ("2016-01-01", "2024-12-31"))


def test_assert_oos_disjoint_accepts_disjoint():
    assert fl.assert_oos_disjoint(("2025-01-01", "2025-12-31"), ("2016-01-01", "2024-12-31")) is None


def test_check_lookahead_flags_future_data_marker():
    m = fl.Mutation("regime_view", "regime_view", {"requires_future": True}, True)
    reason = fl.check_lookahead(m)
    assert reason is not None and "future" in reason.lower()


def test_check_lookahead_flags_non_walk_forward_fit():
    m = fl.Mutation("overlay", "overlay", {"kind": "correlation", "walk_forward": False}, False)
    assert fl.check_lookahead(m) is not None


def test_check_lookahead_passes_normal_cache_reusing_mutation():
    assert fl.check_lookahead(fl.Mutation("tau", "tau", 0.10, False)) is None


def test_verify_passes_on_clean_oos_window():
    strat, fac = _synth_oos()
    res = fl.verify(
        fl.FactorConfig(), strat, fac,
        recall_premium=0.0,
        baseline_calmar=1.0, baseline_maxdd=-0.10,
        oos_calmar=2.0, oos_maxdd=-0.10,
    )
    assert res.verdict.passed is True
    assert res.verdict.first_failure is None
    assert res.appraisal is not None and res.appraisal > 0
    assert res.recall_premium == 0.0


def test_verify_recall_gate_fails_on_large_premium():
    strat, fac = _synth_oos()
    res = fl.verify(
        fl.FactorConfig(), strat, fac,
        recall_premium=0.9,  # memorization premium far from zero
        baseline_calmar=1.0, baseline_maxdd=-0.10,
        oos_calmar=2.0, oos_maxdd=-0.10,
    )
    assert res.verdict.passed is False
    assert res.verdict.first_failure.startswith("recall:")


def test_verify_lookahead_returns_fail_never_silent_pass():
    strat, fac = _synth_oos()
    res = fl.verify(
        fl.FactorConfig(), strat, fac,
        recall_premium=0.0,
        baseline_calmar=1.0, baseline_maxdd=-0.10,
        oos_calmar=2.0, oos_maxdd=-0.10,
        lookahead_reason="regime detector refit on full history",
    )
    assert res.verdict.passed is False
    assert res.verdict.first_failure.startswith("lookahead:")


def test_verify_signature_has_no_in_sample_metric_input():
    # R3.5: selection MUST NOT use in-sample Sharpe/return. Guard against a future
    # regression re-introducing an in-sample lever into the verify surface.
    import inspect

    params = set(inspect.signature(fl.verify).parameters)
    assert not any("in_sample" in p or "insample" in p for p in params), params


# --- Task 6.3: keep/revert + control gate + loop-until-dry + ledger -----------
# Pure/deterministic: verify_fn is INJECTED (scripted evals), so NO DB/NIM.

from macro_framework.skill_metric import GateVerdict  # noqa: E402


def _pass_verdict() -> GateVerdict:
    return GateVerdict(True, True, True, True, True, None, {})


def _fail_verdict(reason: str = "skill: t=0.1 < 2") -> GateVerdict:
    return GateVerdict(False, False, True, True, True, reason, {})


def _scripted_verify(script):
    """Build a verify_fn keyed by (param, value); mutation=None -> baseline entry.

    `script[key]` -> (appraisal, verdict, control_appraisal). Baseline key is None.
    """
    def verify_fn(config, mutation):
        key = None if mutation is None else (mutation.param, mutation.value)
        appraisal, verdict, control = script[key]
        return fl.LoopEval(appraisal=appraisal, verdict=verdict, control_appraisal=control)
    return verify_fn


def _single_mutation_registry(mutations):
    """registry_fn over a fixed candidate list, excluding no-ops vs the given config
    (mimics the real ``mutation_registry`` so a KEEP does not re-offer itself)."""
    return lambda config: [m for m in mutations if getattr(config, m.param) != m.value]


def test_loop_records_seeded_baseline_as_iteration_zero():
    seed = fl.FactorConfig()
    verify_fn = _scripted_verify({None: (0.5, _pass_verdict(), 0.2)})
    ledger = fl.run_loop(seed, verify_fn, dry_rounds=1, registry_fn=_single_mutation_registry([]))
    assert len(ledger) == 1
    base = ledger[0]
    assert base.iteration == 0
    assert base.mutation is None  # baseline has no mutation
    assert base.appraisal == pytest.approx(0.5)
    assert base.decision == "KEEP"  # seeded starting best


def test_loop_keeps_on_improved_gated_beats_control():
    seed = fl.FactorConfig()
    m = fl.Mutation("tau", "tau", 0.10, False)
    verify_fn = _scripted_verify({
        None: (0.30, _pass_verdict(), 0.10),
        ("tau", 0.10): (0.50, _pass_verdict(), 0.20),  # improves, gated, beats control
    })
    ledger = fl.run_loop(seed, verify_fn, dry_rounds=1, registry_fn=_single_mutation_registry([m]))
    kept = ledger[1]
    assert kept.decision == "KEEP"
    assert kept.mutation == m
    assert kept.appraisal == pytest.approx(0.50)


def test_loop_reverts_when_improves_but_fails_gate():
    seed = fl.FactorConfig()
    m = fl.Mutation("tau", "tau", 0.10, False)
    verify_fn = _scripted_verify({
        None: (0.30, _pass_verdict(), 0.10),
        ("tau", 0.10): (0.90, _fail_verdict(), 0.20),  # improves + beats control but FAILS gate
    })
    ledger = fl.run_loop(seed, verify_fn, dry_rounds=1, registry_fn=_single_mutation_registry([m]))
    assert ledger[1].decision == "REVERT"
    assert ledger[1].verdict.passed is False


def test_loop_reverts_when_gated_improves_but_loses_to_control():
    seed = fl.FactorConfig()
    m = fl.Mutation("tau", "tau", 0.10, False)
    verify_fn = _scripted_verify({
        None: (0.30, _pass_verdict(), 0.10),
        ("tau", 0.10): (0.50, _pass_verdict(), 0.60),  # improves + gated but control out-earns it
    })
    ledger = fl.run_loop(seed, verify_fn, dry_rounds=1, registry_fn=_single_mutation_registry([m]))
    assert ledger[1].decision == "REVERT"  # 6.6: must beat non-LLM control


def test_loop_stops_after_dry_rounds_consecutive_reverts():
    seed = fl.FactorConfig()
    muts = [fl.Mutation("tau", "tau", v, False) for v in (0.10, 0.20, 0.025)]
    # every mutation fails the gate -> all REVERT
    script = {None: (0.30, _pass_verdict(), 0.10)}
    for mm in muts:
        script[(mm.param, mm.value)] = (0.90, _fail_verdict(), 0.20)
    verify_fn = _scripted_verify(script)
    ledger = fl.run_loop(seed, verify_fn, dry_rounds=2, registry_fn=_single_mutation_registry(muts))
    # baseline + exactly 2 reverts, then stop (third mutation never tried)
    assert [e.decision for e in ledger] == ["KEEP", "REVERT", "REVERT"]


def test_loop_resets_dry_counter_on_keep():
    seed = fl.FactorConfig()
    m_revert = fl.Mutation("tau", "tau", 0.20, False)
    m_keep = fl.Mutation("blend", "blend", 0.40, False)
    m_revert2 = fl.Mutation("tau", "tau", 0.025, False)
    verify_fn = _scripted_verify({
        None: (0.30, _pass_verdict(), 0.10),
        ("tau", 0.20): (0.20, _pass_verdict(), 0.10),   # worse -> REVERT (dry=1)
        ("blend", 0.40): (0.80, _pass_verdict(), 0.10),  # better -> KEEP (dry resets)
        ("tau", 0.025): (0.10, _pass_verdict(), 0.10),   # worse -> REVERT (dry=1)
    })
    ledger = fl.run_loop(
        seed, verify_fn, dry_rounds=2,
        registry_fn=_single_mutation_registry([m_revert, m_keep, m_revert2]),
    )
    # The KEEP resets the dry counter: it takes TWO reverts AFTER the keep to stop.
    # Without a reset, the pre-keep revert would have counted and the loop would
    # have terminated one iteration earlier.
    assert [e.decision for e in ledger] == ["KEEP", "REVERT", "KEEP", "REVERT", "REVERT"]


def test_loop_is_deterministic():
    seed = fl.FactorConfig()
    m = fl.Mutation("tau", "tau", 0.10, False)
    script = {None: (0.30, _pass_verdict(), 0.10), ("tau", 0.10): (0.50, _pass_verdict(), 0.20)}
    reg = _single_mutation_registry([m])
    a = fl.run_loop(seed, _scripted_verify(script), dry_rounds=1, registry_fn=reg)
    b = fl.run_loop(seed, _scripted_verify(script), dry_rounds=1, registry_fn=reg)
    assert a == b


def test_loop_ledger_entries_are_serializable():
    import json

    seed = fl.FactorConfig()
    m = fl.Mutation("tau", "tau", 0.10, False)
    verify_fn = _scripted_verify({
        None: (0.30, _pass_verdict(), 0.10),
        ("tau", 0.10): (0.50, _pass_verdict(), 0.20),
    })
    ledger = fl.run_loop(seed, verify_fn, dry_rounds=1, registry_fn=_single_mutation_registry([m]))
    # ledger entries reduce to JSON-able primitives (mutation fields + verdict values)
    rows = [
        {
            "iteration": e.iteration,
            "mutation": None if e.mutation is None else dataclasses.asdict(e.mutation),
            "appraisal": e.appraisal,
            "passed": e.verdict.passed,
            "first_failure": e.verdict.first_failure,
            "decision": e.decision,
        }
        for e in ledger
    ]
    json.dumps(rows)

