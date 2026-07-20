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
