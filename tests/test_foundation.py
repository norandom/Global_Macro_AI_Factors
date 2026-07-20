"""Foundation smoke tests for the track-a-macro-steering feature (task 1.1).

Verifies the recall_guard dependency is installed and its public scorer surface
is importable, and that the host macro_framework package imports. Requirements: 1.1, 6.1.
"""

from __future__ import annotations


def test_recall_guard_public_surface_importable() -> None:
    """recall_guard@v0.1.0 is resolved and exposes the scorer facade (1.1)."""
    from recall_guard import ConfigurationError, GuardedScore, MemoryGuardedScorer

    assert MemoryGuardedScorer.__name__ == "MemoryGuardedScorer"
    assert hasattr(MemoryGuardedScorer, "calibrate")
    assert {f for f in GuardedScore.__dataclass_fields__} >= {
        "p_memorized",
        "memguard_confidence",
        "fail_reason",
        "signal",
        "raw_confidence",
    }
    assert issubclass(ConfigurationError, Exception)


def test_macro_framework_imports() -> None:
    """The existing host package still imports unchanged (6.1)."""
    import macro_framework  # noqa: F401


def test_new_library_modules_reachable_via_package() -> None:
    """skill_metric and regime_overlay symbols export through the package (8.2)."""
    import macro_framework as mf

    for name in (
        "basket_residual",
        "market_attribution",
        "BasketResidual",
        "MarketAttribution",
        "GateConfig",
        "GateVerdict",
        "evaluate_gates",
        "IDIO_FLOOR",
        "ewma_correlation_matrix",
        "avg_pairwise_correlation",
        "correlation_scale",
        "derisk_cash_pin",
    ):
        assert name in mf.__all__, f"{name} missing from macro_framework.__all__"
        assert hasattr(mf, name), f"macro_framework.{name} not reachable"
