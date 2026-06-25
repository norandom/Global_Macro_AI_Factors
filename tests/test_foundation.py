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
