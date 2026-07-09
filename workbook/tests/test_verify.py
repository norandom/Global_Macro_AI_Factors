"""verify.compare: agreement -> ok record; injected discrepancy -> flagged
record returned as data with a human-readable message (R7.2)."""

from __future__ import annotations

import dataclasses
import math

import pandas as pd
import pytest

from factor_workbook.verify import Check, checks_frame, compare


class TestAgreement:
    def test_exact_agreement_is_ok(self) -> None:
        check = compare("S3 auc", 0.512345, 0.512345)
        assert check.ok is True
        assert check.name == "S3 auc"
        assert check.published == 0.512345
        assert check.rederived == 0.512345
        assert check.tolerance == 1e-6

    def test_within_relative_tolerance_is_ok(self) -> None:
        assert compare("x", 100.0, 100.0 + 100.0 * 1e-7).ok is True

    def test_custom_tolerance(self) -> None:
        assert compare("x", 1.0, 1.005, tol=1e-2).ok is True
        assert compare("x", 1.0, 1.005, tol=1e-3).ok is False

    def test_both_zero_is_ok(self) -> None:
        assert compare("x", 0.0, 0.0).ok is True


class TestDiscrepancy:
    def test_injected_discrepancy_is_flagged_not_raised(self) -> None:
        check = compare("S3 guarded_tilt", 0.5, 0.6)
        assert check.ok is False
        # message names the check, both values, and the tolerance
        assert "S3 guarded_tilt" in check.message
        assert "0.5" in check.message
        assert "0.6" in check.message
        assert "1e-06" in check.message

    def test_neither_value_is_preferred(self) -> None:
        check = compare("x", 0.5, 0.6)
        assert check.published == 0.5
        assert check.rederived == 0.6

    def test_ok_record_has_empty_message(self) -> None:
        assert compare("x", 1.0, 1.0).message == ""


class TestEdgeCases:
    def test_published_zero_falls_back_to_absolute_tolerance(self) -> None:
        assert compare("x", 0.0, 5e-7).ok is True
        assert compare("x", 0.0, 5e-6).ok is False

    def test_both_nan_is_ok(self) -> None:
        assert compare("x", float("nan"), float("nan")).ok is True

    def test_one_sided_nan_is_flagged(self) -> None:
        assert compare("x", float("nan"), 0.5).ok is False
        assert compare("x", 0.5, float("nan")).ok is False

    def test_none_is_flagged_with_message(self) -> None:
        check = compare("x", None, 0.5)
        assert check.ok is False
        assert "missing" in check.message.lower()
        assert compare("x", 0.5, None).ok is False
        assert compare("x", None, None).ok is False

    def test_equal_infinities_are_ok(self) -> None:
        assert compare("x", math.inf, math.inf).ok is True

    def test_infinity_vs_finite_is_flagged(self) -> None:
        assert compare("x", math.inf, 0.5).ok is False
        assert compare("x", -math.inf, math.inf).ok is False

    def test_non_numeric_garbage_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            compare("x", "0.5", 0.5)  # type: ignore[arg-type]

    def test_check_is_frozen(self) -> None:
        check = compare("x", 1.0, 1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            check.ok = False  # type: ignore[misc]


class TestChecksFrame:
    def test_columns_and_rows(self) -> None:
        checks = [compare("a", 1.0, 1.0), compare("b", 0.5, 0.6)]
        frame = checks_frame(checks)
        assert list(frame.columns) == [
            "name", "published", "rederived", "tolerance", "ok", "message",
        ]
        assert list(frame["name"]) == ["a", "b"]
        assert list(frame["ok"]) == [True, False]

    def test_empty_list_keeps_columns(self) -> None:
        frame = checks_frame([])
        assert list(frame.columns) == [
            "name", "published", "rederived", "tolerance", "ok", "message",
        ]
        assert len(frame) == 0
        assert isinstance(frame, pd.DataFrame)


def test_numpy_scalars_are_accepted_as_numeric() -> None:
    """np.int64/np.float64 (pandas aggregation outputs) pass the type gate."""
    import numpy as np

    from factor_workbook.verify import compare

    check = compare("counts", np.int64(5), np.float64(5.0))
    assert check.ok
    flagged = compare("counts", np.int64(5), np.int64(7))
    assert not flagged.ok and "counts" in flagged.message
