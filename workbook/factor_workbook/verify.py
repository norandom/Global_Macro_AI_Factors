"""Verification framework: published-vs-re-derived comparison records (R7.2).

A disagreement between a published figure and its re-derived value is data,
never an exception: ``compare`` returns a :class:`Check` whose ``ok``/``message``
fields the sheet (FW_CHECKS) displays. Neither value is silently preferred.
``compare`` raises ``TypeError`` only on non-numeric input, which indicates a
programming error upstream rather than a legitimate discrepancy.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import asdict, dataclass

import pandas as pd

_COLUMNS = ["name", "published", "rederived", "tolerance", "ok", "message"]


@dataclass(frozen=True)
class Check:
    """One published-vs-re-derived comparison, rendered as a flag row.

    Attributes:
        name: Human-readable check name, e.g. ``"S3 guarded_tilt equals raw*(1-p)"``.
        published: The value shipped with the release (may be None/NaN).
        rederived: The value re-derived from released data (may be None/NaN).
        tolerance: Relative tolerance used (absolute when published == 0).
        ok: True when the values agree within tolerance.
        message: Empty when ok; otherwise names the check, both values, and
            the tolerance for display in the sheet.
    """

    name: str
    published: float | None
    rederived: float | None
    tolerance: float
    ok: bool
    message: str


def _agree(published: float, rederived: float, tol: float) -> bool:
    if math.isnan(published) or math.isnan(rederived):
        # NaN matches NaN only (both out-of-window is agreement).
        return math.isnan(published) and math.isnan(rederived)
    if published == rederived:  # covers equal infinities
        return True
    if math.isinf(published) or math.isinf(rederived):
        return False
    if published == 0.0:
        return abs(rederived) <= tol  # relative tol undefined at 0 -> absolute
    return abs(published - rederived) <= tol * abs(published)


def compare(
    name: str,
    published: float | None,
    rederived: float | None,
    *,
    tol: float = 1e-6,
) -> Check:
    """Pair a published figure with its re-derived value and flag disagreement.

    Args:
        name: Human-readable check name (appears in the failure message).
        published: Published summary value; None means missing.
        rederived: Value re-derived from released data; None means missing.
        tol: Relative tolerance (absolute fallback when published is 0).

    Returns:
        A :class:`Check`; a failed check is returned, never raised (R7.2).

    Raises:
        TypeError: If either value is neither a number nor None — a
            programming error, not a data discrepancy.
    """
    for label, value in (("published", published), ("rederived", rederived)):
        # numbers.Real admits numpy scalars (np.int64/np.float64) — pandas
        # aggregations feed these in from the step views (review 2026-07-09).
        if value is not None and not isinstance(value, numbers.Real):
            raise TypeError(f"{name}: {label} value is not numeric: {value!r}")

    if published is None or rederived is None:
        missing = [
            label
            for label, value in (("published", published), ("rederived", rederived))
            if value is None
        ]
        message = (
            f"{name}: {' and '.join(missing)} value missing "
            f"(published={published!r}, re-derived={rederived!r}, tolerance={tol:g})"
        )
        return Check(name, published, rederived, tol, False, message)

    ok = _agree(float(published), float(rederived), tol)
    message = (
        ""
        if ok
        else (
            f"{name}: published={published!r} vs re-derived={rederived!r} "
            f"disagree beyond tolerance {tol:g}"
        )
    )
    return Check(name, published, rederived, tol, ok, message)


def checks_frame(checks: list[Check]) -> pd.DataFrame:
    """Render checks as a DataFrame the add-in spills into the sheet.

    Args:
        checks: Comparison records, typically one per headline figure.

    Returns:
        DataFrame with columns name/published/rederived/tolerance/ok/message,
        one row per check (columns preserved when empty).
    """
    return pd.DataFrame([asdict(c) for c in checks], columns=_COLUMNS)
