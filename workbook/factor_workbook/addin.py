"""Thin PyXLL worksheet-function layer (R1.2, R1.3, R1.5, R7.2, R7.4).

On the Windows Excel machine this module is listed in ``pyxll.cfg`` under
``modules = factor_workbook.addin``; PyXLL registers every ``@xl_func`` here
as a worksheet function. The async pair (``FW_LOAD``, ``FW_STEP``) returns
Python objects that PyXLL caches automatically and hands to downstream
formulas as object-cache handles, keeping the UI live during loads. Outside
Excel the PyPI stub makes ``@xl_func`` a pass-through, so every function is
importable and callable end-to-end in the root test run (R7.4).

Version semantics (R1.5): ``FW_LOAD`` constructs a NEW client on every call —
the tag cell feeding it is the only version state. Editing that cell re-loads
everything downstream; nothing is remembered across tags.

Error semantics (R1.4): data errors never raise through the UDF boundary —
a failed fetch renders in-cell as ``#ERROR <asset>: <cause> — <detail>``.

Token rule (R1.3): no worksheet function accepts a token argument; the
dormant authenticated path lives entirely inside the release client.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
from pyxll import xl_func

from factor_workbook.contract import SchemaError
from factor_workbook.release import Provenance, ReleaseClient, ReleaseError
from factor_workbook.steps import StepView, build_s1, build_s2, build_s3, build_s4, build_s5
from factor_workbook.verify import checks_frame

_STEP_BUILDERS = {
    "S1": build_s1,
    "S2": build_s2,
    "S3": build_s3,
    "S4": build_s4,
    "S5": build_s5,
}


def _error(asset: str, cause: str, detail: str) -> str:
    """The in-cell per-asset error string (R1.4)."""
    return f"#ERROR {asset}: {cause} — {detail}"


@xl_func("string tag: object", name="FW_LOAD")
async def fw_load(tag: str) -> ReleaseClient:
    """Load one release version, returning the client handle (R1.5).

    Constructs a NEW :class:`ReleaseClient` per call: the tag cell drives
    the version, and editing it re-loads everything — no hidden version
    state. Asynchronous so Excel's UI stays live.

    Args:
        tag: Explicit release version identifier, e.g. ``"data-v1"``.

    Returns:
        The client handle (a PyXLL object-cache handle in Excel).
    """
    return ReleaseClient(tag)


@xl_func("object client, string step: object", name="FW_STEP")
async def fw_step(client: ReleaseClient, step: str) -> StepView | str:
    """Build one storyboard step view from a loaded client handle.

    Args:
        client: Handle from ``FW_LOAD``.
        step: One of ``"S1"``..``"S5"``.

    Returns:
        The :class:`StepView` handle, or an in-cell error string for an
        unknown step or a failed retrieval (R1.4).
    """
    builder = _STEP_BUILDERS.get(step.strip().upper())
    if builder is None:
        return _error("step", "invalid", f"unknown step {step!r} — expected one of {sorted(_STEP_BUILDERS)}")
    try:
        return builder(client)
    except ReleaseError as exc:
        return _error(exc.error.asset, exc.error.cause, exc.error.detail)
    except SchemaError as exc:
        # A contract breach is a per-asset data error too (R1.4): render it
        # in-cell rather than raising through the UDF boundary.
        return _error("contract", "schema", str(exc))


@xl_func("object step_view, string table: dataframe<index=True>", name="FW_TABLE")
def fw_table(step_view: StepView, table: str) -> pd.DataFrame | str:
    """Expand one named table from a step view into the sheet (on demand).

    Args:
        step_view: Handle from ``FW_STEP``.
        table: Name of the table to expand.

    Returns:
        The table as a DataFrame, or an in-cell error string listing the
        available table names when the name is unknown.
    """
    frame = step_view.tables.get(table)
    if frame is None:
        return _error("table", "invalid", f"unknown table {table!r} — available: {', '.join(sorted(step_view.tables))}")
    return frame


@xl_func("object step_view: dataframe<index=True>", name="FW_CHECKS")
def fw_checks(step_view: StepView) -> pd.DataFrame:
    """The step's published-vs-re-derived verification flags as a frame (R7.2)."""
    return checks_frame(step_view.checks)


@xl_func("object step_view: string", name="FW_FRAMING")
def fw_framing(step_view: StepView) -> str:
    """The step's mandated framing text for the sheet header (R7.3)."""
    return step_view.framing


@xl_func("object client: dataframe<index=True>", name="FW_PROVENANCE")
def fw_provenance(client: ReleaseClient) -> pd.DataFrame:
    """Everything loaded so far, one provenance row per asset (R1.2)."""
    columns = [f.name for f in dataclasses.fields(Provenance)]
    rows = [dataclasses.asdict(p) for p in client.provenance_table()]
    return pd.DataFrame(rows, columns=columns)


@xl_func("object client: string", name="FW_VERSION")
def fw_version(client: ReleaseClient) -> str:
    """The loaded release tag, for the provenance header row."""
    return client.tag
