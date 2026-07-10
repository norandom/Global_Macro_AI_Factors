"""Offline tests for the Excel function surface (task 5.1).

Under the PyXLL PyPI stub (no Excel), ``@xl_func`` is a pass-through: the
async wrappers stay plain coroutines (driven with ``asyncio.run``) and the
sync wrappers stay plain functions — so every exposed ``FW_*`` function is
exercised end-to-end on the fixtures in the root test run (R7.4).

Covered: FW_LOAD constructs a new client per call from the tag cell (R1.5),
FW_STEP builds every S1-S5 view, FW_TABLE expands named tables on demand,
FW_CHECKS/FW_PROVENANCE render as data frames (R1.2, R7.2), no wrapper
accepts a token argument (R1.3), and error states surface as per-asset
in-sheet strings — never exceptions through the UDF boundary (R1.4).
"""

import asyncio
import dataclasses
import inspect
import io
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from factor_workbook import addin
from factor_workbook.release import FetchError, Provenance, ReleaseError
from factor_workbook.steps import StepView
from factor_workbook.verify import checks_frame

FIXTURES = Path(__file__).parent / "fixtures"

STEPS = ["S0", "S1", "S2", "S3", "S4", "S5"]

FW_FUNCTIONS = [
    addin.fw_load,
    addin.fw_step,
    addin.fw_table,
    addin.fw_checks,
    addin.fw_framing,
    addin.fw_provenance,
    addin.fw_version,
]


class FakeClient:
    """ReleaseClient stand-in over the fixtures, recording provenance."""

    def __init__(self, tag: str = "data-v1"):
        self.tag = tag
        self._provenance: dict[str, Provenance] = {}

    def fetch(self, asset: str) -> tuple[bytes, Provenance]:
        data = (FIXTURES / asset).read_bytes()
        provenance = Provenance(
            tag=self.tag,
            asset=asset,
            url=f"fixture://{asset}",
            fetched_at="2026-07-09T00:00:00+00:00",
            sha256="0" * 64,
            from_cache=False,
        )
        self._provenance[asset] = provenance
        return data, provenance

    def fetch_tar_member(self, asset: str, member: str) -> tuple[bytes, Provenance]:
        data, provenance = self.fetch(asset)
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise KeyError(member)
                return extracted.read(), provenance
        except (tarfile.TarError, KeyError) as exc:
            raise ReleaseError(FetchError(asset, "unpack", f"member {member!r}: {exc}")) from exc

    def provenance_table(self) -> list[Provenance]:
        return list(self._provenance.values())


class FailingClient(FakeClient):
    """Every fetch fails with the typed per-asset error (R1.4)."""

    def fetch(self, asset: str) -> tuple[bytes, Provenance]:
        raise ReleaseError(FetchError(asset, "missing", f"HTTP 404 at fixture://{asset}"))


@pytest.fixture(autouse=True)
def _fake_release_client(monkeypatch):
    """Route FW_LOAD's client construction to the fixture-backed fake."""
    monkeypatch.setattr(addin, "ReleaseClient", FakeClient)


@pytest.fixture(scope="module")
def client():
    return FakeClient()


@pytest.fixture(scope="module")
def views(client):
    """Every step view built once through FW_STEP for read-only assertions."""
    return {step: asyncio.run(addin.fw_step(client, step)) for step in STEPS}


def test_no_fw_function_accepts_a_token_argument():
    """R1.3: no worksheet function takes a token parameter — ever."""
    for func in FW_FUNCTIONS:
        params = inspect.signature(func).parameters
        assert "token" not in params, f"{func.__name__} must not accept a token"


def test_fw_load_constructs_a_new_client_per_call():
    """R1.5: the tag cell drives an explicit re-load — no hidden version state."""
    first = asyncio.run(addin.fw_load("data-v1"))
    second = asyncio.run(addin.fw_load("data-v1"))
    assert isinstance(first, FakeClient)
    assert first is not second
    assert first.tag == "data-v1"
    assert asyncio.run(addin.fw_load("data-v2")).tag == "data-v2"


def test_fw_version_returns_the_tag(client):
    assert addin.fw_version(client) == "data-v1"


@pytest.mark.parametrize("step", STEPS)
def test_fw_step_builds_every_step_view(views, step):
    view = views[step]
    assert isinstance(view, StepView)
    assert view.title.startswith(step)


def test_fw_step_unknown_step_returns_error_string(client):
    result = asyncio.run(addin.fw_step(client, "S9"))
    assert isinstance(result, str)
    assert result.startswith("#ERROR")
    assert "S9" in result and "S1" in result and "S5" in result


def test_fw_step_release_error_surfaces_as_per_asset_message():
    """R1.4: a failed fetch renders in-cell, never raises through the UDF."""
    result = asyncio.run(addin.fw_step(FailingClient(), "S2"))
    assert isinstance(result, str)
    assert result.startswith("#ERROR")
    assert "missing" in result and "HTTP 404" in result


class DataV1Client(FakeClient):
    """data-v1 stand-in: the data-v2 static assets do not exist on this tag."""

    def fetch(self, asset: str) -> tuple[bytes, Provenance]:
        if asset.startswith("static_bh"):
            raise ReleaseError(
                FetchError(asset, "missing", f"HTTP 404 at fixture://{asset}")
            )
        return super().fetch(asset)


def test_fw_step_s0_on_data_v1_surfaces_per_asset_error():
    """Task 7.1 observable: on data-v1 the S0 step renders the canonical
    per-asset #ERROR string (the static assets did not exist yet) — never an
    exception through the UDF boundary (R1.4)."""
    result = asyncio.run(addin.fw_step(DataV1Client(), "S0"))
    assert isinstance(result, str)
    assert result.startswith("#ERROR static_bh_stats.json: missing — ")
    assert "404" in result


def test_fw_table_expands_named_tables_on_demand(views):
    for view in views.values():
        for name, expected in view.tables.items():
            frame = addin.fw_table(view, name)
            assert isinstance(frame, pd.DataFrame)
            pd.testing.assert_frame_equal(frame, expected)


def test_fw_table_unknown_name_lists_available_keys(views):
    view = views["S2"]
    result = addin.fw_table(view, "nope")
    assert isinstance(result, str)
    assert result.startswith("#ERROR")
    for name in view.tables:
        assert name in result


def test_fw_checks_renders_the_verification_flags_frame(views):
    for view in views.values():
        frame = addin.fw_checks(view)
        assert isinstance(frame, pd.DataFrame)
        pd.testing.assert_frame_equal(frame, checks_frame(view.checks))
        assert len(frame) == len(view.checks)


def test_fw_framing(views):
    for view in views.values():
        assert addin.fw_framing(view) == view.framing


def test_fw_provenance_renders_a_frame_of_provenance_rows(client, views):
    frame = addin.fw_provenance(client)
    assert isinstance(frame, pd.DataFrame)
    columns = [f.name for f in dataclasses.fields(Provenance)]
    assert list(frame.columns) == columns
    assert len(frame) == len(client.provenance_table())
    assert len(frame) > 0
    assert (frame["tag"] == "data-v1").all()


def test_fw_provenance_empty_client_keeps_columns():
    frame = addin.fw_provenance(FakeClient())
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 0
    assert list(frame.columns) == [f.name for f in dataclasses.fields(Provenance)]
