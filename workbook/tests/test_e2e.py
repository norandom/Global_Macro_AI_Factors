"""End-to-end validation over the whole stack + opt-in live-network test (task 6.1).

One flow drives the FULL chain through the Excel function surface — FW_LOAD ->
FW_STEP x5 -> FW_TABLE / FW_CHECKS / FW_FRAMING / FW_PROVENANCE — on the shipped
fixtures (R7.4): every identity-based verification check passes and every
subset-based check renders as a visible flag, never an exception (R7.2).

R7.1 (no inference): a fresh subprocess importing every factor_workbook module
pulls in neither recall_guard, openai, dspy, nor macro_framework; a source scan
confirms no module even names macro_framework in an import statement.

Live network (design Testing Strategy, opt-in marker): one test marked
``live_network`` and gated on ``FW_LIVE_NETWORK=1`` fetches the real ``data-v1``
``norecall_screen_results.json`` from GitHub and validates it against the
schema contract — excluded from every default run, no root-config change.
"""

import asyncio
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from factor_workbook import addin, contract
from factor_workbook.release import ReleaseClient
from factor_workbook.steps import StepView
from test_addin import FakeClient  # provenance-recording fixture-backed client

WORKBOOK_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = WORKBOOK_DIR / "factor_workbook"

STEPS = ["S1", "S2", "S3", "S4", "S5"]
# One representative table per step, expanded through FW_TABLE.
STEP_TABLES = {
    "S1": "certification",
    "S2": "naive_eval",
    "S3": "views_v1",
    "S4": "equity",
    "S5": "contrast",
}
# The sole identity-based check on fixture row subsets: a per-row identity of
# the published table, so it passes even on 5-row fixtures (test_steps.py).
IDENTITY_CHECKS = {"S3 guarded_tilt equals raw*(1-p)"}


@pytest.fixture(scope="module")
def pipeline():
    """The whole fixture-driven pipeline once, through the FW_* surface only."""
    real = addin.ReleaseClient
    addin.ReleaseClient = FakeClient  # module-scoped monkeypatch
    try:
        client = asyncio.run(addin.fw_load("data-v1"))
        views = {step: asyncio.run(addin.fw_step(client, step)) for step in STEPS}
    finally:
        addin.ReleaseClient = real
    return client, views


# --------------------------------------------------------------------------- #
# Full chain: load -> contract -> re-derive -> verify -> all five step views   #
# --------------------------------------------------------------------------- #


def test_fw_load_returns_the_fixture_client(pipeline):
    client, _ = pipeline
    assert isinstance(client, FakeClient)
    assert addin.fw_version(client) == "data-v1"


def test_every_step_view_builds_through_the_addin_surface(pipeline):
    _, views = pipeline
    for step in STEPS:
        view = views[step]
        assert isinstance(view, StepView), f"{step} did not build: {view!r}"
        framing = addin.fw_framing(view)
        assert framing
        if step == "S1":  # S1's mandated framing is the empty-certified-set statement
            assert "certified set is empty" in framing.lower()
        else:
            assert "no forecast-accuracy claim" in framing.lower()
        table = addin.fw_table(view, STEP_TABLES[step])
        assert isinstance(table, pd.DataFrame) and not table.empty


def test_identity_based_checks_pass_and_subset_checks_render_as_flags(pipeline):
    """R7.2 end-to-end: on the shipped fixtures every identity-based check is
    ok; every subset-based disagreement is a rendered flag with a visible
    message — and FW_CHECKS spills all of them as data."""
    _, views = pipeline
    seen = set()
    for step in STEPS:
        view = views[step]
        frame = addin.fw_checks(view)
        assert list(frame.columns) == [
            "name", "published", "rederived", "tolerance", "ok", "message",
        ]
        assert len(frame) == len(view.checks)
        for check in view.checks:
            seen.add(check.name)
            if check.name in IDENTITY_CHECKS:
                assert check.ok is True, check.message
                assert check.message == ""
            elif not check.ok:
                assert check.message, f"{check.name}: failed check must render a flag"
    assert IDENTITY_CHECKS <= seen
    # fixture subsets genuinely disagree with full-data figures: flags exist
    flagged = [c for v in views.values() for c in v.checks if not c.ok]
    assert flagged


def test_provenance_covers_everything_loaded(pipeline):
    """R1.2 end-to-end: after the full run FW_PROVENANCE lists one populated
    row per asset that fed the five views."""
    client, _ = pipeline
    frame = addin.fw_provenance(client)
    assert not frame.empty
    assert (frame["tag"] == "data-v1").all()
    assert frame["asset"].is_unique
    assert frame["sha256"].str.fullmatch("[0-9a-f]{64}").all()
    assert frame["url"].str.len().gt(0).all()
    # the flow touched every layer's assets: screen, naive eval, factors, sim
    assets = set(frame["asset"])
    for expected in (
        "norecall_screen_results.json",
        "naive_directional_eval_openai_gpt-oss-20b.parquet",
        "factor_loadings_v1.parquet",
        "factor_equity_v1.parquet",
        "factor_contrast_v1.parquet",
    ):
        assert expected in assets, expected


# --------------------------------------------------------------------------- #
# R7.1 — no inference: lean import surface                                     #
# --------------------------------------------------------------------------- #


def test_importing_the_whole_package_pulls_no_inference_stack():
    """Fresh interpreter: importing every factor_workbook module loads neither
    recall_guard, openai, dspy, nor macro_framework (R7.1)."""
    code = textwrap.dedent(
        """
        import sys
        import factor_workbook
        import factor_workbook.addin
        import factor_workbook.certification
        import factor_workbook.contract
        import factor_workbook.rederive
        import factor_workbook.release
        import factor_workbook.steps
        import factor_workbook.vendored_ssr
        import factor_workbook.verify
        forbidden = sorted(
            m for m in sys.modules
            if m.split(".")[0] in {"recall_guard", "openai", "dspy", "macro_framework"}
        )
        assert not forbidden, forbidden
        print("NO-INFERENCE-OK")
        """
    )
    env = dict(os.environ, PYTHONPATH=str(WORKBOOK_DIR))
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0, proc.stderr
    assert "NO-INFERENCE-OK" in proc.stdout


def test_no_module_imports_macro_framework():
    """Source-level guard: no factor_workbook module names macro_framework in
    an import statement — the vendored SSR is verbatim code, not a link."""
    pattern = re.compile(r"^\s*(?:from|import)\s+macro_framework", re.MULTILINE)
    sources = sorted(PACKAGE_DIR.glob("*.py"))
    assert sources  # the scan actually saw the package
    offenders = [p.name for p in sources if pattern.search(p.read_text())]
    assert offenders == []


# --------------------------------------------------------------------------- #
# Opt-in live-network test (design: Network integration marker)                #
# --------------------------------------------------------------------------- #


def test_live_network_marker_is_registered(pytestconfig):
    """The opt-in marker is declared, so default runs stay warning-free."""
    markers = pytestconfig.getini("markers")
    assert any(str(line).startswith("live_network") for line in markers)


@pytest.mark.live_network
@pytest.mark.skipif(
    os.environ.get("FW_LIVE_NETWORK") != "1",
    reason="live-network test: set FW_LIVE_NETWORK=1 to run",
)
def test_live_fetch_validates_real_release_asset(tmp_path):
    """Opt-in end-to-end: fetch the real data-v1 norecall_screen_results.json
    from GitHub, validate it against the schema contract, and check the
    provenance record (R1.1, R1.2) — public path only, no credentials."""
    client = ReleaseClient("data-v1", cache_dir=tmp_path, token_provider=lambda: None)
    obj, provenance = contract.load_json(client, "norecall_screen_results")
    assert obj["screen"]
    assert len(obj["candidates"]) == 5
    assert len(obj["results"]) == 5
    assert provenance.tag == "data-v1"
    assert provenance.asset == "norecall_screen_results.json"
    assert provenance.url.startswith(
        "https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v1/"
    )
    assert re.fullmatch("[0-9a-f]{64}", provenance.sha256)
    assert provenance.from_cache is False
    assert (tmp_path / "data-v1" / "norecall_screen_results.json").exists()


@pytest.mark.live_network
@pytest.mark.skipif(
    os.environ.get("FW_LIVE_NETWORK") != "1",
    reason="live-network test: set FW_LIVE_NETWORK=1 to run",
)
def test_live_all_five_step_views_agree_on_full_data(tmp_path):
    """Opt-in FULL integration: build every step view against the real data-v1
    release and require every attached verification check to agree — the
    cross-task guarantee the fixture path (subsets) cannot give. Added after
    feature validation 2026-07-09 caught S1 count-check false alarms that the
    narrower single-asset live test missed."""
    from factor_workbook.steps import build_s1, build_s2, build_s3, build_s4, build_s5

    client = ReleaseClient("data-v1", cache_dir=tmp_path, token_provider=lambda: None)
    failed: list[str] = []
    for build in (build_s1, build_s2, build_s3, build_s4, build_s5):
        view = build(client)
        assert view.tables and view.framing
        failed.extend(c.message for c in view.checks if not c.ok)
    assert not failed, "checks failing on pristine full data:\n" + "\n".join(failed)


def test_fw_step_renders_schema_error_in_cell():
    """A contract breach surfaces as the canonical in-cell #ERROR string, not
    an exception through the UDF boundary (R1.4 — validation 2026-07-09)."""
    import asyncio

    from factor_workbook import addin
    from factor_workbook.contract import SchemaError

    class _BreachingClient:
        def fetch(self, asset):
            raise SchemaError(f"{asset}: missing column 'p_memorized' (expected float64)")

        def fetch_tar_member(self, asset, member):
            raise SchemaError(f"{asset}: missing member {member}")

    result = asyncio.run(addin.fw_step(_BreachingClient(), "S3"))
    assert isinstance(result, str)
    assert result.startswith("#ERROR contract: schema")
    assert "missing column" in result
