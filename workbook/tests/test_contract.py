"""Offline tests for the schema-contract registry and typed loaders (task 2.2).

Every ``ASSET_SPECS`` entry is validated against its checked-in fixture (a
schema-true subset of the real release asset), and corrupted fixtures must
produce the asset+column-specific ``SchemaError`` message that is the
foundation of the discrepancy detector (R1.4).
"""

import io
import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from factor_workbook.contract import (
    ASSET_SPECS,
    AssetSpec,
    SchemaError,
    load_frame,
    load_json,
)
from factor_workbook.release import Provenance

FIXTURES = Path(__file__).parent / "fixtures"
EVIDENCE_MODEL = "openai_gpt-oss-20b"
# 120b carries raw_ref_delta as all-null float64 (20b/phi-4 surface it as
# all-null object) — the fixture keeps one member of each flavor so the
# tolerant dtype rule is regression-tested against real published data.
EVIDENCE_MODELS = ("openai_gpt-oss-20b", "openai_gpt-oss-120b")


class FakeClient:
    """ReleaseClient stand-in serving checked-in fixture bytes offline."""

    tag = "data-v1"

    def __init__(self, overrides: dict[str, bytes] | None = None):
        self._overrides = overrides or {}

    def fetch(self, asset: str) -> tuple[bytes, Provenance]:
        data = self._overrides.get(asset)
        if data is None:
            data = (FIXTURES / asset).read_bytes()
        provenance = Provenance(
            tag=self.tag,
            asset=asset,
            url=f"fixture://{asset}",
            fetched_at="2026-07-09T00:00:00+00:00",
            sha256="0" * 64,
            from_cache=False,
        )
        return data, provenance

    def fetch_tar_member(self, asset: str, member: str) -> tuple[bytes, Provenance]:
        data, provenance = self.fetch(asset)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
            extracted = tar.extractfile(member)
            assert extracted is not None, member
            return extracted.read(), provenance


def frame_keys():
    return sorted(k for k, s in ASSET_SPECS.items() if s.kind in ("parquet", "tar_parquet"))


def json_keys():
    return sorted(k for k, s in ASSET_SPECS.items() if s.kind in ("json", "tar_json"))


def rewrite_parquet(asset: str, mutate) -> bytes:
    """Read a parquet fixture, apply ``mutate(df) -> df``, return new bytes."""
    df = mutate(pd.read_parquet(FIXTURES / asset))
    buf = io.BytesIO()
    df.to_parquet(buf)
    return buf.getvalue()


# --- registry coverage -----------------------------------------------------


def test_registry_covers_all_consumed_assets():
    """Fixtures cover every consumed asset incl. evidence bundle + screen results."""
    assets = {spec.asset for spec in ASSET_SPECS.values()}
    # one fixture file per distinct release asset
    for asset in assets:
        assert (FIXTURES / asset).is_file(), f"missing fixture for {asset}"
    assert "norecall_screen_evidence.tar.gz" in assets
    assert "norecall_screen_results.json" in assets
    # every spec is a frozen AssetSpec with a sane kind
    for key, spec in ASSET_SPECS.items():
        assert isinstance(spec, AssetSpec), key
        assert spec.kind in ("parquet", "json", "tar_parquet", "tar_json"), key
        assert spec.columns, key


def test_evidence_members_parameterized_by_model():
    spec = ASSET_SPECS["norecall_screen_evidence"]
    assert spec.member is not None and "{model}" in spec.member


# --- every registry entry validates against its fixture ---------------------


@pytest.mark.parametrize("key", frame_keys())
def test_frame_specs_validate_against_fixtures(key):
    df, provenance = load_frame(FakeClient(), key, model=EVIDENCE_MODEL)
    spec = ASSET_SPECS[key]
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= spec.min_rows
    assert df.index.name == spec.index
    assert set(df.columns) == set(spec.columns)
    assert provenance.asset == spec.asset


@pytest.mark.parametrize("key", json_keys())
def test_json_specs_validate_against_fixtures(key):
    obj, provenance = load_json(FakeClient(), key, model=EVIDENCE_MODEL)
    assert isinstance(obj, dict)
    assert provenance.asset == ASSET_SPECS[key].asset


def test_min_rows_is_conservative_and_expected_rows_records_production():
    """min_rows is fixture-safe; full production counts live in expected_rows."""
    for key, spec in ASSET_SPECS.items():
        assert spec.min_rows <= spec.expected_rows or spec.expected_rows == 0, key
    assert ASSET_SPECS["factor_loadings_v1"].expected_rows == 72
    assert ASSET_SPECS["factor_views_v1"].expected_rows == 284
    assert ASSET_SPECS["factor_targets_v1"].expected_rows == 2828
    assert ASSET_SPECS["factor_equity_v1"].expected_rows == 2717
    assert ASSET_SPECS["norecall_screen_evidence"].expected_rows == 521


@pytest.mark.parametrize("model", EVIDENCE_MODELS)
def test_evidence_fixture_keeps_all_feature_columns(model):
    df, _ = load_frame(FakeClient(), "norecall_screen_evidence", model=model)
    for col in ("arm", "prompt", "reply", "n_tokens", "included", "dropped_reason"):
        assert col in df.columns
    assert [c for c in df.columns if c.startswith("raw_")], "raw_* features missing"
    assert [c for c in df.columns if c.startswith("std_")], "std_* features missing"


@pytest.mark.parametrize("model", EVIDENCE_MODELS)
def test_evidence_bundle_validates_per_model(model):
    """Every model-parameterized member validates — incl. 120b's all-null
    float64 raw_ref_delta, which must satisfy the contract (regression)."""
    client = FakeClient()
    df, _ = load_frame(client, "norecall_screen_evidence", model=model)
    assert df["raw_ref_delta"].isna().all()
    load_json(client, "norecall_screen_evidence_baseline", model=model)
    load_json(client, "norecall_screen_evidence_summary", model=model)


def test_decision_log_variant_metas():
    """v1/v2/nonpit share the per-date shape but carry variant-specific metas."""
    client = FakeClient()
    v1, _ = load_json(client, "factor_decision_log_v1")
    v2, _ = load_json(client, "factor_decision_log_v2")
    nonpit, _ = load_json(client, "factor_nonpit_diagnostic_decision_log_v1")
    for log in (v1, v2, nonpit):
        for field in ("p_memorized", "parse_ok", "steered", "conviction", "loadings", "views"):
            assert isinstance(log[field], dict)
    assert "cutoff_date" in v1["meta"]
    assert "prompt_version" in v2["meta"]
    assert "variant" in nonpit["meta"]


# --- fail-fast corrupted-fixture messages -----------------------------------


def test_missing_column_names_asset_column_and_dtype():
    corrupted = rewrite_parquet(
        "factor_loadings_v1.parquet", lambda df: df.drop(columns=["inflation"])
    )
    client = FakeClient({"factor_loadings_v1.parquet": corrupted})
    with pytest.raises(SchemaError) as exc:
        load_frame(client, "factor_loadings_v1")
    message = str(exc.value)
    assert "factor_loadings_v1.parquet" in message
    assert "inflation" in message
    assert "float64" in message


def test_wrong_dtype_names_asset_column_and_expected_dtype():
    corrupted = rewrite_parquet(
        "factor_loadings_v1.parquet", lambda df: df.assign(inflation=df["inflation"].astype(str))
    )
    client = FakeClient({"factor_loadings_v1.parquet": corrupted})
    with pytest.raises(SchemaError) as exc:
        load_frame(client, "factor_loadings_v1")
    message = str(exc.value)
    assert "factor_loadings_v1.parquet" in message
    assert "inflation" in message
    assert "float64" in message


def test_unexpected_column_is_flagged():
    corrupted = rewrite_parquet(
        "factor_scores_v1.parquet", lambda df: df.assign(surprise=1.0)
    )
    client = FakeClient({"factor_scores_v1.parquet": corrupted})
    with pytest.raises(SchemaError) as exc:
        load_frame(client, "factor_scores_v1")
    message = str(exc.value)
    assert "factor_scores_v1.parquet" in message
    assert "surprise" in message


def test_wrong_index_names_asset():
    corrupted = rewrite_parquet(
        "factor_loadings_v1.parquet", lambda df: df.rename_axis("timestamp")
    )
    client = FakeClient({"factor_loadings_v1.parquet": corrupted})
    with pytest.raises(SchemaError) as exc:
        load_frame(client, "factor_loadings_v1")
    message = str(exc.value)
    assert "factor_loadings_v1.parquet" in message
    assert "date" in message


def test_too_few_rows_is_flagged():
    corrupted = rewrite_parquet("factor_loadings_v1.parquet", lambda df: df.head(1))
    client = FakeClient({"factor_loadings_v1.parquet": corrupted})
    with pytest.raises(SchemaError) as exc:
        load_frame(client, "factor_loadings_v1")
    assert "factor_loadings_v1.parquet" in str(exc.value)


def test_corrupted_json_names_asset_and_key():
    log = json.loads((FIXTURES / "factor_decision_log_v1.json").read_text())
    del log["meta"]["nim_model"]
    client = FakeClient({"factor_decision_log_v1.json": json.dumps(log).encode()})
    with pytest.raises(SchemaError) as exc:
        load_json(client, "factor_decision_log_v1")
    message = str(exc.value)
    assert "factor_decision_log_v1.json" in message
    assert "meta.nim_model" in message


def test_corrupted_tar_json_names_asset_and_key():
    member = f"evidence/{EVIDENCE_MODEL}/summary.json"
    good, _ = FakeClient().fetch_tar_member("norecall_screen_evidence.tar.gz", member)
    summary = json.loads(good)
    del summary["controlled_auc"]  # surgical corruption
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = json.dumps(summary).encode()
        info = tarfile.TarInfo(member)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    client = FakeClient({"norecall_screen_evidence.tar.gz": buf.getvalue()})
    with pytest.raises(SchemaError) as exc:
        load_json(client, "norecall_screen_evidence_summary", model=EVIDENCE_MODEL)
    message = str(exc.value)
    assert "norecall_screen_evidence.tar.gz" in message
    assert "controlled_auc" in message


# --- tolerant dtype comparison ----------------------------------------------


def test_datetime_unit_tolerance():
    """datetime64[ms] and [ns] both satisfy a datetime64 contract column."""
    corrupted = rewrite_parquet(
        "factor_loadings_v1.parquet",
        lambda df: df.set_axis(df.index.astype("datetime64[ns]")).rename_axis("date"),
    )
    client = FakeClient({"factor_loadings_v1.parquet": corrupted})
    df, _ = load_frame(client, "factor_loadings_v1")  # must not raise
    assert len(df) >= ASSET_SPECS["factor_loadings_v1"].min_rows


# --- loader/kind discipline --------------------------------------------------


def test_load_frame_rejects_json_keys():
    with pytest.raises(ValueError):
        load_frame(FakeClient(), "factor_stability_v1")


def test_load_json_rejects_frame_keys():
    with pytest.raises(ValueError):
        load_json(FakeClient(), "factor_loadings_v1")


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        load_frame(FakeClient(), "no_such_key")


def test_evidence_requires_model():
    with pytest.raises(ValueError):
        load_frame(FakeClient(), "norecall_screen_evidence")
