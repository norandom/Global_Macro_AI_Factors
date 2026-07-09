"""Offline tests for the versioned release client (task 2.1, R1.1–R1.6).

All transport is mocked (``requests.get`` monkeypatched) and tarballs are
built in memory — no live network in the default run. Covers: successful
fetch with provenance, every error class (network/missing/auth/unpack),
the no-stale-substitution rule, the dormant token path, and the
keyring-then-environment token lookup order.
"""

import hashlib
import io
import os
import sys
import tarfile
import types

import pytest

from factor_workbook import release
from factor_workbook.release import (
    FetchError,
    Provenance,
    ReleaseClient,
    ReleaseError,
    default_token_provider,
)

TAG = "data-v1"
ASSET = "factor_views_v1.parquet"
PUBLIC_URL = (
    "https://github.com/norandom/Global_Macro_AI_Factors/releases/download/"
    f"{TAG}/{ASSET}"
)
API_RELEASE_URL = (
    "https://api.github.com/repos/norandom/Global_Macro_AI_Factors/"
    f"releases/tags/{TAG}"
)


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code: int, content: bytes = b"", json_data=None):
        self.status_code = status_code
        self.content = content
        self._json = json_data

    def json(self):
        return self._json


def install_transport(monkeypatch, handler):
    """Replace ``requests.get`` with ``handler(url, headers)``; return the call log."""
    calls: list[tuple[str, dict]] = []

    def fake_get(url, headers=None, timeout=None, **kwargs):
        headers = dict(headers or {})
        calls.append((url, headers))
        return handler(url, headers)

    monkeypatch.setattr(release.requests, "get", fake_get)
    return calls


def make_targz(members: dict[str, bytes]) -> bytes:
    """Build an in-memory ``.tar.gz`` with the given member files."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def no_token() -> None:
    return None


# --- successful fetch + provenance (R1.1, R1.2) ---------------------------


def test_fetch_success_records_provenance(monkeypatch, tmp_path):
    payload = b"parquet-bytes"
    install_transport(monkeypatch, lambda url, h: FakeResponse(200, payload))
    client = ReleaseClient(TAG, cache_dir=tmp_path, token_provider=no_token)

    data, prov = client.fetch(ASSET)

    assert data == payload
    assert isinstance(prov, Provenance)
    assert prov.tag == TAG
    assert prov.asset == ASSET
    assert prov.url == PUBLIC_URL
    assert prov.sha256 == hashlib.sha256(payload).hexdigest()
    assert prov.from_cache is False
    assert prov.fetched_at  # ISO timestamp present
    assert client.provenance_table() == [prov]


def test_provenance_table_accumulates_per_asset(monkeypatch, tmp_path):
    install_transport(monkeypatch, lambda url, h: FakeResponse(200, b"x"))
    client = ReleaseClient(TAG, cache_dir=tmp_path, token_provider=no_token)

    client.fetch("a.parquet")
    client.fetch("b.json")

    assert [p.asset for p in client.provenance_table()] == ["a.parquet", "b.json"]


def test_second_fetch_served_from_cache(monkeypatch, tmp_path):
    payload = b"cached-bytes"
    install_transport(monkeypatch, lambda url, h: FakeResponse(200, payload))
    client = ReleaseClient(TAG, cache_dir=tmp_path, token_provider=no_token)
    client.fetch(ASSET)

    def boom(url, headers):  # any further network use is a failure
        raise AssertionError("network must not be touched on a cache hit")

    install_transport(monkeypatch, boom)
    fresh = ReleaseClient(TAG, cache_dir=tmp_path, token_provider=no_token)
    data, prov = fresh.fetch(ASSET)

    assert data == payload
    assert prov.from_cache is True
    assert prov.sha256 == hashlib.sha256(payload).hexdigest()


# --- error taxonomy (R1.4) --------------------------------------------------


def test_network_error_raises_typed(monkeypatch, tmp_path):
    def down(url, headers):
        raise release.requests.ConnectionError("boom")

    install_transport(monkeypatch, down)
    client = ReleaseClient(TAG, cache_dir=tmp_path, token_provider=no_token)

    with pytest.raises(ReleaseError) as exc:
        client.fetch(ASSET)
    assert exc.value.error == FetchError(ASSET, "network", exc.value.error.detail)
    assert exc.value.error.cause == "network"
    assert exc.value.error.asset == ASSET


def test_missing_asset_404(monkeypatch, tmp_path):
    install_transport(monkeypatch, lambda url, h: FakeResponse(404))
    client = ReleaseClient(TAG, cache_dir=tmp_path, token_provider=no_token)

    with pytest.raises(ReleaseError) as exc:
        client.fetch(ASSET)
    assert exc.value.error.cause == "missing"
    assert exc.value.error.asset == ASSET


def test_auth_refusal_403_without_token(monkeypatch, tmp_path):
    install_transport(monkeypatch, lambda url, h: FakeResponse(403))
    client = ReleaseClient(TAG, cache_dir=tmp_path, token_provider=no_token)

    with pytest.raises(ReleaseError) as exc:
        client.fetch(ASSET)
    assert exc.value.error.cause == "auth"


def test_server_error_maps_to_network(monkeypatch, tmp_path):
    install_transport(monkeypatch, lambda url, h: FakeResponse(500))
    client = ReleaseClient(TAG, cache_dir=tmp_path, token_provider=no_token)

    with pytest.raises(ReleaseError) as exc:
        client.fetch(ASSET)
    assert exc.value.error.cause == "network"


def test_unpack_error_on_corrupt_tarball(monkeypatch, tmp_path):
    install_transport(monkeypatch, lambda url, h: FakeResponse(200, b"not a tar"))
    client = ReleaseClient(TAG, cache_dir=tmp_path, token_provider=no_token)

    with pytest.raises(ReleaseError) as exc:
        client.fetch_tar_member("bundle.tar.gz", "evidence.parquet")
    assert exc.value.error.cause == "unpack"
    assert exc.value.error.asset == "bundle.tar.gz"


def test_unpack_error_on_missing_member(monkeypatch, tmp_path):
    tar_bytes = make_targz({"other.json": b"{}"})
    install_transport(monkeypatch, lambda url, h: FakeResponse(200, tar_bytes))
    client = ReleaseClient(TAG, cache_dir=tmp_path, token_provider=no_token)

    with pytest.raises(ReleaseError) as exc:
        client.fetch_tar_member("bundle.tar.gz", "evidence.parquet")
    assert exc.value.error.cause == "unpack"
    assert "evidence.parquet" in exc.value.error.detail


def test_fetch_tar_member_success(monkeypatch, tmp_path):
    member_bytes = b"member-payload"
    tar_bytes = make_targz({"dir/evidence.parquet": member_bytes})
    install_transport(monkeypatch, lambda url, h: FakeResponse(200, tar_bytes))
    client = ReleaseClient(TAG, cache_dir=tmp_path, token_provider=no_token)

    data, prov = client.fetch_tar_member("bundle.tar.gz", "dir/evidence.parquet")

    assert data == member_bytes
    assert prov.asset == "bundle.tar.gz"
    assert prov.sha256 == hashlib.sha256(tar_bytes).hexdigest()


# --- no stale substitution (R1.4) + per-tag client (R1.5) -------------------


def test_failed_refresh_never_serves_other_tag_cache(monkeypatch, tmp_path):
    install_transport(monkeypatch, lambda url, h: FakeResponse(200, b"v1-data"))
    ReleaseClient(TAG, cache_dir=tmp_path, token_provider=no_token).fetch(ASSET)

    install_transport(monkeypatch, lambda url, h: FakeResponse(404))
    v2 = ReleaseClient("data-v2", cache_dir=tmp_path, token_provider=no_token)

    with pytest.raises(ReleaseError) as exc:
        v2.fetch(ASSET)
    assert exc.value.error.cause == "missing"
    assert v2.provenance_table() == []  # nothing recorded for the failure


def test_failed_fetch_with_empty_cache_raises_not_substitutes(monkeypatch, tmp_path):
    def down(url, headers):
        raise release.requests.Timeout("timed out")

    install_transport(monkeypatch, down)
    client = ReleaseClient(TAG, cache_dir=tmp_path, token_provider=no_token)

    with pytest.raises(ReleaseError):
        client.fetch(ASSET)
    assert client.provenance_table() == []


def test_tag_is_immutable_new_client_per_version(tmp_path):
    client = ReleaseClient(TAG, cache_dir=tmp_path, token_provider=no_token)
    assert client.tag == TAG
    with pytest.raises(AttributeError):
        client.tag = "data-v2"


# --- dormant token path (R1.3) ----------------------------------------------


def test_token_provider_not_consulted_on_success(monkeypatch, tmp_path):
    def forbidden() -> str:
        raise AssertionError("token provider must not be consulted on success")

    install_transport(monkeypatch, lambda url, h: FakeResponse(200, b"ok"))
    client = ReleaseClient(TAG, cache_dir=tmp_path, token_provider=forbidden)
    data, _ = client.fetch(ASSET)
    assert data == b"ok"


def test_token_retry_via_api_on_refusal(monkeypatch, tmp_path):
    token = "sekrit-token-value"
    asset_api_url = "https://api.github.com/repos/norandom/Global_Macro_AI_Factors/releases/assets/1"
    payload = b"private-bytes"

    def handler(url, headers):
        if url == PUBLIC_URL:
            return FakeResponse(404)
        if url == API_RELEASE_URL:
            assert headers["Authorization"] == f"Bearer {token}"
            return FakeResponse(
                200, json_data={"assets": [{"name": ASSET, "url": asset_api_url}]}
            )
        if url == asset_api_url:
            assert headers["Authorization"] == f"Bearer {token}"
            assert headers["Accept"] == "application/octet-stream"
            return FakeResponse(200, payload)
        raise AssertionError(f"unexpected url {url}")

    install_transport(monkeypatch, handler)
    client = ReleaseClient(TAG, cache_dir=tmp_path, token_provider=lambda: token)

    data, prov = client.fetch(ASSET)

    assert data == payload
    assert prov.url == asset_api_url
    # the token never leaks into provenance or the cache (R1.3)
    for value in (prov.tag, prov.asset, prov.url, prov.fetched_at, prov.sha256):
        assert token not in str(value)
    for path in tmp_path.rglob("*"):
        assert token not in str(path)


def test_token_never_in_error_detail(monkeypatch, tmp_path):
    token = "sekrit-token-value"

    def handler(url, headers):
        if url == PUBLIC_URL:
            return FakeResponse(403)
        return FakeResponse(403)  # API refuses the token too

    install_transport(monkeypatch, handler)
    client = ReleaseClient(TAG, cache_dir=tmp_path, token_provider=lambda: token)

    with pytest.raises(ReleaseError) as exc:
        client.fetch(ASSET)
    assert exc.value.error.cause == "auth"
    assert token not in exc.value.error.detail
    assert token not in str(exc.value)


# --- token lookup order: keyring then environment (R1.3) --------------------


def _fake_keyring(monkeypatch, value):
    fake = types.ModuleType("keyring")
    fake.get_password = lambda service, user: (
        value if (service, user) == ("factor-workbook", "github") else None
    )
    monkeypatch.setitem(sys.modules, "keyring", fake)


def test_default_provider_prefers_keyring(monkeypatch):
    _fake_keyring(monkeypatch, "from-keyring")
    monkeypatch.setenv("GITHUB_TOKEN", "from-env")
    assert default_token_provider() == "from-keyring"


def test_default_provider_falls_back_to_env(monkeypatch):
    _fake_keyring(monkeypatch, None)
    monkeypatch.setenv("GITHUB_TOKEN", "from-env")
    assert default_token_provider() == "from-env"


def test_default_provider_env_when_keyring_unimportable(monkeypatch):
    monkeypatch.setitem(sys.modules, "keyring", None)  # forces ImportError
    monkeypatch.setenv("GITHUB_TOKEN", "from-env")
    assert default_token_provider() == "from-env"


def test_default_provider_none_when_nothing_configured(monkeypatch):
    monkeypatch.setitem(sys.modules, "keyring", None)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert default_token_provider() is None
