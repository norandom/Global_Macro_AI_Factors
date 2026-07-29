"""Versioned GitHub Release client: fetch, disk cache, tar.gz unpack,
provenance records, and the dormant token path (R1.1–R1.6).

Assets are addressed exclusively by an explicit release tag via the public
download URL ``https://github.com/norandom/Global_Macro_AI_Factors/releases/
download/<tag>/<asset>``. Release tags are immutable, so the on-disk cache is
keyed by ``(tag, asset)`` and a cache hit is exact, never stale. A failed
fetch raises a typed :class:`ReleaseError` and never serves substitute data
(R1.4). Switching versions constructs a new client (R1.5); the client only
ever writes its own cache directory (R1.6).

Dormant authenticated path (R1.3): when the unauthenticated address is
refused (HTTP 403/404) and a token provider yields a token, the client
retries once through the GitHub API: it resolves the release by tag
(``/repos/<owner>/<repo>/releases/tags/<tag>``) and downloads the matching
asset endpoint with headers ``Authorization: Bearer <token>`` and
``Accept: application/octet-stream``. The token never appears in provenance,
error details, or any persisted artifact.

Manifest-aware integrity for ``data-v4`` (task 10.2, R7.3–R8.8): asset names
must be listed in the release's completed ``publication_manifest.json`` and
the retrieved bytes must match the manifest's SHA-256 digest — recomputed
here from the observed bytes, never taken from caller-supplied lineage —
before they are cached or returned. Integrity failures raise ``ReleaseError``
with cause ``"integrity"``, evict the offending cache entry, and record
nothing. Historical tags ``data-v1``–``data-v3`` keep direct loading
unchanged (``verification="historical_direct"``).
"""

import hashlib
import io
import json
import os
import re
import shutil
import tarfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TypeGuard

import requests

_REPO = "norandom/Global_Macro_AI_Factors"
_DOWNLOAD_URL = "https://github.com/" + _REPO + "/releases/download/{tag}/{asset}"
_API_RELEASE_URL = "https://api.github.com/repos/" + _REPO + "/releases/tags/{tag}"
_TIMEOUT = 30.0
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
_DATA_V4_TAG = "data-v4"
_PUBLICATION_MANIFEST = "publication_manifest.json"
_PUBLICATION_MANIFEST_SCHEMA = "publication_manifest.v1"
_IMMUTABLE_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_MUTABLE_TAGS = frozenset({"latest", "current", "data-current"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RELEASE_CONTROL_FILES = frozenset({_PUBLICATION_MANIFEST, "SHA256SUMS", "COMPLETED"})


def _is_public_basename(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value) and PurePosixPath(value).name == value


@dataclass(frozen=True)
class Provenance:
    """Retrieval record for one loaded asset (R1.2).

    Attributes:
        tag: Release version identifier, e.g. ``"data-v1"``.
        asset: Release asset name.
        url: Resolved download address (never contains a token).
        fetched_at: ISO-8601 UTC timestamp of this retrieval.
        sha256: Hex digest of the retrieved bytes.
        from_cache: Whether the bytes came from the on-disk cache.
        expected_sha256: Manifest digest for verified assets, otherwise None.
        verified: Whether the bytes passed manifest integrity verification.
        verification: Verification method or historical direct-loading marker.
    """

    tag: str
    asset: str
    url: str
    fetched_at: str
    sha256: str
    from_cache: bool
    expected_sha256: str | None = None
    verified: bool = False
    verification: str = "historical_direct"


@dataclass(frozen=True)
class FetchError:
    """Typed per-asset failure (R1.4).

    Attributes:
        asset: The asset whose retrieval failed.
        cause: One of ``"network"``, ``"missing"``, ``"auth"``, ``"unpack"``,
            or ``"integrity"``.
        detail: Human-readable detail (never contains a token).
    """

    asset: str
    cause: str
    detail: str


class ReleaseError(Exception):
    """Raised on any failed retrieval; carries the typed :class:`FetchError`."""

    def __init__(self, error: FetchError) -> None:
        self.error = error
        super().__init__(f"{error.asset}: {error.cause}: {error.detail}")


def default_token_provider() -> str | None:
    """Look up a GitHub token: system keychain first, then environment.

    Tries ``keyring`` (service ``factor-workbook``, user ``github``) with a
    lazy import so the optional extra is never required, then falls back to
    the ``GITHUB_TOKEN`` environment variable. Returns None when neither is
    configured.
    """
    try:
        import keyring  # ponytail: lazy — keyring is an optional extra

        token = keyring.get_password("factor-workbook", "github")
        if token:
            return token
    except Exception:
        pass
    return os.environ.get("GITHUB_TOKEN") or None


class ReleaseClient:
    """Read-only, cached, provenance-tracked access to one release version.

    The tag is immutable per client: changing the release version is an
    explicit action that constructs a new client (R1.5).

    Args:
        tag: Explicit release version identifier, e.g. ``"data-v1"``.
        cache_dir: On-disk cache root; defaults to ``workbook/.cache/``.
        token_provider: Zero-argument callable yielding a token or None;
            defaults to :func:`default_token_provider`. Consulted only when
            the unauthenticated address is refused.
    """

    def __init__(
        self,
        tag: str,
        cache_dir: Path | None = None,
        token_provider: Callable[[], str | None] | None = None,
    ) -> None:
        if (
            not isinstance(tag, str)
            or not _IMMUTABLE_TAG_RE.fullmatch(tag)
            or tag.casefold() in _MUTABLE_TAGS
        ):
            raise ValueError("tag must be an explicit immutable release tag")
        self._tag = tag
        self._cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
        self._token_provider = token_provider or default_token_provider
        self._provenance: dict[str, Provenance] = {}
        self._manifest_assets: dict[str, tuple[str, int | None]] | None = None

    @property
    def tag(self) -> str:
        """The immutable release version this client serves."""
        return self._tag

    def fetch(self, asset: str) -> tuple[bytes, Provenance]:
        """Retrieve one release asset's bytes with its provenance record.

        Serves the exact ``(tag, asset)`` cache entry when present (release
        tags are immutable); otherwise downloads, caches, and records
        provenance. A failure raises :class:`ReleaseError` and never yields
        substitute or stale data (R1.4).

        Raises:
            ReleaseError: With cause ``network``, ``missing``, ``auth``, or
                ``integrity``.
        """
        if not _is_public_basename(asset):
            raise ReleaseError(
                FetchError(str(asset), "integrity", "asset must be a public basename")
            )
        if self._tag == _DATA_V4_TAG and asset == _PUBLICATION_MANIFEST:
            raise ReleaseError(
                FetchError(asset, "integrity", "publication manifest is release-control metadata")
            )

        if self._tag == _DATA_V4_TAG:
            expected_sha256, expected_size = self._manifest_entry(asset)
            return self._fetch_verified(asset, expected_sha256, expected_size)
        return self._fetch_historical(asset)

    def _fetch_historical(self, asset: str) -> tuple[bytes, Provenance]:
        cache_path = self._cache_path(asset)
        if cache_path.exists():
            data = cache_path.read_bytes()
            return data, self._record(
                asset,
                self._public_url(asset),
                data,
                from_cache=True,
            )

        data, url = self._download(asset)
        self._write_cache(cache_path, data)
        return data, self._record(asset, url, data, from_cache=False)

    def _fetch_verified(
        self,
        asset: str,
        expected_sha256: str,
        expected_size: int | None,
    ) -> tuple[bytes, Provenance]:
        cache_path = self._cache_path(asset)
        from_cache = cache_path.exists()
        if from_cache:
            data = cache_path.read_bytes()
            url = self._public_url(asset)
        else:
            data, url = self._download(asset)

        try:
            self._verify_bytes(
                asset,
                data,
                expected_sha256,
                expected_size,
                source="cache" if from_cache else "download",
            )
        except ReleaseError:
            if from_cache:
                self._invalidate(cache_path)
            raise
        if not from_cache:
            self._write_cache(cache_path, data)

        return data, self._record(
            asset,
            url,
            data,
            from_cache=from_cache,
            expected_sha256=expected_sha256,
            verified=True,
            verification="publication_manifest_sha256",
        )

    def fetch_tar_member(self, asset: str, member: str) -> tuple[bytes, Provenance]:
        """Retrieve one member file from a bundled ``.tar.gz`` release asset.

        Raises:
            ReleaseError: Fetch failures as in :meth:`fetch`; a corrupt
                archive or missing member raises with cause ``unpack``.
        """
        data, provenance = self.fetch(asset)
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise KeyError(member)
                return extracted.read(), provenance
        except (tarfile.TarError, KeyError) as exc:
            raise ReleaseError(
                FetchError(asset, "unpack", f"member {member!r}: {exc}")
            ) from exc

    def provenance_table(self) -> list[Provenance]:
        """Provenance of everything loaded so far, one row per asset (R1.2)."""
        return list(self._provenance.values())

    def _manifest_entry(self, asset: str) -> tuple[str, int | None]:
        assets = self._load_manifest()
        try:
            return assets[asset]
        except KeyError as exc:
            raise ReleaseError(
                FetchError(asset, "integrity", f"asset is not listed in {_PUBLICATION_MANIFEST}")
            ) from exc

    def _load_manifest(self) -> dict[str, tuple[str, int | None]]:
        if self._manifest_assets is not None:
            return self._manifest_assets

        cache_path = self._cache_path(_PUBLICATION_MANIFEST)
        from_cache = cache_path.exists()
        data = (
            cache_path.read_bytes()
            if from_cache
            else self._download(_PUBLICATION_MANIFEST)[0]
        )
        try:
            assets = self._parse_manifest(data)
        except ReleaseError:
            self._invalidate_tag_cache()
            raise
        if not from_cache:
            self._write_cache(cache_path, data)

        self._manifest_assets = assets
        return assets

    def _parse_manifest(self, data: bytes) -> dict[str, tuple[str, int | None]]:
        try:
            manifest = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReleaseError(
                FetchError(_PUBLICATION_MANIFEST, "integrity", f"invalid JSON: {exc}")
            ) from exc
        if not isinstance(manifest, Mapping):
            raise self._manifest_error("top level must be an object")
        schema_id = manifest.get("schema_id")
        if schema_id != _PUBLICATION_MANIFEST_SCHEMA:
            raise self._manifest_error(
                f"schema_id must be {_PUBLICATION_MANIFEST_SCHEMA!r}"
            )
        if manifest.get("release_tag") != self._tag:
            raise self._manifest_error(f"release_tag must equal {self._tag!r}")
        if manifest.get("completed") is not True:
            raise self._manifest_error("completed must be true")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise self._manifest_error("artifacts must be a non-empty list")

        parsed: dict[str, tuple[str, int | None]] = {}
        for index, row in enumerate(artifacts):
            if not isinstance(row, Mapping):
                raise self._manifest_error(f"artifacts[{index}] must be an object")
            path = row.get("path")
            digest = row.get("sha256")
            size = row.get("size")
            if not _is_public_basename(path):
                raise self._manifest_error(
                    f"artifacts[{index}].path must be a public basename"
                )
            if path in _RELEASE_CONTROL_FILES:
                raise self._manifest_error(f"artifacts[{index}].path is a control file")
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise self._manifest_error(
                    f"artifacts[{index}].sha256 must be a lowercase SHA-256 digest"
                )
            if size is not None and (
                not isinstance(size, int) or isinstance(size, bool) or size < 0
            ):
                raise self._manifest_error(
                    f"artifacts[{index}].size must be a non-negative integer"
                )
            if path in parsed:
                raise self._manifest_error(f"duplicate artifact path {path!r}")
            parsed[path] = (digest, size)
        return parsed

    @staticmethod
    def _manifest_error(detail: str) -> ReleaseError:
        return ReleaseError(FetchError(_PUBLICATION_MANIFEST, "integrity", detail))

    def _verify_bytes(
        self,
        asset: str,
        data: bytes,
        expected_sha256: str,
        expected_size: int | None,
        *,
        source: str,
    ) -> None:
        actual_sha256 = hashlib.sha256(data).hexdigest()
        size_matches = expected_size is None or len(data) == expected_size
        if actual_sha256 != expected_sha256 or not size_matches:
            size_detail = ""
            if not size_matches:
                size_detail = f"; expected size {expected_size}, got {len(data)}"
            raise ReleaseError(
                FetchError(
                    asset,
                    "integrity",
                    f"{source} verification failed: expected sha256 "
                    f"{expected_sha256}, got {actual_sha256}{size_detail}",
                )
            )

    def _download(self, asset: str) -> tuple[bytes, str]:
        url = self._public_url(asset)
        response = self._get(asset, url)
        if response.status_code == 200:
            return response.content, url
        if response.status_code in (403, 404):
            token = self._token_provider()
            if token is None:
                cause = "missing" if response.status_code == 404 else "auth"
                raise ReleaseError(
                    FetchError(asset, cause, f"HTTP {response.status_code} at {url}")
                )
            return self._fetch_via_api(asset, token)
        raise ReleaseError(
            FetchError(asset, "network", f"HTTP {response.status_code} at {url}")
        )

    def _cache_path(self, asset: str) -> Path:
        return self._cache_dir / self._tag / asset

    @staticmethod
    def _write_cache(cache_path: Path, data: bytes) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)

    @staticmethod
    def _invalidate(cache_path: Path) -> None:
        try:
            cache_path.unlink()
        except FileNotFoundError:
            pass

    def _invalidate_tag_cache(self) -> None:
        shutil.rmtree(self._cache_dir / self._tag, ignore_errors=True)

    def _public_url(self, asset: str) -> str:
        return _DOWNLOAD_URL.format(tag=self._tag, asset=asset)

    def _get(self, asset: str, url: str, headers: dict[str, str] | None = None):
        try:
            return requests.get(url, headers=headers, timeout=_TIMEOUT)
        except requests.RequestException as exc:
            raise ReleaseError(FetchError(asset, "network", str(exc))) from exc

    def _fetch_via_api(self, asset: str, token: str) -> tuple[bytes, str]:
        """Authenticated retry via the GitHub API asset endpoint (R1.3)."""
        auth = {"Authorization": f"Bearer {token}"}
        release_url = _API_RELEASE_URL.format(tag=self._tag)
        response = self._get(asset, release_url, headers=auth)
        if response.status_code in (401, 403):
            raise ReleaseError(
                FetchError(asset, "auth", f"HTTP {response.status_code} from release API for tag {self._tag}")
            )
        if response.status_code == 404:
            raise ReleaseError(
                FetchError(asset, "missing", f"release tag {self._tag} not found via API")
            )
        if response.status_code != 200:
            raise ReleaseError(
                FetchError(asset, "network", f"HTTP {response.status_code} from release API")
            )

        matches = [a for a in response.json().get("assets", []) if a.get("name") == asset]
        if not matches:
            raise ReleaseError(
                FetchError(asset, "missing", f"asset not present in release {self._tag}")
            )
        asset_url = matches[0]["url"]

        response = self._get(
            asset, asset_url, headers={**auth, "Accept": "application/octet-stream"}
        )
        if response.status_code in (401, 403):
            raise ReleaseError(
                FetchError(asset, "auth", f"HTTP {response.status_code} at {asset_url}")
            )
        if response.status_code == 404:
            raise ReleaseError(FetchError(asset, "missing", f"HTTP 404 at {asset_url}"))
        if response.status_code != 200:
            raise ReleaseError(
                FetchError(asset, "network", f"HTTP {response.status_code} at {asset_url}")
            )
        return response.content, asset_url

    def _record(
        self,
        asset: str,
        url: str,
        data: bytes,
        *,
        from_cache: bool,
        expected_sha256: str | None = None,
        verified: bool = False,
        verification: str = "historical_direct",
    ) -> Provenance:
        provenance = Provenance(
            tag=self._tag,
            asset=asset,
            url=url,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            sha256=hashlib.sha256(data).hexdigest(),
            from_cache=from_cache,
            expected_sha256=expected_sha256,
            verified=verified,
            verification=verification,
        )
        self._provenance[asset] = provenance
        return provenance
