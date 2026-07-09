"""Versioned GitHub Release client: fetch, disk cache, tar.gz unpack,
provenance records, and the dormant token path (R1.1–R1.6).

Assets are addressed exclusively by an explicit release tag via the public
download URL ``https://github.com/norandom/Global_Macro_AI_Factors/releases/
download/<tag>/<asset>``. Release tags are immutable, so the on-disk cache is
keyed by ``(tag, asset)`` and a cache hit is exact — never stale. A failed
fetch raises a typed :class:`ReleaseError` and never serves substitute data
(R1.4). Switching versions constructs a new client (R1.5); the client only
ever writes its own cache directory (R1.6).

Dormant authenticated path (R1.3): when the unauthenticated address is
refused (HTTP 403/404) and a token provider yields a token, the client
retries once through the GitHub API — it resolves the release by tag
(``/repos/<owner>/<repo>/releases/tags/<tag>``) and downloads the matching
asset endpoint with headers ``Authorization: Bearer <token>`` and
``Accept: application/octet-stream``. The token never appears in provenance,
error details, or any persisted artifact.
"""

import hashlib
import io
import os
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

_REPO = "norandom/Global_Macro_AI_Factors"
_DOWNLOAD_URL = "https://github.com/" + _REPO + "/releases/download/{tag}/{asset}"
_API_RELEASE_URL = "https://api.github.com/repos/" + _REPO + "/releases/tags/{tag}"
_TIMEOUT = 30.0
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


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
    """

    tag: str
    asset: str
    url: str
    fetched_at: str
    sha256: str
    from_cache: bool


@dataclass(frozen=True)
class FetchError:
    """Typed per-asset failure (R1.4).

    Attributes:
        asset: The asset whose retrieval failed.
        cause: One of ``"network"``, ``"missing"``, ``"auth"``, ``"unpack"``.
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
        self._tag = tag
        self._cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
        self._token_provider = token_provider or default_token_provider
        self._provenance: dict[str, Provenance] = {}

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
            ReleaseError: With cause ``network``, ``missing``, or ``auth``.
        """
        cache_path = self._cache_dir / self._tag / asset
        if cache_path.exists():
            data = cache_path.read_bytes()
            return data, self._record(asset, self._public_url(asset), data, from_cache=True)

        url = self._public_url(asset)
        response = self._get(asset, url)
        if response.status_code == 200:
            data = response.content
        elif response.status_code in (403, 404):
            token = self._token_provider()
            if token is None:
                cause = "missing" if response.status_code == 404 else "auth"
                raise ReleaseError(FetchError(asset, cause, f"HTTP {response.status_code} at {url}"))
            data, url = self._fetch_via_api(asset, token)
        else:
            raise ReleaseError(
                FetchError(asset, "network", f"HTTP {response.status_code} at {url}")
            )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
        return data, self._record(asset, url, data, from_cache=False)

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

    def _record(self, asset: str, url: str, data: bytes, *, from_cache: bool) -> Provenance:
        provenance = Provenance(
            tag=self._tag,
            asset=asset,
            url=url,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            sha256=hashlib.sha256(data).hexdigest(),
            from_cache=from_cache,
        )
        self._provenance[asset] = provenance
        return provenance
