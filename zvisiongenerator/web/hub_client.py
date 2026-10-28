"""Anonymous, bounded reads of Hugging Face for the Discover LoRAs routes.

Every read goes through one :data:`Fetch` callable, so the rules live in one place and tests replace it: the URL is
built here on ``huggingface.co``, never carries the user's token or any other credential, and its body is cut off at
a byte cap. :func:`validated_repo`, :func:`validated_file` and :func:`validated_revision` are the only way a request
value reaches a URL.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import re
import threading
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode, urljoin, urlsplit

if TYPE_CHECKING:
    import httpx

__all__ = [
    "HEADER_READ_BYTES",
    "HUB_ROOT",
    "CachedFetch",
    "Fetch",
    "HubError",
    "HubResponse",
    "api_url",
    "http_fetch",
    "resolve_url",
    "validated_file",
    "validated_repo",
    "validated_revision",
]

HUB_ROOT = "https://huggingface.co"
# The most of a remote safetensors file a header read takes.
HEADER_READ_BYTES = 2 * 1024 * 1024
# The most of a JSON answer from the Hub a read takes.
JSON_READ_BYTES = 8 * 1024 * 1024

_TIMEOUT_SECS = 15.0
_MAX_REDIRECTS = 5
# Where a Hub download may redirect to: the Hub itself and its CDN (``*.hf.co``).
_HUB_HOSTS = ("huggingface.co", "hf.co")
_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_MAX_FILE_CHARS = 300
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class HubError(RuntimeError):
    """Report that Hugging Face could not answer: *status* is its HTTP status, or None when it could not be reached."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class HubResponse:
    """One answer from the Hub: its status, lower-cased headers and at most the capped number of body bytes.

    Attributes:
        status: HTTP status of the final response.
        body: The body, cut off at the cap.
        headers: The response headers, names lower-cased.
        truncated: Whether the body was cut off at the cap.
    """

    status: int
    body: bytes
    headers: Mapping[str, str]
    truncated: bool = False


type Fetch = Callable[..., HubResponse]


def validated_repo(repo: str) -> str:
    """Return *repo* when it is an ``owner/name`` repository ID, else raise.

    Raises:
        ValueError: If *repo* is not a plain repository ID (a path, a URL or anything with ``..`` is refused).
    """
    if not isinstance(repo, str) or ".." in repo or _REPO.fullmatch(repo) is None:
        raise ValueError("repo must be a Hugging Face repository ID such as 'owner/name'.")
    return repo


def validated_file(file: str) -> str:
    """Return *file* when it is a relative ``.safetensors`` path in a repository, else raise.

    Raises:
        ValueError: If *file* has no ``.safetensors`` ending, contains ``..``, starts with ``/`` or ``\\``,
            holds a backslash or a control character, or is too long.
    """
    if (
        not isinstance(file, str)
        or not file.endswith(".safetensors")
        or len(file) > _MAX_FILE_CHARS
        or ".." in file
        or file.startswith(("/", "\\"))
        or "\\" in file
        or _CONTROL.search(file)
        or any(not part for part in file.split("/"))
    ):
        raise ValueError("file must be a .safetensors path inside the repository, without '..' and without a leading '/' or '\\'.")
    return file


def validated_revision(revision: str) -> str:
    """Return *revision* when it is a 40-character commit hash, else raise.

    Raises:
        ValueError: If *revision* is not a commit hash.
    """
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("revision must be a 40-character commit hash.")
    return revision


def api_url(path: str, params: Mapping[str, Any] | None = None, *, repeated: Mapping[str, tuple[str, ...]] | None = None) -> str:
    """Return the Hub API URL for *path* (for example ``models``), with *params* and the *repeated* multi-value parameters encoded."""
    pairs: list[tuple[str, str]] = [(key, str(value)) for key, value in (params or {}).items() if value is not None]
    pairs += [(key, value) for key, values in (repeated or {}).items() for value in values]
    query = f"?{urlencode(pairs)}" if pairs else ""
    return f"{HUB_ROOT}/api/{path}{query}"


def resolve_url(repo: str, revision: str, file: str) -> str:
    """Return the Hub URL that serves *file* of *repo* at the commit *revision*; every part is validated."""
    return f"{HUB_ROOT}/{validated_repo(repo)}/resolve/{validated_revision(revision)}/{quote(validated_file(file), safe='/')}"


def _allowed_url(url: str) -> bool:
    """Return whether *url* is https on the Hub or its CDN."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    return parts.scheme == "https" and parts.username is None and any(host == base or host.endswith(f".{base}") for base in _HUB_HOSTS)


def http_fetch(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    max_bytes: int = JSON_READ_BYTES,
    transport: httpx.BaseTransport | None = None,
) -> HubResponse:
    """GET *url* without any credential and return at most *max_bytes* of its body.

    The reader follows redirects itself and only to https URLs on the Hub or its CDN. It stops reading once the cap
    is reached, whether or not the server honoured a ``Range`` header, and never sends ``Authorization``.

    Args:
        url: A ``https://huggingface.co`` URL.
        headers: Extra request headers such as ``Range``; any ``Authorization`` or ``Cookie`` entry is dropped.
        max_bytes: The most body bytes to read.
        transport: An ``httpx`` transport, for tests.

    Raises:
        ValueError: If *url* is not on the Hub, or a redirect leaves it.
        HubError: If the Hub cannot be reached or redirects too often.
    """
    from huggingface_hub.utils import httpx

    if not _allowed_url(url):
        raise ValueError("Only https://huggingface.co URLs can be read.")
    sent = {key: value for key, value in (headers or {}).items() if key.lower() not in {"authorization", "cookie", "proxy-authorization"}}
    try:
        with httpx.Client(timeout=_TIMEOUT_SECS, follow_redirects=False, transport=transport) as client:
            current = url
            for _ in range(_MAX_REDIRECTS + 1):
                with client.stream("GET", current, headers=sent) as response:
                    if response.is_redirect and response.headers.get("location"):
                        current = urljoin(current, response.headers["location"])
                        if not _allowed_url(current):
                            raise ValueError("Hugging Face redirected outside huggingface.co.")
                        continue
                    return _read_capped(response, max_bytes)
    except httpx.HTTPError as exc:
        raise HubError("Could not reach Hugging Face.") from exc
    raise HubError("Hugging Face redirected too many times.")


def _read_capped(response: httpx.Response, max_bytes: int) -> HubResponse:
    """Read *response*'s body up to *max_bytes* and report whether more was left."""
    chunks: list[bytes] = []
    total = 0
    truncated = False
    for chunk in response.iter_bytes():
        room = max_bytes - total
        if len(chunk) > room:
            chunks.append(chunk[:room])
            total = max_bytes
            truncated = True
            break
        chunks.append(chunk)
        total += len(chunk)
    return HubResponse(response.status_code, b"".join(chunks), {key.lower(): value for key, value in response.headers.items()}, truncated)


class CachedFetch:
    """Wrap a :data:`Fetch` and keep its successful answers briefly in memory, so a page of results is read from the Hub once.

    The cache is bounded in bytes as well as in entries: an answer with a body over *max_entry_bytes* is not kept,
    the bodies kept never add up to more than *max_total_bytes*, and an answer past its time to live is dropped as
    soon as the cache is next used.

    Args:
        fetch: The function that reads the Hub.
        ttl_secs: How long an answer is kept.
        max_entries: The most answers kept; the least recently used goes first.
        max_entry_bytes: The largest body worth keeping.
        max_total_bytes: The most body bytes kept in all; the least recently used go first.
        clock: Monotonic clock.
    """

    def __init__(
        self,
        fetch: Fetch = http_fetch,
        *,
        ttl_secs: float = 300.0,
        max_entries: int = 128,
        max_entry_bytes: int = 1024 * 1024,
        max_total_bytes: int = 16 * 1024 * 1024,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fetch = fetch
        self._ttl_secs = ttl_secs
        self._max_entries = max_entries
        self._max_entry_bytes = max_entry_bytes
        self._max_total_bytes = max_total_bytes
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, str, int], tuple[float, HubResponse]] = OrderedDict()
        self._bytes = 0

    def __call__(self, url: str, *, headers: Mapping[str, str] | None = None, max_bytes: int = JSON_READ_BYTES) -> HubResponse:
        """Return the answer for *url*, from memory when a recent one is kept."""
        key = (url, (headers or {}).get("Range", ""), max_bytes)
        now = self._clock()
        with self._lock:
            self._purge(now)
            kept = self._entries.get(key)
            if kept is not None:
                self._entries.move_to_end(key)
                return kept[1]
        response = self._fetch(url, headers=headers, max_bytes=max_bytes)
        if response.status in (200, 206) and len(response.body) <= self._max_entry_bytes:
            with self._lock:
                self._drop(key)
                self._entries[key] = (now, response)
                self._bytes += len(response.body)
                while len(self._entries) > self._max_entries or self._bytes > self._max_total_bytes:
                    self._drop(next(iter(self._entries)))
        return response

    def _drop(self, key: tuple[str, str, int]) -> None:
        """Forget the answer kept for *key*, if any (the caller holds the lock)."""
        removed = self._entries.pop(key, None)
        if removed is not None:
            self._bytes -= len(removed[1].body)

    def _purge(self, now: float) -> None:
        """Forget every answer older than the time to live (the caller holds the lock)."""
        for key in [key for key, (kept_at, _) in self._entries.items() if now - kept_at >= self._ttl_secs]:
            self._drop(key)
