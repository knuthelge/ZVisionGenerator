"""Read LoRAs from Hugging Face for Discover: adapters per base model, one repository's files and card, and a file's family.

Everything here takes a :data:`~zvisiongenerator.web.hub_client.Fetch`, so it reads the Hub anonymously and boundedly
and tests need no network. Gated and private repositories, and repositories the content filter
(:mod:`~zvisiongenerator.web.listing_filter`) hides, are dropped from every listing and refused as a detail.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import re
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

from zvisiongenerator.utils.lora_family import detect_lora_family
from zvisiongenerator.utils.lora_trigger import card_triggers
from zvisiongenerator.utils.model_files import disk_size_label
from zvisiongenerator.utils.safetensors_stream import header_and_metadata_from_bytes
from zvisiongenerator.web.listing_filter import ListingFilter, is_filtered
from zvisiongenerator.web.hub_client import HEADER_READ_BYTES, HUB_ROOT, Fetch, HubError, api_url, resolve_url, validated_file, validated_repo

__all__ = [
    "ADULT_TAG",
    "BASE_MODELS",
    "SORTS",
    "FamilyCheck",
    "RepoDetail",
    "RepoFile",
    "base_repos",
    "check_family",
    "choose_files",
    "clean_repo_name",
    "list_adapters",
    "preview_url",
    "read_repo",
]

ADULT_TAG = "not-for-all-audiences"
PAGE_SIZE = 30
# How many more Hub pages a listing reads when every row of a page was dropped.
_EXTRA_PAGES = 2
MAX_QUERY_CHARS = 100
# The official base repositories whose adapters the Hub lists, by family and size or version; ``None`` is the whole family.
# A family listed by size or version has no ``None`` entry: asking for it without either lists every one.
BASE_MODELS: dict[str, dict[str | None, tuple[str, ...]]] = {
    "zimage": {None: ("Tongyi-MAI/Z-Image-Turbo",)},
    "krea2": {None: ("krea/Krea-2-Turbo",)},
    "flux1": {None: ("black-forest-labs/FLUX.1-dev",)},
    "flux2_klein": {"4b": ("black-forest-labs/FLUX.2-klein-4B",), "9b": ("black-forest-labs/FLUX.2-klein-9B",)},
    "ideogram4": {None: ("ideogram-ai/ideogram-4-fp8",)},
    "moss": {None: ("OpenMOSS-Team/MOSS-Image-SFT",)},
    "ltx": {"2.3": ("Lightricks/LTX-2.3",), "2.5": ("Lightricks/LTX-2.5",)},
    "h3": {None: ("MiniMaxAI/MiniMax-H3",)},
}
# The sorts a listing offers, by the name the route takes, and the Hub's own name for each.
SORTS: dict[str, str] = {"downloads": "downloads", "likes": "likes", "trending": "trendingScore", "newest": "createdAt"}
_LIST_FIELDS = ("downloads", "likes", "trendingScore", "tags", "gated", "private", "cardData", "siblings", "createdAt", "lastModified", "author")
_SORT_VALUES: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "downloads": lambda row: row["downloads"],
    "likes": lambda row: row["likes"],
    "trending": lambda row: row["trending_score"],
    "newest": lambda row: str(row["created_at"] or ""),
}
_PREVIEW_IMAGES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})
_PREVIEW_VIDEOS = frozenset({".mp4", ".webm"})
_PREVIEW_HOSTS = frozenset({"huggingface.co", "cdn-uploads.huggingface.co"})
_MAX_PREVIEWS = 6
# A safetensors file opens with its header length, 8 bytes, little-endian.
_LENGTH_BYTES = 8
_CURSOR_VALUE = re.compile(r"[A-Za-z0-9+/=_.\-]{1,2048}")
_STEP = re.compile(r"(?<![0-9])(\d{4,})(?![0-9])")
_STEP_WORD = re.compile(r"step[_\- ]?(\d+)", re.IGNORECASE)
_MARKED = re.compile(r"(?<![a-z])(recommended|final|best)(?![a-z])", re.IGNORECASE)
_NAME_BREAKS = re.compile(r"[-_]+")


@dataclass(frozen=True)
class RepoFile:
    """One ``.safetensors`` file of a repository.

    Attributes:
        file: Path in the repository.
        size_bytes: Size in bytes, or None when the Hub did not say.
        sha256: The Hub's ``lfs.sha256``, or None for a file stored without LFS.
        step: The training step in its name, or None.
        marked: Whether its name says recommended, final or best.
    """

    file: str
    size_bytes: int | None
    sha256: str | None
    step: int | None
    marked: bool


@dataclass(frozen=True)
class RepoDetail:
    """What Discover shows about one repository, read from its model card and file list at one commit."""

    repo: str
    revision: str
    author: str
    downloads: int
    likes: int
    licence: str | None
    base_model: str | None
    instance_prompt: str | None
    previews: tuple[str, ...]
    preview_videos: tuple[str, ...]
    files: tuple[RepoFile, ...]


@dataclass(frozen=True)
class FamilyCheck:
    """Which model a LoRA file was made for, as its header says.

    Attributes:
        status: ``"match"`` (made for the asked model), ``"mismatch"`` (made for another), or ``"unknown"``.
        family: The family the header names, or None.
        size_variant: The size within the family, or None.
    """

    status: str
    family: str | None
    size_variant: str | None


def base_repos(family: str, size: str | None = None, version: str | None = None) -> tuple[str, ...]:
    """Return the official base repositories whose adapters belong to *family* (and *size* or *version* when the family has them).

    Raises:
        ValueError: If Hugging Face search is not offered for the family, size or version.
    """
    bases = BASE_MODELS.get(family)
    if bases is None:
        raise ValueError(f"Hugging Face search is not offered for the '{family}' family.")
    if None in bases:
        return bases[None]
    variant = size if size is not None else version
    if variant is None:
        return tuple(repo for repos in bases.values() for repo in repos)
    if variant not in bases:
        raise ValueError(f"Hugging Face search is not offered for the '{family}' family at '{variant}'.")
    return bases[variant]


def clean_repo_name(repo: str) -> str:
    """Return the repository's name as words: ``owner/Technically-Color_Z`` gives ``Technically Color Z``."""
    return " ".join(_NAME_BREAKS.sub(" ", repo.rsplit("/", 1)[-1]).split())


def preview_url(raw: object, repo: str, revision: str = "main") -> str | None:
    """Return a Hub resolve URL for a card's preview *raw*, or None when it is not an image or video on the Hub.

    A path inside the repository is resolved against *revision*; a Hub ``blob`` URL becomes a ``resolve`` URL; a
    URL on any other host, or any other kind of file, is dropped.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    parts = urlsplit(text)
    if parts.scheme or parts.netloc:
        if parts.scheme != "https" or (parts.hostname or "").lower() not in _PREVIEW_HOSTS or parts.username is not None:
            return None
        path = parts.path.replace("/blob/", "/resolve/", 1)
        url = f"https://{parts.hostname.lower()}{quote(path, safe='/%')}"
    else:
        relative = parts.path.lstrip("/")
        if not relative or ".." in relative.split("/") or "\\" in relative:
            return None
        url = f"{HUB_ROOT}/{validated_repo(repo)}/resolve/{revision}/{quote(relative, safe='/')}"
    suffix = "." + url.rsplit(".", 1)[-1].lower() if "." in url.rsplit("/", 1)[-1] else ""
    return url if suffix in _PREVIEW_IMAGES | _PREVIEW_VIDEOS else None


def _previews(card: Mapping[str, Any], repo: str, revision: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the card's widget preview ``(images, videos)``, each deduplicated, in card order and capped."""
    images: list[str] = []
    videos: list[str] = []
    for widget in card.get("widget") or ():
        output = widget.get("output") if isinstance(widget, Mapping) else None
        url = preview_url(output.get("url") if isinstance(output, Mapping) else None, repo, revision)
        if url is None:
            continue
        bucket = videos if "." + url.rsplit(".", 1)[-1].lower() in _PREVIEW_VIDEOS else images
        if url not in bucket:
            bucket.append(url)
    return tuple(images[:_MAX_PREVIEWS]), tuple(videos[:_MAX_PREVIEWS])


def _get_json(fetch: Fetch, url: str) -> Any:
    """Return the JSON the Hub answers *url* with.

    Raises:
        HubError: If the Hub answers with an error status, an oversized body or something that is not JSON.
    """
    response = fetch(url)
    if response.status == 404:
        raise HubError("That repository was not found on Hugging Face.", 404)
    if response.status != 200 or response.truncated:
        raise HubError(f"Hugging Face answered with status {response.status}.", response.status)
    try:
        return json.loads(response.body)
    except ValueError as exc:
        raise HubError("Hugging Face sent an answer that could not be read.", response.status) from exc


def _card(item: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a Hub model's card data, or an empty mapping when it has none."""
    card = item.get("cardData")
    return card if isinstance(card, Mapping) else {}


def _strings(value: object) -> list[str]:
    """Return *value* as a list of strings: a string alone, the strings of a list, or nothing."""
    if isinstance(value, str):
        return [value]
    return [entry for entry in value if isinstance(entry, str)] if isinstance(value, list) else []


def _names(item: Mapping[str, Any]) -> list[str]:
    """Return the names the content filter reads: the repository ID, tags, card tags, card trigger words and file names."""
    card = _card(item)
    files = [entry.get("rfilename") for entry in item.get("siblings") or () if isinstance(entry, Mapping)]
    return [*_strings(item.get("id")), *_strings(item.get("tags")), *_strings(card.get("tags")), *_strings(card.get("instance_prompt")), *_strings(files)]


def _is_listable(item: Mapping[str, Any], rules: ListingFilter) -> bool:
    """Return whether a repository may be listed: not gated, not private and not hidden by *rules*."""
    if item.get("gated") not in (False, None) or item.get("private"):
        return False
    if rules.hide_adult_tagged and (ADULT_TAG in _strings(item.get("tags")) or ADULT_TAG in _strings(_card(item).get("tags"))):
        return False
    return not is_filtered(_names(item), rules)


def _safetensors(siblings: Sequence[Any]) -> list[str]:
    """Return the ``.safetensors`` paths among a repository's file entries."""
    return [entry["rfilename"] for entry in siblings if isinstance(entry, Mapping) and isinstance(entry.get("rfilename"), str) and entry["rfilename"].endswith(".safetensors")]


def _licence(card: Mapping[str, Any]) -> str | None:
    """Return the card's licence id, or None when it names none."""
    value = card.get("license")
    return value if isinstance(value, str) and value.strip() else None


def _first_preview(card: Mapping[str, Any], repo: str) -> str | None:
    """Return the first image preview of a card, or None."""
    images, _ = _previews(card, repo, "main")
    return images[0] if images else None


def _row(item: Mapping[str, Any], rules: ListingFilter) -> dict[str, Any] | None:
    """Return the listing row for one Hub model, or None when it must not be listed."""
    repo = item.get("id")
    try:
        validated_repo(repo)
    except ValueError:
        return None
    files = _safetensors(item.get("siblings") or ())
    if not files or not _is_listable(item, rules):
        return None
    card = _card(item)
    return {
        "repo": repo,
        "name": clean_repo_name(repo),
        "author": repo.split("/", 1)[0],
        "downloads": int(item.get("downloads") or 0),
        "likes": int(item.get("likes") or 0),
        "files": len(files),
        "licence": _licence(card),
        "thumbnail": _first_preview(card, repo),
        "trending_score": float(item.get("trendingScore") or 0),
        "created_at": item.get("createdAt"),
        "updated_at": item.get("lastModified"),
    }


def _encode_cursor(pending: Mapping[str, str]) -> str | None:
    """Return the opaque cursor for the repositories that have a next page, or None when none has."""
    if not pending:
        return None
    return base64.urlsafe_b64encode(json.dumps(sorted(pending.items())).encode("ascii")).decode("ascii")


def _decode_cursor(cursor: str, allowed: Sequence[str]) -> dict[str, str]:
    """Return ``{base repo: Hub cursor}`` from a cursor made by :func:`_encode_cursor`.

    Raises:
        ValueError: If *cursor* is not one this module made for these base repositories.
    """
    try:
        pairs = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
        pending = {base: value for base, value in pairs}
    except (ValueError, TypeError) as exc:
        raise ValueError("cursor is not valid.") from exc
    if not all(base in allowed and isinstance(value, str) and _CURSOR_VALUE.fullmatch(value) for base, value in pending.items()):
        raise ValueError("cursor is not valid.")
    return pending


def _next_cursor(link_header: str | None) -> str | None:
    """Return the Hub's ``cursor`` of the ``rel="next"`` link, or None on the last page."""
    for part in (link_header or "").split(","):
        url, _, rel = part.partition(";")
        if 'rel="next"' in rel:
            values = parse_qs(urlsplit(url.strip().strip("<>")).query).get("cursor")
            return values[0] if values else None
    return None


def list_adapters(
    fetch: Fetch,
    *,
    family: str,
    size: str | None = None,
    version: str | None = None,
    query: str = "",
    sort: str = "downloads",
    cursor: str | None = None,
    rules: ListingFilter,
) -> dict[str, Any]:
    """Return one page of the LoRAs the Hub lists as adapters of the official base model(s) of *family*.

    The page merges one Hub page per base repository, ordered by *sort*, with a repository that is an adapter of several
    of them listed once; gated and private repositories, repositories *rules* hides, and repositories without a
    ``.safetensors`` file are left out, so a page can hold fewer than ``PAGE_SIZE`` rows. When every row of
    those Hub pages is left out, the next ones are read, up to ``_EXTRA_PAGES`` times, so a page is empty only when the
    Hub has no more or nothing in those pages could be listed.

    Args:
        fetch: Reads the Hub.
        family: The model family.
        size: The size within the family, for families that have sizes.
        version: The version within the family (LTX ``2.3`` or ``2.5``), for families that have versions.
        query: Words to search for; blank lists everything.
        sort: A key of ``SORTS``.
        cursor: The ``next_cursor`` of the previous page, or None for the first page.
        rules: The content filter at its current level.

    Returns:
        ``{"items": [...], "next_cursor": str | None, "base_models": [...]}``.

    Raises:
        ValueError: If the family, size, sort or cursor is not valid.
        HubError: If the Hub cannot answer.
    """
    bases = base_repos(family, size, version)
    if sort not in SORTS:
        raise ValueError(f"sort must be one of {', '.join(SORTS)}.")
    pending = dict.fromkeys(bases, "") if not cursor else _decode_cursor(cursor, bases)
    rows: list[dict[str, Any]] = []
    following: dict[str, str] = {}
    seen: set[str] = set()
    for _ in range(1 + _EXTRA_PAGES):
        following = {}
        for base, hub_cursor in pending.items():
            items, next_cursor = _adapter_page(fetch, base, sort=sort, query=query, hub_cursor=hub_cursor)
            for item in items:
                row = _row(item, rules) if isinstance(item, Mapping) else None
                if row is not None and row["repo"] not in seen:
                    seen.add(row["repo"])
                    rows.append(row)
            if next_cursor is not None:
                following[base] = next_cursor
        if rows or not following:
            break
        pending = following
    if len(bases) > 1:
        rows.sort(key=lambda row: _SORT_VALUES[sort](row), reverse=True)
    return {"items": rows, "next_cursor": _encode_cursor(following), "base_models": list(bases)}


def _adapter_page(fetch: Fetch, base: str, *, sort: str, query: str, hub_cursor: str) -> tuple[list[Any], str | None]:
    """Return one Hub page of the adapters of *base* and the Hub cursor of the next page, or None on the last page.

    Raises:
        HubError: If the Hub answers with an error status, an oversized body or something that is not a JSON list.
    """
    params = {"filter": f"base_model:adapter:{base}", "sort": SORTS[sort], "direction": -1, "limit": PAGE_SIZE, "search": query.strip()[:MAX_QUERY_CHARS] or None, "cursor": hub_cursor or None}
    response = fetch(api_url("models", params, repeated={"expand[]": _LIST_FIELDS}))
    if response.status != 200 or response.truncated:
        raise HubError(f"Hugging Face answered with status {response.status}.", response.status)
    try:
        items = json.loads(response.body)
    except ValueError as exc:
        raise HubError("Hugging Face sent an answer that could not be read.", response.status) from exc
    return (items if isinstance(items, list) else []), _next_cursor(response.headers.get("link"))


def choose_files(files: Sequence[RepoFile]) -> tuple[RepoFile | None, tuple[RepoFile, ...], tuple[RepoFile, ...]]:
    """Return ``(pre-curated file, choices, folded files)`` for a repository's ``.safetensors`` files.

    The pre-curated file is the first one marked recommended, final or best; else the first without a step number; else
    the one with the highest step. Choices are the pre-curated file and every other file that is marked or has no step
    number (real variants), in file order. The rest are training snapshots, folded.
    """
    if not files:
        return None, (), ()
    marked = [entry for entry in files if entry.marked]
    unstepped = [entry for entry in files if entry.step is None]
    chosen = (marked or unstepped or [max(files, key=lambda entry: entry.step or 0)])[0]
    choices = tuple(entry for entry in files if entry is chosen or entry.marked or entry.step is None)
    folded = tuple(entry for entry in files if entry not in choices)
    return chosen, choices, folded


def _step_of(file: str) -> int | None:
    """Return the training step in *file*'s name (``…_000002500…`` or ``…step_1200…``), or None when it has none."""
    stem = file.rsplit("/", 1)[-1].removesuffix(".safetensors")
    word = _STEP_WORD.search(stem)
    if word is not None:
        return int(word.group(1))
    digits = _STEP.findall(stem)
    return int(digits[-1]) if digits else None


def _repo_files(siblings: Sequence[Any]) -> tuple[RepoFile, ...]:
    """Return the ``.safetensors`` entries of a repository's file list with the size and hash the Hub gives them."""
    found = []
    for entry in siblings:
        name = entry.get("rfilename") if isinstance(entry, Mapping) else None
        if not isinstance(name, str) or not name.endswith(".safetensors"):
            continue
        try:
            validated_file(name)
        except ValueError:
            continue
        lfs = entry.get("lfs") if isinstance(entry.get("lfs"), Mapping) else {}
        size = entry.get("size")
        sha = lfs.get("sha256")
        found.append(
            RepoFile(name, size if isinstance(size, int) else None, sha if isinstance(sha, str) and len(sha) == 64 else None, _step_of(name), _MARKED.search(name.rsplit("/", 1)[-1]) is not None)
        )
    return tuple(found)


def read_repo(fetch: Fetch, repo: str, *, rules: ListingFilter) -> RepoDetail:
    """Return the detail of the repository *repo*.

    Args:
        fetch: Reads the Hub.
        repo: The repository ID.
        rules: The content filter at its current level.

    Raises:
        ValueError: If *repo* is not a repository ID, or the repository is gated, private or hidden by *rules*.
        HubError: If the Hub cannot answer, or has no such repository.
    """
    validated_repo(repo)
    data = _get_json(fetch, api_url(f"models/{repo}", {"blobs": "true"}))
    if not isinstance(data, Mapping):
        raise HubError("Hugging Face sent an answer that could not be read.")
    if not _is_listable(data, rules):
        raise ValueError("Discover does not offer that repository: it is gated, private or hidden by the content filter.")
    revision = data.get("sha")
    if not isinstance(revision, str) or len(revision) != 40:
        raise HubError("Hugging Face did not say which commit that repository is at.")
    card = _card(data)
    images, videos = _previews(card, repo, revision)
    base = card.get("base_model")
    if isinstance(base, list):
        base = next((entry for entry in base if isinstance(entry, str)), None)
    return RepoDetail(
        repo=repo,
        revision=revision,
        author=repo.split("/", 1)[0],
        downloads=int(data.get("downloads") or 0),
        likes=int(data.get("likes") or 0),
        licence=_licence(card),
        base_model=base if isinstance(base, str) else None,
        instance_prompt=card_triggers(card.get("instance_prompt")),
        previews=images,
        preview_videos=videos,
        files=_repo_files(data.get("siblings") or ()),
    )


def file_json(entry: RepoFile) -> dict[str, Any]:
    """Return a repository file as the API's JSON."""
    return {
        "file": entry.file,
        "size_bytes": entry.size_bytes,
        "size_label": None if entry.size_bytes is None else disk_size_label(entry.size_bytes),
        "sha256": entry.sha256,
        "step": entry.step,
        "marked": entry.marked,
    }


def check_family(fetch: Fetch, *, repo: str, revision: str, file: str, family: str, size: str | None = None) -> FamilyCheck:
    """Read the header of one repository file by HTTP range and say which model it was made for.

    Only the header is read: first the 8-byte length, then exactly that many bytes. A header that would take the read
    past ``HEADER_READ_BYTES`` is not read. The answer is ``match`` when the detected family is *family*
    (and *size*, when the family has sizes and the file names one), ``mismatch`` when it names another family or size,
    and ``unknown`` when the header cannot be read or names no family.

    Raises:
        ValueError: If *repo*, *revision* or *file* is not valid.
        HubError: If the Hub cannot answer.
    """
    url = resolve_url(repo, revision, file)
    prefix = _read_range(fetch, url, 0, _LENGTH_BYTES)
    if len(prefix) < _LENGTH_BYTES:
        return FamilyCheck("unknown", None, None)
    length = int.from_bytes(prefix, "little")
    if length == 0 or _LENGTH_BYTES + length > HEADER_READ_BYTES:
        return FamilyCheck("unknown", None, None)
    body = _read_range(fetch, url, _LENGTH_BYTES, length)
    try:
        header, metadata = header_and_metadata_from_bytes(prefix + body)
    except ValueError:
        return FamilyCheck("unknown", None, None)
    found = detect_lora_family(header, metadata)
    if found.family is None:
        return FamilyCheck("unknown", None, None)
    if found.family != family:
        return FamilyCheck("mismatch", found.family, found.size_variant)
    if size is not None and found.size_variant is not None and found.size_variant != size:
        return FamilyCheck("mismatch", found.family, found.size_variant)
    return FamilyCheck("match", found.family, found.size_variant)


def _read_range(fetch: Fetch, url: str, start: int, count: int) -> bytes:
    """Return the *count* bytes of *url* that begin at *start*, read by range; fewer when the file ends first.

    Raises:
        HubError: If the Hub does not answer with the bytes.
    """
    response = fetch(url, headers={"Range": f"bytes={start}-{start + count - 1}"}, max_bytes=count)
    if response.status in (401, 403, 404):
        raise HubError("That file was not found on Hugging Face.", response.status)
    if response.status not in (200, 206):
        raise HubError(f"Hugging Face answered with status {response.status}.", response.status)
    if response.status == 200 and start > 0:
        # The server ignored the range and sent the file from its first byte: the bytes asked for are not in this answer.
        return b""
    return response.body[:count]
