"""Read the curated list: the hand-checked Hugging Face LoRAs that ship with each release.

The list is the packaged ``curated_loras.json``. Every entry names one ``.safetensors`` file at a pinned commit of a
Hugging Face repository, with the facts needed to show it (name, description, previews, licence), to download and verify
it (repo, revision, file, size, SHA-256) and to use it (trigger, recommended weight, the family it was made for).

:func:`parse_curated_loras` turns the parsed JSON into frozen :class:`CuratedLora` values and does no I/O;
:func:`load_curated_loras` reads and caches the packaged file; the ``curated_*`` lookups take an optional *entries*
sequence, so they work on a fixture as well as on the packaged list.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
import json
from pathlib import Path
from typing import Any

__all__ = [
    "DATA_FILE",
    "LICENCES",
    "LTX_VERSIONS",
    "CURATED_FAMILIES",
    "PREVIEW_HOSTS",
    "SECTIONS",
    "SIZE_VARIANTS",
    "WORKFLOWS",
    "CuratedLora",
    "load_curated_loras",
    "parse_curated_loras",
    "curated_by_id",
    "curated_by_name",
    "curated_for_family",
]

DATA_FILE = Path(__file__).with_name("curated_loras.json")
_SCHEMA = 1

# The families the list covers, as ``utils/lora_family.py`` and the video presets name them.
CURATED_FAMILIES = ("zimage", "krea2", "flux2_klein", "ideogram4", "moss", "ltx", "h3")
# The size variants of a family that has them, as ``detect_lora_family`` reports them.
SIZE_VARIANTS: dict[str, tuple[str, ...]] = {"flux2_klein": ("4b", "9b")}
# The versions of a family that has them: ``utils/video_model_detect.py`` gives LTX-2.3 and LTX-2.5 one family.
LTX_VERSIONS: dict[str, tuple[str, ...]] = {"ltx": ("2.3", "2.5")}
SECTIONS = ("style", "tool")
# The workflow a tool needs, by the workspace's workflow ids; None when it works in any.
WORKFLOWS = ("txt2img", "img2img", "txt2vid", "img2vid", "extend")

# Every licence the list carries, with the label shown to people. A licence outside this table cannot be listed: read its
# terms, add it here.
LICENCES: dict[str, str] = {
    "apache-2.0": "Apache 2.0",
    "mit": "MIT",
    "cc-by-4.0": "CC BY 4.0",
    "creativeml-openrail-m": "CreativeML OpenRAIL-M",
    "openrail": "OpenRAIL",
    "wtfpl": "WTFPL",
    "cc-by-nc-4.0": "CC BY-NC 4.0",
    "flux-non-commercial": "FLUX non-commercial",
    "flux-2-research": "FLUX.2 research",
    "krea-2-community-license": "Krea 2 community licence",
    "minimax-h3-community-license": "MiniMax H3 community licence",
    "ltx-2-community-license": "LTX-2 community licence",
    "ideogram-community": "Ideogram community licence",
    "other": "Other (see the model card)",
}
# The hosts a preview may be on: the Hub, and its upload CDN for repositories whose card images live only there.
PREVIEW_HOSTS = frozenset({"huggingface.co", "cdn-uploads.huggingface.co"})


@dataclass(frozen=True)
class CuratedLora:
    """One curated LoRA.

    Attributes:
        id: Stable identifier, kept across releases.
        name: Local name: the file name stem the LoRA downloads as, and what presets, Reuse and ``--lora`` refer to.
        family: Family it was made for (see ``CURATED_FAMILIES``).
        size_variant: Size within the family (``4b``, ``9b``), or None when the family has no sizes.
        version: Version within the family (LTX ``2.3``, ``2.5``), or None when the family has no versions.
        display_name: Name shown to people.
        author: Hugging Face user or organisation that published it.
        description: One plain-English line about what it looks like or does.
        section: ``style`` (it looks like something) or ``tool`` (it does something).
        what_it_does: For a tool, one line on what it does; None for a style.
        workflow: For a tool, the workflow it needs (see ``WORKFLOWS``); None when any works or for a style.
        repo: Hugging Face repository ID.
        revision: Pinned 40-character commit of the repository.
        file: Path of the ``.safetensors`` file in the repository at that commit.
        size_bytes: Size of the file in bytes.
        sha256: SHA-256 of the file, as the Hub reports it.
        trigger: The trigger words to put in the prompt, comma-separated, or None when it needs none.
        weight: Recommended weight.
        licence: Licence id (a key of ``LICENCES``).
        non_commercial: Whether the licence forbids commercial use.
        downloads: The repository's Hugging Face download count when the list was last refreshed (``make check-curated``).
        platforms: Platforms it was tested on, as ``sys.platform`` values (``darwin``, ``win32``, ``linux``).
        note: A short remark about the file's format, or None.
        previews: Preview image URLs on the Hub.
        preview_videos: Preview video URLs, listed only when the repository has fewer than two preview images.
    """

    id: str
    name: str
    family: str
    size_variant: str | None
    version: str | None
    display_name: str
    author: str
    description: str
    section: str
    what_it_does: str | None
    workflow: str | None
    repo: str
    revision: str
    file: str
    size_bytes: int
    sha256: str
    trigger: str | None
    weight: float
    licence: str
    non_commercial: bool
    downloads: int
    platforms: tuple[str, ...]
    note: str | None
    previews: tuple[str, ...]
    preview_videos: tuple[str, ...]


def parse_curated_loras(data: Mapping[str, Any]) -> tuple[CuratedLora, ...]:
    """Return the entries of the parsed curated-list JSON *data*, in list order.

    Raises:
        ValueError: When the schema version is unknown or an entry lacks a field or has one of the wrong type.
    """
    if data.get("schema") != _SCHEMA:
        raise ValueError(f"Unsupported curated-list schema {data.get('schema')!r}; expected {_SCHEMA}.")
    return tuple(_parse_entry(raw) for raw in data.get("entries", ()))


def _parse_entry(raw: Mapping[str, Any]) -> CuratedLora:
    """Return the entry for one JSON object, naming the entry in the error when it is malformed."""
    label = raw.get("id", "<no id>")
    try:
        return CuratedLora(
            id=raw["id"],
            name=raw["name"],
            family=raw["family"],
            size_variant=raw["size_variant"],
            version=raw["version"],
            display_name=raw["display_name"],
            author=raw["author"],
            description=raw["description"],
            section=raw["section"],
            what_it_does=raw["what_it_does"],
            workflow=raw["workflow"],
            repo=raw["repo"],
            revision=raw["revision"],
            file=raw["file"],
            size_bytes=int(raw["size_bytes"]),
            sha256=raw["sha256"],
            trigger=raw["trigger"],
            weight=float(raw["weight"]),
            licence=raw["licence"],
            non_commercial=bool(raw["non_commercial"]),
            downloads=int(raw["downloads"]),
            platforms=tuple(raw["platforms"]),
            note=raw["note"],
            previews=tuple(raw["previews"]),
            preview_videos=tuple(raw["preview_videos"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Curated-list entry {label!r} is malformed: {error!r}.") from error


@cache
def load_curated_loras() -> tuple[CuratedLora, ...]:
    """Return the packaged curated list, read once."""
    return parse_curated_loras(json.loads(DATA_FILE.read_text(encoding="utf-8")))


def curated_for_family(
    family: str,
    *,
    size_variant: str | None = None,
    version: str | None = None,
    entries: Sequence[CuratedLora] | None = None,
) -> tuple[CuratedLora, ...]:
    """Return the entries made for *family*, in list order.

    Args:
        family: Family id, for example ``zimage`` or ``ltx``.
        size_variant: Keep only entries of this size (``4b``, ``9b``); None keeps every size.
        version: Keep only entries of this version (``2.3``, ``2.5``); None keeps every version.
        entries: The list to search; the packaged list when None.
    """
    pool = load_curated_loras() if entries is None else entries
    return tuple(entry for entry in pool if entry.family == family and (size_variant is None or entry.size_variant == size_variant) and (version is None or entry.version == version))


def curated_by_id(curated_id: str, *, entries: Sequence[CuratedLora] | None = None) -> CuratedLora | None:
    """Return the entry with id *curated_id*, or None."""
    pool = load_curated_loras() if entries is None else entries
    return next((entry for entry in pool if entry.id == curated_id), None)


def curated_by_name(name: str, *, entries: Sequence[CuratedLora] | None = None) -> CuratedLora | None:
    """Return the entry whose local name is *name*, or None."""
    pool = load_curated_loras() if entries is None else entries
    return next((entry for entry in pool if entry.name == name), None)
