"""Describe installed LoRA files: where each came from and which trigger text goes with it.

The trigger comes from the first of these that names one, in this order of trust:

1. a trigger the user typed;
2. the curated entry's trigger (a file is a curated entry when its name and size match);
3. the model card's ``instance_prompt``, saved when the file was downloaded;
4. a header ``ss_tag_frequency`` that holds a single caption.

The file name and ``ss_output_name`` are never a trigger.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote
import warnings

from zvisiongenerator.utils.curated_loras import LICENCES, CuratedLora, curated_by_id, curated_by_name
from zvisiongenerator.utils.model_files import disk_size_label
from zvisiongenerator.web.discover_hub import clean_repo_name
from zvisiongenerator.web.hub_client import HUB_ROOT
from zvisiongenerator.web.lora_index import lookup as lookup_lora_headers
from zvisiongenerator.web.lora_sources import SOURCES, LoraSourceRow, LoraSources, LoraSourcesError

__all__ = ["INSTALLED_THUMBNAIL_ROUTE", "describe_installed", "installed_details", "resolve_trigger"]

INSTALLED_THUMBNAIL_ROUTE = "/api/discover/installed-thumbnails"


def resolve_trigger(*, curated: CuratedLora | None, row: LoraSourceRow | None, header: str | None) -> tuple[str | None, str | None]:
    """Return ``(trigger, where it came from)`` by the order of trust, or ``(None, None)``.

    Args:
        curated: The curated entry the file is, if any.
        row: The file's stored source row, if any and still valid.
        header: The single caption in the file's header, if any.

    Returns:
        The trigger and its origin: ``"user"``, ``"curated"``, ``"card"`` or ``"header"``.
    """
    if row is not None and row.user_trigger:
        return row.user_trigger, "user"
    if curated is not None and curated.trigger:
        return curated.trigger, "curated"
    if row is not None and row.auto_trigger and row.trigger_source == "card":
        return row.auto_trigger, "card"
    if row is not None and row.auto_trigger and row.trigger_source == "curated" and curated is None:
        return row.auto_trigger, "curated"
    if header:
        return header, "header"
    return None, None


def _hub_facts(name: str, row: LoraSourceRow) -> dict[str, Any]:
    """Return what the Installed tab shows of a Hugging Face download without a connection: the repository's name, author, download count, licence, link and preview picture."""
    return {
        "name": clean_repo_name(row.repo or name),
        "author": row.author,
        "downloads": row.downloads,
        "licence": row.licence,
        "licence_label": None if row.licence is None else LICENCES.get(row.licence, row.licence),
        "model_card_url": f"{HUB_ROOT}/{row.repo}",
        "thumbnail_url": f"{INSTALLED_THUMBNAIL_ROUTE}/{quote(name, safe='')}?v={int(row.downloaded_at)}" if row.has_image else None,
    }


def _names_trigger(curated: CuratedLora | None, row: LoraSourceRow | None) -> bool:
    """Return whether the curated entry or the stored row names a trigger, so the header's caption is not needed."""
    return (curated is not None and bool(curated.trigger)) or (row is not None and bool(row.auto_trigger or row.user_trigger))


def describe_installed(loras_dir: Path, names: Sequence[str], rows: Mapping[str, LoraSourceRow]) -> dict[str, dict[str, Any]]:
    """Return, for each LoRA in *names*, ``{"curated_id", "title", "trigger", "trigger_source", "source", "hub", "size_bytes", "size_label"}``.

    ``title`` is the name to show: the curated entry's, or the repository's name for a Hugging Face download, else None.

    ``source`` is None for a file the app did not download; otherwise ``{"repo", "revision", "file", "sha256",
    "curated_id"}`` as recorded when it was downloaded. ``hub`` holds the facts kept from a Hugging Face download that is
    not a curated entry (see :func:`_hub_facts`), else None. A stored row only counts while the file still has the size it
    was stored with.
    """
    known: dict[str, tuple[int | None, LoraSourceRow | None, CuratedLora | None]] = {}
    for name in names:
        try:
            size = (loras_dir / f"{name}.safetensors").stat().st_size
        except OSError:
            size = None
        row = rows.get(name)
        if row is not None and row.size_bytes != size:
            row = None
        candidate = curated_by_name(name)
        curated = candidate if candidate is not None and size is not None and candidate.size_bytes == size else None
        if curated is None and row is not None and row.curated_id is not None:
            curated = curated_by_id(row.curated_id)
        known[name] = (size, row, curated)
    # A header caption is the last resort, so only a file nothing else names a trigger for is looked up (in the index, which reads a header once per change).
    unsettled = [name for name, (_, row, curated) in known.items() if not _names_trigger(curated, row)]
    captions = lookup_lora_headers(loras_dir, unsettled, prune=False)
    described: dict[str, dict[str, Any]] = {}
    for name, (size, row, curated) in known.items():
        trigger, origin = resolve_trigger(curated=curated, row=row, header=captions[name].caption if name in captions else None)
        hub = _hub_facts(name, row) if row is not None and row.repo is not None and curated is None else None
        described[name] = {
            "curated_id": curated.id if curated is not None else None,
            "title": curated.display_name if curated is not None else None if hub is None else hub["name"],
            "trigger": trigger,
            "trigger_source": origin,
            "source": None if row is None or row.repo is None else {"repo": row.repo, "revision": row.revision, "file": row.file, "sha256": row.sha256, "curated_id": row.curated_id},
            "hub": hub,
            "size_bytes": size,
            "size_label": None if size is None else disk_size_label(size),
        }
    return described


def installed_details(loras_dir: Path, names: Sequence[str], *, sources: LoraSources = SOURCES) -> dict[str, dict[str, Any]]:
    """Return :func:`describe_installed` for *names*, reading their stored rows from *sources*.

    The stored rows only add to what a listing shows: when they cannot be read, every LoRA is described without them.
    """
    try:
        rows = sources.rows(names)
    except LoraSourcesError as exc:
        warnings.warn(f"The LoRA sources are unavailable ({exc}); LoRAs are listed without their sources.", stacklevel=2)
        rows = {}
    return describe_installed(loras_dir, names, rows)
