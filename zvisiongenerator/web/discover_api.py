"""Build the JSON of the Discover LoRAs routes from the curated list, the Hub readers and the installer.

The functions here take what they need and return plain data; the routes in ``server.py`` only parse the request,
call them and map their errors to statuses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from zvisiongenerator.utils.model_files import disk_size_label
from zvisiongenerator.utils.curated_loras import LICENCES, CuratedLora, load_curated_loras, curated_for_family
from zvisiongenerator.web.discover_hub import FamilyCheck, RepoDetail, RepoFile, choose_files, clean_repo_name, file_json
from zvisiongenerator.web.hub_client import HUB_ROOT
from zvisiongenerator.web.lora_download import KIND, HubFacts, LoraInstaller, LoraSource

__all__ = [
    "build_curated",
    "check_json",
    "hf_source",
    "installed_sizes",
    "list_json",
    "curated_json",
    "curated_repo_index",
    "curated_source",
    "repo_json",
]


def installed_sizes(loras_dir: Path) -> dict[str, int]:
    """Return ``{name: size in bytes}`` of the LoRA files in *loras_dir*."""
    sizes: dict[str, int] = {}
    try:
        for path in loras_dir.iterdir():
            if path.suffix == ".safetensors":
                try:
                    sizes[path.stem] = path.stat().st_size
                except OSError:
                    continue
    except OSError:
        return {}
    return sizes


def curated_repo_index(entries: Sequence[CuratedLora] | None = None) -> dict[tuple[str, str], str]:
    """Return ``{(repo, file): curated id}`` of the curated list."""
    return {(entry.repo, entry.file): entry.id for entry in (load_curated_loras() if entries is None else entries)}


def curated_source(entry: CuratedLora) -> LoraSource:
    """Return the installable source of a curated entry: its pinned commit, file, size and SHA-256, and its curated trigger."""
    return LoraSource(entry.repo, entry.revision, entry.file, entry.size_bytes, entry.sha256, curated_id=entry.id, trigger=entry.trigger, trigger_source="curated" if entry.trigger else None)


def hf_source(detail: RepoDetail, entry: RepoFile) -> LoraSource:
    """Return the installable source of *entry*, a file of the repository *detail*, with the model card's trigger and the facts to keep with the file."""
    facts = HubFacts(detail.author, detail.downloads, detail.licence, detail.previews[0] if detail.previews else None)
    return LoraSource(detail.repo, detail.revision, entry.file, entry.size_bytes, entry.sha256, trigger=detail.instance_prompt, trigger_source="card" if detail.instance_prompt else None, facts=facts)


def _licence_fields(licence: str | None) -> dict[str, Any]:
    """Return a licence id with the label shown to people (an id outside the curated table is its own label)."""
    return {"licence": licence, "licence_label": None if licence is None else LICENCES.get(licence, licence)}


def _curated_state(entry: CuratedLora, sizes: Mapping[str, int], active: Mapping[str, dict[str, Any]]) -> tuple[str, dict[str, Any] | None]:
    """Return ``(state, running lane item)`` of a curated entry.

    The state is ``downloading`` (its download is queued or running), ``installed`` (a file of its name and size is
    there), ``name_taken`` (its name is held by another file or download) or ``available``.
    """
    folded_active = {name.casefold(): item for name, item in active.items()}
    folded_sizes = {name.casefold(): size for name, size in sizes.items()}
    running = folded_active.get(entry.name.casefold())
    if running is not None:
        return ("downloading", running) if running.get("curated_id") == entry.id else ("name_taken", None)
    size = folded_sizes.get(entry.name.casefold())
    if size is None:
        return "available", None
    return ("installed", None) if size == entry.size_bytes else ("name_taken", None)


def curated_json(entry: CuratedLora, state: str, download: dict[str, Any] | None) -> dict[str, Any]:
    """Return a curated entry as the API's JSON, with its *state* on this machine."""
    return {
        "id": entry.id,
        "name": entry.name,
        "family": entry.family,
        "size_variant": entry.size_variant,
        "version": entry.version,
        "display_name": entry.display_name,
        "author": entry.author,
        "description": entry.description,
        "section": entry.section,
        "what_it_does": entry.what_it_does,
        "workflow": entry.workflow,
        "repo": entry.repo,
        "revision": entry.revision,
        "file": entry.file,
        "size_bytes": entry.size_bytes,
        "size_label": disk_size_label(entry.size_bytes),
        "trigger": entry.trigger,
        "weight": entry.weight,
        **_licence_fields(entry.licence),
        "non_commercial": entry.non_commercial,
        "downloads": entry.downloads,
        "platforms": list(entry.platforms),
        "note": entry.note,
        "previews": list(entry.previews),
        "preview_videos": list(entry.preview_videos),
        "model_card_url": f"{HUB_ROOT}/{entry.repo}",
        "state": state,
        "download": download,
    }


def build_curated(
    family: str,
    size: str | None,
    version: str | None,
    *,
    sizes: Mapping[str, int],
    active: Sequence[dict[str, Any]],
    entries: Sequence[CuratedLora] | None = None,
) -> list[dict[str, Any]]:
    """Return the curated entries made for the model as JSON, each with its state on this machine, in curated order."""
    chosen = curated_for_family(family, size_variant=size, version=version, entries=entries)
    held = {item["name"]: item for item in active if item.get("kind") == KIND}
    result = []
    for entry in chosen:
        state, download = _curated_state(entry, sizes, held)
        result.append(curated_json(entry, state, download))
    return result


def list_json(page: Mapping[str, Any], *, curated: Mapping[tuple[str, str], str]) -> dict[str, Any]:
    """Return a Hub listing page with the repositories that have a curated entry marked by ``curated_id``."""
    by_repo: dict[str, str] = {}
    for (repo, _file), curated_id in curated.items():
        by_repo.setdefault(repo, curated_id)
    items = [{**item, "curated_id": by_repo.get(item["repo"])} for item in page["items"]]
    return {"items": items, "next_cursor": page["next_cursor"], "base_models": page["base_models"]}


def repo_json(detail: RepoDetail, *, installer: LoraInstaller, curated: Mapping[tuple[str, str], str]) -> dict[str, Any]:
    """Return a repository's detail as JSON: its files (the pre-curated one, the real choices and the folded snapshots), card facts, and the name it would be saved under."""
    chosen, choices, folded = choose_files(detail.files)
    rows = installer.repo_rows(detail.repo)

    def describe(entry: RepoFile) -> dict[str, Any]:
        decision = installer.decide(hf_source(detail, entry), repo_rows=rows)
        return {
            **file_json(entry),
            "curated_id": curated.get((detail.repo, entry.file)),
            "save_as": decision.name,
            "renamed": decision.renamed,
            "state": {"free": "available", "running": "downloading", "installed": "installed"}[decision.status],
        }

    return {
        "repo": detail.repo,
        "name": clean_repo_name(detail.repo),
        "author": detail.author,
        "revision": detail.revision,
        "downloads": detail.downloads,
        "likes": detail.likes,
        **_licence_fields(detail.licence),
        "base_model": detail.base_model,
        "instance_prompt": detail.instance_prompt,
        "previews": list(detail.previews),
        "preview_videos": list(detail.preview_videos),
        "model_card_url": f"{HUB_ROOT}/{detail.repo}",
        "preselected": None if chosen is None else chosen.file,
        "files": [describe(entry) for entry in choices],
        "other_files": [describe(entry) for entry in folded],
    }


def check_json(result: FamilyCheck, *, family: str, size: str | None) -> dict[str, Any]:
    """Return a family check as JSON: ``status`` (``match``, ``mismatch`` or ``unknown``), what the file names, and what was asked."""
    return {"status": result.status, "family": result.family, "size_variant": result.size_variant, "expected": {"family": family, "size_variant": size}}
