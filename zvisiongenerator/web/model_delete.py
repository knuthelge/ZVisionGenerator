"""Delete installed models, LoRAs, prompt-enhancer models, and HuggingFace downloads from the Web UI.

Deletes act only on what the user picked. A converted model's folder is removed with its links, never the
HuggingFace files those links point to; a HuggingFace download is removed from the cache while its alias
stays in config, so the model can be downloaded again. A model built on first use (``moss``, ``moss-sft``, ``ideo``)
is made of hard links to its source repos in the cache, so deleting its row also deletes the source repos that no other
built model needs, and the hidden folder its build downloads into (Ideogram 4 Instant's transformer). Either way the model's stored quants
(``<name>@q4`` / ``<name>@q8``, and the text-encoder copy ``<name>@mlx-te4`` of H3 on macOS) go too, since nothing else uses them. When aliases share one download (``ltx-4``
and ``ltx-8`` on Windows/Linux), deleting one removes only its own stored quants while another still has some.
"""

from __future__ import annotations

import os
import shutil
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zvisiongenerator.downloads.built_sources import BUILT_MODEL_NAMES, built_model_downloads, built_model_source_repos, built_model_sources
from zvisiongenerator.seedvr2_loader import seedvr2_weights_dir
from zvisiongenerator.utils.config import enhancer_tier_models, video_companion_repos
from zvisiongenerator.utils.model_files import find_local_model_dir, huggingface_cache_repo_dir
from zvisiongenerator.utils.paths import parse_huggingface_repo_reference
from zvisiongenerator.utils.seedvr2 import SeedVR2Weights
from zvisiongenerator.utils.stored_quant import stored_quant_dirs_for
from zvisiongenerator.web.model_inventory import ImageInventoryEntry, VideoInventoryEntry


@dataclass(frozen=True)
class DeleteTarget:
    """Describe what deleting one inventory model removes from disk."""

    kind: str  # "installed", "huggingface", or "copy" (only an alias's stored quants, its download is shared)
    path: Path
    repo_id: str | None = None
    stored_quants: tuple[Path, ...] = ()
    companion_repos: tuple[str, ...] = ()  # other HuggingFace repos removed with this one: an LTX model's companions, or the source repos of a model built on first use
    shared_with: tuple[str, ...] = ()  # for "copy": the models still using the download
    build_downloads: tuple[Path, ...] = ()  # hidden folders next to a model built on first use that hold files it downloaded, removed with it

    @property
    def repo_ids(self) -> tuple[str, ...]:
        """The HuggingFace repos deleting this target removes."""
        return (*((self.repo_id,) if self.repo_id else ()), *self.companion_repos)

    @property
    def on_disk(self) -> bool:
        """Whether there is anything to remove: the target's own path, or a repo or download removed with it."""
        return self.path.exists() or self.path.is_symlink() or bool(self.companion_repos or self.build_downloads)


def model_delete_target(
    entry: ImageInventoryEntry | VideoInventoryEntry,
    models_dir: Path,
    *,
    find_local_dir: Callable[[str], Path | None] = find_local_model_dir,
    sharing: tuple[str, ...] = (),
    companion_repos: tuple[str, ...] = (),
    source_repos: tuple[str, ...] = (),
    build_downloads: tuple[Path, ...] = (),
) -> DeleteTarget | None:
    """Return what deleting *entry* would remove, or ``None`` when the model is not deletable.

    Installed models delete their folder in *models_dir*. Aliases delete their HuggingFace cache download, whether it
    is complete or an interrupted, partial one (an LTX MLX repo counts even while its separate text encoder is
    missing); aliases pointing at a local directory are not deletable from the UI.

    Args:
        entry: The model to delete.
        models_dir: The models folder.
        find_local_dir: Resolver for fully downloaded model directories; a repo with no such directory still counts while its cache folder exists.
        sharing: Other models that load from the same download (see :func:`models_sharing`). While one of them has
            stored quants the download stays, and only *entry*'s own stored quants are deleted.
        companion_repos: Other HuggingFace repos downloaded for this model, removed with its download.
        source_repos: For a model built on first use, the cache repos it is made from that nothing else needs (see
            :func:`built_sources_freed_with`). Those on disk are removed with its built folder; with no built folder
            they are the whole target, so a model that is downloaded but not built yet can be deleted too.
        build_downloads: For a model built on first use, the hidden folders of files its build downloaded outside the cache
            (see :func:`built_downloads_freed_with`). Those on disk are removed with it, and alone they make a not yet built
            model deletable.
    """
    stored_quants = stored_quant_dirs_for(models_dir, entry.name)
    sources = tuple(repo_id for repo_id in source_repos if huggingface_cache_repo_dir(repo_id).is_dir())
    downloads = tuple(path for path in build_downloads if path.is_dir())
    repo = parse_huggingface_repo_reference(entry.resolved_path)
    # A built-on-first-use row that is not built has no folder of its own: its target is the sources and downloads it left on disk.
    if entry.source == "installed" or (repo is None and (sources or downloads)):
        path = models_dir / entry.name
        if not _is_direct_child(path, models_dir):
            return None
        return DeleteTarget("installed", path, stored_quants=stored_quants, companion_repos=sources, build_downloads=downloads)
    # Whatever is in the cache counts, so an interrupted or partial download (H3 on macOS fetches only parts of its repository) can be cleared too.
    if repo is None or (find_local_dir(entry.resolved_path) is None and not huggingface_cache_repo_dir(repo.repo_id).is_dir()):
        return None
    users = tuple(name for name in sharing if stored_quant_dirs_for(models_dir, name))
    if users:
        return DeleteTarget("copy", stored_quants[0], stored_quants=stored_quants[1:], shared_with=users) if stored_quants else None
    return DeleteTarget("huggingface", huggingface_cache_repo_dir(repo.repo_id), repo_id=repo.repo_id, stored_quants=stored_quants, companion_repos=companion_repos)


def delete_scope(
    entry: ImageInventoryEntry | VideoInventoryEntry,
    inventory: Iterable[ImageInventoryEntry | VideoInventoryEntry],
    app_config: dict[str, Any],
    *,
    models_dir: Path,
    platform_key: str,
) -> dict[str, tuple[str, ...]]:
    """Return the ``sharing``, ``companion_repos``, ``source_repos`` and ``build_downloads`` arguments of :func:`model_delete_target` for *entry*."""
    return {
        "sharing": models_sharing(entry, inventory),
        "companion_repos": video_companion_repos(entry.family, app_config, entry.resolved_path),
        "source_repos": built_sources_freed_with(entry, tuple(inventory), app_config, models_dir=models_dir, platform_key=platform_key),
        "build_downloads": built_downloads_freed_with(entry, models_dir=models_dir),
    }


def built_sources_freed_with(
    entry: ImageInventoryEntry | VideoInventoryEntry,
    inventory: tuple[ImageInventoryEntry | VideoInventoryEntry, ...],
    app_config: dict[str, Any],
    *,
    models_dir: Path,
    platform_key: str,
) -> tuple[str, ...]:
    """Return the source repos that deleting the built-on-first-use model *entry* frees, or ``()`` for any other model.

    A repo is kept only while another such model with a built folder still needs it. A row that is not built does
    not pin anything, even when its sources happen to be in the cache. So deleting ``moss`` while ``moss-sft`` is
    built deletes only the Turbo LoRA repo. An invalid source block in *app_config* warns and frees nothing.
    """
    if not _is_built_model_row(entry, models_dir):
        return ()
    others = [other for other in inventory if other.name != entry.name and other.source == "installed" and _is_built_model_row(other, models_dir)]
    try:
        own = built_model_sources(entry.name, app_config, platform_key=platform_key)
        kept = {repo_id.casefold() for other in others for repo_id in built_model_sources(other.name, app_config, platform_key=platform_key)}
    except ValueError as exc:
        warnings.warn(f"Could not read the sources of a model built on first use: {exc}", stacklevel=2)
        return ()
    return tuple(repo_id for repo_id in own if repo_id.casefold() not in kept)


def built_downloads_freed_with(entry: ImageInventoryEntry | VideoInventoryEntry, *, models_dir: Path) -> tuple[Path, ...]:
    """Return the hidden download folders that deleting the built-on-first-use model *entry* removes, or ``()`` for any other model.

    They exist while a build downloads (and stay after an interrupted one), so an unbuilt row can own files on disk.
    """
    return built_model_downloads(entry.name, models_dir) if _is_built_model_row(entry, models_dir) else ()


def models_sharing(entry: ImageInventoryEntry | VideoInventoryEntry, inventory: Iterable[ImageInventoryEntry | VideoInventoryEntry]) -> tuple[str, ...]:
    """Return the names of the other aliases in *inventory* that resolve to *entry*'s HuggingFace download."""
    if entry.source == "installed":
        return ()
    return tuple(other.name for other in inventory if other.name != entry.name and other.source != "installed" and other.resolved_path == entry.resolved_path)


def seedvr2_delete_target(model: str, models_dir: Path, weights: SeedVR2Weights) -> DeleteTarget | None:
    """Return what deleting SeedVR2 *model* removes, its weights folder, or ``None`` when the folder is absent.

    Only the folder goes (the stored copy, or the downloaded files): the Hugging Face cache is never touched.
    """
    folder = seedvr2_weights_dir(model, models_dir, weights)
    return DeleteTarget("installed", folder) if folder.is_dir() else None


def enhancer_delete_target(repo: str) -> DeleteTarget | None:
    """Return what deleting the enhancer model *repo* removes: its whole cache folder, or ``None`` when absent or local.

    A partial download counts, so an interrupted first use can be cleared too.
    """
    reference = parse_huggingface_repo_reference(repo)
    if reference is None:
        return None
    path = huggingface_cache_repo_dir(reference.repo_id)
    return DeleteTarget("huggingface", path, repo_id=reference.repo_id) if path.is_dir() else None


def used_repo_ids(
    inventory: Iterable[ImageInventoryEntry | VideoInventoryEntry],
    app_config: dict[str, Any],
    *,
    platform_key: str,
    extra: Iterable[str] = (),
) -> frozenset[str]:
    """Return the HuggingFace repos the app loads, case-folded: aliases, their LTX companions, enhancer tiers, the sources of models built on first use and *extra*."""
    entries = [entry for entry in inventory if entry.source != "installed"]
    references = [entry.resolved_path for entry in entries]
    references += [model[0] for model in enhancer_tier_models(app_config, platform_key=platform_key).values()]
    ids = {repo.repo_id for reference in references if (repo := parse_huggingface_repo_reference(reference)) is not None}
    ids.update(repo_id for entry in entries for repo_id in video_companion_repos(entry.family, app_config, entry.resolved_path))
    ids.update(built_model_source_repos(app_config, platform_key=platform_key))
    ids.update(extra)
    return frozenset(repo_id.casefold() for repo_id in ids)


def cache_repos_linked_from(models_dir: Path, cache_dir: Path) -> frozenset[str]:
    """Return the cache folders (``models--org--name``) that installed models link into, which deleting would break."""
    try:
        caches = {Path(os.path.abspath(cache_dir)), cache_dir.resolve()}
        children = list(models_dir.iterdir())
    except OSError:
        return frozenset()
    links = [child for child in children if child.is_symlink()]
    for child in children:
        if child.is_dir() and not child.is_symlink():
            for root, dirs, files in os.walk(child, followlinks=False):
                links += [entry for name in (*dirs, *files) if (entry := Path(root, name)).is_symlink()]
    destinations = [destination for link in links for destination in _link_destinations(link)]
    return frozenset(destination.relative_to(cache).parts[0] for destination in destinations for cache in caches if destination.is_relative_to(cache) and destination != cache) - {"blobs"}


def installed_models_linking_to(target: Path, models_dir: Path) -> tuple[str, ...]:
    """Return installed models with a link into *target*, which deleting *target* would break."""
    try:
        targets = {Path(os.path.abspath(target)), target.resolve()}
        children = sorted(child for child in models_dir.iterdir() if child.is_dir() and not child.is_symlink())
    except OSError:
        return ()
    return tuple(child.name for child in children if _links_into(child, targets))


def delete_model(target: DeleteTarget) -> None:
    """Remove *target*, its stored quants, companion repos and build downloads from disk; a symlinked folder is unlinked rather than emptied.

    Raises:
        FileNotFoundError: If nothing is on disk to delete.
    """
    path = target.path
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        if target.kind == "huggingface":
            _delete_hub_repo(path)
        else:
            # rmtree unlinks symlinks inside the folder without following them, so linked base files survive.
            shutil.rmtree(path)
    elif not (target.companion_repos or target.build_downloads):
        raise FileNotFoundError(f"Nothing to delete at {path}")
    for extra in target.stored_quants:
        shutil.rmtree(extra, ignore_errors=True)
    for repo_id in target.companion_repos:
        try:
            _delete_hub_repo(huggingface_cache_repo_dir(repo_id))
        except OSError:
            pass
    for download in target.build_downloads:
        shutil.rmtree(download, ignore_errors=True)


def delete_lora(loras_dir: Path, name: str) -> Path:
    """Delete the LoRA file *name* from *loras_dir*; a symlinked file is unlinked, not its target.

    Raises:
        FileNotFoundError: If no LoRA with that name exists.
    """
    path = loras_dir / f"{name}.safetensors"
    if not _is_direct_child(path, loras_dir) or not (path.is_file() or path.is_symlink()):
        raise FileNotFoundError(f"LoRA not found: {name}")
    path.unlink()
    return path


def _is_built_model_row(entry: ImageInventoryEntry | VideoInventoryEntry, models_dir: Path) -> bool:
    """Return whether *entry* is a model built on first use: its name and its folder in *models_dir* say so."""
    return entry.name in BUILT_MODEL_NAMES and Path(entry.resolved_path).resolve() == (models_dir / entry.name).resolve()


def _delete_hub_repo(repo_dir: Path) -> None:
    """Delete a HuggingFace cache repo folder, with the files it holds in the cache-wide shared blob store.

    huggingface_hub keeps Xet downloads once in ``<cache>/blobs/`` and links each repo to them, so removing only
    the repo folder frees nothing. Its delete strategy, scoped to this one folder, removes it and collects the
    shared files no other repo links to. (``HFCacheInfo.delete_revisions`` is not used: it matches commit hashes
    across the whole cache, which would also delete a mirror repo with the same history.)
    """
    from huggingface_hub import DeleteCacheStrategy

    DeleteCacheStrategy(expected_freed_size=0, blobs=frozenset(), refs=frozenset(), repos=frozenset({repo_dir}), snapshots=frozenset(), cache_dir=repo_dir.parent).execute()
    if repo_dir.is_dir():
        shutil.rmtree(repo_dir)


def _is_direct_child(path: Path, parent: Path) -> bool:
    """Return whether *path* names an entry directly inside *parent* (no traversal or nested names)."""
    return path.parent == parent and path.name not in {"", ".", ".."}


def _links_into(model_dir: Path, targets: set[Path]) -> bool:
    for root, dirs, files in os.walk(model_dir, followlinks=False):
        for name in (*dirs, *files):
            entry = Path(root, name)
            if entry.is_symlink() and any(destination.is_relative_to(target) for destination in _link_destinations(entry) for target in targets):
                return True
    return False


def _link_destinations(link: Path) -> tuple[Path, ...]:
    """Return where *link* points: its own target and the fully resolved path.

    Both count because a link into a repo's Xet file resolves on into the cache-wide shared store (``<cache>/blobs/``),
    which no longer names the repo.
    """
    try:
        first_hop = Path(os.path.normpath(link.parent / os.readlink(link)))
    except OSError:
        return (Path(os.path.realpath(link)),)
    return (first_hop, Path(os.path.realpath(link)))
