"""Delete installed models, LoRAs, and HuggingFace model downloads from the Web UI.

Deletes act only on what the user picked. A converted model's folder is removed with its links, never the
HuggingFace files those links point to; a HuggingFace download is removed from the cache while its alias
stays in config, so the model can be downloaded again. Either way the model's stored quants
(``<name>@q4`` / ``<name>@q8``) go too, since nothing else uses them. When aliases share one download (``ltx-4``
and ``ltx-8`` on Windows/Linux), deleting one removes only its own stored quants while another still has some.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zvisiongenerator.seedvr2_loader import seedvr2_copy_dir
from zvisiongenerator.utils.config import ltx_companion_repos
from zvisiongenerator.utils.model_files import find_local_model_dir, huggingface_cache_repo_dir
from zvisiongenerator.utils.paths import parse_huggingface_repo_reference
from zvisiongenerator.utils.stored_quant import stored_quant_dirs_for
from zvisiongenerator.web.model_inventory import ImageInventoryEntry, VideoInventoryEntry


@dataclass(frozen=True)
class DeleteTarget:
    """Describe what deleting one inventory model removes from disk."""

    kind: str  # "installed", "huggingface", or "copy" (only an alias's stored quants, its download is shared)
    path: Path
    repo_id: str | None = None
    stored_quants: tuple[Path, ...] = ()
    companions: tuple[Path, ...] = ()  # other HuggingFace downloads removed with this one
    shared_with: tuple[str, ...] = ()  # for "copy": the models still using the download


def model_delete_target(
    entry: ImageInventoryEntry | VideoInventoryEntry,
    models_dir: Path,
    *,
    find_local_dir: Callable[[str], Path | None] = find_local_model_dir,
    sharing: tuple[str, ...] = (),
    companion_repos: tuple[str, ...] = (),
) -> DeleteTarget | None:
    """Return what deleting *entry* would remove, or ``None`` when the model is not deletable.

    Installed models delete their folder in *models_dir*. Aliases delete their HuggingFace cache download once
    the repo's own weights are fully downloaded (an LTX MLX repo counts even while its separate text encoder is
    missing); aliases pointing at a local directory are not deletable from the UI.

    Args:
        entry: The model to delete.
        models_dir: The models folder.
        find_local_dir: Resolver for fully downloaded model directories.
        sharing: Other models that load from the same download (see :func:`models_sharing`). While one of them has
            stored quants the download stays, and only *entry*'s own stored quants are deleted.
        companion_repos: Other HuggingFace repos downloaded for this model, removed with its download.
    """
    stored_quants = stored_quant_dirs_for(models_dir, entry.name)
    if entry.source == "installed":
        path = models_dir / entry.name
        return DeleteTarget("installed", path, stored_quants=stored_quants) if _is_direct_child(path, models_dir) else None
    repo = parse_huggingface_repo_reference(entry.resolved_path)
    if repo is None or find_local_dir(entry.resolved_path) is None:
        return None
    users = tuple(name for name in sharing if stored_quant_dirs_for(models_dir, name))
    if users:
        return DeleteTarget("copy", stored_quants[0], stored_quants=stored_quants[1:], shared_with=users) if stored_quants else None
    companions = tuple(huggingface_cache_repo_dir(repo_id) for repo_id in companion_repos)
    return DeleteTarget("huggingface", huggingface_cache_repo_dir(repo.repo_id), repo_id=repo.repo_id, stored_quants=stored_quants, companions=companions)


def delete_scope(entry: ImageInventoryEntry | VideoInventoryEntry, inventory: Iterable[ImageInventoryEntry | VideoInventoryEntry], app_config: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Return the ``sharing`` and ``companion_repos`` arguments of :func:`model_delete_target` for *entry*."""
    return {"sharing": models_sharing(entry, inventory), "companion_repos": ltx_companion_repos(app_config, entry.resolved_path)}


def models_sharing(entry: ImageInventoryEntry | VideoInventoryEntry, inventory: Iterable[ImageInventoryEntry | VideoInventoryEntry]) -> tuple[str, ...]:
    """Return the names of the other aliases in *inventory* that resolve to *entry*'s HuggingFace download."""
    if entry.source == "installed":
        return ()
    return tuple(other.name for other in inventory if other.name != entry.name and other.source != "installed" and other.resolved_path == entry.resolved_path)


def seedvr2_delete_target(model: str, models_dir: Path) -> DeleteTarget | None:
    """Return what deleting the stored SeedVR2 copy of *model* removes, or ``None`` when no copy is on disk.

    Only the copy goes: the Hugging Face cache is never touched, since the copy replaced those downloads.
    """
    copy = seedvr2_copy_dir(model, models_dir)
    return DeleteTarget("installed", copy) if copy.is_dir() else None


def installed_models_linking_to(target: Path, models_dir: Path) -> tuple[str, ...]:
    """Return installed models with a link into *target*, which deleting *target* would break."""
    try:
        resolved_target = target.resolve()
        children = sorted(child for child in models_dir.iterdir() if child.is_dir() and not child.is_symlink())
    except OSError:
        return ()
    return tuple(child.name for child in children if _links_into(child, resolved_target))


def delete_model(target: DeleteTarget) -> None:
    """Remove *target* and its stored quants from disk; a symlinked folder is unlinked rather than emptied.

    Raises:
        FileNotFoundError: If nothing is on disk to delete.
    """
    path = target.path
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        # rmtree unlinks symlinks inside the folder without following them, so linked base files survive.
        shutil.rmtree(path)
    else:
        raise FileNotFoundError(f"Nothing to delete at {path}")
    for extra in (*target.stored_quants, *target.companions):
        shutil.rmtree(extra, ignore_errors=True)


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


def _is_direct_child(path: Path, parent: Path) -> bool:
    """Return whether *path* names an entry directly inside *parent* (no traversal or nested names)."""
    return path.parent == parent and path.name not in {"", ".", ".."}


def _links_into(model_dir: Path, resolved_target: Path) -> bool:
    for root, dirs, files in os.walk(model_dir, followlinks=False):
        for name in (*dirs, *files):
            entry = Path(root, name)
            if entry.is_symlink() and Path(os.path.realpath(entry)).is_relative_to(resolved_target):
                return True
    return False
