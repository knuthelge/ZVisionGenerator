"""Delete installed models, LoRAs, and HuggingFace model downloads from the Web UI.

Deletes act only on what the user picked. A converted model's folder is removed with its links, never the
HuggingFace files those links point to; a HuggingFace download is removed from the cache while its alias
stays in config, so the model can be downloaded again.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from zvisiongenerator.utils.model_files import find_local_model_dir, huggingface_cache_repo_dir
from zvisiongenerator.utils.paths import parse_huggingface_repo_reference
from zvisiongenerator.web.model_inventory import ImageInventoryEntry, VideoInventoryEntry


@dataclass(frozen=True)
class DeleteTarget:
    """Describe what deleting one inventory model removes from disk."""

    kind: str  # "installed" or "huggingface"
    path: Path
    repo_id: str | None = None


def model_delete_target(
    entry: ImageInventoryEntry | VideoInventoryEntry,
    models_dir: Path,
    *,
    find_local_dir: Callable[[str], Path | None] = find_local_model_dir,
) -> DeleteTarget | None:
    """Return what deleting *entry* would remove, or ``None`` when the model is not deletable.

    Installed models delete their folder in *models_dir*. Aliases delete their HuggingFace cache download once
    the repo's own weights are fully downloaded (an LTX MLX repo counts even while its separate text encoder is
    missing); aliases pointing at a local directory are not deletable from the UI.
    """
    if entry.source == "installed":
        path = models_dir / entry.name
        return DeleteTarget("installed", path) if _is_direct_child(path, models_dir) else None
    repo = parse_huggingface_repo_reference(entry.resolved_path)
    if repo is None or find_local_dir(entry.resolved_path) is None:
        return None
    return DeleteTarget("huggingface", huggingface_cache_repo_dir(repo.repo_id), repo_id=repo.repo_id)


def installed_models_linking_to(target: Path, models_dir: Path) -> tuple[str, ...]:
    """Return installed models with a link into *target*, which deleting *target* would break."""
    try:
        resolved_target = target.resolve()
        children = sorted(child for child in models_dir.iterdir() if child.is_dir() and not child.is_symlink())
    except OSError:
        return ()
    return tuple(child.name for child in children if _links_into(child, resolved_target))


def delete_model(target: DeleteTarget) -> None:
    """Remove *target* from disk; a symlinked folder is unlinked rather than emptied.

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
