"""Read the HuggingFace cache: the model repos it holds, what deleting each frees, and shared files nothing uses.

huggingface_hub keeps Xet downloads once in ``<cache>/blobs/`` and links each repo's ``blobs/`` entries to them,
so a repo's own folder can be small while its files live in that shared store.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from zvisiongenerator.utils.model_files import directory_bytes


@dataclass(frozen=True)
class CachedRepo:
    """Describe one model repo folder in the HuggingFace cache."""

    repo_id: str
    path: Path
    freed_bytes: int  # what deleting the folder frees: its own files, plus shared files nothing else references


def cached_model_repos(cache_dir: Path) -> tuple[CachedRepo, ...]:
    """Return the model repos in *cache_dir*, sorted by repo id; none when the cache folder is missing."""
    try:
        repo_dirs = sorted(child for child in cache_dir.iterdir() if child.name.startswith("models--") and child.is_dir() and not child.is_symlink())
    except OSError:
        return ()
    return tuple(CachedRepo(_repo_id(repo_dir), repo_dir, directory_bytes(repo_dir) + _freed_shared_bytes(repo_dir, cache_dir)) for repo_dir in repo_dirs)


def repo_disk_bytes(repo_dir: Path) -> int:
    """Return the disk use of a cache repo folder, including its files in the shared store (counted in full when shared)."""
    return directory_bytes(repo_dir) + sum(_file_bytes(target) for target in set(_shared_links(repo_dir).values()))


def stranded_shared_files(cache_dir: Path) -> dict[Path, int]:
    """Return the files in the shared store that no repo links to any more, with their sizes.

    They are left when a repo folder is deleted without huggingface_hub's own delete.
    """
    from huggingface_hub.utils import _shared_blobs

    return _shared_blobs.unreferenced_shared_blobs(cache_dir)


def free_stranded_shared_files(cache_dir: Path) -> int:
    """Delete the shared files no repo links to, and return the bytes freed.

    huggingface_hub re-checks each file's references under its lock before deleting it, so a file a download
    links in the meantime stays.
    """
    from huggingface_hub.utils import _shared_blobs

    return sum(_shared_blobs.sweep_shared_blob(path, cache_dir=cache_dir) for path in stranded_shared_files(cache_dir))


def _repo_id(repo_dir: Path) -> str:
    """Return the repo id a cache folder holds (``models--org--name`` is ``org/name``), as huggingface_hub reads it."""
    return "/".join(repo_dir.name.split("--")[1:])


def _shared_links(repo_dir: Path) -> dict[Path, Path]:
    """Return the repo's ``blobs/`` entries that link into the shared store, mapped to the files they link to."""
    try:
        entries = [entry for entry in (repo_dir / "blobs").iterdir() if entry.is_symlink()]
    except OSError:
        return {}
    return {Path(os.path.abspath(entry)): Path(os.path.realpath(entry)) for entry in entries}


def _freed_shared_bytes(repo_dir: Path, cache_dir: Path) -> int:
    """Return the shared-store bytes deleting *repo_dir* frees, by huggingface_hub's own rule for its sweep.

    A file counts only when its reference manifest is readable and lists nothing outside this repo, so files
    another repo (of any type) still uses, or with unreadable references, stay out of the total as they stay on disk.
    """
    from huggingface_hub.utils import _shared_blobs

    links = _shared_links(repo_dir)
    return sum(_shared_blobs.expected_shared_blob_freed_size(target, blob_paths_to_delete=set(links), cache_dir=cache_dir) for target in set(links.values()))


def _file_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0
