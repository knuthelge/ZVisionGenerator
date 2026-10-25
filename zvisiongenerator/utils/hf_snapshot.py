"""Find or download the files of one pinned Hugging Face revision, whichever platform asks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from zvisiongenerator.utils.h3_preset import HubFiles
from zvisiongenerator.utils.model_files import cached_snapshot_dirs
from zvisiongenerator.utils.paths import parse_huggingface_repo_reference

__all__ = ["pack_for_reference", "snapshot_download", "snapshot_with"]


def snapshot_download(**kwargs: Any) -> str:
    """Download the files of one Hub revision and return the snapshot folder."""
    from huggingface_hub import snapshot_download as hub_snapshot_download

    return hub_snapshot_download(**kwargs)


def snapshot_with(files: HubFiles, download: Callable[..., str]) -> Path:
    """Return the snapshot folder that holds all of *files*: a cached one, else a fresh download of just those files."""
    for snapshot in cached_snapshot_dirs(files.repo_id, files.revision):
        if all((snapshot / name).is_file() for name in files.files):
            return snapshot
    return Path(download(repo_id=files.repo_id, revision=files.revision, allow_patterns=list(files.files)))


def pack_for_reference(packs: Mapping[str, HubFiles], model_ref: str) -> HubFiles | None:
    """Return the pinned pack that the Hugging Face reference *model_ref* names, or ``None`` when it names none.

    A pack is named by its repo id, whatever the letter case, with no revision or exactly its pinned one. A reference
    pinned to another revision names no pack, and neither does a local path. When several packs share a repo, the
    first one that matches wins.

    Args:
        model_ref: A local path or a Hugging Face ``owner/repo[@revision]`` reference.
        packs: The pinned packs, each with the files to fetch.
    """
    reference = parse_huggingface_repo_reference(model_ref)
    if reference is None:
        return None
    repo_id = reference.repo_id.casefold()
    return next((pack for pack in packs.values() if pack.repo_id.casefold() == repo_id and reference.revision in (None, pack.revision)), None)
