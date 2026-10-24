"""Find or download the files of one pinned Hugging Face revision, whichever platform asks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from zvisiongenerator.utils.h3_preset import HubFiles
from zvisiongenerator.utils.model_files import cached_snapshot_dirs

__all__ = ["snapshot_download", "snapshot_with"]


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
