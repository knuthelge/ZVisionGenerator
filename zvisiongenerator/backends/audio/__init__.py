"""Audio backends: one module per model family."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from zvisiongenerator.utils.h3_preset import HubFiles
from zvisiongenerator.utils.hf_snapshot import cached_snapshot

if TYPE_CHECKING:
    from zvisiongenerator.utils.audio_preset import AudioFiles

__all__ = ["snapshot_dir"]


def snapshot_dir(files: AudioFiles) -> Path:
    """Return the cached snapshot folder that holds every literal file of *files* (the download step fetched them).

    A folder pattern such as ``t5gemma-b-b-ul2/*`` is not checked file by file. Loading never downloads.

    Raises:
        FileNotFoundError: If no cached snapshot at the pinned revision holds the files.
    """
    literal = tuple(name for name in files.patterns if "*" not in name)
    folder = cached_snapshot(HubFiles(files.repo_id, files.revision, literal))
    if folder is None:
        raise FileNotFoundError(f"{files.repo_id} at {files.revision} is not downloaded.")
    return folder
