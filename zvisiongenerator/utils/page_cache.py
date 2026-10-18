"""Drop model files from the operating system's page cache once they are written or read."""

from __future__ import annotations

import os
from pathlib import Path

from zvisiongenerator.utils.stored_quant import fsync_file

__all__ = ["drop_cached_file", "drop_cached_files"]


def drop_cached_files(directory: Path) -> None:
    """Write the files in *directory* through to disk and drop them from the page cache, where the platform can.

    The kernel keeps every model file read or written in memory until it needs the space. Model files can be
    larger than system memory, and reclaiming them under load is memory pressure that can get processes killed.
    """
    for path in directory.rglob("*"):
        if path.is_file():
            drop_cached_file(path)


def drop_cached_file(path: Path) -> None:
    """Write *path* through to disk and drop it from the page cache, where the platform can.

    Pages a process still maps stay cached, so call this once the file is closed.
    """
    if not hasattr(os, "posix_fadvise"):  # Linux only
        return
    fsync_file(path)  # dirty pages cannot be dropped
    try:
        with open(path, "rb") as handle:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass  # the cache is only a memory aid
