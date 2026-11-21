"""Tell whether a case fetched or built a model while it ran: the model folders grew."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

DOWNLOAD_THRESHOLD_BYTES = 16 * 1024**2
"""How much the model folders must grow during a case to count as a download or a build; a log line or a lock file is far below it."""


def storage_bytes(roots: Iterable[Path]) -> int:
    """Return the bytes of the files under *roots*, following symlinks to folders and counting each file once (a Hugging Face snapshot links to its blobs).

    A root or entry that cannot be read counts as empty.
    """
    seen: set[tuple[int, int]] = set()
    total = 0
    pending = [str(root) for root in roots]
    while pending:
        try:
            with os.scandir(pending.pop()) as entries:
                listing = list(entries)
        except OSError:
            continue
        for entry in listing:
            try:
                info = entry.stat()
                if info.st_ino == 0:  # Windows leaves the file id out of a directory listing; the path's own stat has it
                    info = os.stat(entry.path)
                is_folder = entry.is_dir()
            except OSError:
                continue
            key = (info.st_dev, info.st_ino)
            if key in seen:
                continue
            seen.add(key)
            if is_folder:
                pending.append(entry.path)
            else:
                total += info.st_size
    return total


def model_storage_roots() -> list[Path]:
    """Return the folders a download or a build writes to: the Hugging Face hub cache and the app's ``models`` folder."""
    from huggingface_hub.constants import HF_HUB_CACHE

    from zvisiongenerator.utils.paths import get_ziv_data_dir

    return [Path(HF_HUB_CACHE), get_ziv_data_dir() / "models"]


def model_storage_bytes() -> int:
    """Return the bytes the model folders hold now."""
    return storage_bytes(model_storage_roots())
