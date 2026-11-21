"""Size LoRA files on disk."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

__all__ = ["lora_files_bytes"]


def lora_files_bytes(paths: Iterable[str | Path]) -> int:
    """Return the size in bytes of the LoRA files at *paths* (a folder counts its files); a path that is not there counts nothing."""
    total = 0
    for path in map(Path, paths):
        if path.is_file():
            total += path.stat().st_size
        elif path.is_dir():
            total += sum(child.stat().st_size for child in path.iterdir() if child.is_file())
    return total
