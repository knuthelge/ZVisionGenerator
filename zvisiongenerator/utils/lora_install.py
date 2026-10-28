"""Verify a downloaded LoRA file and move it into the LoRA folder."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

__all__ = ["install_staged", "sha256_file"]

_CHUNK_BYTES = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the lower-case hex SHA-256 of the file at *path*, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def install_staged(staged: Path, destination: Path, *, expected_sha256: str | None = None, expected_size: int | None = None) -> str:
    """Check the staged file against what the source promised, then rename it onto *destination* in one step.

    The rename is the only change to the LoRA folder, so a failed check leaves no file there. The staging folder
    must be on the destination's drive for the rename to be atomic.

    Args:
        staged: The downloaded file.
        destination: Where it is installed; it must not exist.
        expected_sha256: The SHA-256 the file must have, or None when the source states none.
        expected_size: The size in bytes the file must have, or None when the source states none.

    Returns:
        The file's SHA-256.

    Raises:
        FileNotFoundError: If *staged* is missing.
        FileExistsError: If *destination* exists.
        ValueError: If the size or the SHA-256 differs from what the source promised.
    """
    if not staged.is_file():
        raise FileNotFoundError(f"The downloaded file is missing: {staged.name}")
    size = staged.stat().st_size
    if expected_size is not None and size != expected_size:
        raise ValueError(f"The downloaded file is {size} bytes, not the {expected_size} bytes it should be.")
    digest = sha256_file(staged)
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError("The downloaded file does not match its published SHA-256, so it was discarded.")
    if destination.exists():
        raise FileExistsError(f"A LoRA named '{destination.stem}' already exists.")
    os.replace(staged, destination)
    return digest
