"""Atomic file replacement shared by user-config and prompt-file writers."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import uuid


def write_text_atomic(path: Path, text: str) -> None:
    """Atomically replace *path* with *text*.

    Concurrent readers see either the old or the new file, never a truncated one. A symlinked *path* is
    written through to its target, and an existing file's permissions are kept: the temp file is created
    owner-only and given the target's mode before any content is written, so a 0600 file never leaks as 0644.
    """
    target = Path(path).resolve()
    temp_path = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        mode: int | None = stat.S_IMODE(target.stat().st_mode)
    except FileNotFoundError:
        mode = None
    try:
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if mode is not None else 0o666)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            if mode is not None:
                os.chmod(temp_path, mode)
            handle.write(text)
        temp_path.replace(target)
    finally:
        if temp_path.exists():
            temp_path.unlink()
