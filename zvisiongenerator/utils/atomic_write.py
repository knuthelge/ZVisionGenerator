"""Atomic file replacement shared by user-config and prompt-file writers."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import uuid

from zvisiongenerator.utils.filename import unique_output_path


def write_text_atomic(path: Path, text: str) -> None:
    """Atomically replace *path* with *text* as UTF-8, with the care of :func:`write_bytes_atomic`.

    Newlines are written as the platform's line separator, as a text-mode ``open`` does.
    """
    write_bytes_atomic(path, text.replace("\n", os.linesep).encode("utf-8"))


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Atomically replace *path* with *data*.

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
        # O_BINARY (Windows only) keeps the C runtime from translating newlines in the written bytes.
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        fd = os.open(temp_path, flags, 0o600 if mode is not None else 0o666)
        with os.fdopen(fd, "wb") as handle:
            if mode is not None:
                os.chmod(temp_path, mode)
            handle.write(data)
        temp_path.replace(target)
    finally:
        temp_path.unlink(missing_ok=True)


def write_new_file(path: Path, data: bytes, *, stem: str) -> Path:
    """Write *data* to a file that did not exist, moving to the next free counter name when *path* is taken.

    The file is created exclusively, so a name taken since it was chosen (another job, same second) is never
    overwritten: on a clash the path becomes ``unique_output_path(path.parent, stem, path.suffix)`` and the create
    is retried. A partial file is removed when the write fails, and the error is re-raised.

    Args:
        path: Preferred location; its directory must exist.
        data: Bytes to write.
        stem: File stem the counter names are built from.

    Returns:
        The path that was written.
    """
    while True:
        try:
            handle = open(path, "xb")
        except FileExistsError:
            path = unique_output_path(path.parent, stem, path.suffix)
            continue
        try:
            with handle:
                handle.write(data)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path
