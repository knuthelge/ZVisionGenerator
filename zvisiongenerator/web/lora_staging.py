"""Hold and sweep the staging folders of LoRA downloads (``.download-<name>``, beside the LoRA files).

A download stages its file in a folder of its own. While the download is in use, :class:`StagingClaim` keeps a lock on a
lock file inside that folder: ``flock`` on macOS and Linux (the way the temporary LoRA copies of ``utils/lora_files.py``
are locked), ``msvcrt.locking`` on Windows. The operating system drops the lock when the process dies. :func:`sweep_abandoned`
removes only the folders it can lock itself, so a folder a killed server left behind goes while the folder of a download
another UI sharing the data folder is running stays.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import IO

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None
try:
    import msvcrt
except ImportError:  # macOS and Linux
    msvcrt = None

__all__ = ["STAGING_PREFIX", "StagingClaim", "sweep_abandoned"]

STAGING_PREFIX = ".download-"
_LOCK_NAME = ".in-use"
# A sweep may remove the folder between a claim creating it and locking its file; the claim tries again.
_CLAIM_ATTEMPTS = 5


def _try_lock(handle: IO[bytes]) -> bool:
    """Lock *handle*'s file exclusively without waiting; False when another open file holds it."""
    try:
        if fcntl is not None:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        return False
    return True


def _remove_holding(folder: Path, handle: IO[bytes]) -> None:
    """Remove *folder* with everything in it, then close *handle* (which drops its lock).

    ``flock`` is held until the folder is gone, so no other claim can slip in. Windows cannot delete a file that is
    open, so there the lock file is closed first.
    """
    if fcntl is None:
        handle.close()
    shutil.rmtree(folder, ignore_errors=True)
    handle.close()


class StagingClaim:
    """The right to use one staging folder: the folder is emptied, then locked until :meth:`release`.

    Args:
        folder: The staging folder of a download.
    """

    def __init__(self, folder: Path) -> None:
        self._folder = folder
        self._handle: IO[bytes] | None = None
        self._acquired = False

    @property
    def folder(self) -> Path:
        """The folder this claim stages into: the one it was made for, or the fresh one :meth:`acquire` used instead."""
        return self._folder

    def acquire(self) -> Path:
        """Create the folder empty, lock it and return it.

        An old folder that cannot be emptied (a file held open by a virus scanner, say) does not stop the download: it
        stages into a fresh folder of its own beside it, and that is the folder returned.

        Raises:
            RuntimeError: If another download (of another UI sharing the data folder) holds the folder.
            OSError: If the folder cannot be made.
        """
        folder = self._folder
        for _ in range(_CLAIM_ATTEMPTS):
            folder.mkdir(parents=True, exist_ok=True)
            lock = folder / _LOCK_NAME
            try:
                handle = open(lock, "ab")  # noqa: SIM115 - held open for the life of the claim
            except FileNotFoundError:
                continue
            if not _try_lock(handle):
                handle.close()
                raise RuntimeError(f"Another download is already using {folder.name}.")
            if not _same_file(handle, lock):
                handle.close()  # a sweep removed the folder after it was opened: start again
                continue
            try:
                _empty(folder)
            except OSError:
                handle.close()
                folder = Path(tempfile.mkdtemp(prefix=f"{self._folder.name}.", dir=self._folder.parent))
                continue
            self._folder = folder
            self._handle = handle
            self._acquired = True
            return folder
        raise RuntimeError(f"Could not set up {self._folder.name} to download into.")

    def release(self) -> None:
        """Remove the folder and everything staged in it, and let go of the lock; does nothing when the claim was never acquired."""
        if not self._acquired:
            return  # the folder, if there is one, belongs to someone else
        handle, self._handle = self._handle, None
        self._acquired = False
        if handle is None:
            shutil.rmtree(self._folder, ignore_errors=True)
        else:
            _remove_holding(self._folder, handle)


def _same_file(handle: IO[bytes], path: Path) -> bool:
    """Return whether *path* is still the file *handle* has open."""
    try:
        return os.fstat(handle.fileno()).st_ino == path.stat().st_ino
    except OSError:
        return False


def _empty(folder: Path) -> None:
    """Remove everything in *folder* except its lock file.

    Raises:
        OSError: If something cannot be removed.
    """
    for path in folder.iterdir():
        if path.name == _LOCK_NAME:
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    if any(path.name != _LOCK_NAME for path in folder.iterdir()):
        raise OSError(f"Could not empty {folder.name}.")


def sweep_abandoned(loras_dir: Path) -> None:
    """Remove the staging folders under *loras_dir* that no download holds: what a crashed or killed server left behind.

    A folder is removed only when its lock file can be locked here, so one in use by another process is left alone,
    and an empty folder with no lock file (a claim that never got that far) is removed too.
    """
    try:
        folders = [path for path in loras_dir.iterdir() if path.name.startswith(STAGING_PREFIX) and path.is_dir() and not path.is_symlink()]
    except OSError:
        return
    for folder in folders:
        try:
            handle = open(folder / _LOCK_NAME, "rb")  # noqa: SIM115 - closed by _remove_holding or below
        except FileNotFoundError:
            with contextlib.suppress(OSError):
                folder.rmdir()
            continue
        except OSError:
            continue
        if _try_lock(handle):
            _remove_holding(folder, handle)
        else:
            handle.close()
