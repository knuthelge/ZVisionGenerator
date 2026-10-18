"""Place model files in a new folder by hard link, copying only where a link is impossible."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

__all__ = ["link_or_copy", "link_tree"]


def link_tree(source: Path, target: Path) -> None:
    """Hard-link *source* (a file or directory, following symlinks) to *target*, copying where linking fails."""
    if source.is_dir():
        shutil.copytree(source, target, copy_function=link_or_copy, dirs_exist_ok=True)
    else:
        link_or_copy(source, target)


def link_or_copy(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
    """Hard-link the file *source* resolves to as *target*, or copy it when a link is impossible."""
    try:
        os.link(os.path.realpath(source), target)
    except OSError:
        shutil.copy2(source, target)
