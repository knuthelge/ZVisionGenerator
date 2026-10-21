"""Free-space estimate for the main download drive."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from zvisiongenerator.downloads.target import DownloadTarget, TargetCheck


@dataclass(frozen=True)
class SpaceNeed:
    """Bytes a download plan needs against the free bytes on the drive holding ``path``."""

    path: Path
    needed_bytes: int
    free_bytes: int

    @property
    def enough(self) -> bool:
        """Whether the drive has room for the plan."""
        return self.free_bytes >= self.needed_bytes


def space_need(
    items: Sequence[tuple[DownloadTarget, TargetCheck]],
    *,
    cache_dir: Path,
    disk_usage: Callable[[Path], object] = shutil.disk_usage,
) -> SpaceNeed | None:
    """Estimate the space the missing files need on the main download drive.

    The main drive is the Hub cache's when any remaining target uses the cache, else the first remaining local
    folder's. Every remaining target's missing bytes are summed (unknown sizes count as 0), whichever drive it writes to.

    Args:
        items: Each target with its dry-run check.
        cache_dir: The Hub cache folder.
        disk_usage: Returns an object with ``free`` bytes for a path.

    Returns:
        The estimate, or ``None`` when nothing is missing.
    """
    remaining = [(target, check) for target, check in items if check.missing_files > 0]
    if not remaining:
        return None
    if any(target.local_dir is None for target, _ in remaining):
        destination = cache_dir
    else:
        destination = Path(remaining[0][0].local_dir or "")
    path = _existing_ancestor(destination)
    needed = sum(check.missing_bytes or 0 for _, check in remaining)
    return SpaceNeed(path=path, needed_bytes=needed, free_bytes=disk_usage(path).free)


def _existing_ancestor(path: Path) -> Path:
    """Return *path* or its nearest ancestor that exists."""
    while not path.exists() and path != path.parent:
        path = path.parent
    return path
