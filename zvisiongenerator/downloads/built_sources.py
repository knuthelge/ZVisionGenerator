"""Name the hidden folders that models built on first use download into.

Ideogram 4 Instant downloads its single-file transformer into a hidden folder beside its built model. The cache repos the
models are made from are declared in the model metadata (see ``downloads.declaration``).
"""

from __future__ import annotations

from pathlib import Path

from zvisiongenerator.utils.ideogram4_instant import INSTANT_DIR_NAME, instant_downloads_dir

__all__ = ["built_model_downloads"]


def built_model_downloads(name: str, models_dir: Path) -> tuple[Path, ...]:
    """Return the hidden folders in *models_dir* that hold files the build of the model called *name* downloaded outside the cache.

    Ideogram 4 Instant downloads its single-file transformer next to the model, where an interrupted download resumes
    and the finished build removes it. MOSS-Image downloads only into the cache. Any other name has none. Only the
    folders that exist are returned.
    """
    folders = (instant_downloads_dir(models_dir / name),) if name == INSTANT_DIR_NAME else ()
    return tuple(folder for folder in folders if folder.is_dir())
