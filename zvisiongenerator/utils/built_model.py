"""What the models that are built on first use share: the build record, the models-directory check and the partial-build runner.

A built model (Ideogram 4 Instant, MOSS-Image) is assembled in a hidden partial folder next to its final place; a build
record written last marks the folder finished, and promoting the partial replaces any older build.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any

from zvisiongenerator.utils.paths import get_ziv_data_dir
from zvisiongenerator.utils.stored_quant import discard_partial, flush_to_disk, partial_dir, promote_partial, sweep_stale_partials

__all__ = [
    "BUILDING_PHASE",
    "BUILD_RECORD_NAME",
    "build_in_partial",
    "is_foreign_folder",
    "is_models_dir_child",
    "refuse_foreign_folder",
    "read_build_record",
    "write_record",
]

# Written last into a finished build; a folder without it (or with an outdated one) is rebuilt.
BUILD_RECORD_NAME = "ziv-build.json"
# Reported while a model is built (downloaded, converted or linked), before it loads.
BUILDING_PHASE = "building"


def is_models_dir_child(path: str | Path, names: Collection[str]) -> bool:
    """Return whether *path* is a folder called one of *names* directly in the models directory."""
    candidate = Path(path).expanduser()
    return candidate.name in names and _same_path(candidate.parent, get_ziv_data_dir() / "models")


def _same_path(first: Path, second: Path) -> bool:
    """Return whether *first* and *second* are the same location."""
    try:
        return first.resolve() == second.resolve()
    except OSError:
        return False


def is_foreign_folder(path: str | Path) -> bool:
    """Return whether *path* holds something that is not a build of ours: it exists, is not empty and has no build record.

    A missing or empty folder is free to build in, and a folder with an outdated record is an earlier build that a
    new build replaces.
    """
    candidate = Path(path).expanduser()
    if not candidate.exists() and not candidate.is_symlink():
        return False
    if candidate.is_dir():
        try:
            if not any(candidate.iterdir()):
                return False
        except OSError:
            return True
    return read_build_record(candidate) is None


def refuse_foreign_folder(target: Path) -> None:
    """Raise unless *target* is free to build in, so a build never replaces a folder that is not one of ours.

    Raises:
        RuntimeError: If :func:`is_foreign_folder` says *target* holds another model or other files.
    """
    if is_foreign_folder(target):
        raise RuntimeError(f"Cannot build '{target.name}': the folder {target} already exists and was not built by ziv. Rename or move it, then run again.")


def read_build_record(model_dir: Path) -> dict[str, Any] | None:
    """Return *model_dir*'s build record, or ``None`` when it has none or it cannot be read."""
    try:
        record = json.loads((model_dir / BUILD_RECORD_NAME).read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    return record if isinstance(record, dict) else None


def write_record(model_dir: Path, record: dict[str, Any]) -> None:
    """Write *record* as *model_dir*'s build record."""
    (model_dir / BUILD_RECORD_NAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def build_in_partial(target: Path, assemble: Callable[[Path], bool], write_build_record: Callable[[Path], None]) -> bool:
    """Build a model folder in a partial folder next to *target* and promote it, replacing any older build.

    Args:
        target: The model folder to create.
        assemble: Downloads and fills the partial folder it is given, creating it first; returns False when it was
            stopped, which discards the partial.
        write_build_record: Marks the flushed partial folder as finished.

    Returns:
        Whether the folder is in place (False when *assemble* stopped).

    Raises:
        RuntimeError: If *target* exists and is not an earlier build (see :func:`refuse_foreign_folder`); nothing is downloaded.
    """
    refuse_foreign_folder(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    sweep_stale_partials(target)
    partial = partial_dir(target)
    try:
        if not assemble(partial):
            discard_partial(partial)
            return False
        flush_to_disk(partial)
        write_build_record(partial)
    except BaseException:
        discard_partial(partial)
        raise
    promote_partial(partial, target)
    return True
