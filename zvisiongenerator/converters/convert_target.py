"""Where a video checkpoint conversion writes, and how it reports progress.

A converted model is written into a hidden partial folder and moved into place once complete, so an interrupted
conversion never leaves a model that is listed and fails to load. Progress goes to standard output as
``Converting: <stage>`` lines, with ``(done/total)`` where the stage counts tensors; the Web UI reads them.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from zvisiongenerator.utils.stored_quant import discard_partial, parse_stored_quant_name, partial_dir, promote_partial, sweep_stale_partials

__all__ = ["PROGRESS_PREFIX", "Progress", "checked_target", "progress_line", "write_into"]

PROGRESS_PREFIX = "Converting: "
_STAGES = {
    "download": "Getting the base model files",
    "connectors": "Checking the text connectors",
    "transformer": "Writing the transformer",
    "text_encoder": "Writing the text encoder",
    "components": "Linking the other components",
}
# Print a counted stage at most this many times.
_PROGRESS_STEPS = 50


def checked_target(models_dir: Path, name: str, *, force: bool, config: dict[str, Any]) -> Path:
    """Return the folder ``<models_dir>/<name>`` a conversion may write, after checking the name.

    Raises:
        ValueError: If *name* is a configured alias or a stored-copy name (``<name>@q4`` and the like).
        FileExistsError: If the folder exists and *force* is not set.
    """
    if name in (config.get("model_aliases") or {}):
        raise ValueError(f"'{name}' is a model alias; choose another name for the converted model.")
    if parse_stored_quant_name(name) is not None:
        raise ValueError(f"'{name}' is reserved for stored copies (<name>@q4 / <name>@q8 / <name>@mlx-te4); choose another name.")
    target = models_dir / name
    if target.exists() and not force:
        raise FileExistsError(f"A model named '{name}' already exists. Choose another name, or pass --force to replace it.")
    return target


def write_into(target: Path, write: Callable[[Path], None]) -> Path:
    """Run *write* on a fresh partial folder of *target* and move it into place once it returns; return *target*.

    An existing *target* is replaced only then. A failure (or interruption) removes the partial folder.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    sweep_stale_partials(target)
    partial = partial_dir(target)
    try:
        partial.mkdir()
        write(partial)
        promote_partial(partial, target)
    except BaseException:
        discard_partial(partial)
        raise
    return target


def progress_line(stage: str, done: int, total: int) -> str:
    """Return the progress line of *stage* (a key of the stages), with ``(done/total)`` when the stage counts tensors."""
    label = _STAGES.get(stage, stage)
    return f"{PROGRESS_PREFIX}{label} ({done}/{total})" if total > 1 else f"{PROGRESS_PREFIX}{label}"


class Progress:
    """Print a conversion's progress lines: each stage once, and a counted stage about every 2%."""

    def __init__(self) -> None:
        self._stage: str | None = None
        self._printed = 0

    def __call__(self, stage: str, done: int, total: int) -> None:
        step = max(total // _PROGRESS_STEPS, 1)
        if stage != self._stage or done == total or done - self._printed >= step:
            self._stage, self._printed = stage, done
            print(progress_line(stage, done, total), flush=True)
