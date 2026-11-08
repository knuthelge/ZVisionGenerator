"""Model metadata: the facts about the built-in models, read from the packaged ``model_metadata.yaml``.

The file is read directly and never merged with the user's ``config.yaml``, so a user file cannot change a fact. The parse is
cached and shared: callers read it and never write into it.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any, Final

import yaml

__all__ = [
    "METADATA_FILE",
    "PLATFORM_KEYS",
    "YAML_LOADER",
    "for_platform",
    "load_model_metadata",
    "model_entry",
    "model_ids",
]

METADATA_FILE: Final = Path(__file__).with_name("model_metadata.yaml")
"""The packaged metadata file."""

YAML_LOADER: Final = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
"""libyaml's safe loader when PyYAML was built with it (about 8x faster on the packaged files), else the pure-Python one."""

PLATFORM_KEYS: Final = frozenset({"darwin", "win32", "linux"})
"""The keys of a per-platform value: ``sys.platform`` values the app runs on."""


def load_model_metadata() -> dict[str, Any]:
    """Return the packaged metadata, parsed once per path and shared between all callers.

    The result is the cached parse itself, not a copy: never mutate it or anything reachable from it. Copy before editing.
    """
    return _load(METADATA_FILE)


@cache
def _load(path: Path) -> dict[str, Any]:
    """Parse the metadata file at *path*; cached per path."""
    return yaml.load(path.read_text(encoding="utf-8"), Loader=YAML_LOADER) or {}


def model_entry(model_id: str) -> dict[str, Any] | None:
    """Return the ``models`` entry of *model_id*, or ``None`` when the metadata has none.

    The entry is part of the shared parse: read-only; copy before editing.
    """
    return load_model_metadata().get("models", {}).get(model_id)


def model_ids(kind: str | None = None) -> tuple[str, ...]:
    """Return the ids of the ``models`` entries in file order; only those of *kind* when given."""
    models = load_model_metadata().get("models", {})
    return tuple(model_id for model_id, entry in models.items() if kind is None or entry["kind"] == kind)


def for_platform(value: Any, platform_key: str) -> Any:
    """Return *value* as it applies on *platform_key*.

    A mapping whose keys are all platform keys holds one value per platform: the one for *platform_key*, or ``None`` where
    the platform has none. Any other value applies to every platform and is returned as it is. The result is part of the
    shared parse: read-only; copy before editing.
    """
    if isinstance(value, Mapping) and value and set(value) <= PLATFORM_KEYS:
        return value.get(platform_key)
    return value
