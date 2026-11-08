"""Model metadata: the facts about the built-in models, read from the packaged ``model_metadata.yaml``.

The file is read directly and never merged with the user's ``config.yaml``, so a user file cannot change a fact. The parse is
cached and shared: callers read it and never write into it. Only the readers that merge settings into facts copy, at the merge.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from functools import cache
from pathlib import Path
from typing import Any, Final

import yaml

__all__ = [
    "METADATA_FILE",
    "PLATFORM_KEYS",
    "YAML_LOADER",
    "alias_map",
    "builtin_aliases",
    "built_model_ids",
    "family_block",
    "for_platform",
    "image_model_family",
    "load_model_metadata",
    "model_capabilities",
    "model_display_name",
    "model_download_size_label",
    "model_entry",
    "model_ids",
]

METADATA_FILE: Final = Path(__file__).with_name("model_metadata.yaml")
"""The packaged metadata file."""

YAML_LOADER: Final = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
"""libyaml's safe loader when PyYAML was built with it (about 8x faster on the packaged files), else the pure-Python one."""

PLATFORM_KEYS: Final = frozenset({"darwin", "win32", "linux"})
"""The keys of a per-platform value: ``sys.platform`` values the app runs on."""

_PLATFORM_ORDER: Final = ("darwin", "win32", "linux")
"""The platform keys in the order an alias value lists them."""

_ALIAS_KINDS: Final = frozenset({"image", "video"})
"""The kinds of ``models`` entry that are model aliases."""

_MODEL_ROLE: Final = "model"
"""The role of a download target that is the model itself (the default role)."""

_OVERLAY_SKIPPED_KEYS: Final = frozenset({"model_type"})
"""The keys of a version block that describe the version and are not overlaid on the family block."""


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
    return load_model_metadata()["models"].get(model_id)


def model_ids(kind: str | None = None) -> tuple[str, ...]:
    """Return the ids of the ``models`` entries in file order; only those of *kind* when given."""
    models = load_model_metadata()["models"]
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


def builtin_aliases() -> dict[str, str | dict[str, str]]:
    """Return the built-in model aliases in file order, each as the source it loads, in the shape of a ``model_aliases`` value.

    An alias is an image or video entry. Its source on a platform is the ``ref`` of the first ``role: model`` target of its
    downloads there, or the entry's own id (the folder it is built into) for a ``build`` entry. The value is that string when
    every platform agrees, else a mapping of platform key to source. The result is a new dict, but nothing in it is shared
    with the cached parse.
    """
    aliases: dict[str, str | dict[str, str]] = {}
    for model_id, entry in load_model_metadata()["models"].items():
        if entry["kind"] not in _ALIAS_KINDS:
            continue
        sources = {platform_key: _alias_source(model_id, entry, platform_key) for platform_key in _PLATFORM_ORDER}
        distinct = set(sources.values())
        aliases[model_id] = distinct.pop() if len(distinct) == 1 else sources
    return aliases


def alias_map(config: Mapping[str, Any]) -> dict[str, str | dict[str, str]]:
    """Return every model alias: the built-in ones in file order, then the user's ``model_aliases`` whose names are not built in.

    A built-in name always wins, so a user alias that reuses one is ignored.
    """
    aliases = builtin_aliases()
    for name, value in (config.get("model_aliases") or {}).items():
        aliases.setdefault(name, value)
    return aliases


def model_display_name(name: str) -> str:
    """Return how the model *name* reads in the Web UI: its entry's ``name``, else *name* itself."""
    entry = model_entry(name)
    return entry["name"] if entry is not None else name


def model_download_size_label(name: str, platform_key: str) -> str | None:
    """Return the one-time download size of the model *name* on *platform_key*, such as ``"35 GB"``, or ``None`` when it has none."""
    entry = model_entry(name)
    return for_platform(entry.get("download_size"), platform_key) if entry is not None else None


def image_model_family(name: str) -> str | None:
    """Return the image family the metadata names for the image model *name*, or ``None`` when it names none.

    Only some image entries name a family; the Web UI lists those without the network round-trip that detecting a family from
    the repo takes. A model of another kind has none.
    """
    entry = model_entry(name)
    return entry.get("family") if entry is not None and entry["kind"] == "image" else None


def model_capabilities(family: str, variant: str | None = None) -> dict[str, Any]:
    """Return an image model family's capability flags, dimension limits and negative prompt hint from the model metadata, with permissive defaults.

    The family's values apply to every variant; a capability key set in ``variants[variant]`` overrides them. Without
    *variant* the result is the family's own, which is what callers that only know the family look up. The result is
    a new dict.
    """
    preset = load_model_metadata()["image_families"].get(family, {})
    variant_preset = (preset.get("variants") or {}).get(variant, {}) if variant else {}

    def capability(key: str, default: Any) -> Any:
        return variant_preset.get(key, preset.get(key, default))

    return {
        "supports_negative_prompt": capability("supports_negative_prompt", False),
        "supports_img2img": capability("supports_img2img", True),
        "supports_upscale": capability("supports_upscale", True),
        "supports_quantize": capability("supports_quantize", True),
        "supports_scheduler": capability("supports_scheduler", True),
        "supports_json_prompt": capability("supports_json_prompt", False),
        "supports_first_sigma": capability("supports_first_sigma", False),
        "max_reference_images": capability("max_reference_images", 0),
        "negative_prompt_hint": capability("negative_prompt_hint", None),
        "dimension_min": capability("dimension_min", 16),
        "dimension_max": capability("dimension_max", None),
        "dimension_step": capability("dimension_step", 16),
    }


def built_model_ids() -> tuple[str, ...]:
    """Return the ids of the models built on first use (the entries with a ``build``), in file order."""
    return tuple(model_id for model_id, entry in load_model_metadata()["models"].items() if "build" in entry)


def family_block(family: str, version: str | None) -> dict[str, Any]:
    """Return a copy of the ``video_families`` block of *family* with the version *version* overlaid; the family block alone for ``None``.

    The overlay merges the way user config does: dicts merge key by key, every other value (a list, ``None``, an empty list)
    replaces. The ``versions`` key is not part of the result, nor is a version's ``model_type``. The result is the caller's
    to edit: it shares nothing with the cached parse.

    Raises:
        KeyError: If the metadata has no such family or the family has no such version.
    """
    families = load_model_metadata()["video_families"]
    if family not in families:
        raise KeyError(f"No video family '{family}' in the model metadata.")
    block = families[family]
    merged = {key: deepcopy(value) for key, value in block.items() if key != "versions"}
    if version is not None:
        versions = block.get("versions", {})
        if version not in versions:
            raise KeyError(f"No version '{version}' of video family '{family}' in the model metadata.")
        _overlay(merged, {key: deepcopy(value) for key, value in versions[version].items() if key not in _OVERLAY_SKIPPED_KEYS})
    return merged


def _alias_source(model_id: str, entry: Mapping[str, Any], platform_key: str) -> str:
    """Return the source the alias *model_id* loads on *platform_key*."""
    if "build" in entry:
        return model_id
    return next(target["ref"] for target in for_platform(entry["downloads"], platform_key) if target.get("role", _MODEL_ROLE) == _MODEL_ROLE)


def _overlay(base: dict[str, Any], overlay: Mapping[str, Any]) -> None:
    """Merge *overlay* into *base* in place: dicts merge recursively, every other value replaces."""
    for key, value in overlay.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            _overlay(base[key], value)
        else:
            base[key] = value
