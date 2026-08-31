"""Discover authoritative backend model inventory for Web UI contracts."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zvisiongenerator.converters.list_assets import list_models, list_video_models
from zvisiongenerator.utils.image_model_detect import detect_image_model
from zvisiongenerator.utils.paths import resolve_model_path
from zvisiongenerator.utils.video_model_detect import detect_video_model


@dataclass(frozen=True)
class ImageInventoryEntry:
    """Describe one authoritative image-model inventory item."""

    name: str
    family: str
    size: str | None
    source: str
    resolved_path: str


@dataclass(frozen=True)
class VideoInventoryEntry:
    """Describe one authoritative video-model inventory item."""

    name: str
    family: str
    supports_i2v: bool
    source: str
    resolved_path: str


def declared_image_family(app_config: dict[str, Any], alias_name: str) -> str | None:
    """Return the config-declared image family for *alias_name*, if any.

    Declared families let the Web UI surface known aliases (e.g. ``ideo``)
    without a network round-trip through ``detect_image_model``.

    Args:
        app_config: Loaded config.yaml dict.
        alias_name: The model alias to look up.

    Returns:
        The declared family string, or ``None`` when the alias has no
        declared family.
    """
    families = app_config.get("model_alias_families", {})
    value = families.get(alias_name)
    return value if isinstance(value, str) and value else None


def discover_image_inventory(
    app_config: dict[str, Any],
    data_dir: Path,
    *,
    list_installed: Any = None,
    resolve_alias_path: Any = None,
    detect_model: Any = None,
) -> tuple[ImageInventoryEntry, ...]:
    """Return the canonical image inventory across installed models and aliases."""
    list_installed = list_models if list_installed is None else list_installed
    resolve_alias_path = resolve_model_path if resolve_alias_path is None else resolve_alias_path
    detect_model = detect_image_model if detect_model is None else detect_model
    entries: dict[str, ImageInventoryEntry] = {
        entry.name: ImageInventoryEntry(
            name=entry.name,
            family=getattr(entry, "family", "unknown"),
            size=getattr(entry, "size", None),
            source="installed",
            resolved_path=str((data_dir / "models" / entry.name).resolve()),
        )
        for entry in list_installed(data_dir)
    }
    aliases = app_config.get("model_aliases", {})
    for alias_name in sorted(aliases):
        if alias_name in entries:
            continue
        resolved_path = _resolve_alias_path(alias_name, aliases, resolve_alias_path)
        if resolved_path is None:
            continue
        declared = declared_image_family(app_config, alias_name)
        if declared is not None:
            family, size = declared, None
        else:
            try:
                info = detect_model(resolved_path)
            except Exception:
                continue
            family = getattr(info, "family", "unknown")
            size = getattr(info, "size", None)
            if family == "unknown":
                continue
        entries[alias_name] = ImageInventoryEntry(
            name=alias_name,
            family=family,
            size=size,
            source="alias",
            resolved_path=resolved_path,
        )
    return tuple(entries[name] for name in sorted(entries))


def discover_video_inventory(
    app_config: dict[str, Any],
    data_dir: Path,
    *,
    list_installed: Any = None,
    resolve_alias_path: Any = None,
    detect_model: Any = None,
) -> tuple[VideoInventoryEntry, ...]:
    """Return the canonical video inventory across installed models and aliases."""
    list_installed = list_video_models if list_installed is None else list_installed
    resolve_alias_path = resolve_model_path if resolve_alias_path is None else resolve_alias_path
    detect_model = detect_video_model if detect_model is None else detect_model
    entries: dict[str, VideoInventoryEntry] = {
        entry.name: VideoInventoryEntry(
            name=entry.name,
            family=getattr(entry, "family", "unknown"),
            supports_i2v=getattr(entry, "supports_i2v", False),
            source="installed",
            resolved_path=str((data_dir / "models" / entry.name).resolve()),
        )
        for entry in list_installed(data_dir)
    }
    aliases = app_config.get("model_aliases", {})
    for alias_name in sorted(aliases):
        if alias_name in entries:
            continue
        resolved_path = _resolve_alias_path(alias_name, aliases, resolve_alias_path)
        if resolved_path is None:
            continue
        try:
            info = detect_model(resolved_path)
        except Exception:
            continue
        if getattr(info, "family", "unknown") == "unknown":
            continue
        entries[alias_name] = VideoInventoryEntry(
            name=alias_name,
            family=getattr(info, "family", "unknown"),
            supports_i2v=getattr(info, "supports_i2v", False),
            source="alias",
            resolved_path=resolved_path,
        )
    return tuple(entries[name] for name in sorted(entries))


def inventory_names(entries: tuple[ImageInventoryEntry, ...] | tuple[VideoInventoryEntry, ...]) -> tuple[str, ...]:
    """Return the stable ordered names from an inventory sequence."""
    return tuple(entry.name for entry in entries)


def resolve_default_inventory_name(
    preferred: str | None,
    entries: tuple[ImageInventoryEntry, ...] | tuple[VideoInventoryEntry, ...],
) -> str | None:
    """Resolve the effective default inventory name from the canonical list."""
    names = inventory_names(entries)
    if preferred in names:
        return preferred
    return names[0] if names else None


def _resolve_alias_path(alias_name: str, aliases: dict[str, Any], resolver: Any) -> str | None:
    try:
        return resolver(alias_name, aliases=aliases, platform_key=sys.platform)
    except RuntimeError:
        return None
    except ValueError:
        return None
