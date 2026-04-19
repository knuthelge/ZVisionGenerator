"""Resolve platform metadata and platform-aware alias values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

AliasValue = str | dict[str, str | dict[str, str]]
AliasMap = dict[str, AliasValue]


@dataclass(frozen=True)
class PlatformInfo:
    """Store immutable platform metadata."""

    key: str
    label: str


def resolve_alias(
    alias_value: AliasValue,
    platform_key: str,
    *,
    platform_labels: dict[str, str] | None = None,
) -> str:
    """Resolve a single polymorphic alias for a platform.

    Args:
        alias_value: Raw alias value from config.
        platform_key: Platform key such as ``darwin`` or ``win32``.
        platform_labels: Optional human-readable platform labels.

    Returns:
        The resolved repository identifier.

    Raises:
        ValueError: If the alias is unavailable or missing for the platform.
    """
    if isinstance(alias_value, str):
        return alias_value

    if platform_key not in alias_value:
        available_platforms = sorted(alias_value)
        display_names = [platform_labels.get(key, key) if platform_labels else key for key in available_platforms]
        available_text = ", ".join(display_names)
        requested_label = platform_labels.get(platform_key, platform_key) if platform_labels else platform_key
        raise ValueError(f"Model alias is not available for {requested_label}. Available platforms: {available_text}")

    resolved_value = alias_value[platform_key]
    if isinstance(resolved_value, str):
        return resolved_value

    message = resolved_value.get("message")
    if message:
        raise ValueError(message)

    platform_label = platform_labels.get(platform_key, platform_key) if platform_labels else platform_key
    raise ValueError(f"Model alias is unavailable for {platform_label}.")


def get_platform_info(config: dict[str, Any], platform_key: str) -> PlatformInfo:
    """Look up platform metadata from config.

    Args:
        config: Loaded config mapping.
        platform_key: Platform key such as ``darwin`` or ``win32``.

    Returns:
        Platform info for the requested key. Unknown keys fall back to the raw key.
    """
    platforms = config.get("platforms", {})
    if isinstance(platforms, dict):
        label = platforms.get(platform_key)
        if isinstance(label, str) and label.strip():
            return PlatformInfo(key=platform_key, label=label)
    return PlatformInfo(key=platform_key, label=platform_key)


__all__ = ["AliasMap", "AliasValue", "PlatformInfo", "get_platform_info", "resolve_alias"]
