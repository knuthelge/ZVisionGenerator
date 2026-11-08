"""Resolve platform-aware alias values."""

from __future__ import annotations

from typing import Final

PLATFORM_LABELS: Final = {"darwin": "macOS", "win32": "Windows", "linux": "Linux"}
"""How each platform key reads in messages."""

AliasValue = str | dict[str, str]
AliasMap = dict[str, AliasValue]


def resolve_alias(alias_value: AliasValue, platform_key: str) -> str:
    """Resolve a single polymorphic alias for a platform.

    Args:
        alias_value: Raw alias value from config.
        platform_key: Platform key such as ``darwin`` or ``win32``.

    Returns:
        The resolved repository identifier.

    Raises:
        ValueError: If the alias is unavailable or missing for the platform.
    """
    if isinstance(alias_value, str):
        return alias_value

    if platform_key not in alias_value:
        available_platforms = sorted(alias_value)
        raise ValueError(f"Model alias is not available for {platform_key}. Available platforms: {', '.join(available_platforms)}")

    return alias_value[platform_key]


__all__ = ["PLATFORM_LABELS", "AliasMap", "AliasValue", "resolve_alias"]
