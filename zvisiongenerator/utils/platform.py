"""Platform metadata — PlatformInfo dataclass and config-driven resolvers."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlatformInfo:
    """Immutable platform metadata resolved from config."""

    key: str
    label: str
    message: str


def get_platform_info(
    config: dict[str, Any],
    platform: str | None = None,
) -> PlatformInfo:
    """Resolve platform metadata from config.

    Args:
        config: Parsed config dict (from load_config()).
        platform: Platform key to resolve. Defaults to sys.platform.

    Returns:
        PlatformInfo for the requested platform. If the platform is not
        in config, returns a PlatformInfo with label=platform key and
        a descriptive fallback message.
    """
    key = platform if platform is not None else sys.platform
    platforms = config.get("platforms", {})
    entry = platforms.get(key)
    if entry is None:
        supported = ", ".join(e.get("label", k) for k, e in platforms.items())
        return PlatformInfo(
            key=key,
            label=key,
            message=f"Platform '{key}' is not supported. Supported platforms: {supported}.",
        )
    return PlatformInfo(
        key=key,
        label=entry.get("label", key),
        message=entry.get("message", ""),
    )


def get_platform_label(
    config: dict[str, Any],
    platform: str | None = None,
) -> str:
    """Return the human-readable label for a platform key.

    Falls back to the raw platform string if not in config.
    Does NOT raise — always returns a usable string.
    """
    return get_platform_info(config, platform).label


def get_all_platform_labels(config: dict[str, Any]) -> dict[str, str]:
    """Return {platform_key: label} for all platforms defined in config.

    Used by paths.py and list_assets.py to replace hardcoded dicts.
    """
    return {k: e.get("label", k) for k, e in config.get("platforms", {}).items()}
