"""Resolve backend-owned ratio and size defaults for Web UI contracts."""

from __future__ import annotations

from typing import Any

from zvisiongenerator.utils.config import select_ratio_size_defaults


_IMAGE_FALLBACK_RATIO = "2:3"
_IMAGE_FALLBACK_SIZE = "m"
_VIDEO_FALLBACK_RATIO = "16:9"
_VIDEO_FALLBACK_SIZE = "m"


def resolve_image_ratio_size_defaults(web_config: Any) -> tuple[str, str]:
    """Resolve the effective image ratio and size from layered config and options."""
    generation = getattr(web_config, "app_config", {}).get("generation", {})
    return select_ratio_size_defaults(
        generation.get("default_ratio"),
        tuple(getattr(web_config, "image_ratios", ())),
        dict(getattr(web_config, "image_size_options", {})),
        generation.get("default_size"),
        fallback_ratio=_IMAGE_FALLBACK_RATIO,
        fallback_size=_IMAGE_FALLBACK_SIZE,
    )


def resolve_video_ratio_size_defaults(web_config: Any) -> tuple[str, str]:
    """Resolve the effective video ratio and size from layered config and options."""
    generation = getattr(web_config, "app_config", {}).get("video_generation", {})
    return select_ratio_size_defaults(
        generation.get("default_ratio"),
        tuple(getattr(web_config, "video_ratios", ())),
        dict(getattr(web_config, "video_size_options", {})),
        generation.get("default_size"),
        fallback_ratio=_VIDEO_FALLBACK_RATIO,
        fallback_size=_VIDEO_FALLBACK_SIZE,
    )


def default_image_size_for_ratio(web_config: Any, ratio: str) -> str:
    """Resolve the effective default image size for a selected ratio."""
    generation = getattr(web_config, "app_config", {}).get("generation", {})
    _resolved_ratio, size = select_ratio_size_defaults(
        ratio,
        tuple(getattr(web_config, "image_ratios", ())),
        dict(getattr(web_config, "image_size_options", {})),
        generation.get("default_size"),
        fallback_ratio=_IMAGE_FALLBACK_RATIO,
        fallback_size=_IMAGE_FALLBACK_SIZE,
    )
    return size


def default_video_size_for_ratio(web_config: Any, ratio: str) -> str:
    """Resolve the effective default video size for a selected ratio."""
    generation = getattr(web_config, "app_config", {}).get("video_generation", {})
    _resolved_ratio, size = select_ratio_size_defaults(
        ratio,
        tuple(getattr(web_config, "video_ratios", ())),
        dict(getattr(web_config, "video_size_options", {})),
        generation.get("default_size"),
        fallback_ratio=_VIDEO_FALLBACK_RATIO,
        fallback_size=_VIDEO_FALLBACK_SIZE,
    )
    return size


def build_image_size_labels(web_config: Any) -> list[dict[str, str]]:
    """Build display labels for the effective image ratio's size options."""
    ratio, _size = resolve_image_ratio_size_defaults(web_config)
    image_sizes = getattr(web_config, "app_config", {}).get("sizes", {}).get(ratio, {})
    return [
        {
            "value": size_name,
            "label": f"{size_name} ({size_config['width']}x{size_config['height']})",
        }
        for size_name, size_config in image_sizes.items()
        if isinstance(size_config, dict) and "width" in size_config and "height" in size_config
    ]
