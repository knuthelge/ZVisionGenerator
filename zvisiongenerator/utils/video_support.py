"""Refuse a video request that the model's family cannot do on this platform."""

from __future__ import annotations

from typing import Any

from zvisiongenerator.backends import video_capabilities
from zvisiongenerator.utils.config import video_family

__all__ = ["unsupported_video_request"]


def unsupported_video_request(config: dict[str, Any], family: str, *, image: bool, extend: bool, review: bool, end_image: bool = False, upscale: bool = False, audio_track: bool = False) -> str | None:
    """Return the reason the family cannot run the requested workflow here, or ``None`` when it can.

    A family the config does not describe is not refused here; the platform checks of the caller still apply.

    Args:
        config: The loaded config.
        family: The detected video model family.
        image: Whether the request animates a reference image.
        extend: Whether the request extends a clip.
        review: Whether the request pauses after the half-size render of an upscale.
        end_image: Whether the request ends the clip on a last-frame image.
        upscale: Whether the request upscales the clip.
        audio_track: Whether the request carries an audio track for the clip to follow.

    Returns:
        A message naming the family and what it cannot do, or ``None``.

    Raises:
        ValueError: If the family's description in the config is malformed.
    """
    described = video_family(config, family)
    capabilities = video_capabilities(config, family)
    if described is None or capabilities is None:
        return None
    label = described.label
    if image and not capabilities.i2v:
        return f"{label} does not support image-to-video. Pick another video model."
    if end_image and not capabilities.end_frame:
        return f"{label} does not support a last-frame image. Pick another video model."
    if extend and not capabilities.extend:
        return f"{label} can't extend a clip. Pick another video model."
    if audio_track and not capabilities.audio_track:
        return f"{label} can't follow an audio track. Pick another video model."
    if upscale and not capabilities.upscale:
        return f"{label} can't upscale a clip on this platform."
    if upscale and audio_track and not capabilities.audio_track_upscale:
        return f"{label} can't upscale a clip made from your track. Turn Upscale off."
    if review and not capabilities.upscale_review:
        return f"{label} can't pause for review before upscaling on this platform."
    return None
