"""Clip geometry of the MiniMax H3 transformer: how a canvas and a frame count become latent cells and rows."""

from __future__ import annotations

from dataclasses import dataclass

from zvisiongenerator.utils.h3_memory import (
    H3_AUDIO_CHANNELS,
    H3_PATCH,
    H3_ROW_PIXELS,
    H3_SPATIAL_RATIO,
    h3_audio_latents,
    h3_latent_frames,
)

__all__ = ["Geometry", "geometry"]


@dataclass(frozen=True)
class Geometry:
    """The latent geometry of one clip canvas: what the transformer sees as rows."""

    width: int
    height: int
    num_frames: int
    latent_frames: int
    latent_height: int
    latent_width: int
    audio_latents: int

    @property
    def video_rows(self) -> int:
        """The video rows: one per ``2 x 2`` patch of every latent frame."""
        return self.latent_frames * (self.latent_height // H3_PATCH) * (self.latent_width // H3_PATCH)

    @property
    def audio_rows(self) -> int:
        """The audio rows: the audio latents of both channels."""
        return self.audio_latents * H3_AUDIO_CHANNELS


def geometry(width: int, height: int, num_frames: int) -> Geometry:
    """Return the latent geometry of a ``width x height x num_frames`` clip.

    Raises:
        ValueError: If the canvas is not a positive multiple of 32, or *num_frames* is not of the form ``17 * n + 5``.
    """
    if width < 1 or height < 1 or width % H3_ROW_PIXELS or height % H3_ROW_PIXELS:
        raise ValueError(f"Canvas {width}x{height} must be a positive multiple of {H3_ROW_PIXELS}.")
    return Geometry(
        width=width,
        height=height,
        num_frames=num_frames,
        latent_frames=h3_latent_frames(num_frames),
        latent_height=height // H3_SPATIAL_RATIO,
        latent_width=width // H3_SPATIAL_RATIO,
        audio_latents=h3_audio_latents(num_frames),
    )
