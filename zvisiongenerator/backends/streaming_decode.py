"""Decode LTX-2 video latents a few frames at a time, straight to 8-bit frames in system memory.

diffusers' frame-wise decode (``AutoencoderKLLTX2Video._temporal_tiled_decode``) keeps every decoded tile on the GPU,
each overlapping the next, and joins them at the end: about three times the decoded clip at its peak. This module
runs the same tiles with the same blend, but holds only the previous tile for blending and moves each finished chunk
to the CPU as 8-bit frames, so decoding memory stays flat however long the clip is.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import torch

__all__ = ["decode_frames"]


def decode_frames(vae: Any, latents: torch.Tensor) -> list[np.ndarray]:
    """Decode one clip's denormalized *latents* ``(1, channels, frames, height, width)`` to ``height x width x 3`` uint8 frames.

    The tiling follows the VAE's own settings (``tile_sample_min_num_frames``, ``tile_sample_stride_num_frames``,
    spatial tiling when ``use_tiling`` is on), so the frames equal those of diffusers' frame-wise decode.
    """
    num_sample_frames = (latents.shape[2] - 1) * vae.temporal_compression_ratio + 1
    frames: list[np.ndarray] = []
    for chunk in _decoded_chunks(vae, latents.to(vae.dtype)):
        frames.extend(_to_uint8(chunk))
    return frames[:num_sample_frames]


def _decoded_chunks(vae: Any, latents: torch.Tensor) -> Iterator[torch.Tensor]:
    """Yield the decoded clip ``(1, 3, frames, height, width)`` in time order, one blended tile at a time."""
    tile_frames = vae.tile_sample_min_num_frames // vae.temporal_compression_ratio
    stride = vae.tile_sample_stride_num_frames // vae.temporal_compression_ratio
    blend = vae.tile_sample_min_num_frames - vae.tile_sample_stride_num_frames
    previous = None
    for start in range(0, latents.shape[2], stride):
        decoded = _decode_tile(vae, latents[:, :, start : start + tile_frames + 1])
        if previous is None:
            yield decoded[:, :, : vae.tile_sample_stride_num_frames + 1]
        else:
            decoded = vae.blend_t(previous, decoded[:, :, :-1], blend)
            yield decoded[:, :, : vae.tile_sample_stride_num_frames]
        previous = decoded


def _decode_tile(vae: Any, tile: torch.Tensor) -> torch.Tensor:
    """Decode one span of latent frames, in spatial tiles when the VAE tiles and the span is wider or taller than a tile."""
    min_height = vae.tile_sample_min_height // vae.spatial_compression_ratio
    min_width = vae.tile_sample_min_width // vae.spatial_compression_ratio
    if vae.use_tiling and (tile.shape[-1] > min_width or tile.shape[-2] > min_height):
        return vae.tiled_decode(tile, None, return_dict=True).sample
    return vae.decoder(tile, None)


def _to_uint8(chunk: torch.Tensor) -> list[np.ndarray]:
    """Map decoded frames in ``[-1, 1]`` to uint8 ``height x width x 3`` arrays on the CPU, as diffusers' image output does."""
    import torch

    video = (chunk[0].float() * 0.5 + 0.5).clamp(0, 1).mul(255).round().to(torch.uint8)  # (3, frames, height, width)
    return list(video.permute(1, 2, 3, 0).cpu().numpy())
