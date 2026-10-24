"""Decode MiniMax H3 video latents chunk by chunk, each finished chunk moved to system memory as 8-bit frames.

diffusers' ``AutoencoderKLMiniMaxH3.decode`` keeps every decoded chunk on the GPU and joins them at the end. The
functions here run the same chunking (a copy of the VAE's private ``_decode``, using its ``_decode_clip`` and
``_blend``) but convert each finished chunk to uint8 and move it to the CPU at once, so GPU memory stays flat
however long the clip is. The latent denormalisation, the ImageNet revert and the clamp follow diffusers'
``MiniMaxH3VideoDecodeStep`` (``decoders.py``).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import torch

__all__ = ["PIXEL_MEAN", "PIXEL_STD", "decode_chunked", "decode_video", "denormalise", "iter_decoded_chunks", "load_video_vae", "to_uint8"]

PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
_VAE_INDEX = "diffusion_pytorch_model.safetensors.index.json"


def load_video_vae(vae_dir: Path, device: Any) -> Any:
    """Load the decoder side of the video VAE in bf16 (the cast-sensitive parameters stay fp32), reading only the decoder tensors.

    The encoder and ``quant_conv`` are replaced by identities and their weights are never read. The VAE stays on
    the CPU and moves to *device* when ``post_quant_conv`` first runs.

    Args:
        vae_dir: The ``vae`` folder (``config.json`` and the safetensors weights, sharded or single).
        device: Where the decoder runs.

    Returns:
        The VAE in eval mode with gradients off.
    """
    import torch
    from diffusers import AutoencoderKLMiniMaxH3

    from zvisiongenerator.backends.cuda.block_streaming import load_when_used
    from zvisiongenerator.backends.cuda.quant import empty_diffusers_model, fill_weights
    from zvisiongenerator.utils.safetensors_stream import iter_tensors

    config = json.loads((vae_dir / "config.json").read_text(encoding="utf-8"))
    vae = empty_diffusers_model(config, torch.bfloat16, AutoencoderKLMiniMaxH3)
    vae.encoder = torch.nn.Identity()
    vae.quant_conv = torch.nn.Identity()
    keep = set(vae._keep_in_fp32_modules or ())
    for name, parameter in vae.named_parameters():  # as from_pretrained(dtype=...) keeps them: any name part in the list, bare parameters such as scale1 and scale2 included
        if any(part in keep for part in name.split(".")):
            parameter.data = parameter.data.to(torch.float32)
    vae.requires_grad_(False)

    def wanted(name: str) -> bool:
        return not name.startswith(("encoder.", "quant_conv."))

    fill_weights(vae, (item for path in _weight_files(vae_dir) for item in iter_tensors(path, wanted)), str(vae_dir))
    vae.eval()
    load_when_used(vae, [vae.post_quant_conv], device)
    return vae


def _weight_files(vae_dir: Path) -> list[Path]:
    """Return the safetensors files of *vae_dir*: those its index lists, else every ``*.safetensors`` in it."""
    index = vae_dir / _VAE_INDEX
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return [vae_dir / name for name in sorted(set(weight_map.values()))]
    return sorted(vae_dir.glob("*.safetensors"))


def _stats(values: tuple[float, ...] | list[float], device: Any, dims: int) -> torch.Tensor:
    """Return *values* as a tensor on *device* shaped to broadcast over a *dims*-dimensional channel-first tensor."""
    import torch

    return torch.tensor(list(values), device=device).view(1, -1, *([1] * (dims - 2)))


def to_uint8(video: torch.Tensor) -> torch.Tensor:
    """Return decoder output ``(1, 3, T, H, W)`` as ``(T, H, W, 3)`` uint8: ImageNet revert, clamp to [0, 1], scale and round."""
    import torch

    mean, std = _stats(PIXEL_MEAN, video.device, 5), _stats(PIXEL_STD, video.device, 5)
    pixels = (video.float() * std + mean).clamp(0, 1)
    return (pixels * 255).round().to(torch.uint8)[0].permute(1, 2, 3, 0)


def denormalise(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    """Return *latents* in the decoder's space: ``latents * std + mean`` with the VAE's own per-channel statistics."""
    mean, std = _stats(vae.config.latents_mean, latents.device, 5), _stats(vae.config.latents_std, latents.device, 5)
    return latents * std + mean


def iter_decoded_chunks(vae: Any, z: torch.Tensor) -> Iterator[torch.Tensor]:
    """Yield the finished (cross-faded) pixel chunks of ``AutoencoderKLMiniMaxH3._decode`` one at a time, in order, without joining them.

    The chunks include the frames ``_decode`` cuts again at the end when it pads a partial last chunk;
    :func:`decode_chunked` drops them.

    Args:
        vae: The video VAE.
        z: Denormalised latents ``(1, channels, frames, height, width)``.
    """
    from diffusers.models.modeling_utils import get_parameter_dtype

    z = z.to(get_parameter_dtype(vae.post_quant_conv))
    tokens_chunk_size = vae.tokens_chunk_size
    token_drop = vae.config.token_drop
    chunk_num_frames = tokens_chunk_size * vae.temporal_compression_ratio
    num_tokens = z.shape[2] + token_drop
    pad_tokens = (-num_tokens) % tokens_chunk_size
    num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
    if pad_tokens > 0:
        z = _pad_latents(z, pad_tokens)
    overlap = None
    for index in range(num_chunks):
        start = index * tokens_chunk_size
        clip = vae._decode_clip(z[:, :, start : start + tokens_chunk_size + vae.token_overlap])
        for part in range(int(token_drop > 0) + 1):
            frame_start = part * chunk_num_frames
            chunk = clip[:, :, frame_start : frame_start + chunk_num_frames][:, :, vae.frame_pre_padding :]
            if part == 0:
                if overlap is not None:
                    chunk = vae._blend(overlap, chunk, vae.frame_overlap, dim=-3)
                yield chunk
            else:
                overlap = chunk
    if overlap is not None:
        yield overlap


def _pad_latents(z: torch.Tensor, count: int) -> torch.Tensor:
    """Append the last latent frame of *z* *count* times (the decode pads a partial last chunk this way)."""
    import torch

    return torch.cat([z, z[:, :, -1:].repeat(1, 1, count, 1, 1)], dim=2)


def _trailing_pad_frames(vae: Any, latent_frames: int) -> int:
    """Return how many trailing pixel frames ``_decode`` cuts again after padding *latent_frames* to a whole chunk."""
    pad_tokens = (-(latent_frames + vae.config.token_drop)) % vae.tokens_chunk_size
    if pad_tokens == 0:
        return 0
    intra_tail = vae.config.clip_length % vae.temporal_compression_ratio
    return sum(intra_tail if intra_tail and (latent_frames + k) % vae.tokens_chunk_size == 0 else vae.temporal_compression_ratio for k in range(pad_tokens))


def decode_chunked(vae: Any, latents: torch.Tensor) -> list[np.ndarray]:
    """Decode *latents* chunk by chunk, each finished chunk converted to uint8 and moved to the CPU at once.

    Args:
        vae: The video VAE.
        latents: Normalised latents ``(1, channels, frames, height, width)``.

    Returns:
        One contiguous ``height x width x 3`` uint8 array per frame, the padding frames trimmed.
    """
    z = denormalise(vae, latents)
    frames: list[np.ndarray] = []
    for chunk in iter_decoded_chunks(vae, z):
        frames.extend(to_uint8(chunk).cpu().contiguous().numpy())
    pad = _trailing_pad_frames(vae, int(z.shape[2]))
    return frames[:-pad] if pad else frames


def decode_video(vae: Any, latents: torch.Tensor) -> list[np.ndarray]:
    """Decode *latents* to ``height x width x 3`` uint8 frames in system memory, without gradients.

    Args:
        vae: The video VAE from :func:`load_video_vae`.
        latents: Normalised latents ``(1, channels, frames, height, width)``.

    Returns:
        The frames in order; ``17 * n + 5`` frames for ``5 * n + 2`` latent frames.

    Raises:
        ValueError: If *latents* is not a single five-dimensional clip.
    """
    import torch

    if latents.ndim != 5 or latents.shape[0] != 1:
        raise ValueError(f"Expected latents shaped (1, channels, frames, height, width); got {tuple(latents.shape)}.")
    with torch.no_grad():  # without it autograd keeps every activation of the decoder blocks
        return decode_chunked(vae, latents)
