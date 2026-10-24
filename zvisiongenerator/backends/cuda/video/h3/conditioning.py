"""Encode the conditioning media of MiniMax H3 clips: keyframes, an extension tail and its audio.

Keyframes and the tail go through the video VAE's encoder with the recipe the released model was conditioned on
(diffusers' ``encode_vae_condition``): ImageNet-normalised pixels, a posterior *sample* drawn under a fresh generator
seeded 42 whatever the request seed is, rounded to float16, then normalised with the VAE's latent statistics. The tail
audio goes through the mono audio VAE (the two stereo channels as a batch of two) taking the posterior mean, as the
reference-audio encode of ``MiniMaxH3Ref2VAReferenceEncoderStep`` does.

Everything returned lives on the CPU in float32. :func:`prenoise` mixes conditioning latents with noise the way the
released model does before it pins them in the sequence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.backends.cuda.video.h3.decode import PIXEL_MEAN, PIXEL_STD, _weight_files
from zvisiongenerator.backends.cuda.video.runtime import fit_image
from zvisiongenerator.utils.h3_memory import H3_AUDIO_CHANNELS, H3_TAIL_FRAMES, h3_audio_latents, h3_latent_frames

if TYPE_CHECKING:
    import numpy as np
    import torch
    from PIL import Image

__all__ = [
    "KEYFRAME_ENCODE_SEED",
    "TAIL_AUDIO_LATENTS",
    "TAIL_FRAMES",
    "TAIL_LATENT_FRAMES",
    "encode_keyframe",
    "encode_tail",
    "encode_tail_audio",
    "fit_keyframe",
    "load_vae_encoder",
    "prenoise",
    "tail_audio_rows",
    "tail_first_audio_order",
]

KEYFRAME_ENCODE_SEED = 42  # the seed the released model samples every conditioning posterior under
TAIL_FRAMES = H3_TAIL_FRAMES  # an extension carries 17 * 1 + 5 frames of the source, the smallest valid clip
TAIL_LATENT_FRAMES = h3_latent_frames(TAIL_FRAMES)  # 7
TAIL_AUDIO_LATENTS = h3_audio_latents(TAIL_FRAMES)  # 37 per channel
_ENCODER_PREFIXES = ("encoder.", "quant_conv.")


def load_vae_encoder(vae_dir: Path, device: Any) -> Any:
    """Load the encoder side of the video VAE on *device* in eval mode with gradients off, reading only the encoder tensors.

    The decoder and ``post_quant_conv`` are replaced by identities and their weights are never read. The encoder and
    ``quant_conv`` are in the model's float32 keep-list, so they stay float32 whatever the compute dtype; spatial
    tiling stays on, as the model was released.

    Args:
        vae_dir: The ``vae`` folder (``config.json`` and the safetensors weights, sharded or single).
        device: Where the encoder runs.
    """
    import torch
    from diffusers import AutoencoderKLMiniMaxH3

    from zvisiongenerator.backends.cuda.quant import empty_diffusers_model, fill_weights
    from zvisiongenerator.utils.safetensors_stream import iter_tensors

    config = json.loads((vae_dir / "config.json").read_text(encoding="utf-8"))
    vae = empty_diffusers_model(config, torch.bfloat16, AutoencoderKLMiniMaxH3)
    vae.decoder = torch.nn.Identity()
    vae.post_quant_conv = torch.nn.Identity()
    vae.requires_grad_(False)

    def wanted(name: str) -> bool:
        return name.startswith(_ENCODER_PREFIXES)

    fill_weights(vae, (item for path in _weight_files(vae_dir) for item in iter_tensors(path, wanted)), str(vae_dir))
    return vae.to(device).eval()


def fit_keyframe(image: Image.Image, width: int, height: int) -> Image.Image:
    """Scale *image* to cover *width* x *height* with its aspect ratio kept and crop the overflow from the centre.

    The crop is the one LTX image conditioning uses; an image already at the target size comes back unchanged.
    """
    return fit_image(image.convert("RGB"), width, height)


def _check_canvas(encoder: Any, width: int, height: int) -> None:
    """Raise ``ValueError`` unless the canvas is a positive multiple of the encoder's spatial compression ratio."""
    ratio = encoder.spatial_compression_ratio
    if width < 1 or height < 1 or width % ratio or height % ratio:
        raise ValueError(f"Canvas {width}x{height} must be a positive multiple of {ratio}.")


def _encode_condition(encoder: Any, pixels: torch.Tensor) -> torch.Tensor:
    """Return the normalised latents ``(1, C, T, h, w)`` of uint8 *pixels* ``(1, 3, T, H, W)``: diffusers' ``encode_vae_condition``."""
    import torch

    stats_shape = (1, -1, 1, 1, 1)
    latents_mean = torch.tensor(encoder.config.latents_mean).view(*stats_shape)
    latents_std = torch.tensor(encoder.config.latents_std).view(*stats_shape)
    pixel_mean = torch.tensor(PIXEL_MEAN, device=pixels.device).view(*stats_shape)
    pixel_std = torch.tensor(PIXEL_STD, device=pixels.device).view(*stats_shape)
    with torch.no_grad():
        normalised = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
        posterior = encoder.encode(normalised, return_dict=False)[0]
        latents = posterior.sample(generator=torch.Generator().manual_seed(KEYFRAME_ENCODE_SEED))
        latents = latents.to(torch.float16).float().cpu()
    return (latents - latents_mean) / latents_std


def encode_keyframe(encoder: Any, image: Image.Image, width: int, height: int) -> torch.Tensor:
    """Encode *image*, centre-cropped to *width* x *height*, as one conditioning latent frame.

    Args:
        encoder: The video VAE from :func:`load_vae_encoder`.
        image: The keyframe, any size; it is fitted by :func:`fit_keyframe`.
        width: Canvas width in pixels, a multiple of the VAE's spatial compression (16).
        height: Canvas height in pixels, likewise.

    Returns:
        Normalised float32 CPU latents ``(1, channels, 1, height / 16, width / 16)``.

    Raises:
        ValueError: If the canvas is not a positive multiple of the compression ratio.
    """
    import numpy as np
    import torch

    _check_canvas(encoder, width, height)
    device = next(encoder.encoder.parameters()).device
    array = np.asarray(fit_keyframe(image, width, height))
    pixels = torch.from_numpy(array.copy()).to(device).permute(2, 0, 1)[None, :, None]
    return _encode_condition(encoder, pixels)


def encode_tail(encoder: Any, frames: np.ndarray) -> torch.Tensor:
    """Encode the last :data:`TAIL_FRAMES` frames of a source clip as :data:`TAIL_LATENT_FRAMES` conditioning latent frames.

    Args:
        encoder: The video VAE from :func:`load_vae_encoder`.
        frames: ``(22, height, width, 3)`` uint8 RGB frames already at the canvas size.

    Returns:
        Normalised float32 CPU latents ``(1, channels, 7, height / 16, width / 16)``.

    Raises:
        ValueError: If *frames* is not 22 uint8 RGB frames of a canvas the VAE can encode.
    """
    import torch

    if frames.ndim != 4 or frames.shape[0] != TAIL_FRAMES or frames.shape[3] != 3 or frames.dtype.name != "uint8":
        raise ValueError(f"The tail must be {TAIL_FRAMES} uint8 RGB frames shaped (22, height, width, 3); got {tuple(frames.shape)} {frames.dtype}.")
    _check_canvas(encoder, int(frames.shape[2]), int(frames.shape[1]))
    device = next(encoder.encoder.parameters()).device
    pixels = torch.from_numpy(frames.copy()).to(device).permute(3, 0, 1, 2)[None]
    return _encode_condition(encoder, pixels)


def encode_tail_audio(audio_vae: Any, waveform: torch.Tensor) -> torch.Tensor:
    """Encode the audio of the source tail as :data:`TAIL_AUDIO_LATENTS` normalised latents per stereo channel.

    The mono audio VAE codes each channel as a batch item and its posterior mean is used. The latents of each channel
    are trimmed, or padded by repeating the last one, to :data:`TAIL_AUDIO_LATENTS`.

    Args:
        audio_vae: The audio VAE from ``load_audio_vae``.
        waveform: Stereo float samples ``(2, samples)`` at the audio VAE's sample rate (32 kHz).

    Returns:
        Float32 CPU rows ``(2 * 37, latent_channels)`` in channel-major order: channel 0's 37 latents, then channel 1's.

    Raises:
        ValueError: If *waveform* is not ``(2, samples)``.
    """
    import torch

    if waveform.ndim != 2 or waveform.shape[0] != H3_AUDIO_CHANNELS:
        raise ValueError(f"Expected a stereo waveform shaped (2, samples); got {tuple(waveform.shape)}.")
    device = next(audio_vae.parameters()).device
    mean = torch.tensor(audio_vae.config.latents_mean).view(1, 1, -1)
    std = torch.tensor(audio_vae.config.latents_std).view(1, 1, -1)
    with torch.no_grad():
        posterior = audio_vae.encode(waveform.to(device)[:, None], return_dict=False)[0]
        latents = posterior.mode().float().cpu().transpose(1, 2)  # (2, frames, channels)
    normalised = (latents - mean) / std
    if normalised.shape[1] < TAIL_AUDIO_LATENTS:
        padding = normalised[:, -1:].expand(-1, TAIL_AUDIO_LATENTS - normalised.shape[1], -1)
        normalised = torch.cat([normalised, padding], dim=1)
    return normalised[:, :TAIL_AUDIO_LATENTS].reshape(H3_AUDIO_CHANNELS * TAIL_AUDIO_LATENTS, -1).contiguous()


def tail_audio_rows(audio_latents: int, tail: int = TAIL_AUDIO_LATENTS) -> torch.Tensor:
    """Return the positions of the held tail rows inside the channel-major audio block of a clip.

    The block is ``audio_latents`` rows of channel 0 followed by ``audio_latents`` rows of channel 1; the tail is the
    first *tail* rows of each channel, so the result is ``0..tail-1`` and ``audio_latents..audio_latents + tail - 1``.

    Raises:
        ValueError: If *tail* is not between 1 and *audio_latents*.
    """
    positions, held = _audio_positions(audio_latents, tail)
    return positions[held]


def tail_first_audio_order(audio_latents: int, tail: int = TAIL_AUDIO_LATENTS) -> torch.Tensor:
    """Return every position of the audio block with the held tail rows first: channel 0's tail, channel 1's tail, then the rest ascending.

    Pass the result (offset by the block's start in the sequence) as the audio indices of ``row_timesteps`` with
    ``condition_audio_rows = 2 * tail``: its leading entries are the conditioning rows.

    Raises:
        ValueError: If *tail* is not between 1 and *audio_latents*.
    """
    import torch

    positions, held = _audio_positions(audio_latents, tail)
    return torch.cat([positions[held], positions[~held]])


def _audio_positions(audio_latents: int, tail: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(positions of the audio block, mask of its held tail rows)`` for a channel-major block."""
    import torch

    if not 1 <= tail <= audio_latents:
        raise ValueError(f"The audio tail ({tail}) must be between 1 and the clip's audio latents ({audio_latents}).")
    positions = torch.arange(H3_AUDIO_CHANNELS * audio_latents)
    return positions, (positions % audio_latents) < tail


def prenoise(scheduler: Any, latents: torch.Tensor, t: float, generator: torch.Generator | None) -> torch.Tensor:
    """Return *latents* mixed with fresh noise to level *t*: ``t * latents + (1 - t) * noise``.

    The released model holds video conditioning at ``t = 0.999`` and audio conditioning at ``1.0`` (clean). Noise is
    drawn from *generator* even at ``t = 1.0``, so the draws a run consumes do not depend on the level.

    Args:
        scheduler: A ``MiniMaxH3Scheduler``; only its ``scale_noise`` is used.
        latents: Clean conditioning latents.
        t: The level to hold them at, in ``[0, 1]``.
        generator: The request's generator, or ``None`` for the global one.
    """
    from diffusers.utils.torch_utils import randn_tensor

    noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
    return scheduler.scale_noise(latents, t, noise)
