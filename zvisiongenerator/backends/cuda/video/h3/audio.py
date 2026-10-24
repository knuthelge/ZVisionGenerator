"""Load the MiniMax H3 audio VAE and decode generated audio latents to a stereo waveform.

The audio VAE is mono and takes the two stereo channels as two batch items. The latent denormalisation follows
diffusers' ``MiniMaxH3AudioDecodeStep`` (``decoders.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

__all__ = ["AUDIO_SAMPLE_RATE", "decode_audio", "load_audio_vae"]

AUDIO_SAMPLE_RATE = 32000


def load_audio_vae(audio_vae_dir: Path, device: Any) -> Any:
    """Load the audio VAE on *device* in eval mode with gradients off.

    The weights stay fp32 whatever dtype is asked: the model's ``_keep_in_fp32_modules`` covers every module, since
    the decoder is audibly worse in bf16.

    Args:
        audio_vae_dir: The ``audio_vae`` folder (``config.json`` and weights).
        device: Where the decoder runs.
    """
    import torch
    from diffusers import AutoencoderKLMiniMaxH3Audio

    vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(str(audio_vae_dir), torch_dtype=torch.bfloat16).to(device).eval()
    vae.requires_grad_(False)
    return vae


def decode_audio(audio_vae: Any, audio_latents: torch.Tensor) -> torch.Tensor:
    """Decode audio latents to a stereo waveform, without gradients.

    Args:
        audio_vae: The audio VAE from :func:`load_audio_vae`.
        audio_latents: Normalised latents ``(2, channels, frames)``, the two stereo channels as a batch.

    Returns:
        A float32 CPU waveform ``(2, samples)`` at :data:`AUDIO_SAMPLE_RATE`, the layout ``encode_video`` takes.

    Raises:
        ValueError: If *audio_latents* is not ``(2, channels, frames)``.
    """
    import torch

    if audio_latents.ndim != 3 or audio_latents.shape[0] != 2:
        raise ValueError(f"Expected audio latents shaped (2, channels, frames); got {tuple(audio_latents.shape)}.")
    mean = torch.tensor(audio_vae.config.latents_mean, device=audio_latents.device).view(1, -1, 1)
    std = torch.tensor(audio_vae.config.latents_std, device=audio_latents.device).view(1, -1, 1)
    with torch.no_grad():
        waveform = audio_vae.decode(audio_latents * std + mean, return_dict=False)[0]
    return waveform.float().permute(1, 0, 2)[0].cpu()
