"""The 3D latent upscaler of the MiniMax H3 upscale: a network that enlarges the stage-1 video latents.

The network code is ported from LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler (``nodes/minimax_h3_latent_upscaler_3d.py``,
MIT licence, Copyright (c) 2026 LBH-123-AI). The port matches that file at upstream commit
``98934c0b5f4a213323c4d1d0e47eb3d2f27cb743``, the last commit that changed it. Attribute names are kept unchanged so the
published weights (LBH-123-AI/Minimax_h3_latent_Upscaler, Apache-2.0) load strictly.

Only the architecture of the one pinned weights file is kept: the sizes are fixed defaults, and the shape inference the
ComfyUI node does from the state dict is dropped.

This module imports torch at module level, so the backend imports it lazily.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["LatentResizer3D", "load_latent_upsampler", "upscale_latents"]

_CONFIG_KEY = "video_model_presets.h3.diffusers.latent_upsampler"
_GROUPS = 32  # groups of every GroupNorm


def _normalization(channels: int) -> nn.GroupNorm:
    """Return the group normalisation the network uses between its convolutions."""
    return nn.GroupNorm(_GROUPS, channels)


class ResBlockEmb3D(nn.Module):
    """A residual block of two 3D convolutions whose second normalisation is modulated by the scale embedding."""

    def __init__(self, channels: int, emb_channels: int, dropout: float = 0, out_channels: int | None = None) -> None:
        """Build the block.

        Args:
            channels: Input channels.
            emb_channels: Size of the scale embedding.
            dropout: Dropout probability before the second convolution (inactive in eval mode).
            out_channels: Output channels; the input channels when ``None``.
        """
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(_normalization(channels), nn.SiLU(), nn.Conv3d(channels, self.out_channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels))
        self.out_norm = _normalization(self.out_channels)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(p=dropout), nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1))
        self.skip = nn.Conv3d(channels, self.out_channels, 1) if self.out_channels != channels else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Return *x* plus its two convolutions, with the second normalisation scaled and shifted by *emb*."""
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h


class TemporalConv(nn.Module):
    """A residual depthwise convolution along the time axis, followed by a pointwise one."""

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        """Build the block with a ``kernel_size`` window along time."""
        super().__init__()
        padding = kernel_size // 2
        self.norm = _normalization(channels)
        self.dwconv = nn.Conv3d(channels, channels, kernel_size=(kernel_size, 1, 1), padding=(padding, 0, 0), groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return *x* plus its temporal convolution."""
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return x + h


class LatentResizer3D(nn.Module):
    """Enlarge normalised video latents ``(B, 24, T, H, W)`` spatially: residual blocks, a trilinear resize, residual blocks.

    The defaults are those of the pinned weights (345,280,216 parameters).
    """

    def __init__(
        self,
        in_channels: int = 24,
        in_blocks: int = 12,
        out_blocks: int = 12,
        channels: int = 512,
        dropout: float = 0.1,
        temporal_every: int = 2,
        temporal_kernel: int = 5,
    ) -> None:
        """Build the network; the arguments other than the defaults exist for tiny test instances."""
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            self.in_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))
        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            self.out_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))
        self.norm_out = _normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, scale: float, size: tuple[int, int, int]) -> torch.Tensor:
        """Return *x* resized to the ``(T, H, W)`` *size*, with the embedding of the enlargement factor *scale*."""
        scale_emb = torch.tensor([scale - 1], dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)
        x = self.conv_in(x)
        for block in self.in_blocks:
            x = block(x, emb.expand(x.shape[0], -1)) if isinstance(block, ResBlockEmb3D) else block(x)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        for block in self.out_blocks:
            x = block(x, emb.expand(x.shape[0], -1)) if isinstance(block, ResBlockEmb3D) else block(x)
        x = self.norm_out(x)
        x = F.silu(x)
        return self.conv_out(x)


def load_latent_upsampler(path: Path, device: torch.device | str, *, build: Callable[[], LatentResizer3D] = LatentResizer3D) -> LatentResizer3D:
    """Load the upscaler weights at *path* strictly, in fp16 on *device*, ready to run (eval mode, no gradients).

    Args:
        path: The pinned ``.safetensors`` file.
        device: The device to put the network on.
        build: Builds the network to load into; the default is the pinned architecture. Tests pass a tiny one.

    Raises:
        RuntimeError: If the file's tensors do not match the network, which means it is not the pinned weights file.
    """
    from safetensors.torch import load_file

    state = load_file(str(path), device="cpu")
    with torch.device("meta"):
        net = build()
    try:
        net.load_state_dict(state, strict=True, assign=True)
    except RuntimeError as exc:
        raise RuntimeError(f"{path} does not match the H3 latent upscaler; check {_CONFIG_KEY}: {exc}") from exc
    return net.to(device=device, dtype=torch.float16).eval().requires_grad_(False)


@torch.inference_mode()
def upscale_latents(net: nn.Module, latents: torch.Tensor, mean: Sequence[float], std: Sequence[float], scale: float = 2.0) -> torch.Tensor:
    """Return *latents* ``(1, 24, T, H, W)`` enlarged *scale* times along height and width, as float32 on the CPU.

    The sampler's latents are already in the video VAE's normalised space. They are normalised a second time with the
    VAE's *mean* and *std* before the network and de-normalised after it. The double normalisation is deliberate: it is
    what the LBH reference node feeds the network, and what the upscale was validated with. Do not remove it.

    Args:
        net: The upscaler (see :func:`load_latent_upsampler`).
        latents: The sampler's latents.
        mean: The VAE's ``latents_mean``, one value per channel.
        std: The VAE's ``latents_std``, one value per channel.
        scale: The enlargement factor.
    """
    parameter = next(net.parameters())
    stats = {"device": parameter.device, "dtype": parameter.dtype}
    mean_t = torch.as_tensor(list(mean), **stats).view(1, -1, 1, 1, 1)
    std_t = torch.as_tensor(list(std), **stats).view(1, -1, 1, 1, 1)
    x = latents.to(**stats)
    frames, height, width = x.shape[-3:]
    size = (frames, round(height * scale), round(width * scale))
    out = net((x - mean_t) / std_t, scale, size) * std_t + mean_t
    return out.float().cpu()
