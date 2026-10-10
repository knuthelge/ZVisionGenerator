"""Basic layers of the diffusion video decoder: RMSNorm, SwiGLU, timestep MLP, AdaLN, upsample."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.model.video_vae.diffusion_decoder.patching import pixel_shuffle_3d


class RMSNorm(nn.Module):
    """``x / sqrt(mean(x^2) + eps) * weight`` computed in fp32, returned in the input dtype (torch semantics)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x.astype(mx.float32), self.weight.astype(mx.float32), self.eps).astype(x.dtype)


class SwiGLU(nn.Module):
    """``w_down(silu(w_gate(x)) * w_up(x))``, no biases."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w_down(nn.silu(self.w_gate(x)) * self.w_up(x))


def sinusoidal_timestep_embedding(t: mx.array, dim: int) -> mx.array:
    """``[cos(t*f) | sin(t*f)]`` with ``f = exp(-ln(1e4) * arange(dim/2) / (dim/2))`` (fp32)."""
    import numpy as np

    half = dim // 2
    # Use numpy to generate freqs to match test reference exactly (avoids MLX vs numpy precision diff)
    freqs_np = np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float32) / half)
    freqs = mx.array(freqs_np)
    t_cast = t.astype(mx.float32)
    # Reshape for broadcasting: (B, 1) * (1, half) -> (B, half)
    args = t_cast.reshape(-1, 1) * freqs.reshape(1, -1)
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding -> Linear -> SiLU -> Linear. Keys ``mlp.0.*`` / ``mlp.2.*`` as in the pack."""

    def __init__(self, freq_dim: int, hidden: int) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = [nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)]

    def __call__(self, t: mx.array) -> mx.array:
        h = sinusoidal_timestep_embedding(t, self.freq_dim).astype(self.mlp[0].weight.dtype)
        return self.mlp[2](self.mlp[1](self.mlp[0](h)))


class SharedAdaLN(nn.Module):
    """``proj(silu(t_emb))`` split into 7 modulation vectors ``(B, 1, 1, 1, dim)``.

    Order: scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp, gate_ctx (gates unused).
    """

    def __init__(self, hidden: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, 7 * dim)
        self.dim = dim

    def __call__(self, t_emb: mx.array) -> list[mx.array]:
        h = self.proj(nn.silu(t_emb))
        b = h.shape[0]
        return [c.reshape(b, 1, 1, 1, self.dim) for c in mx.split(h, 7, axis=-1)]


class LinearPixelShuffleUpsample(nn.Module):
    """Linear (``C -> prod(stride) * C / reduction``) then pixel shuffle; optional leading-frame drop."""

    def __init__(self, in_channels: int, stride: tuple[int, int, int], reduction: int) -> None:
        super().__init__()
        self.stride = stride
        p = stride[0] * stride[1] * stride[2]
        self.proj = nn.Linear(in_channels, p * in_channels // reduction)

    def __call__(self, x: mx.array, *, drop_leading_frame: bool) -> mx.array:
        y = pixel_shuffle_3d(self.proj(x), self.stride)
        if self.stride[0] == 2 and drop_leading_frame:
            y = y[:, 1:]
        return y
