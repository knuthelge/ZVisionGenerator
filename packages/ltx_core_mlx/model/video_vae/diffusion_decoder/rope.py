"""Axial rotary embedding of the diffusion decoder (upstream ``rope_math`` / ``det_attn_rope``).

Head channels are split ``(d_t, d_h, d_w)`` = (16, 24, 24) for head_dim 64; each block is
rotated by its axis position with **interleaved** pairs ``(x[2i], x[2i+1])`` in fp32.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def rope_dim_split(head_dim: int) -> tuple[int, int, int]:
    d_t = (head_dim // 4) // 2 * 2
    d_hw = (head_dim - d_t) // 2
    if d_hw % 2 == 1:
        raise ValueError(
            f"head_dim {head_dim} gives an odd RoPE split {(d_t, d_hw, d_hw)}; interleaved rotation needs even dims"
        )
    return d_t, d_hw, d_hw


def inv_freqs(dim: int) -> mx.array:
    """``1 / 10000 ** (arange(0, dim, 2) / dim)`` computed in float64, stored fp32."""
    f = 1.0 / 10000.0 ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    return mx.array(f.astype(np.float32))


def _rotate(x: mx.array, angle: mx.array) -> mx.array:
    """Rotate interleaved pairs of the last axis of ``x`` by ``angle`` (broadcastable to ``x[..., ::2]``)."""
    if x.shape[-1] == 0:
        return x

    xe, xo = x[..., 0::2], x[..., 1::2]
    cos, sin = mx.cos(angle), mx.sin(angle)
    re, ro = xe * cos - xo * sin, xe * sin + xo * cos
    return mx.stack([re, ro], axis=-1).reshape(x.shape)


def apply_axial_rope(
    x: mx.array,
    t_pos: mx.array,
    h_pos: mx.array,
    w_pos: mx.array,
    split: tuple[int, int, int],
    inv: tuple[mx.array, mx.array, mx.array],
) -> mx.array:
    """Rotate ``x`` of shape ``(B, T, H, W, heads, head_dim)`` by axis positions (fp32 math)."""
    d_t, d_h, _ = split
    xf = x.astype(mx.float32)
    parts = [xf[..., :d_t], xf[..., d_t : d_t + d_h], xf[..., d_t + d_h :]]
    # angle shapes broadcast against (B, T, H, W, heads, d/2)
    angles = [
        (t_pos[:, None] * inv[0][None, :])[None, :, None, None, None, :],
        (h_pos[:, None] * inv[1][None, :])[None, None, :, None, None, :],
        (w_pos[:, None] * inv[2][None, :])[None, None, None, :, None, :],
    ]
    out = mx.concatenate([_rotate(p, a) for p, a in zip(parts, angles, strict=True)], axis=-1)
    return out.astype(x.dtype)
