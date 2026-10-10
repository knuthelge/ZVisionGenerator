"""Pixel patchify and pixel-shuffle rearrangements of the diffusion decoder.

Mirrors upstream ``ops.patchify`` (``b c (f p) (h q) (w r) -> b (c p r q) f h w`` with ``p=1``)
and ``LinearPixelShuffleUpsample``'s ``b t h w (c p1 p2 p3) -> b (t p1) (h p2) (w p3) c``. The
channel orders are load-bearing: ``r`` (the W sub-index) precedes ``q`` (the H sub-index).
"""

from __future__ import annotations

import mlx.core as mx


def patchify_pixels(x: mx.array, patch: int) -> mx.array:
    """``(B, C, F, H, W)`` pixels -> ``(B, F, H/p, W/p, C*p*p)`` tokens, channel ``c*p*p + r*p + q``."""
    b, c, f, h, w = x.shape
    p = patch
    x = x.reshape(b, c, f, h // p, p, w // p, p)  # (b, c, f, hq, q, wr, r)
    x = x.transpose(0, 2, 3, 5, 1, 6, 4)  # (b, f, hq, wr, c, r, q)
    return x.reshape(b, f, h // p, w // p, c * p * p)


def unpatchify_pixels(tokens: mx.array, patch: int, out_channels: int) -> mx.array:
    """Exact inverse of :func:`patchify_pixels`: ``(B, F, H/p, W/p, C*p*p)`` -> ``(B, C, F, H, W)``."""
    b, f, hq, wr, _ = tokens.shape
    p, c = patch, out_channels
    x = tokens.reshape(b, f, hq, wr, c, p, p)  # (b, f, hq, wr, c, r, q)
    x = x.transpose(0, 4, 1, 2, 6, 3, 5)  # (b, c, f, hq, q, wr, r)
    return x.reshape(b, c, f, hq * p, wr * p)


def pixel_shuffle_3d(x: mx.array, stride: tuple[int, int, int]) -> mx.array:
    """``(B, T, H, W, C*p1*p2*p3)`` -> ``(B, T*p1, H*p2, W*p3, C)``, channel ``c*p1p2p3 + i1*p2p3 + i2*p3 + i3``."""
    b, t, h, w, cp = x.shape
    p1, p2, p3 = stride
    c = cp // (p1 * p2 * p3)
    x = x.reshape(b, t, h, w, c, p1, p2, p3)
    x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)  # (b, t, p1, h, p2, w, p3, c)
    return x.reshape(b, t * p1, h * p2, w * p3, c)
