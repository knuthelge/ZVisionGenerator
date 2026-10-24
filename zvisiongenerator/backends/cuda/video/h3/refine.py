"""Helpers of the MiniMax H3 windowed refine: the noise drawn once, the audio and keyframes of each window, and the cross-fade.

The upscale refines the enlarged latents in overlapping windows of latent frames (:func:`~zvisiongenerator.utils.h3_memory.h3_refine_windows`),
each as a clip of its own geometry. These functions are the pieces around that loop; the loop itself, which loads the
transformer, lives in the backend. This module imports torch at module level, so the backend imports it lazily.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from zvisiongenerator.backends.cuda.video.h3.sampler import Conditioning
from zvisiongenerator.utils.h3_memory import h3_audio_latents

__all__ = ["audio_window", "draw_refine_noise", "stitch", "window_audio_rows", "window_conditioning"]


def draw_refine_noise(video_shape: Sequence[int], audio_shape: Sequence[int], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw the refine noise of the whole clip once: the video ``(1, 24, T, h, w)`` first, then the audio ``(2, 32, N)``.

    One CPU generator seeded ``seed + 1`` draws both in float32, so every window slices the same noise and a seed
    reproduces the clip. The ``+ 1`` keeps the draw apart from the stage-1 noise of *seed*.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    video = torch.randn(tuple(video_shape), generator=generator, dtype=torch.float32)
    audio = torch.randn(tuple(audio_shape), generator=generator, dtype=torch.float32)
    return video, audio


def audio_window(audio_latents: int, start: int, latent_frames: int, window_frames: int) -> tuple[int, int]:
    """Return ``(offset, count)`` of the audio latents per channel that belong to the window starting at latent frame *start*.

    The count is the audio a clip of *window_frames* frames carries, and the offset is the window's proportional
    position in the clip's *latent_frames*, held inside the *audio_latents* the clip has. A single window gives
    ``(0, audio_latents)``.

    Args:
        audio_latents: Audio latents per channel of the whole clip.
        start: The window's first latent frame.
        latent_frames: Latent frames of the whole clip.
        window_frames: Pixel frames of the window.
    """
    count = h3_audio_latents(window_frames)
    return min(round(start / latent_frames * audio_latents), audio_latents - count), count


def window_audio_rows(audio: torch.Tensor, offset: int, count: int) -> torch.Tensor:
    """Return the *count* latents from *offset* of both channels of *audio* ``(2, 32, N)`` as channel-major rows ``(2 * count, 32)``.

    Channel 0's latents come first, then channel 1's, as the sampler packs audio rows. It serves the audio latents and
    the audio noise alike.
    """
    return audio[:, :, offset : offset + count].permute(0, 2, 1).reshape(-1, audio.shape[1])


def window_conditioning(keyframes: Sequence[torch.Tensor], anchors: Sequence[str], index: int, count: int) -> Conditioning:
    """Return the keyframes window *index* of *count* holds: a ``"first"`` one in window 0 only and a ``"last"`` one in the final window only.

    A single window holds both. The windows in between hold none.
    """
    held = [(keyframe, anchor) for keyframe, anchor in zip(keyframes, anchors, strict=True) if (anchor == "first" and index == 0) or (anchor == "last" and index == count - 1)]
    return Conditioning(keyframes=tuple(keyframe for keyframe, _ in held), anchors=tuple(anchor for _, anchor in held))


def stitch(parts: Sequence[torch.Tensor], windows: Sequence[tuple[int, int]], latent_frames: int, overlap: int) -> torch.Tensor:
    """Cross-fade the refined *parts* of *windows* into one latent ``(1, 24, latent_frames, h, w)``.

    Each part weighs 1 inside its window and ramps linearly (``linspace(0, 1, overlap + 2)[1:-1]``) over *overlap* frames
    at each edge it shares with a neighbour. The result is the weighted sum divided by the weight sum, so a last window
    pulled back to overlap more than *overlap* frames still blends correctly.
    """
    first = parts[0]
    total = torch.zeros(*first.shape[:2], latent_frames, *first.shape[3:], dtype=torch.float32)
    weight = torch.zeros(1, 1, latent_frames, 1, 1)
    ramp = torch.linspace(0, 1, overlap + 2)[1:-1]
    for index, (part, (start, end)) in enumerate(zip(parts, windows, strict=True)):
        frame_weight = torch.ones(end - start)
        if index > 0 and overlap:
            frame_weight[:overlap] = ramp
        if index < len(parts) - 1 and overlap:
            frame_weight[-overlap:] = ramp.flip(0)
        frame_weight = frame_weight.view(1, 1, -1, 1, 1)
        total[:, :, start:end] += part.float() * frame_weight
        weight[:, :, start:end] += frame_weight
    return total / weight
