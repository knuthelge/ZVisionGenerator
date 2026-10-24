"""Row and frame arithmetic of the MiniMax H3 transformer sequence.

H3 sees one packed sequence of text rows, audio rows and video rows. The counts follow from the clip size alone, so the
geometry of a canvas, the memory estimate and the sampler all share these pure functions. They mirror diffusers'
``video_latent_num_frames`` and ``audio_latent_num_frames`` (checked by a contract test against the pinned release).
"""

from __future__ import annotations

__all__ = [
    "H3_AUDIO_CHANNELS",
    "H3_AUDIO_LATENTS_PER_SECOND",
    "H3_FPS",
    "H3_FRAMES_PER_CHUNK",
    "H3_LATENTS_PER_CHUNK",
    "H3_PATCH",
    "H3_ROW_PIXELS",
    "H3_SPATIAL_RATIO",
    "H3_TEXT_TOKENS",
    "h3_audio_latents",
    "h3_latent_frames",
    "h3_sequence_rows",
    "h3_video_rows",
]

H3_FRAMES_PER_CHUNK = 17  # a clip is 17 * n + 5 frames
H3_LATENTS_PER_CHUNK = 5  # each further 17 frames add 5 latent frames; the leading 5 frames give 2
H3_FPS = 24
H3_AUDIO_LATENTS_PER_SECOND = 40
H3_SPATIAL_RATIO = 16  # pixels per latent cell
H3_PATCH = 2  # latent cells per row, along each spatial axis
H3_ROW_PIXELS = H3_SPATIAL_RATIO * H3_PATCH  # 32 pixels per row, along each spatial axis
H3_AUDIO_CHANNELS = 2  # the mono audio VAE codes each of the two channels on its own
H3_TEXT_TOKENS = 100  # approximation of the prompt length in rows


def h3_latent_frames(num_frames: int) -> int:
    """Return the latent frame count of a *num_frames* clip.

    Raises:
        ValueError: If *num_frames* is not of the form ``17 * n + 5``.
    """
    if num_frames < H3_LATENTS_PER_CHUNK or num_frames % H3_FRAMES_PER_CHUNK != H3_LATENTS_PER_CHUNK:
        raise ValueError(f"num_frames must be of the form {H3_FRAMES_PER_CHUNK} * n + {H3_LATENTS_PER_CHUNK}, got {num_frames}.")
    return (num_frames - H3_LATENTS_PER_CHUNK) // H3_FRAMES_PER_CHUNK * H3_LATENTS_PER_CHUNK + 2


def h3_audio_latents(num_frames: int) -> int:
    """Return the number of audio latents that cover a *num_frames* clip."""
    return round(num_frames / H3_FPS * H3_AUDIO_LATENTS_PER_SECOND)


def _cell_rows(width: int, height: int) -> int:
    """Return the rows of one latent frame of a ``width x height`` canvas.

    Raises:
        ValueError: If the canvas is not a positive multiple of 32 on both sides.
    """
    if width < 1 or height < 1 or width % H3_ROW_PIXELS or height % H3_ROW_PIXELS:
        raise ValueError(f"Canvas {width}x{height} must be a positive multiple of {H3_ROW_PIXELS}.")
    return (width // H3_ROW_PIXELS) * (height // H3_ROW_PIXELS)


def h3_video_rows(width: int, height: int, num_frames: int) -> int:
    """Return the video rows of a ``width x height x num_frames`` clip.

    Raises:
        ValueError: If the canvas is not a multiple of 32 or *num_frames* is not ``17 * n + 5``.
    """
    return h3_latent_frames(num_frames) * _cell_rows(width, height)


def h3_sequence_rows(width: int, height: int, num_frames: int, *, keyframes: int = 0, tail_frames: int = 0) -> int:
    """Return the length of the packed sequence: text, audio and video rows, plus the rows of conditioning keyframes.

    Each keyframe adds one latent frame of rows. An extend tail adds none: its frames and audio are the first frames
    of the *num_frames* window, so *tail_frames* is only checked, not counted.

    Raises:
        ValueError: If the canvas or frame count is invalid, *keyframes* is negative, or *tail_frames* is neither 0 nor
            a valid clip length within *num_frames*.
    """
    if keyframes < 0:
        raise ValueError(f"keyframes must not be negative, got {keyframes}.")
    if tail_frames:
        h3_latent_frames(tail_frames)
        if tail_frames > num_frames:
            raise ValueError(f"tail_frames ({tail_frames}) must not exceed num_frames ({num_frames}).")
    audio_rows = h3_audio_latents(num_frames) * H3_AUDIO_CHANNELS
    keyframe_rows = keyframes * _cell_rows(width, height)
    return H3_TEXT_TOKENS + audio_rows + h3_video_rows(width, height, num_frames) + keyframe_rows
