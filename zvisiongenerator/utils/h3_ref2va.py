"""Shared MiniMax H3 Ref2VA helpers: reference preparation, the task tag and the packed layout.

Both backends (Windows/Linux and macOS) prepare a reference, tag the prompt and lay out the packed sequence through
these functions, so the two platforms agree. Torch, PIL and diffusers load lazily inside the functions that need them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from PIL import Image

__all__ = ["Ref2vaSequence", "h3_reference_size", "load_h3_reference", "ref2va_sequence", "task_tag"]

# The short edge a reference is encoded at. The released checkpoint was trained at 2048; 1024 fits the 10 GB budget.
_REFERENCE_SHORT_EDGE = 1024
# Both sides of a reference are a multiple of this, the pixel grid of one transformer row.
_SIDE_MULTIPLE = 32

# The model's Ref2VA prompt guide names the task in a bracketed tag; the bracket takes an optional audio clause.
_TASK_TAGS = {
    "t2v": "reference generation",
    "i2v": "reference generation + keyframe completion",
    "extend": "video continuation + reference generation",
}
# Extend's held clip is the first video of the packed sequence; the sentence tells the model to continue it without a cut.
_EXTEND_CONTINUATION = "The target video continues <Video 1> as one continuous shot with no cut."


@dataclass(frozen=True)
class Ref2vaSequence:
    """Packed Ref2VA sequence: the transformer inputs plus how many leading video rows are held rather than generated.

    Attributes:
        position_ids: Rotary position of every row, shape ``(rows, 3)``.
        token_tags: Modality tag of every row.
        video_indices: Row index of every video row, reference and keyframe rows first, then the target.
        audio_indices: Row index of every target audio row.
        text_indices: Row index of every text row.
        keyframe_rows: Leading video rows that are held: the reference blocks, then the keyframe rows.
    """

    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    keyframe_rows: int


def h3_reference_size(width: int, height: int) -> tuple[int, int]:
    """Return the ``(width, height)`` a ``width x height`` picture has once prepared as a reference.

    The short edge becomes 1024 pixels and both sides are rounded to a multiple of 32.
    """
    scale = _REFERENCE_SHORT_EDGE / min(width, height)
    return round(width * scale / _SIDE_MULTIPLE) * _SIDE_MULTIPLE, round(height * scale / _SIDE_MULTIPLE) * _SIDE_MULTIPLE


def load_h3_reference(path: str | Path) -> Image.Image:
    """Return the reference image at *path*, upright, RGB and scaled to the Ref2VA reference size.

    The short edge becomes 1024 pixels and both sides are rounded to a multiple of 32, with LANCZOS.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    from PIL import Image

    from zvisiongenerator.utils.input_images import open_image_upright

    image = open_image_upright(path)
    target = h3_reference_size(*image.size)
    return image if image.size == target else image.resize(target, Image.Resampling.LANCZOS)


def task_tag(mode: str, audio_track: bool) -> str:
    """Return the Ref2VA task tag for the input *mode* the backend presents.

    The tag is bracketed; ``"extend"`` follows it with the sentence that continues ``<Video 1>`` in one shot.

    Args:
        mode: ``"t2v"``, ``"i2v"`` or ``"extend"``.
        audio_track: Whether a held audio track is reused, which adds `` + audio reuse`` inside the brackets.
    """
    tag = f"[{_TASK_TAGS[mode]}{' + audio reuse' if audio_track else ''}]"
    return f"{tag} {_EXTEND_CONTINUATION}" if mode == "extend" else tag


def ref2va_sequence(
    text_token_tags: Sequence[int] | torch.Tensor,
    reference_shapes: Sequence[tuple[int, int, int]],
    *,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    audio_latents: int,
    anchors: Sequence[str],
    patch: tuple[int, int, int],
    audio_channels: int,
    audio_tag: int,
    video_tag: int,
) -> Ref2vaSequence:
    """Build the packed layout ``[text | references | keyframes | target audio | target video]``.

    The reference part is diffusers' own Ref2VA layout. For each of *anchors*, one latent frame of rows is spliced in
    right after the reference blocks, in order, and the target and audio indices shift by that many rows. A ``"first"``
    keyframe takes the first target frame's rotary positions; a ``"last"`` one takes the positions of fl2va's ``"last"``
    anchor (the time of the last target latent frame), shifted to this layout's rotary origin.

    Args:
        text_token_tags: Modality tag of every text row.
        reference_shapes: ``(frames, height, width)`` of each reference's latent, in packed order.
        latent_frames: Target latent frames.
        latent_height: Target latent height.
        latent_width: Target latent width.
        audio_latents: Target audio latents per channel.
        anchors: One of ``"first"`` / ``"last"`` per held keyframe, in packed order; empty for none.
        patch: The transformer's ``(t, h, w)`` patch.
        audio_channels: Channels the soundtrack is packed channel-major over.
        audio_tag: Modality tag of an audio row.
        video_tag: Modality tag of a video row.
    """
    import torch
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3Ref2VAPrepareLayoutStep

    tags = torch.as_tensor(text_token_tags, dtype=torch.long)
    references = [MiniMaxH3ImageReference(image=None) for _ in reference_shapes]
    latents = [torch.empty(1, 1, *shape) for shape in reference_shapes]
    position_ids, token_tags, video_indices, audio_indices, text_indices, reference_rows, _ = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        tags, references, latents, [], latent_frames, latent_height, latent_width, audio_latents, patch, audio_channels, audio_tag, video_tag
    )
    keyframe_rows = reference_rows
    if anchors:
        from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep

        n_text = text_indices.numel()
        frame_rows = (latent_height // patch[1]) * (latent_width // patch[2])
        at = n_text + reference_rows  # right after the reference blocks, before the target audio
        first_target = int(video_indices[reference_rows])
        keyframe_positions = []
        for anchor in anchors:
            if anchor == "first":
                keyframe_positions.append(position_ids[first_target : first_target + frame_rows].clone())
            else:
                plain = MiniMaxH3PrepareLayoutStep.build_packed_sequence(tags, latent_frames, latent_height, latent_width, audio_latents, patch, audio_channels, audio_tag, video_tag, ("last",))[0]
                last_positions = plain[n_text : n_text + frame_rows].clone()
                last_positions[:, 0] += position_ids[first_target, 0] - n_text
                keyframe_positions.append(last_positions)
        total = frame_rows * len(anchors)
        position_ids = torch.cat([position_ids[:at], *keyframe_positions, position_ids[at:]])
        token_tags = torch.cat([token_tags[:at], torch.full((total,), video_tag, dtype=token_tags.dtype), token_tags[at:]])
        video_indices = torch.cat([video_indices[:reference_rows], torch.arange(at, at + total), video_indices[reference_rows:] + total])
        audio_indices = audio_indices + total
        keyframe_rows = reference_rows + total
    return Ref2vaSequence(position_ids, token_tags, video_indices, audio_indices, text_indices, keyframe_rows)
