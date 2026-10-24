"""The MiniMax H3 denoise loop: one transformer forward per step over a packed sequence with pinned conditioning rows.

The sequence is ``[text | keyframe rows | audio | video]``. Keyframes sit in front of the video block; an Extend tail is
the first latent frames of the video block and the first latents of *each* audio channel (audio rows are channel-major,
so the held audio rows are two separate runs). Held rows are mixed with noise once, up front, and then never written:
each step applies the schedulers to the *generated* rows only, selected by a boolean mask.

One forward serves every row, each at its own timestep: generated video and audio rows step down their schedules, the
conditioning rows stay at :data:`~zvisiongenerator.backends.cuda.video.h3.schedule.KEYFRAME_T` (video) and
:data:`~zvisiongenerator.backends.cuda.video.h3.schedule.AUDIO_COND_T` (audio). The loop mirrors diffusers'
``MiniMaxH3LoopDenoiser`` and ``MiniMaxH3LoopSchedulerStep``; the plain and keyframe paths are bit-equal to them.

The transformer is called as a module (``transformer(...)``, never ``transformer.forward``), so forward hooks installed
on it, such as the AdaLN fold's, fire on every step.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from zvisiongenerator.backends.cuda.video.h3.conditioning import (
    TAIL_AUDIO_LATENTS,
    prenoise,
    tail_audio_rows,
    tail_first_audio_order,
)
from zvisiongenerator.backends.cuda.video.h3.schedule import AUDIO_COND_T, KEYFRAME_T, row_timesteps
from zvisiongenerator.backends.cuda.video.h3.text_encoder import TEXT_TAG, VIDEO_TAG
from zvisiongenerator.utils.h3_memory import H3_AUDIO_CHANNELS

if TYPE_CHECKING:
    import torch
    from diffusers import MiniMaxH3Scheduler

    from zvisiongenerator.backends.cuda.video.h3.frames import Geometry

__all__ = ["Conditioning", "Layout", "build_layout", "denoise", "draw_noise", "refine"]

_AUDIO_TAG = 2  # modality tag of an audio row of the packed sequence
_PATCH = (1, 2, 2)  # the transformer's (t, h, w) patch
_VIDEO_CHANNELS = 24
_AUDIO_CHANNELS = 32  # latent channels of one audio row


@dataclass(frozen=True)
class Conditioning:
    """The conditioning media of one clip, already encoded; every field is optional.

    Attributes:
        keyframes: Clean keyframe latents, each ``(1, 24, 1, latent_height, latent_width)``, in packed order.
        anchors: One of ``"first"`` / ``"last"`` per keyframe: which end of the clip it is anchored to.
        tail_latents: Clean latents of an Extend tail ``(1, 24, T, latent_height, latent_width)``: the first *T*
            latent frames of the window.
        tail_audio: Clean tail audio rows ``(2 * 37, 32)``, channel-major: channel 0's 37 latents, then channel 1's.
    """

    keyframes: tuple[torch.Tensor, ...] = ()
    anchors: tuple[str, ...] = ()
    tail_latents: torch.Tensor | None = None
    tail_audio: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if len(self.keyframes) != len(self.anchors):
            raise ValueError(f"Each keyframe needs one anchor, got {len(self.keyframes)} keyframes and {len(self.anchors)} anchors.")
        if self.tail_audio is not None and tuple(self.tail_audio.shape) != (H3_AUDIO_CHANNELS * TAIL_AUDIO_LATENTS, _AUDIO_CHANNELS):
            raise ValueError(f"The tail audio must be {(H3_AUDIO_CHANNELS * TAIL_AUDIO_LATENTS, _AUDIO_CHANNELS)} rows, got {tuple(self.tail_audio.shape)}.")


@dataclass(frozen=True)
class Layout:
    """The packed sequence of one clip: what the transformer reads by name, and which rows are generated.

    Attributes:
        kwargs: The structural inputs of the transformer (``position_ids``, ``token_tags``, ``video_indices``,
            ``audio_indices``, ``text_indices``), on the device.
        timestep_video_idx: Sequence positions of the video rows, conditioning rows first (CPU).
        timestep_audio_idx: Sequence positions of the audio rows, held rows first: channel 0's tail, channel 1's tail,
            then the rest (CPU). Equal to ``audio_indices`` when nothing is held.
        text_rows: Number of text rows.
        keyframe_rows: Number of leading video rows that hold keyframes.
        tail_rows: Number of video rows, right after the keyframes, that hold the Extend tail.
        condition_video_rows: ``keyframe_rows + tail_rows``: the leading video rows that stay pinned.
        condition_audio_rows: Number of held audio rows (two tails, or 0).
        held_audio: Positions of the held rows inside the audio block, ascending (CPU).
        video_generated: Boolean mask of the video rows the schedulers step (device).
        audio_generated: Boolean mask of the audio rows the schedulers step (device).
    """

    kwargs: dict[str, torch.Tensor] = field(repr=False)
    timestep_video_idx: torch.Tensor = field(repr=False)
    timestep_audio_idx: torch.Tensor = field(repr=False)
    text_rows: int
    keyframe_rows: int
    tail_rows: int
    condition_video_rows: int
    condition_audio_rows: int
    held_audio: torch.Tensor = field(repr=False)
    video_generated: torch.Tensor = field(repr=False)
    audio_generated: torch.Tensor = field(repr=False)


def _check_conditioning(geo: Geometry, conditioning: Conditioning) -> int:
    """Raise ``ValueError`` unless the conditioning fits *geo*; return the number of tail latent frames (0 without a tail)."""
    cell = (geo.latent_height, geo.latent_width)
    for index, keyframe in enumerate(conditioning.keyframes):
        if tuple(keyframe.shape) != (1, _VIDEO_CHANNELS, 1, *cell):
            raise ValueError(f"Keyframe {index} must be shaped (1, {_VIDEO_CHANNELS}, 1, {cell[0]}, {cell[1]}) for this canvas, got {tuple(keyframe.shape)}.")
    tail = conditioning.tail_latents
    if tail is None:
        return 0
    if tail.ndim != 5 or tuple(tail.shape[:2]) != (1, _VIDEO_CHANNELS) or tuple(tail.shape[3:]) != cell:
        raise ValueError(f"The tail must be shaped (1, {_VIDEO_CHANNELS}, T, {cell[0]}, {cell[1]}) for this canvas, got {tuple(tail.shape)}.")
    if not 1 <= tail.shape[2] < geo.latent_frames:
        raise ValueError(f"The tail ({tail.shape[2]} latent frames) must leave frames to generate in a window of {geo.latent_frames}.")
    return int(tail.shape[2])


def build_layout(text_token_tags: Sequence[int] | torch.Tensor, geo: Geometry, device: Any, conditioning: Conditioning | None = None) -> Layout:
    """Return the packed layout of a clip with the rows of *conditioning* held.

    The structure comes from diffusers' ``build_packed_sequence`` called with the keyframe anchors. An Extend tail needs
    no extra rows: it is the leading part of the video block, so with the keyframes in front it makes the leading
    video rows the conditioning ones. The audio tail is two runs of the channel-major block, so the timestep order
    puts them first explicitly.

    Args:
        text_token_tags: The modality tag of every text row (``TEXT_TAG``, or ``VIDEO_TAG`` inside a keyframe's vision block).
        geo: The clip geometry of the window (for Extend, the whole window including the tail).
        device: Where the structural tensors and masks live.
        conditioning: The encoded media to hold, or ``None`` for a plain clip.

    Raises:
        ValueError: If the conditioning does not fit the geometry.
    """
    import torch
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep

    conditioning = conditioning or Conditioning()
    tail_frames = _check_conditioning(geo, conditioning)
    tags = torch.as_tensor(list(text_token_tags) if not isinstance(text_token_tags, torch.Tensor) else text_token_tags, dtype=torch.long)
    position_ids, token_tags, video_idx, audio_idx, text_idx, keyframe_rows, _ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        tags,
        geo.latent_frames,
        geo.latent_height,
        geo.latent_width,
        geo.audio_latents,
        _PATCH,
        H3_AUDIO_CHANNELS,
        _AUDIO_TAG,
        VIDEO_TAG,
        tuple(conditioning.anchors),
    )
    tail_rows = tail_frames * (geo.latent_height // _PATCH[1]) * (geo.latent_width // _PATCH[2])
    condition_audio_rows = 0
    held_audio = torch.empty(0, dtype=torch.long)
    timestep_audio_idx = audio_idx
    if conditioning.tail_audio is not None:
        held_audio = tail_audio_rows(geo.audio_latents, TAIL_AUDIO_LATENTS)
        timestep_audio_idx = audio_idx[tail_first_audio_order(geo.audio_latents, TAIL_AUDIO_LATENTS)]
        condition_audio_rows = int(held_audio.numel())
    condition_video_rows = keyframe_rows + tail_rows
    video_generated = torch.arange(video_idx.numel()) >= condition_video_rows
    audio_generated = torch.ones(audio_idx.numel(), dtype=torch.bool)
    audio_generated[held_audio] = False
    return Layout(
        kwargs={
            "position_ids": position_ids.to(device),
            "token_tags": token_tags.to(device),
            "video_indices": video_idx.to(device),
            "audio_indices": audio_idx.to(device),
            "text_indices": text_idx.to(device),
        },
        timestep_video_idx=video_idx,
        timestep_audio_idx=timestep_audio_idx,
        text_rows=int(text_idx.numel()),
        keyframe_rows=keyframe_rows,
        tail_rows=tail_rows,
        condition_video_rows=condition_video_rows,
        condition_audio_rows=condition_audio_rows,
        held_audio=held_audio,
        video_generated=video_generated.to(device),
        audio_generated=audio_generated.to(device),
    )


def _patchify(latents: torch.Tensor) -> torch.Tensor:
    """Return float32 latents ``(1, C, T, h, w)`` packed into transformer rows, frame-major then row-major."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents

    return patchify_video_latents(latents.float(), _PATCH)


def draw_noise(geo: Geometry, generator: torch.Generator | None, device: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw the noise of a whole window: the video as a latent tensor first, then the audio directly in row layout.

    The order is part of what a seed reproduces, as in diffusers' ``MiniMaxH3PrepareLatentsStep``.

    Args:
        geo: The clip geometry of the window.
        generator: The request's generator, or ``None`` for the global one.
        device: Where the rows are placed (a CPU generator draws on the CPU and moves the result).

    Returns:
        ``(video rows (video_rows, 96), audio rows (2 * audio_latents, 32))`` in float32.
    """
    import torch
    from diffusers.utils.torch_utils import randn_tensor

    video = randn_tensor((1, _VIDEO_CHANNELS, geo.latent_frames, geo.latent_height, geo.latent_width), generator=generator, device=device, dtype=torch.float32)
    audio = randn_tensor((geo.audio_rows, _AUDIO_CHANNELS), generator=generator, device=device, dtype=torch.float32)
    return _patchify(video), audio


def _initial_rows(geo: Geometry, layout: Layout, conditioning: Conditioning, scheduler: MiniMaxH3Scheduler, generator: torch.Generator | None, device: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the packed ``(video, audio)`` rows at step 0: held rows pre-noised once, generated rows pure noise.

    Draw order, as the reference: one conditioning draw per keyframe, then the tail video, then the tail audio, then
    the window noise (video, audio). Audio is drawn even at level 1.0, so a draw never depends on the level.
    """
    import torch

    keyframes = [_patchify(prenoise(scheduler, latents.float(), KEYFRAME_T, generator)) for latents in conditioning.keyframes]
    tail_video = _patchify(prenoise(scheduler, conditioning.tail_latents.float(), KEYFRAME_T, generator)) if conditioning.tail_latents is not None else None
    tail_audio = prenoise(scheduler, conditioning.tail_audio.float(), AUDIO_COND_T, generator) if conditioning.tail_audio is not None else None
    video, audio = draw_noise(geo, generator, device)
    if tail_video is not None:
        video[: layout.tail_rows] = tail_video.to(device)
    if tail_audio is not None:
        audio[layout.held_audio.to(device)] = tail_audio.to(device)
    return torch.cat([*(rows.to(device) for rows in keyframes), video]), audio


def _step_generated(scheduler: MiniMaxH3Scheduler, velocity: torch.Tensor, timestep: torch.Tensor, rows: torch.Tensor, generated: torch.Tensor) -> None:
    """Step the *generated* rows of *rows* in place with the scheduler; the other rows are not read or written."""
    rows[generated] = scheduler.step(velocity[generated].float(), timestep, rows[generated], return_dict=False)[0]


def _unpatchify(rows: torch.Tensor, geo: Geometry) -> torch.Tensor:
    """Return video *rows* as latents ``(1, 24, frames, latent_height, latent_width)``: the inverse of the patchify."""
    patch_t, patch_h, patch_w = _PATCH
    rows = rows.reshape(-1, geo.latent_frames // patch_t, geo.latent_height // patch_h, geo.latent_width // patch_w, _VIDEO_CHANNELS, patch_t, patch_h, patch_w)
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(-1, _VIDEO_CHANNELS, geo.latent_frames, geo.latent_height, geo.latent_width).contiguous()


def _run_steps(
    transformer: Any,
    embeds: torch.Tensor,
    layout: Layout,
    schedulers: tuple[MiniMaxH3Scheduler, MiniMaxH3Scheduler],
    video: torch.Tensor,
    audio: torch.Tensor,
    start: int,
    step_callback: Callable[[dict[str, Any]], None] | None,
    phase: str,
) -> None:
    """Run the steps ``start`` to the end of the schedules over the packed rows, stepping the generated rows of *video* and *audio* in place.

    One forward serves every row. The callback reports ``current_step = step - start + 1`` of ``total_steps = len - start``.
    The schedulers are stepped, so the caller passes copies. *embeds* is already on the rows' device.
    """
    import torch

    video_scheduler, audio_scheduler = schedulers
    if len(video_scheduler.timesteps) != len(audio_scheduler.timesteps):
        raise ValueError(f"The video and audio schedules must have the same length, got {len(video_scheduler.timesteps)} and {len(audio_scheduler.timesteps)}.")
    device = video.device
    total = len(video_scheduler.timesteps) - start
    with torch.no_grad():
        for step in range(start, len(video_scheduler.timesteps)):
            t, audio_t = video_scheduler.timesteps[step], audio_scheduler.timesteps[step]
            timestep, timestep_indices = row_timesteps(
                layout.timestep_video_idx,
                layout.timestep_audio_idx,
                layout.text_rows,
                float(t),
                float(audio_t),
                condition_video_rows=layout.condition_video_rows,
                condition_audio_rows=layout.condition_audio_rows,
            )
            video_velocity, audio_velocity = transformer(
                hidden_states=video[None],
                audio_hidden_states=audio[None],
                encoder_hidden_states=embeds,
                timestep=timestep.to(device),
                timestep_indices=timestep_indices.to(device),
                return_dict=False,
                **layout.kwargs,
            )
            _step_generated(video_scheduler, video_velocity[0], t, video, layout.video_generated)
            _step_generated(audio_scheduler, audio_velocity[0], audio_t, audio, layout.audio_generated)
            if step_callback is not None:
                step_callback({"phase": phase, "current_step": step - start + 1, "total_steps": total})


def denoise(
    transformer: Any,
    embeds: torch.Tensor,
    geo: Geometry,
    schedulers: tuple[MiniMaxH3Scheduler, MiniMaxH3Scheduler],
    seed: int | None,
    device: Any,
    *,
    conditioning: Conditioning | None = None,
    step_callback: Callable[[dict[str, Any]], None] | None = None,
    phase: str = "video",
    text_token_tags: Sequence[int] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full schedule over the packed sequence and return the final video and audio latents.

    The schedulers are copied first, so the caller's stay reusable. The conditioning rows are never written. For
    Extend the result is the whole window: its first latent frames and the first latents of each audio channel are the
    held tail (pre-noised), which the caller drops. Keyframes are not part of the result.

    Args:
        transformer: The H3 transformer; called as a module so its hooks fire.
        embeds: Text conditioning ``(1, rows, hidden)``.
        geo: The clip geometry of the window.
        schedulers: The ``(video, audio)`` schedulers from ``build_schedulers``.
        seed: Seed of the noise (a CPU generator), or ``None`` for unseeded.
        device: Where the sequence lives.
        conditioning: The encoded keyframes and Extend tail to hold, or ``None``.
        step_callback: Called after every step with ``{"phase", "current_step", "total_steps"}``.
        phase: The ``phase`` the callback reports.
        text_token_tags: The modality tag of every row of *embeds* (``Presentation.token_tags``); all text when ``None``.

    Returns:
        ``(video latents (1, 24, frames, latent_height, latent_width), audio latents (2, 32, audio_latents))``, float32.

    Raises:
        ValueError: If the conditioning does not fit the geometry, or the schedulers differ in length.
        Exception: Anything the transformer or *step_callback* raises (cancellation, out of memory) propagates.
    """
    import torch

    video_scheduler, audio_scheduler = (copy.deepcopy(scheduler) for scheduler in schedulers)
    conditioning = conditioning or Conditioning()
    if text_token_tags is None:
        text_token_tags = [TEXT_TAG] * embeds.shape[1]
    layout = build_layout(text_token_tags, geo, device, conditioning)
    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    video, audio = _initial_rows(geo, layout, conditioning, video_scheduler, generator, device)
    embeds = embeds.to(device)
    _run_steps(transformer, embeds, layout, (video_scheduler, audio_scheduler), video, audio, 0, step_callback, phase)
    audio_latents = audio.reshape(H3_AUDIO_CHANNELS, geo.audio_latents, _AUDIO_CHANNELS).permute(0, 2, 1).contiguous()
    return _unpatchify(video[layout.keyframe_rows :], geo), audio_latents


def refine(
    transformer: Any,
    embeds: torch.Tensor,
    geo: Geometry,
    schedulers: tuple[MiniMaxH3Scheduler, MiniMaxH3Scheduler],
    latents: torch.Tensor,
    audio_rows: torch.Tensor,
    noise: tuple[torch.Tensor, torch.Tensor],
    *,
    start: int,
    device: Any,
    conditioning: Conditioning | None = None,
    keyframe_seed: int | None = None,
    step_callback: Callable[[dict[str, Any]], None] | None = None,
    phase: str = "video",
    text_token_tags: Sequence[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Re-noise clean latents to step *start* of the schedules, run the steps from there to the end and return the refined video latents.

    The video latents and the audio rows are mixed with the given noise at ``timesteps[start]`` (``scale_noise``), the
    keyframes are held at :data:`~zvisiongenerator.backends.cuda.video.h3.schedule.KEYFRAME_T`, and the loop is the one
    :func:`denoise` runs. The audio is stepped with the video, because the transformer reads it, and then discarded.
    The schedulers are copied first, so the caller's stay reusable.

    Args:
        transformer: The H3 transformer; called as a module so its hooks fire.
        embeds: Text conditioning ``(1, rows, hidden)``.
        geo: The clip geometry of the window the latents cover.
        schedulers: The ``(video, audio)`` schedulers from ``build_schedulers``.
        latents: Clean video latents ``(1, 24, geo.latent_frames, latent_height, latent_width)``.
        audio_rows: Clean audio rows ``(geo.audio_rows, 32)``, channel-major.
        noise: ``(video noise shaped like latents, audio noise shaped like audio_rows)``.
        start: The schedule step to re-noise to; the loop runs ``start`` to the end.
        device: Where the sequence lives.
        conditioning: The keyframes to hold, or ``None``. An Extend tail is refused.
        keyframe_seed: Seed of the keyframes' pre-noise, or ``None`` for the global generator.
        step_callback: Called after every step with ``{"phase", "current_step", "total_steps"}``, counted from *start*.
        phase: The ``phase`` the callback reports.
        text_token_tags: The modality tag of every row of *embeds*; all text when ``None``.

    Returns:
        The refined video latents ``(1, 24, geo.latent_frames, latent_height, latent_width)``, float32 (keyframes excluded).

    Raises:
        ValueError: If *start* is outside the schedule, the latents, audio rows or noise have the wrong shape, the
            schedulers differ in length, or the conditioning holds an Extend tail or does not fit the geometry.
        Exception: Anything the transformer or *step_callback* raises (cancellation, out of memory) propagates.
    """
    import torch

    video_scheduler, audio_scheduler = (copy.deepcopy(scheduler) for scheduler in schedulers)
    conditioning = conditioning or Conditioning()
    if conditioning.tail_latents is not None or conditioning.tail_audio is not None:
        raise ValueError("A refine cannot hold an Extend tail.")
    if not 0 <= start < len(video_scheduler.timesteps):
        raise ValueError(f"start must be within the {len(video_scheduler.timesteps)} steps of the schedule, got {start}.")
    video_shape = (1, _VIDEO_CHANNELS, geo.latent_frames, geo.latent_height, geo.latent_width)
    audio_shape = (geo.audio_rows, _AUDIO_CHANNELS)
    if tuple(latents.shape) != video_shape:
        raise ValueError(f"The latents must be shaped {video_shape} for this window, got {tuple(latents.shape)}.")
    if tuple(audio_rows.shape) != audio_shape:
        raise ValueError(f"The audio rows must be shaped {audio_shape} for this window, got {tuple(audio_rows.shape)}.")
    noise_video, noise_audio = noise
    if tuple(noise_video.shape) != video_shape or tuple(noise_audio.shape) != audio_shape:
        raise ValueError(f"The noise must be shaped {video_shape} and {audio_shape}, got {tuple(noise_video.shape)} and {tuple(noise_audio.shape)}.")
    if text_token_tags is None:
        text_token_tags = [TEXT_TAG] * embeds.shape[1]
    layout = build_layout(text_token_tags, geo, device, conditioning)
    generator = torch.Generator(device="cpu").manual_seed(keyframe_seed) if keyframe_seed is not None else None
    keyframes = [_patchify(prenoise(video_scheduler, latent.float(), KEYFRAME_T, generator)) for latent in conditioning.keyframes]
    renoised_video = video_scheduler.scale_noise(latents.float(), float(video_scheduler.timesteps[start]), noise_video.float())
    renoised_audio = audio_scheduler.scale_noise(audio_rows.float(), float(audio_scheduler.timesteps[start]), noise_audio.float())
    video = torch.cat([*(rows.to(device) for rows in keyframes), _patchify(renoised_video).to(device)])
    audio = renoised_audio.to(device)
    _run_steps(transformer, embeds.to(device), layout, (video_scheduler, audio_scheduler), video, audio, start, step_callback, phase)
    return _unpatchify(video[layout.keyframe_rows :], geo)
