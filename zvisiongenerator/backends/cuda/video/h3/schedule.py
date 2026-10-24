"""Sigma schedules and per-row timesteps of the MiniMax H3 sampler.

One transformer forward serves every modality: the generated video and audio rows step down their own schedules while
the conditioning rows (keyframes, an extend tail) stay pinned at their noise-augmentation level.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from diffusers import MiniMaxH3Scheduler

    from zvisiongenerator.utils.h3_preset import H3Schedule

__all__ = ["AUDIO_COND_T", "KEYFRAME_T", "ROW_SETS", "build_schedulers", "row_timesteps"]

KEYFRAME_T = 0.999  # timestep of the video conditioning rows: a nearly clean keyframe or tail
AUDIO_COND_T = 1.0  # timestep of the audio conditioning rows: fully clean
ROW_SETS = ("plain", "keyframe", "extend")  # which conditioning rows a clip has: none, a keyframe, or a video and audio tail


def _set_schedule(scheduler: MiniMaxH3Scheduler, shift: float, num_inference_steps: int) -> None:
    """Set *scheduler*'s shift, then its timesteps (the shift must be set first: it applies to the next ``set_timesteps``)."""
    scheduler.set_shift(shift)
    scheduler.set_timesteps(num_inference_steps)


def build_schedulers(schedule: H3Schedule, config_dir: Path | None = None) -> tuple[MiniMaxH3Scheduler, MiniMaxH3Scheduler]:
    """Return the ``(video, audio)`` schedulers set up for *schedule*.

    With *config_dir* (the H3 snapshot) the schedulers load the ``scheduler/`` and ``audio_scheduler/`` configs; the
    explicit shifts then override the configured ones, so the result does not depend on them.

    Raises:
        ValueError: If either scheduler does not give ``schedule.forwards`` timesteps (for instance when two sigmas
            collapse), since the loop pairs the two and would silently truncate.
    """
    from diffusers import MiniMaxH3Scheduler

    if config_dir is not None:
        video = MiniMaxH3Scheduler.from_pretrained(str(Path(config_dir) / "scheduler"))
        audio = MiniMaxH3Scheduler.from_pretrained(str(Path(config_dir) / "audio_scheduler"))
    else:
        video, audio = MiniMaxH3Scheduler(shift=schedule.video_shift), MiniMaxH3Scheduler(shift=schedule.audio_shift)
    _set_schedule(video, schedule.video_shift, schedule.num_inference_steps)
    _set_schedule(audio, schedule.audio_shift, schedule.num_inference_steps)
    for name, scheduler in (("video", video), ("audio", audio)):
        if len(scheduler.timesteps) != schedule.forwards:
            raise ValueError(f"The {name} schedule gives {len(scheduler.timesteps)} forwards, expected {schedule.forwards}.")
    return video, audio


def row_timesteps(
    video_idx: torch.Tensor,
    audio_idx: torch.Tensor,
    text_tokens: int,
    t: float,
    audio_t: float,
    *,
    keyframe_t: float = KEYFRAME_T,
    audio_cond_t: float = AUDIO_COND_T,
    condition_video_rows: int = 0,
    condition_audio_rows: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(distinct timesteps, per-row index)`` of one forward, for the packed sequence of the given row indices.

    *video_idx* and *audio_idx* are the sequence positions of the video and audio rows, conditioning rows first (the
    first *condition_video_rows* and *condition_audio_rows* of them). Those rows get *keyframe_t* and *audio_cond_t*;
    the other video and audio rows get *t* and *audio_t*; text rows take *t*.
    """
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep

    return MiniMaxH3SetTimestepsStep.build_row_timesteps(
        video_idx, audio_idx, condition_video_rows, condition_audio_rows, text_tokens, float(t), float(audio_t), float(keyframe_t), float(audio_cond_t)
    )
