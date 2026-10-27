"""Tests for the H3 sigma schedules and row timesteps, with contracts pinned to diffusers 0.41.0."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from zvisiongenerator.backends.cuda.video.h3.frames import geometry  # noqa: E402
from zvisiongenerator.backends.cuda.video.h3.schedule import (  # noqa: E402
    AUDIO_COND_T,
    KEYFRAME_T,
    ROW_SETS,
    build_schedulers,
    row_timesteps,
)
from zvisiongenerator.utils.h3_memory import h3_audio_latents, h3_latent_frames  # noqa: E402
from zvisiongenerator.utils.h3_preset import H3Schedule  # noqa: E402

_VIDEO_8 = [0.0, 0.023255825, 0.052631557, 0.090909064, 0.142857134, 0.217391312, 0.333333313, 0.538461566]
_AUDIO_8 = [0.0, 0.045454562, 0.100000024, 0.166666687, 0.25, 0.357142866, 0.5, 0.699999988]
_VIDEO_4 = [0.0, 0.052631557, 0.142857134, 0.333333313]
_AUDIO_4 = [0.0, 0.100000024, 0.25, 0.5]


def _values(timesteps) -> list[float]:
    return [float(value) for value in timesteps]


def test_diffusers_is_the_pinned_release():
    assert diffusers.__version__ == "0.41.0"


def test_constants():
    assert (KEYFRAME_T, AUDIO_COND_T) == (0.999, 1.0)
    assert ROW_SETS == ("plain", "keyframe", "extend")


@pytest.mark.parametrize(
    ("forwards", "video", "audio"),
    [(8, _VIDEO_8, _AUDIO_8), (4, _VIDEO_4, _AUDIO_4)],
)
def test_timesteps_literals(forwards, video, audio):
    video_scheduler, audio_scheduler = build_schedulers(H3Schedule(forwards, 6.0, 3.0))
    assert _values(video_scheduler.timesteps) == pytest.approx(video, abs=1e-6)
    assert _values(audio_scheduler.timesteps) == pytest.approx(audio, abs=1e-6)
    assert len(video_scheduler.timesteps) == len(audio_scheduler.timesteps) == forwards


def test_config_dir_loads_both_configs_and_the_explicit_shift_wins(tmp_path):
    # The saved configs carry the released shifts (12 video, 3 audio); the schedule's shifts must override them.
    diffusers.MiniMaxH3Scheduler(shift=12.0).save_pretrained(tmp_path / "scheduler")
    diffusers.MiniMaxH3Scheduler(shift=12.0).save_pretrained(tmp_path / "audio_scheduler")
    video, audio = build_schedulers(H3Schedule(8, 6.0, 3.0), tmp_path)
    assert _values(video.timesteps) == pytest.approx(_VIDEO_8, abs=1e-6)
    assert _values(audio.timesteps) == pytest.approx(_AUDIO_8, abs=1e-6)


def test_config_dir_is_read_from_the_two_subfolders(tmp_path):
    with pytest.raises(OSError):
        build_schedulers(H3Schedule(8, 6.0, 3.0), tmp_path)


def test_a_schedule_that_collapses_sigmas_is_refused():
    # A shift this large collapses neighbouring float32 sigmas into fewer forwards than the schedule states.
    with pytest.raises(ValueError, match="expected 400"):
        build_schedulers(H3Schedule(400, 1e9, 1e9))


def _idx(start: int, count: int):
    return torch.arange(start, start + count)


def test_row_timesteps_plain():
    unique, inverse = row_timesteps(_idx(2, 2), _idx(4, 2), 2, 0.25, 0.4)
    assert unique.tolist() == pytest.approx([0.25, 0.4])
    assert inverse.tolist() == [0, 0, 0, 0, 1, 1]


def test_row_timesteps_keyframe_row_is_pinned():
    unique, inverse = row_timesteps(_idx(2, 4), _idx(6, 4), 2, 0.25, 0.4, condition_video_rows=1)
    assert unique.tolist() == pytest.approx([0.25, 0.4, KEYFRAME_T])
    assert inverse.tolist() == [0, 0, 2, 0, 0, 0, 1, 1, 1, 1]


def test_row_timesteps_extend_tail_pins_video_and_audio():
    unique, inverse = row_timesteps(_idx(2, 4), _idx(6, 4), 2, 0.25, 0.4, condition_video_rows=2, condition_audio_rows=1)
    assert unique.tolist() == pytest.approx([0.25, 0.4, KEYFRAME_T, AUDIO_COND_T])
    assert inverse.tolist() == [0, 0, 2, 2, 0, 0, 3, 1, 1, 1]


def test_row_timesteps_conditioning_levels_can_be_overridden():
    unique, _ = row_timesteps(_idx(0, 2), _idx(2, 2), 0, 0.25, 0.4, keyframe_t=0.9, audio_cond_t=0.8, condition_video_rows=1, condition_audio_rows=1)
    assert unique.tolist() == pytest.approx([0.25, 0.4, 0.8, 0.9])


def test_row_timesteps_accepts_tensor_scalars():
    t = build_schedulers(H3Schedule(8, 6.0, 3.0))[0].timesteps[3]
    unique, _ = row_timesteps(_idx(1, 2), _idx(3, 2), 1, t, t)
    assert unique.dtype == torch.float32
    assert unique.tolist() == pytest.approx([float(t)])


# --- contracts against diffusers 0.41.0 ---------------------------------------------------------------------------


@pytest.mark.parametrize("frames", [22, 124, 413])
def test_arithmetic_equals_the_diffusers_functions(frames):
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import audio_latent_num_frames, video_latent_num_frames

    assert h3_latent_frames(frames) == video_latent_num_frames(frames, 17, 5)
    assert h3_audio_latents(frames) == audio_latent_num_frames(frames)
    assert geometry(64, 32, frames).latent_frames == video_latent_num_frames(frames, 17, 5)


@pytest.mark.parametrize("frames", [21, 125])
def test_frame_refusal_matches_diffusers(frames):
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import video_latent_num_frames

    with pytest.raises(ValueError):
        video_latent_num_frames(frames, 17, 5)
    with pytest.raises(ValueError):
        h3_latent_frames(frames)


def test_diffusers_constants_match_ours():
    from diffusers.modular_pipelines.minimax_h3 import modular_pipeline as mp
    from zvisiongenerator.utils import h3_memory

    assert mp.MINIMAX_H3_FPS == h3_memory.H3_FPS
    assert mp.MINIMAX_H3_AUDIO_LATENTS_PER_SECOND == h3_memory.H3_AUDIO_LATENTS_PER_SECOND
    assert mp.MINIMAX_H3_AUDIO_CHANNELS == h3_memory.H3_AUDIO_CHANNELS


def test_build_row_timesteps_signature_and_semantics_are_unchanged():
    import inspect

    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep

    names = list(inspect.signature(MiniMaxH3SetTimestepsStep.build_row_timesteps).parameters)
    assert names == [
        "video_indices",
        "audio_indices",
        "num_condition_video_rows",
        "num_condition_audio_rows",
        "num_text_tokens",
        "video_timestep",
        "audio_timestep",
        "condition_video_timestep",
        "condition_audio_timestep",
    ]
    # Text rows inherit the video timestep and the conditioning rows lead each index list.
    unique, inverse = MiniMaxH3SetTimestepsStep.build_row_timesteps(_idx(1, 2), _idx(3, 2), 1, 1, 1, 0.25, 0.4, 0.999, 1.0)
    assert unique.tolist() == pytest.approx([0.25, 0.4, 0.999, 1.0])
    assert inverse.tolist() == [0, 2, 0, 3, 1]
