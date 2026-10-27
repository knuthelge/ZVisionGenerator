"""Tests for the H3 denoise loop, with contracts pinned to diffusers 0.41.0 and no real weights.

Most tests drive the loop with a stand-in transformer that records every forward and returns a velocity that depends on
the rows alone. The plain and keyframe paths are also compared bit for bit with diffusers' own loop blocks on a tiny
random-init transformer.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers.modular_pipelines.minimax_h3 import modular_pipeline as reference_pipeline  # noqa: E402
from diffusers.modular_pipelines.minimax_h3.before_denoise import (  # noqa: E402
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
    patchify_video_latents,
)
from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopDenoiser, MiniMaxH3LoopSchedulerStep  # noqa: E402
from diffusers.utils.torch_utils import randn_tensor  # noqa: E402

from tests._h3_helpers import tiny_transformer  # noqa: E402

from zvisiongenerator.backends.cuda.video.h3 import sampler as sampler_module  # noqa: E402
from zvisiongenerator.backends.cuda.video.h3.conditioning import (  # noqa: E402
    TAIL_AUDIO_LATENTS,
    TAIL_LATENT_FRAMES,
    prenoise,
    tail_audio_rows,
    tail_first_audio_order,
)
from zvisiongenerator.backends.cuda.video.h3.frames import geometry  # noqa: E402
from zvisiongenerator.backends.cuda.video.h3.sampler import (  # noqa: E402
    Conditioning,
    build_layout,
    denoise,
    draw_noise,
)
from zvisiongenerator.backends.cuda.video.h3.schedule import AUDIO_COND_T, KEYFRAME_T, build_schedulers  # noqa: E402
from zvisiongenerator.backends.cuda.video.h3.text_encoder import TEXT_TAG, VIDEO_TAG  # noqa: E402
from zvisiongenerator.utils.h3_preset import H3Schedule  # noqa: E402

_SCHEDULE_8 = H3Schedule(8, 6.0, 3.0)
_SCHEDULE_4 = H3Schedule(4, 6.0, 3.0)
_CPU = torch.device("cpu")
_PLAIN = geometry(64, 64, 22)  # 7 latent frames of 2 x 2 cells, 37 audio latents per channel
_WINDOW = geometry(64, 64, 39)  # 12 latent frames, 65 audio latents per channel: room for a 7-frame tail
_CELLS = (_PLAIN.latent_height // 2) * (_PLAIN.latent_width // 2)  # video rows per latent frame
_TEXT_TAGS = [TEXT_TAG] * 4
_TEXT_DIM = 48


# --- stand-ins -------------------------------------------------------------------------------------------------------


class _Recorder(torch.nn.Module):
    """A stand-in transformer: records each forward's inputs and returns a velocity that depends on the rows alone."""

    def __init__(self, fail_at: int | None = None) -> None:
        super().__init__()
        self.calls: list[dict] = []
        self.fail_at = fail_at

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
        attention_kwargs=None,
        return_dict=True,
    ):
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise RuntimeError("CUDA out of memory.")
        self.calls.append(
            {
                "video": hidden_states.clone(),
                "audio": audio_hidden_states.clone(),
                "text": encoder_hidden_states,
                "timestep": timestep,
                "timestep_indices": timestep_indices,
                "token_tags": token_tags,
                "position_ids": position_ids,
                "video_indices": video_indices,
                "audio_indices": audio_indices,
                "text_indices": text_indices,
                "return_dict": return_dict,
            }
        )
        return hidden_states * 0.25 + 0.5, audio_hidden_states * 0.25 - 0.5


def _embeds(rows: int = 4) -> torch.Tensor:
    return torch.randn(1, rows, _TEXT_DIM, generator=torch.Generator().manual_seed(7))


def _schedulers(schedule: H3Schedule = _SCHEDULE_8):
    return build_schedulers(schedule)


def _latents(frames: int, seed: int, geo=_PLAIN) -> torch.Tensor:
    return torch.randn(1, 24, frames, geo.latent_height, geo.latent_width, generator=torch.Generator().manual_seed(seed))


def _tail_audio(seed: int = 5) -> torch.Tensor:
    return torch.randn(2 * TAIL_AUDIO_LATENTS, 32, generator=torch.Generator().manual_seed(seed))


def _extend() -> Conditioning:
    return Conditioning(tail_latents=_latents(TAIL_LATENT_FRAMES, 3, _WINDOW), tail_audio=_tail_audio())


def _keyframes(*anchors: str) -> Conditioning:
    return Conditioning(keyframes=tuple(_latents(1, 10 + index) for index, _ in enumerate(anchors)), anchors=anchors)


def _run(geo=_PLAIN, schedule: H3Schedule = _SCHEDULE_8, *, conditioning: Conditioning | None = None, seed: int = 11, transformer=None, **kwargs):
    transformer = transformer or _Recorder()
    result = denoise(transformer, _embeds(), geo, _schedulers(schedule), seed, _CPU, conditioning=conditioning, text_token_tags=_TEXT_TAGS, **kwargs)
    return transformer, result


# --- the diffusers contracts ------------------------------------------------------------------------------------------


def test_build_packed_sequence_signature_is_pinned():
    # The sampler is a caller of this private static; a changed signature must fail here, not in a run.
    parameters = list(inspect.signature(MiniMaxH3PrepareLayoutStep.build_packed_sequence).parameters)
    assert parameters == [
        "text_token_tags",
        "num_latent_frames",
        "latent_height",
        "latent_width",
        "num_audio_latents",
        "patch_size",
        "audio_channels",
        "audio_tag",
        "video_tag",
        "keyframe_anchors",
    ]


@pytest.mark.parametrize("schedule", [_SCHEDULE_8, _SCHEDULE_4], ids=["8-step", "4-step"])
def test_the_reference_keyframe_level_is_the_constant_on_every_step(schedule):
    # The reference holds conditioning rows at max(t, 0.999); the sampler uses the constant. They agree because the video
    # timestep t = 1 - sigma only reaches 0.999 after the last step. A schedule that changes this must fail here.
    assert reference_pipeline.MiniMaxH3ModularPipeline.keyframe_noise_aug.fget(None) == KEYFRAME_T
    video_scheduler, _ = build_schedulers(schedule)
    assert [max(float(t), 0.999) for t in video_scheduler.timesteps] == [KEYFRAME_T] * schedule.forwards


def test_modality_tags_and_patch_match_diffusers():
    assert sampler_module._AUDIO_TAG == reference_pipeline.MINIMAX_H3_AUDIO_TAG
    assert VIDEO_TAG == reference_pipeline.MINIMAX_H3_VIDEO_TAG
    assert TEXT_TAG == reference_pipeline.MINIMAX_H3_TEXT_TAG
    assert sampler_module._PATCH == (1, 2, 2)


# --- the layout ------------------------------------------------------------------------------------------------------


def test_plain_layout_is_the_reference_layout():
    layout = build_layout(_TEXT_TAGS, _PLAIN, _CPU, None)
    expected = MiniMaxH3PrepareLayoutStep.build_packed_sequence(torch.tensor(_TEXT_TAGS), 7, 4, 4, 37, (1, 2, 2), 2, 2, 0, ())
    for name, value in zip(("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices"), expected[:5], strict=True):
        assert torch.equal(layout.kwargs[name], value), name
    assert (layout.keyframe_rows, layout.tail_rows, layout.condition_video_rows, layout.condition_audio_rows) == (0, 0, 0, 0)
    assert layout.video_generated.all() and layout.audio_generated.all()
    assert torch.equal(layout.timestep_audio_idx, expected[3])


@pytest.mark.parametrize("anchors", [("first",), ("last",), ("first", "last")], ids=["first", "last", "both"])
def test_keyframe_layout_matches_build_packed_sequence_with_the_keyframe_args(anchors):
    layout = build_layout(_TEXT_TAGS, _PLAIN, _CPU, _keyframes(*anchors))
    expected = MiniMaxH3PrepareLayoutStep.build_packed_sequence(torch.tensor(_TEXT_TAGS), 7, 4, 4, 37, (1, 2, 2), 2, 2, 0, anchors)
    for name, value in zip(("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices"), expected[:5], strict=True):
        assert torch.equal(layout.kwargs[name], value), name
    assert layout.keyframe_rows == layout.condition_video_rows == expected[5] == len(anchors) * _CELLS
    assert layout.condition_audio_rows == 0
    assert layout.video_generated.sum() == _PLAIN.video_rows
    assert not layout.video_generated[: layout.keyframe_rows].any()


def test_vision_rows_in_the_text_keep_their_video_tag():
    tags = [TEXT_TAG, VIDEO_TAG, VIDEO_TAG, TEXT_TAG]
    layout = build_layout(tags, _PLAIN, _CPU, _keyframes("first"))
    assert layout.kwargs["token_tags"][:4].tolist() == tags


def test_extend_layout_puts_the_tail_first_in_both_modalities():
    layout = build_layout(_TEXT_TAGS, _WINDOW, _CPU, _extend())
    audio_block = layout.kwargs["audio_indices"]
    assert layout.keyframe_rows == 0
    assert layout.tail_rows == layout.condition_video_rows == TAIL_LATENT_FRAMES * _CELLS
    assert layout.condition_audio_rows == 2 * TAIL_AUDIO_LATENTS == 74
    # The video tail is the leading part of the video block, already first in the video indices.
    assert torch.equal(layout.timestep_video_idx, layout.kwargs["video_indices"])
    assert layout.video_generated.sum() == (_WINDOW.latent_frames - TAIL_LATENT_FRAMES) * _CELLS
    # Audio: channel 0's 37 tail rows and channel 1's, then the rest ascending.
    assert torch.equal(layout.timestep_audio_idx, audio_block[tail_first_audio_order(_WINDOW.audio_latents)])
    assert torch.equal(layout.timestep_audio_idx[:74], audio_block[tail_audio_rows(_WINDOW.audio_latents)])
    assert sorted(layout.timestep_audio_idx.tolist()) == audio_block.tolist()
    held = layout.held_audio.tolist()
    assert held == [*range(37), *range(_WINDOW.audio_latents, _WINDOW.audio_latents + 37)]
    assert layout.audio_generated.sum() == _WINDOW.audio_rows - 74
    assert not layout.audio_generated[held].any()


def test_extend_with_a_keyframe_holds_both_in_front():
    conditioning = Conditioning(keyframes=(_latents(1, 10, _WINDOW),), anchors=("last",), tail_latents=_latents(7, 3, _WINDOW), tail_audio=_tail_audio())
    layout = build_layout(_TEXT_TAGS, _WINDOW, _CPU, conditioning)
    assert (layout.keyframe_rows, layout.tail_rows, layout.condition_video_rows) == (_CELLS, 7 * _CELLS, 8 * _CELLS)


@pytest.mark.parametrize(
    ("conditioning", "match"),
    [
        (lambda: Conditioning(keyframes=(_latents(1, 1),), anchors=()), "one anchor"),
        (lambda: Conditioning(tail_audio=torch.zeros(10, 32)), "tail audio"),
        (lambda: Conditioning(keyframes=(torch.zeros(1, 24, 1, 2, 2),), anchors=("first",)), "Keyframe 0"),
        (lambda: Conditioning(tail_latents=torch.zeros(1, 24, 3, 2, 2)), "tail must be shaped"),
        (lambda: Conditioning(tail_latents=_latents(7, 1)), "must leave frames"),
    ],
    ids=["anchors", "audio-shape", "keyframe-canvas", "tail-canvas", "tail-fills-window"],
)
def test_bad_conditioning_is_refused(conditioning, match):
    with pytest.raises(ValueError, match=match):
        build_layout(_TEXT_TAGS, _PLAIN, _CPU, conditioning())


# --- plain runs ------------------------------------------------------------------------------------------------------


def test_a_plain_run_does_one_forward_per_step_with_the_payload():
    transformer, (video, audio) = _run()
    assert len(transformer.calls) == 8
    layout = build_layout(_TEXT_TAGS, _PLAIN, _CPU, None)
    for call in transformer.calls:
        assert call["video"].shape == (1, _PLAIN.video_rows, 96)
        assert call["audio"].shape == (1, _PLAIN.audio_rows, 32)
        assert call["text"].shape == (1, 4, _TEXT_DIM)
        assert call["return_dict"] is False
        assert call["timestep"].dtype == torch.float32
        assert call["timestep_indices"].shape == (layout.kwargs["position_ids"].shape[0],)
        for name in ("token_tags", "position_ids", "video_indices", "audio_indices", "text_indices"):
            assert torch.equal(call[name], layout.kwargs[name]), name
    assert video.shape == (1, 24, 7, 4, 4) and video.dtype == torch.float32
    assert audio.shape == (2, 32, 37) and audio.dtype == torch.float32


def test_a_plain_run_uses_the_generated_levels_only():
    transformer, _ = _run()
    video_scheduler, audio_scheduler = _schedulers()
    for call, t, audio_t in zip(transformer.calls, video_scheduler.timesteps, audio_scheduler.timesteps, strict=True):
        levels = set(call["timestep"].tolist())
        assert levels == {float(t), float(audio_t)}


def test_the_callback_reports_every_step():
    events: list[dict] = []
    _run(step_callback=events.append, phase="refine")
    assert events == [{"phase": "refine", "current_step": step, "total_steps": 8} for step in range(1, 9)]
    events.clear()
    _run(schedule=_SCHEDULE_4, step_callback=events.append)  # the phase defaults to video
    assert events == [{"phase": "video", "current_step": step, "total_steps": 4} for step in range(1, 5)]


def test_a_forward_hook_can_rewrite_the_arguments():
    # A pre-hook (the AdaLN fold's way in) sees the keyword arguments and may replace them.
    transformer = _Recorder()
    calls: list[dict] = []

    def hook(module, args, kwargs):
        calls.append(kwargs)
        return args, {**kwargs, "audio_hidden_states": kwargs["audio_hidden_states"] * 0.0}

    transformer.register_forward_pre_hook(hook, with_kwargs=True)
    _run(transformer=transformer)
    assert len(calls) == 8
    assert all(not call["audio"].any() for call in transformer.calls)


def test_a_callback_exception_propagates_and_ends_the_run():
    transformer = _Recorder()

    class Cancelled(Exception):
        pass

    def cancel(event):
        if event["current_step"] == 3:
            raise Cancelled

    with pytest.raises(Cancelled):
        _run(transformer=transformer, step_callback=cancel)
    assert len(transformer.calls) == 3


def test_a_transformer_failure_propagates():
    transformer = _Recorder(fail_at=2)
    with pytest.raises(RuntimeError, match="out of memory"):
        _run(transformer=transformer)
    assert len(transformer.calls) == 2


def test_a_run_is_deterministic_and_the_seed_matters():
    _, first = _run(seed=3)
    _, again = _run(seed=3)
    _, other = _run(seed=4)
    assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))
    assert not torch.equal(first[0], other[0]) and not torch.equal(first[1], other[1])


def test_the_callers_schedulers_are_left_reusable():
    schedulers = _schedulers()
    results = [denoise(_Recorder(), _embeds(), _PLAIN, schedulers, 3, _CPU, text_token_tags=_TEXT_TAGS) for _ in range(2)]
    assert schedulers[0].step_index is None and schedulers[1].step_index is None
    assert all(torch.equal(a, b) for a, b in zip(*results, strict=True))


def test_unseeded_runs_draw_from_the_global_generator():
    torch.manual_seed(0)
    _, first = _run(seed=None)
    torch.manual_seed(0)
    _, again = _run(seed=None)
    assert torch.equal(first[0], again[0])


def test_text_tags_default_to_all_text():
    recorder = _Recorder()
    denoise(recorder, _embeds(5), _PLAIN, _schedulers(), 1, _CPU)
    assert recorder.calls[0]["token_tags"][:5].tolist() == [TEXT_TAG] * 5


def test_noise_is_drawn_video_first_then_audio_from_one_generator():
    video, audio = draw_noise(_PLAIN, torch.Generator().manual_seed(9), _CPU)
    generator = torch.Generator().manual_seed(9)
    expected_video = randn_tensor((1, 24, 7, 4, 4), generator=generator, device=_CPU, dtype=torch.float32)
    expected_audio = randn_tensor((74, 32), generator=generator, device=_CPU, dtype=torch.float32)
    assert torch.equal(video, patchify_video_latents(expected_video, (1, 2, 2)))
    assert torch.equal(audio, expected_audio)


# --- keyframes -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("anchors", [("first",), ("first", "last")], ids=["one", "two"])
def test_keyframe_rows_are_prenoised_in_draw_order_and_never_written(anchors):
    conditioning = _keyframes(*anchors)
    transformer, (video, audio) = _run(conditioning=conditioning, seed=21)
    scheduler = _schedulers()[0]
    generator = torch.Generator().manual_seed(21)
    expected = torch.cat([patchify_video_latents(prenoise(scheduler, latents, KEYFRAME_T, generator), (1, 2, 2)) for latents in conditioning.keyframes])
    held = len(anchors) * _CELLS
    assert len(transformer.calls) == 8
    for call in transformer.calls:
        assert call["video"].shape == (1, held + _PLAIN.video_rows, 96)
        assert torch.equal(call["video"][0, :held], expected)
        # The conditioning rows are at the keyframe level, the generated ones at the video level.
        levels = call["timestep"][call["timestep_indices"][call["video_indices"]]]
        assert torch.all(levels[:held] == torch.tensor(KEYFRAME_T, dtype=torch.float32))
    assert video.shape == (1, 24, 7, 4, 4)  # the keyframes are not part of the result


# --- extend ----------------------------------------------------------------------------------------------------------


def test_extend_holds_the_tail_rows_of_both_channels_across_every_step():
    conditioning = _extend()
    transformer, (video, audio) = _run(_WINDOW, conditioning=conditioning, seed=5)
    layout = build_layout(_TEXT_TAGS, _WINDOW, _CPU, conditioning)
    scheduler = _schedulers()[0]
    generator = torch.Generator().manual_seed(5)
    expected_video = patchify_video_latents(prenoise(scheduler, conditioning.tail_latents, KEYFRAME_T, generator), (1, 2, 2))
    expected_audio = prenoise(scheduler, conditioning.tail_audio, AUDIO_COND_T, generator)
    assert len(transformer.calls) == 8
    for step, call in enumerate(transformer.calls):
        assert torch.equal(call["video"][0, : layout.tail_rows], expected_video), step
        assert torch.equal(call["audio"][0, layout.held_audio], expected_audio), step
        # Channel 0 rows 0..36 and channel 1 rows 0..36: each channel's tail, explicitly.
        for channel in range(2):
            start = channel * _WINDOW.audio_latents
            assert torch.equal(call["audio"][0, start : start + 37], expected_audio[channel * 37 : (channel + 1) * 37]), (step, channel)
    # The generated rows moved and the held ones did not, in the result as well.
    first, last = transformer.calls[0], transformer.calls[-1]
    assert not torch.equal(first["video"][0, layout.tail_rows :], last["video"][0, layout.tail_rows :])
    assert not torch.equal(first["audio"][0, layout.audio_generated], last["audio"][0, layout.audio_generated])
    assert video.shape == (1, 24, 12, 4, 4) and audio.shape == (2, 32, 65)
    assert torch.equal(sampler_module._patchify(video)[: layout.tail_rows], expected_video)
    assert torch.equal(audio.permute(0, 2, 1).reshape(-1, 32)[layout.held_audio], expected_audio)


def test_extend_timesteps_pin_the_tail_and_step_the_rest():
    conditioning = _extend()
    transformer, _ = _run(_WINDOW, conditioning=conditioning)
    layout = build_layout(_TEXT_TAGS, _WINDOW, _CPU, conditioning)
    video_scheduler, audio_scheduler = _schedulers()
    keyframe_level = torch.tensor(KEYFRAME_T, dtype=torch.float32)
    for call, t, audio_t in zip(transformer.calls, video_scheduler.timesteps, audio_scheduler.timesteps, strict=True):
        per_row = call["timestep"][call["timestep_indices"]]
        video_levels = per_row[call["video_indices"]]
        audio_levels = per_row[call["audio_indices"]]
        assert torch.all(video_levels[: layout.tail_rows] == keyframe_level)
        assert torch.all(video_levels[layout.tail_rows :] == torch.tensor(float(t), dtype=torch.float32))
        assert torch.all(audio_levels[layout.held_audio] == 1.0)
        assert torch.all(audio_levels[layout.audio_generated] == torch.tensor(float(audio_t), dtype=torch.float32))


def test_extend_matches_a_reference_loop_with_the_held_rows_moved_to_the_front():
    # The reference loop slices leading rows; permuting the audio rows tail-first makes it the oracle for the mask.
    conditioning = _extend()
    _, (video, audio) = _run(_WINDOW, conditioning=conditioning, seed=8)
    layout = build_layout(_TEXT_TAGS, _WINDOW, _CPU, conditioning)
    video_scheduler, audio_scheduler = _schedulers()
    generator = torch.Generator().manual_seed(8)
    tail_video = patchify_video_latents(prenoise(video_scheduler, conditioning.tail_latents, KEYFRAME_T, generator), (1, 2, 2))
    tail_audio = prenoise(video_scheduler, conditioning.tail_audio, AUDIO_COND_T, generator)
    rows_video, rows_audio = draw_noise(_WINDOW, generator, _CPU)
    rows_video[: layout.tail_rows] = tail_video
    order = tail_first_audio_order(_WINDOW.audio_latents)
    rows_audio[layout.held_audio] = tail_audio
    front_audio = rows_audio[order].clone()
    for t, audio_t in zip(video_scheduler.timesteps, audio_scheduler.timesteps, strict=True):
        velocity_video = rows_video * 0.25 + 0.5
        velocity_audio = front_audio * 0.25 - 0.5
        rows_video[layout.tail_rows :] = video_scheduler.step(velocity_video[layout.tail_rows :].float(), t, rows_video[layout.tail_rows :], return_dict=False)[0]
        front_audio[74:] = audio_scheduler.step(velocity_audio[74:].float(), audio_t, front_audio[74:], return_dict=False)[0]
    restored = torch.empty_like(front_audio)
    restored[order] = front_audio
    assert torch.equal(sampler_module._patchify(video), rows_video)
    assert torch.equal(audio.permute(0, 2, 1).reshape(-1, 32), restored)


# --- the reference loop on a tiny transformer --------------------------------------------------------------------------


def _reference_denoise(model, geo, schedule: H3Schedule, seed: int, anchors: tuple[str, ...], keyframes: tuple[torch.Tensor, ...]):
    """Run diffusers' own loop blocks (denoiser and scheduler step) on the same draws; return the final ``(video rows, audio rows)``."""
    video_scheduler, audio_scheduler = build_schedulers(schedule)
    generator = torch.Generator().manual_seed(seed)
    conditions = [
        patchify_video_latents(video_scheduler.scale_noise(latents, 0.999, randn_tensor(latents.shape, generator=generator, device=_CPU, dtype=torch.float32)), (1, 2, 2)) for latents in keyframes
    ]
    video = patchify_video_latents(randn_tensor((1, 24, geo.latent_frames, geo.latent_height, geo.latent_width), generator=generator, device=_CPU, dtype=torch.float32), (1, 2, 2))
    audio = randn_tensor((geo.audio_rows, 32), generator=generator, device=_CPU, dtype=torch.float32)
    video = torch.cat([*conditions, video])
    position_ids, token_tags, video_idx, audio_idx, text_idx, condition_rows, _ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        torch.tensor(_TEXT_TAGS), geo.latent_frames, geo.latent_height, geo.latent_width, geo.audio_latents, (1, 2, 2), 2, 2, 0, anchors
    )
    plan = [
        MiniMaxH3SetTimestepsStep.build_row_timesteps(video_idx, audio_idx, condition_rows, 0, 4, float(t), float(audio_t), max(float(t), 0.999), 1.0)
        for t, audio_t in zip(video_scheduler.timesteps, audio_scheduler.timesteps, strict=True)
    ]
    state = SimpleNamespace(
        latents=video,
        audio_latents=audio,
        prompt_embeds=_embeds(),
        row_timestep_plan=plan,
        denoiser_input_fields={"position_ids": position_ids, "token_tags": token_tags, "video_indices": video_idx, "audio_indices": audio_idx, "text_indices": text_idx},
        attention_kwargs=None,
        audio_timesteps=audio_scheduler.timesteps,
        num_condition_video_rows=condition_rows,
        num_condition_audio_rows=0,
    )
    components = SimpleNamespace(transformer=model, scheduler=video_scheduler, audio_scheduler=audio_scheduler)
    denoiser, updater = MiniMaxH3LoopDenoiser(), MiniMaxH3LoopSchedulerStep()
    for i, t in enumerate(video_scheduler.timesteps):
        denoiser(components, state, i, t)
        updater(components, state, i, t)
    return state.latents[condition_rows:], state.audio_latents


def test_a_plain_run_equals_the_diffusers_loop_on_a_tiny_transformer():
    model = tiny_transformer()
    video, audio = denoise(model, _embeds(), _PLAIN, _schedulers(), 17, _CPU, text_token_tags=_TEXT_TAGS)
    expected_video, expected_audio = _reference_denoise(model, _PLAIN, _SCHEDULE_8, 17, (), ())
    assert torch.equal(sampler_module._patchify(video), expected_video)
    assert torch.equal(audio.permute(0, 2, 1).reshape(-1, 32), expected_audio)


def test_a_keyframe_run_equals_the_diffusers_loop_on_a_tiny_transformer():
    model = tiny_transformer()
    conditioning = _keyframes("first", "last")
    video, audio = denoise(model, _embeds(), _PLAIN, _schedulers(), 17, _CPU, conditioning=conditioning, text_token_tags=_TEXT_TAGS)
    expected_video, expected_audio = _reference_denoise(model, _PLAIN, _SCHEDULE_8, 17, conditioning.anchors, conditioning.keyframes)
    assert torch.equal(sampler_module._patchify(video), expected_video)
    assert torch.equal(audio.permute(0, 2, 1).reshape(-1, 32), expected_audio)


def test_an_extend_run_through_the_real_transformer_is_finite():
    # The only test that pushes the extend layout (tail-first audio indices, held audio rows, four timestep levels) through the real model.
    result = denoise(tiny_transformer(), _embeds(), _WINDOW, _schedulers(), 2, _CPU, conditioning=_extend(), text_token_tags=_TEXT_TAGS)
    assert all(torch.isfinite(part).all() for part in result)


# --- refine ----------------------------------------------------------------------------------------------------------


class _ZeroVelocity(_Recorder):
    """A stand-in transformer that records its forwards and predicts no change."""

    def forward(self, hidden_states, audio_hidden_states, *args, **kwargs):
        super().forward(hidden_states, audio_hidden_states, *args, **kwargs)
        return torch.zeros_like(hidden_states), torch.zeros_like(audio_hidden_states)


def _refine_inputs(geo=_PLAIN, seed: int = 31):
    latents = _latents(geo.latent_frames, seed, geo)
    generator = torch.Generator().manual_seed(seed + 1)
    audio = torch.randn(geo.audio_rows, 32, generator=generator)
    noise = (torch.randn(latents.shape, generator=generator), torch.randn(audio.shape, generator=generator))
    return latents, audio, noise


def _refine(geo=_PLAIN, *, start: int = 6, transformer=None, conditioning: Conditioning | None = None, **kwargs):
    transformer = transformer or _ZeroVelocity()
    latents, audio, noise = _refine_inputs(geo)
    result = sampler_module.refine(transformer, _embeds(), geo, _schedulers(), latents, audio, noise, start=start, device=_CPU, conditioning=conditioning, text_token_tags=_TEXT_TAGS, **kwargs)
    return transformer, result, (latents, audio, noise)


def test_a_refine_with_no_velocity_returns_the_renoised_latents():
    _, result, (latents, _audio, (noise, _noise_audio)) = _refine(start=6)
    video_scheduler = _schedulers()[0]
    t6 = float(video_scheduler.timesteps[6])
    assert 1 - t6 == pytest.approx(0.667, abs=1e-3)
    assert result.shape == latents.shape
    assert result.dtype == torch.float32
    assert torch.allclose(result, t6 * latents + (1 - t6) * noise, atol=1e-5)


def test_a_refine_from_step_six_runs_the_last_two_forwards_and_reports_them():
    events: list[dict] = []
    transformer, _, _ = _refine(start=6, step_callback=events.append, phase="video_upscale_stage_2")
    assert len(transformer.calls) == 2
    assert events == [
        {"phase": "video_upscale_stage_2", "current_step": 1, "total_steps": 2},
        {"phase": "video_upscale_stage_2", "current_step": 2, "total_steps": 2},
    ]


def test_a_refine_runs_the_timesteps_from_start_on():
    video_scheduler, audio_scheduler = _schedulers()
    transformer, _, _ = _refine(start=6)
    for call, step in zip(transformer.calls, (6, 7), strict=True):
        levels = call["timestep"][call["timestep_indices"][call["video_indices"]]]
        audio_levels = call["timestep"][call["timestep_indices"][call["audio_indices"]]]
        assert torch.allclose(levels, torch.tensor(float(video_scheduler.timesteps[step])))
        assert torch.allclose(audio_levels, torch.tensor(float(audio_scheduler.timesteps[step])))


def test_a_refine_renoises_the_audio_rows_too():
    transformer, _, (_latents, audio, (_noise, noise_audio)) = _refine(start=6)
    audio_scheduler = _schedulers()[1]
    t6 = float(audio_scheduler.timesteps[6])
    assert torch.allclose(transformer.calls[0]["audio"][0], t6 * audio + (1 - t6) * noise_audio, atol=1e-5)


def test_a_refine_holds_its_keyframes_at_every_forward_prenoised_with_the_keyframe_seed():
    conditioning = _keyframes("first", "last")
    transformer, result, _ = _refine(start=6, conditioning=conditioning, keyframe_seed=77)
    scheduler = _schedulers()[0]
    generator = torch.Generator().manual_seed(77)
    expected = torch.cat([patchify_video_latents(prenoise(scheduler, latents, KEYFRAME_T, generator), (1, 2, 2)) for latents in conditioning.keyframes])
    held = 2 * _CELLS
    assert len(transformer.calls) == 2
    for call in transformer.calls:
        assert call["video"].shape == (1, held + _PLAIN.video_rows, 96)
        assert torch.equal(call["video"][0, :held], expected)
        levels = call["timestep"][call["timestep_indices"][call["video_indices"]]]
        assert torch.all(levels[:held] == torch.tensor(KEYFRAME_T, dtype=torch.float32))
    assert result.shape == (1, 24, 7, 4, 4)  # the keyframes are not part of the result


def test_a_refine_leaves_the_callers_schedulers_reusable():
    schedulers = _schedulers()
    before = [scheduler.timesteps.clone() for scheduler in schedulers]
    latents, audio, noise = _refine_inputs()
    sampler_module.refine(_ZeroVelocity(), _embeds(), _PLAIN, schedulers, latents, audio, noise, start=6, device=_CPU, text_token_tags=_TEXT_TAGS)
    assert all(torch.equal(a, b.timesteps) for a, b in zip(before, schedulers, strict=True))
    assert all(scheduler.step_index is None for scheduler in schedulers)


def test_a_refine_callback_exception_propagates():
    class Cancelled(Exception):
        pass

    def cancel(event):
        if event["current_step"] == 1:
            raise Cancelled

    transformer = _ZeroVelocity()
    with pytest.raises(Cancelled):
        _refine(transformer=transformer, step_callback=cancel)
    assert len(transformer.calls) == 1


@pytest.mark.parametrize("start", [-1, 8])
def test_a_refine_start_outside_the_schedule_is_refused(start):
    with pytest.raises(ValueError, match="start"):
        _refine(start=start)


def test_a_refine_refuses_inputs_that_do_not_fit_the_window():
    latents, audio, (noise, noise_audio) = _refine_inputs()
    args = (_ZeroVelocity(), _embeds(), _PLAIN, _schedulers())
    with pytest.raises(ValueError, match="latents"):
        sampler_module.refine(*args, latents[:, :, :3], audio, (noise, noise_audio), start=6, device=_CPU)
    with pytest.raises(ValueError, match="audio rows"):
        sampler_module.refine(*args, latents, audio[:-1], (noise, noise_audio), start=6, device=_CPU)
    with pytest.raises(ValueError, match="noise"):
        sampler_module.refine(*args, latents, audio, (noise[:, :, :3], noise_audio), start=6, device=_CPU)
    with pytest.raises(ValueError, match="noise"):
        sampler_module.refine(*args, latents, audio, (noise, noise_audio[:-1]), start=6, device=_CPU)


def test_a_refine_refuses_an_extend_tail():
    with pytest.raises(ValueError, match="tail"):
        _refine(geo=_WINDOW, conditioning=_extend())
