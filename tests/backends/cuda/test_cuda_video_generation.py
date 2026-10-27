"""Tests for text-to-video and image-to-video generation on the diffusers LTX backend, including the two-stage upscale, audio and progress contract."""

from __future__ import annotations


import pytest
from PIL import Image

from tests.backends.cuda._helpers import (
    install_fake_weights,
    _AudioTextPipeline,
    _DISTILLED_SIGMAS,
    _FakeTextPipeline,
    _NEUTRAL_GUIDANCE,
    _NoRatePipeline,
    _STAGE2_SIGMAS,
    _UPSAMPLER_SOURCE,
    _guidance,
    _load_upscale_model,
    _make_runtime,
    _patch_runtime,
    _patch_upsampler_io,
    _text_to_video_args,
)

from zvisiongenerator.backends.cuda.video.backend import (
    _DISTILLED_GUIDANCE,
    DiffusersVideoBackend,
    _stage1_sigmas,
)


@pytest.fixture(autouse=True)
def fake_weights(monkeypatch):
    """Fake the weights, placement and memory checks of ``load_model`` (see ``_cuda_helpers.install_fake_weights``)."""
    return install_fake_weights(monkeypatch)


@pytest.mark.parametrize("steps", [4, 8])
def test_text_to_video_exports_path_and_reports_progress(monkeypatch, tmp_path, steps):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime()
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._load_runtime_dependencies", lambda: runtime)
    model, _ = backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", low_memory=False)
    events: list[dict[str, object]] = []
    output_path = tmp_path / "video.mp4"

    result = backend.text_to_video(
        model=model,
        prompt="a river",
        width=704,
        height=448,
        num_frames=49,
        seed=123,
        steps=steps,
        output_path=str(output_path),
        step_callback=events.append,
    )

    assert result == output_path
    assert runtime.encoded["frames"] == ["frame-1", "frame-2"]
    assert runtime.encoded["output_path"] == str(output_path)
    assert runtime.encoded["fps"] == 24
    call = model.text_to_video.calls[0]
    assert call["prompt_embeds"] == "embeds:a river"
    assert call["prompt_attention_mask"] == "mask:a river"
    assert "prompt" not in call
    assert call["width"] == 704
    assert call["height"] == 448
    assert call["num_frames"] == 49
    assert call["sigmas"] == _stage1_sigmas(_DISTILLED_SIGMAS, steps)
    assert call["num_inference_steps"] == len(call["sigmas"]) == steps
    assert _guidance(call) == _NEUTRAL_GUIDANCE
    assert call["frame_rate"] == 24
    assert call["generator"].seed == 123
    assert events == [{"phase": "video", "current_step": step, "total_steps": steps} for step in range(1, steps + 1)]
    assert runtime.torch.cuda.empty_cache_calls == 1


def test_distilled_guidance_turns_every_guidance_term_off():
    # diffusers' own defaults for the other terms (audio 7, STG 1, modality 3) over-cook a distilled model.
    assert dict(_DISTILLED_GUIDANCE) == _NEUTRAL_GUIDANCE


@pytest.mark.parametrize(
    ("steps", "expected"),
    [
        (9, list(_DISTILLED_SIGMAS)),
        (8, list(_DISTILLED_SIGMAS)),
        (7, [1.0, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]),
        (5, [1.0, 0.975, 0.909375, 0.725, 0.421875]),
        (3, [1.0, 0.725, 0.421875]),
        (1, [1.0]),
    ],
)
def test_stage1_sigmas_keeps_start_and_tail(steps, expected):
    assert _stage1_sigmas(_DISTILLED_SIGMAS, steps) == expected


def test_stage1_sigmas_rejects_fewer_than_one_step():
    with pytest.raises(ValueError, match="at least 1"):
        _stage1_sigmas(_DISTILLED_SIGMAS, 0)


def test_image_to_video_uses_loaded_image(monkeypatch, tmp_path):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime()
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._load_runtime_dependencies", lambda: runtime)
    model, _ = backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="i2v", low_memory=False)
    input_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "blue").save(input_path)
    output_path = tmp_path / "i2v.mp4"

    result = backend.image_to_video(
        model=model,
        image_path=str(input_path),
        prompt="animate this",
        width=704,
        height=448,
        num_frames=49,
        seed=99,
        steps=4,
        output_path=str(output_path),
    )

    assert result == output_path
    call = model.image_to_video.calls[0]
    assert call["prompt_embeds"] == "embeds:animate this"
    assert call["image"].size == (704, 448)  # fitted to the video size
    assert call["sigmas"] == _stage1_sigmas(_DISTILLED_SIGMAS, 4)
    assert call["num_inference_steps"] == 4
    assert call["guidance_scale"] == 1.0
    assert _guidance(call) == _NEUTRAL_GUIDANCE


@pytest.mark.parametrize(("source_size", "kept_colors"), [((300, 100), {"green"}), ((100, 300), {"green"}), ((200, 200), {"green"})])
def test_image_to_video_crops_the_image_to_the_video_aspect_instead_of_stretching_it(monkeypatch, tmp_path, source_size, kept_colors):
    _patch_runtime(monkeypatch, _make_runtime())
    model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="i2v", low_memory=False)
    # A green center square with red margins along the long side: a stretch would keep the red, a center crop drops it.
    width, height = source_size
    side = min(width, height)
    image = Image.new("RGB", source_size, "red")
    image.paste(Image.new("RGB", (side, side), (0, 255, 0)), ((width - side) // 2, (height - side) // 2))
    image.save(tmp_path / "input.png")

    DiffusersVideoBackend().image_to_video(model, str(tmp_path / "input.png"), "animate this", 64, 64, 9, 1, 8, str(tmp_path / "out.mp4"))

    fitted = model.image_to_video.calls[0]["image"]
    colors = {"green" if pixel[1] > 200 and pixel[0] < 50 else "red" if pixel[0] > 200 else "other" for pixel in fitted.get_flattened_data()}
    assert fitted.size == (64, 64)
    assert colors == kept_colors


def _run_upscaled_text_to_video(monkeypatch, tmp_path, *, low_memory=False, **overrides):
    runtime = _make_runtime()
    model = _load_upscale_model(monkeypatch, low_memory=low_memory, runtime=runtime)
    DiffusersVideoBackend().text_to_video(model=model, **_text_to_video_args(tmp_path, **overrides))
    return model, runtime


def test_upscale_requires_stage1_steps_and_stage1_steps_require_upscale(monkeypatch, tmp_path):
    backend = DiffusersVideoBackend()
    model = _load_upscale_model(monkeypatch, low_memory=False)

    with pytest.raises(ValueError, match="stage1_steps"):
        backend.text_to_video(model=model, **_text_to_video_args(tmp_path))
    assert model.text_to_video.calls == []

    _patch_runtime(monkeypatch, _make_runtime())
    plain, _ = backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", low_memory=False)
    with pytest.raises(RuntimeError, match="upsampler"):
        backend.text_to_video(model=plain, stage1_steps=4, **_text_to_video_args(tmp_path))
    assert plain.text_to_video.calls == []


def test_upscale_t2v_runs_half_size_stage1_then_full_size_refine(monkeypatch, tmp_path):
    runtime = _make_runtime()
    model = _load_upscale_model(monkeypatch, low_memory=False, runtime=runtime)

    result = DiffusersVideoBackend().text_to_video(model=model, **_text_to_video_args(tmp_path, stage1_steps=4))

    stage1, stage2 = model.text_to_video.calls
    assert (stage1["width"], stage1["height"]) == (256, 192)
    assert stage1["output_type"] == "latent"
    assert stage1["sigmas"] == _stage1_sigmas(_DISTILLED_SIGMAS, 4)
    assert stage1["num_inference_steps"] == 4
    assert stage1["guidance_scale"] == 1.0
    assert "latents" not in stage1
    assert (stage2["width"], stage2["height"]) == (512, 384)
    assert stage2["output_type"] == "latent"  # decoded by streaming_decode
    assert stage2["sigmas"] == list(_STAGE2_SIGMAS)
    assert stage2["num_inference_steps"] == 3
    assert _guidance(stage1) == _guidance(stage2) == _NEUTRAL_GUIDANCE
    assert stage2["num_frames"] == stage1["num_frames"] == 49
    assert stage2["generator"] is stage1["generator"]
    assert stage1["generator"].seed == 77
    # the refine pass starts from the upsampled stage-1 latents and its audio
    assert stage2["latents"] == "upsampled-latent"
    assert stage2["audio_latents"] == "audio-latent"
    assert stage2["noise_scale"] == _STAGE2_SIGMAS[0]
    (upsample,) = model.upsample_pipeline.calls
    assert upsample == {"latents": "video-latent", "height": 192, "width": 256, "num_frames": 49, "output_type": "latent"}
    # the stage-2 frames are exported
    assert result == tmp_path / "upscaled.mp4"
    assert runtime.encoded["frames"] == ["frame-1", "frame-2"]
    assert runtime.encoded["fps"] == 24
    assert runtime.torch.cuda.empty_cache_calls == 1


def test_upscale_refine_uses_swapped_scheduler_and_restores_it(monkeypatch, tmp_path):
    model = _load_upscale_model(monkeypatch, low_memory=False)
    pipeline = model.text_to_video
    original = pipeline.scheduler

    DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, **_text_to_video_args(tmp_path))

    stage1_scheduler, stage2_scheduler = pipeline.schedulers_at_call
    assert stage1_scheduler is original
    assert stage2_scheduler is not original
    assert stage2_scheduler.config is original.config
    assert stage2_scheduler.overrides == {"use_dynamic_shifting": False, "shift_terminal": None}
    assert pipeline.scheduler is original


def test_upscale_restores_scheduler_when_the_step_callback_raises(monkeypatch, tmp_path):
    model = _load_upscale_model(monkeypatch, low_memory=False)
    pipeline = model.text_to_video
    original = pipeline.scheduler

    def _cancel_in_stage2(event):
        if event["phase"] == "video_upscale_stage_2":
            raise KeyboardInterrupt("quit")

    with pytest.raises(KeyboardInterrupt):
        DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, step_callback=_cancel_in_stage2, **_text_to_video_args(tmp_path))

    assert pipeline.schedulers_at_call[1] is not original
    assert pipeline.scheduler is original


def test_upscale_leaves_the_scheduler_alone_when_stage1_is_cancelled(monkeypatch, tmp_path):
    model = _load_upscale_model(monkeypatch, low_memory=False)
    pipeline = model.text_to_video
    original = pipeline.scheduler

    def _cancel(_event):
        raise KeyboardInterrupt("quit")

    with pytest.raises(KeyboardInterrupt):
        DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, step_callback=_cancel, **_text_to_video_args(tmp_path))

    assert len(pipeline.calls) == 1
    assert model.upsample_pipeline.calls == []
    assert pipeline.scheduler is original


def test_upscale_i2v_refines_on_the_image_pipeline(monkeypatch, tmp_path):
    runtime = _make_runtime()
    model = _load_upscale_model(monkeypatch, low_memory=False, mode="i2v", runtime=runtime)
    pipeline = model.image_to_video
    input_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "blue").save(input_path)

    DiffusersVideoBackend().image_to_video(model=model, image_path=str(input_path), stage1_steps=4, **_text_to_video_args(tmp_path))

    stage1, stage2 = pipeline.calls
    assert (stage1["width"], stage1["height"]) == (256, 192)
    assert stage1["output_type"] == "latent"
    assert stage1["image"].size == (512, 384)  # fitted to the final size; the pipeline scales it to stage 1's, same aspect
    assert (stage2["width"], stage2["height"]) == (512, 384)
    assert stage2["latents"] == "upsampled-latent"
    assert stage2["audio_latents"] == "audio-latent"
    assert stage2["output_type"] == "latent"  # decoded by streaming_decode
    assert model.upsample_pipeline.vae is pipeline.vae
    assert runtime.encoded["frames"] == ["image-frame-1"]


@pytest.mark.parametrize(("width", "height"), [(520, 384), (512, 392)])
def test_upscale_rejects_sizes_not_divisible_by_64(monkeypatch, tmp_path, width, height):
    model = _load_upscale_model(monkeypatch, low_memory=False)

    with pytest.raises(ValueError, match="divisible by 64"):
        DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, **_text_to_video_args(tmp_path, width=width, height=height))

    assert model.text_to_video.calls == []
    assert model.upsample_pipeline.calls == []


def test_text_to_video_muxes_the_first_waveform_at_the_vocoder_rate(monkeypatch, tmp_path):
    runtime = _make_runtime(text_pipeline=_AudioTextPipeline)
    _patch_runtime(monkeypatch, runtime)
    backend = DiffusersVideoBackend()
    model, _ = backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", low_memory=False)

    result = backend.text_to_video(model=model, **_text_to_video_args(tmp_path))

    assert result == tmp_path / "upscaled.mp4"
    assert runtime.encoded["frames"] == ["frame-1", "frame-2"]
    assert runtime.encoded["fps"] == 24
    assert runtime.encoded["output_path"] == str(tmp_path / "upscaled.mp4")
    assert runtime.encoded["audio"].name == "clip-0"
    assert runtime.encoded["audio"].conversions == ["float", "cpu"]
    assert runtime.encoded["audio_sample_rate"] == 24000


def test_upscale_muxes_the_refined_stage_audio(monkeypatch, tmp_path):
    runtime = _make_runtime(text_pipeline=_AudioTextPipeline)
    model = _load_upscale_model(monkeypatch, low_memory=False, runtime=runtime)

    DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, **_text_to_video_args(tmp_path))

    assert runtime.encoded["audio"].name == "clip-0"
    assert runtime.encoded["audio_sample_rate"] == 24000


def test_audio_without_a_vocoder_rate_is_dropped_with_a_warning(monkeypatch, tmp_path):
    runtime = _make_runtime(text_pipeline=_NoRatePipeline)
    _patch_runtime(monkeypatch, runtime)
    backend = DiffusersVideoBackend()
    model, _ = backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", low_memory=False)

    with pytest.warns(UserWarning, match="without sound"):
        backend.text_to_video(model=model, **_text_to_video_args(tmp_path))

    assert runtime.encoded["frames"] == ["frame-1", "frame-2"]
    assert runtime.encoded["audio"] is None
    assert runtime.encoded["audio_sample_rate"] is None


def test_output_without_audio_is_encoded_as_frames_only(monkeypatch, tmp_path):
    runtime = _make_runtime()
    _patch_runtime(monkeypatch, runtime)
    backend = DiffusersVideoBackend()
    model, _ = backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", low_memory=False)

    backend.text_to_video(model=model, **_text_to_video_args(tmp_path))

    assert runtime.encoded["frames"] == ["frame-1", "frame-2"]
    assert runtime.encoded["audio"] is None
    assert runtime.encoded["audio_sample_rate"] is None


@pytest.mark.parametrize(("stage1_steps", "stage1_len"), [(8, 8), (4, 4), (1, 1), (12, 8)])
def test_upscale_progress_matches_two_stage_contract(monkeypatch, tmp_path, stage1_steps, stage1_len):
    events: list[dict[str, object]] = []
    total = stage1_len + 3

    _run_upscaled_text_to_video(monkeypatch, tmp_path, stage1_steps=stage1_steps, step_callback=events.append)

    expected = [{"phase": "video_upscale_stage_1", "current_step": step, "total_steps": total} for step in range(1, stage1_len + 1)]
    expected += [{"phase": "video_upscale_stage_2", "current_step": stage1_len + step, "total_steps": total} for step in range(1, 4)]
    assert events == expected
    assert events[-1]["current_step"] == events[-1]["total_steps"]


def test_upscale_i2v_progress_matches_two_stage_contract(monkeypatch, tmp_path):
    model = _load_upscale_model(monkeypatch, low_memory=False, mode="i2v")
    input_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "blue").save(input_path)
    events: list[dict[str, object]] = []

    DiffusersVideoBackend().image_to_video(model=model, image_path=str(input_path), stage1_steps=4, step_callback=events.append, **_text_to_video_args(tmp_path))

    assert [(e["phase"], e["current_step"], e["total_steps"]) for e in events] == [
        ("video_upscale_stage_1", 1, 7),
        ("video_upscale_stage_1", 2, 7),
        ("video_upscale_stage_1", 3, 7),
        ("video_upscale_stage_1", 4, 7),
        ("video_upscale_stage_2", 5, 7),
        ("video_upscale_stage_2", 6, 7),
        ("video_upscale_stage_2", 7, 7),
    ]


@pytest.mark.parametrize(("low_memory", "expected_moves"), [(True, ["cuda", "cpu"]), (False, [])])
def test_upscale_moves_the_upsampler_to_the_gpu_only_in_low_memory_mode(monkeypatch, tmp_path, low_memory, expected_moves):
    model, _ = _run_upscaled_text_to_video(monkeypatch, tmp_path, low_memory=low_memory, stage1_steps=4)

    upsampler = model.upsample_pipeline.latent_upsampler
    assert upsampler.moves == expected_moves
    assert upsampler.moves_at_call == [expected_moves[:1]]


def test_upscale_returns_the_upsampler_to_the_cpu_when_the_upsample_fails(monkeypatch, tmp_path):
    model = _load_upscale_model(monkeypatch, low_memory=True)
    model.upsample_pipeline.error = RuntimeError("out of memory")
    pipeline = model.text_to_video
    original = pipeline.scheduler

    with pytest.raises(RuntimeError, match="out of memory"):
        DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, **_text_to_video_args(tmp_path))

    assert model.upsample_pipeline.latent_upsampler.moves == ["cuda", "cpu"]
    assert len(pipeline.calls) == 1
    assert pipeline.scheduler is original


def test_upscale_runs_both_stages_on_the_same_lora_bearing_pipeline(monkeypatch, tmp_path):
    lora_path = tmp_path / "style.safetensors"
    lora_path.write_text("weights", encoding="utf-8")
    _patch_runtime(monkeypatch, _make_runtime())
    _patch_upsampler_io(monkeypatch)
    backend = DiffusersVideoBackend()
    model, _ = backend.load_model(
        "diffusers/LTX-2.3-Distilled-Diffusers",
        quantize=4,
        mode="t2v",
        low_memory=False,
        loras=[(str(lora_path), 0.5)],
        upscale=True,
        latent_upsampler=_UPSAMPLER_SOURCE,
    )

    backend.text_to_video(model=model, stage1_steps=4, **_text_to_video_args(tmp_path))

    (pipeline,) = _FakeTextPipeline.loaded_instances
    assert pipeline is model.text_to_video
    assert pipeline.loaded_loras == [(str(lora_path), "lora_0")]
    assert len(pipeline.calls) == 2
