"""Tests for the shared diffusers LTX video backend."""

from __future__ import annotations

import importlib
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from PIL import Image
import pytest

import zvisiongenerator.backends.video_diffusers as video_diffusers_module
from zvisiongenerator.backends.video_diffusers import (
    _DISTILLED_GUIDANCE,
    DiffusersVideoBackend,
    _build_generation_kwargs,
    _UPSAMPLER_CONFIG,
    _build_upsample_pipeline,
    _configure_torch_runtime,
    _load_latent_upsampler,
    _read_state_dict,
    _resolve_pipeline_classes,
    _resolve_upsampler_weights,
    _stage1_sigmas,
    _validate_diffusers_version,
)

_DISTILLED_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875)
_STAGE2_SIGMAS = (0.909375, 0.725, 0.421875)
_EXTRA_GUIDANCE_KWARGS = ("audio_guidance_scale", "stg_scale", "audio_stg_scale", "modality_scale", "audio_modality_scale")


class _FakeVae:
    def __init__(self):
        self.tiling_enabled = False

    def enable_tiling(self) -> None:
        self.tiling_enabled = True


class _FakeScheduler:
    """Stand-in for FlowMatchEulerDiscreteScheduler: from_config records the overrides it was built with."""

    def __init__(self, config: dict[str, Any] | None = None, overrides: dict[str, Any] | None = None):
        self.config = config or {"shift": 1.0, "use_dynamic_shifting": True, "shift_terminal": 0.1}
        self.overrides = overrides

    @classmethod
    def from_config(cls, config, **overrides):
        return cls(config, overrides)


class _FakeInferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeGenerator:
    def __init__(self, device: str):
        self.device = device
        self.seed: int | None = None

    def manual_seed(self, seed: int):
        self.seed = seed
        return self


class _FakeCuda:
    def __init__(self, *, available: bool = True, bf16: bool = True):
        self.available = available
        self.bf16 = bf16
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return self.available

    def is_bf16_supported(self) -> bool:
        return self.bf16

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakeTorch:
    def __init__(self, *, cuda_available: bool = True, bf16: bool = True, cuda_version: str | None = "12.6"):
        self.cuda = _FakeCuda(available=cuda_available, bf16=bf16)
        self.version = SimpleNamespace(cuda=cuda_version)
        self.backends = SimpleNamespace(cudnn=SimpleNamespace(benchmark=False))
        self.bfloat16 = "bfloat16"
        self.float16 = "float16"
        self.Generator = _FakeGenerator

    def inference_mode(self):
        return _FakeInferenceMode()

    def set_float32_matmul_precision(self, _value: str) -> None:
        return None


class _BaseFakePipeline:
    loaded_instances: list["_BaseFakePipeline"] = []

    def __init__(self):
        self.model_path: str | None = None
        self.load_kwargs: dict[str, object] = {}
        self.cpu_offload_enabled = False
        self.to_calls: list[tuple[object, object]] = []
        self.vae = _FakeVae()
        self.scheduler = _FakeScheduler()
        self.schedulers_at_call: list[object] = []
        self.loaded_loras: list[tuple[str, str | None]] = []
        self.adapter_calls: list[tuple[object, object]] = []
        self.calls: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        instance = cls()
        instance.model_path = model_path
        instance.load_kwargs = kwargs
        cls.loaded_instances.append(instance)
        return instance

    def enable_model_cpu_offload(self) -> None:
        self.cpu_offload_enabled = True

    def to(self, *, device=None, dtype=None):
        self.to_calls.append((device, dtype))
        return self

    def load_lora_weights(self, path: str, adapter_name: str | None = None) -> None:
        self.loaded_loras.append((path, adapter_name))

    def set_adapters(self, adapter_names, adapter_weights=None) -> None:
        self.adapter_calls.append((adapter_names, adapter_weights))


def _run_fake_denoise(pipeline: _BaseFakePipeline, kwargs: dict[str, Any], frames: list[list[str]]) -> SimpleNamespace:
    """Record one pipeline call, fire the step callback per step and return latents or frames by output type."""

    pipeline.calls.append(kwargs)
    pipeline.schedulers_at_call.append(pipeline.scheduler)
    callback = kwargs.get("callback_on_step_end")
    if callback is not None:
        for step in range(int(kwargs["num_inference_steps"])):
            callback(pipeline, step, None, {})
    if kwargs["output_type"] == "latent":
        return SimpleNamespace(frames="video-latent", audio="audio-latent")
    return SimpleNamespace(frames=frames)


class _FakeTextPipeline(_BaseFakePipeline):
    def __call__(self, **kwargs):
        return _run_fake_denoise(self, kwargs, [["frame-1", "frame-2"]])


class _FakeImagePipeline(_BaseFakePipeline):
    def __call__(self, **kwargs):
        return _run_fake_denoise(self, kwargs, [["image-frame-1"]])


class _FakeUpsamplePipeline(_BaseFakePipeline):
    """Stand-in for LTX2LatentUpsamplePipeline: built from a shared VAE and an upsampler, never from_pretrained."""

    def __init__(self, vae=None, latent_upsampler=None):
        super().__init__()
        self.vae = vae
        self.latent_upsampler = latent_upsampler
        self.error: Exception | None = None

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.latent_upsampler.moves_at_call.append(list(self.latent_upsampler.moves))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(frames="upsampled-latent")


class _FakeUpsamplerModel:
    """Stand-in for LTX2LatentUpsamplerModel recording its constructor kwargs, weights and placement."""

    instances: list["_FakeUpsamplerModel"] = []
    load_error: Exception | None = None

    def __init__(self, **config):
        self.config = config
        self.state_dict_calls: list[tuple[dict[str, Any], bool]] = []
        self.to_calls: list[dict[str, Any]] = []
        self.moves: list[str] = []
        self.moves_at_call: list[list[str]] = []
        self.eval_calls = 0
        _FakeUpsamplerModel.instances.append(self)

    def load_state_dict(self, state_dict, strict=True):
        if _FakeUpsamplerModel.load_error is not None:
            raise _FakeUpsamplerModel.load_error
        self.state_dict_calls.append((state_dict, strict))

    def to(self, *devices, **kwargs):
        self.moves.extend(devices)
        if kwargs:
            self.to_calls.append(kwargs)
        return self

    def eval(self):
        self.eval_calls += 1
        return self


class _FakeWaveform:
    """Stand-in for a vocoder waveform tensor; records the conversions applied before encoding."""

    def __init__(self, name: str):
        self.name = name
        self.conversions: list[str] = []

    def float(self):
        self.conversions.append("float")
        return self

    def cpu(self):
        self.conversions.append("cpu")
        return self


class _AudioTextPipeline(_FakeTextPipeline):
    """Text pipeline whose decoded output carries a batch of waveforms, like LTX2Pipeline with its vocoder."""

    def __init__(self, sample_rate: int | None = 24000):
        super().__init__()
        self.vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=sample_rate))

    def __call__(self, **kwargs):
        result = _run_fake_denoise(self, kwargs, [["frame-1", "frame-2"]])
        if kwargs["output_type"] != "latent":
            result.audio = [_FakeWaveform("clip-0"), _FakeWaveform("clip-1")]
        return result


class _NoRatePipeline(_AudioTextPipeline):
    def __init__(self):
        super().__init__(sample_rate=None)


class _NoScalePipeline(_FakeTextPipeline):
    def set_adapters(self, adapter_names, adapter_weights=None) -> None:  # noqa: ARG002
        raise AttributeError


@pytest.fixture(autouse=True)
def _clear_pipeline_state():
    for pipeline_class in (_FakeTextPipeline, _FakeImagePipeline, _NoScalePipeline, _AudioTextPipeline, _NoRatePipeline):
        pipeline_class.loaded_instances = []
    _FakeUpsamplerModel.instances = []
    _FakeUpsamplerModel.load_error = None


def _make_runtime(
    *,
    torch: _FakeTorch | None = None,
    text_pipeline=_FakeTextPipeline,
    image_pipeline=_FakeImagePipeline,
    upsample_pipeline=_FakeUpsamplePipeline,
    upsampler_model=_FakeUpsamplerModel,
    version: str = "0.38.0",
):
    encoded: dict[str, object] = {}

    def _encode(video, fps, output_path, audio=None, audio_sample_rate=None):
        encoded.update(frames=video, fps=fps, output_path=output_path, audio=audio, audio_sample_rate=audio_sample_rate)

    runtime = SimpleNamespace(
        torch=torch or _FakeTorch(),
        image_module=Image,
        encode_video=_encode,
        pipeline_classes=SimpleNamespace(
            text_to_video=text_pipeline,
            image_to_video=image_pipeline,
            latent_upsample_pipeline=upsample_pipeline,
            latent_upsampler_model=upsampler_model,
        ),
        diffusers_version=version,
        distilled_sigmas=_DISTILLED_SIGMAS,
        stage2_sigmas=_STAGE2_SIGMAS,
    )
    runtime.encoded = encoded
    return runtime


_UPSAMPLER_SOURCE = ("example-org/example-upsampler", "example-x2.safetensors")
_UPSAMPLER_STATE = {"initial_conv.weight": "tensor"}


def _patch_runtime(monkeypatch, runtime):
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._load_runtime_dependencies", lambda: runtime)


def _patch_upsampler_io(monkeypatch, events: list[str] | None = None):
    """Replace the weight download and read with fakes; return the recorded download requests."""

    downloads: list[tuple[str, str]] = []

    def _download(repo_id, filename):
        downloads.append((repo_id, filename))
        if events is not None:
            events.append("download")
        return "/fake/upsampler.safetensors"

    def _read(path):
        assert path == "/fake/upsampler.safetensors"
        return dict(_UPSAMPLER_STATE)

    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._resolve_upsampler_weights", _download)
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._read_state_dict", _read)
    return downloads


def _load_upscale_model(monkeypatch, *, low_memory, mode="t2v", runtime=None):
    _patch_runtime(monkeypatch, runtime or _make_runtime())
    _patch_upsampler_io(monkeypatch)
    model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode=mode, low_memory=low_memory, upscale=True, latent_upsampler=_UPSAMPLER_SOURCE)
    return model


def test_load_model_rejects_missing_cuda(monkeypatch):
    backend = DiffusersVideoBackend()
    monkeypatch.setattr(
        "zvisiongenerator.backends.video_diffusers._load_runtime_dependencies",
        lambda: _make_runtime(torch=_FakeTorch(cuda_available=False)),
    )

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v")


@pytest.mark.parametrize(("cuda_version", "expects_hint"), [("13.0", True), ("12.6", False), (None, False)])
def test_missing_cuda_error_asks_for_a_driver_update_only_for_cuda_13_builds(monkeypatch, cuda_version, expects_hint):
    from zvisiongenerator.backends.cuda_driver import cuda_driver_hint

    backend = DiffusersVideoBackend()
    monkeypatch.setattr(
        "zvisiongenerator.backends.video_diffusers._load_runtime_dependencies",
        lambda: _make_runtime(torch=_FakeTorch(cuda_available=False, cuda_version=cuda_version)),
    )

    with pytest.raises(RuntimeError, match="CUDA is not available") as excinfo:
        backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v")

    assert (cuda_driver_hint(False, "13.0") in str(excinfo.value)) is expects_hint


def test_load_model_prefers_runtime_classes_and_configures_pipeline(monkeypatch):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime()
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._load_runtime_dependencies", lambda: runtime)
    _patch_upsampler_io(monkeypatch)

    model, model_info = backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", low_memory=True, upscale=True, latent_upsampler=_UPSAMPLER_SOURCE)

    assert model_info.family == "ltx"
    assert model.text_to_video is _FakeTextPipeline.loaded_instances[0]
    assert isinstance(model.upsample_pipeline, _FakeUpsamplePipeline)
    assert model.text_to_video.cpu_offload_enabled is True


@pytest.mark.parametrize("low_memory", [True, False])
def test_load_model_enables_tiling_on_the_vae(monkeypatch, low_memory):
    runtime = _make_runtime()
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._load_runtime_dependencies", lambda: runtime)

    model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", low_memory=low_memory)

    pipeline = model.text_to_video
    assert pipeline.vae.tiling_enabled is True
    assert pipeline.cpu_offload_enabled is low_memory
    assert bool(pipeline.to_calls) is not low_memory


def test_load_model_without_a_vae_still_loads(monkeypatch):
    class _NoVaePipeline(_FakeTextPipeline):
        def __init__(self):
            super().__init__()
            del self.vae

    runtime = _make_runtime(text_pipeline=_NoVaePipeline)
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._load_runtime_dependencies", lambda: runtime)

    model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", low_memory=False)

    assert not hasattr(model.text_to_video, "vae")


def test_text_to_video_exports_path_and_reports_progress(monkeypatch, tmp_path):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime()
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._load_runtime_dependencies", lambda: runtime)
    model, _ = backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", low_memory=False)
    events: list[dict[str, object]] = []
    output_path = tmp_path / "video.mp4"

    result = backend.text_to_video(
        model=model,
        prompt="a river",
        width=704,
        height=448,
        num_frames=49,
        seed=123,
        steps=6,
        output_path=str(output_path),
        step_callback=events.append,
    )

    assert result == output_path
    assert runtime.encoded["frames"] == ["frame-1", "frame-2"]
    assert runtime.encoded["output_path"] == str(output_path)
    assert runtime.encoded["fps"] == 24
    call = model.text_to_video.calls[0]
    assert call["prompt"] == "a river"
    assert call["width"] == 704
    assert call["height"] == 448
    assert call["num_frames"] == 49
    assert call["num_inference_steps"] == 6
    assert call["sigmas"] == _stage1_sigmas(_DISTILLED_SIGMAS, 6)
    assert call["frame_rate"] == 24
    assert call["generator"].seed == 123
    assert events == [{"phase": "video", "current_step": step, "total_steps": 6} for step in range(1, 7)]
    assert runtime.torch.cuda.empty_cache_calls == 1


def _run_text_to_video(monkeypatch, tmp_path, *, steps, step_callback=None):
    runtime = _make_runtime()
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._load_runtime_dependencies", lambda: runtime)
    backend = DiffusersVideoBackend()
    model, _ = backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", low_memory=False)
    backend.text_to_video(
        model=model,
        prompt="a river",
        width=704,
        height=448,
        num_frames=49,
        seed=1,
        steps=steps,
        output_path=str(tmp_path / "video.mp4"),
        step_callback=step_callback,
    )
    return model.text_to_video.calls[0]


def test_text_to_video_passes_distilled_sigmas_and_cfg_one(monkeypatch, tmp_path):
    call = _run_text_to_video(monkeypatch, tmp_path, steps=8)

    assert call["sigmas"] == list(_DISTILLED_SIGMAS)
    assert call["num_inference_steps"] == len(call["sigmas"])
    assert call["guidance_scale"] == 1.0
    assert not set(_EXTRA_GUIDANCE_KWARGS) & set(call)


def test_distilled_guidance_is_guidance_scale_only():
    assert dict(_DISTILLED_GUIDANCE) == {"guidance_scale": 1.0}


def test_short_steps_report_len_sigmas_progress(monkeypatch, tmp_path):
    events: list[dict[str, object]] = []

    call = _run_text_to_video(monkeypatch, tmp_path, steps=4, step_callback=events.append)

    assert call["sigmas"] == _stage1_sigmas(_DISTILLED_SIGMAS, 4)
    assert events == [{"phase": "video", "current_step": step, "total_steps": 4} for step in range(1, 5)]


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


def test_build_generation_kwargs_keeps_input_kwargs() -> None:
    extra_kwargs = {"image": "image-bytes"}
    generator = _FakeGenerator("cpu")
    callback = MagicMock()

    kwargs = _build_generation_kwargs(
        prompt="animate this",
        width=704,
        height=448,
        num_frames=49,
        fps=24,
        sigmas=(1.0, 0.5),
        generator=generator,
        output_type="pil",
        callback=callback,
        extra_kwargs=extra_kwargs,
    )

    assert kwargs["image"] == "image-bytes"
    assert kwargs["generator"] is generator
    assert kwargs["callback_on_step_end"] is callback
    assert kwargs["sigmas"] == [1.0, 0.5]
    assert kwargs["num_inference_steps"] == 2
    assert kwargs["guidance_scale"] == 1.0
    assert kwargs["frame_rate"] == 24
    assert kwargs["output_type"] == "pil"
    assert extra_kwargs == {"image": "image-bytes"}


def test_image_to_video_uses_loaded_image(monkeypatch, tmp_path):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime()
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._load_runtime_dependencies", lambda: runtime)
    model, _ = backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="i2v", low_memory=False)
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
    assert call["prompt"] == "animate this"
    assert call["image"].size == (8, 8)
    assert call["sigmas"] == _stage1_sigmas(_DISTILLED_SIGMAS, 4)
    assert call["num_inference_steps"] == 4
    assert call["guidance_scale"] == 1.0
    assert not set(_EXTRA_GUIDANCE_KWARGS) & set(call)


def _text_to_video_args(tmp_path, **overrides):
    return {"prompt": "a city", "width": 512, "height": 384, "num_frames": 49, "seed": 77, "steps": 8, "output_path": str(tmp_path / "upscaled.mp4"), **overrides}


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
    plain, _ = backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", low_memory=False)
    with pytest.raises(RuntimeError, match="upsampler"):
        backend.text_to_video(model=plain, stage1_steps=4, **_text_to_video_args(tmp_path))
    assert plain.text_to_video.calls == []


def test_upscale_t2v_runs_half_size_stage1_then_full_size_refine(monkeypatch, tmp_path):
    model, _ = _run_upscaled_text_to_video(monkeypatch, tmp_path, stage1_steps=4)

    stage1, stage2 = model.text_to_video.calls
    assert (stage1["width"], stage1["height"]) == (256, 192)
    assert stage1["output_type"] == "latent"
    assert stage1["sigmas"] == _stage1_sigmas(_DISTILLED_SIGMAS, 4)
    assert stage1["num_inference_steps"] == 4
    assert stage1["guidance_scale"] == 1.0
    assert "latents" not in stage1
    assert (stage2["width"], stage2["height"]) == (512, 384)
    assert stage2["output_type"] == "pil"
    assert stage2["sigmas"] == list(_STAGE2_SIGMAS)
    assert stage2["num_inference_steps"] == 3
    assert stage2["guidance_scale"] == 1.0
    assert not set(_EXTRA_GUIDANCE_KWARGS) & (set(stage1) | set(stage2))
    assert stage2["num_frames"] == stage1["num_frames"] == 49
    assert stage2["generator"] is stage1["generator"]
    assert stage1["generator"].seed == 77


def test_upscale_passes_stage1_audio_latents_and_noise_scale_to_stage2(monkeypatch, tmp_path):
    model, _ = _run_upscaled_text_to_video(monkeypatch, tmp_path, stage1_steps=4)

    stage2 = model.text_to_video.calls[1]
    assert stage2["latents"] == "upsampled-latent"
    assert stage2["audio_latents"] == "audio-latent"
    assert stage2["noise_scale"] == _STAGE2_SIGMAS[0]


def test_upscale_upsampler_gets_half_size_and_stage1_latents(monkeypatch, tmp_path):
    model, _ = _run_upscaled_text_to_video(monkeypatch, tmp_path, stage1_steps=4)

    (call,) = model.upsample_pipeline.calls
    assert call == {"latents": "video-latent", "height": 192, "width": 256, "num_frames": 49, "output_type": "latent"}


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
    assert stage1["image"].size == (8, 8)
    assert (stage2["width"], stage2["height"]) == (512, 384)
    assert stage2["latents"] == "upsampled-latent"
    assert stage2["audio_latents"] == "audio-latent"
    assert stage2["output_type"] == "pil"
    assert model.upsample_pipeline.vae is pipeline.vae
    assert runtime.encoded["frames"] == ["image-frame-1"]


@pytest.mark.parametrize(("width", "height"), [(520, 384), (512, 392)])
def test_upscale_rejects_sizes_not_divisible_by_64(monkeypatch, tmp_path, width, height):
    model = _load_upscale_model(monkeypatch, low_memory=False)

    with pytest.raises(ValueError, match="divisible by 64"):
        DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, **_text_to_video_args(tmp_path, width=width, height=height))

    assert model.text_to_video.calls == []
    assert model.upsample_pipeline.calls == []


def test_upscale_exports_stage2_frames(monkeypatch, tmp_path):
    runtime = _make_runtime()
    model = _load_upscale_model(monkeypatch, low_memory=False, runtime=runtime)

    result = DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, **_text_to_video_args(tmp_path))

    assert result == tmp_path / "upscaled.mp4"
    assert runtime.encoded["frames"] == ["frame-1", "frame-2"]
    assert runtime.encoded["fps"] == 24
    assert runtime.torch.cuda.empty_cache_calls == 1


def test_text_to_video_muxes_the_first_waveform_at_the_vocoder_rate(monkeypatch, tmp_path):
    runtime = _make_runtime(text_pipeline=_AudioTextPipeline)
    _patch_runtime(monkeypatch, runtime)
    backend = DiffusersVideoBackend()
    model, _ = backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", low_memory=False)

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
    model, _ = backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", low_memory=False)

    with pytest.warns(UserWarning, match="without sound"):
        backend.text_to_video(model=model, **_text_to_video_args(tmp_path))

    assert runtime.encoded["frames"] == ["frame-1", "frame-2"]
    assert runtime.encoded["audio"] is None
    assert runtime.encoded["audio_sample_rate"] is None


def test_output_without_audio_is_encoded_as_frames_only(monkeypatch, tmp_path):
    runtime = _make_runtime()
    _patch_runtime(monkeypatch, runtime)
    backend = DiffusersVideoBackend()
    model, _ = backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", low_memory=False)

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


def test_resolve_pipeline_classes_requires_ltx2():
    ltx2 = SimpleNamespace(LTX2Pipeline=object, LTX2ImageToVideoPipeline=object, LTX2LatentUpsamplePipeline=object)

    classes = _resolve_pipeline_classes(ltx2, _FakeUpsamplerModel)

    assert classes.text_to_video is object
    assert classes.image_to_video is object
    assert classes.latent_upsample_pipeline is object
    assert classes.latent_upsampler_model is _FakeUpsamplerModel
    for missing in ("LTX2Pipeline", "LTX2ImageToVideoPipeline", "LTX2LatentUpsamplePipeline"):
        partial = SimpleNamespace(**{name: object for name in vars(ltx2) if name != missing})
        with pytest.raises(RuntimeError, match="LTX-2"):
            _resolve_pipeline_classes(partial, _FakeUpsamplerModel)
    with pytest.raises(RuntimeError, match="LTX-2"):
        _resolve_pipeline_classes(SimpleNamespace(LTXPipeline=object, LTXImageToVideoPipeline=object), _FakeUpsamplerModel)


def test_upscale_load_downloads_upsampler_before_the_pipeline(monkeypatch):
    events: list[str] = []

    class _OrderedPipeline(_FakeTextPipeline):
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            events.append("from_pretrained")
            return super().from_pretrained(model_path, **kwargs)

    runtime = _make_runtime(text_pipeline=_OrderedPipeline)
    _patch_runtime(monkeypatch, runtime)
    downloads = _patch_upsampler_io(monkeypatch, events)

    DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", upscale=True, latent_upsampler=_UPSAMPLER_SOURCE)

    assert events == ["download", "from_pretrained"]
    assert downloads == [_UPSAMPLER_SOURCE]


def test_failed_upsampler_download_stops_before_the_pipeline_loads(monkeypatch):
    _patch_runtime(monkeypatch, _make_runtime())

    def _fail(repo_id, filename):
        raise RuntimeError(f"Failed to download the LTX latent upsampler '{filename}' from '{repo_id}': offline")

    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._resolve_upsampler_weights", _fail)

    with pytest.raises(RuntimeError, match="latent upsampler"):
        DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", upscale=True, latent_upsampler=_UPSAMPLER_SOURCE)

    assert _FakeTextPipeline.loaded_instances == []


def test_upscale_load_builds_upsampler_with_x2_config_and_strict_weights(monkeypatch):
    _load_upscale_model(monkeypatch, low_memory=False)

    (upsampler,) = _FakeUpsamplerModel.instances
    assert upsampler.config == dict(_UPSAMPLER_CONFIG)
    assert upsampler.config["use_rational_resampler"] is False
    assert upsampler.state_dict_calls == [(_UPSAMPLER_STATE, True)]
    assert upsampler.eval_calls == 1


@pytest.mark.parametrize(("low_memory", "expected_to"), [(False, {"device": "cuda", "dtype": "bfloat16"}), (True, {"dtype": "bfloat16"})])
def test_upsampler_device_follows_low_memory(monkeypatch, low_memory, expected_to):
    model = _load_upscale_model(monkeypatch, low_memory=low_memory)

    assert model.upsample_pipeline.latent_upsampler.to_calls == [expected_to]


@pytest.mark.parametrize("mode", ["t2v", "i2v"])
@pytest.mark.parametrize("low_memory", [False, True])
def test_upsample_pipeline_shares_the_generation_vae(monkeypatch, mode, low_memory):
    model = _load_upscale_model(monkeypatch, low_memory=low_memory, mode=mode)

    generation = model.text_to_video or model.image_to_video
    upsample = model.upsample_pipeline
    assert upsample.vae is generation.vae
    assert upsample.to_calls == []
    assert upsample.cpu_offload_enabled is False


def test_upscale_load_requires_upsampler_source(monkeypatch):
    _patch_runtime(monkeypatch, _make_runtime())
    downloads = _patch_upsampler_io(monkeypatch)

    with pytest.raises(ValueError, match="latent_upsampler"):
        DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", upscale=True)

    assert downloads == []
    assert _FakeTextPipeline.loaded_instances == []


def test_upsampler_weight_mismatch_is_reported(monkeypatch):
    _FakeUpsamplerModel.load_error = RuntimeError("Missing key(s) in state_dict")

    with pytest.raises(RuntimeError, match="latent_upsampler") as excinfo:
        _load_upscale_model(monkeypatch, low_memory=False)

    assert "Missing key(s)" in str(excinfo.value)


def test_non_upscale_load_never_touches_the_upsampler(monkeypatch):
    _patch_runtime(monkeypatch, _make_runtime())
    downloads = _patch_upsampler_io(monkeypatch)

    model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", latent_upsampler=_UPSAMPLER_SOURCE)

    assert model.upsample_pipeline is None
    assert downloads == []
    assert _FakeUpsamplerModel.instances == []


def test_load_latent_upsampler_helper_returns_the_eval_model():
    model = _load_latent_upsampler(_FakeUpsamplerModel, _UPSAMPLER_STATE, torch_dtype="float16", low_memory=True)

    assert isinstance(model, _FakeUpsamplerModel)
    assert model.to_calls == [{"dtype": "float16"}]


def test_build_upsample_pipeline_passes_the_vae_and_upsampler_only():
    generation = SimpleNamespace(vae=_FakeVae())
    upsampler = _FakeUpsamplerModel()

    pipeline = _build_upsample_pipeline(_FakeUpsamplePipeline, generation, upsampler)

    assert pipeline.vae is generation.vae
    assert pipeline.latent_upsampler is upsampler


def test_resolve_upsampler_weights_downloads_the_named_file(monkeypatch):
    import huggingface_hub

    calls: list[dict[str, str]] = []

    def _fake_download(**kwargs):
        calls.append(kwargs)
        return "/cache/example-x2.safetensors"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)

    assert _resolve_upsampler_weights(*_UPSAMPLER_SOURCE) == "/cache/example-x2.safetensors"
    assert calls == [{"repo_id": _UPSAMPLER_SOURCE[0], "filename": _UPSAMPLER_SOURCE[1]}]


def test_resolve_upsampler_weights_reports_download_failures(monkeypatch):
    import huggingface_hub

    def _fail(**_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fail)

    with pytest.raises(RuntimeError, match="offline") as excinfo:
        _resolve_upsampler_weights(*_UPSAMPLER_SOURCE)

    assert _UPSAMPLER_SOURCE[0] in str(excinfo.value)
    assert _UPSAMPLER_SOURCE[1] in str(excinfo.value)


def test_read_state_dict_loads_the_safetensors_file(monkeypatch):
    import safetensors.torch

    monkeypatch.setattr(safetensors.torch, "load_file", lambda path: {"path": path})

    assert _read_state_dict("/cache/example-x2.safetensors") == {"path": "/cache/example-x2.safetensors"}


def test_load_model_applies_lora_weights(monkeypatch, tmp_path):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime()
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._load_runtime_dependencies", lambda: runtime)
    lora_path = tmp_path / "style.safetensors"
    lora_path.write_text("weights", encoding="utf-8")

    model, _ = backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", loras=[(str(lora_path), 0.75)])

    assert model.text_to_video.loaded_loras == [(str(lora_path), "lora_0")]
    assert model.text_to_video.adapter_calls == [(["lora_0"], [0.75])]


class _LoraNoMatchPipeline(_FakeTextPipeline):
    """A pipeline whose LoRA loads raise *error* for the paths in *unmatched* and drop its offload hook when they do."""

    unmatched: tuple[str, ...] = ()
    error: type[Exception] = ValueError

    def __init__(self):
        super().__init__()
        self.transformer = SimpleNamespace(_hf_hook=object())

    def load_lora_weights(self, path: str, adapter_name: str | None = None) -> None:
        self.transformer._hf_hook = None
        if path in self.unmatched:
            raise self.error("no module matches")
        super().load_lora_weights(path, adapter_name=adapter_name)


def _write_loras(tmp_path, *names):
    paths = []
    for name in names:
        path = tmp_path / name
        path.write_text("weights", encoding="utf-8")
        paths.append(str(path))
    return paths


def test_a_lora_matching_no_layer_is_skipped_with_a_warning(monkeypatch, tmp_path, fake_peft_no_match):
    first, second = _write_loras(tmp_path, "other-model.safetensors", "style.safetensors")
    _LoraNoMatchPipeline.unmatched = (first,)
    _LoraNoMatchPipeline.error = fake_peft_no_match
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._load_runtime_dependencies", lambda: _make_runtime(text_pipeline=_LoraNoMatchPipeline))

    with pytest.warns(UserWarning, match="other-model.safetensors: none of its LoRA tensors match"):
        model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", loras=[(first, 1.0), (second, 0.5)])

    assert model.text_to_video.loaded_loras == [(second, "lora_1")]
    assert model.text_to_video.adapter_calls == [(["lora_1"], [0.5])]


@pytest.mark.parametrize(("low_memory", "restored"), [(True, True), (False, False)])
def test_offload_hooks_are_restored_after_loras_load_only_in_low_memory_mode(monkeypatch, tmp_path, fake_peft_no_match, low_memory, restored):
    (path,) = _write_loras(tmp_path, "other-model.safetensors")
    _LoraNoMatchPipeline.unmatched = (path,)
    _LoraNoMatchPipeline.error = fake_peft_no_match
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._load_runtime_dependencies", lambda: _make_runtime(text_pipeline=_LoraNoMatchPipeline))
    enable_calls: list[bool] = []
    monkeypatch.setattr(_LoraNoMatchPipeline, "enable_model_cpu_offload", lambda self: enable_calls.append(True))

    with pytest.warns(UserWarning):
        DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", low_memory=low_memory, loras=[(path, 1.0)])

    # low_memory enables offload once while configuring the pipeline; the restore adds a second call.
    assert len(enable_calls) == (2 if restored else 0)


def test_other_lora_load_errors_still_fail_the_load(monkeypatch, tmp_path, fake_peft_no_match):
    (path,) = _write_loras(tmp_path, "bad.safetensors")
    _LoraNoMatchPipeline.unmatched = (path,)
    _LoraNoMatchPipeline.error = ValueError
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._load_runtime_dependencies", lambda: _make_runtime(text_pipeline=_LoraNoMatchPipeline))

    with pytest.raises(RuntimeError, match="Failed to load LoRA"):
        DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", loras=[(path, 1.0)])


def test_load_model_rejects_missing_or_unsupported_lora(monkeypatch, tmp_path):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime()
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._load_runtime_dependencies", lambda: runtime)

    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", loras=[(str(tmp_path / "missing.safetensors"), 1.0)])

    bad_lora = tmp_path / "bad.txt"
    bad_lora.write_text("weights", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported LoRA format"):
        backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", loras=[(str(bad_lora), 1.0)])


def test_load_model_rejects_lora_scales_when_adapter_api_cannot_apply_them(monkeypatch, tmp_path):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime(text_pipeline=_NoScalePipeline)
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._load_runtime_dependencies", lambda: runtime)
    lora_path = tmp_path / "style.safetensors"
    lora_path.write_text("weights", encoding="utf-8")

    with pytest.raises(RuntimeError, match="explicit LoRA scales"):
        backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", loras=[(str(lora_path), 0.5)])


def test_load_model_rejects_old_diffusers_versions(monkeypatch):
    with pytest.raises(RuntimeError, match=r"diffusers>=0\.38\.0"):
        _validate_diffusers_version("0.37.1")


def test_import_does_not_set_cuda_alloc_conf(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)

    importlib.reload(video_diffusers_module)

    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ


def test_configure_torch_runtime_sets_cuda_alloc_default(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)

    _configure_torch_runtime(_FakeTorch())

    assert "expandable_segments:True" in __import__("os").environ["PYTORCH_CUDA_ALLOC_CONF"]


def test_configure_torch_runtime_preserves_existing_cuda_alloc_conf(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "custom:value")

    _configure_torch_runtime(_FakeTorch())

    assert __import__("os").environ["PYTORCH_CUDA_ALLOC_CONF"] == "custom:value"


def test_load_model_does_not_validate_diffusers_version_twice(monkeypatch):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime(version="0.36.0")
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._load_runtime_dependencies", lambda: runtime)
    validator = MagicMock(side_effect=AssertionError("duplicate validation"))
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._validate_diffusers_version", validator)

    backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v")

    validator.assert_not_called()
