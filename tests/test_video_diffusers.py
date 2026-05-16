"""Tests for the shared diffusers LTX video backend."""

from __future__ import annotations

import importlib
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

from PIL import Image
import pytest

import zvisiongenerator.backends.video_diffusers as video_diffusers_module
from zvisiongenerator.backends.video_diffusers import DiffusersVideoBackend, _build_generation_kwargs, _configure_torch_runtime, _validate_diffusers_version


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
        self.vae_slicing_enabled = False
        self.vae_tiling_enabled = False
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

    def enable_vae_slicing(self) -> None:
        self.vae_slicing_enabled = True

    def enable_vae_tiling(self) -> None:
        self.vae_tiling_enabled = True

    def to(self, *, device=None, dtype=None):
        self.to_calls.append((device, dtype))
        return self

    def load_lora_weights(self, path: str, adapter_name: str | None = None) -> None:
        self.loaded_loras.append((path, adapter_name))

    def set_adapters(self, adapter_names, adapter_weights=None) -> None:
        self.adapter_calls.append((adapter_names, adapter_weights))


class _FakeTextPipeline(_BaseFakePipeline):
    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        callback = kwargs.get("callback_on_step_end")
        total_steps = int(kwargs.get("num_inference_steps", 1))
        if callback is not None:
            callback(self, 0, None, {})
            callback(self, max(total_steps - 1, 0), None, {})
        return SimpleNamespace(frames=[["frame-1", "frame-2"]])


class _FakeImagePipeline(_BaseFakePipeline):
    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        callback = kwargs.get("callback_on_step_end")
        total_steps = int(kwargs.get("num_inference_steps", 1))
        if callback is not None:
            callback(self, 0, None, {})
            callback(self, max(total_steps - 1, 0), None, {})
        return SimpleNamespace(frames=[["image-frame-1"]])


class _FakeUpscalerPipeline(_BaseFakePipeline):
    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(frames=[["upscaled-frame"]])


class _NoScalePipeline(_FakeTextPipeline):
    def set_adapters(self, adapter_names, adapter_weights=None) -> None:  # noqa: ARG002
        raise AttributeError


@pytest.fixture(autouse=True)
def _clear_pipeline_state():
    for pipeline_class in (_FakeTextPipeline, _FakeImagePipeline, _FakeUpscalerPipeline, _NoScalePipeline):
        pipeline_class.loaded_instances = []


def _make_runtime(*, torch: _FakeTorch | None = None, text_pipeline=_FakeTextPipeline, image_pipeline=_FakeImagePipeline, upscaler_pipeline=_FakeUpscalerPipeline, version: str = "0.37.1"):
    exported: dict[str, object] = {}

    def _export(frames, output_video_path=None, fps=10):
        exported["frames"] = frames
        exported["output_video_path"] = output_video_path
        exported["fps"] = fps
        return output_video_path

    runtime = SimpleNamespace(
        torch=torch or _FakeTorch(),
        image_module=Image,
        export_to_video=_export,
        pipeline_classes=SimpleNamespace(
            text_to_video=text_pipeline,
            image_to_video=image_pipeline,
            latent_upscaler=upscaler_pipeline,
            family_name="LTX2",
        ),
        diffusers_version=version,
    )
    runtime.exported = exported
    return runtime


def test_load_model_rejects_missing_cuda(monkeypatch):
    backend = DiffusersVideoBackend()
    monkeypatch.setattr(
        "zvisiongenerator.backends.video_diffusers._load_runtime_dependencies",
        lambda: _make_runtime(torch=_FakeTorch(cuda_available=False)),
    )

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        backend.load_model("dg845/LTX-2.3-Diffusers", mode="t2v")


def test_load_model_prefers_runtime_classes_and_configures_pipeline(monkeypatch):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime()
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._load_runtime_dependencies", lambda: runtime)

    model, model_info = backend.load_model("dg845/LTX-2.3-Diffusers", mode="t2v", low_memory=True, upscale=True)

    assert model_info.family == "ltx"
    assert model.text_to_video is _FakeTextPipeline.loaded_instances[0]
    assert model.latent_upscaler is _FakeUpscalerPipeline.loaded_instances[0]
    assert model.text_to_video.cpu_offload_enabled is True
    assert model.text_to_video.vae_slicing_enabled is True
    assert model.text_to_video.vae_tiling_enabled is True


def test_text_to_video_exports_path_and_reports_progress(monkeypatch, tmp_path):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime()
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._load_runtime_dependencies", lambda: runtime)
    model, _ = backend.load_model("dg845/LTX-2.3-Diffusers", mode="t2v", low_memory=False)
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
    assert runtime.exported["frames"] == ["frame-1", "frame-2"]
    assert runtime.exported["output_video_path"] == str(output_path)
    assert runtime.exported["fps"] == 24
    call = model.text_to_video.calls[0]
    assert call["prompt"] == "a river"
    assert call["width"] == 704
    assert call["height"] == 448
    assert call["num_frames"] == 49
    assert call["num_inference_steps"] == 6
    assert call["frame_rate"] == 24
    assert call["generator"].seed == 123
    assert events[0] == {"phase": "video", "current_step": 1, "total_steps": 6}
    assert events[-1] == {"phase": "video", "current_step": 6, "total_steps": 6}
    assert runtime.torch.cuda.empty_cache_calls == 1


def test_build_generation_kwargs_keeps_input_kwargs_and_drops_torch() -> None:
    extra_kwargs = {"image": "image-bytes", "torch": _FakeTorch()}

    kwargs = _build_generation_kwargs(
        _FakeImagePipeline(),
        prompt="animate this",
        width=704,
        height=448,
        num_frames=49,
        steps=4,
        seed=99,
        fps=24,
        step_callback=None,
        phase="video",
        extra_kwargs=extra_kwargs,
    )

    assert kwargs["image"] == "image-bytes"
    assert kwargs["generator"].seed == 99
    assert "torch" not in kwargs
    assert extra_kwargs == {"image": "image-bytes", "torch": extra_kwargs["torch"]}


def test_image_to_video_uses_loaded_image(monkeypatch, tmp_path):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime()
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._load_runtime_dependencies", lambda: runtime)
    model, _ = backend.load_model("dg845/LTX-2.3-Diffusers", mode="i2v", low_memory=False)
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


def test_upscale_uses_latent_upscaler_and_requires_stage1_steps(monkeypatch, tmp_path):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime()
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._load_runtime_dependencies", lambda: runtime)
    model, _ = backend.load_model("dg845/LTX-2.3-Diffusers", mode="t2v", low_memory=False, upscale=True)
    events: list[dict[str, object]] = []
    output_path = tmp_path / "upscaled.mp4"

    with pytest.raises(ValueError, match="stage1_steps"):
        backend.text_to_video(
            model=model,
            prompt="a city",
            width=704,
            height=448,
            num_frames=49,
            seed=77,
            steps=8,
            output_path=str(output_path),
            step_callback=events.append,
        )

    result = backend.text_to_video(
        model=model,
        prompt="a city",
        width=704,
        height=448,
        num_frames=49,
        seed=77,
        steps=8,
        output_path=str(output_path),
        step_callback=events.append,
        stage1_steps=4,
    )

    assert result == output_path
    assert model.text_to_video.calls[0]["num_inference_steps"] == 4
    assert model.latent_upscaler.calls[0]["video"] == ["frame-1", "frame-2"]
    assert runtime.exported["frames"] == ["upscaled-frame"]
    assert events[-1] == {"phase": "video_upscale_stage_2", "current_step": 5, "total_steps": 5}


def test_load_model_fails_when_upscaler_api_is_missing(monkeypatch):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime(upscaler_pipeline=None)
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._load_runtime_dependencies", lambda: runtime)

    with pytest.raises(RuntimeError, match="latent upscaler"):
        backend.load_model("dg845/LTX-2.3-Diffusers", mode="t2v", upscale=True)


def test_load_model_applies_lora_weights(monkeypatch, tmp_path):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime()
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._load_runtime_dependencies", lambda: runtime)
    lora_path = tmp_path / "style.safetensors"
    lora_path.write_text("weights", encoding="utf-8")

    model, _ = backend.load_model("dg845/LTX-2.3-Diffusers", mode="t2v", loras=[(str(lora_path), 0.75)])

    assert model.text_to_video.loaded_loras == [(str(lora_path), "lora_0")]
    assert model.text_to_video.adapter_calls == [(["lora_0"], [0.75])]


def test_load_model_rejects_missing_or_unsupported_lora(monkeypatch, tmp_path):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime()
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._load_runtime_dependencies", lambda: runtime)

    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        backend.load_model("dg845/LTX-2.3-Diffusers", mode="t2v", loras=[(str(tmp_path / "missing.safetensors"), 1.0)])

    bad_lora = tmp_path / "bad.txt"
    bad_lora.write_text("weights", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported LoRA format"):
        backend.load_model("dg845/LTX-2.3-Diffusers", mode="t2v", loras=[(str(bad_lora), 1.0)])


def test_load_model_rejects_lora_scales_when_adapter_api_cannot_apply_them(monkeypatch, tmp_path):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime(text_pipeline=_NoScalePipeline)
    monkeypatch.setattr("zvisiongenerator.backends.video_diffusers._load_runtime_dependencies", lambda: runtime)
    lora_path = tmp_path / "style.safetensors"
    lora_path.write_text("weights", encoding="utf-8")

    with pytest.raises(RuntimeError, match="explicit LoRA scales"):
        backend.load_model("dg845/LTX-2.3-Diffusers", mode="t2v", loras=[(str(lora_path), 0.5)])


def test_load_model_rejects_old_diffusers_versions(monkeypatch):
    with pytest.raises(RuntimeError, match=r"diffusers>=0\.37\.1"):
        _validate_diffusers_version("0.36.0")


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

    backend.load_model("dg845/LTX-2.3-Diffusers", mode="t2v")

    validator.assert_not_called()
