"""Tests for the shared diffusers LTX video backend."""

from __future__ import annotations

import importlib
import os
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
from PIL import Image
import pytest
from conftest import make_extend_source

import zvisiongenerator.backends.cuda.video.backend as video_diffusers_module
import zvisiongenerator.backends.cuda.video.sampler as _sampler_module
from zvisiongenerator.backends.cuda.video.backend import (
    _DISTILLED_GUIDANCE,
    DiffusersVideoBackend,
    _build_generation_kwargs,
    _UPSAMPLER_CONFIG,
    _absent_optional_components,
    _build_upsample_pipeline,
    _pass_scheduler,
    _repo_dir,
    _configure_torch_runtime,
    _load_latent_upsampler,
    _read_state_dict,
    _resolve_pipeline_classes,
    _resolve_upsampler_weights,
    _stage1_sigmas,
    _TextEncoderSource,
    _validate_diffusers_version,
)

# The originals, which the fake_weights fixture replaces on the module.
from zvisiongenerator.backends.cuda.video.backend import _check_system_memory as real_check_system_memory
from zvisiongenerator.backends.cuda.video.memory_estimate import ltx_system_estimate as real_ltx_system_estimate
from zvisiongenerator.backends.cuda.video.backend import _download_hub_file as real_download_hub_file
from zvisiongenerator.backends.cuda.video.backend import _encode_prompt as real_encode_prompt
from zvisiongenerator.backends.cuda.video.backend import _place_components as real_place_components
from zvisiongenerator.backends.cuda.video.backend import _prepared_weights as real_prepared_weights
from zvisiongenerator.backends.cuda.video.backend import _place_transformer as real_place_transformer

_PIN = "0123456789abcdef0123456789abcdef01234567"
_DISTILLED_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875)
_STAGE2_SIGMAS = (0.909375, 0.725, 0.421875)
# Every guidance term at its neutral value: no classifier-free, spatio-temporal or modality guidance.
_NEUTRAL_GUIDANCE = {"guidance_scale": 1.0, "audio_guidance_scale": 1.0, "stg_scale": 0.0, "audio_stg_scale": 0.0, "modality_scale": 1.0, "audio_modality_scale": 1.0}


def _guidance(call):
    return {key: call.get(key) for key in _NEUTRAL_GUIDANCE}


class _FakeLatent(str):
    """A latent stand-in that survives dtype moves (``.to``)."""

    def to(self, *_args, **_kwargs):
        return self


class _FakeVae:
    def __init__(self):
        self.tiling_enabled = False
        self.decoded_frames: list[str] = []  # what the fake streaming decode returns for this pipeline
        self.decoded_latents: list[str] = []

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

    def device(self, name: str) -> str:
        return f"device:{name}"

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
        self.text_encoder = None
        self.encoded: list[dict[str, object]] = []

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

    def encode_prompt(self, *, prompt, **kwargs):
        self.encoded.append({"prompt": prompt, "text_encoder": self.text_encoder, **kwargs})
        return f"embeds:{prompt}", f"mask:{prompt}", "embeds:", "mask:"

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
    pipeline.vae.decoded_frames = frames[0]
    if kwargs["output_type"] == "latent":
        return SimpleNamespace(frames=_FakeLatent("video-latent"), audio=_FakeLatent("audio-latent"))
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


class _FakeAudioVae:
    dtype = "bfloat16"

    def __init__(self):
        self.decoded: list[str] = []

    def decode(self, latents, return_dict=True):
        self.decoded.append(latents)
        return (f"mel:{latents}",)


class _FakeVocoder:
    """Turns a mel spectrogram into a batch of waveforms, like LTX2VocoderWithBWE."""

    def __init__(self, sample_rate: int | None):
        self.config = SimpleNamespace(output_sampling_rate=sample_rate)
        self.mels: list[str] = []

    def __call__(self, mel):
        self.mels.append(mel)
        return [_FakeWaveform("clip-0"), _FakeWaveform("clip-1")]


class _AudioTextPipeline(_FakeTextPipeline):
    """Text pipeline with an audio VAE and a vocoder, like LTX2Pipeline."""

    def __init__(self, sample_rate: int | None = 24000):
        super().__init__()
        self.audio_vae = _FakeAudioVae()
        self.vocoder = _FakeVocoder(sample_rate)


class _NoRatePipeline(_AudioTextPipeline):
    def __init__(self):
        super().__init__(sample_rate=None)


class _FakeVideoCondition:
    """Stand-in for LTX2VideoCondition: records the frames, latent index, strength and CRF it was built with."""

    def __init__(self, *, frames, index, strength, crf=None):
        self.frames = frames
        self.index = index
        self.strength = strength
        self.crf = crf


class _NumpyWaveform:
    """A vocoder waveform ``(channels, samples)`` that converts to numpy like a torch tensor."""

    def __init__(self, samples: np.ndarray):
        self.samples = samples

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.samples


class _WindowVocoder(_FakeVocoder):
    """Vocoder returning a ramp of ``samples`` per channel, so a cut's position in the window can be read back."""

    def __init__(self, samples: int):
        super().__init__(48000)
        self.samples = samples

    def __call__(self, mel):
        self.mels.append(mel)
        ramp = np.linspace(-0.5, 0.5, self.samples, dtype=np.float32)
        return [_NumpyWaveform(np.stack([ramp, ramp]))]


class _FakeConditionPipeline(_BaseFakePipeline):
    """Condition pipeline that decodes to ``window`` numbered frames, with audio at 48 kHz, like LTX2ConditionPipeline."""

    window = 25 + 16
    audio_sampling_rate = 16000
    audio_hop_length = 160
    audio_vae_temporal_compression_ratio = 4

    def __init__(self):
        super().__init__()
        self.audio_vae = _FakeAudioVae()
        self.audio_vae.config = SimpleNamespace(sample_rate=16000)
        self.vocoder = _WindowVocoder(round(self.window / 24 * 48000))
        self.transformer_at_call: list[object] = []
        self.keyframe_pixels: list[int] = []

    def _prepare_keyframe_coords(self, **kwargs):
        return kwargs["pixel_frame_idx"]

    def __call__(self, **kwargs):
        self.transformer_at_call.append(self.transformer)
        # As diffusers places an appended (non-zero index) keyframe: latent index -> pixel (index - 1) * 8 + 1.
        latent_frames = (kwargs["num_frames"] - 1) // 8 + 1
        for condition in kwargs.get("conditions", []):
            if condition.index % latent_frames:
                self.keyframe_pixels.append(self._prepare_keyframe_coords(pixel_frame_idx=(condition.index % latent_frames - 1) * 8 + 1, fps=kwargs["frame_rate"]))
        return _run_fake_denoise(self, kwargs, [[np.full((4, 4, 3), index, dtype=np.uint8) for index in range(self.window)]])


class _NoScalePipeline(_FakeTextPipeline):
    def set_adapters(self, adapter_names, adapter_weights=None) -> None:  # noqa: ARG002
        raise AttributeError


@pytest.fixture(autouse=True)
def _clear_pipeline_state():
    for pipeline_class in (_FakeTextPipeline, _FakeImagePipeline, _FakeConditionPipeline, _NoScalePipeline, _AudioTextPipeline, _NoRatePipeline):
        pipeline_class.loaded_instances = []
    _FakeUpsamplerModel.instances = []
    _FakeUpsamplerModel.load_error = None


@pytest.fixture(autouse=True)
def fake_weights(monkeypatch):
    """Keep load_model off the network and disk: fake repo folder, checkpoint, memory check, weights and placement."""

    calls = SimpleNamespace(
        placements=[],
        prepared=[],
        prepared_names=[],
        memory_checks=[],
        repo_dirs=[],
        transformers=[],
        text_encoders=[],
        released=0,
        offloaded=[],
        events=[],
        fits_together=True,
        lazy=[],
        unload_before=[],
        run_wise=[],
    )

    def _prepared(model_name, model_path, repo_dir, source, quantize, device):
        calls.prepared.append((source, quantize))
        calls.prepared_names.append(model_name)
        return Path("/fake/cache/transformer"), Path("/fake/cache/text-encoder")

    def _repo_dir(model_path, *, with_transformer):
        calls.repo_dirs.append(with_transformer)
        return Path("/fake/repo")

    def _check_memory(estimate, quantize, *, converted):
        calls.memory_checks.append((estimate[1], quantize))
        return calls.fits_together

    def _load_transformer(config, *, prepared, bits, compute_dtype):
        calls.transformers.append({"prepared": prepared, "bits": bits})
        calls.events.append("load transformer")
        return f"transformer-{len(calls.transformers)}"

    def _load_text_encoder(repo_dir, *, prepared, compute_dtype):
        encoder = MagicMock(name=f"text-encoder-{len(calls.text_encoders)}")
        calls.text_encoders.append({"repo_dir": repo_dir, "prepared": prepared, "encoder": encoder})
        calls.events.append("load text encoder")
        return encoder

    def _release():
        calls.released += 1
        calls.events.append("release")

    monkeypatch.setattr(video_diffusers_module, "_download_hub_file", lambda repo, filename, label: Path(f"/fake/hub/{filename}"))
    monkeypatch.setattr(video_diffusers_module, "_repo_dir", _repo_dir)
    monkeypatch.setattr(video_diffusers_module, "_check_system_memory", _check_memory)
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.memory_estimate.ltx_system_estimate", lambda repo_dir, source, quantize: ("estimate", source, quantize))
    monkeypatch.setattr(video_diffusers_module, "_prepared_weights", _prepared)
    monkeypatch.setattr(video_diffusers_module, "_transformer_config", lambda repo_dir: {})
    monkeypatch.setattr(video_diffusers_module, "_load_connectors", lambda repo_dir, dtype: "connectors")
    monkeypatch.setattr(video_diffusers_module, "_place_components", lambda pipeline, torch, *, low_memory: calls.placements.append(("components", low_memory)))
    monkeypatch.setattr(video_diffusers_module, "_place_transformer", lambda transformer, torch, *, quantize, low_memory: calls.placements.append((quantize, low_memory)))
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.weights.load_transformer", _load_transformer)
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.weights.load_text_encoder", _load_text_encoder)
    monkeypatch.setattr("zvisiongenerator.backends.cuda.block_streaming.group_offload", lambda module, device, kind=None: calls.offloaded.append(module))
    monkeypatch.setattr("zvisiongenerator.backends.cuda.memory.release_memory", _release)
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.streaming_decode.decode_frames", lambda vae, latents: vae.decoded_latents.append(latents) or list(vae.decoded_frames))
    monkeypatch.setattr("zvisiongenerator.backends.cuda.block_streaming.load_when_used", lambda component, entry_points, device: calls.lazy.append((component, list(entry_points))))
    monkeypatch.setattr("zvisiongenerator.backends.cuda.block_streaming.unload_before", lambda module, components: calls.unload_before.append((module, list(components))))
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.timestep_runs.modulate_by_runs", lambda transformer: calls.run_wise.append(transformer))
    return calls


def _make_runtime(
    *,
    torch: _FakeTorch | None = None,
    text_pipeline=_FakeTextPipeline,
    image_pipeline=_FakeImagePipeline,
    condition_pipeline=_FakeConditionPipeline,
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
            condition=condition_pipeline,
            video_condition=_FakeVideoCondition,
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
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._load_runtime_dependencies", lambda: runtime)


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

    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._resolve_upsampler_weights", _download)
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._read_state_dict", _read)
    return downloads


def _load_upscale_model(monkeypatch, *, low_memory, mode="t2v", runtime=None):
    _patch_runtime(monkeypatch, runtime or _make_runtime())
    _patch_upsampler_io(monkeypatch)
    model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode=mode, low_memory=low_memory, upscale=True, latent_upsampler=_UPSAMPLER_SOURCE)
    return model


def test_load_model_rejects_missing_cuda(monkeypatch):
    backend = DiffusersVideoBackend()
    monkeypatch.setattr(
        "zvisiongenerator.backends.cuda.video.backend._load_runtime_dependencies",
        lambda: _make_runtime(torch=_FakeTorch(cuda_available=False)),
    )

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v")


@pytest.mark.parametrize(("cuda_version", "expects_hint"), [("13.0", True), ("12.6", False), (None, False)])
def test_missing_cuda_error_asks_for_a_driver_update_only_for_cuda_13_builds(monkeypatch, cuda_version, expects_hint):
    from zvisiongenerator.backends.cuda.driver import CUDA_INSTALL_COMMAND, cuda_driver_hint

    backend = DiffusersVideoBackend()
    monkeypatch.setattr(
        "zvisiongenerator.backends.cuda.video.backend._load_runtime_dependencies",
        lambda: _make_runtime(torch=_FakeTorch(cuda_available=False, cuda_version=cuda_version)),
    )

    with pytest.raises(RuntimeError, match="CUDA") as excinfo:
        backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v")

    assert (cuda_driver_hint(False, "13.0") in str(excinfo.value)) is expects_hint
    assert (CUDA_INSTALL_COMMAND in str(excinfo.value)) is (cuda_version is None)


def test_load_model_prefers_runtime_classes_and_configures_pipeline(monkeypatch):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime()
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._load_runtime_dependencies", lambda: runtime)
    _patch_upsampler_io(monkeypatch)

    model, model_info = backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", low_memory=True, upscale=True, latent_upsampler=_UPSAMPLER_SOURCE)

    assert model_info.family == "ltx"
    assert model.text_to_video is _FakeTextPipeline.loaded_instances[0]
    assert isinstance(model.upsample_pipeline, _FakeUpsamplePipeline)
    assert model.text_to_video.cpu_offload_enabled is False


def test_load_model_builds_the_pipeline_without_a_text_encoder(monkeypatch, fake_weights):
    _patch_runtime(monkeypatch, _make_runtime())

    model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v")

    pipeline = model.text_to_video
    assert pipeline.load_kwargs["text_encoder"] is None
    assert pipeline.load_kwargs["connectors"] == "connectors"
    assert pipeline.transformer == "transformer-1"  # attached after the pipeline is built
    assert model.text_encoder == _TextEncoderSource(repo_dir=Path("/fake/repo"), prepared=Path("/fake/cache/text-encoder"), torch_dtype="bfloat16")


@pytest.mark.parametrize("quantize", [4, 8])
def test_load_reads_the_prepared_copies_of_the_configured_checkpoint(monkeypatch, fake_weights, quantize):
    _patch_runtime(monkeypatch, _make_runtime())
    checkpoint = ("example-org/ltx", "ltx-distilled.safetensors")

    model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", quantize=quantize, transformer_source=checkpoint)

    assert fake_weights.prepared == [(Path("/fake/hub/ltx-distilled.safetensors"), quantize)]
    assert fake_weights.transformers == [{"prepared": Path("/fake/cache/transformer"), "bits": quantize}]
    assert model.text_encoder.prepared == Path("/fake/cache/text-encoder")
    assert fake_weights.placements == [("components", True), (quantize, True)]
    assert fake_weights.repo_dirs == [False]  # the repo's own transformer weights are not downloaded


def test_the_picked_name_names_the_prepared_copy(monkeypatch, fake_weights):
    _patch_runtime(monkeypatch, _make_runtime())

    DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, model_name="ltx-4", mode="t2v")

    assert fake_weights.prepared_names == ["ltx-4"]


class TestPreparedCopiesLiveWithTheAlias:
    def _prepare(self, monkeypatch, tmp_path, name, bits, writes):
        def _write(kind):
            def _writer(*args):
                folder = args[2] if kind == "transformer" else args[1]
                writes.append((name, kind))
                (folder / "weights.safetensors").write_text(f"{kind}-{name}")

            return _writer

        monkeypatch.setattr("zvisiongenerator.utils.paths.get_ziv_data_dir", lambda: tmp_path)
        monkeypatch.setattr("zvisiongenerator.backends.cuda.video.weights.write_transformer", _write("transformer"))
        monkeypatch.setattr("zvisiongenerator.backends.cuda.video.weights.write_text_encoder", _write("text_encoder"))
        monkeypatch.setattr("zvisiongenerator.backends.cuda.video.weights.transformer_sources", lambda source: [source])
        monkeypatch.setattr("zvisiongenerator.backends.cuda.video.weights.text_encoder_sources", lambda repo: [repo / "te.safetensors"])
        monkeypatch.setattr(video_diffusers_module, "_transformer_config", lambda repo_dir: {})
        # The NF4 storage format names the installed bitsandbytes version, which is not installed on macOS.
        monkeypatch.setattr("importlib.metadata.version", lambda _name: "0.50.0")
        return real_prepared_weights(name, "org/repo", tmp_path, tmp_path / "ckpt.safetensors", bits, "cuda")

    def test_copies_are_the_alias_stored_copy_and_share_one_text_encoder(self, monkeypatch, tmp_path):
        (tmp_path / "ckpt.safetensors").write_text("checkpoint")
        (tmp_path / "te.safetensors").write_text("text encoder")
        writes: list[tuple[str, str]] = []

        transformer4, text_encoder4 = self._prepare(monkeypatch, tmp_path, "ltx-4", 4, writes)
        transformer8, text_encoder8 = self._prepare(monkeypatch, tmp_path, "ltx-8", 8, writes)

        assert (transformer4, text_encoder4) == (tmp_path / "models" / "ltx-4@q4" / "transformer", tmp_path / "models" / "ltx-4@q4" / "text_encoder")
        assert transformer8 == tmp_path / "models" / "ltx-8@q8" / "transformer"
        assert writes == [("ltx-4", "transformer"), ("ltx-4", "text_encoder"), ("ltx-8", "transformer")]
        assert (text_encoder8 / "weights.safetensors").stat().st_ino == (text_encoder4 / "weights.safetensors").stat().st_ino


def test_without_a_checkpoint_the_repos_own_transformer_is_prepared(monkeypatch, fake_weights):
    _patch_runtime(monkeypatch, _make_runtime())

    DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=8, mode="t2v", low_memory=False)

    assert fake_weights.prepared == [(Path("/fake/repo/transformer"), 8)]
    assert fake_weights.repo_dirs == [True]
    assert fake_weights.placements == [("components", False), (8, False)]


def test_load_model_checks_system_memory_before_loading_weights(monkeypatch, fake_weights):
    _patch_runtime(monkeypatch, _make_runtime())

    def _refuse(estimate, quantize, *, converted):
        raise RuntimeError("needs more memory")

    monkeypatch.setattr(video_diffusers_module, "_check_system_memory", _refuse)

    with pytest.raises(RuntimeError, match="needs more memory"):
        DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", quantize=4)

    assert fake_weights.prepared == []
    assert fake_weights.transformers == []


@pytest.mark.parametrize("quantize", [None, 6])
def test_load_model_needs_q4_or_q8_and_names_the_models(monkeypatch, fake_weights, quantize):
    _patch_runtime(monkeypatch, _make_runtime())

    with pytest.raises(ValueError, match="ltx-4"):
        DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", quantize=quantize)

    assert fake_weights.repo_dirs == []  # refused before anything downloads


class TestTakingTurns:
    """When the text encoder and transformer do not fit in system memory together, they never are."""

    def _load(self, monkeypatch, fake_weights, **kwargs):
        fake_weights.fits_together = False
        _patch_runtime(monkeypatch, _make_runtime())
        model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", mode="t2v", quantize=4, **kwargs)
        return model

    def test_the_transformer_waits_until_the_first_prompt_is_encoded(self, monkeypatch, fake_weights, tmp_path):
        model = self._load(monkeypatch, fake_weights)
        assert fake_weights.transformers == []

        DiffusersVideoBackend().text_to_video(model, prompt="a river", width=512, height=384, num_frames=9, seed=1, steps=8, output_path=str(tmp_path / "a.mp4"))

        assert fake_weights.events == ["release", "load text encoder", "release", "load transformer", "release"]
        assert model.text_to_video.calls[0]["prompt_embeds"] == "embeds:a river"

    def test_without_low_memory_the_gpu_held_transformer_never_takes_turns(self, monkeypatch, fake_weights):
        model = self._load(monkeypatch, fake_weights, low_memory=False)

        assert model.transformer.swap is False
        assert len(fake_weights.transformers) == 1  # attached at load

    def test_a_new_prompt_releases_the_transformer_before_the_text_encoder_loads(self, monkeypatch, fake_weights, tmp_path):
        model = self._load(monkeypatch, fake_weights)
        backend = DiffusersVideoBackend()
        args = {"width": 512, "height": 384, "num_frames": 9, "seed": 1, "steps": 8}
        backend.text_to_video(model, prompt="a river", output_path=str(tmp_path / "a.mp4"), **args)
        fake_weights.events.clear()

        backend.text_to_video(model, prompt="a river", output_path=str(tmp_path / "b.mp4"), **args)
        assert fake_weights.events == []  # a repeated prompt keeps both its embeddings and the transformer

        backend.text_to_video(model, prompt="a city", output_path=str(tmp_path / "c.mp4"), **args)
        # The transformer goes first, the text encoder comes and goes, then the transformer returns.
        assert fake_weights.events == ["release", "load text encoder", "release", "load transformer", "release"]
        assert model.text_to_video.encoded[-1]["text_encoder"] is not None

    def test_loras_are_unloaded_with_the_transformer_and_applied_again_after(self, monkeypatch, fake_weights, tmp_path):
        (lora,) = _write_loras(tmp_path, "style.safetensors")
        model = self._load(monkeypatch, fake_weights, loras=[(lora, 0.5)])
        pipeline = model.text_to_video
        unloads: list[bool] = []
        pipeline.unload_lora_weights = lambda: unloads.append(True)
        args = {"width": 512, "height": 384, "num_frames": 9, "seed": 1, "steps": 8}

        DiffusersVideoBackend().text_to_video(model, prompt="a river", output_path=str(tmp_path / "a.mp4"), **args)
        DiffusersVideoBackend().text_to_video(model, prompt="a city", output_path=str(tmp_path / "b.mp4"), **args)

        assert unloads == [True]
        assert pipeline.loaded_loras == [(lora, "lora_0"), (lora, "lora_0")]

    def test_a_missing_lora_still_fails_the_load(self, monkeypatch, fake_weights, tmp_path):
        with pytest.raises(FileNotFoundError):
            self._load(monkeypatch, fake_weights, loras=[(str(tmp_path / "missing.safetensors"), 1.0)])


def _placement_pipeline():
    pipeline = MagicMock()
    pipeline.vae = MagicMock(spec=["to", "enable_tiling"])
    return pipeline


def _record_streaming(monkeypatch):
    streamed: list[tuple[str, object, object]] = []
    monkeypatch.setattr("zvisiongenerator.backends.cuda.block_streaming.group_offload", lambda module, device, kind=None: streamed.append(("group", module, kind)))
    monkeypatch.setattr("zvisiongenerator.backends.cuda.block_streaming.stream_blocks", lambda blocks, device: streamed.append(("blocks", blocks, None)))
    monkeypatch.setattr("zvisiongenerator.backends.cuda.block_streaming.move_outside", lambda model, blocks, device: streamed.append(("outside", model, None)))
    return streamed


def test_low_memory_streams_the_connectors_and_moves_the_vaes_to_the_gpu_only_when_used(monkeypatch, fake_weights):
    streamed = _record_streaming(monkeypatch)
    pipeline = _placement_pipeline()
    pipeline.vae = MagicMock(spec=["to", "enable_tiling", "encoder", "decoder"])

    real_place_components(pipeline, _FakeTorch(), low_memory=True)

    assert streamed == [("group", pipeline.connectors, "leaf_level")]
    pipeline.vae.to.assert_not_called()
    assert (pipeline.vae, [pipeline.vae.encoder, pipeline.vae.decoder]) in fake_weights.lazy
    assert any(component is pipeline.vocoder for component, _ in fake_weights.lazy)
    pipeline.vae.enable_tiling.assert_called_once()


def test_the_transformer_sends_the_vaes_back_before_it_runs_in_low_memory_mode(monkeypatch, fake_weights):
    _patch_runtime(monkeypatch, _make_runtime())

    model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", low_memory=True)

    assert fake_weights.unload_before == [("transformer-1", [model.text_to_video.vae])]


def test_the_transformer_modulates_per_token_timesteps_by_runs(monkeypatch, fake_weights):
    _patch_runtime(monkeypatch, _make_runtime())

    DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="i2v")

    assert fake_weights.run_wise == ["transformer-1"]


@pytest.mark.parametrize(("quantize", "streamed_by_blocks"), [(4, True), (8, False), (None, False)])
def test_low_memory_streams_the_transformer_block_by_block(monkeypatch, quantize, streamed_by_blocks):
    streamed = _record_streaming(monkeypatch)
    transformer = MagicMock()

    real_place_transformer(transformer, _FakeTorch(), quantize=quantize, low_memory=True)

    if streamed_by_blocks:
        assert streamed == [("blocks", transformer.transformer_blocks, None), ("outside", transformer, None)]
    else:
        assert streamed == [("group", transformer, "block_level")]
    transformer.to.assert_not_called()


def test_without_low_memory_everything_moves_to_the_gpu(monkeypatch):
    streamed = _record_streaming(monkeypatch)
    pipeline = _placement_pipeline()
    transformer = MagicMock()

    real_place_components(pipeline, _FakeTorch(), low_memory=False)
    real_place_transformer(transformer, _FakeTorch(), quantize=4, low_memory=False)

    assert streamed == []
    pipeline.connectors.to.assert_called_once_with("device:cuda")
    transformer.to.assert_called_once_with("device:cuda")


def test_the_vae_decodes_in_tiles_in_time_as_well_as_space():
    pipeline = MagicMock(spec=["connectors", "vae"])
    pipeline.vae = MagicMock(spec=["to", "enable_tiling", "use_framewise_decoding"])
    pipeline.vae.use_framewise_decoding = False

    real_place_components(pipeline, _FakeTorch(), low_memory=False)

    pipeline.vae.enable_tiling.assert_called_once()
    assert pipeline.vae.use_framewise_decoding is True


def test_every_block_feed_forward_and_rms_norm_is_chunked(monkeypatch, fake_weights):
    torch = pytest.importorskip("torch")
    normalization = pytest.importorskip("diffusers.models.normalization")
    chunked: list[tuple[list[object], int]] = []
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.token_chunking.chunk_tokens", lambda modules, size: chunked.append((list(modules), size)))

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ff = torch.nn.Linear(4, 4)
            self.audio_ff = torch.nn.Linear(4, 4)
            self.norm3 = normalization.RMSNorm(4, eps=1e-6)
            self.attn1 = torch.nn.Module()
            self.attn1.norm_q = normalization.RMSNorm(4, eps=1e-6)
            self.other = torch.nn.LayerNorm(4)

    transformer = torch.nn.Module()
    transformer.transformer_blocks = torch.nn.ModuleList([_Block()])
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.weights.load_transformer", lambda config, **_: transformer)
    _patch_runtime(monkeypatch, _make_runtime())

    DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v")

    block = transformer.transformer_blocks[0]
    ((modules, size),) = chunked
    assert modules == [block.ff, block.audio_ff, block.norm3, block.attn1.norm_q]
    assert size == video_diffusers_module._TOKEN_CHUNK


def _encoding_model(*, low_memory=True, swap=True, prepared=Path("/fake/cache/text-encoder")):
    """A loaded model whose text encoder and transformer take turns (*swap*) or fit together."""
    return SimpleNamespace(
        runtime=_make_runtime(),
        low_memory=low_memory,
        text_encoder=_TextEncoderSource(repo_dir=Path("/fake/repo"), prepared=prepared, torch_dtype="bfloat16"),
        transformer=SimpleNamespace(swap=swap),
        prompt_cache=None,
        resident_text_encoder=None,
    )


def test_a_prompt_is_encoded_with_a_text_encoder_loaded_for_it_and_released_after(fake_weights):
    model = _encoding_model()
    pipeline = _FakeTextPipeline()

    inputs = real_encode_prompt(model, pipeline, "a river")

    assert inputs == {"prompt_embeds": "embeds:a river", "prompt_attention_mask": "mask:a river"}
    assert pipeline.encoded[0]["do_classifier_free_guidance"] is False  # the distilled guidance needs no negative prompt
    (loaded,) = fake_weights.text_encoders
    assert loaded["prepared"] == Path("/fake/cache/text-encoder")
    assert pipeline.encoded[0]["text_encoder"] is loaded["encoder"]  # present while encoding
    assert pipeline.encoded[0]["dtype"] == "bfloat16"  # never the FP8 storage dtype
    assert pipeline.text_encoder is None
    assert fake_weights.released == 1
    assert fake_weights.offloaded == [loaded["encoder"]]


def test_a_repeated_prompt_reuses_its_embeddings_and_a_new_one_loads_the_text_encoder_again(fake_weights):
    model = _encoding_model()
    pipeline = _FakeTextPipeline()

    first = real_encode_prompt(model, pipeline, "a river")
    again = real_encode_prompt(model, pipeline, "a river")
    real_encode_prompt(model, pipeline, "a city")

    assert again is first
    assert len(fake_weights.text_encoders) == 2
    assert [call["prompt"] for call in pipeline.encoded] == ["a river", "a city"]


def test_a_text_encoder_that_fits_beside_the_transformer_stays_loaded_for_the_next_prompt(fake_weights):
    model = _encoding_model(swap=False)
    pipeline = _FakeTextPipeline()

    real_encode_prompt(model, pipeline, "a river")
    real_encode_prompt(model, pipeline, "a city")

    (loaded,) = fake_weights.text_encoders
    assert model.resident_text_encoder is loaded["encoder"]
    assert pipeline.encoded[1]["text_encoder"] is loaded["encoder"]
    assert pipeline.text_encoder is None
    assert fake_weights.released == 0


def test_without_low_memory_the_text_encoder_still_streams_beside_the_transformer(fake_weights):
    real_encode_prompt(_encoding_model(low_memory=False), _FakeTextPipeline(), "a river")

    (loaded,) = fake_weights.text_encoders
    loaded["encoder"].to.assert_not_called()
    assert fake_weights.offloaded == [loaded["encoder"]]


def test_the_text_encoder_is_released_when_encoding_fails(fake_weights):
    pipeline = _FakeTextPipeline()
    pipeline.encode_prompt = MagicMock(side_effect=RuntimeError("out of memory"))

    with pytest.raises(RuntimeError, match="out of memory"):
        real_encode_prompt(_encoding_model(), pipeline, "a river")

    assert pipeline.text_encoder is None
    assert fake_weights.released == 1


def test_without_a_text_encoder_source_the_pipeline_gets_the_prompt():
    model = SimpleNamespace(text_encoder=None, prompt_cache=None)

    assert real_encode_prompt(model, _FakeTextPipeline(), "a river") == {"prompt": "a river"}


def test_two_stage_upscale_encodes_the_prompt_once_for_both_stages(monkeypatch, tmp_path):
    model = _load_upscale_model(monkeypatch, low_memory=True)

    DiffusersVideoBackend().text_to_video(model, **_text_to_video_args(tmp_path), stage1_steps=8)

    pipeline = model.text_to_video
    assert len(pipeline.encoded) == 1
    assert [call["prompt_embeds"] for call in pipeline.calls] == ["embeds:a city", "embeds:a city"]


def _patch_memory(monkeypatch, *, total_gib, block_matrices=10 * 1024**3, rest=0, connectors=0, text_encoder=0):
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(total=total_gib * 1024**3))
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.weights.transformer_elements", lambda source: (block_matrices, rest))
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.weights.component_elements", lambda path: connectors)
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.weights.text_encoder_bytes", lambda path: text_encoder)


def _check_stock_memory(quantize):
    return real_check_system_memory(real_ltx_system_estimate(Path("/fake/repo"), Path("/fake/ckpt"), quantize), quantize, converted=False)


def test_a_load_larger_than_system_memory_is_refused_and_names_the_smaller_level(monkeypatch):
    _patch_memory(monkeypatch, total_gib=12)  # 10 G block weights at q8 and the runtime: 16 GB

    with pytest.raises(RuntimeError, match="ltx-4"):
        _check_stock_memory(8)


def test_a_tight_load_warns(monkeypatch):
    _patch_memory(monkeypatch, total_gib=19)  # 10 G block weights at q8 and the runtime: 16 GB, more than 80% of 19

    with pytest.warns(UserWarning, match="ltx-4"):
        _check_stock_memory(8)


@pytest.mark.parametrize(("total_gib", "together"), [(64, True), (24, False)])
def test_the_check_says_whether_text_encoder_and_transformer_fit_together(monkeypatch, total_gib, together):
    # 10 G block weights and a 10 G text encoder at q8: 10 GB each, 22.5 GB together with the runtime.
    _patch_memory(monkeypatch, total_gib=total_gib, text_encoder=10 * 1024**3)

    assert _check_stock_memory(8) is together


def test_text_to_video_exports_path_and_reports_progress(monkeypatch, tmp_path):
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
        steps=6,
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
    assert call["num_inference_steps"] == 6
    assert call["sigmas"] == _stage1_sigmas(_DISTILLED_SIGMAS, 6)
    assert call["frame_rate"] == 24
    assert call["generator"].seed == 123
    assert events == [{"phase": "video", "current_step": step, "total_steps": 6} for step in range(1, 7)]
    assert runtime.torch.cuda.empty_cache_calls == 1


def _run_text_to_video(monkeypatch, tmp_path, *, steps, step_callback=None):
    runtime = _make_runtime()
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._load_runtime_dependencies", lambda: runtime)
    backend = DiffusersVideoBackend()
    model, _ = backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", low_memory=False)
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
    assert _guidance(call) == _NEUTRAL_GUIDANCE


def test_distilled_guidance_turns_every_guidance_term_off():
    # diffusers' own defaults for the other terms (audio 7, STG 1, modality 3) over-cook a distilled model.
    assert dict(_DISTILLED_GUIDANCE) == _NEUTRAL_GUIDANCE


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
        prompt_inputs={"prompt_embeds": "embeds", "prompt_attention_mask": "mask"},
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
    assert kwargs["prompt_embeds"] == "embeds"
    assert kwargs["prompt_attention_mask"] == "mask"
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
    plain, _ = backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", low_memory=False)
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
    assert stage2["output_type"] == "latent"  # decoded by streaming_decode
    assert stage2["sigmas"] == list(_STAGE2_SIGMAS)
    assert stage2["num_inference_steps"] == 3
    assert _guidance(stage1) == _guidance(stage2) == _NEUTRAL_GUIDANCE
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


def test_resolve_pipeline_classes_requires_ltx2():
    ltx2 = SimpleNamespace(LTX2Pipeline=object, LTX2ImageToVideoPipeline=object, LTX2ConditionPipeline=object, LTX2LatentUpsamplePipeline=object)

    classes = _resolve_pipeline_classes(ltx2, _FakeUpsamplerModel, _FakeVideoCondition)

    assert classes.text_to_video is object
    assert classes.image_to_video is object
    assert classes.condition is object
    assert classes.video_condition is _FakeVideoCondition
    assert classes.latent_upsample_pipeline is object
    assert classes.latent_upsampler_model is _FakeUpsamplerModel
    for missing in ("LTX2Pipeline", "LTX2ImageToVideoPipeline", "LTX2ConditionPipeline", "LTX2LatentUpsamplePipeline"):
        partial = SimpleNamespace(**{name: object for name in vars(ltx2) if name != missing})
        with pytest.raises(RuntimeError, match="LTX-2"):
            _resolve_pipeline_classes(partial, _FakeUpsamplerModel, _FakeVideoCondition)
    with pytest.raises(RuntimeError, match="LTX-2"):
        _resolve_pipeline_classes(SimpleNamespace(LTXPipeline=object, LTXImageToVideoPipeline=object), _FakeUpsamplerModel, _FakeVideoCondition)


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

    DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", upscale=True, latent_upsampler=_UPSAMPLER_SOURCE)

    assert events == ["download", "from_pretrained"]
    assert downloads == [_UPSAMPLER_SOURCE]


def test_failed_upsampler_download_stops_before_the_pipeline_loads(monkeypatch):
    _patch_runtime(monkeypatch, _make_runtime())

    def _fail(repo_id, filename):
        raise RuntimeError(f"Failed to download the LTX latent upsampler '{filename}' from '{repo_id}': offline")

    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._resolve_upsampler_weights", _fail)

    with pytest.raises(RuntimeError, match="latent upsampler"):
        DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", upscale=True, latent_upsampler=_UPSAMPLER_SOURCE)

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
        DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", upscale=True)

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

    model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", latent_upsampler=_UPSAMPLER_SOURCE)

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

    monkeypatch.setattr(video_diffusers_module, "_download_hub_file", real_download_hub_file)
    calls: list[dict[str, str]] = []

    def _fake_download(**kwargs):
        calls.append(kwargs)
        return "/cache/example-x2.safetensors"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)

    assert _resolve_upsampler_weights(*_UPSAMPLER_SOURCE) == "/cache/example-x2.safetensors"
    assert calls == [{"repo_id": _UPSAMPLER_SOURCE[0], "filename": _UPSAMPLER_SOURCE[1]}]


def test_resolve_upsampler_weights_reports_download_failures(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(video_diffusers_module, "_download_hub_file", real_download_hub_file)

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
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._load_runtime_dependencies", lambda: runtime)
    lora_path = tmp_path / "style.safetensors"
    lora_path.write_text("weights", encoding="utf-8")

    model, _ = backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", loras=[(str(lora_path), 0.75)])

    assert model.text_to_video.loaded_loras == [(str(lora_path), "lora_0")]
    assert model.text_to_video.adapter_calls == [(["lora_0"], [0.75])]


class _LoraNoMatchPipeline(_FakeTextPipeline):
    """A pipeline whose LoRA loads raise *error* for the paths in *unmatched*."""

    unmatched: tuple[str, ...] = ()
    error: type[Exception] = ValueError

    def load_lora_weights(self, path: str, adapter_name: str | None = None) -> None:
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
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._load_runtime_dependencies", lambda: _make_runtime(text_pipeline=_LoraNoMatchPipeline))

    with pytest.warns(UserWarning, match="other-model.safetensors: none of its LoRA tensors match"):
        model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", loras=[(first, 1.0), (second, 0.5)])

    assert model.text_to_video.loaded_loras == [(second, "lora_1")]
    assert model.text_to_video.adapter_calls == [(["lora_1"], [0.5])]


@pytest.mark.parametrize("low_memory", [True, False])
def test_loading_loras_never_turns_on_model_cpu_offload(monkeypatch, tmp_path, fake_peft_no_match, low_memory):
    (path,) = _write_loras(tmp_path, "other-model.safetensors")
    _LoraNoMatchPipeline.unmatched = (path,)
    _LoraNoMatchPipeline.error = fake_peft_no_match
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._load_runtime_dependencies", lambda: _make_runtime(text_pipeline=_LoraNoMatchPipeline))

    with pytest.warns(UserWarning):
        model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", low_memory=low_memory, loras=[(path, 1.0)])

    # The transformer streams block by block; whole-component CPU offload would hold every component in memory.
    assert model.text_to_video.cpu_offload_enabled is False


def test_other_lora_load_errors_still_fail_the_load(monkeypatch, tmp_path, fake_peft_no_match):
    (path,) = _write_loras(tmp_path, "bad.safetensors")
    _LoraNoMatchPipeline.unmatched = (path,)
    _LoraNoMatchPipeline.error = ValueError
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._load_runtime_dependencies", lambda: _make_runtime(text_pipeline=_LoraNoMatchPipeline))

    with pytest.raises(RuntimeError, match="Failed to load LoRA"):
        DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", loras=[(path, 1.0)])


def test_load_model_rejects_missing_or_unsupported_lora(monkeypatch, tmp_path):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime()
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._load_runtime_dependencies", lambda: runtime)

    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", loras=[(str(tmp_path / "missing.safetensors"), 1.0)])

    bad_lora = tmp_path / "bad.txt"
    bad_lora.write_text("weights", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported LoRA format"):
        backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", loras=[(str(bad_lora), 1.0)])


def test_load_model_rejects_lora_scales_when_adapter_api_cannot_apply_them(monkeypatch, tmp_path):
    backend = DiffusersVideoBackend()
    runtime = _make_runtime(text_pipeline=_NoScalePipeline)
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._load_runtime_dependencies", lambda: runtime)
    lora_path = tmp_path / "style.safetensors"
    lora_path.write_text("weights", encoding="utf-8")

    with pytest.raises(RuntimeError, match="explicit LoRA scales"):
        backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", loras=[(str(lora_path), 0.5)])


def test_load_model_rejects_old_diffusers_versions(monkeypatch):
    with pytest.raises(RuntimeError, match=r"diffusers>=0\.41\.0"):
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
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._load_runtime_dependencies", lambda: runtime)
    validator = MagicMock(side_effect=AssertionError("duplicate validation"))
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.backend._validate_diffusers_version", validator)

    backend.load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v")

    validator.assert_not_called()


class TestExtendVideo:
    """Continue a clip: pin its tail at the start of a window at its size and rate, write only the new part."""

    @pytest.fixture
    def segments(self, monkeypatch):
        """Replace the tail readers, the audio carry and the segment encoder; return what each received."""
        from contextlib import contextmanager

        calls = SimpleNamespace(tails=[], tail_audio=[], carries=[], pinned=[], restored=[], encoded=[], events=[])
        tail = np.zeros((25, 256, 512, 3), dtype=np.uint8)
        tail_audio = np.zeros((2, 16000), dtype=np.float32)
        carry = SimpleNamespace(token_count=26)

        def _read_tail(source):
            calls.tails.append(source)
            return tail

        def _read_tail_audio(source, sample_rate):
            calls.tail_audio.append((source, sample_rate))
            return tail_audio

        def _encode_carry(pipeline, waveform, token_count):
            calls.carries.append((pipeline, waveform, token_count))
            return carry

        @contextmanager
        def _pin(transformer, pinned):
            calls.pinned.append((transformer, pinned))
            calls.events.append("pin")
            yield
            calls.events.append("unpin")

        def _restore(audio_latents, restored):
            calls.restored.append((audio_latents, restored))
            calls.events.append("restore")

        def _write_frames(frames, command):
            calls.encoded.append((list(frames), command))

        monkeypatch.setattr("zvisiongenerator.utils.video_tail.read_source_tail", _read_tail)
        monkeypatch.setattr("zvisiongenerator.utils.video_tail.read_source_tail_audio", _read_tail_audio)
        monkeypatch.setattr("zvisiongenerator.backends.cuda.video.audio_carry.encode_audio_carry", _encode_carry)
        monkeypatch.setattr("zvisiongenerator.backends.cuda.video.audio_carry.pin_audio_prefix", _pin)
        monkeypatch.setattr("zvisiongenerator.backends.cuda.video.audio_carry.restore_audio_prefix", _restore)
        monkeypatch.setattr("zvisiongenerator.utils.video_splice.write_segment_frames", _write_frames)
        calls.tail, calls.tail_audio_samples, calls.carry = tail, tail_audio, carry
        return calls

    @staticmethod
    def _load(monkeypatch, *, mode="extend", upscale=False):
        _patch_runtime(monkeypatch, _make_runtime())
        if upscale:
            _patch_upsampler_io(monkeypatch)
            return DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode=mode, upscale=True, latent_upsampler=_UPSAMPLER_SOURCE)[0]
        return DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode=mode)[0]

    @staticmethod
    def _extend(model, tmp_path, source=None, **overrides):
        kwargs = {"extend_frames": 16, "seed": 7, "steps": 8, "segment_dir": tmp_path, "audio_output": True, **overrides}
        return DiffusersVideoBackend().extend_video(model, source or make_extend_source(), "She closes the umbrella.", **kwargs)

    def test_extend_mode_loads_the_condition_pipeline(self, monkeypatch):
        model = self._load(monkeypatch)
        (pipeline,) = _FakeConditionPipeline.loaded_instances
        assert model.extend is pipeline
        assert model.text_to_video is None and model.image_to_video is None

    def test_pins_the_tail_at_latent_zero_and_generates_the_window_at_the_source_size_and_rate(self, monkeypatch, tmp_path, segments):
        model = self._load(monkeypatch)
        source = make_extend_source(fps=23.976, fps_text="24000/1001")
        self._extend(model, tmp_path, source)
        (call,) = model.extend.calls
        (condition,) = call["conditions"]
        assert condition.frames is segments.tail
        assert (condition.index, condition.strength) == (0, 1.0)
        assert (call["width"], call["height"], call["num_frames"]) == (512, 256, 41)
        assert call["frame_rate"] == 23.976
        assert segments.tails == [source]

    def test_writes_only_the_new_frames_with_the_source_rate_and_colour_tags(self, monkeypatch, tmp_path, segments):
        model = self._load(monkeypatch)
        source = make_extend_source(fps_text="24/1", color_space="bt709", color_primaries="bt709", color_trc="bt709")
        segment = self._extend(model, tmp_path, source)
        ((frames, command),) = segments.encoded
        assert [int(frame[0, 0, 0]) for frame in frames] == list(range(25, 41))
        assert command[command.index("-r") + 1] == "24/1"
        assert command[command.index("-s") + 1] == "512x256"
        assert "colorspace=bt709" in command[command.index("-vf") + 1]
        assert command[-1] == str(tmp_path / "segment.mp4")
        assert segment.video_path == tmp_path / "segment.mp4"

    def test_the_sound_starts_a_lead_before_the_new_part_and_covers_it_exactly(self, monkeypatch, tmp_path, segments):
        import wave

        from zvisiongenerator.utils.video_extend import audio_lead_samples

        model = self._load(monkeypatch)
        segment = self._extend(model, tmp_path)
        lead = audio_lead_samples(24.0)
        assert segment.audio_path == tmp_path / "segment.wav"
        assert segment.lead_samples == lead
        with wave.open(str(segment.audio_path), "rb") as reader:
            assert reader.getframerate() == 48000
            assert reader.getnframes() == lead + round(16 / 24 * 48000)

    def test_the_sound_under_the_tail_is_pinned_during_the_window_and_restored_before_decoding(self, monkeypatch, tmp_path, segments):
        model = self._load(monkeypatch)
        source = make_extend_source()
        self._extend(model, tmp_path, source)
        assert segments.tail_audio == [(source, 16000)]
        ((pipeline, waveform, token_count),) = segments.carries
        assert pipeline is model.extend and waveform is segments.tail_audio_samples
        assert token_count == round(25 / 24 * 25)
        ((transformer, pinned),) = segments.pinned
        assert transformer is model.extend.transformer_at_call[0] and pinned is segments.carry
        ((audio_latents, restored),) = segments.restored
        assert audio_latents == "audio-latent" and restored is segments.carry
        assert segments.events == ["pin", "unpin", "restore"]
        assert model.extend.audio_vae.decoded == ["audio-latent"]

    def test_the_sound_is_pinned_even_when_none_is_written(self, monkeypatch, tmp_path, segments):
        model = self._load(monkeypatch)
        self._extend(model, tmp_path, audio_output=False)
        assert len(segments.pinned) == 1

    def test_a_silent_source_is_continued_without_a_pin(self, monkeypatch, tmp_path, segments):
        model = self._load(monkeypatch)
        self._extend(model, tmp_path, make_extend_source(has_audio=False))
        assert segments.tail_audio == [] and segments.carries == [] and segments.pinned == []

    @pytest.mark.parametrize(("has_audio", "audio_output"), [(False, True), (True, False)])
    def test_no_sound_is_written_for_a_silent_source_or_without_audio_output(self, monkeypatch, tmp_path, segments, has_audio, audio_output):
        model = self._load(monkeypatch)
        segment = self._extend(model, tmp_path, make_extend_source(has_audio=has_audio), audio_output=audio_output)
        assert segment.audio_path is None
        assert not (tmp_path / "segment.wav").exists()

    @pytest.mark.parametrize("frames", [0, 12, 4])
    def test_refuses_a_length_that_is_not_a_positive_multiple_of_8(self, monkeypatch, tmp_path, segments, frames):
        model = self._load(monkeypatch)
        with pytest.raises(ValueError, match="multiple of 8"):
            self._extend(model, tmp_path, extend_frames=frames)

    def test_refuses_a_model_loaded_for_another_mode(self, monkeypatch, tmp_path, segments):
        model = self._load(monkeypatch, mode="t2v")
        with pytest.raises(RuntimeError, match="extend"):
            self._extend(model, tmp_path)

    def test_refuses_an_upscale_model(self, monkeypatch, tmp_path, segments):
        model = self._load(monkeypatch, upscale=True)
        with pytest.raises(ValueError, match="upscale"):
            self._extend(model, tmp_path)

    def test_a_window_of_the_wrong_length_fails_before_anything_is_written(self, monkeypatch, tmp_path, segments):
        model = self._load(monkeypatch)
        monkeypatch.setattr(_FakeConditionPipeline, "window", 40)
        with pytest.raises(RuntimeError, match="41"):
            self._extend(model, tmp_path)
        assert segments.encoded == []

    def test_text_to_video_still_runs_at_the_model_rate(self, monkeypatch, tmp_path):
        model = self._load(monkeypatch, mode="t2v")
        DiffusersVideoBackend().text_to_video(model, "prompt", 512, 256, 25, 7, 8, str(tmp_path / "out.mp4"))
        assert model.text_to_video.calls[0]["frame_rate"] == model.model_info.default_fps

    @pytest.mark.parametrize("repo", ["diffusers/LTX-2.3-Distilled-Diffusers", "unsloth/LTX-2.5-FP8@" + _PIN])
    def test_an_extend_pass_never_builds_the_ancestral_scheduler_whatever_the_sampler(self, monkeypatch, tmp_path, segments, repo):
        def _never(*_args, **_kwargs):
            raise AssertionError("the ancestral scheduler was built for an Extend pass")

        monkeypatch.setattr(_sampler_module, "ancestral_scheduler", _never)
        _patch_runtime(monkeypatch, _make_runtime())
        model = DiffusersVideoBackend().load_model(repo, quantize=4, mode="extend", sampler="euler_ancestral")[0]
        original = model.extend.scheduler

        self._extend(model, tmp_path)

        assert model.sampler == "euler_ancestral"
        assert model.extend.schedulers_at_call == [original]
        assert model.extend.scheduler is original

    def test_the_extend_conditions_pass_through_the_clip_and_the_one_stage_pass(self, monkeypatch, tmp_path, segments):
        seen: list[bool] = []
        real = video_diffusers_module._pass_scheduler

        def _spy(*args, **kwargs):
            seen.append(kwargs["conditioned"])
            return real(*args, **kwargs)

        monkeypatch.setattr(video_diffusers_module, "_pass_scheduler", _spy)
        self._extend(self._load(monkeypatch), tmp_path)

        assert seen == [True]


class TestEndImage:
    """A clip that ends on a last-frame image runs on the condition pipeline, pinned at its first and last frames."""

    @staticmethod
    def _load(monkeypatch, repo="diffusers/LTX-2.3-Distilled-Diffusers", **kwargs):
        _patch_runtime(monkeypatch, _make_runtime())
        return DiffusersVideoBackend().load_model(repo, quantize=4, **{"mode": "i2v", "end_image": True, **kwargs})[0]

    @staticmethod
    def _images(tmp_path):
        Image.new("RGB", (16, 8), "blue").save(tmp_path / "first.png")
        Image.new("RGB", (8, 16), "red").save(tmp_path / "last.png")
        return str(tmp_path / "first.png"), str(tmp_path / "last.png")

    def _run(self, model, tmp_path, *, num_frames=49, **kwargs):
        first, last = self._images(tmp_path)
        return DiffusersVideoBackend().image_to_video(model, first, "she turns", 128, 64, num_frames, 7, 8, str(tmp_path / "out.mp4"), end_image=last, **kwargs)

    @pytest.mark.parametrize(
        ("mode", "end_image", "slot", "pipeline_class"),
        [("i2v", True, "image_to_video", _FakeConditionPipeline), ("i2v", False, "image_to_video", _FakeImagePipeline), ("t2v", True, "text_to_video", _FakeTextPipeline)],
    )
    def test_only_an_image_to_video_load_for_an_end_image_builds_the_condition_pipeline(self, monkeypatch, mode, end_image, slot, pipeline_class):
        model = self._load(monkeypatch, mode=mode, end_image=end_image)
        assert type(getattr(model, slot)) is pipeline_class
        assert model.end_image is (end_image and mode == "i2v")

    def test_pins_the_first_image_at_latent_zero_and_the_last_after_it_both_fitted_to_the_clip(self, monkeypatch, tmp_path):
        model = self._load(monkeypatch)
        self._run(model, tmp_path)
        (call,) = model.image_to_video.calls
        first, last = call["conditions"]
        assert [(c.index, c.strength) for c in (first, last)] == [(0, 1.0), (-1, 1.0)]
        assert first.frames.size == last.frames.size == (128, 64)
        assert first.frames.getpixel((0, 0)) == (0, 0, 255) and last.frames.getpixel((0, 0)) == (255, 0, 0)
        assert "image" not in call

    @pytest.mark.parametrize("num_frames", [9, 49, 121])
    def test_the_last_image_lands_on_the_last_frame_and_the_placement_is_undone_after(self, monkeypatch, tmp_path, num_frames):
        model = self._load(monkeypatch)
        self._run(model, tmp_path, num_frames=num_frames)
        assert model.image_to_video.keyframe_pixels == [num_frames - 1]
        assert "_prepare_keyframe_coords" not in vars(model.image_to_video)

    @pytest.mark.parametrize(("repo", "crf"), [("diffusers/LTX-2.3-Distilled-Diffusers", None), ("unsloth/LTX-2.5-FP8@" + _PIN, 0)])
    def test_an_image_the_backend_recompressed_is_not_recompressed_again(self, monkeypatch, tmp_path, repo, crf):
        model = self._load(monkeypatch, repo=repo, image_crf=18 if crf == 0 else None)
        self._run(model, tmp_path)
        assert [c.crf for c in model.image_to_video.calls[0]["conditions"]] == [crf, crf]

    @pytest.mark.parametrize("upscale", [False, True])
    def test_every_pass_keeps_the_stock_scheduler_whatever_the_sampler(self, monkeypatch, tmp_path, upscale):
        seen: list[bool] = []
        real = video_diffusers_module._pass_scheduler

        def _spy(*args, **kwargs):
            seen.append(kwargs["conditioned"])
            return real(*args, **kwargs)

        monkeypatch.setattr(video_diffusers_module, "_pass_scheduler", _spy)
        if upscale:
            _patch_upsampler_io(monkeypatch)
        options = {"upscale": True, "latent_upsampler": _UPSAMPLER_SOURCE} if upscale else {}
        model = self._load(monkeypatch, repo="unsloth/LTX-2.5-FP8@" + _PIN, sampler="euler_ancestral", **options)

        self._run(model, tmp_path, stage1_steps=8 if upscale else None)

        assert seen == [True, True] if upscale else [True]

    def test_the_refine_pass_pins_both_images_too(self, monkeypatch, tmp_path):
        _patch_upsampler_io(monkeypatch)
        model = self._load(monkeypatch, upscale=True, latent_upsampler=_UPSAMPLER_SOURCE)
        self._run(model, tmp_path, stage1_steps=8)
        assert [[c.index for c in call["conditions"]] for call in model.image_to_video.calls] == [[0, -1], [0, -1]]
        assert model.image_to_video.keyframe_pixels == [48, 48]

    @pytest.mark.parametrize("loaded_for_end_image", [True, False])
    def test_a_clip_that_does_not_match_the_load_is_refused(self, monkeypatch, tmp_path, loaded_for_end_image):
        model = self._load(monkeypatch, end_image=loaded_for_end_image)
        first, last = self._images(tmp_path)
        with pytest.raises(RuntimeError, match="end_image="):
            DiffusersVideoBackend().image_to_video(model, first, "p", 64, 32, 49, 7, 8, str(tmp_path / "out.mp4"), **({} if loaded_for_end_image else {"end_image": last}))
        assert model.image_to_video.calls == []


class TestUpscaleReview:
    """The half-size render is decoded to a temporary video and shown to ``review`` before stage 2 starts."""

    @staticmethod
    def _recording_runtime():
        """Return a runtime whose encoder writes a file and notes each call, for the preview and the final export."""
        runtime = _make_runtime()
        calls: list[Path] = []

        def _encode(video, fps, output_path, audio=None, audio_sample_rate=None):
            Path(output_path).write_bytes(b"mp4")
            calls.append(Path(output_path))

        runtime.encode_video = _encode
        return runtime, calls

    def test_review_sees_the_decoded_half_size_render_before_stage_2(self, monkeypatch, tmp_path):
        runtime, encoded = self._recording_runtime()
        model = _load_upscale_model(monkeypatch, low_memory=False, runtime=runtime)
        seen: list[dict] = []

        def review(path):
            seen.append({"path": path, "exists": path.is_file(), "calls_so_far": len(model.text_to_video.calls), "upsampled": len(model.upsample_pipeline.calls)})

        DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, review=review, **_text_to_video_args(tmp_path))

        (shown,) = seen
        assert shown["exists"] and shown["path"].suffix == ".mp4"
        assert shown["calls_so_far"] == 1  # only stage 1 had run
        assert shown["upsampled"] == 0
        assert len(model.text_to_video.calls) == 2  # stage 2 followed
        assert encoded[0] == shown["path"] and encoded[1] == tmp_path / "upscaled.mp4"

    def test_the_preview_file_is_deleted_when_review_returns(self, monkeypatch, tmp_path):
        runtime, _ = self._recording_runtime()
        model = _load_upscale_model(monkeypatch, low_memory=False, runtime=runtime)
        shown: list[Path] = []

        DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, review=shown.append, **_text_to_video_args(tmp_path))

        assert not shown[0].exists() and not shown[0].parent.exists()

    def test_a_review_that_raises_skips_stage_2_and_the_export(self, monkeypatch, tmp_path):
        runtime, encoded = self._recording_runtime()
        model = _load_upscale_model(monkeypatch, low_memory=False, runtime=runtime)
        shown: list[Path] = []

        class Discarded(Exception):
            pass

        def review(path):
            shown.append(path)
            raise Discarded

        with pytest.raises(Discarded):
            DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, review=review, **_text_to_video_args(tmp_path))

        assert len(model.text_to_video.calls) == 1  # stage 1 only
        assert model.upsample_pipeline.calls == []
        assert encoded == shown  # the preview was the only encode
        assert not shown[0].exists()
        assert not (tmp_path / "upscaled.mp4").exists()
        assert runtime.torch.cuda.empty_cache_calls == 1  # memory handed back even though the clip was discarded

    def test_image_to_video_reviews_too(self, monkeypatch, tmp_path):
        runtime, _ = self._recording_runtime()
        model = _load_upscale_model(monkeypatch, low_memory=False, mode="i2v", runtime=runtime)
        image = tmp_path / "in.png"
        Image.new("RGB", (64, 64)).save(image)
        shown: list[Path] = []

        DiffusersVideoBackend().image_to_video(model=model, image_path=str(image), stage1_steps=4, review=shown.append, **_text_to_video_args(tmp_path))

        assert len(shown) == 1
        assert len(model.image_to_video.calls) == 2

    @pytest.mark.parametrize(("low_memory", "expects_offload"), [(True, True), (False, False)])
    def test_low_memory_sends_the_codecs_back_to_the_cpu_before_the_wait_for_the_decision(self, monkeypatch, tmp_path, low_memory, expects_offload):
        runtime, _ = self._recording_runtime()
        model = _load_upscale_model(monkeypatch, low_memory=low_memory, runtime=runtime)
        events: list[str] = []
        vae = model.text_to_video.vae
        vae.to = lambda device: events.append(f"vae->{device}")

        DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, review=lambda path: events.append("review"), **_text_to_video_args(tmp_path))

        assert (events[:2] == ["vae->cpu", "review"]) is expects_offload
        assert ("review" in events) and (events.index("review") == (1 if expects_offload else 0))

    def test_without_review_nothing_is_decoded_before_the_final_export(self, monkeypatch, tmp_path):
        runtime, encoded = self._recording_runtime()
        model = _load_upscale_model(monkeypatch, low_memory=False, runtime=runtime)

        DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, **_text_to_video_args(tmp_path))

        assert encoded == [tmp_path / "upscaled.mp4"]


# ── Converted LTX folders (ziv model model --model-type ltx-2.3) ─────────────


def _converted_folder(tmp_path: Path, bits: int = 8) -> Path:
    from zvisiongenerator.utils.ltx_converted import write_converted_ltx

    folder = tmp_path / "myft"
    (folder / "transformer").mkdir(parents=True)
    (folder / "model_index.json").write_text(json.dumps({"_class_name": "LTX2Pipeline"}))
    write_converted_ltx(folder, {"bits": bits, "storage": {4: "nf4", 8: "fp8-e4m3"}[bits], "block_matrices": 10 * 1024**3, "transformer_rest": 0})
    return folder


def test_a_converted_folder_loads_its_own_weights_and_level(monkeypatch, fake_weights, tmp_path):
    _patch_runtime(monkeypatch, _make_runtime())
    folder = _converted_folder(tmp_path, bits=8)
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.memory_estimate.converted_ltx_system_estimate", lambda model_dir, manifest: ("estimate", model_dir, manifest["bits"]))
    monkeypatch.setattr(video_diffusers_module, "_download_hub_file", lambda *a: pytest.fail("a converted folder never reads the configured transformer"))

    # The alias table has no level for this name, and the configured transformer is ignored.
    model, _ = DiffusersVideoBackend().load_model(str(folder), mode="t2v", quantize=None, transformer_source=("org/ltx", "stock.safetensors"), model_name="myft")

    assert fake_weights.prepared == []  # no <name>@q<bits> copy is written
    assert fake_weights.transformers == [{"prepared": folder / "transformer", "bits": 8}]
    assert model.text_encoder.prepared == folder / "text_encoder"
    assert fake_weights.memory_checks == [(folder, 8)]


def test_a_converted_q8_folder_that_does_not_fit_advises_converting_again_at_q4(monkeypatch, tmp_path):
    from zvisiongenerator.backends.cuda.video.memory_estimate import converted_ltx_system_estimate

    _patch_memory(monkeypatch, total_gib=12)
    estimate = converted_ltx_system_estimate(_converted_folder(tmp_path, bits=8), {"bits": 8, "block_matrices": 10 * 1024**3, "transformer_rest": 0})

    with pytest.raises(RuntimeError, match="--quantize 4") as raised:
        real_check_system_memory(estimate, 8, converted=True)
    assert "ltx-4" not in str(raised.value)


def test_a_stock_converted_folder_is_estimated_like_the_alias(monkeypatch, tmp_path):
    from zvisiongenerator.backends.cuda.video.memory_estimate import converted_ltx_system_estimate

    _patch_memory(monkeypatch, total_gib=64, block_matrices=123, rest=45, connectors=6, text_encoder=7)

    stock = real_ltx_system_estimate(Path("/fake/repo"), Path("/fake/ckpt"), 4)
    converted = converted_ltx_system_estimate(tmp_path, {"bits": 4, "block_matrices": 123, "transformer_rest": 45})

    assert converted == stock


def test_text_encoder_copies_are_reused_from_converted_folders_and_aliases(tmp_path):
    from zvisiongenerator.backends.cuda.video.weights import prepared_text_encoders

    for name in ("ltx-4@q4", "myft", ".myft.1234.partial", "zit"):
        (tmp_path / name / "text_encoder").mkdir(parents=True)
    for name in ("ltx-4@q4", "myft", ".myft.1234.partial"):
        (tmp_path / name / "text_encoder" / "ziv-quant.json").write_text("{}")

    assert prepared_text_encoders(tmp_path) == [tmp_path / "ltx-4@q4" / "text_encoder", tmp_path / "myft" / "text_encoder"]


# --- LTX-2.5: revisions, absent optional components, the ancestral sampler and the image CRF ---

_LTX25_REPO = "unsloth/LTX-2.5-FP8@" + _PIN
_LTX23_INDEX = {
    "_class_name": "LTX2Pipeline",
    "connectors": ["ltx2", "LTX2TextConnectors"],
    "text_encoder": ["transformers", "Gemma3ForConditionalGeneration"],
    "transformer": ["ltx2", "LTX2VideoTransformer3DModel"],
}
_LTX25_INDEX = {
    **_LTX23_INDEX,
    "text_encoder": ["transformers", "Gemma4UnifiedForConditionalGeneration"],
    "processor": ["transformers", "Gemma4Processor"],
    "prompt_enhancer": ["ltx2", "LTX2PromptEnhancer"],
    "duration_head": ["ltx2", "LTX2DurationHead"],
    "diffusion_decoder": ["ltx2", "LTX2DiffusionDecoder"],
}
_ABSENT = {"processor": None, "prompt_enhancer": None, "duration_head": None}


def _write_index(folder: Path, index: dict[str, Any] | str) -> Path:
    (folder / "model_index.json").write_text(index if isinstance(index, str) else json.dumps(index), encoding="utf-8")
    return folder


@pytest.fixture
def snapshot_calls(monkeypatch) -> list[tuple[str, dict[str, Any]]]:
    import huggingface_hub

    calls: list[tuple[str, dict[str, Any]]] = []

    def _snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return "/cache/snapshot"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _snapshot_download)
    return calls


class TestRepoRevisions:
    def test_a_pinned_repo_downloads_that_revision_without_the_extra_patterns(self, snapshot_calls):
        extra = ("LTX-2.5-*.safetensors", "samples/*")

        assert _repo_dir(_LTX25_REPO, with_transformer=True, extra_ignore=extra) == Path("/cache/snapshot")

        assert snapshot_calls == [("unsloth/LTX-2.5-FP8", {"revision": _PIN, "ignore_patterns": list(extra)})]

    def test_the_extra_patterns_follow_the_separate_transformer_ones(self, snapshot_calls):
        from zvisiongenerator.backends.cuda.download_plan import SEPARATE_TRANSFORMER_IGNORE

        _repo_dir(_LTX25_REPO, with_transformer=False, extra_ignore=("samples/*",))

        assert snapshot_calls[0][1]["ignore_patterns"] == [*SEPARATE_TRANSFORMER_IGNORE, "samples/*"]

    def test_an_unpinned_repo_passes_no_revision_and_keeps_the_2_3_patterns(self, snapshot_calls):
        from zvisiongenerator.backends.cuda.download_plan import SEPARATE_TRANSFORMER_IGNORE

        _repo_dir("owner/ltx", with_transformer=True)
        _repo_dir("owner/ltx", with_transformer=False)

        assert snapshot_calls == [
            ("owner/ltx", {"ignore_patterns": None}),
            ("owner/ltx", {"ignore_patterns": list(SEPARATE_TRANSFORMER_IGNORE)}),
        ]

    def test_a_local_folder_is_used_as_it_is(self, snapshot_calls, tmp_path):
        assert _repo_dir(str(tmp_path), with_transformer=True) == tmp_path
        assert snapshot_calls == []

    def test_a_failed_download_names_the_reference(self, monkeypatch):
        import huggingface_hub

        def _fail(*_args, **_kwargs):
            raise OSError("offline")

        monkeypatch.setattr(huggingface_hub, "snapshot_download", _fail)

        with pytest.raises(RuntimeError, match="offline") as excinfo:
            _repo_dir(_LTX25_REPO, with_transformer=True)
        assert _LTX25_REPO in str(excinfo.value)

    @pytest.mark.parametrize(("repo_id", "expected"), [("owner/up@" + _PIN, {"revision": _PIN}), ("owner/up", {})])
    def test_the_upsampler_download_takes_the_revision_of_its_reference(self, monkeypatch, repo_id, expected):
        import huggingface_hub

        monkeypatch.setattr(video_diffusers_module, "_download_hub_file", real_download_hub_file)
        calls: list[dict[str, str]] = []
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: calls.append(kwargs) or "/cache/up.safetensors")

        assert _resolve_upsampler_weights(repo_id, "up.safetensors") == "/cache/up.safetensors"

        assert calls == [{"repo_id": "owner/up", "filename": "up.safetensors", **expected}]


class TestRepoDirArguments:
    @pytest.fixture
    def repo_dir_calls(self, monkeypatch) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []

        def _repo_dir_fake(model_path, **kwargs):
            calls.append(kwargs)
            return Path("/fake/repo")

        monkeypatch.setattr(video_diffusers_module, "_repo_dir", _repo_dir_fake)
        return calls

    def test_a_load_without_extra_patterns_passes_no_extra_ignore(self, monkeypatch, repo_dir_calls):
        _patch_runtime(monkeypatch, _make_runtime())

        DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v")
        DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", repo_ignore=())

        assert repo_dir_calls == [{"with_transformer": True}, {"with_transformer": True}]

    def test_the_configured_patterns_reach_the_repo_download(self, monkeypatch, repo_dir_calls):
        _patch_runtime(monkeypatch, _make_runtime())

        DiffusersVideoBackend().load_model(_LTX25_REPO, quantize=4, mode="t2v", repo_ignore=["LTX-2.5-*.safetensors", "samples/*"])

        assert repo_dir_calls == [{"with_transformer": True, "extra_ignore": ("LTX-2.5-*.safetensors", "samples/*")}]


class TestAbsentOptionalComponents:
    def test_an_ltx_2_5_index_leaves_out_the_three_components_the_app_never_loads(self, tmp_path):
        assert _absent_optional_components(_write_index(tmp_path, _LTX25_INDEX)) == _ABSENT

    def test_an_ltx_2_3_index_lists_none(self, tmp_path):
        assert _absent_optional_components(_write_index(tmp_path, _LTX23_INDEX)) == {}

    def test_only_the_components_the_index_lists_are_left_out(self, tmp_path):
        index = {**_LTX23_INDEX, "processor": ["transformers", "Gemma4Processor"]}

        assert _absent_optional_components(_write_index(tmp_path, index)) == {"processor": None}

    @pytest.mark.parametrize("content", [None, "{not json", "[]", '"text"'])
    def test_a_missing_or_unreadable_index_lists_none(self, tmp_path, content):
        if content is not None:
            _write_index(tmp_path, content)

        assert _absent_optional_components(tmp_path) == {}

    def test_the_pipeline_loads_a_2_5_snapshot_with_those_components_set_to_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(video_diffusers_module, "_repo_dir", lambda model_path, **_kwargs: _write_index(tmp_path, _LTX25_INDEX))
        _patch_runtime(monkeypatch, _make_runtime())

        model, _ = DiffusersVideoBackend().load_model(_LTX25_REPO, quantize=4, mode="t2v")

        assert model.text_to_video.load_kwargs == {"transformer": None, "connectors": "connectors", "text_encoder": None, "torch_dtype": "bfloat16", **_ABSENT}

    @pytest.mark.parametrize("index", [_LTX23_INDEX, None])
    def test_a_2_3_load_passes_exactly_the_kwargs_it_always_did(self, monkeypatch, tmp_path, index):
        if index is not None:
            _write_index(tmp_path, index)
        monkeypatch.setattr(video_diffusers_module, "_repo_dir", lambda model_path, **_kwargs: tmp_path)
        _patch_runtime(monkeypatch, _make_runtime())

        model, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v")

        assert model.text_to_video.load_kwargs == {"transformer": None, "connectors": "connectors", "text_encoder": None, "torch_dtype": "bfloat16"}


class TestLoadOptions:
    def test_an_unknown_sampler_is_refused_before_anything_loads(self, monkeypatch):
        monkeypatch.setattr(video_diffusers_module, "_load_runtime_dependencies", lambda: pytest.fail("the runtime loaded"))

        with pytest.raises(ValueError, match="sampler"):
            DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="t2v", sampler="bogus")

    def test_the_sampler_and_the_image_crf_are_kept_on_the_loaded_model(self, monkeypatch):
        _patch_runtime(monkeypatch, _make_runtime())

        model, _ = DiffusersVideoBackend().load_model(_LTX25_REPO, quantize=4, mode="i2v", sampler="euler_ancestral", image_crf=18)
        plain, _ = DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode="i2v")

        assert (model.sampler, model.image_crf) == ("euler_ancestral", 18)
        assert (plain.sampler, plain.image_crf) == (None, None)

    def test_a_load_without_a_quantize_level_names_every_alias(self):
        with pytest.raises(ValueError, match="quantize level") as excinfo:
            DiffusersVideoBackend().load_model("owner/some-model", mode="t2v")

        for alias in ("ltx-4", "ltx-8", "ltx25-4"):
            assert alias in str(excinfo.value)


class _AncestralFactory:
    """Stands in for ``sampler.ancestral_scheduler``: records each call and returns a new scheduler object."""

    def __init__(self):
        self.calls: list[SimpleNamespace] = []
        self.made: list[SimpleNamespace] = []

    def __call__(self, generator, *, num_train_timesteps=1000):
        self.calls.append(SimpleNamespace(generator=generator, num_train_timesteps=num_train_timesteps))
        self.made.append(SimpleNamespace(name=f"ancestral-{len(self.made) + 1}"))
        return self.made[-1]


@pytest.fixture
def ancestral(monkeypatch) -> _AncestralFactory:
    factory = _AncestralFactory()
    monkeypatch.setattr(_sampler_module, "ancestral_scheduler", factory)
    return factory


def _stock_scheduler() -> _FakeScheduler:
    return _FakeScheduler(config=SimpleNamespace(num_train_timesteps=1000))


class TestAncestralSampler:
    @staticmethod
    def _load(monkeypatch, *, mode="t2v", sampler="euler_ancestral", upscale=False, repo="diffusers/LTX-2.3-Distilled-Diffusers"):
        _patch_runtime(monkeypatch, _make_runtime())
        options: dict[str, Any] = {"quantize": 4, "mode": mode, "low_memory": False, "sampler": sampler}
        if upscale:
            _patch_upsampler_io(monkeypatch)
            options.update(upscale=True, latent_upsampler=_UPSAMPLER_SOURCE)
        model, _ = DiffusersVideoBackend().load_model(repo, **options)
        pipeline = model.text_to_video or model.image_to_video
        pipeline.scheduler = _stock_scheduler()
        return model, pipeline

    @staticmethod
    def _image(tmp_path) -> str:
        path = tmp_path / "input.png"
        Image.new("RGB", (8, 8), "blue").save(path)
        return str(path)

    def test_a_one_stage_t2v_pass_runs_on_a_fresh_scheduler_seeded_for_stage_1(self, monkeypatch, tmp_path, ancestral):
        model, pipeline = self._load(monkeypatch)
        stock = pipeline.scheduler

        DiffusersVideoBackend().text_to_video(model=model, **_text_to_video_args(tmp_path))

        (factory_call,) = ancestral.calls
        assert factory_call.generator.seed == 77 + 10_000
        assert factory_call.num_train_timesteps == 1000
        assert pipeline.schedulers_at_call == ancestral.made
        assert pipeline.scheduler is stock

    def test_the_initial_noise_keeps_the_user_seed_and_the_sigmas_are_the_2_3_ones(self, monkeypatch, tmp_path, ancestral):
        model, pipeline = self._load(monkeypatch)
        events: list[dict[str, Any]] = []

        DiffusersVideoBackend().text_to_video(model=model, step_callback=events.append, **_text_to_video_args(tmp_path))

        (call,) = pipeline.calls
        assert call["generator"].seed == 77
        assert call["generator"] is not ancestral.calls[0].generator
        assert call["sigmas"] == _stage1_sigmas(_DISTILLED_SIGMAS, 8)
        assert call["num_inference_steps"] == 8
        assert [event["total_steps"] for event in events] == [8] * 8

    def test_a_one_stage_i2v_pass_runs_on_a_fresh_scheduler_seeded_for_stage_1(self, monkeypatch, tmp_path, ancestral):
        model, pipeline = self._load(monkeypatch, mode="i2v")
        stock = pipeline.scheduler

        DiffusersVideoBackend().image_to_video(model=model, image_path=self._image(tmp_path), **_text_to_video_args(tmp_path))

        assert [call.generator.seed for call in ancestral.calls] == [77 + 10_000]
        assert pipeline.schedulers_at_call == ancestral.made
        assert pipeline.scheduler is stock

    def test_each_stage_of_a_two_stage_run_gets_its_own_scheduler_and_generator(self, monkeypatch, tmp_path, ancestral):
        model, pipeline = self._load(monkeypatch, upscale=True)
        stock = pipeline.scheduler
        monkeypatch.setattr(video_diffusers_module, "_refine_scheduler", lambda _pipeline: pytest.fail("the static-shift scheduler was used"))
        events: list[dict[str, Any]] = []

        DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, step_callback=events.append, **_text_to_video_args(tmp_path))

        stage1, stage2 = pipeline.calls
        assert [call.generator.seed for call in ancestral.calls] == [77 + 10_000, 77 + 20_000]
        assert ancestral.calls[0].generator is not ancestral.calls[1].generator
        assert ancestral.made[0] is not ancestral.made[1]
        assert pipeline.schedulers_at_call == ancestral.made
        assert pipeline.scheduler is stock
        assert stage1["generator"] is stage2["generator"] and stage1["generator"].seed == 77
        assert stage1["sigmas"] == _stage1_sigmas(_DISTILLED_SIGMAS, 4) and stage2["sigmas"] == list(_STAGE2_SIGMAS)
        assert (stage1["num_inference_steps"], stage2["num_inference_steps"]) == (4, 3)
        assert {event["total_steps"] for event in events} == {7}

    def test_the_stock_scheduler_comes_back_when_stage_2_fails(self, monkeypatch, tmp_path, ancestral):
        model, pipeline = self._load(monkeypatch, upscale=True)
        stock = pipeline.scheduler

        def _cancel_in_stage2(event):
            if event["phase"] == "video_upscale_stage_2":
                raise KeyboardInterrupt("quit")

        with pytest.raises(KeyboardInterrupt):
            DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, step_callback=_cancel_in_stage2, **_text_to_video_args(tmp_path))

        assert pipeline.schedulers_at_call == ancestral.made
        assert len(ancestral.made) == 2
        assert pipeline.scheduler is stock

    def test_the_scheduler_scale_comes_from_the_pipelines_own_scheduler(self, monkeypatch, tmp_path, ancestral):
        model, pipeline = self._load(monkeypatch)
        pipeline.scheduler = _FakeScheduler(config=SimpleNamespace(num_train_timesteps=500))

        DiffusersVideoBackend().text_to_video(model=model, **_text_to_video_args(tmp_path))

        assert ancestral.calls[0].num_train_timesteps == 500

    def test_a_run_without_a_sampler_never_builds_the_ancestral_scheduler(self, monkeypatch, tmp_path, ancestral):
        model, pipeline = self._load(monkeypatch, sampler=None, upscale=True)
        stock = pipeline.scheduler

        DiffusersVideoBackend().text_to_video(model=model, stage1_steps=4, **_text_to_video_args(tmp_path))

        assert ancestral.calls == []
        stage1_scheduler, stage2_scheduler = pipeline.schedulers_at_call
        assert stage1_scheduler is stock
        assert stage2_scheduler.overrides == {"use_dynamic_shifting": False, "shift_terminal": None}
        assert pipeline.scheduler is stock

    def test_a_conditioned_pass_keeps_the_stock_scheduler_even_for_the_refinement(self):
        model = SimpleNamespace(sampler="euler_ancestral", runtime=SimpleNamespace(torch=_FakeTorch()))
        pipeline = SimpleNamespace(scheduler=_stock_scheduler())
        stock = pipeline.scheduler

        with _pass_scheduler(model, pipeline, seed=1, stage=1, refine=False, conditioned=True):
            assert pipeline.scheduler is stock
        with _pass_scheduler(model, pipeline, seed=1, stage=2, refine=True, conditioned=True):
            assert pipeline.scheduler is not stock
            assert pipeline.scheduler.overrides == {"use_dynamic_shifting": False, "shift_terminal": None}
        assert pipeline.scheduler is stock


class TestImageCrf:
    @pytest.fixture
    def order(self, monkeypatch) -> list[tuple[str, Any]]:
        events: list[tuple[str, Any]] = []
        real_fit = video_diffusers_module._fit_image

        def _recompress(image, crf):
            events.append(("recompress", (image.mode, image.size, crf)))
            return Image.new("RGB", image.size, "red")

        def _fit(image, width, height):
            events.append(("fit", (image.getpixel((0, 0)), width, height)))
            return real_fit(image, width, height)

        monkeypatch.setattr(video_diffusers_module, "recompress", _recompress)
        monkeypatch.setattr(video_diffusers_module, "_fit_image", _fit)
        return events

    @staticmethod
    def _run(monkeypatch, tmp_path, **options):
        _patch_runtime(monkeypatch, _make_runtime())
        model, _ = DiffusersVideoBackend().load_model(_LTX25_REPO, quantize=4, mode="i2v", low_memory=False, **options)
        input_path = tmp_path / "input.png"
        Image.new("RGBA", (8, 6), "blue").save(input_path)
        DiffusersVideoBackend().image_to_video(model=model, image_path=str(input_path), **_text_to_video_args(tmp_path))
        return model.image_to_video.calls

    def test_the_image_is_recompressed_then_fitted_and_the_pipeline_skips_its_own_pass(self, monkeypatch, tmp_path, order):
        (call,) = self._run(monkeypatch, tmp_path, image_crf=18)

        assert order == [("recompress", ("RGB", (8, 6), 18)), ("fit", ((255, 0, 0), 512, 384))]
        assert call["image_crf"] == 0
        assert call["image"].size == (512, 384)

    def test_a_two_stage_run_passes_the_skip_to_both_stages(self, monkeypatch, tmp_path, order):
        _patch_runtime(monkeypatch, _make_runtime())
        _patch_upsampler_io(monkeypatch)
        model, _ = DiffusersVideoBackend().load_model(_LTX25_REPO, quantize=4, mode="i2v", low_memory=False, image_crf=18, upscale=True, latent_upsampler=_UPSAMPLER_SOURCE)
        input_path = tmp_path / "input.png"
        Image.new("RGB", (8, 6), "blue").save(input_path)

        DiffusersVideoBackend().image_to_video(model=model, image_path=str(input_path), stage1_steps=4, **_text_to_video_args(tmp_path))

        assert [call["image_crf"] for call in model.image_to_video.calls] == [0, 0]
        assert [name for name, _ in order] == ["recompress", "fit"]

    def test_without_a_crf_nothing_is_recompressed_and_the_pipeline_gets_no_crf_kwarg(self, monkeypatch, tmp_path):
        monkeypatch.setattr(video_diffusers_module, "recompress", lambda *_args: pytest.fail("the image was recompressed"))

        (call,) = self._run(monkeypatch, tmp_path)

        assert "image_crf" not in call
        assert call["image"].size == (512, 384)
