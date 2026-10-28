"""Shared fakes, fixtures and builders for the diffusers LTX video backend tests (``test_cuda_video*.py``).

Each test module wraps ``install_fake_weights`` in an autouse ``fake_weights`` fixture; the fixtures are not shared as imports so that no lint rule sees a redefinition.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np

import zvisiongenerator.backends.cuda.video.backend as video_diffusers_module
from zvisiongenerator.backends.cuda.video.backend import DiffusersVideoBackend

# The originals, which the fake_weights fixture replaces on the module; the test modules import them from here.
from zvisiongenerator.backends.cuda.video.backend import _check_system_memory as real_check_system_memory  # noqa: F401
from zvisiongenerator.backends.cuda.video.backend import _download_hub_file as real_download_hub_file  # noqa: F401
from zvisiongenerator.backends.cuda.video.backend import _encode_prompt as real_encode_prompt  # noqa: F401
from zvisiongenerator.backends.cuda.video.backend import _place_components as real_place_components  # noqa: F401
from zvisiongenerator.backends.cuda.video.backend import _place_transformer as real_place_transformer  # noqa: F401
from zvisiongenerator.backends.cuda.video.backend import _prepared_weights as real_prepared_weights  # noqa: F401
from zvisiongenerator.backends.cuda.video.memory_estimate import ltx_system_estimate as real_ltx_system_estimate  # noqa: F401

# The originals, which the fake_weights fixture replaces on the module.

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


def _clear_pipeline_state():
    for pipeline_class in (_FakeTextPipeline, _FakeImagePipeline, _FakeConditionPipeline, _NoScalePipeline, _AudioTextPipeline, _NoRatePipeline):
        pipeline_class.loaded_instances = []
    _FakeUpsamplerModel.instances = []
    _FakeUpsamplerModel.load_error = None


def install_fake_weights(monkeypatch):
    """Keep load_model off the network and disk: fake repo folder, checkpoint, memory check, weights and placement.

    Also resets the fake pipelines' recorded instances and keeps ``load_model`` from writing the CUDA allocator variables into ``os.environ``
    (``test_cuda_memory`` owns that behaviour). Each test module wraps this in its own autouse ``fake_weights`` fixture.
    """
    _clear_pipeline_state()
    monkeypatch.setattr(video_diffusers_module, "_configure_torch_runtime", lambda torch: None)

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


def _text_to_video_args(tmp_path, **overrides):
    return {"prompt": "a city", "width": 512, "height": 384, "num_frames": 49, "seed": 77, "steps": 8, "output_path": str(tmp_path / "upscaled.mp4"), **overrides}


def _patch_memory(monkeypatch, *, total_gib, block_matrices=10 * 1024**3, rest=0, connectors=0, text_encoder=0):
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(total=total_gib * 1024**3))
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.weights.transformer_elements", lambda source: (block_matrices, rest))
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.weights.component_elements", lambda path: connectors)
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.weights.text_encoder_bytes", lambda path: text_encoder)


def _converted_folder(tmp_path: Path, bits: int = 8) -> Path:
    from zvisiongenerator.utils.ltx_converted import write_converted_ltx

    folder = tmp_path / "myft"
    (folder / "transformer").mkdir(parents=True)
    (folder / "model_index.json").write_text(json.dumps({"_class_name": "LTX2Pipeline"}))
    write_converted_ltx(folder, {"bits": bits, "storage": {4: "nf4", 8: "fp8-e4m3"}[bits], "block_matrices": 10 * 1024**3, "transformer_rest": 0})
    return folder
