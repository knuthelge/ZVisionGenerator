"""The Stable Audio 3 backend: the fp16 tensor-by-tensor load, the arguments ``generate`` gets and the unclamped waveform it returns.

The vendored package is real, but its model is a tiny stand-in built by a patched factory and its ``generate`` is a stand-in
that follows the library's contract: it adapts the length to ``sample_size``, calls the step callback, decodes through
``pretransform.decode`` and returns the clamped, truncated result. No real model is loaded and CUDA is never touched.
"""

from __future__ import annotations

import gc
import json
import sys
import weakref
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file
from stable_audio_3.model import StableAudioModel
from torch import nn

import zvisiongenerator.backends as backends_module
from zvisiongenerator.backends.audio import sa3
from zvisiongenerator.backends.audio.sa3 import FAMILY, StableAudio3Backend
from zvisiongenerator.core.audio_backend import AudioCancelled
from zvisiongenerator.core.audio_types import AudioGenerationRequest
from zvisiongenerator.utils.audio_preset import AudioPreset, parse_audio_presets
from tests._torch_flags import current_flags, set_app_flags, set_kit_flags
from tests._audio_presets import SA3_CHECKPOINT_REPO, SA3_CONFIG_REPO, SA3_TEXT_ENCODER_REPO, audio_config, sa3_preset

SAMPLE_RATE = 44100
SAMPLE_SIZE = 16777216
CLAMP_PEAK = 2.0


class _Pretransform(nn.Module):
    """A decoder whose output peaks at ``CLAMP_PEAK``, above the library's clamp, for ``frames`` samples."""

    def __init__(self) -> None:
        super().__init__()
        self.frames = 0

    def decode(self, latents: torch.Tensor, chunked: bool | None = None) -> torch.Tensor:
        """Return a stereo fp16 waveform of ``frames`` samples with one sample at ``CLAMP_PEAK``."""
        waveform = torch.full((1, 2, self.frames), 0.5, dtype=torch.float16)
        waveform[..., 0] = CLAMP_PEAK
        return waveform


class _Diffusion(nn.Module):
    """The model ``create_diffusion_cond_from_config`` returns: a diffusion backbone, a decoder and the call order of ``half`` and ``to``."""

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events
        self.model = nn.Linear(2, 3)
        self.pretransform = _Pretransform()
        self.sample_rate = SAMPLE_RATE

    def half(self) -> _Diffusion:
        """Record the call, then halve."""
        self.events.append("half")
        return super().half()

    def to(self, device: str, *args: Any, **kwargs: Any) -> _Diffusion:  # type: ignore[override]
        """Record the target device and stay where the stand-in is."""
        self.events.append(f"to {device}")
        return self


@pytest.fixture
def events() -> list[str]:
    return []


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> list[Any]:
    """Patch the factory to build a stand-in model; the list collects each ``(config, model)`` built."""
    results: list[Any] = []

    def factory(config: dict[str, Any]) -> _Diffusion:
        model = _Diffusion(events)
        results.append((config, model))
        return model

    monkeypatch.setattr("stable_audio_3.factory.create_diffusion_cond_from_config", factory)
    return results


@pytest.fixture
def generated(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Patch ``StableAudioModel.generate`` with a stand-in that follows the library's contract; the list collects its calls."""
    calls: list[dict[str, Any]] = []

    def generate(self: StableAudioModel, prompt: str, duration: float, steps: int, cfg_scale: float, seed: int, sample_size: int = 5292032, callback: Any = None, **rest: Any) -> torch.Tensor:
        calls.append({"prompt": prompt, "duration": duration, "steps": steps, "cfg_scale": cfg_scale, "seed": seed, "sample_size": sample_size, "callback": callback, **rest})
        for index in range(steps):
            if callback is not None:
                callback({"i": index})
        # The library pads the length by 6 s, caps it at sample_size and truncates the result to the duration.
        self.model.pretransform.frames = min(int((duration + 6) * SAMPLE_RATE), sample_size)
        decoded = self.model.pretransform.decode(torch.zeros(1, 1, 1))
        return decoded.to(torch.float32).clamp(-1, 1)[:, :, : int(duration * SAMPLE_RATE)]

    monkeypatch.setattr(StableAudioModel, "generate", generate)
    return calls


@pytest.fixture
def checkpoint_tensors() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return {"model.weight": torch.randn(3, 2, generator=generator), "model.bias": torch.randn(3, generator=generator)}


def _config_json() -> dict[str, Any]:
    return {
        "sample_size": SAMPLE_SIZE,
        "sample_rate": SAMPLE_RATE,
        "model": {
            "conditioning": {
                "configs": [
                    {"id": "prompt", "type": "t5gemma", "config": {"model_name": "google/t5gemma-b-b-ul2", "repo_id": "somebody/elsewhere", "subfolder": "t5"}},
                    {"id": "seconds_total", "type": "number", "config": {"min_val": 0, "max_val": 512}},
                ]
            }
        },
    }


@pytest.fixture
def snapshots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpoint_tensors: dict[str, torch.Tensor]) -> Path:
    """Lay the three repositories out as Hub snapshots and make the backend find them there."""
    root = tmp_path / "hub"
    config_dir, checkpoint_dir, encoder_dir = (root / repo.split("@")[0].replace("/", "--") for repo in (SA3_CONFIG_REPO, SA3_CHECKPOINT_REPO, SA3_TEXT_ENCODER_REPO))
    (checkpoint_dir / "checkpoints").mkdir(parents=True)
    (encoder_dir / "t5gemma-b-b-ul2").mkdir(parents=True)
    config_dir.mkdir(parents=True)
    (config_dir / "model_config.json").write_text(json.dumps(_config_json()), encoding="utf-8")
    # The text encoder repository has a model_config.json of its own, which must not be used.
    (encoder_dir / "model_config.json").write_text("{}", encoding="utf-8")
    (encoder_dir / "t5gemma-b-b-ul2" / "config.json").write_text("{}", encoding="utf-8")
    save_file(checkpoint_tensors, str(checkpoint_dir / "checkpoints" / "stable_audio_3_medium.safetensors"))

    def snapshot_dirs(repo_id: str, revision: str | None) -> list[Path]:
        folder = root / repo_id.replace("/", "--")
        return [folder] if folder.is_dir() else []

    monkeypatch.setattr("zvisiongenerator.utils.hf_snapshot.cached_snapshot_dirs", snapshot_dirs)
    return root


@pytest.fixture
def preset() -> AudioPreset:
    return parse_audio_presets(audio_config(sa3=sa3_preset()))[0]


def _loaded(preset: AudioPreset, caches: list[str] | None = None) -> StableAudio3Backend:
    backend = StableAudio3Backend(empty_cache=(caches if caches is not None else []).append)
    backend.load(preset, preset.platforms["cuda"], "cuda")
    return backend


def _request(seconds: float | None = 30, seed: int = 11) -> AudioGenerationRequest:
    return AudioGenerationRequest(preset="sa3", kind="sound", prompt="steady rain on a tin roof", seconds=seconds, seed=seed)


def _take(backend: StableAudio3Backend, request: AudioGenerationRequest, progress: list[tuple] | None = None, cancelled: Any = lambda: False):
    return backend.generate(request, lambda *args: (progress if progress is not None else []).append(args), cancelled)


class TestRegistry:
    def test_the_registry_creates_this_backend(self):
        backend = backends_module.get_audio_backend(FAMILY)

        assert isinstance(backend, StableAudio3Backend)


class TestLoad:
    def test_the_weights_are_the_checkpoints_in_fp16(self, preset, snapshots, built, checkpoint_tensors):
        _loaded(preset)

        model = built[0][1]
        for name, expected in checkpoint_tensors.items():
            actual = model.state_dict()[name]
            assert actual.dtype == torch.float16
            assert torch.equal(actual, expected.half())

    def test_the_model_is_built_in_fp16_so_no_fp32_weights_exist(self, preset, snapshots, events, monkeypatch):
        """The factory runs under an fp16 default dtype, so its parameters are never fp32."""
        seen: list[tuple[torch.dtype, torch.dtype]] = []

        def factory(config: dict[str, Any]) -> _Diffusion:
            model = _Diffusion(events)
            seen.append((torch.get_default_dtype(), model.model.weight.dtype))
            return model

        monkeypatch.setattr("stable_audio_3.factory.create_diffusion_cond_from_config", factory)

        _loaded(preset)

        assert seen == [(torch.float16, torch.float16)]

    def test_the_default_dtype_is_restored_when_the_build_fails(self, preset, snapshots, monkeypatch):
        def factory(config: dict[str, Any]) -> _Diffusion:
            raise RuntimeError("boom")

        monkeypatch.setattr("stable_audio_3.factory.create_diffusion_cond_from_config", factory)

        with pytest.raises(RuntimeError, match="boom"):
            _loaded(preset)

        assert torch.get_default_dtype() == torch.float32

    def test_each_tensor_is_in_the_model_before_the_next_is_read(self, preset, snapshots, built, monkeypatch):
        """No state dict of the whole checkpoint is built: a tensor is copied in, then the next is read."""
        real = sa3.iter_tensors
        seen: list[str] = []

        def one_at_a_time(path, wanted=None):
            for name, tensor in real(path, wanted):
                yield name, tensor
                # The generator resumes only after the loop body copied this tensor.
                assert torch.equal(built[0][1].state_dict()[name], tensor.half())
                seen.append(name)

        def refuse(*args: Any, **kwargs: Any):
            raise AssertionError("the whole checkpoint was loaded at once")

        monkeypatch.setattr(sa3, "iter_tensors", one_at_a_time)
        monkeypatch.setattr("safetensors.torch.load_file", refuse)
        monkeypatch.setattr("safetensors.safe_open", refuse)
        monkeypatch.setattr("stable_audio_3.loading_utils.load_file", refuse)

        _loaded(preset)

        assert sorted(seen) == ["model.bias", "model.weight"]

    def test_a_key_with_an_extra_segment_lands_on_the_model_tensor(self, preset, snapshots, built, tmp_path, monkeypatch):
        tensor = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        monkeypatch.setattr(sa3, "iter_tensors", lambda path, wanted=None: iter([("model.wrapped.weight", tensor)]))

        _loaded(preset)

        assert torch.equal(built[0][1].state_dict()["model.weight"], tensor.half())

    def test_the_conditioner_reads_the_local_text_encoder_folder(self, preset, snapshots, built):
        _loaded(preset)

        config = built[0][0]
        t5, number = config["model"]["conditioning"]["configs"]
        assert "repo_id" not in t5["config"] and "subfolder" not in t5["config"]
        assert t5["config"]["model_path"] == str(snapshots / SA3_TEXT_ENCODER_REPO.split("@")[0].replace("/", "--") / "t5gemma-b-b-ul2")
        assert t5["config"]["model_name"] == "google/t5gemma-b-b-ul2"
        assert number["config"] == {"min_val": 0, "max_val": 512}
        assert config["sample_size"] == SAMPLE_SIZE

    def test_the_config_comes_from_the_entry_that_lists_it(self, preset, snapshots, built):
        _loaded(preset)

        assert built[0][0]["sample_rate"] == SAMPLE_RATE  # the text encoder repository's own model_config.json is "{}"

    def test_the_package_is_told_the_model_is_half_precision_without_loras(self, preset, snapshots, built):
        backend = _loaded(preset)

        assert backend._model.model_half is True
        assert backend._model.model is built[0][1]
        assert built[0][1].use_lora is False and built[0][1].lora_names == []
        assert backend._model.device == "cuda"

    def test_works_without_flash_attn(self, preset, snapshots, built, generated, monkeypatch):
        monkeypatch.setitem(sys.modules, "flash_attn", None)  # an import of it raises ImportError

        clip = _take(_loaded(preset), _request(seconds=3))

        assert clip.seconds == 3.0

    def test_a_revision_that_is_not_downloaded_is_named(self, preset, built, monkeypatch):
        monkeypatch.setattr("zvisiongenerator.utils.hf_snapshot.cached_snapshot_dirs", lambda repo_id, revision: [])

        with pytest.raises(FileNotFoundError, match="not downloaded"):
            _loaded(preset)

        assert built == []

    def test_a_file_missing_from_the_snapshot_names_its_repository(self, preset, snapshots, built):
        (snapshots / SA3_CHECKPOINT_REPO.split("@")[0].replace("/", "--") / "checkpoints" / "stable_audio_3_medium.safetensors").unlink()

        with pytest.raises(FileNotFoundError, match="Comfy-Org/stable-audio-3 at [0-9a-f]{40} is not downloaded"):
            _loaded(preset)


class TestGenerate:
    def test_generate_gets_the_presets_settings_and_the_config_sample_size(self, preset, snapshots, built, generated):
        _take(_loaded(preset), _request(seconds=30, seed=11))

        call = generated[0]
        assert (call["prompt"], call["duration"], call["steps"], call["cfg_scale"], call["seed"]) == ("steady rain on a tin roof", 30.0, 8, 1.0, 11)
        assert call["sample_size"] == SAMPLE_SIZE
        assert call["disable_tqdm"] is True

    def test_a_380_second_request_is_not_capped(self, preset, snapshots, built, generated):
        clip = _take(_loaded(preset), _request(seconds=380))

        assert generated[0]["duration"] == 380.0
        assert clip.seconds == 380.0
        assert clip.samples.shape == (2, 380 * SAMPLE_RATE)

    def test_the_waveform_is_returned_from_before_the_clamp(self, preset, snapshots, built, generated):
        clip = _take(_loaded(preset), _request(seconds=30))

        assert clip.preclamp is True
        assert clip.sample_rate == SAMPLE_RATE
        assert clip.samples.dtype.name == "float32"
        assert clip.samples.shape == (2, 30 * SAMPLE_RATE)
        assert float(abs(clip.samples).max()) == CLAMP_PEAK  # the library's own result peaks at 1.0

    def test_the_decoder_is_restored_after_a_failed_take(self, preset, snapshots, built, generated, monkeypatch):
        backend = _loaded(preset)
        pretransform = built[0][1].pretransform
        monkeypatch.setattr(StableAudioModel, "generate", lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("out of memory")))

        with pytest.raises(RuntimeError, match="out of memory"):
            _take(backend, _request(seconds=3))

        assert "decode" not in vars(pretransform)

    def test_progress_counts_the_sampling_steps_of_one_part(self, preset, snapshots, built, generated):
        progress: list[tuple] = []

        _take(_loaded(preset), _request(seconds=3), progress)

        assert progress[:9] == [("Sampling", step, 8, 1, 1) for step in range(9)]

    def test_the_decode_after_the_last_step_is_reported_as_its_own_phase(self, preset, snapshots, built, generated):
        progress: list[tuple] = []

        _take(_loaded(preset), _request(seconds=3), progress)

        assert progress[9:] == [("Decode", 0, 1, 1, 1), ("Decode", 1, 1, 1, 1)]

    def test_the_decode_starts_after_the_last_step_not_before_it(self, preset, snapshots, built, generated):
        progress: list[tuple] = []

        _take(_loaded(preset), _request(seconds=3), progress)

        assert [args[0] for args in progress].index("Decode") == 9

    def test_a_stop_between_steps_raises_and_ends_the_take(self, preset, snapshots, built, generated):
        progress: list[tuple] = []

        with pytest.raises(AudioCancelled):
            _take(_loaded(preset), _request(seconds=3), progress, cancelled=lambda: len(progress) >= 4)

        assert progress[-1] == ("Sampling", 3, 8, 1, 1)

    def test_a_stop_before_the_take_never_starts_it(self, preset, snapshots, built, generated):
        progress: list[tuple] = []

        with pytest.raises(AudioCancelled):
            _take(_loaded(preset), _request(seconds=3), progress, cancelled=lambda: True)

        assert generated == [] and progress == []


class TestClose:
    def test_close_frees_the_model_and_the_device_cache(self, preset, snapshots, built):
        caches: list[str] = []
        backend = _loaded(preset, caches)
        model = weakref.ref(built[0][1])
        built.clear()

        backend.close()
        gc.collect()

        assert model() is None
        assert caches == ["cuda"]

    def test_close_gives_the_global_torch_flags_back(self, preset, snapshots, built, torch_flags_guard):
        set_app_flags()
        before = current_flags()
        backend = _loaded(preset)  # the real StableAudioModel sets the flags in its constructor
        assert current_flags() != before

        backend.close()
        assert current_flags() == before
        backend.close()
        assert current_flags() == before

    def test_a_failed_load_gives_the_global_torch_flags_back(self, preset, snapshots, built, monkeypatch, torch_flags_guard):
        set_app_flags()
        before = current_flags()

        def fail_after_flipping(*args, **kwargs):
            set_kit_flags()
            raise OSError("disk")

        monkeypatch.setattr("stable_audio_3.model.StableAudioModel", fail_after_flipping)
        backend = StableAudio3Backend(empty_cache=lambda device: None)

        with pytest.raises(OSError, match="disk"):
            backend.load(preset, preset.platforms["cuda"], "cuda")

        assert current_flags() == before
