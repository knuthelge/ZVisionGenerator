"""Tests for MOSS-Image on the diffusers/CUDA image backend: loading, placement, the sampling run and shared stored copies.

Everything runs on the CPU with mocks and tiny modules. The module's GPU is replaced by the device ``cpu:0``, so a move
"to the GPU" differs from a move to ``"cpu"`` in the recorded events.
"""

from __future__ import annotations

import ast
import json
import os
import re
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from moss_image.pipeline_moss_image import MOSSImageScheduler, MOSSImageTransformer
from PIL import Image
from torch import nn

import zvisiongenerator.backends.cuda.image.moss as moss
from zvisiongenerator.backends.cuda.image.moss_sampler import Conditioning, initial_noise, noised
from zvisiongenerator.backends.cuda import quant
from zvisiongenerator.backends.cuda.quant import cast_to_fp8
from zvisiongenerator.core.moss import img2img_run_steps, sampler_settings
from zvisiongenerator.image_model_loader import load_image_model
from zvisiongenerator.utils.model_files import is_moss_model
from zvisiongenerator.utils.moss_build import MossSource, write_build_record
from zvisiongenerator.utils.stored_quant import is_current

_GPU = "cpu:0"
_LAYERS = 3
_TEXT_DIM = 32
_SFT = "owner/sft@" + "a" * 40
_FORMAT = "fmt-1"


@pytest.fixture(autouse=True)
def cpu_as_gpu(monkeypatch):
    monkeypatch.setattr(moss, "_DEVICE", _GPU)


@pytest.fixture
def models(monkeypatch, tmp_path) -> Path:
    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path))
    path = tmp_path / "models"
    path.mkdir()
    return path


def _tiny_transformer() -> MOSSImageTransformer:
    torch.manual_seed(0)
    return MOSSImageTransformer(dim=64, ffn_dim=128, num_heads=2, num_layers=_LAYERS, text_dim=_TEXT_DIM, freq_dim=16).eval()


# ---------------------------------------------------------------------------
# Folders on disk
# ---------------------------------------------------------------------------


def _source(sft: str = _SFT) -> MossSource:
    return MossSource(sft, "owner/lora", "b" * 40, "lora.safetensors")


def _built(models: Path, name: str, *, sft: str = _SFT) -> Path:
    """Create a built MOSS-Image folder: an index, a one-file weight set per component and a build record."""
    folder = models / name
    index = {"_class_name": "MOSSImagePipeline", "is_distilled": name == "moss", "transformer": ["pipeline_moss_image", "MOSSImageTransformer"], "text_encoder": ["transformers", "Qwen3_5Model"]}
    folder.mkdir()
    (folder / "model_index.json").write_text(json.dumps(index), encoding="utf-8")
    for component in ("transformer", "text_encoder", "vae", "tokenizer", "scheduler"):
        (folder / component).mkdir()
        (folder / component / "config.json").write_text(json.dumps({"num_layers": _LAYERS}), encoding="utf-8")
    for component in ("transformer", "text_encoder", "vae"):
        (folder / component / "model.safetensors").write_bytes(f"{name}-{component}".encode())
    (folder / "LICENSE").write_text("license", encoding="utf-8")
    write_build_record(folder, _source(sft), turbo=name == "moss")
    return folder


class _Component:
    """A quantized component whose ``save_pretrained`` writes a weight file that names the writer and the call number."""

    def __init__(self, label: str, calls: list[str]) -> None:
        self._label = label
        self._calls = calls

    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.safetensors").write_bytes(f"{self._label}-{len(self._calls)}".encode())
        (Path(path) / "config.json").write_text(json.dumps({"quantization_config": {"label": self._label}}), encoding="utf-8")


@pytest.fixture
def writers(monkeypatch):
    """Replace the quantizing loaders with fakes that count how often each runs."""
    calls: list[str] = []

    def fake(label: str):
        def load(*args, **kwargs):
            calls.append(label)
            return _Component(label, calls)

        return load

    for name, label in (("load_nf4_transformer", "nf4-transformer"), ("load_fp8_transformer", "fp8-transformer"), ("load_nf4_text_encoder", "nf4-encoder")):
        monkeypatch.setattr(moss, name, fake(label))
    monkeypatch.setattr(quant, "release_memory", lambda: None)
    return calls


# ---------------------------------------------------------------------------
# Refusal and load order
# ---------------------------------------------------------------------------


class _Part:
    """A model part that logs the devices it is moved to."""

    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log

    def to(self, device):
        self.log.append(f"{self.name}.to({device})")
        return self

    def eval(self):
        return self


@pytest.fixture
def loading(monkeypatch, models):
    """Patch every loader of ``load_pipeline`` and log the order they run in; return ``(log, loaded)``."""
    import diffusers
    import transformers
    from moss_image import pipeline_moss_image

    from zvisiongenerator.backends.cuda.image import moss_lora

    log: list[str] = []
    loaded: dict[str, object] = {}

    def record(name: str, result):
        def call(*args, **kwargs):
            log.append(name)
            loaded[name] = SimpleNamespace(args=args, kwargs=kwargs, result=result)
            return result

        return call

    def build_pipeline(**components):
        log.append("pipeline")
        loaded["pipeline"] = components
        return SimpleNamespace(**components)

    encoder, transformer, vae = _Part("encoder", log), _Part("transformer", log), _Part("vae", log)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", record("tokenizer", "tokenizer"))
    monkeypatch.setattr(pipeline_moss_image.MOSSImageScheduler, "from_pretrained", record("scheduler", "scheduler"))
    monkeypatch.setattr(diffusers.AutoencoderKLFlux2, "from_pretrained", record("vae", vae))
    monkeypatch.setattr(pipeline_moss_image.MOSSImageTransformer, "from_pretrained", record("transformer-bf16", transformer))
    monkeypatch.setattr(pipeline_moss_image, "MOSSImagePipeline", build_pipeline)
    monkeypatch.setattr(moss_lora, "install_loras", record("loras", None))
    monkeypatch.setattr(moss, "load_nf4_text_encoder", record("encoder", encoder))
    monkeypatch.setattr(moss, "load_nf4_transformer", record("transformer-nf4", transformer))
    monkeypatch.setattr(moss, "load_fp8_transformer", record("transformer-fp8", transformer))
    monkeypatch.setattr(moss, "group_offload", record("offload", None))
    return log, loaded


def _load(path: Path, bits: int | None = None, *, prequantized: bool = False, turbo_lora: Path | None = None, lora_paths=None, lora_weights=None):
    return moss.load_pipeline(str(path), torch.bfloat16, bits, prequantized=prequantized, turbo_lora=turbo_lora, lora_paths=lora_paths, lora_weights=lora_weights)


class TestRefusal:
    @pytest.mark.parametrize("reference", ["OpenMOSS-Team/MOSS-Image-SFT", "elsewhere/moss"])
    def test_a_repo_id_or_a_foreign_folder_is_refused_before_anything_loads(self, loading, tmp_path, reference):
        log, _ = loading
        path = str(tmp_path / reference) if reference.startswith("elsewhere") else reference

        with pytest.raises(RuntimeError, match=r"use -m moss \(Turbo\) or -m moss-sft instead of") as raised:
            _load(Path(path))

        assert path in str(raised.value)
        assert log == []

    def test_a_stored_copy_is_accepted(self, loading, models):
        log, _ = loading
        _built(models, "moss-sft")
        copy = models / "moss-sft@q4"
        copy.mkdir()
        (copy / "transformer").mkdir()
        (copy / "transformer" / "config.json").write_text(json.dumps({"num_layers": _LAYERS}), encoding="utf-8")

        _load(copy, 4, prequantized=True)

        assert "pipeline" in log

    def test_a_turbo_stored_copy_runs_with_the_lora_of_its_base_folder(self, models):
        from zvisiongenerator.backends.cuda.image.backend import _moss_turbo_lora
        from zvisiongenerator.utils.image_model_detect import ImageModelInfo

        base = _built(models, "moss")
        (base / "turbo_lora.safetensors").write_bytes(b"lora")
        copy = models / "moss@q4"
        copy.mkdir()
        turbo = ImageModelInfo(family="moss", is_distilled=True, size=None)

        assert _moss_turbo_lora(str(copy), turbo) == base / "turbo_lora.safetensors"
        assert _moss_turbo_lora(str(base), turbo) == base / "turbo_lora.safetensors"
        assert _moss_turbo_lora(str(copy), ImageModelInfo(family="moss", is_distilled=False, size=None)) is None
        (base / "turbo_lora.safetensors").unlink()
        with pytest.raises(FileNotFoundError, match="LoRA is missing"):
            _moss_turbo_lora(str(copy), turbo)

    def test_the_stored_copy_writer_refuses_a_folder_that_is_not_built(self, tmp_path, writers):
        source = tmp_path / "elsewhere" / "moss"
        source.mkdir(parents=True)

        with pytest.raises(RuntimeError, match="-m moss"):
            moss.write_stored_copy(source, tmp_path / "out", 4, torch.bfloat16, _FORMAT, None)

        assert writers == []


class TestLoadOrder:
    def test_q4_loads_the_nf4_transformer_with_the_keep_list_and_stays_resident(self, loading, models):
        log, loaded = loading
        folder = _built(models, "moss")

        pipeline = _load(folder, 4)

        assert log == ["tokenizer", "scheduler", "vae", "vae.to(cpu:0)", "encoder", "encoder.to(cpu)", "transformer-nf4", "transformer.to(cpu)", "pipeline"]
        kwargs = loaded["transformer-nf4"].kwargs
        assert kwargs["model_class"] is MOSSImageTransformer
        assert kwargs["keep_modules"] == moss.nf4_keep_modules(_LAYERS)
        assert kwargs["prequantized"] is False
        assert pipeline._ziv_resident is True

    def test_q8_loads_fp8_with_the_keep_patterns_and_streams_by_block(self, loading, models):
        log, loaded = loading
        folder = _built(models, "moss-sft")

        pipeline = _load(folder, 8)

        assert log[-3:] == ["transformer.to(cpu)", "pipeline", "offload"]
        assert "transformer-fp8" in log
        assert loaded["transformer-fp8"].kwargs == {"model_class": MOSSImageTransformer, "keep_patterns": moss.fp8_keep_patterns()}
        assert loaded["offload"].args[2] == "block_level" and loaded["offload"].args[0] is loaded["transformer-fp8"].result
        assert pipeline._ziv_resident is False

    def test_unquantized_loads_the_bf16_transformer_and_streams_by_block(self, loading, models):
        log, loaded = loading
        folder = _built(models, "moss-sft")

        pipeline = _load(folder, None)

        assert "transformer-bf16" in log and log[-1] == "offload"
        assert loaded["transformer-bf16"].kwargs["torch_dtype"] == torch.bfloat16
        assert pipeline._ziv_resident is False

    def test_the_text_encoder_is_nf4_at_every_level_and_prequantized_for_a_stored_copy(self, loading, models):
        _, loaded = loading
        folder = _built(models, "moss-sft")

        _load(folder, 8, prequantized=True)

        assert loaded["encoder"].kwargs == {"prequantized": True}

    def test_the_pipeline_starts_with_an_empty_memo(self, loading, models):
        folder = _built(models, "moss")

        pipeline = _load(folder, 8)

        assert pipeline._ziv_memo is None

    def test_adapters_are_installed_before_the_transformer_is_placed(self, loading, models):
        log, loaded = loading
        folder = _built(models, "moss")
        turbo = models / "moss" / "turbo_lora.safetensors"

        _load(folder, 8, turbo_lora=turbo, lora_paths=["/l/a.safetensors", "/l/b.safetensors"], lora_weights=[0.7, 1.2])

        assert log.index("loras") < log.index("pipeline") < log.index("offload")
        files = loaded["loras"].args[1]
        assert [(file.path, file.strength, file.dtype) for file in files] == [
            (str(turbo), 1.0, torch.float32),
            ("/l/a.safetensors", 0.7, torch.bfloat16),
            ("/l/b.safetensors", 1.2, torch.bfloat16),
        ]
        assert loaded["loras"].kwargs == {"device": torch.device(_GPU)}

    def test_user_loras_without_weights_run_at_full_strength_and_sft_has_no_turbo_adapter(self, loading, models):
        _, loaded = loading
        folder = _built(models, "moss-sft")

        _load(folder, 8, lora_paths=["/l/a.safetensors"])

        assert [(file.path, file.strength) for file in loaded["loras"].args[1]] == [("/l/a.safetensors", 1.0)]

    def test_no_adapters_means_no_install(self, loading, models):
        log, _ = loading
        folder = _built(models, "moss-sft")

        _load(folder, 8)

        assert "loras" not in log


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


class _Transformer(nn.Module):
    """Logs moves and forwards; predicts a constant velocity so a run changes its latents."""

    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self.log = log
        self.inputs: list[torch.Tensor] = []

    def to(self, device, *args, **kwargs):
        self.log.append(f"transformer.to({device})")
        return self

    def forward(self, latents, timestep, embeds, mask, skip_layers=None):
        self.log.append("forward(skip)" if skip_layers else "forward")
        self.inputs.append(latents.clone())
        return SimpleNamespace(sample=torch.full_like(latents, 0.1))


class _Encoder:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def to(self, device):
        self.log.append(f"encoder.to({device})")
        return self


class _Vae(nn.Module):
    """FLUX.2-VAE stand-in: unit batch-norm statistics, a fixed latent for encoding and a 1x1 conv plus 8x upsample for decoding."""

    def __init__(self, log: list[str], encoded: torch.Tensor | None = None) -> None:
        super().__init__()
        self.log = log
        self.bn = nn.BatchNorm2d(128)
        self.config = SimpleNamespace(batch_norm_eps=1e-4)
        self.conv = nn.Conv2d(32, 3, 1)
        self.encoded = encoded

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def decode(self, z, return_dict=False):
        self.log.append("decode")
        return (nn.functional.interpolate(self.conv(z), scale_factor=8, mode="nearest"),)

    def encode(self, pixels):
        self.log.append("encode_image")
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: self.encoded))


def _fake_pipeline(log: list[str], *, resident: bool, encoded: torch.Tensor | None = None) -> SimpleNamespace:
    from diffusers.image_processor import VaeImageProcessor

    return SimpleNamespace(
        transformer=_Transformer(log),
        scheduler=MOSSImageScheduler(),
        text_encoder=_Encoder(log),
        tokenizer=MagicMock(),
        vae=_Vae(log, encoded),
        image_processor=VaeImageProcessor(vae_scale_factor=16),
        _ziv_resident=resident,
        _ziv_memo=None,
        _interrupt=False,
    )


@pytest.fixture
def encoding(monkeypatch):
    """Replace the prompt encoder with one that logs each prompt it encodes."""
    from zvisiongenerator.backends.cuda.image import moss_sampler

    prompts: list[str] = []

    def encode(tokenizer, text_encoder, prompt, *, device, max_length):
        text_encoder.log.append(f"encode:{prompt}")
        prompts.append(prompt)
        return Conditioning(torch.zeros(1, 3, _TEXT_DIM), torch.ones(1, 3, dtype=torch.bool))

    monkeypatch.setattr(moss_sampler, "encode_prompt", encode)
    return prompts


def _turbo(steps: int = 4):
    return sampler_settings(turbo=True, steps=steps, guidance=1.0, negative_prompt=None)


def _sft(steps: int = 4, negative: str | None = None):
    return sampler_settings(turbo=False, steps=steps, guidance=3.0, negative_prompt=negative)


@torch.no_grad()
def _run(pipeline, prompt="a cat", sampler=None, make_callback=lambda n: None, **kwargs):
    # The backend calls generate under inference mode.
    return moss.generate(pipeline, prompt, width=64, height=64, seed=7, sampler=sampler or _turbo(), make_callback=make_callback, **kwargs)


class TestGenerate:
    def test_a_resident_transformer_steps_aside_for_the_encoder_and_the_decode(self, encoding):
        log: list[str] = []
        pipeline = _fake_pipeline(log, resident=True)

        _run(pipeline)

        steps = len(pipeline.scheduler.timesteps)
        assert log == ["transformer.to(cpu)", f"encoder.to({_GPU})", "encode:a cat", "encoder.to(cpu)", f"transformer.to({_GPU})", *["forward"] * steps, "transformer.to(cpu)", "decode"]

    def test_a_streamed_transformer_is_never_moved(self, encoding):
        log: list[str] = []
        pipeline = _fake_pipeline(log, resident=False)

        _run(pipeline)

        assert [entry for entry in log if entry.startswith("transformer.to")] == []
        assert log[:3] == [f"encoder.to({_GPU})", "encode:a cat", "encoder.to(cpu)"]

    def test_the_same_prompts_skip_the_encoder_and_its_moves(self, encoding):
        log: list[str] = []
        pipeline = _fake_pipeline(log, resident=True)
        _run(pipeline)
        log.clear()

        _run(pipeline)

        steps = len(pipeline.scheduler.timesteps)
        assert log == [f"transformer.to({_GPU})", *["forward"] * steps, "transformer.to(cpu)", "decode"]
        assert encoding == ["a cat"]

    @pytest.mark.parametrize(("second_prompt", "second_negative"), [("a dog", None), ("a cat", "blurry")])
    def test_a_new_prompt_or_negative_prompt_encodes_again(self, encoding, second_prompt, second_negative):
        pipeline = _fake_pipeline([], resident=False)
        _run(pipeline, sampler=_sft(negative=None))

        _run(pipeline, second_prompt, sampler=_sft(negative=second_negative))

        assert encoding == ["a cat", "", second_prompt, second_negative or ""]

    def test_turbo_takes_one_forward_per_step_and_sft_two(self, encoding):
        turbo_log: list[str] = []
        turbo = _fake_pipeline(turbo_log, resident=False)
        sft_log: list[str] = []
        sft = _fake_pipeline(sft_log, resident=False)

        _run(turbo, sampler=_turbo(4))
        _run(sft, sampler=_sft(4))

        steps = len(turbo.scheduler.timesteps)
        assert turbo_log.count("forward") == steps and turbo_log.count("forward(skip)") == 0
        assert (sft_log.count("forward"), sft_log.count("forward(skip)")) == (steps, steps)
        assert encoding == ["a cat", "a cat", ""]

    def test_the_noise_comes_from_the_seed_and_the_result_is_a_picture_of_the_asked_size(self, encoding):
        log: list[str] = []
        pipeline = _fake_pipeline(log, resident=False)

        image = _run(pipeline)

        expected = initial_noise((1, 128, 4, 4), torch.Generator("cpu").manual_seed(7), _GPU)
        torch.testing.assert_close(pipeline.transformer.inputs[0], expected)
        assert isinstance(image, Image.Image) and image.size == (64, 64)

    def test_the_callback_is_built_for_the_steps_that_run_and_called_each_step(self, encoding):
        pipeline = _fake_pipeline([], resident=False)
        callback = MagicMock()
        sizes: list[int] = []

        _run(pipeline, make_callback=lambda n: sizes.append(n) or callback)

        steps = len(pipeline.scheduler.timesteps)
        assert sizes == [steps]
        assert [call.args[1] for call in callback.call_args_list] == list(range(steps))
        assert all(call.args[0] is pipeline and set(call.args[3]) == {"latents"} for call in callback.call_args_list)

    def test_a_skip_stops_after_the_current_step_and_returns_nothing(self, encoding):
        log: list[str] = []
        pipeline = _fake_pipeline(log, resident=False)

        def stop(pipe, index, timestep, kwargs):
            pipe._interrupt = True

        image = _run(pipeline, make_callback=lambda n: stop)

        assert image is None
        assert log.count("forward") == 1 and "decode" not in log
        assert pipeline._interrupt is False

    def test_a_skipped_resident_run_steps_the_transformer_aside_and_the_next_run_brings_it_back(self, encoding):
        log: list[str] = []
        pipeline = _fake_pipeline(log, resident=True)

        def stop(pipe, index, timestep, kwargs):
            pipe._interrupt = True

        assert _run(pipeline, make_callback=lambda n: stop) is None
        assert log[-1] == "transformer.to(cpu)" and "decode" not in log

        log.clear()
        _run(pipeline)

        steps = len(pipeline.scheduler.timesteps)
        assert log == [f"transformer.to({_GPU})", *["forward"] * steps, "transformer.to(cpu)", "decode"]

    def test_a_failed_resident_run_steps_the_transformer_aside(self, encoding, monkeypatch):
        from zvisiongenerator.backends.cuda.image import moss_sampler

        log: list[str] = []
        pipeline = _fake_pipeline(log, resident=True)

        def fail(*args, **kwargs):
            raise RuntimeError("out of memory")

        monkeypatch.setattr(moss_sampler, "denoise", fail)

        with pytest.raises(RuntimeError):
            _run(pipeline)

        assert log[-1] == "transformer.to(cpu)"

    def test_an_earlier_interrupt_does_not_stop_the_next_run(self, encoding):
        pipeline = _fake_pipeline([], resident=False)
        pipeline._interrupt = True

        assert _run(pipeline) is not None


class TestImageToImage:
    @staticmethod
    def _encoded() -> torch.Tensor:
        """The VAE's unpatchified output for a 64x64 image (32 channels at 1/8 resolution)."""
        return torch.randn(1, 32, 8, 8, generator=torch.Generator().manual_seed(3))

    @staticmethod
    def _clean(encoded: torch.Tensor) -> torch.Tensor:
        """The reference's latents as the loop sees them: patchified and normalised with the (unit) batch-norm statistics."""
        from diffusers import Flux2Pipeline

        return Flux2Pipeline._patchify_latents(encoded) / (1 + 1e-4) ** 0.5

    def test_the_reference_is_noised_to_the_sigma_the_run_starts_at(self, encoding):
        log: list[str] = []
        encoded = self._encoded()
        pipeline = _fake_pipeline(log, resident=False, encoded=encoded)
        sizes: list[int] = []

        _run(pipeline, sampler=_turbo(8), make_callback=lambda n: sizes.append(n), image=Image.new("RGB", (64, 64)), strength=0.5)

        total = len(pipeline.scheduler.timesteps)
        start = total - img2img_run_steps(total, 0.5)
        noise = initial_noise((1, 128, 4, 4), torch.Generator("cpu").manual_seed(7), _GPU)
        expected = noised(self._clean(encoded), noise, float(pipeline.scheduler.sigmas[start]))
        torch.testing.assert_close(pipeline.transformer.inputs[0], expected)
        assert 0 < start < total
        assert sizes == [total - start] and log.count("forward") == total - start
        assert "encode_image" in log

    def test_a_resident_transformer_returns_to_the_gpu_after_the_reference_is_encoded(self, encoding):
        log: list[str] = []
        pipeline = _fake_pipeline(log, resident=True, encoded=self._encoded())

        _run(pipeline, sampler=_turbo(8), image=Image.new("RGB", (64, 64)), strength=0.5)

        assert log.index("encode_image") < log.index(f"transformer.to({_GPU})") < log.index("forward")

    def test_a_skipped_resident_run_then_a_same_prompt_reference_run_encodes_with_the_transformer_on_the_cpu(self, encoding):
        log: list[str] = []
        pipeline = _fake_pipeline(log, resident=True, encoded=self._encoded())

        def stop(pipe, index, timestep, kwargs):
            pipe._interrupt = True

        assert _run(pipeline, make_callback=lambda n: stop) is None
        _run(pipeline, sampler=_turbo(), image=Image.new("RGB", (64, 64)), strength=0.5)

        assert encoding == ["a cat"]  # the second run reused the memoised conditioning
        before_encode = [entry for entry in log[: log.index("encode_image")] if entry.startswith("transformer.to(")]
        assert before_encode[-1] == "transformer.to(cpu)"

    def test_full_strength_with_a_reference_starts_from_pure_noise(self, encoding):
        pipeline = _fake_pipeline([], resident=False, encoded=self._encoded())

        _run(pipeline, image=Image.new("RGB", (64, 64)), strength=1.0)

        noise = initial_noise((1, 128, 4, 4), torch.Generator("cpu").manual_seed(7), _GPU)
        torch.testing.assert_close(pipeline.transformer.inputs[0], noise)


# ---------------------------------------------------------------------------
# write_stored_copy
# ---------------------------------------------------------------------------


class _Backend:
    """Writes stored copies with the real MOSS writer; records what it loads."""

    name = "fake"

    def __init__(self) -> None:
        self.loads: list[str] = []

    def stored_quant_format(self, bits):
        return _FORMAT

    def quantizes_from_files(self, bits, source=None):
        return source is not None and is_moss_model(Path(source))

    def write_quantized_files(self, source, path, bits, cancelled=None):
        moss.write_stored_copy(Path(source), Path(path), bits, torch.bfloat16, _FORMAT, cancelled)

    def load_model(self, model_path, quantize=None, precision="bfloat16", lora_paths=None, lora_weights=None):
        self.loads.append(model_path)
        return MagicMock(), "info"


def _store(backend: _Backend, models: Path, name: str, bits: int) -> Path:
    load_image_model(backend, str(models / name), quantize=bits, models_dir=models)
    return models / f"{name}@q{bits}"


def _inode(copy: Path, component: str) -> int:
    return os.stat(copy / component / "model.safetensors").st_ino


_WRITERS = {4: ("nf4-transformer", "nf4-encoder"), 8: ("fp8-transformer", "nf4-encoder")}


class TestWriteStoredCopy:
    @pytest.mark.parametrize("bits", [4, 8])
    def test_the_transformer_and_then_the_nf4_encoder_are_quantized_and_the_rest_is_linked(self, models, writers, bits):
        source = _built(models, "moss-sft")
        target = models / "out"

        moss.write_stored_copy(source, target, bits, torch.bfloat16, _FORMAT, None)

        assert writers == list(_WRITERS[bits])
        assert (target / "transformer" / "model.safetensors").is_file() and (target / "text_encoder" / "model.safetensors").is_file()
        for component in ("vae", "tokenizer", "scheduler"):
            assert (target / component / "config.json").is_file()
        assert (target / "vae" / "model.safetensors").stat().st_ino == (source / "vae" / "model.safetensors").stat().st_ino
        assert (target / "LICENSE").is_file()

    @pytest.mark.parametrize("bits", [4, 8])
    @pytest.mark.parametrize("order", [("moss-sft", "moss"), ("moss", "moss-sft")])
    def test_the_second_alias_links_the_first_ones_weights_and_quantizes_nothing(self, models, writers, bits, order):
        for name in ("moss", "moss-sft"):
            _built(models, name)
        backend = _Backend()
        first = _store(backend, models, order[0], bits)
        written = list(writers)

        second = _store(backend, models, order[1], bits)

        assert writers == written == list(_WRITERS[bits])
        for component in ("transformer", "text_encoder"):
            assert _inode(first, component) == _inode(second, component)
        for name, copy in zip(order, (first, second), strict=True):
            manifest = json.loads((copy / "ziv-quant.json").read_text(encoding="utf-8"))
            assert manifest["source"] == str(models / name)
            assert is_current(copy, models / name, bits, _FORMAT)
        assert json.loads((models / "moss@q4" if bits == 4 else models / "moss@q8").joinpath("model_index.json").read_text(encoding="utf-8"))["is_distilled"] is True
        assert json.loads((models / f"moss-sft@q{bits}" / "model_index.json").read_text(encoding="utf-8"))["is_distilled"] is False
        assert backend.loads == [str(first), str(second)]

    def test_a_stale_sibling_copy_is_not_shared(self, models, writers):
        for name in ("moss", "moss-sft"):
            _built(models, name)
        backend = _Backend()
        sibling = _store(backend, models, "moss-sft", 4)
        (models / "moss-sft" / "transformer" / "model.safetensors").write_bytes(b"changed weights, other size")
        assert not is_current(sibling, models / "moss-sft", 4, _FORMAT)

        copy = _store(backend, models, "moss", 4)

        assert writers == ["nf4-transformer", "nf4-encoder"] * 2
        assert _inode(copy, "transformer") != _inode(sibling, "transformer")

    def test_a_sibling_built_from_another_sft_pin_is_not_shared(self, models, writers):
        _built(models, "moss")
        _built(models, "moss-sft", sft="owner/sft@" + "c" * 40)
        backend = _Backend()
        _store(backend, models, "moss-sft", 4)

        _store(backend, models, "moss", 4)

        assert writers == ["nf4-transformer", "nf4-encoder"] * 2

    def test_a_stop_after_the_transformer_leaves_the_encoder_and_the_links_unwritten(self, models, writers):
        source = _built(models, "moss-sft")
        target = models / "out"
        answers = iter([False, True])

        moss.write_stored_copy(source, target, 4, torch.bfloat16, _FORMAT, lambda: next(answers))

        assert writers == ["nf4-transformer"]
        assert (target / "transformer").is_dir()
        assert not (target / "text_encoder").exists() and not (target / "vae").exists()

    def test_a_stop_before_anything_writes_nothing(self, models, writers):
        source = _built(models, "moss-sft")

        moss.write_stored_copy(source, models / "out", 8, torch.bfloat16, _FORMAT, lambda: True)

        assert writers == []


# ---------------------------------------------------------------------------
# Keep lists against the real model
# ---------------------------------------------------------------------------


def _tiny_bf16() -> MOSSImageTransformer:
    return _tiny_transformer().to(torch.bfloat16)


def _forward(model: MOSSImageTransformer) -> torch.Tensor:
    latents = torch.randn(1, 128, 4, 4, dtype=torch.bfloat16)
    embeds = torch.randn(1, 5, _TEXT_DIM, dtype=torch.bfloat16)
    with torch.no_grad():
        return model(latents, torch.tensor(500.0), embeds, torch.ones(1, 5, dtype=torch.bool)).sample


class TestKeepLists:
    def test_every_nf4_keep_name_is_a_module_of_the_model(self):
        names = {name for name, _ in _tiny_transformer().named_modules()}

        assert set(moss.nf4_keep_modules(_LAYERS)) <= names

    def test_every_fp8_keep_pattern_matches_a_module_of_the_model(self):
        names = [name for name, _ in _tiny_transformer().named_modules()]

        for pattern in moss.fp8_keep_patterns():
            assert any(re.search(pattern, name) for name in names), pattern

    def test_nf4_keep_modules_list_every_layer_then_the_fixed_modules(self):
        fixed = tuple(name for name in moss.KEEP_IN_COMPUTE_DTYPE if "*" not in name)

        assert moss.nf4_keep_modules(3) == ("layers.0.proj.q", "layers.1.proj.q", "layers.2.proj.q", *fixed)
        assert moss.nf4_keep_modules(0) == fixed
        assert len(moss.nf4_keep_modules(42)) == 42 + len(fixed)

    def test_fp8_keep_patterns_match_only_the_kept_linears(self):
        patterns = [re.compile(p) for p in moss.fp8_keep_patterns()]

        def kept(name: str) -> bool:
            return any(p.search(name) for p in patterns)

        assert kept("layers.7.proj.q") and kept("text_embedder.proj")
        assert not kept("text_embedder.blocks.0.proj")
        assert not kept("layers.7.proj.qkv") and not kept("layers.7.proj.k")

    def test_every_fp8_pattern_is_anchored_and_matches_only_its_own_module(self):
        patterns = [re.compile(p) for p in moss.fp8_keep_patterns()]

        assert len(patterns) == len(moss.KEEP_IN_COMPUTE_DTYPE)
        for name in (n for n in moss.KEEP_IN_COMPUTE_DTYPE if "*" not in n):
            assert sum(bool(p.search(name)) for p in patterns) == 1, name
            assert not any(p.search(f"blocks.{name}") or p.search(f"{name}_extra") for p in patterns), name
        assert all(any(p.search(f"layers.{i}.proj.q") for p in patterns) for i in (0, 9, 41))

    def test_fp8_storage_with_the_keep_patterns_runs_a_forward_and_keeps_the_dtype_readers_in_bf16(self):
        model = cast_to_fp8(_tiny_bf16(), torch.bfloat16, moss.fp8_keep_patterns())

        output = _forward(model)

        assert torch.isfinite(output.float()).all()
        assert model.layers[0].proj["q"].weight.dtype == torch.bfloat16
        assert model.layers[0].proj["k"].weight.dtype == torch.float8_e4m3fn

    def test_fp8_storage_without_the_patterns_breaks_the_forward(self):
        model = cast_to_fp8(_tiny_bf16(), torch.bfloat16)

        with pytest.raises(RuntimeError):
            _forward(model)


_VENDORED = Path(__file__).resolve().parent.parent / "packages" / "moss_image" / "pipeline_moss_image.py"
# The module path each vendored class has inside the transformer ("*" stands for a layer index).
_CLASS_PREFIX = {"MOSSImageTransformer": "", "_Block": "layers.*.", "_TextProjection": "text_embedder."}


def _owner_path(owner: ast.expr) -> str | None:
    """Return the dotted module path of an owner expression such as ``self.proj['q']`` or ``self.time_embedding[0]``."""
    if isinstance(owner, ast.Name) and owner.id == "self":
        return ""
    if isinstance(owner, ast.Attribute):
        parent = _owner_path(owner.value)
        return None if parent is None else f"{parent}.{owner.attr}".lstrip(".")
    if isinstance(owner, ast.Subscript) and isinstance(owner.slice, ast.Constant):
        parent = _owner_path(owner.value)
        return None if parent is None else f"{parent}.{owner.slice.value}".lstrip(".")
    return None


def _rmsnorm_attributes(cls: ast.ClassDef) -> set[str]:
    """Return the names of the attributes a class's ``__init__`` sets to an ``_RMSNorm``."""
    names: set[str] = set()
    for node in ast.walk(cls):
        if not isinstance(node, ast.Assign):
            continue
        targets = node.targets[0].elts if isinstance(node.targets[0], ast.Tuple) else [node.targets[0]]
        values = node.value.elts if isinstance(node.value, ast.Tuple) else [node.value]
        for target, value in zip(targets, values, strict=False):
            if isinstance(value, ast.Call) and getattr(value.func, "id", None) == "_RMSNorm" and isinstance(target, ast.Attribute):
                names.add(target.attr)
    return names


def test_the_layers_the_vendored_model_reads_a_dtype_from_are_exactly_the_keep_list():
    tree = ast.parse(_VENDORED.read_text(encoding="utf-8"))
    readers: set[str] = set()
    for cls in (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in _CLASS_PREFIX):
        norms = _rmsnorm_attributes(cls)
        for node in ast.walk(cls):
            if isinstance(node, ast.Attribute) and node.attr == "dtype" and isinstance(node.value, ast.Attribute) and node.value.attr == "weight":
                path = _owner_path(node.value.value)
                if path is not None and path.split(".")[0] not in norms:
                    readers.add(_CLASS_PREFIX[cls.name] + path)

    assert readers == set(moss.KEEP_IN_COMPUTE_DTYPE)


# ---------------------------------------------------------------------------
# Live preview through the backend's real step callback
# ---------------------------------------------------------------------------


class TestLivePreviewThroughTheBackend:
    """The latents ``generate`` hands to the backend's step callback are the layout the MOSS preview projection reads."""

    @staticmethod
    def _run_through_backend(pipeline, *, steps: int, **kwargs):
        from zvisiongenerator.backends.cuda.image.backend import DiffusersBackend
        from zvisiongenerator.utils.image_model_detect import ImageModelInfo

        payloads: list[dict] = []
        backend = DiffusersBackend()
        backend._model_info = ImageModelInfo(family="moss", is_distilled=True, size=None)
        if "image" in kwargs:
            image = backend.image_to_image(pipeline, kwargs["image"], "a cat", kwargs["strength"], steps, 7, 1.0, step_callback=payloads.append)
        else:
            image = backend.text_to_image(pipeline, "a cat", 128, 128, 7, steps, 1.0, step_callback=payloads.append)
        return image, payloads

    @staticmethod
    def _spy_on_projection(monkeypatch) -> list[tuple[int, ...]]:
        from zvisiongenerator.backends.cuda.image.backend import LivePreview

        # The globals of the module the backend's LivePreview actually lives in (other tests re-import these modules).
        namespace = LivePreview._render.__globals__
        shapes: list[tuple[int, ...]] = []
        real = namespace["render_latent_preview"]

        def spy(pipe, family, latents, height, width):
            shapes.append(tuple(latents.shape))
            return real(pipe, family, latents, height, width)

        monkeypatch.setitem(namespace, "render_latent_preview", spy)
        return shapes

    def test_text_to_image_emits_a_preview_on_every_step_but_the_last_from_128_channel_latents(self, encoding, monkeypatch):
        from zvisiongenerator.core.progress_events import preview_steps

        shapes = self._spy_on_projection(monkeypatch)
        pipeline = _fake_pipeline([], resident=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            image, payloads = self._run_through_backend(pipeline, steps=8)
        assert not [w for w in caught if "Live preview failed" in str(w.message)]

        run = len(pipeline.scheduler.timesteps)
        preview_at = preview_steps(run)
        assert isinstance(image, Image.Image)
        assert [p["current_step"] for p in payloads] == list(range(1, run + 1))
        assert {p["current_step"] for p in payloads if "preview" in p} == set(preview_at) and preview_at
        assert all(p["total_steps"] == run for p in payloads)
        assert all(p["preview"].size == (16, 16) for p in payloads if "preview" in p)
        assert shapes and set(shapes) == {(1, 128, 8, 8)}

    def test_image_to_image_previews_follow_the_steps_that_run_not_the_whole_schedule(self, encoding, monkeypatch):
        from zvisiongenerator.core.progress_events import preview_steps

        self._spy_on_projection(monkeypatch)
        encoded = torch.randn(1, 32, 16, 16, generator=torch.Generator().manual_seed(3))
        pipeline = _fake_pipeline([], resident=False, encoded=encoded)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            image, payloads = self._run_through_backend(pipeline, steps=8, image=Image.new("RGB", (128, 128)), strength=0.5)
        assert not [w for w in caught if "Live preview failed" in str(w.message)]

        total = len(pipeline.scheduler.timesteps)
        run = img2img_run_steps(total, 0.5)
        assert isinstance(image, Image.Image) and run < total
        assert [p["current_step"] for p in payloads] == list(range(1, run + 1))
        assert all(p["total_steps"] == run for p in payloads)
        assert {p["current_step"] for p in payloads if "preview" in p} == set(preview_steps(run))
