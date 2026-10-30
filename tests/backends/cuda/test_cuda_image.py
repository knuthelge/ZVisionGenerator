"""Mock-patched tests for zvisiongenerator.backends.cuda.image.backend on macOS CI.

Exercises DiffusersBackend code paths without real torch/diffusers/CUDA by
mocking all heavy dependencies at the module level before import.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

import zvisiongenerator.schedulers.beta_sigmas  # noqa: F401  (import scipy for real here; its first import under the fake ``torch`` fails and corrupts numpy)
from zvisiongenerator.core.image_types import JobPrompt
from zvisiongenerator.utils.image_model_detect import ImageModelInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_MODULES_KEYS = (
    "torch",
    "torch.cuda",
    "torch.backends",
    "torch.backends.cudnn",
    "diffusers",
    "diffusers.hooks",
    "transformers",
)


def _stub_torch(*, cuda_available=True):
    """Build a fake torch module with enough surface to satisfy image_win."""
    torch_mod = MagicMock()
    torch_mod.bfloat16 = "bf16-sentinel"
    torch_mod.float16 = "fp16-sentinel"
    torch_mod.float32 = "fp32-sentinel"
    torch_mod.device.return_value = MagicMock()
    torch_mod.cuda.is_available.return_value = cuda_available
    torch_mod.cuda.empty_cache = MagicMock()
    torch_mod.backends.cudnn.benchmark = False
    torch_mod.Generator.return_value.manual_seed.return_value = MagicMock()
    torch_mod.inference_mode = lambda: lambda fn: fn  # no-op decorator
    return torch_mod


def _build_fake_modules(torch_mod=None, diffusers_mod=None):
    """Return dict of fake sys.modules entries for torch/diffusers/transformers."""
    t = torch_mod or _stub_torch()
    d = diffusers_mod or MagicMock()
    return {
        "torch": t,
        "torch.cuda": t.cuda,
        "torch.backends": t.backends,
        "torch.backends.cudnn": t.backends.cudnn,
        "diffusers": d,
        "diffusers.hooks": d.hooks,
        "transformers": MagicMock(),
    }


def _fresh_import(fake_modules):
    """Force a fresh import of image_win with *fake_modules* in sys.modules.

    The caller MUST be inside a ``patch.dict(sys.modules, fake_modules)`` context
    so that lazy imports inside methods also resolve to fakes.
    """
    sys.modules.pop("zvisiongenerator.backends.cuda.image.backend", None)
    import zvisiongenerator.backends.cuda.image.backend as mod  # noqa: E402

    return mod


def _make_model_info(family="zimage", is_distilled=False, size=None):
    return ImageModelInfo(family=family, is_distilled=is_distilled, size=size)


# ---------------------------------------------------------------------------
# Fixture: import image_win with all heavy deps mocked (stays active per-test)
# ---------------------------------------------------------------------------


def _t2i(backend, model, **overrides):
    """Call ``text_to_image`` with the usual small arguments, each replaceable by *overrides*."""
    arguments = {"prompt": "a cat", "width": 64, "height": 64, "seed": 42, "steps": 4, "guidance": 0.5, **overrides}
    return backend.text_to_image(model=model, **arguments)


@pytest.fixture()
def win_backend():
    """Yield (module, torch_stub, diffusers_stub) with fakes active for the whole test."""
    torch_mod = _stub_torch()
    diffusers_mod = MagicMock()
    fakes = _build_fake_modules(torch_mod, diffusers_mod)

    with patch.dict(sys.modules, fakes):
        mod = _fresh_import(fakes)
        yield mod, torch_mod, diffusers_mod

    # Clean up cached module so other test files aren't affected
    sys.modules.pop("zvisiongenerator.backends.cuda.image.backend", None)


@pytest.fixture()
def win_backend_no_cuda():
    """Same as win_backend but CUDA reports unavailable."""
    torch_mod = _stub_torch(cuda_available=False)
    diffusers_mod = MagicMock()
    fakes = _build_fake_modules(torch_mod, diffusers_mod)

    with patch.dict(sys.modules, fakes):
        mod = _fresh_import(fakes)
        yield mod, torch_mod, diffusers_mod

    sys.modules.pop("zvisiongenerator.backends.cuda.image.backend", None)


# ---------------------------------------------------------------------------
# load_model guard: generation without load_model raises RuntimeError
# ---------------------------------------------------------------------------


class TestLoadModelGuardMocked:
    """Generation methods must raise RuntimeError when load_model() hasn't been called."""

    def test_text_to_image_without_load_model(self, win_backend):
        mod, _, _ = win_backend
        backend = mod.DiffusersBackend()
        with pytest.raises(RuntimeError, match="load_model"):
            backend.text_to_image(
                model=MagicMock(),
                prompt="test",
                width=64,
                height=64,
                seed=42,
                steps=1,
                guidance=0.5,
            )

    def test_image_to_image_without_load_model(self, win_backend):
        mod, _, _ = win_backend
        backend = mod.DiffusersBackend()
        with pytest.raises(RuntimeError, match="load_model"):
            backend.image_to_image(
                model=MagicMock(),
                image=Image.new("RGB", (64, 64)),
                prompt="test",
                strength=0.5,
                steps=1,
                seed=42,
                guidance=0.5,
            )

    def test_text_to_image_accepts_first_sigma_kwarg_for_signature_parity(self, win_backend):
        mod, _, _ = win_backend
        backend = mod.DiffusersBackend()
        backend._model_info = _make_model_info()
        model = MagicMock()
        model.scheduler = MagicMock(config={})
        result = MagicMock()
        result.images = [Image.new("RGB", (64, 64))]
        model.return_value = result

        image = backend.text_to_image(
            model=model,
            prompt="test",
            width=64,
            height=64,
            seed=42,
            steps=4,
            guidance=0.5,
            first_sigma=1.005,
        )

        assert image == result.images[0]
        assert "first_sigma" not in model.call_args.kwargs


# ---------------------------------------------------------------------------
# load_model: CUDA unavailable → RuntimeError
# ---------------------------------------------------------------------------


class TestLoadModelCudaCheck:
    """load_model() refuses without CUDA before it loads anything (the message wording is test_cuda_driver's job)."""

    def test_raises_when_cuda_unavailable(self, win_backend_no_cuda):
        mod, _, _ = win_backend_no_cuda

        with patch.object(mod, "detect_image_model") as mock_detect, pytest.raises(RuntimeError, match="CUDA"):
            mod.DiffusersBackend().load_model("fake-model-path")

        mock_detect.assert_not_called()


class TestPreparePrompts:
    """The CUDA backend has no text encoder to free: preparing prompts touches nothing."""

    def test_prepare_prompts_does_nothing(self, win_backend):
        mod, _, _ = win_backend
        model, cancelled = MagicMock(), MagicMock()

        result = mod.DiffusersBackend().prepare_prompts(model, (JobPrompt("p", None),), width=64, height=64, cancelled=cancelled)

        assert result is None
        assert model.mock_calls == []
        assert cancelled.mock_calls == []


class TestIdeogram4:
    """Ideogram 4 Instant loads through its own loader at the selected level and runs with its built-in guidance."""

    def test_load_model_uses_the_ideogram_loader_and_tiles_the_vae(self, win_backend):
        mod, _, _ = win_backend
        pipeline = MagicMock()
        info = _make_model_info(family="ideogram4")

        with patch.object(mod, "detect_image_model", return_value=info), patch.object(mod, "load_ideogram4_pipeline", return_value=pipeline) as mock_load:
            result, model_info = mod.DiffusersBackend().load_model("/models/ideo", quantize=4)

        assert result is pipeline and model_info is info
        assert mock_load.call_args.args[0] == "/models/ideo"
        assert mock_load.call_args.args[2] == 4 and mock_load.call_args.kwargs == {"prequantized": False}
        pipeline.vae.enable_tiling.assert_called_once()
        mod.AutoPipelineForText2Image.from_pretrained.assert_not_called()

    def test_a_stored_copy_loads_at_its_own_level_as_prequantized(self, win_backend, tmp_path):
        mod, _, _ = win_backend
        stored = tmp_path / "ideo@q8"
        stored.mkdir()
        (stored / "ziv-quant.json").write_text('{"bits": 8}', encoding="utf-8")

        with patch.object(mod, "detect_image_model", return_value=_make_model_info(family="ideogram4")), patch.object(mod, "load_ideogram4_pipeline", return_value=MagicMock()) as mock_load:
            mod.DiffusersBackend().load_model(str(stored))

        assert mock_load.call_args.args[2] == 8 and mock_load.call_args.kwargs == {"prequantized": True}

    def test_loras_load_onto_the_pipeline_after_it_is_placed(self, win_backend):
        mod, _, _ = win_backend
        pipeline = MagicMock()
        order: list[str] = []

        with (
            patch.object(mod, "detect_image_model", return_value=_make_model_info(family="ideogram4")),
            patch.object(mod, "load_ideogram4_pipeline", side_effect=lambda *args, **kwargs: order.append("load") or pipeline),
            patch.object(mod, "_load_loras", side_effect=lambda *args, **kwargs: order.append("lora")) as mock_loras,
        ):
            mod.DiffusersBackend().load_model("/models/ideo", lora_paths=["/l/a.safetensors"], lora_weights=[0.7])

        assert order == ["load", "lora"]
        assert mock_loras.call_args.args == (pipeline, ["/l/a.safetensors"], [0.7])
        assert mock_loras.call_args.kwargs == {"family": "ideogram4"}

    def _generate(self, mod, **overrides):
        backend = mod.DiffusersBackend()
        backend._model_info = _make_model_info(family="ideogram4")
        pipe = MagicMock(_interrupt=False)
        arguments = {"model": pipe, "prompt": "a car", "width": 1024, "height": 768, "seed": 5, "steps": 8, "guidance": 1.0, **overrides}
        with patch.object(mod, "generate_ideogram4", return_value=MagicMock(images=["image"])) as mock_generate:
            result = backend.text_to_image(**arguments)
        return result, mock_generate, pipe

    def test_text_to_image_runs_the_given_steps(self, win_backend):
        mod, _, _ = win_backend

        result, mock_generate, pipe = self._generate(mod, steps=12)

        assert result == "image"
        assert mock_generate.call_args.kwargs["sampler"].steps == 12
        assert mock_generate.call_args.args[:2] == (pipe, "a car")
        assert (mock_generate.call_args.kwargs["width"], mock_generate.call_args.kwargs["height"]) == (1024, 768)

    def test_another_guidance_is_ignored_with_a_warning(self, win_backend):
        mod, _, _ = win_backend

        with pytest.warns(UserWarning, match="guidance"):
            result, _, _ = self._generate(mod, guidance=5.0, guidance_explicit=True)

        assert result == "image"

    @pytest.mark.parametrize("first_sigma", [None, 1.006])
    def test_the_per_run_first_sigma_is_passed_on(self, win_backend, first_sigma):
        mod, _, _ = win_backend

        _, mock_generate, _ = self._generate(mod, first_sigma=first_sigma)

        assert mock_generate.call_args.kwargs["first_sigma"] == first_sigma

    def test_progress_callbacks_are_built_for_the_sampler_steps(self, win_backend):
        mod, _, _ = win_backend

        _, mock_generate, _ = self._generate(mod, step_callback=MagicMock())

        assert "callback_on_step_end" in mock_generate.call_args.kwargs["callback_kwargs"]

    def test_a_skipped_run_returns_none_and_clears_the_interrupt(self, win_backend):
        mod, _, _ = win_backend
        backend = mod.DiffusersBackend()
        backend._model_info = _make_model_info(family="ideogram4")
        pipe = MagicMock(_interrupt=True)

        with patch.object(mod, "generate_ideogram4", return_value=MagicMock(images=["image"])):
            result = backend.text_to_image(model=pipe, prompt="a car", width=512, height=512, seed=1, steps=8, guidance=1.0, skip_signal=MagicMock())

        assert result is None
        assert pipe._interrupt is False


class TestMoss:
    """MOSS-Image loads through its family module, with Turbo's LoRA, and runs through the family's generate."""

    _BASE = Path("/models/moss")

    @staticmethod
    def _info(*, turbo: bool):
        return _make_model_info(family="moss", is_distilled=turbo)

    def _load(self, mod, *, turbo=True, path="/models/moss", lora=Path("/models/moss/turbo_lora.safetensors"), base=_BASE, **kwargs):
        pipeline = MagicMock()
        with (
            patch.object(mod, "detect_image_model", return_value=self._info(turbo=turbo)),
            patch.object(mod, "built_moss_base", return_value=base),
            patch.object(mod, "turbo_lora_file", return_value=lora),
            patch.object(mod, "load_moss_pipeline", return_value=pipeline) as mock_load,
            patch.object(mod, "_load_loras") as mock_loras,
        ):
            result = mod.DiffusersBackend().load_model(path, **kwargs)
        return result, pipeline, mock_load, mock_loras

    def test_turbo_loads_with_its_lora_and_the_user_loras_go_to_the_family_loader(self, win_backend):
        mod, _, _ = win_backend

        (result, info), pipeline, mock_load, mock_loras = self._load(mod, quantize=8, lora_paths=["/l/a.safetensors"], lora_weights=[0.5])

        assert result is pipeline and info.family == "moss"
        assert mock_load.call_args.args[0] == "/models/moss" and mock_load.call_args.args[2] == 8
        assert mock_load.call_args.kwargs == {"prequantized": False, "turbo_lora": Path("/models/moss/turbo_lora.safetensors"), "lora_paths": ["/l/a.safetensors"], "lora_weights": [0.5]}
        mock_loras.assert_not_called()
        pipeline.vae.enable_tiling.assert_called_once()
        mod.AutoPipelineForText2Image.from_pretrained.assert_not_called()

    def test_sft_has_no_turbo_lora(self, win_backend):
        mod, _, _ = win_backend

        _, _, mock_load, _ = self._load(mod, turbo=False, path="/models/moss-sft", base=Path("/models/moss-sft"), lora=None)

        assert mock_load.call_args.kwargs["turbo_lora"] is None

    def test_a_stored_copy_loads_at_its_own_level_as_prequantized(self, win_backend, tmp_path):
        mod, _, _ = win_backend
        stored = tmp_path / "moss@q4"
        stored.mkdir()
        (stored / "ziv-quant.json").write_text('{"bits": 4}', encoding="utf-8")

        _, _, mock_load, _ = self._load(mod, path=str(stored))

        assert mock_load.call_args.args[2] == 4 and mock_load.call_args.kwargs["prequantized"] is True

    def test_a_turbo_folder_without_its_lora_fails_naming_the_remedy(self, win_backend):
        mod, _, _ = win_backend

        with pytest.raises(FileNotFoundError, match="-m moss"):
            self._load(mod, lora=None)

    def test_a_folder_that_is_not_built_is_left_to_the_family_loader_to_refuse(self, win_backend):
        mod, _, _ = win_backend

        _, _, mock_load, _ = self._load(mod, path="org/MOSS-Image-SFT", base=None, lora=None)

        mock_load.assert_called_once()
        assert mock_load.call_args.kwargs["turbo_lora"] is None

    @pytest.mark.parametrize(("bits", "expected"), [(8, True), (4, True), (6, False)])
    def test_moss_is_stored_from_files_at_q4_and_q8(self, win_backend, tmp_path, bits, expected):
        mod, _, _ = win_backend
        (tmp_path / "model_index.json").write_text('{"_class_name": "MOSSImagePipeline"}', encoding="utf-8")

        assert mod.DiffusersBackend().quantizes_from_files(bits, source=str(tmp_path)) is expected

    def test_other_families_are_stored_from_files_only_at_q8(self, win_backend, tmp_path):
        mod, _, _ = win_backend
        (tmp_path / "model_index.json").write_text('{"_class_name": "ZImagePipeline"}', encoding="utf-8")
        backend = mod.DiffusersBackend()

        assert backend.quantizes_from_files(8, source=str(tmp_path)) is True
        assert backend.quantizes_from_files(4, source=str(tmp_path)) is False
        assert backend.quantizes_from_files(4, source=None) is False

    @pytest.mark.parametrize("bits", [4, 8])
    def test_write_quantized_files_uses_the_family_writer(self, win_backend, tmp_path, bits):
        mod, _, _ = win_backend
        (tmp_path / "model_index.json").write_text('{"_class_name": "MOSSImagePipeline"}', encoding="utf-8")
        cancelled = MagicMock(return_value=False)

        with (
            patch.object(mod.importlib.metadata, "version", side_effect=lambda package: "1.0"),
            patch.object(mod, "write_moss_stored_copy") as mock_moss,
            patch.object(mod, "write_fp8_copy") as mock_fp8,
        ):
            mod.DiffusersBackend().write_quantized_files(str(tmp_path), "/models/.moss@q.partial", bits, cancelled)

        source, target, passed_bits, _dtype, backend_format, passed_cancelled = mock_moss.call_args.args
        assert (source, target, passed_bits, passed_cancelled) == (tmp_path, Path("/models/.moss@q.partial"), bits, cancelled)
        assert backend_format == "diffusers-1.0+transformers-1.0" + ("+bitsandbytes-1.0" if bits == 4 else "")
        mock_fp8.assert_not_called()

    def _generate(self, mod, *, turbo=True, result="image", callback_steps=4, **overrides):
        """Run text_to_image with the family's generate faked; also build its callback for *callback_steps* steps.

        Returns ``(image, generate mock, callback)``.
        """
        backend = mod.DiffusersBackend()
        backend._model_info = self._info(turbo=turbo)
        arguments = {"model": MagicMock(), "prompt": "a car", "width": 1024, "height": 768, "seed": 5, "steps": 4, "guidance": 1.0, **overrides}
        built: list[Any] = []

        def fake_generate(*args, make_callback, **kwargs):
            built.append(make_callback(callback_steps))
            return result

        with patch.object(mod, "generate_moss", side_effect=fake_generate) as mock_generate, patch.object(mod, "create_live_preview", return_value=None):
            image = backend.text_to_image(**arguments)
        return image, mock_generate, built[0]

    def test_turbo_runs_the_given_steps_without_guidance_or_a_negative_prompt(self, win_backend):
        mod, _, _ = win_backend

        image, mock_generate, _ = self._generate(mod, steps=6, negative_prompt="blurry")

        sampler = mock_generate.call_args.kwargs["sampler"]
        assert image == "image"
        assert (sampler.steps, sampler.guidance, sampler.negative_prompt) == (6, 1.0, None)
        assert mock_generate.call_args.args[1] == "a car"
        assert {key: mock_generate.call_args.kwargs[key] for key in ("width", "height", "seed", "strength")} == {"width": 1024, "height": 768, "seed": 5, "strength": 1.0}
        assert mock_generate.call_args.kwargs["image"] is None

    def test_turbo_ignores_another_guidance_with_a_warning(self, win_backend):
        mod, _, _ = win_backend

        with pytest.warns(UserWarning, match="guidance"):
            _, mock_generate, _ = self._generate(mod, guidance=4.0)

        assert mock_generate.call_args.kwargs["sampler"].guidance == 1.0

    def test_sft_guides_with_the_negative_prompt_and_skips_a_layer_in_that_branch(self, win_backend):
        mod, _, _ = win_backend

        _, mock_generate, _ = self._generate(mod, turbo=False, steps=50, guidance=3.0, negative_prompt=None)

        sampler = mock_generate.call_args.kwargs["sampler"]
        assert (sampler.steps, sampler.guidance, sampler.negative_prompt, sampler.skip_layers) == (50, 3.0, "", (22,))

    def test_without_progress_or_skip_no_callback_is_built(self, win_backend):
        mod, _, _ = win_backend

        _, _, callback = self._generate(mod)

        assert callback is None

    def test_the_progress_callback_counts_the_steps_the_run_takes(self, win_backend):
        mod, _, _ = win_backend
        progress = MagicMock()

        _, _, callback = self._generate(mod, step_callback=progress, callback_steps=3)
        callback(MagicMock(num_timesteps=None), 1, 0.5, {"latents": MagicMock()})

        progress.assert_called_once_with({"current_step": 2, "total_steps": 3})

    def test_the_skip_signal_interrupts_the_pipeline(self, win_backend):
        mod, _, _ = win_backend
        skip = MagicMock()
        skip.check.return_value = True
        pipe = MagicMock(num_timesteps=None, _interrupt=False)

        _, _, callback = self._generate(mod, skip_signal=skip)
        callback(pipe, 0, 0.5, {"latents": MagicMock()})

        assert pipe._interrupt is True

    def test_a_skipped_run_returns_none(self, win_backend):
        mod, _, _ = win_backend

        image, _, _ = self._generate(mod, result=None, skip_signal=MagicMock())

        assert image is None

    def test_image_to_image_rounds_the_reference_size_down_to_16_and_passes_the_strength(self, win_backend):
        mod, _, _ = win_backend
        backend = mod.DiffusersBackend()
        backend._model_info = self._info(turbo=True)
        reference = Image.new("RGB", (1000, 790))

        with patch.object(mod, "generate_moss", return_value="image") as mock_generate:
            result = backend.image_to_image(MagicMock(), reference, "a car", 0.4, 8, 5, 1.0)

        assert result == "image"
        assert (mock_generate.call_args.kwargs["width"], mock_generate.call_args.kwargs["height"]) == (992, 784)
        assert mock_generate.call_args.kwargs["image"] is reference and mock_generate.call_args.kwargs["strength"] == 0.4
        assert mock_generate.call_args.kwargs["sampler"].steps == 8


# ---------------------------------------------------------------------------
# load_model: full-precision path (no quantize)
# ---------------------------------------------------------------------------


class TestLoadModelFullPrecision:
    """load_model without quantize uses group offloading path."""

    def test_full_precision_loads_pipeline(self, win_backend):
        mod, _, _ = win_backend

        fake_pipeline = MagicMock(spec=["vae", "transformer", "text_encoder"])
        mod.AutoPipelineForText2Image.from_pretrained.return_value = fake_pipeline

        with patch.object(mod, "detect_image_model", return_value=_make_model_info()):
            backend = mod.DiffusersBackend()
            pipeline, info = backend.load_model("fake-model-path")

        mod.AutoPipelineForText2Image.from_pretrained.assert_called_once()
        assert info.family == "zimage"
        assert pipeline is fake_pipeline
        # Tiling is set on the VAE itself, so pipelines without the wrapper methods (Krea 2) tile too.
        fake_pipeline.vae.enable_slicing.assert_called_once()
        fake_pipeline.vae.enable_tiling.assert_called_once()

    @pytest.mark.parametrize(("top_level_children", "expected"), [(["Embedding", "ModuleList", "RMSNorm"], "block_level"), (["Qwen3VLVisionModel", "Qwen3VLTextModel"], "leaf_level")])
    def test_text_encoder_streams_by_block_only_with_a_top_level_layer_list(self, win_backend, top_level_children, expected):
        """Block-level offloading needs the layers in a top-level list; nested layouts (Krea 2's Qwen3-VL) stream by leaf."""
        mod, _, diffusers_mod = win_backend
        fake_pipeline = MagicMock()
        fake_pipeline.text_encoder.children.return_value = [type(name, (), {})() for name in top_level_children]
        mod.AutoPipelineForText2Image.from_pretrained.return_value = fake_pipeline

        with patch.object(mod, "detect_image_model", return_value=_make_model_info()):
            mod.DiffusersBackend().load_model("fake-model-path")

        offload_types = {call.args[0]: call.kwargs["offload_type"] for call in diffusers_mod.hooks.apply_group_offloading.call_args_list}
        assert offload_types == {fake_pipeline.transformer: "block_level", fake_pipeline.text_encoder: expected}

    def test_precision_mapping(self, win_backend):
        """torch_dtype is mapped from precision string."""
        mod, _, _ = win_backend

        fake_pipeline = MagicMock()
        mod.AutoPipelineForText2Image.from_pretrained.return_value = fake_pipeline

        with patch.object(mod, "detect_image_model", return_value=_make_model_info()):
            backend = mod.DiffusersBackend()
            backend.load_model("fake-model-path", precision="float16")

        call_kwargs = mod.AutoPipelineForText2Image.from_pretrained.call_args[1]
        assert call_kwargs["dtype"] == "fp16-sentinel"


# ---------------------------------------------------------------------------
# load_model: quantized paths (4-bit, 8-bit)
# ---------------------------------------------------------------------------


class TestLoadModelQuantized:
    """q4 loads NF4 onto the GPU; q8 stores weights in FP8 and streams them like unquantized ones."""

    def test_q4_loads_nf4(self, win_backend):
        mod, _, _ = win_backend
        fake_pipeline = MagicMock()

        with (
            patch.object(mod, "detect_image_model", return_value=_make_model_info()),
            patch.object(mod, "_load_nf4", return_value=fake_pipeline) as mock_nf4,
        ):
            pipeline, _info = mod.DiffusersBackend().load_model("fake-model-path", quantize=4)

        assert pipeline is fake_pipeline
        assert mock_nf4.call_args.kwargs["prequantized"] is False

    def test_q8_casts_to_fp8_and_streams(self, win_backend):
        mod, _, diffusers_mod = win_backend
        components = {"text_encoder": MagicMock(), "transformer": MagicMock()}

        with (
            patch.object(mod, "detect_image_model", return_value=_make_model_info()),
            patch.object(mod, "load_fp8_components", return_value=components) as mock_fp8,
            patch.object(mod, "_load_nf4") as mock_nf4,
        ):
            mod.DiffusersBackend().load_model("fake-model-path", quantize=8)

        mock_fp8.assert_called_once()
        mock_nf4.assert_not_called()
        call_kwargs = mod.AutoPipelineForText2Image.from_pretrained.call_args.kwargs
        assert call_kwargs["text_encoder"] is components["text_encoder"]
        assert call_kwargs["transformer"] is components["transformer"]
        assert diffusers_mod.hooks.apply_group_offloading.call_count == 2

    @pytest.mark.parametrize(("bits", "loader"), [(8, "load_fp8_components"), (4, "_load_nf4")])
    def test_stored_copy_loads_at_its_recorded_level(self, win_backend, tmp_path, bits, loader):
        mod, _, _ = win_backend
        stored = tmp_path / f"atlas@q{bits}"
        stored.mkdir()
        (stored / "ziv-quant.json").write_text(f'{{"bits": {bits}}}')
        mock_return = {"text_encoder": MagicMock(), "transformer": MagicMock()} if bits == 8 else MagicMock()

        with (
            patch.object(mod, "detect_image_model", return_value=_make_model_info()),
            patch.object(mod, loader, return_value=mock_return) as mock_loader,
        ):
            mod.DiffusersBackend().load_model(str(stored), quantize=None)

        mock_loader.assert_called_once()
        if bits == 4:
            assert mock_loader.call_args.kwargs["prequantized"] is True

    def test_unsupported_quantize_warns_and_loads_at_full_precision(self, win_backend):
        mod, _, _ = win_backend
        mod.AutoPipelineForText2Image.from_pretrained.return_value = MagicMock()

        with patch.object(mod, "detect_image_model", return_value=_make_model_info()), pytest.warns(UserWarning):
            mod.DiffusersBackend().load_model("fake-model-path", quantize=3)

        assert "text_encoder" not in mod.AutoPipelineForText2Image.from_pretrained.call_args.kwargs


class TestStoredQuants:
    def test_format_names_the_libraries_each_level_depends_on(self, win_backend):
        """A bitsandbytes upgrade must not make FP8 (q8) copies stale."""
        mod, _, _ = win_backend
        with patch.object(mod.importlib.metadata, "version", side_effect=lambda package: "1.0"):
            backend = mod.DiffusersBackend()
            assert backend.stored_quant_format(8) == "diffusers-1.0+transformers-1.0"
            assert backend.stored_quant_format(4) == "diffusers-1.0+transformers-1.0+bitsandbytes-1.0"
            assert backend.stored_quant_format(6) is None

    def test_q8_is_written_from_files_and_q4_saved_from_the_model(self, win_backend):
        mod, _, _ = win_backend
        backend = mod.DiffusersBackend()
        assert backend.quantizes_from_files(8) is True
        assert backend.quantizes_from_files(4) is False

    def test_save_quantized_saves_the_pipeline(self, win_backend):
        mod, _, _ = win_backend
        pipeline = MagicMock()
        mod.DiffusersBackend().save_quantized(pipeline, "/models/zit@q4")
        pipeline.save_pretrained.assert_called_once_with("/models/zit@q4")

    def test_write_quantized_files_writes_the_fp8_copy(self, win_backend):
        mod, _, _ = win_backend
        cancelled = MagicMock(return_value=False)
        with patch.object(mod, "write_fp8_copy") as mock_write:
            mod.DiffusersBackend().write_quantized_files("/models/zit", "/models/.zit@q8.partial", 8, cancelled)
        source, target, _dtype, passed_cancelled = mock_write.call_args.args
        assert (str(source), str(target), passed_cancelled) == ("/models/zit", "/models/.zit@q8.partial", cancelled)

    def test_write_quantized_files_rejects_q4(self, win_backend):
        mod, _, _ = win_backend
        with pytest.raises(ValueError):
            mod.DiffusersBackend().write_quantized_files("/models/zit", "/models/.zit@q4.partial", 4)


# ---------------------------------------------------------------------------
# load_model: LoRA loading
# ---------------------------------------------------------------------------


class TestLoadModelLoRA:
    """LoRA files go to diffusers by path; only files with LoKr tensors are split, their LoKr layers applied last."""

    @pytest.fixture
    def loras(self, win_backend):
        """Patch LoRA file reading and the LoKr module; yield (module, parts by path, LoKr paths, LoKr module stub)."""
        mod, _, _ = win_backend
        parts: dict[str, SimpleNamespace] = {}
        lokr_paths: set[str] = set()
        lokr_module = MagicMock()
        lokr_module.split_lora_tensors.side_effect = lambda state: state
        lokr_module.apply_lokr.return_value = ()
        safetensors_torch = MagicMock()
        safetensors_torch.load_file.side_effect = lambda path: parts[path]
        fakes = {"safetensors.torch": safetensors_torch, "zvisiongenerator.backends.cuda.image.lokr": lokr_module}
        with patch.dict(sys.modules, fakes), patch.object(mod, "_has_lokr", side_effect=lambda path: path in lokr_paths):
            yield mod, parts, lokr_paths, lokr_module

    @staticmethod
    def _parts(rest=None, lora=None, lokr=None, unsupported=()):
        return SimpleNamespace(rest=rest or {}, lora=lora or {}, lokr=lokr or {}, unsupported=tuple(unsupported))

    @staticmethod
    def _load(mod, paths, weights, pipeline=None, registers=True, *, quantize=None, family="zimage"):
        pipeline = pipeline or MagicMock()
        # diffusers registers each adapter it loads tensors for (unless *registers* is off).
        pipeline.get_list_adapters.side_effect = lambda: {"transformer": [call.kwargs["adapter_name"] for call in pipeline.load_lora_weights.call_args_list] if registers else []}
        mod.AutoPipelineForText2Image.from_pretrained.return_value = pipeline
        with (
            patch.object(mod, "detect_image_model", return_value=_make_model_info(family=family)),
            patch.object(mod, "_load_nf4", return_value=pipeline),
            patch.object(mod, "load_fp8_components", return_value={}),
        ):
            mod.DiffusersBackend().load_model("fake-model-path", quantize=quantize, lora_paths=paths, lora_weights=weights)
        return pipeline

    def test_files_without_lokr_load_by_path(self, loras):
        mod, _, _, lokr_module = loras

        pipeline = self._load(mod, ["/l/a.safetensors", "/l/b.bin"], [0.8, 0.5])

        assert [call.args[0] for call in pipeline.load_lora_weights.call_args_list] == ["/l/a.safetensors", "/l/b.bin"]
        pipeline.set_adapters.assert_called_once_with(["lora_0", "lora_1"], adapter_weights=[0.8, 0.5])
        lokr_module.apply_lokr.assert_not_called()

    def test_a_file_the_converter_cannot_read_goes_to_diffusers_under_its_normalised_names(self, loras):
        mod, _, _, lokr_module = loras
        renamed = self._parts(rest={"diffusion_model.a.lora_A.weight": 1, "diffusion_model.a.lora_B.weight": 2}, lora={"diffusion_model.a.lora_A.weight": 1})
        seen: list[tuple[str, str]] = []

        def normalised(path, family):
            seen.append((path, family))
            return renamed if path == "/l/peft.safetensors" else None

        with patch.object(mod, "normalised_lora_state", side_effect=normalised):
            pipeline = self._load(mod, ["/l/peft.safetensors", "/l/plain.safetensors"], [0.6, 1.0], family="flux2_klein")

        assert seen == [("/l/peft.safetensors", "flux2_klein"), ("/l/plain.safetensors", "flux2_klein")]
        assert [call.args[0] for call in pipeline.load_lora_weights.call_args_list] == [renamed.rest, "/l/plain.safetensors"]
        pipeline.set_adapters.assert_called_once_with(["lora_0", "lora_1"], adapter_weights=[0.6, 1.0])
        lokr_module.apply_lokr.assert_not_called()

    def test_a_rejected_file_is_retried_from_its_normalised_tensors(self, loras):
        mod, _, _, _ = loras
        renamed = self._parts(rest={"a": 1, "d": 2}, lora={"a": 1}, unsupported=["layers.0.diff"])
        pipeline = MagicMock()
        pipeline.load_lora_weights.side_effect = lambda source, **_k: (_ for _ in ()).throw(ValueError("leftover keys")) if source is renamed.rest else None

        with patch.object(mod, "normalised_lora_state", return_value=renamed), pytest.warns(UserWarning, match="unsupported type"):
            self._load(mod, ["/l/peft.safetensors"], None, pipeline)

        assert [call.args[0] for call in pipeline.load_lora_weights.call_args_list] == [renamed.rest, renamed.lora]

    def test_lokr_files_load_their_other_tensors_first_and_lokr_layers_last(self, loras):
        mod, parts, lokr_paths, lokr_module = loras
        lokr_paths.add("/l/mixed.safetensors")
        parts["/l/mixed.safetensors"] = self._parts(rest={"b": 1}, lora={"b": 1}, lokr={"blocks.0": {}})
        events: list[str] = []
        pipeline = MagicMock()
        pipeline.load_lora_weights.side_effect = lambda source, **_k: events.append(f"lora {source}")
        lokr_module.apply_lokr.side_effect = lambda *_a, **_k: events.append("lokr") or ()

        self._load(mod, ["/l/mixed.safetensors", "/l/plain.safetensors"], [0.7, 1.0], pipeline)

        assert events == ["lora {'b': 1}", "lora /l/plain.safetensors", "lokr"]
        assert lokr_module.apply_lokr.call_args.args[1:3] == ({"blocks.0": {}}, 0.7)
        pipeline.set_adapters.assert_called_once_with(["lora_0", "lora_1"], adapter_weights=[0.7, 1.0])

    def test_lokr_only_file_adds_no_diffusers_adapter(self, loras):
        mod, parts, lokr_paths, lokr_module = loras
        lokr_paths.add("/l/lokr.safetensors")
        parts["/l/lokr.safetensors"] = self._parts(lokr={"blocks.0": {}})

        pipeline = self._load(mod, ["/l/lokr.safetensors"], [1.0])

        pipeline.load_lora_weights.assert_not_called()
        pipeline.set_adapters.assert_not_called()
        lokr_module.apply_lokr.assert_called_once()

    def test_tensors_diffusers_rejects_are_skipped_with_a_warning(self, loras):
        mod, parts, _, _ = loras
        parts["/l/diff.safetensors"] = self._parts(rest={"a": 1, "d": 2}, lora={"a": 1}, unsupported=["cap_embedder.0.diff"])

        def reject_whole_files(source, **_kwargs):
            if isinstance(source, str):
                raise ValueError("leftover keys")

        pipeline = MagicMock()
        pipeline.load_lora_weights.side_effect = reject_whole_files

        with pytest.warns(UserWarning, match="diff.safetensors"):
            self._load(mod, ["/l/diff.safetensors"], None, pipeline)

        assert pipeline.load_lora_weights.call_args.args[0] == {"a": 1}
        pipeline.set_adapters.assert_called_once_with(["lora_0"], adapter_weights=[1.0])

    def test_a_file_diffusers_loads_nothing_from_adds_no_adapter(self, loras):
        mod, _, _, _ = loras

        with pytest.warns(UserWarning, match="none of its LoRA tensors"):
            pipeline = self._load(mod, ["/l/other-model.safetensors"], [1.0], registers=False)

        pipeline.set_adapters.assert_not_called()

    def test_a_rejected_file_with_nothing_to_leave_out_fails(self, loras):
        mod, parts, _, _ = loras
        parts["/l/bad.safetensors"] = self._parts(rest={"a": 1}, lora={"a": 1})
        pipeline = MagicMock()
        pipeline.load_lora_weights.side_effect = ValueError("not a LoRA for this model")

        with pytest.raises(ValueError):
            self._load(mod, ["/l/bad.safetensors"], None, pipeline)

    @staticmethod
    def _unmatched_for(error, *rejected):
        """Return a ``load_lora_weights`` stand-in that raises *error* for the sources named in *rejected*."""

        def load(source, **_kwargs):
            if source in rejected:
                raise error("no module matches")

        return load

    def test_a_file_none_of_whose_tensors_match_is_skipped_with_a_warning(self, loras, fake_peft_no_match):
        mod, _, _, _ = loras
        pipeline = MagicMock()
        pipeline.load_lora_weights.side_effect = self._unmatched_for(fake_peft_no_match, "/l/other-model.safetensors")

        with pytest.warns(UserWarning, match="other-model.safetensors: none of its LoRA tensors match"):
            self._load(mod, ["/l/other-model.safetensors", "/l/good.safetensors"], [1.0, 0.5], pipeline)

        pipeline.set_adapters.assert_called_once_with(["lora_1"], adapter_weights=[0.5])

    def test_the_split_part_none_of_whose_tensors_match_is_skipped_but_its_lokr_layers_apply(self, loras, fake_peft_no_match):
        mod, parts, lokr_paths, lokr_module = loras
        lokr_paths.add("/l/mixed.safetensors")
        parts["/l/mixed.safetensors"] = self._parts(rest={"b": 1}, lora={"b": 1}, lokr={"blocks.0": {}})
        pipeline = MagicMock()
        pipeline.load_lora_weights.side_effect = lambda source, **_k: (_ for _ in ()).throw(fake_peft_no_match("no module matches"))

        with pytest.warns(UserWarning, match="mixed.safetensors: none of its LoRA tensors match"):
            self._load(mod, ["/l/mixed.safetensors"], [1.0], pipeline)

        pipeline.set_adapters.assert_not_called()
        lokr_module.apply_lokr.assert_called_once()

    def test_the_retry_without_unsupported_tensors_may_match_nothing_too(self, loras, fake_peft_no_match):
        mod, parts, _, _ = loras
        parts["/l/diff.safetensors"] = self._parts(rest={"a": 1, "d": 2}, lora={"a": 1}, unsupported=["cap_embedder.0.diff"])

        def load(source, **_kwargs):
            raise ValueError("leftover keys") if isinstance(source, str) else fake_peft_no_match("no module matches")

        pipeline = MagicMock()
        pipeline.load_lora_weights.side_effect = load

        with pytest.warns(UserWarning) as caught:
            self._load(mod, ["/l/diff.safetensors"], None, pipeline)

        assert any("none of its LoRA tensors match" in str(w.message) for w in caught)
        pipeline.set_adapters.assert_not_called()

    def test_a_plain_value_error_still_fails_when_peft_is_loaded(self, loras, fake_peft_no_match):
        mod, parts, _, _ = loras
        parts["/l/bad.safetensors"] = self._parts(rest={"a": 1}, lora={"a": 1})
        pipeline = MagicMock()
        pipeline.load_lora_weights.side_effect = ValueError("not a LoRA for this model")

        with pytest.raises(ValueError, match="not a LoRA"):
            self._load(mod, ["/l/bad.safetensors"], None, pipeline)

    @pytest.mark.parametrize(
        ("hook", "quantize", "family", "restored"),
        [
            (None, 4, "krea2", True),
            (object(), 4, "krea2", False),
        ],
    )
    def test_cpu_offload_is_restored_only_for_an_offloaded_pipeline_that_lost_its_hooks(self, loras, fake_peft_no_match, hook, quantize, family, restored):
        mod, _, _, _ = loras
        pipeline = MagicMock()
        pipeline.transformer._hf_hook = hook
        pipeline.load_lora_weights.side_effect = self._unmatched_for(fake_peft_no_match, "/l/other-model.safetensors")

        with pytest.warns(UserWarning):
            self._load(mod, ["/l/other-model.safetensors"], [1.0], pipeline, quantize=quantize, family=family)

        assert pipeline.enable_model_cpu_offload.call_count == int(restored)

    def test_cpu_offload_is_restored_after_the_split_retry_loads(self, loras):
        mod, parts, _, _ = loras
        parts["/l/diff.safetensors"] = self._parts(rest={"a": 1, "d": 2}, lora={"a": 1}, unsupported=["cap_embedder.0.diff"])

        def reject_whole_files(source, **_kwargs):
            if isinstance(source, str):
                raise ValueError("leftover keys")

        pipeline = MagicMock()
        pipeline.transformer._hf_hook = None
        pipeline.load_lora_weights.side_effect = reject_whole_files

        with pytest.warns(UserWarning, match="diff.safetensors"):
            self._load(mod, ["/l/diff.safetensors"], None, pipeline, quantize=4, family="krea2")

        pipeline.set_adapters.assert_called_once_with(["lora_0"], adapter_weights=[1.0])
        pipeline.enable_model_cpu_offload.assert_called_once_with()

    def test_lokr_layers_the_model_lacks_are_skipped_with_a_warning(self, loras):
        mod, parts, lokr_paths, lokr_module = loras
        lokr_paths.add("/l/lokr.safetensors")
        parts["/l/lokr.safetensors"] = self._parts(lokr={"blocks.0": {}, "te.0": {}})
        lokr_module.apply_lokr.return_value = ("te.0",)

        with pytest.warns(UserWarning, match="lokr.safetensors"):
            self._load(mod, ["/l/lokr.safetensors"], [1.0])


# ---------------------------------------------------------------------------
# text_to_image: basic generation path
# ---------------------------------------------------------------------------


class TestTextToImage:
    """text_to_image delegates to the pipeline and returns the first image."""

    @staticmethod
    def _ready_backend(mod):
        backend = mod.DiffusersBackend()
        backend._model_info = _make_model_info()
        fake_result = MagicMock()
        fake_result.images = [Image.new("RGB", (64, 64))]
        mock_model = MagicMock()
        mock_model.return_value = fake_result
        mock_model._interrupt = False
        return backend, mock_model

    def test_returns_the_image_and_passes_guidance_scale(self, win_backend):
        mod, _, _ = win_backend
        backend, mock_model = self._ready_backend(mod)

        result = _t2i(backend, mock_model, guidance=3.5)

        assert isinstance(result, Image.Image)
        mock_model.assert_called_once()
        assert mock_model.call_args[1]["guidance_scale"] == 3.5

    def test_negative_prompt_non_flux(self, win_backend):
        """Non-flux models pass negative_prompt."""
        mod, _, _ = win_backend
        backend, mock_model = self._ready_backend(mod)
        backend._model_info = _make_model_info(family="unknown")

        _t2i(backend, mock_model, negative_prompt="ugly")

        assert mock_model.call_args[1]["negative_prompt"] == "ugly"

    def test_negative_prompt_suppressed_for_flux(self, win_backend):
        """Flux models do NOT receive negative_prompt."""
        mod, _, _ = win_backend
        backend, mock_model = self._ready_backend(mod)
        backend._model_info = _make_model_info(family="flux1")

        _t2i(backend, mock_model, negative_prompt="ugly")

        assert "negative_prompt" not in mock_model.call_args[1]

    def test_skip_signal_returns_none(self, win_backend):
        """When skip_signal fires, result is None."""
        mod, _, _ = win_backend
        backend, mock_model = self._ready_backend(mod)
        mock_model._interrupt = True
        skip = MagicMock()
        skip.check.return_value = True

        assert _t2i(backend, mock_model, skip_signal=skip) is None

    def test_restores_scheduler_after_beta(self, win_backend):
        """Scheduler is restored even when beta scheduler is requested."""
        mod, _, _ = win_backend
        backend, mock_model = self._ready_backend(mod)
        original_scheduler = mock_model.scheduler

        _t2i(backend, mock_model, scheduler="beta")

        assert mock_model.scheduler is original_scheduler

    def test_zimage_beta_passes_beta_sigmas(self, win_backend):
        """Z-Image gets the beta grid as ``sigmas`` so the model shift applies after the beta spacing."""
        mod, _, _ = win_backend
        backend, mock_model = self._ready_backend(mod)
        _t2i(backend, mock_model, steps=6, guidance=0.0, scheduler="beta")
        sigmas = mock_model.call_args[1]["sigmas"]
        assert len(sigmas) == 6
        assert sigmas[0] == pytest.approx(1.0)
        assert sigmas == sorted(sigmas, reverse=True)

    def test_other_families_beta_passes_no_sigmas(self, win_backend):
        mod, _, _ = win_backend
        backend, mock_model = self._ready_backend(mod)
        backend._model_info = _make_model_info(family="flux2")
        _t2i(backend, mock_model, guidance=None, scheduler="beta")
        assert "sigmas" not in mock_model.call_args[1]

    def test_no_scheduler_passes_no_sigmas(self, win_backend):
        mod, _, _ = win_backend
        backend, mock_model = self._ready_backend(mod)
        _t2i(backend, mock_model, guidance=0.0)
        assert "sigmas" not in mock_model.call_args[1]

    def test_default_guidance_for_flux(self, win_backend):
        """Flux models get guidance_scale=1.0 when guidance is None."""
        mod, _, _ = win_backend
        backend, mock_model = self._ready_backend(mod)
        backend._model_info = _make_model_info(family="flux2")
        _t2i(backend, mock_model, guidance=None)
        assert mock_model.call_args[1]["guidance_scale"] == 1.0


# ---------------------------------------------------------------------------
# image_to_image: basic refinement path
# ---------------------------------------------------------------------------


class TestImageToImage:
    """image_to_image creates an img2img pipeline and refines the input."""

    def test_keeps_no_pipeline_between_calls(self, win_backend):
        """The long-lived backend must not keep a finished job's model alive through a cached pipeline."""
        mod, _, _ = win_backend
        backend = mod.DiffusersBackend()
        backend._model_info = _make_model_info()
        fake_i2i_pipe = MagicMock()
        fake_i2i_pipe.return_value.images = [Image.new("RGB", (64, 64))]
        fake_i2i_pipe._interrupt = False
        from_pipe = sys.modules["diffusers"].AutoPipelineForImage2Image.from_pipe
        from_pipe.return_value = fake_i2i_pipe
        model = MagicMock()

        for _ in range(2):
            backend.image_to_image(model=model, image=Image.new("RGB", (64, 64)), prompt="sharper", strength=0.3, steps=4, seed=42, guidance=0.5)

        assert [call.args[0] for call in from_pipe.call_args_list] == [model, model]
        assert all(call.kwargs["dtype"] is model.vae.dtype for call in from_pipe.call_args_list)  # never float32
        assert all(not isinstance(value, MagicMock) for value in vars(backend).values())

    def test_skip_signal_returns_none(self, win_backend):
        """When skip_signal fires during img2img, result is None."""
        mod, _, _ = win_backend
        backend = mod.DiffusersBackend()
        backend._model_info = _make_model_info()

        fake_i2i_pipe = MagicMock()
        fake_result = MagicMock()
        fake_result.images = [Image.new("RGB", (64, 64))]
        fake_i2i_pipe.return_value = fake_result
        fake_i2i_pipe._interrupt = True
        sys.modules["diffusers"].AutoPipelineForImage2Image.from_pipe.return_value = fake_i2i_pipe

        skip = MagicMock()
        skip.check.return_value = True

        result = backend.image_to_image(
            model=MagicMock(),
            image=Image.new("RGB", (64, 64)),
            prompt="sharper",
            strength=0.3,
            steps=4,
            seed=42,
            guidance=0.5,
            skip_signal=skip,
        )
        assert result is None

    def test_negative_prompt_passed_for_non_flux(self, win_backend):
        """Non-flux models receive negative_prompt in img2img."""
        mod, _, _ = win_backend
        backend = mod.DiffusersBackend()
        backend._model_info = _make_model_info(family="unknown")

        fake_i2i_pipe = MagicMock()
        fake_result = MagicMock()
        fake_result.images = [Image.new("RGB", (64, 64))]
        fake_i2i_pipe.return_value = fake_result
        fake_i2i_pipe._interrupt = False
        sys.modules["diffusers"].AutoPipelineForImage2Image.from_pipe.return_value = fake_i2i_pipe

        backend.image_to_image(
            model=MagicMock(),
            image=Image.new("RGB", (64, 64)),
            prompt="sharper",
            strength=0.3,
            steps=4,
            seed=42,
            guidance=0.5,
            negative_prompt="blurry",
        )
        call_kwargs = fake_i2i_pipe.call_args[1]
        assert call_kwargs["negative_prompt"] == "blurry"


# ---------------------------------------------------------------------------
# _make_skip_callback
# ---------------------------------------------------------------------------


class TestMakeSkipCallback:
    """_make_skip_callback integrates skip_signal with pipeline interruption."""

    def test_callback_sets_interrupt_on_skip(self, win_backend):
        mod, _, _ = win_backend
        skip = MagicMock()
        skip.check.return_value = True
        cb = mod._make_skip_callback(skip)

        fake_pipe = MagicMock()
        result = cb(fake_pipe, step=0, timestep=0.0, callback_kwargs={"key": "val"})

        assert fake_pipe._interrupt is True
        assert result == {"key": "val"}


# ---------------------------------------------------------------------------
# Live previews in the step callback
# ---------------------------------------------------------------------------


class TestStepCallbackLivePreview:
    """Step callbacks carry live previews from the backend's LivePreview tracker."""

    def test_preview_is_added_to_the_progress_payload(self, win_backend):
        mod, _, _ = win_backend
        preview_image = Image.new("RGB", (8, 8))
        live_preview = MagicMock()
        live_preview.observe.return_value = preview_image
        events = []
        cb = mod._make_step_callback(None, total_steps=4, step_callback=events.append, live_preview=live_preview)

        fake_pipe = MagicMock()
        latents = object()
        cb(fake_pipe, step=1, timestep=0.0, callback_kwargs={"latents": latents})

        live_preview.observe.assert_called_once_with(fake_pipe, 1, latents)
        assert events == [{"current_step": 2, "total_steps": 4, "preview": preview_image}]

    def test_steps_without_a_preview_report_progress_only(self, win_backend):
        mod, _, _ = win_backend
        live_preview = MagicMock()
        live_preview.observe.return_value = None
        events = []
        cb = mod._make_step_callback(None, total_steps=4, step_callback=events.append, live_preview=live_preview)

        cb(MagicMock(), step=0, timestep=0.0, callback_kwargs={"latents": object()})

        assert events == [{"current_step": 1, "total_steps": 4}]

    def test_partial_runs_report_the_pipeline_step_count(self, win_backend):
        mod, _, _ = win_backend
        events = []
        cb = mod._make_step_callback(None, total_steps=7, step_callback=events.append)

        cb(SimpleNamespace(num_timesteps=3, _interrupt=False), step=2, timestep=0.0, callback_kwargs={})

        assert events == [{"current_step": 3, "total_steps": 3}]

    @pytest.mark.parametrize(("family", "expects_preview"), [("zimage", True), ("flux2_klein", True), ("krea2", True), ("flux1", False)])
    def test_text_to_image_requests_latents_only_for_previewable_families(self, win_backend, family, expects_preview):
        mod, _, _ = win_backend
        backend = mod.DiffusersBackend()
        backend._model_info = _make_model_info(family=family)
        model = MagicMock()
        model.return_value.images = [Image.new("RGB", (64, 64))]
        model._interrupt = False

        backend.text_to_image(model=model, prompt="p", width=64, height=64, seed=1, steps=4, guidance=1.0, step_callback=lambda _: None)

        kwargs = model.call_args.kwargs
        assert callable(kwargs["callback_on_step_end"])
        assert (kwargs.get("callback_on_step_end_tensor_inputs") == ["latents"]) is expects_preview

    def test_no_preview_without_a_progress_callback(self, win_backend):
        mod, _, _ = win_backend
        backend = mod.DiffusersBackend()
        backend._model_info = _make_model_info(family="zimage")
        model = MagicMock()
        model.return_value.images = [Image.new("RGB", (64, 64))]
        model._interrupt = False
        skip = MagicMock()
        skip.check.return_value = False

        backend.text_to_image(model=model, prompt="p", width=64, height=64, seed=1, steps=4, guidance=1.0, skip_signal=skip)

        assert "callback_on_step_end_tensor_inputs" not in model.call_args.kwargs


# ---------------------------------------------------------------------------
# get_backend() returns diffusers on Windows and Linux
# ---------------------------------------------------------------------------


class TestGetBackendDiffusersPlatforms:
    """get_backend() returns the diffusers backend on Windows and Linux (through the real backend map)."""

    @pytest.mark.parametrize("platform", ["win32", "linux"])
    def test_get_backend_returns_diffusers(self, win_backend, platform):
        import zvisiongenerator.backends as backends_mod

        saved = dict(backends_mod.BACKENDS)
        backends_mod.BACKENDS.clear()
        try:
            with patch.object(backends_mod, "sys") as mock_sys:
                mock_sys.platform = platform
                assert backends_mod.get_backend().name == "diffusers"
        finally:
            backends_mod.BACKENDS.clear()
            backends_mod.BACKENDS.update(saved)


# ---------------------------------------------------------------------------
# Krea 2
# ---------------------------------------------------------------------------


class TestKrea2:
    """Krea 2 converts guidance, offloads at NF4, and runs img2img through its text-to-image pipeline."""

    @pytest.mark.parametrize(("guidance", "expected"), [(1.0, 0.0), (0.0, 0.0), (5.5, 4.5)])
    def test_text_to_image_converts_standard_cfg_to_krea2_scale(self, win_backend, guidance, expected):
        mod, _, _ = win_backend
        backend, mock_model = TestTextToImage._ready_backend(mod)
        backend._model_info = _make_model_info(family="krea2")

        backend.text_to_image(model=mock_model, prompt="a fox", width=64, height=64, seed=1, steps=8, guidance=guidance)

        assert mock_model.call_args[1]["guidance_scale"] == expected

    def test_text_to_image_passes_negative_prompt(self, win_backend):
        mod, _, _ = win_backend
        backend, mock_model = TestTextToImage._ready_backend(mod)
        backend._model_info = _make_model_info(family="krea2")

        backend.text_to_image(model=mock_model, prompt="a fox", width=64, height=64, seed=1, steps=8, guidance=3.0, negative_prompt="blurry")

        assert mock_model.call_args[1]["negative_prompt"] == "blurry"

    def test_image_to_image_runs_text_to_image_pipeline_from_noised_latents(self, win_backend):
        mod, _, _ = win_backend
        backend = mod.DiffusersBackend()
        backend._model_info = _make_model_info(family="krea2")
        pipe = MagicMock(vae_scale_factor=8, patch_size=2, _interrupt=False)
        pipe.return_value.images = [Image.new("RGB", (64, 48))]

        with patch.object(mod, "_krea2_img2img_latents", return_value="noised-latents") as mock_latents:
            result = backend.image_to_image(model=pipe, image=Image.new("RGB", (70, 50)), prompt="p", strength=0.4, steps=8, seed=1, guidance=1.0)

        assert result is pipe.return_value.images[0]
        kwargs = pipe.call_args.kwargs
        assert (kwargs["width"], kwargs["height"]) == (64, 48)
        assert kwargs["sigmas"] == pytest.approx([0.5, 0.375, 0.25, 0.125])
        assert kwargs["num_inference_steps"] == 4
        assert kwargs["latents"] == "noised-latents"
        assert kwargs["guidance_scale"] == 0.0
        assert mock_latents.call_args.args[2] == kwargs["sigmas"]
        sys.modules["diffusers"].AutoPipelineForImage2Image.from_pipe.assert_not_called()

    def test_image_to_image_returns_none_when_skipped(self, win_backend):
        mod, _, _ = win_backend
        backend = mod.DiffusersBackend()
        backend._model_info = _make_model_info(family="krea2")
        pipe = MagicMock(vae_scale_factor=8, patch_size=2, _interrupt=True)
        skip = MagicMock()
        skip.check.return_value = False

        with patch.object(mod, "_krea2_img2img_latents", return_value="noised-latents"):
            result = backend.image_to_image(model=pipe, image=Image.new("RGB", (64, 64)), prompt="p", strength=0.5, steps=8, seed=1, guidance=1.0, skip_signal=skip)

        assert result is None
        assert pipe._interrupt is False

    @pytest.mark.parametrize(
        ("steps", "strength", "expected"),
        [(8, 0.4, [0.5, 0.375, 0.25, 0.125]), (8, 1.0, [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]), (8, 0.05, [0.125]), (4, 0.5, [0.5, 0.25])],
    )
    def test_img2img_sigmas_keep_the_tail_of_the_schedule(self, win_backend, steps, strength, expected):
        mod, _, _ = win_backend

        assert mod._img2img_sigmas(steps, strength) == pytest.approx(expected)

    @pytest.mark.parametrize(("requested", "strength"), [(8, 0.3), (8, 0.7), (28, 0.45)])
    def test_img2img_runs_the_requested_steps_after_workflow_inflation(self, win_backend, requested, strength):
        mod, _, _ = win_backend
        inflated = math.floor(requested / strength)  # as image_stages passes steps to image_to_image

        assert len(mod._img2img_sigmas(inflated, strength)) == requested

    def test_load_model_passes_family_to_nf4_loader(self, win_backend):
        mod, _, _ = win_backend

        with (
            patch.object(mod, "detect_image_model", return_value=_make_model_info(family="krea2")),
            patch.object(mod, "_load_nf4", return_value=MagicMock()) as mock_nf4,
        ):
            mod.DiffusersBackend().load_model("unsloth/Krea-2-Turbo", quantize=4)

        assert mock_nf4.call_args.args[2] == "krea2"

    def test_nf4_moves_text_encoder_off_gpu_and_uses_model_offload(self, win_backend):
        mod, _, _ = win_backend
        pipeline = mod.AutoPipelineForText2Image.from_pretrained.return_value

        with patch.object(mod, "load_nf4_components", return_value={}) as mock_components:
            mod._load_nf4("unsloth/Krea-2-Turbo", "bf16-sentinel", "krea2", prequantized=False)

        assert mock_components.call_args.kwargs["text_encoder_to_cpu"] is True
        pipeline.enable_model_cpu_offload.assert_called_once()

    def test_nf4_keeps_other_families_resident(self, win_backend):
        mod, _, _ = win_backend
        pipeline = mod.AutoPipelineForText2Image.from_pretrained.return_value

        with patch.object(mod, "load_nf4_components", return_value={}) as mock_components:
            mod._load_nf4("Tongyi-MAI/Z-Image-Turbo", "bf16-sentinel", "zimage", prequantized=False)

        assert mock_components.call_args.kwargs["text_encoder_to_cpu"] is False
        pipeline.enable_model_cpu_offload.assert_not_called()


# ---------------------------------------------------------------------------
# FLUX.2 Klein
# ---------------------------------------------------------------------------


class TestFlux2KleinImg2Img:
    """FLUX.2 Klein img2img runs through its text-to-image pipeline, whose ``__call__`` has no ``strength``."""

    @staticmethod
    def _strict_pipe():
        """Build a pipeline stand-in with Flux2KleinPipeline's call signature: unknown keywords raise ``TypeError``."""

        def call(prompt, width, height, num_inference_steps, sigmas, latents, guidance_scale, generator, callback_on_step_end=None, callback_on_step_end_tensor_inputs=None):
            return SimpleNamespace(images=[Image.new("RGB", (width, height))])

        return MagicMock(side_effect=call, vae_scale_factor=8, _interrupt=False)

    def test_image_to_image_never_passes_strength(self, win_backend):
        mod, _, _ = win_backend
        backend = mod.DiffusersBackend()
        backend._model_info = _make_model_info(family="flux2_klein")
        pipe = self._strict_pipe()

        with patch.object(mod, "_klein_img2img_latents", return_value="noised-latents") as mock_latents:
            result = backend.image_to_image(model=pipe, image=Image.new("RGB", (70, 50)), prompt="p", strength=0.4, steps=8, seed=1, guidance=None)

        assert result.size == (64, 48)
        kwargs = pipe.call_args.kwargs
        assert kwargs["sigmas"] == pytest.approx([0.5, 0.375, 0.25, 0.125])
        assert kwargs["num_inference_steps"] == 4
        assert kwargs["latents"] == "noised-latents"
        assert kwargs["guidance_scale"] == 1.0
        assert mock_latents.call_args.args[2:5] == (kwargs["sigmas"], 64, 48)
        sys.modules["diffusers"].AutoPipelineForImage2Image.from_pipe.assert_not_called()

    def test_image_to_image_returns_none_when_skipped(self, win_backend):
        mod, _, _ = win_backend
        backend = mod.DiffusersBackend()
        backend._model_info = _make_model_info(family="flux2_klein")
        pipe = self._strict_pipe()
        pipe._interrupt = True

        with patch.object(mod, "_klein_img2img_latents", return_value="noised-latents"):
            result = backend.image_to_image(model=pipe, image=Image.new("RGB", (64, 64)), prompt="p", strength=0.5, steps=8, seed=1, guidance=1.0, skip_signal=MagicMock())

        assert result is None
        assert pipe._interrupt is False

    def test_size_rounds_down_to_the_packed_latent_multiple(self, win_backend):
        mod, _, _ = win_backend
        pipe = MagicMock(vae_scale_factor=8)

        assert mod._klein_size(pipe, Image.new("RGB", (70, 50))) == (64, 48)
        assert mod._klein_size(pipe, Image.new("RGB", (5, 5))) == (16, 16)


class TestFlux2KleinReferences:
    """References reach the Klein pipeline as its ``image`` list, and no other family takes them."""

    @staticmethod
    def _backend(mod, family="flux2_klein"):
        backend = mod.DiffusersBackend()
        backend._model_info = _make_model_info(family=family)
        return backend

    @staticmethod
    def _refs():
        return [Image.new("RGB", (40, 30)), Image.new("RGB", (20, 60))]

    def test_text_to_image_passes_the_references_as_image(self, win_backend):
        mod, _, _ = win_backend
        pipe = MagicMock(return_value=SimpleNamespace(images=[Image.new("RGB", (64, 64))]), _interrupt=False)
        refs = self._refs()

        self._backend(mod).text_to_image(model=pipe, prompt="p", width=64, height=64, seed=1, steps=4, guidance=1.0, reference_images=refs)

        assert pipe.call_args.kwargs["image"] == refs

    def test_text_to_image_without_references_passes_no_image(self, win_backend):
        mod, _, _ = win_backend
        pipe = MagicMock(return_value=SimpleNamespace(images=[Image.new("RGB", (64, 64))]), _interrupt=False)

        self._backend(mod).text_to_image(model=pipe, prompt="p", width=64, height=64, seed=1, steps=4, guidance=1.0)

        assert "image" not in pipe.call_args.kwargs

    def test_start_image_and_references_share_one_call(self, win_backend):
        mod, _, _ = win_backend
        pipe = MagicMock(return_value=SimpleNamespace(images=[Image.new("RGB", (64, 48))]), vae_scale_factor=8, _interrupt=False)
        refs = self._refs()

        with patch.object(mod, "_klein_img2img_latents", return_value="noised-latents"):
            self._backend(mod).image_to_image(model=pipe, image=Image.new("RGB", (70, 50)), prompt="p", strength=0.4, steps=8, seed=1, guidance=1.0, reference_images=refs)

        kwargs = pipe.call_args.kwargs
        assert kwargs["image"] == refs
        assert kwargs["latents"] == "noised-latents"
        assert "strength" not in kwargs
