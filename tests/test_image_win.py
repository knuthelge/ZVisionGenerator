"""Mock-patched tests for zvisiongenerator.backends.image_win on macOS CI.

Exercises DiffusersBackend code paths without real torch/diffusers/CUDA by
mocking all heavy dependencies at the module level before import.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

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
    sys.modules.pop("zvisiongenerator.backends.image_win", None)
    import zvisiongenerator.backends.image_win as mod  # noqa: E402

    return mod


def _make_model_info(family="zimage", is_distilled=False, size=None):
    return ImageModelInfo(family=family, is_distilled=is_distilled, size=size)


# ---------------------------------------------------------------------------
# Fixture: import image_win with all heavy deps mocked (stays active per-test)
# ---------------------------------------------------------------------------


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
    sys.modules.pop("zvisiongenerator.backends.image_win", None)


@pytest.fixture()
def win_backend_no_cuda():
    """Same as win_backend but CUDA reports unavailable."""
    torch_mod = _stub_torch(cuda_available=False)
    diffusers_mod = MagicMock()
    fakes = _build_fake_modules(torch_mod, diffusers_mod)

    with patch.dict(sys.modules, fakes):
        mod = _fresh_import(fakes)
        yield mod, torch_mod, diffusers_mod

    sys.modules.pop("zvisiongenerator.backends.image_win", None)


# ---------------------------------------------------------------------------
# DiffusersBackend construction & protocol surface
# ---------------------------------------------------------------------------


class TestDiffusersBackendConstruction:
    """DiffusersBackend can be instantiated and has the right protocol surface."""

    def test_instantiation(self, win_backend):
        mod, _, _ = win_backend
        backend = mod.DiffusersBackend()
        assert backend.name == "diffusers"
        assert backend._img2img_pipe is None
        assert backend._model_info is None

    def test_has_required_methods(self, win_backend):
        mod, _, _ = win_backend
        backend = mod.DiffusersBackend()
        assert callable(getattr(backend, "load_model", None))
        assert callable(getattr(backend, "text_to_image", None))
        assert callable(getattr(backend, "image_to_image", None))


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


# ---------------------------------------------------------------------------
# load_model: CUDA unavailable → RuntimeError
# ---------------------------------------------------------------------------


class TestLoadModelCudaCheck:
    """load_model() must raise when CUDA is not available."""

    def test_raises_when_cuda_unavailable(self, win_backend_no_cuda):
        mod, _, _ = win_backend_no_cuda
        backend = mod.DiffusersBackend()

        with pytest.raises(RuntimeError, match="CUDA"):
            backend.load_model("fake-model-path")


# ---------------------------------------------------------------------------
# load_model: full-precision path (no quantize)
# ---------------------------------------------------------------------------


class TestLoadModelFullPrecision:
    """load_model without quantize uses group offloading path."""

    def test_full_precision_loads_pipeline(self, win_backend):
        mod, _, _ = win_backend

        fake_pipeline = MagicMock()
        fake_pipeline.enable_vae_slicing = MagicMock()
        fake_pipeline.enable_vae_tiling = MagicMock()
        mod.AutoPipelineForText2Image.from_pretrained.return_value = fake_pipeline

        with patch.object(mod, "detect_image_model", return_value=_make_model_info()):
            backend = mod.DiffusersBackend()
            pipeline, info = backend.load_model("fake-model-path")

        mod.AutoPipelineForText2Image.from_pretrained.assert_called_once()
        assert info.family == "zimage"
        assert pipeline is fake_pipeline
        fake_pipeline.enable_vae_slicing.assert_called_once()
        fake_pipeline.enable_vae_tiling.assert_called_once()

    def test_precision_mapping(self, win_backend):
        """torch_dtype is mapped from precision string."""
        mod, _, _ = win_backend

        fake_pipeline = MagicMock()
        mod.AutoPipelineForText2Image.from_pretrained.return_value = fake_pipeline

        with patch.object(mod, "detect_image_model", return_value=_make_model_info()):
            backend = mod.DiffusersBackend()
            backend.load_model("fake-model-path", precision="float16")

        call_kwargs = mod.AutoPipelineForText2Image.from_pretrained.call_args[1]
        assert call_kwargs["torch_dtype"] == "fp16-sentinel"

    def test_resets_img2img_pipe(self, win_backend):
        """load_model() clears any cached img2img pipeline."""
        mod, _, _ = win_backend

        fake_pipeline = MagicMock()
        mod.AutoPipelineForText2Image.from_pretrained.return_value = fake_pipeline

        with patch.object(mod, "detect_image_model", return_value=_make_model_info()):
            backend = mod.DiffusersBackend()
            backend._img2img_pipe = MagicMock()  # simulate previous load
            backend.load_model("fake-model-path")

        assert backend._img2img_pipe is None


# ---------------------------------------------------------------------------
# load_model: quantized paths (4-bit, 8-bit)
# ---------------------------------------------------------------------------


class TestLoadModelQuantized:
    """load_model with quantize triggers _load_quantized path."""

    def test_quantize_4bit_calls_load_quantized(self, win_backend):
        mod, _, _ = win_backend
        fake_pipeline = MagicMock()

        with (
            patch.object(mod, "detect_image_model", return_value=_make_model_info()),
            patch.object(mod, "_load_quantized", return_value=fake_pipeline) as mock_lq,
        ):
            backend = mod.DiffusersBackend()
            pipeline, info = backend.load_model("fake-model-path", quantize=4)

        mock_lq.assert_called_once()
        assert pipeline is fake_pipeline

    def test_quantize_8bit_calls_load_quantized(self, win_backend):
        mod, _, _ = win_backend
        fake_pipeline = MagicMock()

        with (
            patch.object(mod, "detect_image_model", return_value=_make_model_info()),
            patch.object(mod, "_load_quantized", return_value=fake_pipeline) as mock_lq,
        ):
            backend = mod.DiffusersBackend()
            backend.load_model("fake-model-path", quantize=8)

        mock_lq.assert_called_once()

    def test_unsupported_quantize_falls_back(self, win_backend, capsys):
        """Unsupported quantize value prints a message and falls back."""
        mod, _, _ = win_backend

        fake_pipeline = MagicMock()
        mod.AutoPipelineForText2Image.from_pretrained.return_value = fake_pipeline

        with patch.object(mod, "detect_image_model", return_value=_make_model_info()):
            backend = mod.DiffusersBackend()
            backend.load_model("fake-model-path", quantize=3)

        captured = capsys.readouterr()
        assert "Unsupported quantize" in captured.out


# ---------------------------------------------------------------------------
# load_model: LoRA loading
# ---------------------------------------------------------------------------


class TestLoadModelLoRA:
    """LoRA paths are forwarded to pipeline.load_lora_weights / set_adapters."""

    def test_lora_loaded_and_set(self, win_backend):
        mod, _, _ = win_backend

        fake_pipeline = MagicMock()
        mod.AutoPipelineForText2Image.from_pretrained.return_value = fake_pipeline

        with patch.object(mod, "detect_image_model", return_value=_make_model_info()):
            backend = mod.DiffusersBackend()
            backend.load_model(
                "fake-model-path",
                lora_paths=["/path/lora1.safetensors", "/path/lora2.safetensors"],
                lora_weights=[0.8, 0.5],
            )

        assert fake_pipeline.load_lora_weights.call_count == 2
        fake_pipeline.set_adapters.assert_called_once_with(
            ["lora_0", "lora_1"],
            adapter_weights=[0.8, 0.5],
        )


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

    def test_returns_image(self, win_backend):
        mod, _, _ = win_backend
        backend, mock_model = self._ready_backend(mod)
        result = backend.text_to_image(
            model=mock_model,
            prompt="a cat",
            width=64,
            height=64,
            seed=42,
            steps=4,
            guidance=0.5,
        )
        assert isinstance(result, Image.Image)
        mock_model.assert_called_once()

    def test_passes_guidance_scale(self, win_backend):
        mod, _, _ = win_backend
        backend, mock_model = self._ready_backend(mod)
        backend.text_to_image(
            model=mock_model,
            prompt="a cat",
            width=64,
            height=64,
            seed=42,
            steps=4,
            guidance=3.5,
        )
        call_kwargs = mock_model.call_args[1]
        assert call_kwargs["guidance_scale"] == 3.5

    def test_negative_prompt_non_flux(self, win_backend):
        """Non-flux models pass negative_prompt."""
        mod, _, _ = win_backend
        backend, mock_model = self._ready_backend(mod)
        backend._model_info = _make_model_info(family="unknown")
        backend.text_to_image(
            model=mock_model,
            prompt="a cat",
            width=64,
            height=64,
            seed=42,
            steps=4,
            guidance=0.5,
            negative_prompt="ugly",
        )
        call_kwargs = mock_model.call_args[1]
        assert call_kwargs["negative_prompt"] == "ugly"

    def test_negative_prompt_suppressed_for_flux(self, win_backend):
        """Flux models do NOT receive negative_prompt."""
        mod, _, _ = win_backend
        backend, mock_model = self._ready_backend(mod)
        backend._model_info = _make_model_info(family="flux1")
        backend.text_to_image(
            model=mock_model,
            prompt="a cat",
            width=64,
            height=64,
            seed=42,
            steps=4,
            guidance=0.5,
            negative_prompt="ugly",
        )
        call_kwargs = mock_model.call_args[1]
        assert "negative_prompt" not in call_kwargs

    def test_skip_signal_returns_none(self, win_backend):
        """When skip_signal fires, result is None."""
        mod, _, _ = win_backend
        backend, _ = self._ready_backend(mod)
        mock_model = MagicMock()
        mock_model._interrupt = True
        fake_result = MagicMock()
        fake_result.images = [Image.new("RGB", (64, 64))]
        mock_model.return_value = fake_result

        skip = MagicMock()
        skip.check.return_value = True

        result = backend.text_to_image(
            model=mock_model,
            prompt="a cat",
            width=64,
            height=64,
            seed=42,
            steps=4,
            guidance=0.5,
            skip_signal=skip,
        )
        assert result is None

    def test_restores_scheduler_after_beta(self, win_backend):
        """Scheduler is restored even when beta scheduler is requested."""
        mod, _, _ = win_backend
        backend, mock_model = self._ready_backend(mod)
        original_scheduler = mock_model.scheduler

        backend.text_to_image(
            model=mock_model,
            prompt="a cat",
            width=64,
            height=64,
            seed=42,
            steps=4,
            guidance=0.5,
            scheduler="beta",
        )
        assert mock_model.scheduler is original_scheduler

    def test_default_guidance_for_flux(self, win_backend):
        """Flux models get guidance_scale=1.0 when guidance is None."""
        mod, _, _ = win_backend
        backend, mock_model = self._ready_backend(mod)
        backend._model_info = _make_model_info(family="flux2")
        backend.text_to_image(
            model=mock_model,
            prompt="a cat",
            width=64,
            height=64,
            seed=42,
            steps=4,
            guidance=None,
        )
        call_kwargs = mock_model.call_args[1]
        assert call_kwargs["guidance_scale"] == 1.0


# ---------------------------------------------------------------------------
# image_to_image: basic refinement path
# ---------------------------------------------------------------------------


class TestImageToImage:
    """image_to_image creates an img2img pipeline and refines the input."""

    def test_returns_image(self, win_backend):
        mod, _, _ = win_backend
        backend = mod.DiffusersBackend()
        backend._model_info = _make_model_info()

        fake_i2i_pipe = MagicMock()
        fake_result = MagicMock()
        fake_result.images = [Image.new("RGB", (64, 64))]
        fake_i2i_pipe.return_value = fake_result
        fake_i2i_pipe._interrupt = False

        # The lazy import ``from diffusers import AutoPipelineForImage2Image``
        # resolves to sys.modules["diffusers"].AutoPipelineForImage2Image
        # which is already a MagicMock (our fake_modules).  Wire it up:
        sys.modules["diffusers"].AutoPipelineForImage2Image.from_pipe.return_value = fake_i2i_pipe

        result = backend.image_to_image(
            model=MagicMock(),
            image=Image.new("RGB", (64, 64)),
            prompt="sharper",
            strength=0.3,
            steps=4,
            seed=42,
            guidance=0.5,
        )
        assert isinstance(result, Image.Image)

    def test_reuses_img2img_pipe(self, win_backend):
        """Second call should reuse the cached _img2img_pipe."""
        mod, _, _ = win_backend
        backend = mod.DiffusersBackend()
        backend._model_info = _make_model_info()

        fake_i2i_pipe = MagicMock()
        fake_result = MagicMock()
        fake_result.images = [Image.new("RGB", (64, 64))]
        fake_i2i_pipe.return_value = fake_result
        fake_i2i_pipe._interrupt = False

        # Pre-set the pipe so it's already cached
        backend._img2img_pipe = fake_i2i_pipe

        result = backend.image_to_image(
            model=MagicMock(),
            image=Image.new("RGB", (64, 64)),
            prompt="sharper",
            strength=0.3,
            steps=4,
            seed=42,
            guidance=0.5,
        )
        assert isinstance(result, Image.Image)

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

        backend._img2img_pipe = fake_i2i_pipe

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
        backend._img2img_pipe = fake_i2i_pipe

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

    def test_callback_noop_when_not_skipped(self, win_backend):
        mod, _, _ = win_backend
        skip = MagicMock()
        skip.check.return_value = False
        cb = mod._make_skip_callback(skip)

        fake_pipe = MagicMock()
        result = cb(fake_pipe, step=0, timestep=0.0, callback_kwargs={"key": "val"})

        assert result == {"key": "val"}


# ---------------------------------------------------------------------------
# get_backend() returns diffusers on win32
# ---------------------------------------------------------------------------


class TestGetBackendWin32:
    """get_backend() returns the DiffusersBackend when platform is win32."""

    def test_get_backend_returns_diffusers_on_win32(self, win_backend):
        mod, _, _ = win_backend

        import zvisiongenerator.backends as backends_mod

        saved = dict(backends_mod.BACKENDS)
        backends_mod.BACKENDS.clear()
        try:
            with patch.object(backends_mod, "sys") as mock_sys:
                mock_sys.platform = "win32"
                backend = backends_mod.get_backend()
                assert backend.name == "diffusers"
        finally:
            backends_mod.BACKENDS.clear()
            backends_mod.BACKENDS.update(saved)


# ---------------------------------------------------------------------------
# _make_bnb_configs
# ---------------------------------------------------------------------------


class TestMakeBnbConfigs:
    """_make_bnb_configs returns two configs for 4-bit or 8-bit."""

    def test_4bit_configs(self, win_backend):
        mod, _, _ = win_backend
        te_cfg, tx_cfg = mod._make_bnb_configs(4, "bf16-sentinel")
        assert te_cfg is not None
        assert tx_cfg is not None

    def test_8bit_configs(self, win_backend):
        mod, _, _ = win_backend
        te_cfg, tx_cfg = mod._make_bnb_configs(8, "bf16-sentinel")
        assert te_cfg is not None
        assert tx_cfg is not None
