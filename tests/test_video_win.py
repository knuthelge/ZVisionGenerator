"""Mock-patched tests for zvisiongenerator.backends.video_win on macOS CI.

Exercises LtxCudaVideoBackend code paths without real torch/CUDA/ltx-pipelines
by mocking all heavy dependencies at the module level before import.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from zvisiongenerator.core.video_backend import VideoBackend
from zvisiongenerator.utils.video_model_detect import VideoModelInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_MODULES_KEYS = (
    "torch",
    "torch.cuda",
    "torch.backends",
    "torch.backends.cudnn",
    "ltx_core",
    "ltx_core.loader",
    "ltx_core.loader.sd_ops",
    "ltx_core.quantization",
    "ltx_core.model",
    "ltx_core.model.video_vae",
    "ltx_pipelines",
    "ltx_pipelines.distilled",
    "ltx_pipelines.distilled_single_stage",
    "ltx_pipelines.utils",
    "ltx_pipelines.utils.args",
    "ltx_pipelines.utils.constants",
    "ltx_pipelines.utils.media_io",
    "huggingface_hub",
)


def _stub_torch(*, cuda_available=True):
    """Build a fake torch module with enough surface to satisfy video_win."""
    torch_mod = MagicMock()
    torch_mod.cuda.is_available.return_value = cuda_available
    torch_mod.cuda.empty_cache = MagicMock()
    torch_mod.cuda.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
    torch_mod.backends.cudnn.benchmark = False
    torch_mod.device.return_value = MagicMock()
    torch_mod.inference_mode = lambda: lambda fn: fn  # no-op decorator
    return torch_mod


def _build_fake_modules(torch_mod=None):
    """Return dict of fake sys.modules entries for torch/ltx_core/ltx_pipelines."""
    t = torch_mod or _stub_torch()
    ltx_core = MagicMock()
    ltx_pipelines = MagicMock()
    return {
        "torch": t,
        "torch.cuda": t.cuda,
        "torch.backends": t.backends,
        "torch.backends.cudnn": t.backends.cudnn,
        "ltx_core": ltx_core,
        "ltx_core.loader": ltx_core.loader,
        "ltx_core.loader.sd_ops": ltx_core.loader.sd_ops,
        "ltx_core.quantization": ltx_core.quantization,
        "ltx_core.model": ltx_core.model,
        "ltx_core.model.video_vae": ltx_core.model.video_vae,
        "ltx_pipelines": ltx_pipelines,
        "ltx_pipelines.distilled": ltx_pipelines.distilled,
        "ltx_pipelines.distilled_single_stage": ltx_pipelines.distilled_single_stage,
        "ltx_pipelines.utils": ltx_pipelines.utils,
        "ltx_pipelines.utils.args": ltx_pipelines.utils.args,
        "ltx_pipelines.utils.constants": ltx_pipelines.utils.constants,
        "ltx_pipelines.utils.media_io": ltx_pipelines.utils.media_io,
        "huggingface_hub": MagicMock(),
    }


def _fresh_import(fake_modules):
    """Force a fresh import of video_win with *fake_modules* in sys.modules."""
    sys.modules.pop("zvisiongenerator.backends.video_win", None)
    import zvisiongenerator.backends.video_win as mod

    return mod


_DUMMY_MODEL_INFO = VideoModelInfo(
    family="ltx",
    backend="ltx",
    supports_i2v=True,
    default_fps=24,
    frame_alignment=8,
    resolution_alignment=32,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def win_video():
    """Yield (module, torch_stub, fakes) with fakes active for the whole test."""
    torch_mod = _stub_torch()
    fakes = _build_fake_modules(torch_mod)

    with patch.dict(sys.modules, fakes):
        mod = _fresh_import(fakes)
        yield mod, torch_mod, fakes

    sys.modules.pop("zvisiongenerator.backends.video_win", None)


@pytest.fixture()
def win_video_no_cuda():
    """Same as win_video but CUDA reports unavailable."""
    torch_mod = _stub_torch(cuda_available=False)
    fakes = _build_fake_modules(torch_mod)

    with patch.dict(sys.modules, fakes):
        mod = _fresh_import(fakes)
        yield mod, torch_mod, fakes

    sys.modules.pop("zvisiongenerator.backends.video_win", None)


# ---------------------------------------------------------------------------
# _resolve_model_paths() tests
# ---------------------------------------------------------------------------


class TestResolveModelPaths:
    """Verify _resolve_model_paths() discovers model components from a directory."""

    def test_resolve_model_paths_finds_all_components(self, tmp_path, win_video):
        mod, _, _fakes = win_video
        (tmp_path / "checkpoint.safetensors").touch()
        (tmp_path / "gemma-3").mkdir()
        (tmp_path / "spatial_upscaler.safetensors").touch()

        ckpt, gemma, upsampler = mod._resolve_model_paths(str(tmp_path))
        assert Path(ckpt).name == "checkpoint.safetensors"
        assert Path(gemma).name == "gemma-3"
        assert Path(upsampler).name == "spatial_upscaler.safetensors"

    def test_resolve_model_paths_prefers_distilled_checkpoint(self, tmp_path, win_video):
        mod, _, _fakes = win_video
        (tmp_path / "ltx-dev.safetensors").touch()
        (tmp_path / "ltx-distilled.safetensors").touch()
        (tmp_path / "gemma-2-2b-it").mkdir()
        (tmp_path / "spatial_upscaler_x2.safetensors").touch()

        ckpt, _, _ = mod._resolve_model_paths(str(tmp_path))
        assert "distilled" in Path(ckpt).name.lower()

    def test_resolve_model_paths_missing_checkpoint(self, tmp_path, win_video):
        mod, _, _fakes = win_video
        (tmp_path / "gemma-2-2b-it").mkdir()
        (tmp_path / "spatial_upscaler.safetensors").touch()

        with pytest.raises(FileNotFoundError, match="checkpoint"):
            mod._resolve_model_paths(str(tmp_path))

    def test_resolve_model_paths_missing_gemma(self, tmp_path, win_video):
        mod, _, _fakes = win_video
        (tmp_path / "model.safetensors").touch()
        (tmp_path / "spatial_upscaler.safetensors").touch()

        with pytest.raises(FileNotFoundError, match="gemma"):
            mod._resolve_model_paths(str(tmp_path))

    def test_resolve_model_paths_missing_upsampler_required(self, tmp_path, win_video):
        mod, _, _fakes = win_video
        (tmp_path / "model.safetensors").touch()
        (tmp_path / "gemma-2-2b-it").mkdir()

        with pytest.raises(FileNotFoundError, match="spatial upsampler"):
            mod._resolve_model_paths(str(tmp_path), require_upsampler=True)

    def test_resolve_model_paths_missing_upsampler_optional(self, tmp_path, win_video):
        mod, _, _fakes = win_video
        (tmp_path / "model.safetensors").touch()
        (tmp_path / "gemma-2-2b-it").mkdir()

        ckpt, gemma, upsampler = mod._resolve_model_paths(str(tmp_path), require_upsampler=False)
        assert Path(ckpt).name == "model.safetensors"
        assert Path(gemma).name == "gemma-2-2b-it"
        assert upsampler is None

    def test_resolve_model_paths_nonexistent_dir(self, tmp_path, win_video):
        mod, _, _fakes = win_video
        with pytest.raises(FileNotFoundError, match="Model directory"):
            mod._resolve_model_paths(str(tmp_path / "does-not-exist"))

    def test_resolve_model_paths_excludes_upscaler_from_checkpoint(self, tmp_path, win_video):
        mod, _, _fakes = win_video
        (tmp_path / "model.safetensors").touch()
        (tmp_path / "spatial_upscaler_x2.safetensors").touch()
        (tmp_path / "gemma-2-2b-it").mkdir()

        ckpt, _, upsampler = mod._resolve_model_paths(str(tmp_path))
        assert Path(ckpt).name == "model.safetensors"
        assert "spatial_upscal" in Path(upsampler).name


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    """LtxCudaVideoBackend satisfies the VideoBackend Protocol."""

    def test_protocol_compliance(self, win_video):
        mod, _, _fakes = win_video
        backend = mod.LtxCudaVideoBackend()
        assert isinstance(backend, VideoBackend)


# ---------------------------------------------------------------------------
# load_model() tests
# ---------------------------------------------------------------------------


class TestLoadModel:
    """Verify load_model() pipeline construction and guard checks."""

    def test_load_model_calls_single_stage_pipeline(self, tmp_path, win_video):
        mod, _, fakes = win_video
        # Set up mock directory — no upsampler needed for single-stage
        (tmp_path / "ltx-distilled.safetensors").touch()
        (tmp_path / "gemma-2-2b-it").mkdir()

        fake_pipeline = MagicMock()
        mock_cls = MagicMock(return_value=fake_pipeline)
        fakes["ltx_pipelines.distilled_single_stage"].DistilledSingleStagePipeline = mock_cls

        with patch.object(mod, "detect_video_model", return_value=_DUMMY_MODEL_INFO):
            backend = mod.LtxCudaVideoBackend()
            pipeline, info = backend.load_model(str(tmp_path))

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert "distilled" in call_kwargs["distilled_checkpoint_path"]
        assert "spatial_upsampler_path" not in call_kwargs
        assert pipeline is fake_pipeline
        assert info == _DUMMY_MODEL_INFO

    def test_load_model_upscale_calls_distilled_pipeline(self, tmp_path, win_video):
        mod, _, fakes = win_video
        (tmp_path / "ltx-distilled.safetensors").touch()
        (tmp_path / "gemma-2-2b-it").mkdir()
        (tmp_path / "spatial_upscaler.safetensors").touch()

        fake_pipeline = MagicMock()
        mock_cls = MagicMock(return_value=fake_pipeline)
        fakes["ltx_pipelines.distilled"].DistilledPipeline = mock_cls

        with patch.object(mod, "detect_video_model", return_value=_DUMMY_MODEL_INFO):
            backend = mod.LtxCudaVideoBackend()
            pipeline, info = backend.load_model(str(tmp_path), upscale=True)

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert "distilled" in call_kwargs["distilled_checkpoint_path"]
        assert call_kwargs["spatial_upsampler_path"] is not None
        assert pipeline is fake_pipeline
        assert info == _DUMMY_MODEL_INFO

    def test_load_model_raises_without_cuda(self, win_video_no_cuda):
        mod, _, _fakes = win_video_no_cuda
        backend = mod.LtxCudaVideoBackend()

        with pytest.raises(RuntimeError, match="CUDA"):
            backend.load_model("fake-model-path")

    def test_load_model_with_loras(self, tmp_path, win_video):
        mod, _, fakes = win_video
        (tmp_path / "model.safetensors").touch()
        (tmp_path / "gemma-2-2b-it").mkdir()

        fake_pipeline = MagicMock()
        mock_cls = MagicMock(return_value=fake_pipeline)
        fakes["ltx_pipelines.distilled_single_stage"].DistilledSingleStagePipeline = mock_cls

        with patch.object(mod, "detect_video_model", return_value=_DUMMY_MODEL_INFO):
            backend = mod.LtxCudaVideoBackend()
            backend.load_model(str(tmp_path), loras=[("/path/lora.safetensors", 0.8)])

        call_kwargs = mock_cls.call_args[1]
        loras = call_kwargs["loras"]
        assert len(loras) == 1
        fakes["ltx_core.loader"].LoraPathStrengthAndSDOps.assert_called_once()


# ---------------------------------------------------------------------------
# text_to_video() tests
# ---------------------------------------------------------------------------


class TestTextToVideo:
    """Verify text_to_video() delegates to the pipeline correctly."""

    @staticmethod
    def _ready_backend(mod, fakes):
        """Return a backend with _model_info set and pipeline mocked."""
        backend = mod.LtxCudaVideoBackend()
        backend._model_info = _DUMMY_MODEL_INFO

        fake_pipeline = MagicMock()
        fake_pipeline.return_value = (MagicMock(), None)  # (video_iter, audio)

        fakes["ltx_pipelines.utils.media_io"].encode_video = MagicMock()
        fakes["ltx_pipelines.utils.constants"].DISTILLED_SIGMAS = list(range(20))
        fakes["ltx_core.model.video_vae"].TilingConfig.default.return_value = MagicMock()
        fakes["ltx_core.model.video_vae"].get_video_chunks_number.return_value = 1

        return backend, fake_pipeline

    def test_text_to_video_calls_pipeline(self, win_video):
        mod, _, fakes = win_video
        backend, fake_pipeline = self._ready_backend(mod, fakes)

        backend.text_to_video(
            model=fake_pipeline,
            prompt="a sunset",
            width=704,
            height=448,
            num_frames=49,
            seed=42,
            steps=8,
            output_path="/tmp/out.mp4",
        )

        fake_pipeline.assert_called_once()
        call_kwargs = fake_pipeline.call_args[1]
        assert call_kwargs["prompt"] == "a sunset"
        assert call_kwargs["seed"] == 42
        assert call_kwargs["height"] == 448
        assert call_kwargs["width"] == 704
        assert call_kwargs["num_frames"] == 49
        assert call_kwargs["images"] == []
        # steps=8 should produce sigmas[:9]
        assert call_kwargs["sigmas"] == list(range(9))

    def test_text_to_video_stage1_steps_calls_upscaled(self, win_video):
        mod, _, fakes = win_video
        backend, fake_pipeline = self._ready_backend(mod, fakes)

        backend.text_to_video(
            model=fake_pipeline,
            prompt="a sunset",
            width=704,
            height=448,
            num_frames=49,
            seed=42,
            steps=8,
            output_path="/tmp/out.mp4",
            stage1_steps=4,
        )

        fake_pipeline.assert_called_once()
        call_kwargs = fake_pipeline.call_args[1]
        # stage1_steps=4 should produce stage_1_sigmas[:5]
        assert call_kwargs["stage_1_sigmas"] == list(range(5))
        assert "sigmas" not in call_kwargs

    def test_text_to_video_returns_path(self, win_video):
        mod, _, fakes = win_video
        backend, fake_pipeline = self._ready_backend(mod, fakes)

        result = backend.text_to_video(
            model=fake_pipeline,
            prompt="a sunset",
            width=704,
            height=448,
            num_frames=49,
            seed=42,
            steps=8,
            output_path="/tmp/out.mp4",
        )

        assert result == Path("/tmp/out.mp4")

    def test_text_to_video_oom_returns_none(self, win_video):
        mod, torch_mod, fakes = win_video
        backend, fake_pipeline = self._ready_backend(mod, fakes)

        # Raise OOM from pipeline call
        oom_error = torch_mod.cuda.OutOfMemoryError
        fake_pipeline.side_effect = oom_error("out of memory")

        result = backend.text_to_video(
            model=fake_pipeline,
            prompt="a sunset",
            width=704,
            height=448,
            num_frames=49,
            seed=42,
            steps=8,
            output_path="/tmp/out.mp4",
        )

        assert result is None


# ---------------------------------------------------------------------------
# image_to_video() tests
# ---------------------------------------------------------------------------


class TestImageToVideo:
    """Verify image_to_video() passes image conditioning."""

    def test_image_to_video_passes_image_conditioning(self, win_video):
        mod, _, fakes = win_video
        backend = mod.LtxCudaVideoBackend()
        backend._model_info = _DUMMY_MODEL_INFO

        fake_pipeline = MagicMock()
        fake_pipeline.return_value = (MagicMock(), None)

        fakes["ltx_pipelines.utils.media_io"].encode_video = MagicMock()
        fakes["ltx_pipelines.utils.constants"].DISTILLED_SIGMAS = list(range(20))
        fakes["ltx_core.model.video_vae"].TilingConfig.default.return_value = MagicMock()
        fakes["ltx_core.model.video_vae"].get_video_chunks_number.return_value = 1

        fake_ici = MagicMock()
        fakes["ltx_pipelines.utils.args"].ImageConditioningInput = MagicMock(return_value=fake_ici)

        backend.image_to_video(
            model=fake_pipeline,
            image_path="/tmp/input.png",
            prompt="animate this",
            width=704,
            height=448,
            num_frames=49,
            seed=42,
            steps=8,
            output_path="/tmp/out.mp4",
        )

        fakes["ltx_pipelines.utils.args"].ImageConditioningInput.assert_called_once_with(
            path="/tmp/input.png",
            frame_idx=0,
            strength=1.0,
        )
        call_kwargs = fake_pipeline.call_args[1]
        assert call_kwargs["images"] == [fake_ici]


# ---------------------------------------------------------------------------
# CUDA package isolation
# ---------------------------------------------------------------------------


class TestCudaPackageIsolation:
    """Verify CUDA-only packages aren't imported at module level."""

    def test_cuda_packages_not_imported(self):
        """After importing zvisiongenerator, ltx_core/ltx_pipelines must not be in sys.modules."""
        import zvisiongenerator  # noqa: F401

        assert "ltx_core" not in sys.modules, "ltx_core was imported at module level"
        assert "ltx_pipelines" not in sys.modules, "ltx_pipelines was imported at module level"
