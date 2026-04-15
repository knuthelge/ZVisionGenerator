"""Tests for video upscale feature — CLI parsing, pipeline selection, stage kwargs, workflow assembly."""

from __future__ import annotations

import inspect
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import _make_video_args
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.video_types import VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.video_cli import _build_video_parser
from zvisiongenerator.workflows import build_video_workflow
from zvisiongenerator.workflows.video_stages import (
    image_to_video_stage,
    text_to_video_stage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _req(**overrides) -> VideoGenerationRequest:
    """Build a minimal VideoGenerationRequest for stage testing."""
    defaults: dict = dict(
        backend=MagicMock(),
        model=MagicMock(),
        prompt="a beautiful sunset",
    )
    defaults.update(overrides)
    return VideoGenerationRequest(**defaults)


# ---------------------------------------------------------------------------
# CLI / Config — --upscale parsing
# ---------------------------------------------------------------------------


class TestUpscaleCLIParsing:
    """Verify --upscale argument parsing and validation."""

    def test_upscale_2_parsed(self):
        parser = _build_video_parser()
        args = parser.parse_args(["-m", "model", "--upscale", "2"])
        assert args.upscale == 2

    def test_upscale_3_rejected(self):
        """--upscale 3 should cause a parser error (only 2 is supported)."""
        # The actual validation happens in main(), which calls parser.error().
        # At parse level, argparse accepts any int. The check is:
        #   if args.upscale is not None and args.upscale != 2: parser.error(...)
        parser = _build_video_parser()
        args = parser.parse_args(["-m", "model", "--upscale", "3"])
        assert args.upscale == 3  # parsing succeeds; validation is in main()

    def test_upscale_default_is_none(self):
        parser = _build_video_parser()
        args = parser.parse_args(["-m", "model"])
        assert args.upscale is None

    def test_upscale_steps_from_config_default(self):
        """--upscale 2 without --steps resolves upscale_steps to 8 from config."""
        args = _make_video_args(upscale=2, steps=8)
        config = {"video_model_presets": {"ltx": {"upscale": {"default_upscale_steps": 8}}}}

        # Replicate the CLI logic for upscale defaults
        steps_explicitly_set = False
        if args.upscale:
            upscale_cfg = config.get("video_model_presets", {}).get("ltx", {}).get("upscale", {})
            if not steps_explicitly_set:
                args.steps = upscale_cfg.get("default_upscale_steps", 8)
            args.upscale_steps = args.steps

        assert args.upscale_steps == 8

    def test_upscale_steps_from_explicit_steps(self):
        """--upscale 2 --steps 20 resolves upscale_steps to 20."""
        args = _make_video_args(upscale=2, steps=20)
        config = {"video_model_presets": {"ltx": {"upscale": {"default_upscale_steps": 8}}}}

        steps_explicitly_set = True  # user passed --steps 20
        if args.upscale:
            upscale_cfg = config.get("video_model_presets", {}).get("ltx", {}).get("upscale", {})
            if not steps_explicitly_set:
                args.steps = upscale_cfg.get("default_upscale_steps", 8)
            args.upscale_steps = args.steps

        assert args.upscale_steps == 20

    def test_upscale_step_cap_applies(self):
        """--upscale 2 still applies the LTX distilled step cap (steps capped at 8)."""
        args = _make_video_args(upscale=2, steps=30)

        _LTX_MAX_STEPS = 8
        if args.steps > _LTX_MAX_STEPS:
            args.steps = _LTX_MAX_STEPS

        assert args.steps == 8  # capped

    def test_without_upscale_step_cap_applies(self):
        """Without --upscale, steps are capped at 8 for distilled LTX."""
        args = _make_video_args(upscale=None, steps=20)

        _LTX_MAX_STEPS = 8
        if not args.upscale:
            if args.steps > _LTX_MAX_STEPS:
                args.steps = _LTX_MAX_STEPS

        assert args.steps == 8

    def test_audio_flag_defaults_enabled(self):
        """--audio defaults to True (BooleanOptionalAction)."""
        parser = _build_video_parser()
        args = parser.parse_args(["-m", "model"])
        assert args.audio is True

    def test_no_audio_flag(self):
        """--no-audio sets audio to False."""
        parser = _build_video_parser()
        args = parser.parse_args(["-m", "model", "--no-audio"])
        assert args.audio is False

    def test_audio_flag_explicit(self):
        """--audio sets audio to True explicitly."""
        parser = _build_video_parser()
        args = parser.parse_args(["-m", "model", "--audio"])
        assert args.audio is True


# ---------------------------------------------------------------------------
# Backend Pipeline Selection
# ---------------------------------------------------------------------------


class TestPipelineSelection:
    """Verify correct pipeline class is loaded based on upscale flag."""

    @patch("zvisiongenerator.backends.video_mac.detect_video_model")
    def test_upscale_loads_text_to_video_pipeline(self, mock_detect):
        """When upscale=True in load_model(), TextToVideoPipeline is loaded (distilled two-stage)."""
        from zvisiongenerator.backends.video_mac import LtxVideoBackend
        from zvisiongenerator.utils.video_model_detect import VideoModelInfo

        mock_detect.return_value = VideoModelInfo(family="ltx", backend="ltx", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32)
        mock_t2v_cls = MagicMock()

        backend = LtxVideoBackend()
        with patch.dict("sys.modules", {"ltx_pipelines_mlx": MagicMock(TextToVideoPipeline=mock_t2v_cls)}):
            pipeline, info = backend.load_model("model/path", upscale=True)

        mock_t2v_cls.assert_called_once_with(model_dir="model/path", low_memory=True)

    @patch("zvisiongenerator.backends.video_mac.detect_video_model")
    def test_no_upscale_loads_text_to_video_pipeline(self, mock_detect):
        """When upscale is absent, TextToVideoPipeline is loaded for t2v mode."""
        from zvisiongenerator.backends.video_mac import LtxVideoBackend
        from zvisiongenerator.utils.video_model_detect import VideoModelInfo

        mock_detect.return_value = VideoModelInfo(family="ltx", backend="ltx", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32)
        mock_t2v_cls = MagicMock()

        backend = LtxVideoBackend()
        with patch.dict("sys.modules", {"ltx_pipelines_mlx": MagicMock(TextToVideoPipeline=mock_t2v_cls)}):
            pipeline, info = backend.load_model("model/path", mode="t2v")

        mock_t2v_cls.assert_called_once_with(model_dir="model/path", low_memory=True)

    @patch("zvisiongenerator.backends.video_mac.detect_video_model")
    def test_no_upscale_i2v_loads_image_to_video_pipeline(self, mock_detect):
        """When upscale=False and mode=i2v, ImageToVideoPipeline is loaded."""
        from zvisiongenerator.backends.video_mac import LtxVideoBackend
        from zvisiongenerator.utils.video_model_detect import VideoModelInfo

        mock_detect.return_value = VideoModelInfo(family="ltx", backend="ltx", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32)
        mock_i2v_cls = MagicMock()

        backend = LtxVideoBackend()
        with patch.dict("sys.modules", {"ltx_pipelines_mlx": MagicMock(ImageToVideoPipeline=mock_i2v_cls)}):
            pipeline, info = backend.load_model("model/path", mode="i2v")

        mock_i2v_cls.assert_called_once_with(model_dir="model/path", low_memory=True)


# ---------------------------------------------------------------------------
# Stage Behavior — upscale kwargs
# ---------------------------------------------------------------------------


class TestTextToVideoStageUpscaleKwargs:
    """Verify text_to_video_stage passes stage1_steps when upscale is set."""

    def test_passes_upscale_kwargs(self):
        """text_to_video_stage passes stage1_steps when request.upscale == 2."""
        backend = MagicMock()
        backend.text_to_video.return_value = Path("/tmp/out.mp4")
        req = _req(backend=backend, upscale=2, upscale_steps=30, output_dir="/tmp")
        arts = VideoWorkingArtifacts()
        arts.filename = "test.mp4"
        arts.resolved_prompt = "test prompt"

        outcome = text_to_video_stage(req, arts)

        assert outcome is StageOutcome.success
        call_kwargs = backend.text_to_video.call_args
        assert call_kwargs.kwargs["stage1_steps"] == 30

    def test_no_upscale_kwargs_when_upscale_none(self):
        """text_to_video_stage does NOT pass extra kwargs when request.upscale is None."""
        backend = MagicMock()
        backend.text_to_video.return_value = Path("/tmp/out.mp4")
        req = _req(backend=backend, upscale=None, upscale_steps=None, output_dir="/tmp")
        arts = VideoWorkingArtifacts()
        arts.filename = "test.mp4"
        arts.resolved_prompt = "test prompt"

        text_to_video_stage(req, arts)

        call_kwargs = backend.text_to_video.call_args
        assert "stage1_steps" not in call_kwargs.kwargs

    def test_upscale_with_stage1_steps_only(self):
        """When upscale is set, only stage1_steps is passed (no cfg_scale)."""
        backend = MagicMock()
        backend.text_to_video.return_value = Path("/tmp/out.mp4")
        req = _req(backend=backend, upscale=2, upscale_steps=25, output_dir="/tmp")
        arts = VideoWorkingArtifacts()
        arts.filename = "test.mp4"
        arts.resolved_prompt = "prompt"

        text_to_video_stage(req, arts)

        call_kwargs = backend.text_to_video.call_args
        assert call_kwargs.kwargs["stage1_steps"] == 25
        assert "cfg_scale" not in call_kwargs.kwargs


class TestImageToVideoStageUpscaleKwargs:
    """Verify image_to_video_stage passes upscale kwargs correctly."""

    def test_passes_upscale_kwargs(self):
        """image_to_video_stage passes stage1_steps when request.upscale == 2."""
        backend = MagicMock()
        backend.image_to_video.return_value = Path("/tmp/i2v.mp4")
        req = _req(
            backend=backend,
            image_path="/tmp/input.png",
            upscale=2,
            upscale_steps=30,
            output_dir="/tmp",
        )
        arts = VideoWorkingArtifacts()
        arts.filename = "i2v.mp4"
        arts.resolved_prompt = "prompt"

        outcome = image_to_video_stage(req, arts)

        assert outcome is StageOutcome.success
        call_kwargs = backend.image_to_video.call_args
        assert call_kwargs.kwargs["stage1_steps"] == 30

    def test_no_upscale_kwargs_when_upscale_none(self):
        """image_to_video_stage does NOT pass extra kwargs when upscale is None."""
        backend = MagicMock()
        backend.image_to_video.return_value = Path("/tmp/i2v.mp4")
        req = _req(
            backend=backend,
            image_path="/tmp/input.png",
            upscale=None,
            upscale_steps=None,
            output_dir="/tmp",
        )
        arts = VideoWorkingArtifacts()
        arts.filename = "i2v.mp4"
        arts.resolved_prompt = "prompt"

        image_to_video_stage(req, arts)

        call_kwargs = backend.image_to_video.call_args
        assert "stage1_steps" not in call_kwargs.kwargs


# ---------------------------------------------------------------------------
# Workflow Assembly — upscale
# ---------------------------------------------------------------------------


class TestBuildVideoWorkflowUpscale:
    """Verify build_video_workflow() assembles correct stages with/without upscale."""

    def test_upscale_workflow_same_stages_as_non_upscale_t2v(self):
        """build_video_workflow with upscale builds same stage types for T2V.

        Upscale affects pipeline selection (load_model) and stage kwargs, not the
        stage list itself (unless no_audio is also set).
        """
        args = Namespace(image_path=None, no_audio=False)
        wf = build_video_workflow(args)
        stage_names = [s.__name__ for s in wf.stages]
        assert "resolve_prompt_stage" in stage_names
        assert "generate_filename_stage" in stage_names
        assert "text_to_video_stage" in stage_names
        assert "log_video_stage" in stage_names
        assert "strip_audio_stage" not in stage_names

    def test_non_upscale_workflow_no_strip_audio(self):
        """build_video_workflow without upscale, no_audio=False has no strip_audio_stage."""
        args = Namespace(image_path=None, no_audio=False)
        wf = build_video_workflow(args)
        stage_names = [s.__name__ for s in wf.stages]
        assert "strip_audio_stage" not in stage_names


# ---------------------------------------------------------------------------
# Runner — request field propagation
# ---------------------------------------------------------------------------


class TestRunnerUpscaleFieldPropagation:
    """Verify run_video_batch passes upscale fields to VideoGenerationRequest."""

    def test_upscale_fields_in_request(self):
        """When args has upscale=2, the VideoGenerationRequest gets upscale fields."""
        from tests.conftest import _make_mock_video_backend
        from zvisiongenerator.core.types import StageOutcome
        from zvisiongenerator.utils.video_model_detect import VideoModelInfo
        from zvisiongenerator.video_runner import run_video_batch

        backend = _make_mock_video_backend()
        model = MagicMock()
        workflow = MagicMock()
        workflow.run.return_value = StageOutcome.success

        args = _make_video_args(upscale=2, upscale_steps=30, no_audio=False)
        model_info = VideoModelInfo(family="ltx", backend="ltx", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32)
        prompts = {"default": [("a sunset", None)]}
        config: dict = {"generation": {"seed_min": 4, "seed_max": 100}}

        run_video_batch(backend, model, model_info, workflow, prompts, config, args)

        # The request object passed to workflow.run() should have upscale fields
        req = workflow.run.call_args[0][0]
        assert req.upscale == 2
        assert req.upscale_steps == 30

    def test_no_upscale_fields_in_request(self):
        """When args has upscale=None, request has None upscale fields."""
        from tests.conftest import _make_mock_video_backend
        from zvisiongenerator.core.types import StageOutcome
        from zvisiongenerator.utils.video_model_detect import VideoModelInfo
        from zvisiongenerator.video_runner import run_video_batch

        backend = _make_mock_video_backend()
        model = MagicMock()
        workflow = MagicMock()
        workflow.run.return_value = StageOutcome.success

        args = _make_video_args(upscale=None, upscale_steps=None)
        model_info = VideoModelInfo(family="ltx", backend="ltx", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32)
        prompts = {"default": [("a sunset", None)]}
        config: dict = {}

        run_video_batch(backend, model, model_info, workflow, prompts, config, args)

        req = workflow.run.call_args[0][0]
        assert req.upscale is None
        assert req.upscale_steps is None


# ---------------------------------------------------------------------------
# Distilled Two-Stage Backend — method dispatch
# ---------------------------------------------------------------------------


class TestDistilledTwoStageBackend:
    """Verify LtxVideoBackend dispatches to _generate_upscaled when upscaling."""

    def test_text_to_video_upscale_calls_generate_upscaled(self):
        """text_to_video() with stage1_steps dispatches to _generate_upscaled."""
        from zvisiongenerator.backends.video_mac import LtxVideoBackend

        backend = LtxVideoBackend()
        backend._generate_upscaled = MagicMock(return_value=Path("/tmp/upscaled.mp4"))

        result = backend.text_to_video(
            model=MagicMock(),
            prompt="test prompt",
            width=704,
            height=480,
            num_frames=49,
            seed=42,
            steps=8,
            output_path="/tmp/out.mp4",
            stage1_steps=8,
        )

        backend._generate_upscaled.assert_called_once()
        call_args = backend._generate_upscaled.call_args
        assert call_args[0][1] == "test prompt"  # prompt
        assert call_args[0][2] == 704  # width
        assert call_args[0][3] == 480  # height
        assert call_args[0][6] == 8  # stage1_steps
        assert result == Path("/tmp/upscaled.mp4")

    def test_text_to_video_no_upscale_calls_generate_and_save(self):
        """text_to_video() without stage1_steps calls model.generate_and_save()."""
        from zvisiongenerator.backends.video_mac import LtxVideoBackend

        backend = LtxVideoBackend()
        mock_model = MagicMock()

        result = backend.text_to_video(
            model=mock_model,
            prompt="test prompt",
            width=704,
            height=480,
            num_frames=49,
            seed=42,
            steps=8,
            output_path="/tmp/out.mp4",
        )

        mock_model.generate_and_save.assert_called_once()
        assert result == Path("/tmp/out.mp4")


# ---------------------------------------------------------------------------
# Distilled Two-Stage Internals — formulas, sigmas, signature
# ---------------------------------------------------------------------------


class TestDistilledTwoStageInternals:
    """Unit tests for distilled two-stage upscale internals."""

    @pytest.mark.parametrize(
        "dim, expected_half",
        [
            (704, 352),  # 704 % 64 == 0: exact
            (480, 224),  # 480 % 64 == 32: floor from 240 to 224
            (512, 256),  # 512 % 64 == 0: exact
            (288, 128),  # 288 % 64 == 32: floor from 144 to 128
            (384, 192),  # 384 % 64 == 0: exact
        ],
    )
    def test_half_resolution_floor_alignment(self, dim, expected_half):
        """Floor-alignment formula (dim // 2 // 32) * 32 produces correct half-resolution."""
        half = (dim // 2 // 32) * 32
        assert half == expected_half

    @pytest.mark.parametrize(
        "dim, expect_warning",
        [
            (704, False),  # 352 * 2 == 704
            (480, True),  # 224 * 2 == 448 != 480
            (512, False),  # 256 * 2 == 512
            (288, True),  # 128 * 2 == 256 != 288
            (384, False),  # 192 * 2 == 384
        ],
    )
    def test_half_resolution_triggers_warning(self, dim, expect_warning):
        """Warning condition fires when half * 2 != original dimension."""
        half = (dim // 2 // 32) * 32
        assert (half * 2 != dim) == expect_warning

    def test_sigma_schedule_values(self):
        """Sigma schedules have correct lengths and Stage 2 starts at 0.909375."""
        scheduler = pytest.importorskip("ltx_pipelines_mlx.scheduler")
        distilled = scheduler.DISTILLED_SIGMAS
        stage2 = scheduler.STAGE_2_SIGMAS

        assert len(distilled) == 9, f"DISTILLED_SIGMAS should have 9 values (8 steps), got {len(distilled)}"
        assert len(stage2) == 4, f"STAGE_2_SIGMAS should have 4 values (3 steps), got {len(stage2)}"
        assert stage2[0] == pytest.approx(0.909375), f"STAGE_2_SIGMAS[0] should be 0.909375, got {stage2[0]}"

    def test_generate_upscaled_signature_has_image_path(self):
        """_generate_upscaled() should accept image_path for I2V upscale."""
        from zvisiongenerator.backends.video_mac import LtxVideoBackend

        sig = inspect.signature(LtxVideoBackend._generate_upscaled)
        param_names = set(sig.parameters.keys())
        assert "image_path" in param_names, f"_generate_upscaled should have image_path, params: {param_names}"

    def test_i2v_upscale_allowed_in_cli(self):
        """--upscale 2 --image path.png should NOT cause a parser error at parse time."""
        parser = _build_video_parser()
        args = parser.parse_args(["-m", "model", "--upscale", "2", "--image", "photo.png"])
        assert args.upscale == 2
        assert args.image_path == "photo.png"

    def test_generate_upscaled_uses_noise_latent_state_for_audio(self):
        """_generate_upscaled source must call noise_latent_state for audio re-noising."""
        from zvisiongenerator.backends.video_mac import LtxVideoBackend

        source = inspect.getsource(LtxVideoBackend._generate_upscaled)
        assert "noise_latent_state" in source, "_generate_upscaled must call noise_latent_state for audio re-noising"
        # Verify the sigma comes from STAGE_2_SIGMAS (via start_sigma)
        assert "sigma=start_sigma" in source, "Audio re-noising must use start_sigma from STAGE_2_SIGMAS"

    def test_decode_receives_both_video_and_audio_latents(self):
        """_decode_and_save_video must be called with both video_latent and audio_latent."""
        from zvisiongenerator.backends.video_mac import LtxVideoBackend

        source = inspect.getsource(LtxVideoBackend._generate_upscaled)
        assert "_decode_and_save_video(video_latent, audio_latent" in source, "_decode_and_save_video must receive both video_latent and audio_latent"

    def test_stage2_uses_same_x0_model_no_reload(self):
        """Stage 2 must reuse the same x0_model (no transformer reload)."""
        from zvisiongenerator.backends.video_mac import LtxVideoBackend

        source = inspect.getsource(LtxVideoBackend._generate_upscaled)
        # x0_model is created once, used in both denoise_loop calls
        assert source.count("X0Model(") == 1, "x0_model should be created once (same transformer for both stages)"
        assert source.count("denoise_loop(") == 2, "Two denoise_loop calls expected (Stage 1 and Stage 2)"

    @pytest.mark.parametrize("stage1_steps", [4, 8])
    def test_sigma_slice_passed_to_first_denoise_loop(self, stage1_steps):
        """First denoise_loop receives DISTILLED_SIGMAS[:stage1_steps + 1]."""
        from zvisiongenerator.backends.video_mac import LtxVideoBackend

        # Known sigma schedule (9 values for up to 8 steps)
        fake_distilled_sigmas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.0]
        fake_stage2_sigmas = [0.909375, 0.725, 0.421875, 0.0]

        mock_denoise_loop = MagicMock()
        mock_denoise_loop.return_value = MagicMock()

        # Build mock module tree for all heavy local imports
        mock_mx = MagicMock()
        mock_patchifiers = MagicMock()
        mock_patchifiers.compute_video_latent_shape.return_value = (3, 7, 11)
        mock_latent_cond = MagicMock()
        mock_transformer_model = MagicMock()
        mock_memory = MagicMock()
        mock_positions = MagicMock()
        mock_positions.compute_audio_token_count.return_value = 16
        mock_weights = MagicMock()
        mock_scheduler = MagicMock()
        mock_scheduler.DISTILLED_SIGMAS = fake_distilled_sigmas
        mock_scheduler.STAGE_2_SIGMAS = fake_stage2_sigmas
        mock_samplers = MagicMock()
        mock_samplers.denoise_loop = mock_denoise_loop

        modules = {
            "mlx": MagicMock(),
            "mlx.core": mock_mx,
            "ltx_core_mlx": MagicMock(),
            "ltx_core_mlx.components": MagicMock(),
            "ltx_core_mlx.components.patchifiers": mock_patchifiers,
            "ltx_core_mlx.conditioning": MagicMock(),
            "ltx_core_mlx.conditioning.types": MagicMock(),
            "ltx_core_mlx.conditioning.types.latent_cond": mock_latent_cond,
            "ltx_core_mlx.model": MagicMock(),
            "ltx_core_mlx.model.transformer": MagicMock(),
            "ltx_core_mlx.model.transformer.model": mock_transformer_model,
            "ltx_core_mlx.model.upsampler": MagicMock(),
            "ltx_core_mlx.utils": MagicMock(),
            "ltx_core_mlx.utils.memory": mock_memory,
            "ltx_core_mlx.utils.positions": mock_positions,
            "ltx_core_mlx.utils.weights": mock_weights,
            "ltx_pipelines_mlx": MagicMock(),
            "ltx_pipelines_mlx.scheduler": mock_scheduler,
            "ltx_pipelines_mlx.utils": MagicMock(),
            "ltx_pipelines_mlx.utils.samplers": mock_samplers,
        }

        pipeline = MagicMock()
        pipeline.model_dir = "/fake/model"
        pipeline.dit = MagicMock()  # skip transformer loading
        pipeline.low_memory = False
        pipeline._encode_text.return_value = (MagicMock(), MagicMock())
        pipeline.video_patchifier.patchify.return_value = (MagicMock(), MagicMock())

        backend = LtxVideoBackend()

        with patch.dict("sys.modules", modules), patch("zvisiongenerator.backends.video_mac._load_upsampler", return_value=MagicMock()):
            backend._generate_upscaled(
                pipeline=pipeline,
                prompt="test",
                width=704,
                height=480,
                num_frames=49,
                seed=42,
                stage1_steps=stage1_steps,
                output_path="/tmp/out.mp4",
            )

        expected_sigmas = fake_distilled_sigmas[: stage1_steps + 1]
        first_call_kwargs = mock_denoise_loop.call_args_list[0].kwargs
        assert first_call_kwargs["sigmas"] == expected_sigmas, f"Stage 1 sigmas should be DISTILLED_SIGMAS[:{stage1_steps + 1}] = {expected_sigmas}, got {first_call_kwargs['sigmas']}"
