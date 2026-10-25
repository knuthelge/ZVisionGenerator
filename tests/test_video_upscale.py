"""Tests for video upscale feature — CLI parsing, stage kwargs, workflow assembly, and the macOS LTX backend."""

from __future__ import annotations

from conftest import _make_plan, make_extend_source

import os
import sys
from argparse import Namespace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from tests.conftest import _make_video_args
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.backends.mac.video import (
    LtxVideoBackend,
    _drop_leading_frames,
    _generation_scope,
    _LOW_RAM_THRESHOLD_BYTES,
    _LtxModel,
    _resolve_model_dir,
    _TwoStageProgress,
    _use_low_ram_streaming,
)
from zvisiongenerator.core.video_types import ExtendSegment, VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.utils.config import load_config
from zvisiongenerator.utils.h3_preset import HubFiles
from zvisiongenerator.utils.video_model_detect import VideoModelInfo
from zvisiongenerator.video_cli import _build_video_parser, _resolve_upscale_steps
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
        """The parser preserves unsupported values so CLI validation can reject them."""
        parser = _build_video_parser()
        args = parser.parse_args(["-m", "model", "--upscale", "3"])
        assert args.upscale == 3

    def test_upscale_default_is_none(self):
        parser = _build_video_parser()
        args = parser.parse_args(["-m", "model"])
        assert args.upscale is None

    def test_upscale_steps_from_config_default(self):
        """--upscale 2 without --steps resolves upscale_steps to 8 from config."""
        args = _make_video_args(upscale=2, steps=50)
        config = load_config()

        _resolve_upscale_steps(args, config, "ltx", steps_explicitly_set=False)

        assert args.steps == 8
        assert args.upscale_steps == 8

    def test_upscale_steps_from_explicit_steps(self):
        """--upscale 2 --steps 20 resolves upscale_steps to 20."""
        args = _make_video_args(upscale=2, steps=20)
        config = load_config()

        _resolve_upscale_steps(args, config, "wan", steps_explicitly_set=True)

        assert args.upscale_steps == 20

    def test_upscale_step_cap_applies(self):
        """--upscale 2 still applies the LTX distilled step cap (steps capped at 8)."""
        args = _make_video_args(upscale=2, steps=30)

        _resolve_upscale_steps(args, load_config(), "ltx", steps_explicitly_set=True)

        assert args.steps == 8
        assert args.upscale_steps == 8

    def test_without_upscale_step_cap_applies(self):
        """Without --upscale, steps are capped at 8 for distilled LTX."""
        args = _make_video_args(upscale=None, steps=20)

        _resolve_upscale_steps(args, load_config(), "ltx", steps_explicitly_set=True)

        assert args.steps == 8
        assert args.upscale_steps is None

    def test_i2v_upscale_allowed_in_cli(self):
        """--upscale 2 --image path.png should NOT cause a parser error at parse time."""
        parser = _build_video_parser()
        args = parser.parse_args(["-m", "model", "--upscale", "2", "--image", "photo.png"])
        assert args.upscale == 2
        assert args.image_path == "photo.png"

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

        run_video_batch(backend, model, model_info, workflow, prompts, config, args, plan=_make_plan(prompts, args, config=config))

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

        run_video_batch(backend, model, model_info, workflow, prompts, config, args, plan=_make_plan(prompts, args, config=config))

        req = workflow.run.call_args[0][0]
        assert req.upscale is None
        assert req.upscale_steps is None


# ---------------------------------------------------------------------------
# macOS LTX backend — LTX modules are mocked, no model is ever loaded
# ---------------------------------------------------------------------------

_FPS = 24.0
_LATENT_SHAPE = (3, 7, 11)  # (F, H, W)
_AUDIO_TOKENS = 16
_FAKE_DISTILLED_SIGMAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.0]
_FAKE_LTX25_SIGMAS = [1.0, 0.95, 0.85, 0.75, 0.6, 0.45, 0.3, 0.1, 0.0]
_FAKE_STAGE_2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]


class _ImageConditioningInput(NamedTuple):
    """Stand-in with the fields of the upstream conditioning input."""

    path: str
    frame_idx: int = 0
    strength: float = 1.0
    crf: int | None = None


def _model_info(default_fps: int = 24) -> VideoModelInfo:
    return VideoModelInfo(family="ltx", backend="ltx", supports_i2v=True, default_fps=default_fps, frame_alignment=8, resolution_alignment=32)


@pytest.fixture
def ltx(monkeypatch):
    """Replace every LTX/MLX module the backend imports lazily with mocks sharing one call recorder.

    ``h.calls`` is the shared parent: ``h.calls.method_calls`` lists, in order, the module functions and
    pipeline methods the backend invoked.
    """
    h = SimpleNamespace(calls=MagicMock())

    h.mx = MagicMock()
    mlx = MagicMock()
    mlx.core = h.mx
    patchifiers = MagicMock()
    patchifiers.compute_video_latent_shape.return_value = _LATENT_SHAPE
    patchifiers.snap_output_dimensions.side_effect = lambda height, width, **_: (height, width)
    h.patchifiers = patchifiers
    positions = MagicMock()
    positions.compute_audio_token_count.return_value = _AUDIO_TOKENS
    positions.compute_video_positions.return_value = "video_positions"
    positions.compute_audio_positions.return_value = "audio_positions"
    h.positions = positions
    h.memory = MagicMock()
    h.x0_model = MagicMock()
    transformer = MagicMock(X0Model=h.x0_model)

    h.scheduler = MagicMock(DISTILLED_SIGMAS=_FAKE_DISTILLED_SIGMAS, STAGE_2_SIGMAS=_FAKE_STAGE_2_SIGMAS, LTX_2_5_DISTILLED_SIGMAS=_FAKE_LTX25_SIGMAS)
    h.scheduler.shorten_schedule.return_value = ["shortened"]
    h.orchestration = MagicMock()
    h.orchestration.combined_image_conditionings.side_effect = lambda *a, **k: [MagicMock(clean_latent="clean_a"), MagicMock(clean_latent="clean_b")]
    args = MagicMock(ImageConditioningInput=_ImageConditioningInput)
    h.video_state = MagicMock(name="video_state")
    h.audio_state = MagicMock(name="audio_state")
    helpers = MagicMock()
    helpers.create_noised_state.side_effect = [h.video_state, h.audio_state]
    h.output = MagicMock()
    samplers = MagicMock()
    samplers.denoise_loop.return_value = h.output
    samplers.euler_ancestral_denoising_loop.return_value = h.output
    h.diffusion_steps = MagicMock()
    h.distilled = MagicMock(ANCESTRAL_ETA=1.0, ANCESTRAL_S_NOISE=1.0, ANCESTRAL_NOISE_SEED_OFFSET=10000)
    h.generation = MagicMock(is_ltx25_pack=MagicMock(return_value=False))  # a bare MagicMock return value is truthy

    # Modules the extension path imports lazily.
    h.conditioning_types = MagicMock()
    h.conditioning_types.VideoConditionByLatentIndex.side_effect = lambda **fields: SimpleNamespace(**fields)
    h.audio_vae = MagicMock()
    h.audio_vae.encode_audio.return_value = "raw_audio_latent"
    h.audio_io = MagicMock()
    h.audio_io.load_audio.return_value = SimpleNamespace(waveform="tail_waveform", sample_rate=16000)
    h.extend_distilled = MagicMock()
    h.extend_distilled.carry_audio_tokens.side_effect = lambda fps, count: round(count / fps * 25)
    h.extend_distilled.fit_audio_latent.side_effect = lambda latent, tokens: np.zeros((1, 8, tokens, 16))
    h.retake = MagicMock()
    h.retake.encode_video_tensor.return_value = "video_latent"
    h.media_io = MagicMock()
    h.media_io.video_preprocess.return_value = "video_tensor"
    h.video_vae = MagicMock()
    h.video_vae._compute_decode_tiling.return_value = None
    h.streamed = []
    h.video_vae.stream_chunks_to_ffmpeg.side_effect = lambda chunks, proc: h.streamed.extend(chunks)

    h.calls.attach_mock(h.mx.eval, "mx_eval")
    h.calls.attach_mock(h.retake.encode_video_tensor, "encode_video_tensor")
    h.calls.attach_mock(h.audio_io.load_audio, "load_audio")
    h.calls.attach_mock(h.audio_vae.encode_audio, "encode_audio")
    h.calls.attach_mock(h.extend_distilled.fit_audio_latent, "fit_audio_latent")
    h.calls.attach_mock(h.memory.aggressive_cleanup, "aggressive_cleanup")
    h.calls.attach_mock(h.orchestration.combined_image_conditionings, "combined_image_conditionings")
    h.calls.attach_mock(h.scheduler.shorten_schedule, "shorten_schedule")
    h.calls.attach_mock(helpers.create_noised_state, "create_noised_state")
    h.calls.attach_mock(samplers.denoise_loop, "denoise_loop")
    h.calls.attach_mock(samplers.euler_ancestral_denoising_loop, "euler_ancestral_denoising_loop")
    h.helpers = helpers
    h.samplers = samplers

    modules = {
        "mlx": mlx,
        "mlx.core": h.mx,
        "ltx_core_mlx": MagicMock(),
        "ltx_core_mlx.components": MagicMock(),
        "ltx_core_mlx.components.diffusion_steps": h.diffusion_steps,
        "ltx_core_mlx.components.patchifiers": patchifiers,
        "ltx_core_mlx.conditioning": MagicMock(),
        "ltx_core_mlx.conditioning.types": h.conditioning_types,
        "ltx_core_mlx.model": MagicMock(),
        "ltx_core_mlx.model.audio_vae": h.audio_vae,
        "ltx_core_mlx.model.video_vae": MagicMock(),
        "ltx_core_mlx.model.video_vae.video_vae": h.video_vae,
        "ltx_core_mlx.model.transformer": MagicMock(),
        "ltx_core_mlx.model.transformer.model": transformer,
        "ltx_core_mlx.utils": MagicMock(),
        "ltx_core_mlx.utils.audio": h.audio_io,
        "ltx_core_mlx.utils.memory": h.memory,
        "ltx_core_mlx.utils.positions": positions,
        "ltx_pipelines_mlx": MagicMock(),
        "ltx_pipelines_mlx.distilled": h.distilled,
        "ltx_pipelines_mlx.extend_distilled": h.extend_distilled,
        "ltx_pipelines_mlx.retake": h.retake,
        "ltx_pipelines_mlx.scheduler": h.scheduler,
        "ltx_pipelines_mlx.utils": MagicMock(),
        "ltx_pipelines_mlx.utils._orchestration": h.orchestration,
        "ltx_pipelines_mlx.utils.args": args,
        "ltx_pipelines_mlx.utils.generation": h.generation,
        "ltx_pipelines_mlx.utils.helpers": helpers,
        "ltx_pipelines_mlx.utils.media_io": h.media_io,
        "ltx_pipelines_mlx.utils.samplers": samplers,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return h


def _pipeline(h, *, low_memory: bool = False) -> MagicMock:
    """Build a mock one-stage pipeline whose method calls are recorded on ``h.calls``."""
    pipe = MagicMock()
    pipe.low_memory = low_memory
    pipe.prompt_encoder.return_value = ("video_embeds", "audio_embeds")
    pipe.image_conditioner.resolve_crf.side_effect = lambda images: list(images)
    h.encoder = MagicMock(name="vae_encoder")

    def _run_conditioner(fn, free_after):
        result = fn(h.encoder)
        h.calls.conditioner_returned()
        return result

    pipe.image_conditioner.side_effect = _run_conditioner
    h.calls.attach_mock(pipe.prompt_encoder, "prompt_encoder")
    h.calls.attach_mock(pipe.load, "load")
    h.calls.attach_mock(pipe._pre_denoise_flush, "_pre_denoise_flush")
    h.calls.attach_mock(pipe._decode_and_save_video, "_decode_and_save_video")
    return pipe


def _call_names(h) -> list[str]:
    return [c[0] for c in h.calls.method_calls if "." not in c[0]]


def _handle(pipe, *, upscale: bool = False, frame_rate: float = _FPS, low_ram_streaming: bool = False, ltx25: bool = False) -> _LtxModel:
    return _LtxModel(pipeline=pipe, frame_rate=frame_rate, upscale=upscale, low_ram_streaming=low_ram_streaming, ltx25=ltx25)


def _run_one_stage(pipe, *, image_path: str | None = None, step_callback=None, steps: int = 8, ltx25: bool = False):
    backend = LtxVideoBackend()
    kwargs = dict(
        model=_handle(pipe, ltx25=ltx25),
        prompt="a prompt",
        width=704,
        height=480,
        num_frames=49,
        seed=42,
        steps=steps,
        output_path="/tmp/out.mp4",
        step_callback=step_callback,
    )
    if image_path is None:
        return backend.text_to_video(**kwargs)
    return backend.image_to_video(image_path=image_path, **kwargs)


class TestComputeDtypeScope:
    """A generation runs the transformer in float16 unless the user chose a compute dtype."""

    @pytest.fixture(autouse=True)
    def _no_inherited_setting(self, monkeypatch):
        monkeypatch.delenv("LTX2_COMPUTE_DTYPE", raising=False)

    def test_defaults_to_float16_inside_the_scope_and_clears_it_after(self):
        with _generation_scope(_handle(MagicMock())):
            assert os.environ["LTX2_COMPUTE_DTYPE"] == "float16"

        assert "LTX2_COMPUTE_DTYPE" not in os.environ

    @pytest.mark.parametrize("value", ["float32", "bfloat16"])
    def test_keeps_a_value_the_user_set(self, monkeypatch, value):
        monkeypatch.setenv("LTX2_COMPUTE_DTYPE", value)

        with _generation_scope(_handle(MagicMock())):
            assert os.environ["LTX2_COMPUTE_DTYPE"] == value

        assert os.environ["LTX2_COMPUTE_DTYPE"] == value

    def test_clears_the_default_even_when_the_generation_fails(self):
        with pytest.raises(RuntimeError), _generation_scope(_handle(MagicMock())):
            raise RuntimeError("boom")

        assert "LTX2_COMPUTE_DTYPE" not in os.environ


class TestLoadModel:
    """Verify load_model builds the pipeline that matches the upscale flag and carries the frame rate."""

    @pytest.fixture
    def classes(self, monkeypatch):
        base, distilled = MagicMock(name="BasePipeline"), MagicMock(name="DistilledPipeline")
        base.return_value = SimpleNamespace(kind="base", model_dir="model/path")
        distilled.return_value = SimpleNamespace(kind="distilled", model_dir="model/path")
        monkeypatch.setitem(sys.modules, "ltx_pipelines_mlx", MagicMock(BasePipeline=base, DistilledPipeline=distilled))
        # A bare MagicMock return value is truthy, so the pack check is stubbed to a plain LTX-2.3 answer.
        generation = MagicMock(is_ltx25_pack=MagicMock(return_value=False))
        monkeypatch.setitem(sys.modules, "ltx_pipelines_mlx.utils", MagicMock(generation=generation))
        monkeypatch.setitem(sys.modules, "ltx_pipelines_mlx.utils.generation", generation)
        # A large Mac: no streaming, and the real MLX cache limit is never touched.
        monkeypatch.setattr("zvisiongenerator.backends.mac.video.memory_size_bytes", lambda: 64 * 1024**3)
        return SimpleNamespace(base=base, distilled=distilled, is_ltx25_pack=generation.is_ltx25_pack)

    @pytest.mark.parametrize("mode", ["t2v", "i2v"])
    def test_without_upscale_builds_base_pipeline_for_either_mode(self, classes, mode):
        with patch("zvisiongenerator.backends.mac.video.detect_video_model", return_value=_model_info()):
            handle, info = LtxVideoBackend().load_model("model/path", mode=mode)

        classes.base.assert_called_once_with(model_dir="model/path", low_memory=True, low_ram_streaming=False)
        classes.distilled.assert_not_called()
        assert handle.pipeline.kind == "base"
        assert handle.upscale is False

    def test_upscale_builds_distilled_pipeline(self, classes):
        with patch("zvisiongenerator.backends.mac.video.detect_video_model", return_value=_model_info()):
            handle, _ = LtxVideoBackend().load_model("model/path", mode="i2v", upscale=True, low_memory=False)

        classes.distilled.assert_called_once_with(model_dir="model/path", low_memory=False, low_ram_streaming=False)
        classes.base.assert_not_called()
        assert handle.pipeline.kind == "distilled"
        assert handle.upscale is True

    @pytest.mark.parametrize("upscale", [False, True])
    def test_loras_are_set_before_load(self, classes, upscale):
        loras = [("lora.safetensors", 0.7)]
        with patch("zvisiongenerator.backends.mac.video.detect_video_model", return_value=_model_info()):
            handle, _ = LtxVideoBackend().load_model("model/path", upscale=upscale, loras=loras)

        assert handle.pipeline._pending_loras == loras

    def test_no_loras_leaves_pending_loras_unset(self, classes):
        with patch("zvisiongenerator.backends.mac.video.detect_video_model", return_value=_model_info()):
            handle, _ = LtxVideoBackend().load_model("model/path")

        assert not hasattr(handle.pipeline, "_pending_loras")

    @pytest.mark.parametrize("default_fps", [24, 30])
    def test_frame_rate_is_the_detected_default_fps(self, classes, default_fps):
        info = _model_info(default_fps)
        with patch("zvisiongenerator.backends.mac.video.detect_video_model", return_value=info):
            handle, returned_info = LtxVideoBackend().load_model("model/path")

        assert handle.frame_rate == float(default_fps)
        assert isinstance(handle.frame_rate, float)
        assert returned_info is info

    @pytest.mark.parametrize(
        ("upscale", "memory_bytes", "low_memory", "expected"),
        [
            (False, _LOW_RAM_THRESHOLD_BYTES, True, True),
            (True, _LOW_RAM_THRESHOLD_BYTES, True, True),
            (False, _LOW_RAM_THRESHOLD_BYTES + 1, True, False),
            (False, 32 * 1024**3, True, False),
            (True, 16 * 1024**3, False, False),
            (False, None, True, False),
        ],
    )
    def test_low_ram_streaming_reaches_the_constructor_and_the_handle(self, classes, monkeypatch, upscale, memory_bytes, low_memory, expected):
        monkeypatch.setattr("zvisiongenerator.backends.mac.video.memory_size_bytes", lambda: memory_bytes)
        monkeypatch.setattr("zvisiongenerator.backends.mac.video.mlx_cache_limit", MagicMock())
        with patch("zvisiongenerator.backends.mac.video.detect_video_model", return_value=_model_info()):
            handle, _ = LtxVideoBackend().load_model("model/path", upscale=upscale, low_memory=low_memory)

        ctor = classes.distilled if upscale else classes.base
        ctor.assert_called_once_with(model_dir="model/path", low_memory=low_memory, low_ram_streaming=expected)
        assert handle.low_ram_streaming is expected

    def test_streaming_construction_runs_inside_the_cache_limit_scope(self, classes, monkeypatch):
        events: list[str] = []
        scope = MagicMock()
        scope.return_value.__enter__.side_effect = lambda: events.append("enter")
        scope.return_value.__exit__.side_effect = lambda *a: events.append("exit")
        classes.base.side_effect = lambda **_: events.append("construct") or SimpleNamespace(model_dir="model/path")
        monkeypatch.setattr("zvisiongenerator.backends.mac.video.memory_size_bytes", lambda: _LOW_RAM_THRESHOLD_BYTES)
        monkeypatch.setattr("zvisiongenerator.backends.mac.video.mlx_cache_limit", scope)
        with patch("zvisiongenerator.backends.mac.video.detect_video_model", return_value=_model_info()):
            LtxVideoBackend().load_model("model/path")

        scope.assert_called_once_with(0)
        assert events == ["enter", "construct", "exit"]

    def test_non_streaming_construction_leaves_the_cache_limit_alone(self, classes, monkeypatch):
        scope = MagicMock()
        monkeypatch.setattr("zvisiongenerator.backends.mac.video.mlx_cache_limit", scope)
        with patch("zvisiongenerator.backends.mac.video.detect_video_model", return_value=_model_info()):
            LtxVideoBackend().load_model("model/path")

        scope.assert_not_called()

    def test_missing_package_raises_runtime_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ltx_pipelines_mlx", None)  # makes the import raise ImportError
        monkeypatch.setattr("zvisiongenerator.backends.mac.video.memory_size_bytes", lambda: 64 * 1024**3)  # MLX is absent off macOS

        with pytest.raises(RuntimeError, match="ltx-pipelines-mlx"):
            LtxVideoBackend().load_model("model/path")

    def test_unpinned_reference_reaches_the_pipeline_unchanged(self, classes, monkeypatch):
        find, download = MagicMock(), MagicMock()
        monkeypatch.setattr("zvisiongenerator.backends.mac.video.find_local_model_dir", find)
        monkeypatch.setattr("zvisiongenerator.backends.mac.video._snapshot_download", download)
        with patch("zvisiongenerator.backends.mac.video.detect_video_model", return_value=_model_info()):
            LtxVideoBackend().load_model("owner/repo")

        classes.base.assert_called_once_with(model_dir="owner/repo", low_memory=True, low_ram_streaming=False)
        find.assert_not_called()
        download.assert_not_called()

    @pytest.mark.parametrize("upscale", [False, True])
    def test_pinned_reference_builds_the_pipeline_from_the_cached_snapshot(self, classes, monkeypatch, tmp_path, upscale):
        find, download = MagicMock(return_value=tmp_path), MagicMock()
        monkeypatch.setattr("zvisiongenerator.backends.mac.video.find_local_model_dir", find)
        monkeypatch.setattr("zvisiongenerator.backends.mac.video._snapshot_download", download)
        with patch("zvisiongenerator.backends.mac.video.detect_video_model", return_value=_model_info()) as detect:
            LtxVideoBackend().load_model("owner/repo@abc123", upscale=upscale)

        ctor = classes.distilled if upscale else classes.base
        ctor.assert_called_once_with(model_dir=str(tmp_path), low_memory=True, low_ram_streaming=False)
        find.assert_called_once_with("owner/repo@abc123")
        download.assert_not_called()
        detect.assert_called_once_with("owner/repo@abc123")

    def test_pinned_reference_missing_from_the_cache_is_downloaded_at_its_revision(self, classes, monkeypatch, tmp_path):
        download = MagicMock(return_value=str(tmp_path))
        monkeypatch.setattr("zvisiongenerator.backends.mac.video.find_local_model_dir", MagicMock(return_value=None))
        monkeypatch.setattr("zvisiongenerator.backends.mac.video._snapshot_download", download)
        with patch("zvisiongenerator.backends.mac.video.detect_video_model", return_value=_model_info()):
            LtxVideoBackend().load_model("owner/repo@abc123")

        download.assert_called_once_with(repo_id="owner/repo", revision="abc123")
        classes.base.assert_called_once_with(model_dir=str(tmp_path), low_memory=True, low_ram_streaming=False)

    def test_local_folder_with_an_at_sign_is_not_treated_as_a_pin(self, classes, monkeypatch, tmp_path):
        folder = tmp_path / "owner" / "repo@abc123"
        folder.mkdir(parents=True)
        find, download = MagicMock(), MagicMock()
        monkeypatch.setattr("zvisiongenerator.backends.mac.video.find_local_model_dir", find)
        monkeypatch.setattr("zvisiongenerator.backends.mac.video._snapshot_download", download)
        monkeypatch.chdir(tmp_path)
        with patch("zvisiongenerator.backends.mac.video.detect_video_model", return_value=_model_info()):
            LtxVideoBackend().load_model("owner/repo@abc123")

        classes.base.assert_called_once_with(model_dir="owner/repo@abc123", low_memory=True, low_ram_streaming=False)
        find.assert_not_called()
        download.assert_not_called()

    def _pack_folder(self, root, *, bundled):
        folder = root / "pack"
        folder.mkdir()
        (folder / "connector.safetensors").write_bytes(b"")
        if bundled:
            (folder / "text_encoder.safetensors").write_bytes(b"")
            (folder / "text_encoder_config.json").write_text("{}")
        return folder

    @pytest.mark.parametrize("upscale", [False, True])
    def test_a_2_3_folder_gets_the_gemma_3_folder_as_its_text_encoder(self, classes, monkeypatch, tmp_path, upscale):
        pack = self._pack_folder(tmp_path, bundled=False)
        gemma = tmp_path / "gemma"
        gemma.mkdir()
        find = MagicMock(return_value=gemma)
        monkeypatch.setattr("zvisiongenerator.backends.mac.video.find_local_model_dir", find)
        with patch("zvisiongenerator.backends.mac.video.detect_video_model", return_value=_model_info()):
            LtxVideoBackend().load_model(str(pack), upscale=upscale, mlx_text_encoder="owner/gemma@" + "c" * 40)

        ctor = classes.distilled if upscale else classes.base
        ctor.assert_called_once_with(model_dir=str(pack), low_memory=True, low_ram_streaming=False, gemma_model_id=str(gemma))
        find.assert_called_once_with("owner/gemma@" + "c" * 40)

    def test_the_gemma_3_revision_is_downloaded_with_the_mlx_lm_patterns_when_it_is_not_cached(self, classes, monkeypatch, tmp_path):
        from zvisiongenerator.backends.mac.download_plan import MLX_LM_ALLOW_PATTERNS

        pack = self._pack_folder(tmp_path, bundled=False)
        download = MagicMock(return_value=str(tmp_path / "gemma"))
        monkeypatch.setattr("zvisiongenerator.backends.mac.video.find_local_model_dir", MagicMock(return_value=None))
        monkeypatch.setattr("zvisiongenerator.backends.mac.video._snapshot_download", download)
        with patch("zvisiongenerator.backends.mac.video.detect_video_model", return_value=_model_info()):
            LtxVideoBackend().load_model(str(pack), mlx_text_encoder="owner/gemma@" + "c" * 40)

        download.assert_called_once_with(repo_id="owner/gemma", revision="c" * 40, allow_patterns=list(MLX_LM_ALLOW_PATTERNS))
        assert classes.base.call_args.kwargs["gemma_model_id"] == str(tmp_path / "gemma")

    def test_an_unpinned_reference_to_a_pack_repo_gets_the_pinned_pack_folder_and_the_pinned_gemma_3(self, classes, monkeypatch, tmp_path):
        pin = "d" * 40
        pack = HubFiles("owner/pack", pin, ("connector.safetensors",))
        folder = self._pack_folder(tmp_path, bundled=False)
        gemma = tmp_path / "gemma"
        gemma.mkdir()
        monkeypatch.setattr("zvisiongenerator.utils.hf_snapshot.cached_snapshot_dirs", lambda repo, revision: [folder])
        monkeypatch.setattr("zvisiongenerator.backends.mac.video.find_local_model_dir", MagicMock(return_value=gemma))
        with patch("zvisiongenerator.backends.mac.video.detect_video_model", return_value=_model_info()):
            LtxVideoBackend().load_model("Owner/Pack", mlx_packs={f"owner/pack@{pin}": pack}, mlx_text_encoder="owner/gemma@" + "c" * 40)

        classes.base.assert_called_once_with(model_dir=str(folder), low_memory=True, low_ram_streaming=False, gemma_model_id=str(gemma))

    def test_a_2_5_folder_with_its_own_text_encoder_gets_no_gemma_3(self, classes, monkeypatch, tmp_path):
        pack = self._pack_folder(tmp_path, bundled=True)
        find, download = MagicMock(), MagicMock()
        monkeypatch.setattr("zvisiongenerator.backends.mac.video.find_local_model_dir", find)
        monkeypatch.setattr("zvisiongenerator.backends.mac.video._snapshot_download", download)
        classes.base.return_value.model_dir = pack
        classes.is_ltx25_pack.return_value = True
        with patch("zvisiongenerator.backends.mac.video.detect_video_model", return_value=_model_info()):
            LtxVideoBackend().load_model(str(pack), mlx_text_encoder="owner/gemma@" + "c" * 40)

        classes.base.assert_called_once_with(model_dir=str(pack), low_memory=True, low_ram_streaming=False)
        find.assert_not_called()
        download.assert_not_called()

    def test_without_a_configured_text_encoder_the_pipeline_default_applies(self, classes, tmp_path):
        pack = self._pack_folder(tmp_path, bundled=False)
        with patch("zvisiongenerator.backends.mac.video.detect_video_model", return_value=_model_info()):
            LtxVideoBackend().load_model(str(pack))

        classes.base.assert_called_once_with(model_dir=str(pack), low_memory=True, low_ram_streaming=False)

    def test_a_pinned_pack_is_resolved_to_its_listed_files_and_gets_the_gemma_3_folder(self, classes, monkeypatch, tmp_path):
        pack_files = HubFiles("owner/pack", "d" * 40, ("connector.safetensors", "transformer-distilled-1.1.safetensors"))
        snapshot = self._pack_folder(tmp_path, bundled=False)
        (snapshot / "transformer-distilled-1.1.safetensors").write_bytes(b"")
        gemma = tmp_path / "gemma"
        gemma.mkdir()
        monkeypatch.setattr("zvisiongenerator.utils.hf_snapshot.cached_snapshot_dirs", lambda repo, revision: [snapshot])
        monkeypatch.setattr("zvisiongenerator.backends.mac.video.find_local_model_dir", MagicMock(return_value=gemma))
        download = MagicMock()
        monkeypatch.setattr("zvisiongenerator.backends.mac.video._snapshot_download", download)
        reference = "owner/pack@" + "d" * 40
        with patch("zvisiongenerator.backends.mac.video.detect_video_model", return_value=_model_info()):
            LtxVideoBackend().load_model(reference, mlx_packs={reference: pack_files}, mlx_text_encoder="owner/gemma@" + "c" * 40)

        classes.base.assert_called_once_with(model_dir=str(snapshot), low_memory=True, low_ram_streaming=False, gemma_model_id=str(gemma))
        download.assert_not_called()

    def test_a_2_3_pack_is_recorded_as_not_ltx25(self, classes):
        with patch("zvisiongenerator.backends.mac.video.detect_video_model", return_value=_model_info()):
            handle, _ = LtxVideoBackend().load_model("model/path")

        classes.is_ltx25_pack.assert_called_once_with("model/path")
        assert handle.ltx25 is False

    @pytest.mark.parametrize("upscale", [False, True])
    def test_a_2_5_pack_with_its_encoder_at_the_root_is_recorded_as_ltx25(self, classes, tmp_path, upscale):
        (tmp_path / "text_encoder.safetensors").write_bytes(b"")
        (tmp_path / "text_encoder_config.json").write_text("{}")
        classes.base.return_value.model_dir = classes.distilled.return_value.model_dir = tmp_path
        classes.is_ltx25_pack.return_value = True
        with patch("zvisiongenerator.backends.mac.video.detect_video_model", return_value=_model_info()):
            handle, _ = LtxVideoBackend().load_model("model/path", upscale=upscale)

        classes.is_ltx25_pack.assert_called_once_with(tmp_path)
        assert handle.ltx25 is True

    def test_a_2_5_pack_without_its_encoder_at_the_root_is_refused(self, classes, tmp_path):
        (tmp_path / "text_encoder_config.json").write_text("{}")  # the weights are missing
        classes.base.return_value.model_dir = tmp_path
        classes.is_ltx25_pack.return_value = True
        with patch("zvisiongenerator.backends.mac.video.detect_video_model", return_value=_model_info()):
            with pytest.raises(ValueError, match="text_encoder.safetensors"):
                LtxVideoBackend().load_model("model/path")


class TestLowRamStreamingGate:
    """Verify the pure memory-size gate for transformer streaming."""

    @pytest.mark.parametrize(
        ("low_memory", "memory_bytes", "expected"),
        [
            (True, 24 * 1024**3, True),
            (True, 24 * 1024**3 + 1, False),
            (True, 32 * 1024**3, False),
            (True, 16 * 1024**3, True),
            (False, 16 * 1024**3, False),
            (True, None, False),
        ],
    )
    def test_gate_table(self, low_memory, memory_bytes, expected):
        assert _use_low_ram_streaming(low_memory, memory_bytes) is expected

    def test_explicit_threshold_overrides_the_default(self):
        assert _use_low_ram_streaming(True, 24 * 1024**3, threshold_bytes=16 * 1024**3) is False
        assert _use_low_ram_streaming(True, 32 * 1024**3, threshold_bytes=32 * 1024**3) is True


class TestGenerationCacheLimit:
    """Verify generation scopes the MLX cache limit only when the handle streams."""

    @pytest.fixture
    def set_limit(self, ltx):
        ltx.mx.set_cache_limit.return_value = "previous"
        return ltx.mx.set_cache_limit

    @pytest.mark.parametrize("upscale", [False, True])
    def test_streaming_generation_sets_zero_then_restores(self, ltx, set_limit, upscale):
        pipe = _pipeline(ltx) if not upscale else MagicMock()
        stage1 = 8 if upscale else None

        LtxVideoBackend().text_to_video(
            model=_handle(pipe, upscale=upscale, low_ram_streaming=True), prompt="p", width=704, height=480, num_frames=49, seed=1, steps=8, output_path="/tmp/o.mp4", stage1_steps=stage1
        )

        assert set_limit.call_args_list == [call(0), call("previous")]

    @pytest.mark.parametrize("upscale", [False, True])
    def test_non_streaming_generation_does_not_touch_the_cache_limit(self, ltx, set_limit, upscale):
        pipe = _pipeline(ltx) if not upscale else MagicMock()
        stage1 = 8 if upscale else None

        LtxVideoBackend().text_to_video(model=_handle(pipe, upscale=upscale), prompt="p", width=704, height=480, num_frames=49, seed=1, steps=8, output_path="/tmp/o.mp4", stage1_steps=stage1)

        set_limit.assert_not_called()

    def test_streaming_generation_restores_when_the_pipeline_raises(self, ltx, set_limit):
        pipe = MagicMock()
        pipe.generate_and_save.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            LtxVideoBackend().text_to_video(
                model=_handle(pipe, upscale=True, low_ram_streaming=True), prompt="p", width=704, height=480, num_frames=49, seed=1, steps=8, output_path="/tmp/o.mp4", stage1_steps=8
            )

        assert set_limit.call_args_list == [call(0), call("previous")]


class TestOneStageGeneration:
    """Verify the BasePipeline composition used when upscale is off."""

    def test_call_order(self, ltx):
        pipe = _pipeline(ltx)

        _run_one_stage(pipe)

        assert _call_names(ltx) == [
            "prompt_encoder",
            "load",
            "create_noised_state",
            "create_noised_state",
            "_pre_denoise_flush",
            "shorten_schedule",
            "denoise_loop",
            "_decode_and_save_video",
        ]

    def test_low_memory_cleanup_follows_flush_and_encoder_is_freed(self, ltx):
        pipe = _pipeline(ltx, low_memory=True)

        _run_one_stage(pipe)

        names = _call_names(ltx)
        assert names[names.index("_pre_denoise_flush") + 1] == "aggressive_cleanup"
        assert pipe.prompt_encoder.call_args == call("a prompt", free_after=True)

    def test_no_cleanup_without_low_memory(self, ltx):
        pipe = _pipeline(ltx, low_memory=False)

        _run_one_stage(pipe)

        assert "aggressive_cleanup" not in _call_names(ltx)
        assert pipe.prompt_encoder.call_args == call("a prompt", free_after=False)

    def test_flush_receives_the_noised_states(self, ltx):
        pipe = _pipeline(ltx)

        _run_one_stage(pipe)

        pipe._pre_denoise_flush.assert_called_once_with(ltx.video_state, ltx.audio_state)
        loop_kwargs = ltx.samplers.denoise_loop.call_args.kwargs
        assert loop_kwargs["video_state"] is ltx.video_state
        assert loop_kwargs["audio_state"] is ltx.audio_state
        assert loop_kwargs["video_text_embeds"] == "video_embeds"
        assert loop_kwargs["audio_text_embeds"] == "audio_embeds"

    def test_sigmas_come_from_shorten_schedule(self, ltx):
        pipe = _pipeline(ltx)

        _run_one_stage(pipe, steps=6)

        ltx.scheduler.shorten_schedule.assert_called_once_with(_FAKE_DISTILLED_SIGMAS, 6, keep="start")
        assert ltx.samplers.denoise_loop.call_args.kwargs["sigmas"] is ltx.scheduler.shorten_schedule.return_value

    def test_a_2_3_handle_never_calls_the_ancestral_loop(self, ltx):
        _run_one_stage(_pipeline(ltx), image_path="/tmp/first.png")

        ltx.samplers.euler_ancestral_denoising_loop.assert_not_called()
        ltx.diffusion_steps.EulerAncestralDiffusionStep.assert_not_called()

    @pytest.mark.parametrize("image_path", [None, "/tmp/first.png"])
    def test_a_2_5_clip_runs_the_ancestral_loop_on_the_2_5_schedule(self, ltx, image_path):
        pipe = _pipeline(ltx)
        pipe.dit = object()
        callback = MagicMock()

        _run_one_stage(pipe, image_path=image_path, steps=6, ltx25=True, step_callback=callback)

        ltx.samplers.denoise_loop.assert_not_called()
        ltx.scheduler.shorten_schedule.assert_called_once_with(_FAKE_LTX25_SIGMAS, 6, keep="start")
        ltx.samplers.euler_ancestral_denoising_loop.assert_called_once()
        kwargs = ltx.samplers.euler_ancestral_denoising_loop.call_args.kwargs
        assert kwargs["sigmas"] is ltx.scheduler.shorten_schedule.return_value
        assert kwargs["noise_seed"] == 42 + 10000
        assert kwargs["stepper"] is ltx.diffusion_steps.EulerAncestralDiffusionStep.return_value
        ltx.diffusion_steps.EulerAncestralDiffusionStep.assert_called_once_with(eta=1.0, s_noise=1.0)
        assert kwargs["transformer"] is ltx.x0_model.return_value
        ltx.x0_model.assert_called_once_with(pipe.dit)
        assert kwargs["video_state"] is ltx.video_state
        assert kwargs["audio_state"] is ltx.audio_state
        assert kwargs["video_text_embeds"] == "video_embeds"
        assert kwargs["audio_text_embeds"] == "audio_embeds"
        assert kwargs["on_step"] is not None

    def test_a_2_5_clip_keeps_the_initial_noise_on_the_plain_seeds(self, ltx):
        _run_one_stage(_pipeline(ltx), ltx25=True)

        seeds = [c.kwargs["seed"] for c in ltx.helpers.create_noised_state.call_args_list]
        assert seeds == [42, 43]

    def test_decode_uses_frame_rate_and_seed(self, ltx):
        pipe = _pipeline(ltx)

        result = _run_one_stage(pipe)

        pipe._decode_and_save_video.assert_called_once_with(
            pipe.video_patchifier.unpatchify.return_value,
            pipe.audio_patchifier.unpatchify.return_value,
            "/tmp/out.mp4",
            frame_rate=_FPS,
            seed=42,
        )
        assert result == Path("/tmp/out.mp4")

    def test_noised_states_use_frame_rate_positions_and_offset_audio_seed(self, ltx):
        pipe = _pipeline(ltx)

        _run_one_stage(pipe)

        ltx.positions.compute_video_positions.assert_called_once_with(*_LATENT_SHAPE, frame_rate=_FPS)
        ltx.positions.compute_audio_token_count.assert_called_once_with(49, frame_rate=_FPS)
        video_call, audio_call = ltx.helpers.create_noised_state.call_args_list
        F, H, W = _LATENT_SHAPE
        assert video_call.kwargs["base_shape"] == (1, F * H * W, 128)
        assert video_call.kwargs["positions"] == "video_positions"
        assert video_call.kwargs["seed"] == 42
        assert audio_call.kwargs["base_shape"] == (1, _AUDIO_TOKENS, 128)
        assert audio_call.kwargs["positions"] == "audio_positions"
        assert audio_call.kwargs["seed"] == 43
        assert audio_call.kwargs["conditionings"] == []

    def test_noised_states_start_from_pure_noise_with_legacy_blend(self, ltx):
        pipe = _pipeline(ltx)

        _run_one_stage(pipe)

        for state_call in ltx.helpers.create_noised_state.call_args_list:
            assert state_call.kwargs["legacy_scalar_blend"] is True
            assert state_call.kwargs["sigma"] == 1.0
            assert state_call.kwargs["initial_latent"] is None

    def test_denoise_loop_runs_the_pipeline_transformer(self, ltx):
        pipe = _pipeline(ltx)

        _run_one_stage(pipe)

        ltx.x0_model.assert_called_once_with(pipe.dit)
        assert ltx.samplers.denoise_loop.call_args.kwargs["model"] is ltx.x0_model.return_value

    def test_unpatchify_receives_only_the_generated_tokens(self, ltx):
        pipe = _pipeline(ltx)

        _run_one_stage(pipe)

        F, H, W = _LATENT_SHAPE
        sliced = ltx.output.video_latent.__getitem__
        sliced.assert_called_once_with((slice(None), slice(None, F * H * W), slice(None)))
        pipe.video_patchifier.unpatchify.assert_called_once_with(sliced.return_value, _LATENT_SHAPE)
        pipe.audio_patchifier.unpatchify.assert_called_once_with(ltx.output.audio_latent)

    def test_low_memory_frees_the_decoders_after_decoding(self, ltx):
        pipe = _pipeline(ltx, low_memory=True)

        _run_one_stage(pipe)

        assert pipe.vae_decoder is None
        assert pipe.audio_decoder is None
        assert pipe.vocoder is None

    def test_decoders_stay_loaded_without_low_memory(self, ltx):
        pipe = _pipeline(ltx, low_memory=False)

        _run_one_stage(pipe)

        assert pipe.vae_decoder is not None
        assert pipe.audio_decoder is not None
        assert pipe.vocoder is not None

    def test_t2v_has_no_conditionings(self, ltx):
        pipe = _pipeline(ltx)

        _run_one_stage(pipe)

        pipe.image_conditioner.assert_not_called()
        assert ltx.helpers.create_noised_state.call_args_list[0].kwargs["conditionings"] == []

    def test_i2v_conditions_the_image_at_frame_zero(self, ltx):
        pipe = _pipeline(ltx)
        input_image = _ImageConditioningInput(path="/tmp/in.png", frame_idx=0, strength=1.0)

        _run_one_stage(pipe, image_path="/tmp/in.png")

        pipe.image_conditioner.resolve_crf.assert_called_once_with([input_image])
        F, H, W = _LATENT_SHAPE
        ltx.orchestration.combined_image_conditionings.assert_called_once_with(
            [input_image],
            enc_h=H * 32,
            enc_w=W * 32,
            spatial_dims=_LATENT_SHAPE,
            video_encoder=ltx.encoder,
            frame_rate=_FPS,
        )
        assert pipe.image_conditioner.call_args.kwargs == {"free_after": False}

    def test_i2v_materializes_clean_latents_before_the_encoder_is_freed(self, ltx):
        pipe = _pipeline(ltx)

        _run_one_stage(pipe, image_path="/tmp/in.png")

        names = [c[0] for c in ltx.calls.method_calls if c[0] in {"combined_image_conditionings", "mx_eval", "conditioner_returned"}]
        assert names == ["combined_image_conditionings", "mx_eval", "conditioner_returned"]
        assert ltx.mx.eval.call_args.args == ("clean_a", "clean_b")

    def test_i2v_conditionings_reach_the_video_state(self, ltx):
        pipe = _pipeline(ltx)
        produced = [MagicMock(clean_latent="x")]
        ltx.orchestration.combined_image_conditionings.side_effect = lambda *a, **k: produced

        _run_one_stage(pipe, image_path="/tmp/in.png")

        assert ltx.helpers.create_noised_state.call_args_list[0].kwargs["conditionings"] is produced

    def test_i2v_frees_the_encoder_under_low_memory(self, ltx):
        pipe = _pipeline(ltx, low_memory=True)

        _run_one_stage(pipe, image_path="/tmp/in.png")

        assert pipe.image_conditioner.call_args.kwargs == {"free_after": True}

    def test_progress_is_one_based_video_generate(self, ltx):
        pipe = _pipeline(ltx)
        events: list[dict] = []

        _run_one_stage(pipe, step_callback=events.append)

        on_step = ltx.samplers.denoise_loop.call_args.kwargs["on_step"]
        on_step(0, 8, MagicMock(), 1.0)
        on_step(7, 8, MagicMock(), 0.1)
        assert events == [
            {"phase": "video_generate", "current_step": 1, "total_steps": 8},
            {"phase": "video_generate", "current_step": 8, "total_steps": 8},
        ]

    def test_no_callback_passes_no_on_step(self, ltx):
        pipe = _pipeline(ltx)

        _run_one_stage(pipe)

        assert ltx.samplers.denoise_loop.call_args.kwargs["on_step"] is None

    def test_stage1_steps_without_upscale_model_raises(self, ltx):
        pipe = _pipeline(ltx)

        with pytest.raises(ValueError, match="stage1_steps"):
            LtxVideoBackend().text_to_video(model=_handle(pipe, upscale=False), prompt="p", width=704, height=480, num_frames=49, seed=1, steps=8, output_path="/tmp/o.mp4", stage1_steps=8)

        pipe.load.assert_not_called()


class TestTwoStageGeneration:
    """Verify upscale runs through DistilledPipeline.generate_and_save with a progress adapter."""

    @pytest.mark.parametrize("image_path", [None, "/tmp/in.png"])
    def test_generate_and_save_receives_exactly_the_supported_kwargs(self, ltx, image_path):
        pipe = MagicMock()
        backend = LtxVideoBackend()
        kwargs = dict(model=_handle(pipe, upscale=True), prompt="p", width=704, height=480, num_frames=49, seed=7, steps=8, output_path="/tmp/o.mp4", stage1_steps=6)

        result = backend.text_to_video(**kwargs) if image_path is None else backend.image_to_video(image_path=image_path, **kwargs)

        pipe.generate_and_save.assert_called_once()
        assert pipe.generate_and_save.call_args.args == ()
        assert pipe.generate_and_save.call_args.kwargs == {
            "prompt": "p",
            "output_path": "/tmp/o.mp4",
            "height": 480,
            "width": 704,
            "num_frames": 49,
            "frame_rate": _FPS,
            "seed": 7,
            "stage1_steps": 6,
            "image": image_path,
        }
        assert result == Path("/tmp/o.mp4")

    def test_frame_rate_comes_from_the_handle(self, ltx):
        pipe = MagicMock()

        LtxVideoBackend().text_to_video(model=_handle(pipe, upscale=True, frame_rate=30.0), prompt="p", width=704, height=480, num_frames=49, seed=7, steps=8, output_path="/tmp/o.mp4", stage1_steps=8)

        assert pipe.generate_and_save.call_args.kwargs["frame_rate"] == 30.0

    def test_missing_stage1_steps_with_upscale_model_raises(self, ltx):
        pipe = MagicMock()

        with pytest.raises(ValueError, match="stage1_steps"):
            LtxVideoBackend().text_to_video(model=_handle(pipe, upscale=True), prompt="p", width=704, height=480, num_frames=49, seed=1, steps=8, output_path="/tmp/o.mp4")

        pipe.generate_and_save.assert_not_called()

    def test_no_callback_leaves_stepwise_unset(self, ltx):
        pipe = MagicMock()
        pipe.stepwise = "stale"

        LtxVideoBackend().text_to_video(model=_handle(pipe, upscale=True), prompt="p", width=704, height=480, num_frames=49, seed=7, steps=8, output_path="/tmp/o.mp4", stage1_steps=8)

        assert pipe.stepwise is None

    def test_callback_receives_cumulative_stage_events(self, ltx):
        pipe = MagicMock()
        events: list[dict] = []

        def _fake_generate_and_save(**_):
            for stage, total in ((1, 8), (2, 3)):
                on_step = pipe.stepwise.bind(latent_frames=3, latent_height=7, latent_width=11, decoder_block=None, patchifier=None, stage=stage)
                for i in range(total):
                    on_step(i, total, MagicMock(), 0.5)

        pipe.generate_and_save.side_effect = _fake_generate_and_save

        LtxVideoBackend().text_to_video(
            model=_handle(pipe, upscale=True),
            prompt="p",
            width=704,
            height=480,
            num_frames=49,
            seed=7,
            steps=8,
            output_path="/tmp/o.mp4",
            step_callback=events.append,
            stage1_steps=8,
        )

        assert [e["current_step"] for e in events] == list(range(1, 12))
        assert {e["total_steps"] for e in events} == {11}
        assert [e["phase"] for e in events] == ["video_upscale_stage_1"] * 8 + ["video_upscale_stage_2"] * 3


class TestTwoStageProgress:
    """Verify the adapter sitting in the pipeline's stepwise slot."""

    def test_stage_one_total_includes_stage_two(self):
        events: list[dict] = []
        progress = _TwoStageProgress(events.append, stage2_total=3)

        progress.bind(stage=1)(0, 5, None, 1.0)

        assert events == [{"phase": "video_upscale_stage_1", "current_step": 1, "total_steps": 8}]

    def test_stage_two_continues_after_stage_one(self):
        events: list[dict] = []
        progress = _TwoStageProgress(events.append, stage2_total=3)
        progress.bind(stage=1)(4, 5, None, 0.1)

        progress.bind(stage=2)(2, 3, None, 0.0)

        assert events[-1] == {"phase": "video_upscale_stage_2", "current_step": 8, "total_steps": 8}

    def test_missing_stage_is_stage_one(self):
        events: list[dict] = []
        progress = _TwoStageProgress(events.append, stage2_total=3)

        progress.bind(latent_frames=3, latent_height=7, latent_width=11, decoder_block=object(), patchifier=object())(0, 8, None, 1.0)

        assert events[0]["phase"] == "video_upscale_stage_1"
        assert events[0]["total_steps"] == 11


# ---------------------------------------------------------------------------
# macOS LTX backend — extending a clip
# ---------------------------------------------------------------------------

_WINDOW_LATENT_SHAPE = (9, 14, 22)  # (F, H, W) of a 73-frame window at 704x448
_EXT_WIDTH, _EXT_HEIGHT = 704, 448
_CARRY = 25


class _FakeWaveform:
    """Stand-in for the decoded audio: indexing gives itself and ``astype`` gives the samples."""

    def __init__(self, samples: np.ndarray) -> None:
        self._samples = samples

    def __getitem__(self, _index):
        return self

    def astype(self, _dtype) -> np.ndarray:
        return self._samples


def _chunks(sizes: list[int]) -> list[np.ndarray]:
    """Return decoded chunks ``(1, 3, T, 2, 2)`` whose pixels hold the frame's index in the window."""
    chunks, start = [], 0
    for size in sizes:
        values = np.arange(start, start + size, dtype=np.float32)
        chunks.append(np.broadcast_to(values[None, None, :, None, None], (1, 3, size, 2, 2)).copy())
        start += size
    return chunks


def _frame_ids(chunks) -> list[int]:
    return [int(v) for chunk in chunks for v in chunk[0, 0, :, 0, 0]]


_source = partial(
    make_extend_source,
    path=Path("/tmp/clip.mp4"),
    width=_EXT_WIDTH,
    height=_EXT_HEIGHT,
    frame_count=120,
    video_start=0.5,
    color_space="bt709",
    color_primaries="bt709",
    color_trc="bt709",
    scene=None,
    prompt=None,
)


def _audio_patchify(latent):
    batch, channels, tokens, width = latent.shape
    return latent.transpose(0, 2, 1, 3).reshape(batch, tokens, channels * width), tokens


@pytest.fixture
def ext(ltx, monkeypatch, tmp_path):
    """Extend the ``ltx`` fixture with the tail reader, the WAV writer and a runner for ``extend_video``."""
    ltx.patchifiers.compute_video_latent_shape.return_value = _WINDOW_LATENT_SHAPE
    ltx.tail = MagicMock(side_effect=lambda source: np.zeros((_CARRY, source.height, source.width, 3), np.uint8))
    ltx.wav = MagicMock()
    monkeypatch.setattr("zvisiongenerator.backends.mac.video.read_source_tail", ltx.tail)
    monkeypatch.setattr("zvisiongenerator.backends.mac.video.write_pcm16_wav", ltx.wav)
    ltx.segment_dir = tmp_path
    ltx.chunks = _chunks([16, 16, 16, 16, 9])
    ltx.waveform = _FakeWaveform(np.zeros((2, 146000), np.float32))
    ltx.state_at_decode = {}

    def pipeline(*, low_memory: bool = True) -> MagicMock:
        pipe = _pipeline(ltx, low_memory=low_memory)
        pipe.dit = object()
        pipe._loaded = True
        pipe.video_patchifier.patchify.return_value = ("video_tokens", (4, 14, 22))
        pipe.video_patchifier.unpatchify.return_value = SimpleNamespace(shape=(1, 128, *_WINDOW_LATENT_SHAPE))
        pipe.audio_patchifier.patchify.side_effect = _audio_patchify
        pipe.audio_conditioner.side_effect = lambda fn, free_after: fn("audio_encoder", "audio_processor")
        pipe.video_decoder_block.video_decoder = "conv"
        decoder = pipe.video_decoder_block.load.return_value

        def tiled_decode(latent, tiling):
            ltx.state_at_decode.update(dit=pipe.dit, loaded=pipe._loaded)
            return ltx.chunks

        decoder.tiled_decode.side_effect = tiled_decode
        pipe.audio_decoder_block.side_effect = lambda latent: ltx.waveform
        ltx.calls.attach_mock(pipe.audio_patchifier.patchify, "audio_patchify")
        return pipe

    def run(pipe=None, *, source=None, extend_frames=48, audio_output=True, step_callback=None, handle=None):
        pipe = pipe or pipeline()
        result = LtxVideoBackend().extend_video(
            handle or _handle(pipe),
            source or _source(),
            "a prompt",
            extend_frames=extend_frames,
            seed=42,
            steps=8,
            segment_dir=ltx.segment_dir,
            audio_output=audio_output,
            step_callback=step_callback,
        )
        return pipe, result

    ltx.pipeline = pipeline
    ltx.run = run
    return ltx


class TestExtendVideo:
    """Verify the carry-conditioned window generation; no model is loaded."""

    def test_returns_the_segment_files(self, ext):
        _, result = ext.run()

        assert result == ExtendSegment(video_path=ext.segment_dir / "segment.mp4", audio_path=ext.segment_dir / "segment.wav", lead_samples=12000)

    def test_reads_the_source_tail(self, ext):
        source = _source()

        ext.run(source=source)

        ext.tail.assert_called_once_with(source)

    @pytest.mark.parametrize("extend_frames", [48, 96])
    def test_window_is_the_carry_plus_the_new_frames(self, ext, extend_frames):
        ext.run(extend_frames=extend_frames)

        ext.patchifiers.compute_video_latent_shape.assert_called_once_with(_CARRY + extend_frames, _EXT_HEIGHT, _EXT_WIDTH)
        ext.positions.compute_audio_token_count.assert_called_once_with(_CARRY + extend_frames, frame_rate=24.0)
        video_call, _ = ext.helpers.create_noised_state.call_args_list
        assert video_call.kwargs["base_shape"] == (1, 9 * 14 * 22, 128)
        assert video_call.kwargs["spatial_dims"] == _WINDOW_LATENT_SHAPE

    @pytest.mark.parametrize("frames", [0, 7, 12, 50])
    def test_extend_frames_must_be_a_positive_multiple_of_eight(self, ext, frames):
        with pytest.raises(ValueError, match="multiple of 8"):
            ext.run(extend_frames=frames)

        ext.tail.assert_not_called()

    def test_video_pin_covers_the_carried_latent_frames_at_full_strength(self, ext):
        pipe, _ = ext.run()

        video_call, _ = ext.helpers.create_noised_state.call_args_list
        (pin,) = video_call.kwargs["conditionings"]
        assert pin.frame_indices == [0, 1, 2, 3]
        assert pin.clean_latent == "video_tokens"
        assert pin.strength == 1.0
        pipe.video_patchifier.patchify.assert_called_once_with("video_latent")

    def test_tail_frames_are_encoded_at_the_displayed_size_inside_the_conditioner(self, ext):
        pipe = ext.pipeline(low_memory=True)

        ext.run(pipe)

        (frames, height, width), _ = ext.media_io.video_preprocess.call_args
        assert len(frames) == _CARRY
        assert (height, width) == (_EXT_HEIGHT, _EXT_WIDTH)
        ext.retake.encode_video_tensor.assert_called_once_with(ext.encoder, "video_tensor")
        assert pipe.image_conditioner.call_args.kwargs == {"free_after": True}

    def test_carry_is_encoded_before_the_pipeline_loads(self, ext):
        ext.run()

        names = _call_names(ext)
        assert names.index("encode_video_tensor") < names.index("load")
        assert names.index("encode_audio") < names.index("load")

    def test_audio_is_fitted_before_it_is_patchified(self, ext):
        ext.run()

        names = _call_names(ext)
        assert names.index("encode_audio") < names.index("fit_audio_latent") < names.index("audio_patchify")

    def test_audio_pin_has_the_carry_token_shape(self, ext):
        ext.run()

        _, audio_call = ext.helpers.create_noised_state.call_args_list
        (pin,) = audio_call.kwargs["conditionings"]
        assert pin.clean_latent.shape == (1, 26, 128)
        assert pin.frame_indices == list(range(26))
        assert pin.strength == 1.0
        assert audio_call.kwargs["spatial_dims"] == (_AUDIO_TOKENS, 1, 1)
        ext.extend_distilled.fit_audio_latent.assert_called_once()
        assert ext.extend_distilled.fit_audio_latent.call_args.args[1] == 26

    def test_audio_state_uses_the_next_seed_and_legacy_blend(self, ext):
        ext.run()

        video_call, audio_call = ext.helpers.create_noised_state.call_args_list
        assert (video_call.kwargs["seed"], audio_call.kwargs["seed"]) == (42, 43)
        assert audio_call.kwargs["legacy_scalar_blend"] is True
        assert audio_call.kwargs["sigma"] == 1.0

    def test_tail_audio_is_read_at_its_position_in_the_file(self, ext):
        source = _source(video_start=0.5, frame_count=120)

        ext.run(source=source)

        ext.audio_io.load_audio.assert_called_once()
        assert ext.audio_io.load_audio.call_args.args == (source.path,)
        kwargs = ext.audio_io.load_audio.call_args.kwargs
        assert kwargs["target_sample_rate"] == 16000
        assert kwargs["start_time"] == pytest.approx(0.5 + (120 - _CARRY) / 24.0)
        assert kwargs["max_duration"] == pytest.approx(_CARRY / 24.0)
        ext.audio_vae.encode_audio.assert_called_once_with("tail_waveform", 16000, "audio_encoder", "audio_processor")

    def test_unreadable_audio_raises_before_generating(self, ext):
        ext.audio_io.load_audio.return_value = None

        with pytest.raises(RuntimeError, match="audio"):
            ext.run()

        ext.samplers.denoise_loop.assert_not_called()

    def test_silent_source_has_no_pin_no_decode_and_no_wav(self, ext):
        pipe, result = ext.run(source=_source(has_audio=False))

        ext.audio_io.load_audio.assert_not_called()
        ext.audio_vae.encode_audio.assert_not_called()
        ext.extend_distilled.fit_audio_latent.assert_not_called()
        pipe.audio_patchifier.patchify.assert_not_called()
        _, audio_call = ext.helpers.create_noised_state.call_args_list
        assert audio_call.kwargs["conditionings"] == []
        assert audio_call.kwargs["spatial_dims"] == _WINDOW_LATENT_SHAPE
        pipe.audio_decoder_block.assert_not_called()
        ext.wav.assert_not_called()
        assert result.audio_path is None

    def test_silent_source_still_pins_the_video(self, ext):
        ext.run(source=_source(has_audio=False))

        video_call, _ = ext.helpers.create_noised_state.call_args_list
        assert len(video_call.kwargs["conditionings"]) == 1

    def test_without_audio_output_the_pin_stays_but_no_wav_is_written(self, ext):
        pipe, result = ext.run(audio_output=False)

        _, audio_call = ext.helpers.create_noised_state.call_args_list
        assert len(audio_call.kwargs["conditionings"]) == 1
        pipe.audio_decoder_block.assert_not_called()
        ext.wav.assert_not_called()
        assert result == ExtendSegment(video_path=ext.segment_dir / "segment.mp4", audio_path=None)

    def test_the_source_frame_rate_overrides_the_handle_for_positions_audio_length_and_decode(self, ext):
        pipe = ext.pipeline()
        ext.video_vae._compute_decode_tiling.return_value = "tiling"

        ext.run(pipe, source=_source(fps=30.0, fps_text="30/1"), handle=_handle(pipe, frame_rate=24.0))

        ext.positions.compute_video_positions.assert_called_once_with(*_WINDOW_LATENT_SHAPE, frame_rate=30.0)
        ext.positions.compute_audio_token_count.assert_called_once_with(_CARRY + 48, frame_rate=30.0)
        ext.extend_distilled.carry_audio_tokens.assert_called_once_with(30.0, _CARRY)
        assert ext.video_vae._compute_decode_tiling.call_args.kwargs == {"frame_rate": 30.0}
        assert ext.audio_io.load_audio.call_args.kwargs["start_time"] == pytest.approx(0.5 + (120 - _CARRY) / 30.0)
        command = ext.video_vae._ffmpeg_sink.call_args.args[0]
        assert command[command.index("-r") + 1] == "30/1"
        pipe._decode_and_save_video.assert_not_called()

    def test_the_segment_is_encoded_with_the_source_size_and_colour_tags(self, ext):
        ext.run(source=_source(color_space="smpte170m", color_primaries="smpte170m", color_trc="bt709"))

        command = ext.video_vae._ffmpeg_sink.call_args.args[0]
        assert command[command.index("-s") + 1] == f"{_EXT_WIDTH}x{_EXT_HEIGHT}"
        assert command[-1] == str(ext.segment_dir / "segment.mp4")
        assert "colorspace=smpte170m" in command[command.index("-vf") + 1]

    def test_the_dit_is_freed_before_the_video_decode(self, ext):
        pipe, _ = ext.run(ext.pipeline(low_memory=True))

        assert ext.state_at_decode == {"dit": None, "loaded": False}
        assert pipe.dit is None

    def test_the_dit_stays_without_low_memory(self, ext):
        pipe = ext.pipeline(low_memory=False)
        dit = pipe.dit

        ext.run(pipe)

        assert ext.state_at_decode == {"dit": dit, "loaded": True}

    def test_decoders_are_freed_afterwards_under_low_memory(self, ext):
        pipe, _ = ext.run(ext.pipeline(low_memory=True))

        assert pipe.vae_decoder is None
        assert pipe.audio_decoder is None
        assert pipe.vocoder is None

    def test_tiled_decode_and_streaming_both_run_inside_the_cache_limit(self, ext):
        events: list[str] = []
        pipe = ext.pipeline()
        decoder = pipe.video_decoder_block.load.return_value
        decode = decoder.tiled_decode.side_effect
        decoder.tiled_decode.side_effect = lambda *args: (events.append("tiled_decode"), decode(*args))[1]
        stream = ext.video_vae.stream_chunks_to_ffmpeg.side_effect
        ext.video_vae.stream_chunks_to_ffmpeg.side_effect = lambda chunks, proc: (events.append("stream"), stream(chunks, proc))[1]
        limit = ext.video_vae.decode_cache_limit.return_value
        limit.__enter__.side_effect = lambda: events.append("limit_on")
        limit.__exit__.side_effect = lambda *exc: events.append("limit_off")
        ext.wav.side_effect = lambda *args: events.append("wav")

        ext.run(pipe)

        assert events == ["limit_on", "tiled_decode", "stream", "limit_off", "wav"]

    @pytest.mark.parametrize("tiling", [None, "temporal-tiles"])
    def test_only_the_new_frames_are_streamed_with_or_without_tiling(self, ext, tiling):
        ext.video_vae._compute_decode_tiling.return_value = tiling
        ext.chunks = _chunks([73]) if tiling is None else _chunks([16, 16, 16, 16, 9])

        ext.run()

        assert _frame_ids(ext.streamed) == list(range(_CARRY, _CARRY + 48))
        assert ext.video_vae._compute_decode_tiling.call_args.args[0] == (1, 128, *_WINDOW_LATENT_SHAPE)

    def test_audio_is_cut_to_the_new_frames(self, ext):
        samples = np.arange(2 * 146000, dtype=np.float32).reshape(2, 146000) / 1e6
        ext.waveform = _FakeWaveform(samples)

        ext.run()

        written, path = ext.wav.call_args.args
        assert written.shape == (2, 12000 + 96000)  # a quarter second of the carried frames' sound, then the new frames' sound
        np.testing.assert_array_equal(written, samples[:, 50000 - 12000 : 146000])
        assert path == ext.segment_dir / "segment.wav"

    def test_short_audio_fades_out_and_is_zero_padded_to_the_new_frames(self, ext):
        ext.waveform = _FakeWaveform(np.ones((2, 144480), np.float32))

        ext.run()

        written = ext.wav.call_args.args[0]
        assert written.shape == (2, 12000 + 96000)
        real_end = 144480 - (50000 - 12000)
        assert written[:, : real_end - 384].min() == 1.0
        assert written[:, real_end - 1].max() == 0.0  # the last real sample is the end of the fade
        assert written[:, real_end:].max() == 0.0

    def test_audio_cut_follows_the_source_frame_rate(self, ext):
        ext.run(source=_source(fps=30.0, fps_text="30/1"), extend_frames=48)

        assert ext.wav.call_args.args[0].shape == (2, 12000 + round(48 / 30 * 48000))

    def test_upscale_model_is_rejected(self, ext):
        pipe = ext.pipeline()

        with pytest.raises(ValueError, match="upscale"):
            ext.run(pipe, handle=_handle(pipe, upscale=True))

        ext.tail.assert_not_called()

    def test_diffusion_decoder_is_rejected(self, ext):
        pipe = ext.pipeline()
        pipe.video_decoder_block.video_decoder = "diffusion"

        with pytest.raises(ValueError, match="diffusion"):
            ext.run(pipe)

        pipe.load.assert_not_called()

    def test_progress_is_reported_as_video_generate(self, ext):
        events: list[dict] = []

        ext.run(step_callback=events.append)

        on_step = ext.samplers.denoise_loop.call_args.kwargs["on_step"]
        on_step(2, 8, MagicMock(), 0.5)
        assert events == [{"phase": "video_generate", "current_step": 3, "total_steps": 8}]

    def test_a_callback_that_raises_stops_the_generation(self, ext):
        class Quit(Exception):
            pass

        def stop(_event):
            raise Quit

        ext.samplers.denoise_loop.side_effect = lambda **kwargs: kwargs["on_step"](0, 8, None, 1.0)

        with pytest.raises(Quit):
            ext.run(step_callback=stop)

        ext.video_vae.stream_chunks_to_ffmpeg.assert_not_called()

    def test_prompt_and_distilled_schedule_are_used(self, ext):
        pipe, _ = ext.run()

        assert pipe.prompt_encoder.call_args == call("a prompt", free_after=True)
        ext.scheduler.shorten_schedule.assert_called_once_with(_FAKE_DISTILLED_SIGMAS, 8, keep="start")

    def test_a_2_5_extension_stays_deterministic_on_the_2_5_schedule(self, ext):
        pipe = ext.pipeline()
        ext.run(pipe, handle=_handle(pipe, ltx25=True))

        ext.samplers.euler_ancestral_denoising_loop.assert_not_called()
        ext.scheduler.shorten_schedule.assert_called_once_with(_FAKE_LTX25_SIGMAS, 8, keep="start")
        ext.samplers.denoise_loop.assert_called_once()
        assert ext.samplers.denoise_loop.call_args.kwargs["sigmas"] is ext.scheduler.shorten_schedule.return_value

    def test_streaming_handle_scopes_the_cache_limit_over_the_carry_encode_and_generation(self, ext):
        ext.mx.set_cache_limit.return_value = "previous"
        pipe = ext.pipeline()
        limits_at_encode = []

        def encode(*_):
            limits_at_encode.append(list(ext.mx.set_cache_limit.call_args_list))
            return "video_latent"

        ext.retake.encode_video_tensor.side_effect = encode

        ext.run(pipe, handle=_handle(pipe, low_ram_streaming=True))

        assert limits_at_encode == [[call(0)]]
        assert ext.mx.set_cache_limit.call_args_list == [call(0), call("previous")]

    def test_non_streaming_handle_leaves_the_cache_limit_alone(self, ext):
        ext.run()

        ext.mx.set_cache_limit.assert_not_called()


class TestDropLeadingFrames:
    """Verify the cut of the carried frames from decoded chunks, whatever the chunking."""

    @pytest.mark.parametrize(
        "sizes",
        [
            [73],
            [16, 16, 16, 16, 9],
            [25, 48],
            [10, 15, 20, 28],
            [24, 1, 48],
            [26, 47],
            [1] * 73,
            [80],
        ],
    )
    def test_leaves_the_frames_after_the_carry(self, sizes):
        kept = list(_drop_leading_frames(_chunks(sizes), _CARRY))

        assert _frame_ids(kept) == list(range(_CARRY, sum(sizes)))

    def test_whole_chunks_inside_the_carry_are_skipped(self):
        kept = list(_drop_leading_frames(_chunks([16, 9, 20]), _CARRY))

        assert [chunk.shape[2] for chunk in kept] == [20]

    def test_a_chunk_after_the_carry_passes_through_unchanged(self):
        chunks = _chunks([25, 16])

        kept = list(_drop_leading_frames(chunks, _CARRY))

        assert kept[0] is chunks[1]

    def test_keeps_the_other_axes(self):
        (kept,) = _drop_leading_frames(_chunks([73]), _CARRY)

        assert kept.shape == (1, 3, 48, 2, 2)

    def test_consumes_the_chunks_lazily(self):
        pulled = []

        def source():
            for chunk in _chunks([16, 16, 16, 16, 9]):
                pulled.append(chunk.shape[2])
                yield chunk

        first = next(_drop_leading_frames(source(), _CARRY))

        assert first.shape[2] == 7
        assert pulled == [16, 16]

    def test_carry_longer_than_the_clip_yields_nothing(self):
        assert list(_drop_leading_frames(_chunks([16, 9]), _CARRY + 1)) == []


class TestResolveModelDir:
    """A pinned MLX pack resolves to exactly its listed files at its revision; other references resolve as before."""

    _PIN = "d" * 40
    _PACK = HubFiles("owner/pack", _PIN, ("a.safetensors", "b.json"))
    _PACKS = {f"owner/pack@{_PIN}": _PACK}

    @staticmethod
    def _snapshot(root, *names):
        root.mkdir(parents=True, exist_ok=True)
        for name in names:
            (root / name).write_bytes(b"x")
        return root

    def test_a_cached_snapshot_with_every_listed_file_needs_no_download(self, monkeypatch, tmp_path):
        snapshot = self._snapshot(tmp_path / "snap", "a.safetensors", "b.json", "unlisted.bin")
        seen = []
        monkeypatch.setattr("zvisiongenerator.utils.hf_snapshot.cached_snapshot_dirs", lambda repo, revision: (seen.append((repo, revision)), [snapshot])[1])
        download, find = MagicMock(), MagicMock()

        resolved = _resolve_model_dir(f"owner/pack@{self._PIN}", packs=self._PACKS, download=download, find_local_dir=find)

        assert resolved == str(snapshot)
        assert seen == [("owner/pack", self._PIN)]
        download.assert_not_called()
        find.assert_not_called()

    def test_a_missing_listed_file_downloads_just_the_listed_files_at_the_pin(self, monkeypatch, tmp_path):
        partial = self._snapshot(tmp_path / "partial", "a.safetensors")
        full = self._snapshot(tmp_path / "full", "a.safetensors", "b.json")
        monkeypatch.setattr("zvisiongenerator.utils.hf_snapshot.cached_snapshot_dirs", lambda repo, revision: [partial])
        download = MagicMock(return_value=str(full))

        resolved = _resolve_model_dir(f"owner/pack@{self._PIN}", packs=self._PACKS, download=download)

        assert resolved == str(full)
        download.assert_called_once_with(repo_id="owner/pack", revision=self._PIN, allow_patterns=["a.safetensors", "b.json"])

    def test_a_revision_without_a_listed_file_is_an_error_naming_the_revision_and_the_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr("zvisiongenerator.utils.hf_snapshot.cached_snapshot_dirs", lambda repo, revision: [])
        download = MagicMock(return_value=str(self._snapshot(tmp_path / "snap", "a.safetensors")))

        with pytest.raises(FileNotFoundError, match=f"owner/pack@{self._PIN}.*b.json"):
            _resolve_model_dir(f"owner/pack@{self._PIN}", packs=self._PACKS, download=download)

    def test_a_pinned_reference_that_is_no_pack_is_the_complete_snapshot_or_a_download(self, tmp_path):
        find = MagicMock(return_value=tmp_path)
        download = MagicMock(return_value=str(tmp_path / "dl"))

        assert _resolve_model_dir("o/r@abc", packs=self._PACKS, find_local_dir=find, download=download) == str(tmp_path)
        find.return_value = None
        assert _resolve_model_dir("o/r@abc", packs=self._PACKS, find_local_dir=find, download=download) == str(tmp_path / "dl")
        download.assert_called_once_with(repo_id="o/r", revision="abc")

    @pytest.mark.parametrize("reference", ["owner/pack", "Owner/PACK", f"OWNER/pack@{_PIN}"])
    def test_a_reference_to_a_pack_repo_resolves_to_the_pack_whatever_its_case_or_pin(self, monkeypatch, tmp_path, reference):
        full = self._snapshot(tmp_path / "full", "a.safetensors", "b.json")
        monkeypatch.setattr("zvisiongenerator.utils.hf_snapshot.cached_snapshot_dirs", lambda repo, revision: [])
        download, find = MagicMock(return_value=str(full)), MagicMock()

        resolved = _resolve_model_dir(reference, packs=self._PACKS, download=download, find_local_dir=find)

        assert resolved == str(full)
        download.assert_called_once_with(repo_id="owner/pack", revision=self._PIN, allow_patterns=["a.safetensors", "b.json"])
        find.assert_not_called()

    def test_a_pack_repo_at_another_revision_is_no_pack_and_downloads_that_revision_whole(self, tmp_path):
        find = MagicMock(return_value=None)
        download = MagicMock(return_value=str(tmp_path))

        assert _resolve_model_dir("owner/pack@" + "e" * 40, packs=self._PACKS, find_local_dir=find, download=download) == str(tmp_path)
        download.assert_called_once_with(repo_id="owner/pack", revision="e" * 40)

    def test_an_unpinned_reference_to_no_pack_and_a_folder_come_back_unchanged(self, tmp_path):
        download = MagicMock()

        assert _resolve_model_dir("owner/other", packs=self._PACKS, download=download) == "owner/other"
        assert _resolve_model_dir(str(tmp_path), packs=self._PACKS, download=download) == str(tmp_path)
        download.assert_not_called()

    def test_a_download_limited_to_patterns_asks_the_hub_for_just_those(self, tmp_path):
        download = MagicMock(return_value=str(tmp_path))

        _resolve_model_dir("o/r@abc", packs={}, find_local_dir=MagicMock(return_value=None), download=download, allow_patterns=("*.json", "model*.safetensors"))

        download.assert_called_once_with(repo_id="o/r", revision="abc", allow_patterns=["*.json", "model*.safetensors"])
