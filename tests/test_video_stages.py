"""Tests for video workflow stage functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.video_types import VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.workflows.video_stages import (
    generate_filename_stage,
    image_to_video_stage,
    log_video_stage,
    resolve_prompt_stage,
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
# resolve_prompt_stage
# ---------------------------------------------------------------------------


class TestResolvePromptStage:
    """Verify random-choice block expansion in video prompts."""

    def test_plain_prompt_passthrough(self):
        req = _req(prompt="a simple prompt")
        arts = VideoWorkingArtifacts()
        outcome = resolve_prompt_stage(req, arts)
        assert outcome is StageOutcome.success
        assert arts.resolved_prompt == "a simple prompt"

    def test_choice_block_resolves(self):
        req = _req(prompt="{red|green|blue} sky")
        arts = VideoWorkingArtifacts()
        outcome = resolve_prompt_stage(req, arts)
        assert outcome is StageOutcome.success
        assert arts.resolved_prompt in ("red sky", "green sky", "blue sky")

    def test_nested_choice_blocks(self):
        req = _req(prompt="{a|{b|c}}")
        arts = VideoWorkingArtifacts()
        outcome = resolve_prompt_stage(req, arts)
        assert outcome is StageOutcome.success
        assert arts.resolved_prompt in ("a", "b", "c")

    def test_whitespace_stripped(self):
        req = _req(prompt="  hello world  ")
        arts = VideoWorkingArtifacts()
        resolve_prompt_stage(req, arts)
        assert arts.resolved_prompt == "hello world"


# ---------------------------------------------------------------------------
# generate_filename_stage
# ---------------------------------------------------------------------------


class TestGenerateFilenameStage:
    """Verify filename format and sanitization."""

    def test_basic_filename(self):
        req = _req(model_name="test-model", seed=42, width=704, height=480, num_frames=49, steps=30, output_format="mp4")
        arts = VideoWorkingArtifacts()
        outcome = generate_filename_stage(req, arts)
        assert outcome is StageOutcome.success
        assert arts.filename.endswith(".mp4")
        assert "_704x480_" in arts.filename
        assert "_49f_" in arts.filename
        assert "_test-model_" in arts.filename
        assert "_steps30_" in arts.filename
        assert "_seed42" in arts.filename

    def test_slash_in_model_name_sanitized(self):
        req = _req(model_name="dgrauet/ltx-2.3-mlx-q4", seed=1, width=512, height=512, num_frames=25, steps=30, output_format="mp4")
        arts = VideoWorkingArtifacts()
        generate_filename_stage(req, arts)
        assert "/" not in arts.filename
        assert "ltx-2.3-mlx-q4" in arts.filename

    def test_spaces_sanitized(self):
        req = _req(model_name="my cool model", seed=7, width=100, height=100, num_frames=9, steps=30, output_format="mp4")
        arts = VideoWorkingArtifacts()
        generate_filename_stage(req, arts)
        assert " " not in arts.filename

    def test_none_model_name(self):
        req = _req(model_name=None, seed=0, width=704, height=480, num_frames=49, steps=30, output_format="mp4")
        arts = VideoWorkingArtifacts()
        generate_filename_stage(req, arts)
        assert arts.filename.endswith(".mp4")
        assert "_49f_" in arts.filename
        assert "_seed0" in arts.filename

    def test_set_name_included(self):
        req = _req(filename_base="sunset", model_name="test", seed=1, width=704, height=480, num_frames=49, steps=30, output_format="mp4")
        arts = VideoWorkingArtifacts()
        generate_filename_stage(req, arts)
        assert arts.filename.startswith("sunset_")


# ---------------------------------------------------------------------------
# text_to_video_stage
# ---------------------------------------------------------------------------


class TestTextToVideoStage:
    """Verify text_to_video_stage delegates to backend correctly."""

    def test_success_sets_artifacts(self):
        backend = MagicMock()
        backend.text_to_video.return_value = Path("/tmp/out.mp4")
        req = _req(backend=backend, prompt="test", output_dir="/tmp", output_format="mp4", seed=42, width=704, height=480, num_frames=49, steps=30)
        arts = VideoWorkingArtifacts()
        arts.filename = "test.mp4"
        arts.resolved_prompt = "test"

        outcome = text_to_video_stage(req, arts)
        assert outcome is StageOutcome.success
        assert arts.video_path == Path("/tmp/out.mp4")
        assert arts.generation_time > 0
        backend.text_to_video.assert_called_once()

    def test_backend_returns_none_fails(self):
        backend = MagicMock()
        backend.text_to_video.return_value = None
        req = _req(backend=backend)
        arts = VideoWorkingArtifacts()
        arts.filename = "test.mp4"
        arts.resolved_prompt = "test"

        outcome = text_to_video_stage(req, arts)
        assert outcome is StageOutcome.failed
        assert arts.video_path is None

    def test_uses_resolved_prompt(self):
        backend = MagicMock()
        backend.text_to_video.return_value = Path("/tmp/x.mp4")
        req = _req(backend=backend, prompt="original")
        arts = VideoWorkingArtifacts()
        arts.filename = "f.mp4"
        arts.resolved_prompt = "resolved version"

        text_to_video_stage(req, arts)
        call_kwargs = backend.text_to_video.call_args
        assert call_kwargs.kwargs.get("prompt") == "resolved version" or call_kwargs[1].get("prompt") == "resolved version"


# ---------------------------------------------------------------------------
# image_to_video_stage
# ---------------------------------------------------------------------------


class TestImageToVideoStage:
    """Verify image_to_video_stage delegates to backend correctly."""

    def test_success_sets_artifacts(self):
        backend = MagicMock()
        backend.image_to_video.return_value = Path("/tmp/i2v.mp4")
        req = _req(backend=backend, image_path="/tmp/input.png", output_dir="/tmp", output_format="mp4")
        arts = VideoWorkingArtifacts()
        arts.filename = "i2v.mp4"
        arts.resolved_prompt = "describe this"

        outcome = image_to_video_stage(req, arts)
        assert outcome is StageOutcome.success
        assert arts.video_path == Path("/tmp/i2v.mp4")
        backend.image_to_video.assert_called_once()

    def test_backend_returns_none_fails(self):
        backend = MagicMock()
        backend.image_to_video.return_value = None
        req = _req(backend=backend, image_path="/tmp/input.png")
        arts = VideoWorkingArtifacts()
        arts.filename = "i2v.mp4"

        outcome = image_to_video_stage(req, arts)
        assert outcome is StageOutcome.failed


# ---------------------------------------------------------------------------
# log_video_stage
# ---------------------------------------------------------------------------


class TestLogVideoStage:
    """Verify log_video_stage runs without error."""

    def test_prints_output(self, capsys):
        req = _req()
        arts = VideoWorkingArtifacts()
        arts.video_path = Path("/tmp/out.mp4")
        arts.generation_time = 12.5

        outcome = log_video_stage(req, arts)
        assert outcome is StageOutcome.success
        captured = capsys.readouterr()
        assert "/tmp/out.mp4" in captured.out
        assert "12.5s" in captured.out

    def test_none_path_prints_unknown(self, capsys):
        req = _req()
        arts = VideoWorkingArtifacts()
        arts.video_path = None

        log_video_stage(req, arts)
        captured = capsys.readouterr()
        assert "unknown" in captured.out
