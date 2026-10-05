"""Tests for video workflow stage functions."""

from __future__ import annotations

import re
import subprocess
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    """Verify the set-name + timestamp filename and its collision counter."""

    def test_set_name_and_timestamp(self, tmp_path):
        req = _req(filename_base="sunset", output_dir=str(tmp_path), output_format="mp4")
        arts = VideoWorkingArtifacts()
        outcome = generate_filename_stage(req, arts)
        assert outcome is StageOutcome.success
        assert re.fullmatch(r"sunset_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.mp4", arts.filename)

    def test_settings_are_not_in_the_name(self, tmp_path):
        req = _req(filename_base="sunset", model_name="dgrauet/ltx-2.3-mlx-q4", seed=42, output_dir=str(tmp_path))
        arts = VideoWorkingArtifacts()
        generate_filename_stage(req, arts)
        assert "ltx" not in arts.filename
        assert "seed" not in arts.filename

    def test_taken_name_gets_counter(self, tmp_path):
        req = _req(filename_base="sunset", output_dir=str(tmp_path), output_format="mp4")
        with patch("zvisiongenerator.utils.filename.generate_filename", return_value="sunset_2026-10-04_14-03-22"):
            (tmp_path / "sunset_2026-10-04_14-03-22.mp4").write_bytes(b"")
            arts = VideoWorkingArtifacts()
            generate_filename_stage(req, arts)
        assert arts.filename == "sunset_2026-10-04_14-03-22_2.mp4"


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
    """Verify log_video_stage embeds config metadata and logs correctly."""

    def test_returns_success_without_mutating_artifacts(self):
        req = _req()
        arts = VideoWorkingArtifacts()
        arts.video_path = Path("/tmp/out.mp4")
        arts.generation_time = 12.5

        with patch("zvisiongenerator.workflows.video_stages.embed_mp4_config"):
            outcome = log_video_stage(req, arts)

        assert outcome is StageOutcome.success
        assert arts.video_path == Path("/tmp/out.mp4")
        assert arts.generation_time == 12.5

    def test_calls_embed_mp4_config_with_video_path(self, tmp_path):
        video_path = tmp_path / "out.mp4"
        video_path.write_bytes(b"fake-video")
        req = _req(
            prompt="a moving scene",
            model_name="ltx-8",
            model_family="ltx",
            width=704,
            height=448,
            num_frames=49,
            seed=77,
            steps=8,
            image_path="/tmp/ref.png",
        )
        arts = VideoWorkingArtifacts(video_path=video_path, generation_time=12.5, filename="out.mp4")

        with patch("zvisiongenerator.workflows.video_stages.embed_mp4_config") as mock_embed:
            outcome = log_video_stage(req, arts)

        assert outcome is StageOutcome.success
        mock_embed.assert_called_once()
        call_path, call_payload = mock_embed.call_args[0]
        assert call_path == video_path
        assert call_payload["prompt"] == "a moving scene"
        assert call_payload["model"] == "ltx-8"
        assert call_payload["seed"] == 77
        assert call_payload["frame_count"] == 49
        assert call_payload["workflow"] == "img2vid"

    def test_warns_and_succeeds_on_ffmpeg_embed_failure(self):
        req = _req(prompt="test", model_name="ltx")
        arts = VideoWorkingArtifacts(video_path=Path("/tmp/out.mp4"), generation_time=5.0)

        with patch("zvisiongenerator.workflows.video_stages.embed_mp4_config") as mock_embed:
            mock_embed.side_effect = subprocess.CalledProcessError(1, "ffmpeg", stderr=b"mux error")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                outcome = log_video_stage(req, arts)

        assert outcome is StageOutcome.success
        assert any("ffmpeg metadata embed failed" in str(w.message) for w in caught)

    def test_none_path_still_succeeds(self):
        req = _req()
        arts = VideoWorkingArtifacts()
        arts.video_path = None
        arts.generation_time = 0.0

        outcome = log_video_stage(req, arts)

        assert outcome is StageOutcome.success
        assert arts.video_path is None
        assert arts.generation_time == 0.0
