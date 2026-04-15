"""Tests for strip_audio() utility and strip_audio_stage — audio stripping in video output."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.video_types import VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.utils.ffmpeg import strip_audio
from zvisiongenerator.workflows import build_video_workflow
from zvisiongenerator.workflows.video_stages import strip_audio_stage


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
# strip_audio_stage
# ---------------------------------------------------------------------------


class TestStripAudioStage:
    """Verify strip_audio_stage behaviour for various request/artifact states."""

    @patch("zvisiongenerator.workflows.video_stages.strip_audio")
    def test_no_audio_false_does_not_call_ffmpeg(self, mock_strip):
        """Returns success and does NOT call ffmpeg when no_audio is False."""
        req = _req(no_audio=False)
        arts = VideoWorkingArtifacts()
        arts.video_path = Path("/tmp/video.mp4")

        outcome = strip_audio_stage(req, arts)

        assert outcome is StageOutcome.success
        mock_strip.assert_not_called()

    @patch("zvisiongenerator.workflows.video_stages.strip_audio")
    def test_no_audio_true_calls_strip_audio(self, mock_strip):
        """Calls strip_audio() when no_audio is True and video_path is set."""
        req = _req(no_audio=True)
        arts = VideoWorkingArtifacts()
        arts.video_path = Path("/tmp/video.mp4")

        outcome = strip_audio_stage(req, arts)

        assert outcome is StageOutcome.success
        mock_strip.assert_called_once_with(Path("/tmp/video.mp4"))

    @patch("zvisiongenerator.workflows.video_stages.strip_audio")
    def test_no_audio_true_but_no_video_path(self, mock_strip):
        """Returns success when no_audio is True but video_path is None."""
        req = _req(no_audio=True)
        arts = VideoWorkingArtifacts()
        arts.video_path = None

        outcome = strip_audio_stage(req, arts)

        assert outcome is StageOutcome.success
        mock_strip.assert_not_called()


# ---------------------------------------------------------------------------
# strip_audio() utility
# ---------------------------------------------------------------------------


class TestStripAudioUtility:
    """Verify strip_audio() calls ffmpeg with correct args and replaces original file."""

    @patch("zvisiongenerator.utils.ffmpeg.subprocess.run")
    def test_calls_ffmpeg_with_correct_command(self, mock_run, tmp_path):
        """strip_audio() calls subprocess with the expected ffmpeg arguments."""
        video = tmp_path / "output.mp4"
        video.write_bytes(b"fake video data")
        expected_tmp = video.with_suffix(".tmp.mp4")

        # Make subprocess.run create the tmp file (simulating ffmpeg)
        def fake_run(cmd, **kwargs):
            Path(cmd[-1]).write_bytes(b"stripped video")
            return MagicMock(returncode=0)

        mock_run.side_effect = fake_run

        strip_audio(video)

        mock_run.assert_called_once_with(
            ["ffmpeg", "-y", "-i", str(video), "-an", "-c:v", "copy", str(expected_tmp)],
            check=True,
            capture_output=True,
        )

    @patch("zvisiongenerator.utils.ffmpeg.subprocess.run")
    def test_replaces_original_file(self, mock_run, tmp_path):
        """strip_audio() replaces the original file with the temp file."""
        video = tmp_path / "output.mp4"
        video.write_bytes(b"original data")
        expected_tmp = video.with_suffix(".tmp.mp4")

        def fake_run(cmd, **kwargs):
            # Simulate ffmpeg creating the tmp file
            Path(cmd[-1]).write_bytes(b"stripped data")
            return MagicMock(returncode=0)

        mock_run.side_effect = fake_run

        strip_audio(video)

        # The original file should now contain the stripped data
        assert video.read_bytes() == b"stripped data"
        # The tmp file should be gone (replaced via rename)
        assert not expected_tmp.exists()

    @patch("zvisiongenerator.utils.ffmpeg.subprocess.run", side_effect=Exception("ffmpeg failed"))
    def test_ffmpeg_failure_propagates(self, mock_run, tmp_path):
        """If ffmpeg fails, the exception propagates."""
        video = tmp_path / "output.mp4"
        video.write_bytes(b"data")

        with pytest.raises(Exception, match="ffmpeg failed"):
            strip_audio(video)


# ---------------------------------------------------------------------------
# Workflow Assembly — strip_audio_stage inclusion
# ---------------------------------------------------------------------------


class TestBuildVideoWorkflowAudio:
    """Verify build_video_workflow() includes/excludes strip_audio_stage."""

    def test_no_audio_true_includes_strip_audio_stage(self):
        """build_video_workflow with no_audio=True includes strip_audio_stage."""
        args = Namespace(image_path=None, no_audio=True)
        wf = build_video_workflow(args)
        stage_names = [s.__name__ for s in wf.stages]
        assert "strip_audio_stage" in stage_names

    def test_no_audio_false_excludes_strip_audio_stage(self):
        """build_video_workflow with no_audio=False does NOT include strip_audio_stage."""
        args = Namespace(image_path=None, no_audio=False)
        wf = build_video_workflow(args)
        stage_names = [s.__name__ for s in wf.stages]
        assert "strip_audio_stage" not in stage_names

    def test_no_audio_attr_missing_excludes_strip_audio(self):
        """build_video_workflow with no no_audio attr defaults to not including strip_audio_stage."""
        args = Namespace(image_path=None)
        wf = build_video_workflow(args)
        stage_names = [s.__name__ for s in wf.stages]
        assert "strip_audio_stage" not in stage_names

    def test_strip_audio_stage_after_generation(self):
        """strip_audio_stage comes after the generation stage and before log."""
        args = Namespace(image_path=None, no_audio=True)
        wf = build_video_workflow(args)
        stage_names = [s.__name__ for s in wf.stages]
        gen_idx = stage_names.index("text_to_video_stage")
        strip_idx = stage_names.index("strip_audio_stage")
        log_idx = stage_names.index("log_video_stage")
        assert gen_idx < strip_idx < log_idx

    def test_i2v_with_no_audio_includes_strip_audio_stage(self):
        """I2V workflow with no_audio=True also includes strip_audio_stage."""
        args = Namespace(image_path="/tmp/input.png", no_audio=True)
        wf = build_video_workflow(args)
        stage_names = [s.__name__ for s in wf.stages]
        assert "image_to_video_stage" in stage_names
        assert "strip_audio_stage" in stage_names


# ---------------------------------------------------------------------------
# Combined: upscale + no_audio
# ---------------------------------------------------------------------------


class TestBuildVideoWorkflowCombined:
    """Verify workflow with both upscale=2 and no_audio=True has correct stages."""

    def test_upscale_and_no_audio_t2v(self):
        """T2V workflow with upscale and no_audio has correct stage order."""
        args = Namespace(image_path=None, no_audio=True)
        wf = build_video_workflow(args)
        stage_names = [s.__name__ for s in wf.stages]
        assert stage_names == [
            "resolve_prompt_stage",
            "generate_filename_stage",
            "text_to_video_stage",
            "strip_audio_stage",
            "log_video_stage",
        ]

    def test_upscale_and_no_audio_i2v(self):
        """I2V workflow with upscale and no_audio has correct stage order."""
        args = Namespace(image_path="/tmp/input.png", no_audio=True)
        wf = build_video_workflow(args)
        stage_names = [s.__name__ for s in wf.stages]
        assert stage_names == [
            "resolve_prompt_stage",
            "generate_filename_stage",
            "image_to_video_stage",
            "strip_audio_stage",
            "log_video_stage",
        ]

    def test_upscale_no_audio_false_t2v(self):
        """T2V with upscale but no_audio=False has no strip_audio_stage."""
        args = Namespace(image_path=None, no_audio=False)
        wf = build_video_workflow(args)
        stage_names = [s.__name__ for s in wf.stages]
        assert stage_names == [
            "resolve_prompt_stage",
            "generate_filename_stage",
            "text_to_video_stage",
            "log_video_stage",
        ]

    def test_runner_no_audio_field_propagated(self):
        """run_video_batch propagates no_audio to VideoGenerationRequest."""
        from tests.conftest import _make_mock_video_backend, _make_video_args
        from zvisiongenerator.utils.video_model_detect import VideoModelInfo
        from zvisiongenerator.video_runner import run_video_batch

        backend = _make_mock_video_backend()
        model = MagicMock()
        workflow = MagicMock()
        workflow.run.return_value = StageOutcome.success

        args = _make_video_args(upscale=2, upscale_steps=30, no_audio=True)
        model_info = VideoModelInfo(family="ltx", backend="ltx", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32)
        prompts = {"default": [("test", None)]}
        config: dict = {"generation": {"seed_min": 4, "seed_max": 100}}

        run_video_batch(backend, model, model_info, workflow, prompts, config, args)

        req = workflow.run.call_args[0][0]
        assert req.upscale == 2
        assert req.no_audio is True
        assert req.upscale_steps == 30
