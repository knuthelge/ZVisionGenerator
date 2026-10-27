"""Tests for VideoGenerationWorkflow and build_video_workflow()."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests._factories import make_extend_source

from zvisiongenerator.core.progress_events import _stage_label
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.video_types import ExtendSegment, ExtendSource, ExtendSpec, VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.core.workflow import GenerationWorkflow
from zvisiongenerator.workflows import build_video_workflow
from zvisiongenerator.workflows.video_stages import (
    image_to_video_stage,
    strip_audio_stage,
    text_to_video_stage,
)


# ---------------------------------------------------------------------------
# VideoGenerationWorkflow.run()
# ---------------------------------------------------------------------------


class TestVideoGenerationWorkflow:
    """Verify workflow executes stages in order and stops on failure."""

    def _make_stage(self, outcome: StageOutcome):
        """Return a stage callable that records its call and returns outcome."""
        mock = MagicMock(return_value=outcome)
        return mock

    def test_stops_on_failed_stage(self):
        s1 = self._make_stage(StageOutcome.success)
        s2 = self._make_stage(StageOutcome.failed)
        s3 = self._make_stage(StageOutcome.success)
        wf = GenerationWorkflow(name="test", stages=[s1, s2, s3])

        req = MagicMock(spec=VideoGenerationRequest)
        arts = MagicMock(spec=VideoWorkingArtifacts)
        result = wf.run(req, arts)

        assert result is StageOutcome.failed
        s1.assert_called_once()
        s2.assert_called_once()
        s3.assert_not_called()

    def test_stops_on_skipped_stage(self):
        s1 = self._make_stage(StageOutcome.success)
        s2 = self._make_stage(StageOutcome.skipped)
        s3 = self._make_stage(StageOutcome.success)
        wf = GenerationWorkflow(name="test", stages=[s1, s2, s3])

        result = wf.run(MagicMock(), MagicMock())
        assert result is StageOutcome.skipped
        s3.assert_not_called()


# ---------------------------------------------------------------------------
# build_video_workflow()
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Scene through the built workflow
# ---------------------------------------------------------------------------


class TestSceneThroughTheWorkflow:
    """Run the real stages with a mocked backend and check the prompt sent and the config embedded."""

    def _run(self, tmp_path, *, enhance=False, image_path=None, **request_overrides):
        backend = MagicMock()
        backend.text_to_video.return_value = tmp_path / "out.mp4"
        backend.image_to_video.return_value = tmp_path / "out.mp4"
        request = VideoGenerationRequest(backend=backend, model=MagicMock(), image_path=image_path, output_dir=str(tmp_path), **request_overrides)
        artifacts = VideoWorkingArtifacts()
        workflow = build_video_workflow(Namespace(image_path=image_path, no_audio=False), enhance=enhance)
        with patch("zvisiongenerator.workflows.video_stages.embed_mp4_config") as embed:
            assert workflow.run(request, artifacts) is StageOutcome.success
        sent = (backend.image_to_video if image_path else backend.text_to_video).call_args.kwargs["prompt"]
        return sent, embed.call_args[0][1]

    def test_scene_only_image_run(self, tmp_path):
        sent, payload = self._run(tmp_path, image_path=str(tmp_path / "ref.png"), prompt="", resolved_prompt="", scene="A castle at dusk.", resolved_scene="A castle at dusk.")
        assert sent == "A castle at dusk." and payload["workflow"] == "img2vid"

    def test_enhanced_pair_is_composed_for_the_model_and_split_in_the_record(self, tmp_path):
        sent, payload = self._run(
            tmp_path,
            enhance=True,
            prompt="a knight rides in",
            resolved_prompt="a knight rides in",
            scene="A castle.",
            resolved_scene="A castle.",
            enhanced_prompt="A knight rides in at a gallop.",
            enhanced_scene="A grey castle at dusk, soft light.",
        )
        assert sent == "A grey castle at dusk, soft light. A knight rides in at a gallop."
        assert payload["prompt"] == "A knight rides in at a gallop."
        assert payload["scene"] == "A grey castle at dusk, soft light."
        assert payload["prompt_input"] == "a knight rides in" and payload["scene_input"] == "A castle."
        assert "castle" not in payload["prompt"]


# ---------------------------------------------------------------------------
# Extend workflow
# ---------------------------------------------------------------------------


def _extend_source(tmp_path: Path, **overrides) -> ExtendSource:
    """Build a checked 24 fps source with sound, recorded with a scene and the action that followed it."""
    return make_extend_source(**{"path": tmp_path / "clip.mp4", **overrides})


def _stage_names(workflow: GenerationWorkflow) -> list[str]:
    return [_stage_label(stage) for stage in workflow.stages]


class TestBuildExtendWorkflow:
    """Verify the extend branch of build_video_workflow(): its stages, their order and the progress labels."""

    def _args(self, tmp_path, **overrides):
        return Namespace(image_path=None, extend_source=_extend_source(tmp_path), extend_frames=48, **overrides)

    def test_stage_order(self, tmp_path):
        workflow = build_video_workflow(self._args(tmp_path))
        assert workflow.name == "video"
        assert _stage_names(workflow) == ["resolve_prompt", "generate_filename", "extend_video", "join_clips", "log_video"]

    def test_enhance_adds_the_prompt_rewrite_after_resolving(self, tmp_path):
        workflow = build_video_workflow(self._args(tmp_path), enhance=True)
        assert _stage_names(workflow) == ["resolve_prompt", "enhance_prompt", "generate_filename", "extend_video", "join_clips", "log_video"]

    @pytest.mark.parametrize("enhance", [False, True])
    def test_never_strips_audio_even_with_no_audio(self, tmp_path, enhance):
        workflow = build_video_workflow(self._args(tmp_path, no_audio=True), enhance=enhance)
        assert strip_audio_stage not in workflow.stages
        assert "strip_audio" not in _stage_names(workflow)

    def test_has_no_generation_stage_of_the_other_workflows(self, tmp_path):
        workflow = build_video_workflow(self._args(tmp_path))
        assert text_to_video_stage not in workflow.stages and image_to_video_stage not in workflow.stages

    def test_other_workflows_are_unchanged(self):
        assert _stage_names(build_video_workflow(Namespace(image_path=None))) == ["resolve_prompt", "generate_filename", "text_to_video", "log_video"]
        assert _stage_names(build_video_workflow(Namespace(image_path="/tmp/a.png", no_audio=True), enhance=True)) == [
            "resolve_prompt",
            "enhance_prompt",
            "generate_filename",
            "image_to_video",
            "strip_audio",
            "log_video",
        ]

    def test_a_source_of_none_is_not_an_extension(self):
        workflow = build_video_workflow(Namespace(image_path=None, extend_source=None, extend_frames=None))
        assert text_to_video_stage in workflow.stages


class TestExtendThroughTheWorkflow:
    """Run the real extend stages with a mocked backend and splice, and check the data flow between them."""

    def _run(self, tmp_path, *, source=None, enhance=False, **request_overrides):
        source = source or _extend_source(tmp_path)
        backend = MagicMock()
        backend.extend_video.side_effect = lambda **kw: ExtendSegment(Path(kw["segment_dir"]) / "seg.mp4", Path(kw["segment_dir"]) / "seg.wav")
        request = VideoGenerationRequest(
            backend=backend,
            model=MagicMock(),
            width=source.width,
            height=source.height,
            num_frames=25 + 48,
            output_dir=str(tmp_path),
            extend=ExtendSpec(source, 48),
            **{"prompt": "", **request_overrides},
        )
        artifacts = VideoWorkingArtifacts()
        workflow = build_video_workflow(Namespace(image_path=None, extend_source=source, extend_frames=48, no_audio=request.no_audio), enhance=enhance)
        with patch("zvisiongenerator.workflows.video_stages.splice_clips", side_effect=lambda src, seg, out, **kw: Path(out)) as splice:
            with patch("zvisiongenerator.workflows.video_stages.embed_mp4_config") as embed:
                outcome = workflow.run(request, artifacts)
        return outcome, backend, splice, embed, artifacts

    def test_segment_flows_from_the_backend_to_the_splice_and_the_record(self, tmp_path):
        outcome, backend, splice, embed, artifacts = self._run(tmp_path, prompt="She walks away.", scene="A street at night.")
        assert outcome is StageOutcome.success
        sent = backend.extend_video.call_args.kwargs
        assert sent["prompt"] == "A street at night. She walks away."
        assert splice.call_args.args[1].video_path.name == "seg.mp4"
        assert splice.call_args.args[2] == artifacts.video_path
        assert splice.call_args.kwargs == {"extend_frames": 48, "include_audio": True}
        payload = embed.call_args[0][1]
        assert (payload["workflow"], payload["scene"], payload["prompt"]) == ("extend", "A street at night.", "She walks away.")
        assert payload["frame_count"] == 97 and payload["extend_frames"] == 48
        assert not Path(sent["segment_dir"]).exists()

    def test_no_audio_reaches_both_the_backend_and_the_splice(self, tmp_path):
        _, backend, splice, *_ = self._run(tmp_path, no_audio=True)
        assert backend.extend_video.call_args.kwargs["audio_output"] is False
        assert splice.call_args.kwargs["include_audio"] is False

    def test_silent_source_asks_for_no_audio(self, tmp_path):
        _, backend, splice, *_ = self._run(tmp_path, source=_extend_source(tmp_path, has_audio=False))
        assert backend.extend_video.call_args.kwargs["audio_output"] is False
        assert splice.call_args.kwargs["include_audio"] is False

    def test_enhanced_action_is_what_the_backend_receives_and_what_is_recorded(self, tmp_path):
        _, backend, _, embed, _ = self._run(
            tmp_path,
            enhance=True,
            prompt="she leaves",
            scene="A street at night.",
            resolved_prompt="she leaves",
            resolved_scene="A street at night.",
            enhanced_prompt="She turns and walks into the rain.",
            enhanced_scene="A street at night.",
        )
        assert backend.extend_video.call_args.kwargs["prompt"] == "A street at night. She turns and walks into the rain."
        payload = embed.call_args[0][1]
        assert payload["prompt"] == "She turns and walks into the rain." and payload["scene"] == "A street at night."
        assert payload["prompt_input"] == "she leaves"

    def test_failed_generation_stops_before_the_splice(self, tmp_path):
        source = _extend_source(tmp_path)
        backend = MagicMock()
        backend.extend_video.return_value = None
        request = VideoGenerationRequest(backend=backend, model=MagicMock(), prompt="", output_dir=str(tmp_path), extend=ExtendSpec(source, 48))
        workflow = build_video_workflow(Namespace(image_path=None, extend_source=source, extend_frames=48))
        with patch("zvisiongenerator.workflows.video_stages.splice_clips") as splice:
            assert workflow.run(request, VideoWorkingArtifacts()) is StageOutcome.failed
        splice.assert_not_called()
