"""Tests for VideoGenerationWorkflow and build_video_workflow()."""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock, patch

from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.video_types import VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.core.workflow import GenerationWorkflow
from zvisiongenerator.workflows import build_video_workflow
from zvisiongenerator.workflows.video_stages import (
    generate_filename_stage,
    image_to_video_stage,
    log_video_stage,
    resolve_prompt_stage,
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

    def test_runs_all_stages_on_success(self):
        s1 = self._make_stage(StageOutcome.success)
        s2 = self._make_stage(StageOutcome.success)
        s3 = self._make_stage(StageOutcome.success)
        wf = GenerationWorkflow(name="test", stages=[s1, s2, s3])

        req = MagicMock(spec=VideoGenerationRequest)
        arts = MagicMock(spec=VideoWorkingArtifacts)
        result = wf.run(req, arts)

        assert result is StageOutcome.success
        s1.assert_called_once_with(req, arts)
        s2.assert_called_once_with(req, arts)
        s3.assert_called_once_with(req, arts)

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

    def test_empty_stages_returns_success(self):
        wf = GenerationWorkflow(name="empty", stages=[])
        result = wf.run(MagicMock(), MagicMock())
        assert result is StageOutcome.success


# ---------------------------------------------------------------------------
# build_video_workflow()
# ---------------------------------------------------------------------------


class TestBuildVideoWorkflow:
    """Verify build_video_workflow() assembles correct stage lists for T2V and I2V."""

    def test_t2v_workflow_no_image(self):
        args = Namespace(image_path=None)
        wf = build_video_workflow(args)
        assert wf.name == "video"
        assert text_to_video_stage in wf.stages
        assert image_to_video_stage not in wf.stages

    def test_i2v_workflow_with_image(self):
        args = Namespace(image_path="/tmp/input.png")
        wf = build_video_workflow(args)
        assert image_to_video_stage in wf.stages
        assert text_to_video_stage not in wf.stages

    def test_common_stages_present(self):
        args = Namespace(image_path=None)
        wf = build_video_workflow(args)
        assert resolve_prompt_stage in wf.stages
        assert generate_filename_stage in wf.stages
        assert log_video_stage in wf.stages

    def test_stage_order_t2v(self):
        args = Namespace(image_path=None)
        wf = build_video_workflow(args)
        stage_names = [s.__name__ for s in wf.stages]
        assert stage_names.index("resolve_prompt_stage") < stage_names.index("generate_filename_stage")
        assert stage_names.index("generate_filename_stage") < stage_names.index("text_to_video_stage")
        assert stage_names.index("text_to_video_stage") < stage_names.index("log_video_stage")


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

    def test_plain_run_sends_the_prompt_and_records_no_scene_keys(self, tmp_path):
        sent, payload = self._run(tmp_path, prompt="a sunset", resolved_prompt="a sunset")
        assert sent == "a sunset"
        assert payload["prompt"] == "a sunset"
        assert not {"scene", "scene_input", "prompt_input"} & payload.keys()

    def test_scene_is_sent_first_but_recorded_apart(self, tmp_path):
        sent, payload = self._run(tmp_path, prompt="a knight rides in", resolved_prompt="a knight rides in", scene="A castle.", resolved_scene="A castle.")
        assert sent == "A castle. a knight rides in"
        assert payload["scene"] == "A castle." and payload["prompt"] == "a knight rides in"
        assert "scene_input" not in payload and "prompt_input" not in payload

    def test_scene_only_text_run(self, tmp_path):
        sent, payload = self._run(tmp_path, prompt="", resolved_prompt="", scene="A castle at dusk.", resolved_scene="A castle at dusk.")
        assert sent == "A castle at dusk." and payload["scene"] == "A castle at dusk."

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
