"""Tests for VideoGenerationWorkflow and build_video_workflow()."""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock

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
