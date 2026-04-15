"""Tests for video batch orchestration (run_video_batch)."""

from __future__ import annotations

from unittest.mock import MagicMock

from tests.conftest import _make_mock_video_backend, _make_video_args
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.utils.video_model_detect import VideoModelInfo
from zvisiongenerator.video_runner import run_video_batch


def _ltx_model_info():
    return VideoModelInfo(family="ltx", backend="ltx", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32)


class TestRunVideoBatch:
    """Verify run_video_batch() loop execution and edge cases."""

    def test_single_prompt_single_run(self):
        backend = _make_mock_video_backend()
        model = MagicMock()
        workflow = MagicMock()
        workflow.run.return_value = StageOutcome.success
        args = _make_video_args(runs=1)
        prompts = {"default": [("a sunset", None)]}
        config: dict = {"generation": {"seed_min": 4, "seed_max": 100}}

        run_video_batch(backend, model, _ltx_model_info(), workflow, prompts, config, args)
        assert workflow.run.call_count == 1

    def test_multiple_prompts_multiple_runs(self):
        backend = _make_mock_video_backend()
        model = MagicMock()
        workflow = MagicMock()
        workflow.run.return_value = StageOutcome.success
        args = _make_video_args(runs=2)
        prompts = {"set1": [("prompt A", None), ("prompt B", None)]}
        config: dict = {"generation": {"seed_min": 4, "seed_max": 100}}

        run_video_batch(backend, model, _ltx_model_info(), workflow, prompts, config, args)
        # 2 runs * 2 prompts = 4 iterations
        assert workflow.run.call_count == 4

    def test_multiple_sets(self):
        backend = _make_mock_video_backend()
        model = MagicMock()
        workflow = MagicMock()
        workflow.run.return_value = StageOutcome.success
        args = _make_video_args(runs=1)
        prompts = {"set_a": [("p1", None)], "set_b": [("p2", None), ("p3", None)]}
        config: dict = {}

        run_video_batch(backend, model, _ltx_model_info(), workflow, prompts, config, args)
        assert workflow.run.call_count == 3

    def test_zero_prompts_early_return(self, capsys):
        backend = _make_mock_video_backend()
        model = MagicMock()
        workflow = MagicMock()
        args = _make_video_args(runs=1)
        prompts: dict = {}
        config: dict = {}

        run_video_batch(backend, model, _ltx_model_info(), workflow, prompts, config, args)
        workflow.run.assert_not_called()
        captured = capsys.readouterr()
        assert "No active prompt sets" in captured.out

    def test_failed_generation_not_counted(self, capsys):
        backend = _make_mock_video_backend()
        model = MagicMock()
        workflow = MagicMock()
        workflow.run.return_value = StageOutcome.failed
        args = _make_video_args(runs=1)
        prompts = {"default": [("fail prompt", None)]}
        config: dict = {}

        run_video_batch(backend, model, _ltx_model_info(), workflow, prompts, config, args)
        captured = capsys.readouterr()
        assert "0/1 videos generated" in captured.out
