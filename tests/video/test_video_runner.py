"""Tests for video batch orchestration (run_video_batch)."""

from __future__ import annotations


import threading
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests._factories import _make_plan, make_extend_source
from tests._factories import _make_mock_video_backend, _make_video_args
from tests.video._helpers import make_ltx_info

from zvisiongenerator.core.job_plan import EnhanceStatus, JobPlan
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.video_types import ExtendSpec
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.utils.prompt_enhance import EnhanceSettings
from zvisiongenerator.video_runner import run_video_batch


class TestRunVideoBatch:
    """Verify run_video_batch() loop execution and edge cases."""

    @pytest.mark.parametrize(
        ("runs", "prompts", "expected"),
        [
            (1, {"default": [("a sunset", None)]}, 1),
            (2, {"set1": [("prompt A", None), ("prompt B", None)]}, 4),
            (1, {"set_a": [("p1", None)], "set_b": [("p2", None), ("p3", None)]}, 3),
        ],
    )
    def test_every_prompt_of_every_set_runs_once_per_run(self, runs, prompts, expected):
        workflow = MagicMock()
        workflow.run.return_value = StageOutcome.success
        args = _make_video_args(runs=runs)
        config: dict = {"generation": {"seed_min": 4, "seed_max": 100}}

        run_video_batch(_make_mock_video_backend(), MagicMock(), make_ltx_info(), workflow, prompts, config, args, plan=_make_plan(prompts, args, config=config))

        assert workflow.run.call_count == expected

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({"upscale": 2, "upscale_steps": 30, "no_audio": True}, (2, 30, True)),
            ({"upscale": None, "upscale_steps": None, "no_audio": False}, (None, None, False)),
        ],
    )
    def test_the_upscale_and_audio_settings_reach_the_request(self, overrides, expected):
        workflow = MagicMock()
        workflow.run.return_value = StageOutcome.success
        args = _make_video_args(**overrides)
        prompts = {"default": [("a sunset", None)]}
        config: dict = {"generation": {"seed_min": 4, "seed_max": 100}}

        run_video_batch(_make_mock_video_backend(), MagicMock(), make_ltx_info(), workflow, prompts, config, args, plan=_make_plan(prompts, args, config=config))

        request = workflow.run.call_args.args[0]
        assert (request.upscale, request.upscale_steps, request.no_audio) == expected

    def test_zero_prompts_early_return(self):
        backend = _make_mock_video_backend()
        model = MagicMock()
        workflow = MagicMock()
        events: list[dict[str, object]] = []
        args = _make_video_args(runs=1)
        prompts: dict = {}
        config: dict = {}

        run_video_batch(backend, model, make_ltx_info(), workflow, prompts, config, args, progress_callback=events.append, plan=_make_plan(prompts, args, config=config))
        workflow.run.assert_not_called()

        assert events == [
            {
                "type": "batch_completed",
                "mode": "video",
                "total_iterations": 0,
                "completed_iterations": 0,
            }
        ]

    def test_failed_generation_emits_failed_terminal_batch_event(self):
        backend = _make_mock_video_backend()
        model = MagicMock()
        workflow = MagicMock()
        workflow.run.return_value = StageOutcome.failed
        events: list[dict[str, object]] = []
        args = _make_video_args(runs=1)
        prompts = {"default": [("fail prompt", None)]}
        config: dict = {}

        run_video_batch(backend, model, make_ltx_info(), workflow, prompts, config, args, progress_callback=events.append, plan=_make_plan(prompts, args, config=config))

        request = workflow.run.call_args.args[0]

        finished_events = [event for event in events if event["type"] == "generation_finished"]
        batch_events = [event for event in events if event["type"] == "batch_failed"]

        assert request.backend is backend
        assert request.model is model
        assert request.prompt == "fail prompt"
        assert request.seed == args.seed

        assert finished_events == [
            {
                "type": "generation_finished",
                "mode": "video",
                "run_index": 0,
                "total_runs": 1,
                "ran_iterations": 1,
                "total_iterations": 1,
                "set_name": "default",
                "prompt_index": 0,
                "total_prompts": 1,
                "prompt": "fail prompt",
                "seed": args.seed,
                "status": "failed",
                "message": "Video generation failed.",
            }
        ]
        assert len(batch_events) == 1
        assert batch_events[0]["message"] == "Video generation failed."
        assert not [event for event in events if event["type"] == "batch_completed"]
        assert batch_events[0]["mode"] == "video"
        assert batch_events[0]["completed_iterations"] == 1
        assert batch_events[0]["total_iterations"] == 1
        assert isinstance(batch_events[0]["total_time"], float)

    def test_step_progress_includes_video_generation_context(self):
        backend = _make_mock_video_backend()
        model = MagicMock()
        events: list[dict[str, object]] = []
        captured_requests = []

        def _emit_step_then_succeed(request, artifacts):
            captured_requests.append(request)
            assert request.step_callback is not None
            request.step_callback({"current_step": 3, "total_steps": request.steps, "phase": "video_generate"})
            artifacts.generation_time = 0.25
            artifacts.filename = "clip.mp4"
            return StageOutcome.success

        workflow = MagicMock()
        workflow.run.side_effect = _emit_step_then_succeed
        args = _make_video_args(runs=1, steps=8)
        prompts = {"default": [("a sunset", None)]}
        config: dict = {"generation": {"seed_min": 4, "seed_max": 100}}

        run_video_batch(backend, model, make_ltx_info(), workflow, prompts, config, args, progress_callback=events.append, plan=_make_plan(prompts, args, config=config))

        step_events = [event for event in events if event["type"] == "step_progress"]

        assert len(captured_requests) == 1
        assert captured_requests[0].backend is backend
        assert captured_requests[0].model is model
        assert captured_requests[0].prompt == "a sunset"
        assert captured_requests[0].seed == args.seed

        assert len(step_events) == 1
        assert step_events[0]["mode"] == "video"
        assert step_events[0]["phase"] == "video_generate"
        assert step_events[0]["current_step"] == 3
        assert step_events[0]["total_steps"] == 8
        assert step_events[0]["run_index"] == 0
        assert step_events[0]["total_runs"] == 1
        assert step_events[0]["ran_iterations"] == 1
        assert step_events[0]["total_iterations"] == 1
        assert step_events[0]["set_name"] == "default"
        assert step_events[0]["prompt_index"] == 0
        assert step_events[0]["total_prompts"] == 1

    def test_success_emits_generation_finished_with_shared_payload(self):
        backend = _make_mock_video_backend()
        model = MagicMock()
        events: list[dict[str, object]] = []
        captured_requests = []

        def _succeed_with_artifacts(request, artifacts):
            captured_requests.append(request)
            artifacts.generation_time = 0.25
            artifacts.filename = "clip.mp4"
            artifacts.video_path = "/tmp/clip.mp4"
            return StageOutcome.success

        workflow = MagicMock()
        workflow.run.side_effect = _succeed_with_artifacts
        args = _make_video_args(runs=1, steps=8, seed=77)
        prompts = {"default": [("a sunset", None)]}
        config: dict = {"generation": {"seed_min": 4, "seed_max": 100}}

        run_video_batch(backend, model, make_ltx_info(), workflow, prompts, config, args, progress_callback=events.append, plan=_make_plan(prompts, args, config=config))

        finished_events = [event for event in events if event["type"] == "generation_finished"]

        assert len(captured_requests) == 1
        assert captured_requests[0].backend is backend
        assert captured_requests[0].model is model
        assert captured_requests[0].prompt == "a sunset"
        assert captured_requests[0].seed == 77

        assert len(finished_events) == 1
        assert finished_events[0]["mode"] == "video"
        assert finished_events[0]["status"] == "success"
        assert finished_events[0]["run_index"] == 0
        assert finished_events[0]["total_runs"] == 1
        assert finished_events[0]["ran_iterations"] == 1
        assert finished_events[0]["total_iterations"] == 1
        assert finished_events[0]["set_name"] == "default"
        assert finished_events[0]["prompt_index"] == 0
        assert finished_events[0]["total_prompts"] == 1
        assert finished_events[0]["prompt"] == "a sunset"
        assert finished_events[0]["seed"] == 77
        assert captured_requests[0].prompt == finished_events[0]["prompt"]
        assert captured_requests[0].seed == finished_events[0]["seed"]
        assert finished_events[0]["filename"] == "clip.mp4"
        assert finished_events[0]["output_path"] == "/tmp/clip.mp4"
        assert finished_events[0]["generation_time"] == 0.25


@pytest.mark.filterwarnings("ignore:Video generation failed")
class TestVideoFailureMessages:
    """Failed generations and a failed batch carry a one-line reason."""

    @staticmethod
    def _run(workflow, *, runs=1, prompts=None):
        events: list[dict[str, object]] = []
        args = _make_video_args(runs=runs)
        prompts = prompts or {"default": [("p", None)]}
        config: dict = {}
        run_video_batch(
            _make_mock_video_backend(),
            MagicMock(),
            make_ltx_info(),
            workflow,
            prompts,
            config,
            args,
            progress_callback=events.append,
            plan=_make_plan(prompts, args, config=config),
        )
        return events

    def test_exception_message_is_one_line_on_finished_and_batch_failed(self):
        workflow = MagicMock()
        workflow.run.side_effect = RuntimeError("CUDA out of memory\n  while allocating\ttensor")

        events = self._run(workflow)

        (finished,) = [e for e in events if e["type"] == "generation_finished"]
        (batch,) = [e for e in events if e["type"] == "batch_failed"]
        assert finished["status"] == "failed"
        assert finished["message"] == "CUDA out of memory while allocating tensor"
        assert batch["message"] == finished["message"]

    def test_exception_without_text_falls_back_to_its_type_name(self):
        workflow = MagicMock()
        workflow.run.side_effect = ValueError()

        events = self._run(workflow)

        (finished,) = [e for e in events if e["type"] == "generation_finished"]
        assert finished["message"] == "ValueError"

    def test_stage_failure_uses_the_message_the_stage_set(self):
        def _fail(request, artifacts):
            artifacts.failure_message = "Extend produced no video segment."
            return StageOutcome.failed

        workflow = MagicMock()
        workflow.run.side_effect = _fail

        events = self._run(workflow)

        (finished,) = [e for e in events if e["type"] == "generation_finished"]
        (batch,) = [e for e in events if e["type"] == "batch_failed"]
        assert finished["message"] == "Extend produced no video segment."
        assert batch["message"] == "Extend produced no video segment."

    def test_batch_failed_carries_the_last_failed_generations_message(self):
        messages = iter(["first reason", "second reason"])

        def _fail(request, artifacts):
            artifacts.failure_message = next(messages)
            return StageOutcome.failed

        workflow = MagicMock()
        workflow.run.side_effect = _fail

        events = self._run(workflow, runs=2)

        finished = [e for e in events if e["type"] == "generation_finished"]
        (batch,) = [e for e in events if e["type"] == "batch_failed"]
        assert [e["message"] for e in finished] == ["first reason", "second reason"]
        assert batch["message"] == "second reason"

    def test_success_and_completed_batch_have_no_message(self):
        def _succeed(request, artifacts):
            artifacts.filename = "clip.mp4"
            return StageOutcome.success

        workflow = MagicMock()
        workflow.run.side_effect = _succeed

        events = self._run(workflow)

        for event in events:
            if event["type"] in {"generation_finished", "batch_completed"}:
                assert "message" not in event


class TestVideoSceneRequests:
    """The runner hands each iteration's planned scene (and the job's recorded inputs) to its request."""

    def _requests(self, prompts, *, plan_scene=None, scene_by_set=None, plan_transform=None, **arg_overrides):
        captured: list = []
        workflow = MagicMock()
        workflow.run.side_effect = lambda request, artifacts: captured.append(request) or StageOutcome.success
        args = _make_video_args(runs=1, **arg_overrides)
        plan = _make_plan(prompts, args, config={}, scene=plan_scene, scene_by_set=scene_by_set)
        if plan_transform is not None:
            plan = plan_transform(plan)
        run_video_batch(_make_mock_video_backend(), MagicMock(), make_ltx_info(), workflow, prompts, {}, args, plan=plan)
        return captured

    def test_plain_job_has_no_scene_fields(self):
        (request,) = self._requests({"s": [("a sunset", None)]})
        assert (request.scene, request.resolved_scene, request.enhanced_scene, request.scene_input, request.prompt_input) == (None,) * 5

    def test_job_scene_reaches_every_request(self):
        requests = self._requests({"s": [("a", None), ("b", None)]}, plan_scene="A {castle|castle}.")
        assert [(r.scene, r.resolved_scene) for r in requests] == [("A {castle|castle}.", "A castle.")] * 2

    def test_entry_scene_beats_the_job_scene(self):
        requests = self._requests({"s": [("a", None), ("b", None)]}, plan_scene="Job.", scene_by_set={"s": ["Entry.", None]})
        assert [r.scene for r in requests] == ["Entry.", "Job."]

    def test_scene_only_job_runs_with_an_empty_prompt(self):
        (request,) = self._requests({"prompt": [("", None)]}, plan_scene="A castle at dusk.")
        assert request.prompt == "" and request.resolved_prompt == "" and request.resolved_scene == "A castle at dusk."

    def test_enhanced_pair_reaches_the_request(self):
        def _enhance(plan):
            return JobPlan(
                iterations=tuple(
                    replace(it, enhance=EnhanceSettings(), enhanced_prompt="A knight rides in.", enhanced_scene="A grey castle.", enhance_status=EnhanceStatus.ENHANCED) for it in plan.iterations
                )
            )

        (request,) = self._requests({"s": [("a knight", None)]}, plan_scene="A castle.", plan_transform=_enhance)
        assert (request.enhanced_prompt, request.enhanced_scene) == ("A knight rides in.", "A grey castle.")
        assert request.scene == "A castle." and request.prompt == "a knight"

    def test_recorded_inputs_come_from_args(self):
        (request,) = self._requests({"s": [("A knight rides in.", None)]}, scene_input="A castle.", prompt_input="a knight rides in")
        assert request.scene_input == "A castle." and request.prompt_input == "a knight rides in"


class TestVideoSceneEndToEnd:
    """Run the real stages behind ``run_video_batch`` and check the one composed prompt sent and the config recorded."""

    def _run(self, tmp_path, prompts, *, enhance=False, plan_transform=None, scene=None, scene_by_set=None):
        from argparse import Namespace
        from unittest.mock import patch

        from zvisiongenerator.workflows import build_video_workflow

        backend = _make_mock_video_backend()
        backend.text_to_video.return_value = tmp_path / "out.mp4"
        args = _make_video_args(runs=1, output=str(tmp_path))
        plan = _make_plan(prompts, args, config={}, scene=scene, scene_by_set=scene_by_set)
        if plan_transform is not None:
            plan = plan_transform(plan)
        workflow = build_video_workflow(Namespace(image_path=None, no_audio=False), enhance=enhance)
        with patch("zvisiongenerator.workflows.video_stages.embed_mp4_config") as embed:
            run_video_batch(backend, MagicMock(), make_ltx_info(), workflow, prompts, {}, args, plan=plan)
        sent = [call.kwargs["prompt"] for call in backend.text_to_video.call_args_list]
        return sent, [call.args[1] for call in embed.call_args_list]

    def test_entry_scene_beats_job_scene_and_is_sent_once(self, tmp_path):
        sent, payloads = self._run(tmp_path, {"s": [("a knight rides in", None), ("a dog runs", None)]}, scene="Job.", scene_by_set={"s": ["Entry.", None]})
        assert sent == ["Entry. a knight rides in", "Job. a dog runs"]
        assert [p["scene"] for p in payloads] == ["Entry.", "Job."]
        assert [p["prompt"] for p in payloads] == ["a knight rides in", "a dog runs"]

    def test_enhanced_pair_is_sent_once_and_recorded_apart(self, tmp_path):
        def _enhance(plan):
            return JobPlan(
                iterations=tuple(
                    replace(it, enhance=EnhanceSettings(), enhanced_prompt="A knight gallops in.", enhanced_scene="A grey castle.", enhance_status=EnhanceStatus.ENHANCED) for it in plan.iterations
                )
            )

        sent, (payload,) = self._run(tmp_path, {"s": [("a knight", None)]}, enhance=True, plan_transform=_enhance, scene="A castle.")
        assert sent == ["A grey castle. A knight gallops in."]
        assert sent[0].count("A grey castle.") == 1 and "A castle." not in sent[0]
        assert (payload["scene"], payload["prompt"]) == ("A grey castle.", "A knight gallops in.")
        assert (payload["scene_input"], payload["prompt_input"]) == ("A castle.", "a knight")

    def test_plain_run_records_no_scene_keys(self, tmp_path):
        sent, (payload,) = self._run(tmp_path, {"s": [("a sunset", None)]})
        assert sent == ["a sunset"]
        assert not {"scene", "scene_input", "prompt_input"} & payload.keys()


class TestVideoControls:
    """Verify the queued controls (Next, Pause, Repeat, Quit) act on a video batch as they do on an image batch."""

    @staticmethod
    def _run(workflow, *, runs: int, skip_signal, on_event=None, **arg_overrides):
        events: list[dict[str, object]] = []

        def _record(event):
            events.append(event)
            if on_event is not None:
                on_event(event)

        args = _make_video_args(runs=runs, steps=8, **arg_overrides)
        prompts = {"default": [("a sunset", None)]}
        config: dict = {"generation": {"seed_min": 4, "seed_max": 100}}
        run_video_batch(
            _make_mock_video_backend(),
            MagicMock(),
            make_ltx_info(),
            workflow,
            prompts,
            config,
            args,
            progress_callback=_record,
            plan=_make_plan(prompts, args, config=config),
            skip_signal=skip_signal,
        )
        return events

    @staticmethod
    def _resume_once_waiting(skip_signal) -> None:
        """Resume *skip_signal* from a thread as soon as it waits for a key."""

        def _resume():
            while not skip_signal.is_waiting_for_resume():
                time.sleep(0.001)
            skip_signal.resume()

        threading.Thread(target=_resume, daemon=True).start()

    def test_quit_queued_before_a_video_stops_the_batch_without_running_it(self):
        skip_signal = SkipSignal()
        skip_signal.queue_action("quit")
        workflow = MagicMock()

        events = self._run(workflow, runs=2, skip_signal=skip_signal)

        workflow.run.assert_not_called()
        assert events[-1] == {"type": "batch_cancelled", "mode": "video", "completed_iterations": 0, "total_iterations": 2}

    def test_quit_stops_the_running_video_at_its_next_step_and_skips_the_rest(self):
        skip_signal = SkipSignal()
        steps_seen: list[int] = []

        def _denoise(request, artifacts):
            for step in range(1, 9):
                if step == 3:
                    skip_signal.queue_action("quit")
                request.step_callback({"current_step": step, "total_steps": 8})
                steps_seen.append(step)
            return StageOutcome.success

        workflow = MagicMock()
        workflow.run.side_effect = _denoise

        events = self._run(workflow, runs=3, skip_signal=skip_signal)

        assert steps_seen == [1, 2]
        assert workflow.run.call_count == 1
        assert not [event for event in events if event["type"] == "generation_finished"]
        assert events[-1]["type"] == "batch_cancelled"
        assert events[-1]["completed_iterations"] == 0

    def test_quit_after_a_finished_video_cancels_with_it_counted(self):
        skip_signal = SkipSignal()
        workflow = MagicMock()
        workflow.run.side_effect = lambda request, artifacts: skip_signal.queue_action("quit") or StageOutcome.success

        events = self._run(workflow, runs=2, skip_signal=skip_signal)

        assert workflow.run.call_count == 1
        assert [e["status"] for e in events if e["type"] == "generation_finished"] == ["success"]
        assert events[-1] == {"type": "batch_cancelled", "mode": "video", "completed_iterations": 1, "total_iterations": 2}

    def test_next_stops_the_running_video_and_the_batch_goes_on(self):
        skip_signal = SkipSignal()
        calls = 0

        def _denoise(request, artifacts):
            nonlocal calls
            calls += 1
            for step in range(1, 5):
                if calls == 1 and step == 2:
                    skip_signal.queue_action("skip")
                    time.sleep(0.01)
                request.step_callback({"current_step": step, "total_steps": 4})
            artifacts.video_path = "/out/clip.mp4"
            return StageOutcome.success

        workflow = MagicMock()
        workflow.run.side_effect = _denoise

        events = self._run(workflow, runs=2, skip_signal=skip_signal)

        finished = [e for e in events if e["type"] == "generation_finished"]
        assert [e["status"] for e in finished] == ["skipped", "success"]
        assert finished[0]["generation_time"] > 0
        assert not finished[0].get("output_path")
        assert finished[1]["output_path"] == "/out/clip.mp4"
        assert events[-1]["type"] == "batch_completed"
        assert events[-1]["completed_iterations"] == 2
        assert "batch_cancelled" not in [e["type"] for e in events]

    def test_a_next_before_a_video_or_on_its_last_step_is_dropped_and_the_video_stays_saved(self):
        skip_signal = SkipSignal()
        skip_signal.queue_action("skip")

        def _denoise(request, artifacts):
            request.step_callback({"current_step": 1, "total_steps": 2})
            skip_signal.queue_action("skip")
            request.step_callback({"current_step": 2, "total_steps": 2})
            return StageOutcome.success

        workflow = MagicMock()
        workflow.run.side_effect = _denoise

        events = self._run(workflow, runs=2, skip_signal=skip_signal)

        assert [e["status"] for e in events if e["type"] == "generation_finished"] == ["success", "success"]
        assert events[-1]["type"] == "batch_completed"

    def test_a_quit_on_the_last_step_saves_the_video_and_then_cancels(self):
        skip_signal = SkipSignal()

        def _denoise(request, artifacts):
            request.step_callback({"current_step": 0, "total_steps": 1})
            skip_signal.queue_action("quit")
            request.step_callback({"current_step": 1, "total_steps": 1})
            artifacts.video_path = "/out/clip.mp4"
            return StageOutcome.success

        workflow = MagicMock()
        workflow.run.side_effect = _denoise

        events = self._run(workflow, runs=2, skip_signal=skip_signal)

        assert [e["status"] for e in events if e["type"] == "generation_finished"] == ["success"]
        assert workflow.run.call_count == 1
        assert events[-1] == {"type": "batch_cancelled", "mode": "video", "completed_iterations": 1, "total_iterations": 2}

    def test_a_next_at_the_end_of_the_first_stage_of_a_two_stage_render_still_stops_it(self):
        skip_signal = SkipSignal()
        steps_seen: list[int] = []

        def _denoise(request, artifacts):
            # Two stages report summed totals (4 + 3), so the end of stage 1 is not the end of the render.
            for step in range(1, 8):
                if step == 4:
                    skip_signal.queue_action("skip")
                request.step_callback({"current_step": step, "total_steps": 7})
                steps_seen.append(step)
            return StageOutcome.success

        workflow = MagicMock()
        workflow.run.side_effect = _denoise

        events = self._run(workflow, runs=1, skip_signal=skip_signal)

        assert steps_seen == [1, 2, 3]
        assert [e["status"] for e in events if e["type"] == "generation_finished"] == ["skipped"]

    def test_pause_queued_before_and_during_a_video_waits_before_and_after_it(self):
        skip_signal = SkipSignal()
        skip_signal.queue_action("pause")
        workflow = MagicMock()
        workflow.run.side_effect = lambda request, artifacts: skip_signal.queue_action("pause") or StageOutcome.success

        def _on_event(event):
            if event["type"] == "job_paused":
                self._resume_once_waiting(skip_signal)

        events = self._run(workflow, runs=1, skip_signal=skip_signal, on_event=_on_event)

        types = [e["type"] for e in events]
        assert types.count("job_paused") == types.count("job_resumed") == 2
        assert {e["mode"] for e in events if e["type"] in ("job_paused", "job_resumed")} == {"video"}
        assert types.index("job_paused") < types.index("job_resumed") < types.index("generation_started")
        assert types.index("generation_finished") < len(types) - 1 - types[::-1].index("job_paused")
        assert workflow.run.call_count == 1
        assert types[-1] == "batch_completed"

    def test_quit_while_paused_cancels_and_no_second_video_starts(self):
        skip_signal = SkipSignal()
        workflow = MagicMock()
        workflow.run.side_effect = lambda request, artifacts: skip_signal.queue_action("pause") or StageOutcome.success

        def _on_event(event):
            if event["type"] == "job_paused":
                skip_signal.queue_action("quit")
                skip_signal.resume()

        events = self._run(workflow, runs=2, skip_signal=skip_signal, on_event=_on_event)

        assert workflow.run.call_count == 1
        assert events[-1] == {"type": "batch_cancelled", "mode": "video", "completed_iterations": 1, "total_iterations": 2}

    def test_repeat_after_a_video_renders_the_same_prompt_again_with_a_new_seed(self):
        skip_signal = SkipSignal()
        requests = []

        def _generate(request, artifacts):
            requests.append(request)
            if len(requests) == 1:
                skip_signal.queue_action("repeat")
            return StageOutcome.success

        workflow = MagicMock()
        workflow.run.side_effect = _generate

        with patch("zvisiongenerator.job_controls.random.randint", return_value=77):
            events = self._run(workflow, runs=2, skip_signal=skip_signal)

        types = [e["type"] for e in events]
        assert len(requests) == 3
        assert [r.prompt for r in requests] == ["a sunset"] * 3
        assert requests[1].seed == 77
        assert requests[1].resolved_prompt == requests[0].resolved_prompt
        assert types.count("prompt_started") == 2
        assert types.count("generation_started") == 3
        repeat = types.index("generation_repeat")
        assert [e["seed"] for e in events if e["type"] == "generation_started"][1] == 77
        assert [e["seed"] for e in events if e["type"] == "generation_finished"][1] == 77
        assert types.index("generation_finished") < repeat < types.index("generation_started", repeat)
        assert events[-1]["type"] == "batch_completed"
        assert events[-1]["completed_iterations"] == 3

    def test_pause_after_a_failed_video_is_honoured_and_repeat_is_dropped(self):
        for action, paused in (("pause", True), ("repeat", False)):
            skip_signal = SkipSignal()
            workflow = MagicMock()
            workflow.run.side_effect = lambda request, artifacts, action=action, skip_signal=skip_signal: skip_signal.queue_action(action) or StageOutcome.failed

            def _on_event(event, skip_signal=skip_signal):
                if event["type"] == "job_paused":
                    self._resume_once_waiting(skip_signal)

            events = self._run(workflow, runs=2, skip_signal=skip_signal, on_event=_on_event)

            types = [e["type"] for e in events]
            assert workflow.run.call_count == 2
            assert ("job_paused" in types) is paused
            assert "generation_repeat" not in types

    def test_a_next_stopped_video_stays_out_of_the_eta_and_a_batch_with_no_success_and_a_failure_fails(self):
        skip_signal = SkipSignal()
        calls = 0

        def _denoise(request, artifacts):
            nonlocal calls
            calls += 1
            if calls == 1:
                skip_signal.queue_action("skip")
                time.sleep(0.01)
                request.step_callback({"current_step": 1, "total_steps": 4})
            return StageOutcome.failed

        workflow = MagicMock()
        workflow.run.side_effect = _denoise

        events = self._run(workflow, runs=2, skip_signal=skip_signal)

        assert [e["status"] for e in events if e["type"] == "generation_finished"] == ["skipped", "failed"]
        assert [e["avg_secs"] for e in events if e["type"] == "prompt_started"] == [None, None]
        assert events[-1]["type"] == "batch_failed"

    def test_repeats_keep_the_eta_positive_and_the_final_counts_within_the_total(self):
        skip_signal = SkipSignal()
        calls = 0

        def _generate(request, artifacts):
            nonlocal calls
            calls += 1
            time.sleep(0.001)
            if calls <= 2:
                skip_signal.queue_action("repeat")
            return StageOutcome.success

        workflow = MagicMock()
        workflow.run.side_effect = _generate

        events = self._run(workflow, runs=2, skip_signal=skip_signal)

        second = [e for e in events if e["type"] == "prompt_started"][1]
        assert second["eta_secs"] == pytest.approx(second["avg_secs"])
        assert {key: events[-1][key] for key in ("type", "completed_iterations", "total_iterations")} == {"type": "batch_completed", "completed_iterations": 4, "total_iterations": 4}

    def test_a_repeat_after_a_discarded_review_is_a_new_attempt_of_the_same_iteration(self):
        skip_signal = SkipSignal()
        decisions = iter(["discard", "approve"])

        def _render(request, artifacts):
            request.review(Path("small.mp4"))
            return StageOutcome.success

        def _on_event(event):
            if event["type"] == "job_paused" and "review" in event:
                decision = next(decisions)
                if decision == "discard":
                    skip_signal.queue_action("repeat")
                skip_signal.set_review_decision(decision)

        workflow = MagicMock()
        workflow.run.side_effect = _render

        events = self._run(workflow, runs=1, skip_signal=skip_signal, on_event=_on_event, upscale=2, review=True)

        assert [e["review"] for e in events if e["type"] == "job_paused"] == [{"iteration": 1, "attempt": 1}, {"iteration": 1, "attempt": 2}]


class TestEndImageBatch:
    """An end image in the arguments reaches each run's request."""

    @pytest.mark.parametrize(("end_image", "expected"), [("/tmp/last.png", "/tmp/last.png"), (None, None)])
    def test_request_carries_the_end_image(self, end_image, expected):
        requests = []
        workflow = MagicMock()
        workflow.run.side_effect = lambda request, artifacts: requests.append(request) or StageOutcome.success
        args = _make_video_args(image_path="/tmp/first.png", end_image_path=end_image)
        prompts = {"default": [("a dog", None)]}
        config: dict = {"generation": {"seed_min": 4, "seed_max": 10**9}}

        run_video_batch(_make_mock_video_backend(), MagicMock(), make_ltx_info(), workflow, prompts, config, args, plan=_make_plan(prompts, args, config=config))

        (request,) = requests
        assert (request.image_path, request.end_image_path) == ("/tmp/first.png", expected)


class TestExtendBatch:
    """Verify an extension job builds its request from the extend arguments and plans each run with its own seed."""

    def _source(self):
        return make_extend_source(scene="A street at night.", prompt="She walks away.")

    def _run(self, *, runs=1, prompts=None, **arg_overrides):
        requests = []
        workflow = MagicMock()
        workflow.run.side_effect = lambda request, artifacts: requests.append(request) or StageOutcome.success
        source = self._source()
        args = _make_video_args(runs=runs, width=512, height=256, num_frames=25 + 48, extend_source=source, extend_frames=48, **arg_overrides)
        prompts = prompts or {"default": [("", None)]}
        config: dict = {"generation": {"seed_min": 4, "seed_max": 10**9}}
        run_video_batch(_make_mock_video_backend(), MagicMock(), make_ltx_info(), workflow, prompts, config, args, plan=_make_plan(prompts, args, config=config, scene=source.scene))
        return requests, source

    def test_request_carries_the_extend_spec_and_the_window_size(self):
        (request,), source = self._run()
        assert request.extend == ExtendSpec(source, 48)
        assert (request.width, request.height, request.num_frames) == (512, 256, 73)
        assert request.image_path is None and request.upscale is None

    def test_planned_scene_reaches_the_request_and_an_empty_prompt_is_accepted(self):
        (request,), _ = self._run()
        assert request.prompt == "" and request.scene == "A street at night." and request.resolved_scene == "A street at night."

    def test_sibling_runs_get_different_seeds(self):
        with patch("zvisiongenerator.preflight.random.randint", side_effect=[1111, 2222]):
            requests, _ = self._run(runs=2, seed=None)
        assert [r.seed for r in requests] == [1111, 2222]
        assert all(r.extend is not None and r.extend.frames == 48 for r in requests)

    def test_fixed_seed_repeats_across_runs(self):
        requests, _ = self._run(runs=2, seed=9)
        assert [r.seed for r in requests] == [9, 9]

    def test_the_request_records_the_picked_model_name_over_the_resolved_model(self):
        (request,), _ = self._run(model="org/resolved-repo", model_name="ltx-4")
        assert request.model_name == "ltx-4"

    def test_other_jobs_have_no_extend(self):
        workflow = MagicMock()
        workflow.run.return_value = StageOutcome.success
        args = _make_video_args()
        prompts = {"default": [("a sunset", None)]}
        run_video_batch(_make_mock_video_backend(), MagicMock(), make_ltx_info(), workflow, prompts, {}, args, plan=_make_plan(prompts, args))
        assert workflow.run.call_args.args[0].extend is None
