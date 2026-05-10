"""Tests for the Web UI runner facade and SSE endpoints."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from PIL import Image

from tests.conftest import _make_args
from tests.conftest import _make_video_args
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import web_runner as web_runner_module


def _wait_for_status(runner: web_runner_module.WebRunner, job_id: str, expected_status: str, *, timeout: float = 1.0) -> dict[str, object]:
    """Poll a job until it reaches the expected status."""
    deadline = time.monotonic() + timeout
    snapshot = runner.get_job_snapshot(job_id)
    while time.monotonic() < deadline:
        snapshot = runner.get_job_snapshot(job_id)
        if snapshot["status"] == expected_status:
            return snapshot
        time.sleep(0.01)
    return snapshot


def _read_sse_events(response) -> list[dict[str, object]]:
    """Parse streamed SSE frames into structured event dictionaries."""
    events: list[dict[str, object]] = []
    frame_lines: list[str] = []

    for line in response.iter_lines():
        if line == "":
            if not frame_lines:
                continue
            frame: dict[str, object] = {}
            for frame_line in frame_lines:
                if frame_line.startswith(":"):
                    continue
                field, _, value = frame_line.partition(":")
                frame[field] = value.lstrip()
            if "data" in frame:
                frame["data"] = json.loads(str(frame["data"]))
            events.append(frame)
            frame_lines = []
            if frame.get("event") in {"job_completed", "job_failed", "job_cancelled"}:
                break
            continue
        frame_lines.append(line)

    return events


def test_importing_web_runner_does_not_install_stdio_wrappers():
    """Importing the runner module should not mutate process-global stdout or stderr."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    module_path = Path(web_runner_module.__file__)
    spec = importlib.util.spec_from_file_location("zvisiongenerator.web._web_runner_import_probe", module_path)
    assert spec is not None and spec.loader is not None
    probe_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = probe_module
    try:
        spec.loader.exec_module(probe_module)
    finally:
        sys.modules.pop(spec.name, None)

    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr


class TestWebRunner:
    """Verify synchronous runners execute safely behind the web facade."""

    def test_submit_image_job_runs_in_background_thread(self, monkeypatch):
        """Image jobs should execute on a worker thread and complete asynchronously."""
        started = threading.Event()
        release = threading.Event()
        worker_thread_ids: list[int] = []
        captured_control_signal = {}

        def _fake_run_batch(backend, model, prompts_data, config, args, model_info, *, progress_callback, enable_interactive_controls, skip_signal):
            worker_thread_ids.append(threading.get_ident())
            captured_control_signal["signal"] = skip_signal
            assert enable_interactive_controls is False
            assert sys.stdout.isatty() is False
            assert sys.stderr.isatty() is False
            assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
            assert os.environ["TQDM_DISABLE"] == "1"
            print("suppressed worker output")
            progress_callback({"type": "batch_started", "mode": "image", "total_iterations": 1, "total_runs": 1})
            started.set()
            release.wait(timeout=1.0)
            progress_callback({"type": "batch_completed", "mode": "image", "completed_iterations": 1, "total_iterations": 1})

        monkeypatch.setattr(web_runner_module, "run_batch", _fake_run_batch)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            started_at = time.monotonic()
            job_id = runner.submit_image_job(
                backend=MagicMock(),
                model=MagicMock(),
                prompts_data={"default": [("prompt", None)]},
                config={},
                args=_make_args(),
                model_info=MagicMock(),
            )
            elapsed = time.monotonic() - started_at

            assert elapsed < 0.2
            assert started.wait(timeout=1.0)

            running_snapshot = runner.get_job_snapshot(job_id)
            assert running_snapshot["status"] == "running"
            assert running_snapshot["supported_controls"] == ["next", "pause", "resume", "repeat", "quit"]
            assert worker_thread_ids == [worker_thread_ids[0]]
            assert worker_thread_ids[0] != threading.get_ident()
            assert captured_control_signal["signal"] is not None

            release.set()
            completed_snapshot = _wait_for_status(runner, job_id, "completed")
            assert completed_snapshot["status"] == "completed"
            assert completed_snapshot["last_event"]["type"] == "job_completed"
            assert completed_snapshot["terminal_event"] == "job_completed"
        finally:
            release.set()
            runner.shutdown()

    def test_submit_video_job_tracks_video_runner_progress(self, monkeypatch):
        """Video jobs should publish progress and terminal state from the wrapped runner."""

        def _fake_run_video_batch(*, backend, model, model_info, workflow, prompts_data, config, args, progress_callback):
            progress_callback({"type": "batch_started", "mode": "video", "total_iterations": 1, "total_runs": 1})
            progress_callback({"type": "batch_completed", "mode": "video", "completed_iterations": 1, "total_iterations": 1})

        monkeypatch.setattr(web_runner_module, "run_video_batch", _fake_run_video_batch)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            job_id = runner.submit_video_job(
                backend=MagicMock(),
                model=MagicMock(),
                model_info=MagicMock(),
                workflow=MagicMock(),
                prompts_data={"default": [("prompt", None)]},
                config={},
                args=_make_video_args(),
            )

            snapshot = _wait_for_status(runner, job_id, "completed")
            assert snapshot["status"] == "completed"
            assert snapshot["event_count"] == 4
            assert snapshot["last_event"]["type"] == "job_completed"
        finally:
            runner.shutdown()

    def test_completed_image_job_includes_output_assets(self, tmp_path, monkeypatch):
        """Successful generated output paths should become gallery-shaped job outputs."""
        output_path = tmp_path / "result.png"

        def _fake_run_batch(backend, model, prompts_data, config, args, model_info, *, progress_callback, enable_interactive_controls, skip_signal):
            Image.new("RGB", (16, 12), color="teal").save(output_path)
            progress_callback({"type": "generation_finished", "mode": "image", "status": "success", "filename": output_path.name, "output_path": str(output_path)})
            progress_callback({"type": "batch_completed", "mode": "image", "completed_iterations": 1, "total_iterations": 1})

        monkeypatch.setattr(web_runner_module, "run_batch", _fake_run_batch)
        monkeypatch.setattr(
            web_runner_module,
            "load_web_config",
            lambda: SimpleNamespace(default_models=SimpleNamespace(image="zit", video="ltx-8"), image_model_options=("zit",), video_model_options=("ltx-8",)),
        )
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            job_id = runner.submit_image_job(
                backend=MagicMock(),
                model=MagicMock(),
                prompts_data={"default": [("prompt", None)]},
                config={},
                args=_make_args(output=str(tmp_path)),
                model_info=MagicMock(),
            )

            snapshot = _wait_for_status(runner, job_id, "completed")

            assert snapshot["status"] == "completed"
            assert snapshot["outputs"] == snapshot["last_event"]["outputs"]
            assert snapshot["outputs"][0]["id"] == "result.png"
            assert snapshot["outputs"][0]["url"] == "/media/result.png"
            assert snapshot["outputs"][0]["media_type"] == "image"
            assert snapshot["result_path"] == str(output_path)
        finally:
            runner.shutdown()

    def test_submit_image_job_all_failed_batch_ends_failed(self, monkeypatch):
        """Image jobs with only failed generations should publish a failed terminal state."""

        def _fake_run_batch(backend, model, prompts_data, config, args, model_info, *, progress_callback, enable_interactive_controls, skip_signal):
            progress_callback({"type": "batch_started", "mode": "image", "total_iterations": 1, "total_runs": 1})
            progress_callback({"type": "generation_finished", "mode": "image", "status": "failed", "filename": "failed.png"})
            progress_callback({"type": "batch_failed", "mode": "image", "completed_iterations": 1, "total_iterations": 1})

        monkeypatch.setattr(web_runner_module, "run_batch", _fake_run_batch)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            job_id = runner.submit_image_job(
                backend=MagicMock(),
                model=MagicMock(),
                prompts_data={"default": [("prompt", None)]},
                config={},
                args=_make_args(),
                model_info=MagicMock(),
            )

            snapshot = _wait_for_status(runner, job_id, "failed")
            assert snapshot["status"] == "failed"
            assert snapshot["terminal_event"] == "job_failed"
            assert snapshot["last_event"]["type"] == "job_failed"
        finally:
            runner.shutdown()

    def test_submit_video_job_all_failed_batch_ends_failed(self, monkeypatch):
        """Video jobs with only failed generations should publish a failed terminal state."""

        def _fake_run_video_batch(*, backend, model, model_info, workflow, prompts_data, config, args, progress_callback):
            progress_callback({"type": "batch_started", "mode": "video", "total_iterations": 1, "total_runs": 1})
            progress_callback({"type": "generation_finished", "mode": "video", "status": "failed"})
            progress_callback({"type": "batch_failed", "mode": "video", "completed_iterations": 1, "total_iterations": 1})

        monkeypatch.setattr(web_runner_module, "run_video_batch", _fake_run_video_batch)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            job_id = runner.submit_video_job(
                backend=MagicMock(),
                model=MagicMock(),
                model_info=MagicMock(),
                workflow=MagicMock(),
                prompts_data={"default": [("prompt", None)]},
                config={},
                args=_make_video_args(),
            )

            snapshot = _wait_for_status(runner, job_id, "failed")
            assert snapshot["status"] == "failed"
            assert snapshot["terminal_event"] == "job_failed"
            assert snapshot["last_event"]["type"] == "job_failed"
        finally:
            runner.shutdown()

    def test_job_history_is_bounded_but_event_count_tracks_total_events(self):
        """Retained history should be capped without losing total event count or last event."""
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01, max_history_events=3)

        try:
            job_id = runner.submit_dummy_job(total_steps=5, delay_seconds=0.001)

            snapshot = _wait_for_status(runner, job_id, "completed")

            assert snapshot["status"] == "completed"
            assert snapshot["event_count"] == 9
            assert snapshot["last_event"]["type"] == "job_completed"
            with runner._jobs_lock:
                retained = list(runner._jobs[job_id].history)
            assert len(retained) == 3
            assert [event["event_id"] for event in retained] == [7, 8, 9]
            assert [event["type"] for event in retained] == ["progress", "batch_completed", "job_completed"]
        finally:
            runner.shutdown()

    def test_terminal_jobs_are_pruned_by_count_without_evicting_active_jobs(self):
        """Terminal retention pruning should never remove still-running jobs."""
        started = threading.Event()
        release = threading.Event()

        def _active_target(progress_callback):
            progress_callback({"type": "batch_started", "mode": "dummy", "total_iterations": 1, "total_runs": 1})
            started.set()
            release.wait(timeout=1.0)
            progress_callback({"type": "batch_completed", "mode": "dummy", "completed_iterations": 1, "total_iterations": 1})

        runner = web_runner_module.WebRunner(max_workers=2, heartbeat_seconds=0.01, max_terminal_jobs=1, terminal_retention_seconds=999.0)

        try:
            active_job_id = runner._submit_job(job_type="dummy", target_factory=_active_target)
            assert started.wait(timeout=1.0)
            first_terminal_id = runner.submit_dummy_job(total_steps=1, delay_seconds=0.001)
            assert _wait_for_status(runner, first_terminal_id, "completed")["status"] == "completed"
            second_terminal_id = runner.submit_dummy_job(total_steps=1, delay_seconds=0.001)
            assert _wait_for_status(runner, second_terminal_id, "completed")["status"] == "completed"

            with runner._jobs_lock:
                assert active_job_id in runner._jobs
                assert first_terminal_id not in runner._jobs
                assert second_terminal_id in runner._jobs
            assert runner.get_job_snapshot(active_job_id)["status"] == "running"
        finally:
            release.set()
            runner.shutdown()

    def test_submit_image_job_rejects_overlapping_exclusive_jobs(self, monkeypatch):
        """The runner should reject a second exclusive generation job while one is active."""
        started = threading.Event()
        release = threading.Event()

        def _fake_run_batch(backend, model, prompts_data, config, args, model_info, *, progress_callback, enable_interactive_controls, skip_signal):
            progress_callback({"type": "batch_started", "mode": "image", "total_iterations": 1, "total_runs": 1})
            started.set()
            release.wait(timeout=1.0)
            progress_callback({"type": "batch_completed", "mode": "image", "completed_iterations": 1, "total_iterations": 1})

        monkeypatch.setattr(web_runner_module, "run_batch", _fake_run_batch)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            runner.submit_image_job(
                backend=MagicMock(),
                model=MagicMock(),
                prompts_data={"default": [("prompt", None)]},
                config={},
                args=_make_args(),
                model_info=MagicMock(),
            )
            assert started.wait(timeout=1.0)

            try:
                runner.submit_image_job(
                    backend=MagicMock(),
                    model=MagicMock(),
                    prompts_data={"default": [("prompt", None)]},
                    config={},
                    args=_make_args(),
                    model_info=MagicMock(),
                )
            except web_runner_module.JobConflictError as exc:
                assert "already running" in str(exc)
            else:
                raise AssertionError("Expected JobConflictError for overlapping exclusive jobs")
        finally:
            release.set()
            runner.shutdown()

    def test_queue_job_control_records_control_event(self, monkeypatch):
        """Supported image controls should be queued and reflected in job history."""
        started = threading.Event()
        release = threading.Event()

        def _fake_run_batch(backend, model, prompts_data, config, args, model_info, *, progress_callback, enable_interactive_controls, skip_signal):
            progress_callback({"type": "batch_started", "mode": "image", "total_iterations": 1, "total_runs": 1})
            started.set()
            release.wait(timeout=1.0)
            progress_callback({"type": "batch_completed", "mode": "image", "completed_iterations": 1, "total_iterations": 1})

        monkeypatch.setattr(web_runner_module, "run_batch", _fake_run_batch)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            job_id = runner.submit_image_job(
                backend=MagicMock(),
                model=MagicMock(),
                prompts_data={"default": [("prompt", None)]},
                config={},
                args=_make_args(),
                model_info=MagicMock(),
            )
            assert started.wait(timeout=1.0)

            response = runner.queue_job_control(job_id, "pause")
            snapshot = runner.get_job_snapshot(job_id)

            assert response == {"job_id": job_id, "action": "pause", "status": "queued"}
            assert snapshot["last_event"]["type"] == "control_queued"
            assert snapshot["last_event"]["action"] == "pause"
        finally:
            release.set()
            runner.shutdown()

    def test_quit_while_paused_wakes_worker_and_cancels_job(self, monkeypatch):
        """A paused job should wake and publish cancellation when quit is queued."""
        paused = threading.Event()

        def _fake_run_batch(backend, model, prompts_data, config, args, model_info, *, progress_callback, enable_interactive_controls, skip_signal):
            progress_callback({"type": "batch_started", "mode": "image", "total_iterations": 1, "total_runs": 1})
            progress_callback({"type": "job_paused", "mode": "image", "completed_iterations": 0, "total_iterations": 1})
            paused.set()
            skip_signal.wait_for_key()
            if skip_signal.consume() == "quit":
                progress_callback({"type": "batch_cancelled", "mode": "image", "completed_iterations": 0, "total_iterations": 1})

        monkeypatch.setattr(web_runner_module, "run_batch", _fake_run_batch)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            job_id = runner.submit_image_job(
                backend=MagicMock(),
                model=MagicMock(),
                prompts_data={"default": [("prompt", None)]},
                config={},
                args=_make_args(),
                model_info=MagicMock(),
            )
            assert paused.wait(timeout=1.0)
            assert runner.get_job_snapshot(job_id)["status"] == "paused"

            runner.queue_job_control(job_id, "quit")

            cancelled_snapshot = _wait_for_status(runner, job_id, "cancelled")
            assert cancelled_snapshot["status"] == "cancelled"
            assert cancelled_snapshot["last_event"]["type"] == "job_cancelled"
            assert cancelled_snapshot["terminal_event"] == "job_cancelled"
        finally:
            runner.shutdown()

    def test_active_exclusive_snapshot_uses_public_job_shape(self, monkeypatch):
        """Reconnect consumers should see the same public snapshot shape for an active job."""
        started = threading.Event()
        release = threading.Event()

        def _fake_run_batch(backend, model, prompts_data, config, args, model_info, *, progress_callback, enable_interactive_controls, skip_signal):
            progress_callback({"type": "batch_started", "mode": "image", "total_iterations": 1, "total_runs": 1})
            progress_callback(
                {
                    "type": "step_progress",
                    "mode": "image",
                    "current_step": 2,
                    "total_steps": 10,
                    "elapsed_secs": 4,
                    "eta_secs": 16,
                    "workflow_stage_name": "denoise",
                    "workflow_stage_index": 1,
                    "run_index": 0,
                    "total_runs": 1,
                }
            )
            started.set()
            release.wait(timeout=1.0)
            progress_callback({"type": "batch_completed", "mode": "image", "completed_iterations": 1, "total_iterations": 1})

        monkeypatch.setattr(web_runner_module, "run_batch", _fake_run_batch)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            job_id = runner.submit_image_job(
                backend=MagicMock(),
                model=MagicMock(),
                prompts_data={"default": [("prompt", None)]},
                config={},
                args=_make_args(),
                model_info=MagicMock(),
            )
            assert started.wait(timeout=1.0)

            snapshot = runner.get_active_exclusive_job_snapshot()

            assert snapshot is not None
            assert snapshot["id"] == job_id
            assert snapshot["job_id"] == job_id
            assert snapshot["status"] == "running"
            assert snapshot["supported_controls"] == ["next", "pause", "resume", "repeat", "quit"]
            assert snapshot["last_event"]["type"] == "step_progress"
            assert snapshot["last_event"]["current_step"] == 2
            assert snapshot["event_count"] == 3
        finally:
            release.set()
            runner.shutdown()


class TestWebServerSse:
    """Verify the FastAPI endpoints expose job state and SSE updates."""

    def test_dummy_job_sse_endpoint_streams_events(self, monkeypatch):
        """The SSE endpoint should emit structured progress frames for a background job."""
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        monkeypatch.setattr(web_server, "web_runner", runner)

        with TestClient(web_server.app) as client:
            job_id = runner.submit_dummy_job(total_steps=2, delay_seconds=0.001)

            snapshot_response = client.get(f"/jobs/{job_id}")
            assert snapshot_response.status_code == 200
            assert snapshot_response.json()["job_id"] == job_id

            with client.stream("GET", f"/jobs/{job_id}/events") as response:
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/event-stream")
                events = _read_sse_events(response)

        event_names = [str(event["event"]) for event in events]
        assert event_names[0] == "job_submitted"
        assert "batch_started" in event_names
        assert "progress" in event_names
        assert "batch_completed" in event_names
        assert event_names[-1] == "job_completed"

        payloads = [event["data"] for event in events]
        assert all(payload["job_id"] == job_id for payload in payloads)
        assert [payload["event_id"] for payload in payloads] == sorted(payload["event_id"] for payload in payloads)
