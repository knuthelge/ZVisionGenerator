"""Tests for the Web UI runner facade and SSE endpoints."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

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
            if frame.get("event") in {"batch_completed", "job_completed", "job_failed"}:
                break
            continue
        frame_lines.append(line)

    return events


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
            assert running_snapshot["supports_controls"] == ["next", "pause", "repeat", "quit"]
            assert worker_thread_ids == [worker_thread_ids[0]]
            assert worker_thread_ids[0] != threading.get_ident()
            assert captured_control_signal["signal"] is not None

            release.set()
            completed_snapshot = _wait_for_status(runner, job_id, "completed")
            assert completed_snapshot["status"] == "completed"
            assert completed_snapshot["last_event"]["type"] == "batch_completed"
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
            assert snapshot["event_count"] == 3
            assert snapshot["last_event"]["type"] == "batch_completed"
        finally:
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


class TestWebServerSse:
    """Verify the FastAPI endpoints expose job state and SSE updates."""

    def test_dummy_job_sse_endpoint_streams_events(self, monkeypatch):
        """The SSE endpoint should emit structured progress frames for a background job."""
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        monkeypatch.setattr(web_server, "web_runner", runner)

        with TestClient(web_server.app) as client:
            create_response = client.post("/jobs/dummy", params={"steps": 2, "delay_seconds": 0.001})

            assert create_response.status_code == 202
            job_id = create_response.json()["job_id"]

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
        assert event_names[-1] == "batch_completed"

        payloads = [event["data"] for event in events]
        assert all(payload["job_id"] == job_id for payload in payloads)
        assert [payload["event_id"] for payload in payloads] == sorted(payload["event_id"] for payload in payloads)
