"""Tests for the Web UI runner facade and SSE endpoints."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import httpx
from huggingface_hub.errors import GatedRepoError
from PIL import Image
import pytest

from tests._factories import _make_args
from tests._factories import _make_plan
from tests._factories import _make_video_args, make_extend_source
from tests.web._helpers import sse_payload, stub_request_job, wait_for_status

from zvisiongenerator.core.image_types import ImageGenerationRequest
from zvisiongenerator.core.video_types import VideoFileUpscaleRequest, VideoGenerationRequest
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils import app_db
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.utils.seedvr2 import FP8_WEIGHTS, MLX_WEIGHTS
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import web_runner as web_runner_module
from zvisiongenerator.web.job_log import JobLog
from zvisiongenerator.workflows import build_video_workflow


@pytest.fixture(autouse=True)
def _no_real_memory_release(monkeypatch):
    """Never run the real accelerator cleanup (on CUDA it imports torch and collects garbage); tests may override it."""
    monkeypatch.setattr(web_runner_module, "release_accelerator_memory", lambda: None)


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


def _submit_image(monkeypatch, runner, *, prompts_data, config, args) -> str:
    """Submit an image request job with preflight and model loading stubbed."""
    stub_request_job(monkeypatch)
    request = ImageGenerationRequest(backend=None, model=None, prompt="prompt", model_family="zimage")
    return runner.submit_image_request_job(request=request, prompts_data=prompts_data, config=config, args=args, model_ref="model")


def _submit_video(monkeypatch, runner, *, prompts_data, config, args) -> str:
    """Submit a video request job with preflight and model loading stubbed."""
    stub_request_job(monkeypatch)
    request = VideoGenerationRequest(backend=None, model=None, prompt="prompt", model_family="ltx")
    return runner.submit_video_request_job(request=request, prompts_data=prompts_data, config=config, args=args, model_ref="model")


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

        def _fake_run_batch(backend, model, prompts_data, config, args, model_info, *, plan, progress_callback, skip_signal):
            worker_thread_ids.append(threading.get_ident())
            captured_control_signal["signal"] = skip_signal
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
            job_id = _submit_image(monkeypatch, runner, prompts_data={"default": [("prompt", None)]}, config={}, args=_make_args())
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
            completed_snapshot = wait_for_status(runner, job_id, "completed")
            assert completed_snapshot["status"] == "completed"
            assert completed_snapshot["last_event"]["type"] == "job_completed"
            assert completed_snapshot["terminal_event"] == "job_completed"
        finally:
            release.set()
            runner.shutdown()

    def test_image_job_reports_saving_a_stored_quant_and_honours_quit_while_saving(self, monkeypatch, tmp_path):
        """The loader's saving phase becomes a model_loading event, and a queued Quit cancels the save."""
        loader_calls: dict = {}

        def _fake_loader(backend, model_ref, *, quantize, models_dir, model_name, lora_paths, lora_weights, on_phase, cancelled, release_memory):
            loader_calls.update(model_ref=model_ref, quantize=quantize, models_dir=models_dir, model_name=model_name, cancelled_before=cancelled())
            on_phase("saving_quant")
            runner.queue_job_control(job_id_holder["id"], "quit")
            loader_calls["cancelled_after"] = cancelled()
            return MagicMock(name="model"), MagicMock(name="model_info")

        def _fake_run_batch(backend, model, prompts_data, config, args, model_info, *, plan, progress_callback, skip_signal):
            skip_signal.consume()
            progress_callback({"type": "batch_cancelled", "mode": "image", "completed_iterations": 0, "total_iterations": 1})

        job_id_holder: dict = {}
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        stub_request_job(monkeypatch)
        monkeypatch.setattr(web_runner_module, "get_ziv_data_dir", lambda: tmp_path)
        monkeypatch.setattr(web_runner_module, "load_image_model", _fake_loader)
        monkeypatch.setattr(web_runner_module, "run_batch", _fake_run_batch)
        start = threading.Event()
        original_preflight = web_runner_module.run_preflight
        monkeypatch.setattr(web_runner_module, "run_preflight", lambda *a, **kw: (start.wait(1.0), original_preflight(*a, **kw))[1])

        try:
            request = ImageGenerationRequest(backend=None, model=None, prompt="prompt", model_family="zimage", model_name="atlas")
            job_id_holder["id"] = runner.submit_image_request_job(request=request, prompts_data={"default": [("prompt", None)]}, config={}, args=_make_args(), model_ref="/models/atlas", quantize=8)
            start.set()
            wait_for_status(runner, job_id_holder["id"], "cancelled")
            events = [event for event in runner._get_job(job_id_holder["id"]).history if event["type"] == "model_loading"]
        finally:
            runner.shutdown()

        assert loader_calls == {
            "model_ref": "/models/atlas",
            "quantize": 8,
            "models_dir": tmp_path / "models",
            "model_name": "atlas",
            "cancelled_before": False,
            "cancelled_after": True,
        }
        assert [event.get("phase") for event in events] == [None, "saving_quant"]
        assert events[1]["quantize"] == 8

    def test_submit_video_job_tracks_video_runner_progress(self, monkeypatch):
        """Video jobs should publish progress and terminal state from the wrapped runner."""

        def _fake_run_video_batch(*, backend, model, model_info, workflow, prompts_data, config, args, plan, progress_callback, skip_signal):
            progress_callback({"type": "batch_started", "mode": "video", "total_iterations": 1, "total_runs": 1})
            progress_callback({"type": "batch_completed", "mode": "video", "completed_iterations": 1, "total_iterations": 1})

        monkeypatch.setattr(web_runner_module, "run_video_batch", _fake_run_video_batch)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            job_id = _submit_video(monkeypatch, runner, prompts_data={"default": [("prompt", None)]}, config={}, args=_make_video_args())

            snapshot = wait_for_status(runner, job_id, "completed")
            assert snapshot["status"] == "completed"
            assert snapshot["event_count"] == 6  # job_submitted, job_started, model_loading, batch_started, batch_completed, job_completed
            assert snapshot["last_event"]["type"] == "job_completed"
        finally:
            runner.shutdown()

    @pytest.mark.parametrize(("scene", "scene_by_set"), [("Night.", {"default": ["A castle at dusk.", None]}), (None, None)])
    def test_submit_video_job_passes_scene_and_scene_by_set_to_preflight(self, monkeypatch, scene, scene_by_set):
        """The job scene and the per-entry prompt-file scenes reach run_preflight; a job without them passes None."""
        seen: dict[str, dict] = {}

        def _fake_run_video_batch(*, progress_callback, **_kw):
            progress_callback({"type": "batch_completed", "mode": "video", "completed_iterations": 1, "total_iterations": 1})

        def _recording_preflight(prompts_data, config, args, **kwargs):
            seen["video"] = kwargs
            return _make_plan(prompts_data, args, config=config)

        monkeypatch.setattr(web_runner_module, "run_video_batch", _fake_run_video_batch)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            stub_request_job(monkeypatch, run_preflight=_recording_preflight)
            request = VideoGenerationRequest(backend=None, model=None, prompt="prompt", model_family="ltx")
            job_id = runner.submit_video_request_job(
                request=request,
                prompts_data={"default": [("one", None), ("two", None)]},
                config={},
                args=_make_video_args(scene=scene),
                model_ref="model",
                scene_by_set=scene_by_set,
            )
            wait_for_status(runner, job_id, "completed")
        finally:
            runner.shutdown()

        assert seen["video"]["scene"] == scene
        assert seen["video"]["scene_by_set"] == scene_by_set

    def test_extend_job_builds_the_extend_workflow_and_loads_the_plain_model(self, monkeypatch):
        """Args that carry an extend source run the extend stages with the plain (not upscale) pipeline, and the source reaches the batch."""
        seen: dict[str, object] = {}

        def _fake_run_video_batch(*, workflow, args, progress_callback, **_kw):
            seen["stages"] = [stage.__name__ for stage in workflow.stages]
            seen["args"] = args
            progress_callback({"type": "batch_completed", "mode": "video", "completed_iterations": 1, "total_iterations": 1})

        source = make_extend_source(path=Path("/clips/clip.mp4"), frame_count=97, scene=None, prompt=None)
        args = _make_video_args(extend_source=source, extend_frames=48, width=512, height=256, num_frames=73, fps=24.0, image_path=None, upscale=None, scene="A street.")
        monkeypatch.setattr(web_runner_module, "run_video_batch", _fake_run_video_batch)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            backend = stub_request_job(monkeypatch)
            monkeypatch.setattr(web_runner_module, "build_video_workflow", build_video_workflow)
            request = VideoGenerationRequest(backend=None, model=None, prompt="prompt", model_family="ltx", width=512, height=256, num_frames=73)
            job_id = runner.submit_video_request_job(request=request, prompts_data={"default": [("", None)]}, config={}, args=args, model_ref="model")
            wait_for_status(runner, job_id, "completed")
        finally:
            runner.shutdown()

        assert "extend_video_stage" in seen["stages"] and "join_clips_stage" in seen["stages"]
        assert not {"text_to_video_stage", "image_to_video_stage", "strip_audio_stage"} & set(seen["stages"])
        assert seen["args"].extend_source is source
        assert "upscale" not in backend.load_model.call_args.kwargs
        assert backend.load_model.call_args.kwargs["mode"] == "extend"

    @pytest.mark.parametrize("end_image_path", ["/images/last.png", None])
    def test_an_image_to_video_job_loads_for_its_last_frame_image_only_when_it_has_one(self, monkeypatch, end_image_path):
        monkeypatch.setattr(web_runner_module, "run_video_batch", lambda **_kw: None)
        backend = stub_request_job(monkeypatch)
        request = VideoGenerationRequest(backend=None, model=None, prompt="prompt", model_family="ltx", image_path="/images/first.png", end_image_path=end_image_path)
        args = _make_video_args(image_path="/images/first.png", end_image_path=end_image_path)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            job_id = runner.submit_video_request_job(request=request, prompts_data={"default": [("prompt", None)]}, config={}, args=args, model_ref="model")
            wait_for_status(runner, job_id, "completed")
        finally:
            runner.shutdown()

        kwargs = backend.load_model.call_args.kwargs
        assert (kwargs["mode"], kwargs["end_image"]) == ("i2v", end_image_path is not None)

    @pytest.mark.parametrize(("upscale", "expect_source"), [(2, True), (None, False)])
    def test_video_job_passes_latent_upsampler_only_when_upscaling(self, monkeypatch, upscale, expect_source):
        """Video jobs hand the configured upsampler source to load_model only for upscale jobs."""
        monkeypatch.setattr(web_runner_module, "run_video_batch", lambda **_kw: None)
        backend = stub_request_job(monkeypatch)
        config = {"video_model_presets": {"ltx": {"diffusers": {"latent_upsampler": {"repo": "your-org/upsampler", "filename": "up.safetensors"}}}}}
        request = VideoGenerationRequest(backend=None, model=None, prompt="prompt", model_family="ltx", upscale=upscale)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            job_id = runner.submit_video_request_job(request=request, prompts_data={"default": [("prompt", None)]}, config=config, args=_make_video_args(), model_ref="model")
            wait_for_status(runner, job_id, "completed")
        finally:
            runner.shutdown()

        kwargs = backend.load_model.call_args.kwargs
        if expect_source:
            assert kwargs["latent_upsampler"] == ("your-org/upsampler", "up.safetensors")
        else:
            assert "latent_upsampler" not in kwargs

    def test_upscale_job_loads_model_and_runs_workflow_once(self, tmp_path, monkeypatch):
        """Upscale jobs load the recorded model (with its quant and LoRAs), run once, and accept only quit."""
        loader_calls: dict = {}
        runs: list = []

        def _fake_loader(backend, model_ref, *, quantize, models_dir, model_name, lora_paths, lora_weights, on_phase, cancelled, release_memory):
            loader_calls.update(model_ref=model_ref, quantize=quantize, model_name=model_name, lora_paths=lora_paths)
            return MagicMock(name="model"), MagicMock(name="model_info")

        def _fake_run_upscale(backend, model, request, workflow, *, progress_callback, skip_signal):
            runs.append((request, workflow))
            progress_callback({"type": "batch_completed", "mode": "image", "completed_iterations": 1, "total_iterations": 1})

        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        stub_request_job(monkeypatch)
        monkeypatch.setattr(web_runner_module, "get_ziv_data_dir", lambda: tmp_path)
        monkeypatch.setattr(web_runner_module, "load_image_model", _fake_loader)
        monkeypatch.setattr(web_runner_module, "run_upscale", _fake_run_upscale)
        request = ImageGenerationRequest(
            backend=None, model=None, prompt="p", model_name="zit", lora_paths=["/l.safetensors"], lora_weights=[0.5], upscale_factor=2, upscale_source="/out/a.png", quantize=8
        )
        workflow = MagicMock(name="workflow")

        try:
            job_id = runner.submit_upscale_job(request=request, workflow=workflow, model_ref="/models/zit", config={}, context={"output_dir": str(tmp_path)})
            snapshot = wait_for_status(runner, job_id, "completed")
        finally:
            runner.shutdown()

        assert snapshot["job_type"] == "upscale"
        assert snapshot["supported_controls"] == ["quit"]
        assert loader_calls == {"model_ref": "/models/zit", "quantize": 8, "model_name": "zit", "lora_paths": ["/l.safetensors"]}
        assert runs == [(request, workflow)]

    @pytest.mark.parametrize(("weights", "replaces_download", "label"), [(MLX_WEIGHTS, True, "SeedVR2 7B q8"), (FP8_WEIGHTS, False, "SeedVR2 7B fp8")])
    def test_seedvr2_upscale_loads_its_own_model_and_reports_the_copy_phases(self, tmp_path, monkeypatch, weights, replaces_download, label):
        """A SeedVR2 request loads through load_seedvr2 (never the image loader); its load events replace the download only where a copy is stored."""
        upscaler = MagicMock(name="upscaler")
        upscaler.weights = weights
        model = MagicMock(name="seedvr2_model")
        load_calls: list = []
        runs: list = []

        def _fake_load(engine, name, *, models_dir, on_phase):
            load_calls.append((engine, name, models_dir))
            for phase in ("downloading", "saving_quant"):
                on_phase(phase)
            return model

        def _fake_run_upscale(backend, loaded, request, workflow, *, progress_callback, skip_signal):
            runs.append((backend, loaded, request, workflow))
            progress_callback({"type": "batch_completed", "mode": "image", "completed_iterations": 1, "total_iterations": 1})

        def _no_image_loader(*_args, **_kwargs):
            raise AssertionError("a SeedVR2 job must not load an image model")

        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        stub_request_job(monkeypatch)
        monkeypatch.setattr(web_runner_module, "get_ziv_data_dir", lambda: tmp_path)
        monkeypatch.setattr(web_runner_module, "get_seedvr2_upscaler", lambda: upscaler)
        monkeypatch.setattr(web_runner_module, "load_seedvr2", _fake_load)
        monkeypatch.setattr(web_runner_module, "load_image_model", _no_image_loader)
        monkeypatch.setattr(web_runner_module, "run_upscale", _fake_run_upscale)
        request = ImageGenerationRequest(backend=None, model=None, prompt="p", upscale_factor=2, upscale_source="/out/a.png", upscale_method="seedvr2", upscale_model="seedvr2-7b", upscale_quantize=8)
        workflow = MagicMock(name="workflow")

        try:
            job_id = runner.submit_upscale_job(request=request, workflow=workflow, model_ref="numz/SeedVR2_comfyUI", config={}, context={"output_dir": str(tmp_path)})
            snapshot = wait_for_status(runner, job_id, "completed")
            events = [event for event in runner._get_job(job_id).history if event["type"] == "model_loading"]
        finally:
            runner.shutdown()

        assert snapshot["status"] == "completed"
        assert load_calls == [(upscaler, "seedvr2-7b", tmp_path / "models")]
        assert runs == [(upscaler, model, request, workflow)]
        assert [(event["phase"], event["quantize"], event["replaces_download"]) for event in events] == [("downloading", 8, replaces_download), ("saving_quant", 8, replaces_download)]
        assert {event["model"] for event in events} == {label}

    def test_seedvr2_upscale_without_an_engine_fails_the_job(self, tmp_path, monkeypatch):
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        stub_request_job(monkeypatch)
        monkeypatch.setattr(web_runner_module, "get_ziv_data_dir", lambda: tmp_path)
        monkeypatch.setattr(web_runner_module, "get_seedvr2_upscaler", lambda: None)
        request = ImageGenerationRequest(backend=None, model=None, prompt="p", upscale_factor=2, upscale_source="/out/a.png", upscale_method="seedvr2", upscale_model="seedvr2-7b")

        try:
            job_id = runner.submit_upscale_job(request=request, workflow=MagicMock(), model_ref="numz/SeedVR2_comfyUI", config={}, context={})
            snapshot = wait_for_status(runner, job_id, "failed")
        finally:
            runner.shutdown()

        assert snapshot["status"] == "failed"

    def test_seedvr2_upscale_quit_during_the_load_cancels_without_a_file(self, tmp_path, monkeypatch):
        """The load cannot be cancelled; a quit queued meanwhile is consumed when the workflow would start."""
        control = SkipSignal()
        events: list = []
        workflow = MagicMock(name="workflow")

        def _quitting_load(engine, name, *, models_dir, on_phase):
            control.queue_action("quit")
            return MagicMock(name="seedvr2_model")

        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        monkeypatch.setattr(web_runner_module, "get_ziv_data_dir", lambda: tmp_path)
        monkeypatch.setattr(web_runner_module, "get_seedvr2_upscaler", lambda: MagicMock(name="upscaler"))
        monkeypatch.setattr(web_runner_module, "load_seedvr2", _quitting_load)
        request = ImageGenerationRequest(backend=None, model=None, prompt="p", upscale_factor=2, upscale_source=str(tmp_path / "a.png"), upscale_method="seedvr2", upscale_model="seedvr2-3b")

        try:
            runner._run_upscale_request(request=request, workflow=workflow, model_ref="numz/SeedVR2_comfyUI", config={}, progress_callback=events.append, control_signal=control)
        finally:
            runner.shutdown()

        workflow.run.assert_not_called()
        assert events[-1]["type"] == "batch_cancelled"
        assert not list(tmp_path.glob("*.png"))

    def _video_file_request(self, tmp_path):
        return VideoFileUpscaleRequest(
            source=make_extend_source(path=tmp_path / "clip.mp4"), recorded_config=None, output_dir=str(tmp_path), width=1536, height=896, upscale_model="seedvr2-7b", softness=0.0, batch_size=25
        )

    def test_video_file_upscale_job_downloads_loads_and_runs_in_video_mode(self, tmp_path, monkeypatch):
        """A video file upscale downloads the tier like the image one, loads it with video-mode events, runs once, and accepts only quit."""
        upscaler = MagicMock(name="upscaler")
        upscaler.weights = FP8_WEIGHTS
        model = MagicMock(name="seedvr2_model")
        targets = (DownloadTarget("numz/SeedVR2_comfyUI", None, ("a.safetensors",)),)
        plans: list = []
        downloads: list = []
        runs: list = []

        def _fake_load(engine, name, *, models_dir, on_phase):
            on_phase("loading")
            return model

        def _fake_run(engine, loaded, request, *, progress_callback, skip_signal):
            runs.append((engine, loaded, request))
            progress_callback({"type": "batch_completed", "mode": "video", "completed_iterations": 1, "total_iterations": 1})

        def _fake_ensure(self, planned, **kwargs):
            downloads.append((planned, kwargs["mode"], kwargs["role"], kwargs["model"]))
            return True

        monkeypatch.setattr(web_runner_module, "seedvr2_plan", lambda name, **kwargs: plans.append((name, kwargs["weights"])) or targets)
        monkeypatch.setattr(web_runner_module.WebRunner, "_ensure_downloads", _fake_ensure)
        monkeypatch.setattr(web_runner_module, "get_ziv_data_dir", lambda: tmp_path)
        monkeypatch.setattr(web_runner_module, "get_seedvr2_upscaler", lambda: upscaler)
        monkeypatch.setattr(web_runner_module, "load_seedvr2", _fake_load)
        monkeypatch.setattr(web_runner_module, "run_video_file_upscale", _fake_run)
        request = self._video_file_request(tmp_path)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            job_id = runner.submit_video_file_upscale_job(request=request, context={"output_dir": str(tmp_path), "media_type": "video"})
            snapshot = wait_for_status(runner, job_id, "completed")
            loading = [event for event in runner._get_job(job_id).history if event["type"] == "model_loading"]
        finally:
            runner.shutdown()

        image_plan = web_runner_module.WebRunner._upscale_download_plan(
            ImageGenerationRequest(backend=None, model=None, prompt="p", upscale_method="seedvr2", upscale_model="seedvr2-7b"), "numz/SeedVR2_comfyUI", {}
        )
        assert snapshot["job_type"] == "upscale"
        assert snapshot["supported_controls"] == ["quit"]
        assert snapshot["media_type"] == "video"
        assert downloads[0] == (targets, "video", "model", "SeedVR2 7B fp8")
        assert downloads[0][0] == image_plan
        assert plans[0] == ("seedvr2-7b", FP8_WEIGHTS)
        assert [(event["mode"], event["model"], event["replaces_download"]) for event in loading] == [("video", "SeedVR2 7B fp8", False)]
        assert runs == [(upscaler, model, request)]

    def test_a_quit_before_a_video_file_upscale_starts_cancels_in_video_mode(self, tmp_path, monkeypatch):
        control = SkipSignal()
        control.queue_action("quit")
        events: list = []
        monkeypatch.setattr(web_runner_module, "run_video_file_upscale", MagicMock(side_effect=AssertionError("must not run")))
        monkeypatch.setattr(web_runner_module, "get_seedvr2_upscaler", MagicMock(side_effect=AssertionError("must not load")))
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            runner._run_video_file_upscale_request(request=self._video_file_request(tmp_path), progress_callback=events.append, control_signal=control)
        finally:
            runner.shutdown()

        assert events == [{"type": "batch_cancelled", "mode": "video", "completed_iterations": 0, "total_iterations": 1}]

    def test_a_stopped_video_file_download_ends_the_job_without_loading(self, tmp_path, monkeypatch):
        monkeypatch.setattr(web_runner_module.WebRunner, "_ensure_downloads", lambda self, *_a, **_kw: False)
        monkeypatch.setattr(web_runner_module, "run_video_file_upscale", MagicMock(side_effect=AssertionError("must not run")))
        monkeypatch.setattr(web_runner_module, "load_seedvr2", MagicMock(side_effect=AssertionError("must not load")))
        monkeypatch.setattr(web_runner_module, "get_seedvr2_upscaler", lambda: MagicMock(name="upscaler"))
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            runner._run_video_file_upscale_request(request=self._video_file_request(tmp_path), progress_callback=lambda _event: None, control_signal=SkipSignal())
        finally:
            runner.shutdown()

    def test_image_job_snapshots_carry_no_media_type(self, tmp_path, monkeypatch):
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        stub_request_job(monkeypatch)
        monkeypatch.setattr(web_runner_module, "run_upscale", lambda *_a, **_kw: None)
        monkeypatch.setattr(web_runner_module, "load_image_model", lambda *_a, **_kw: (MagicMock(), MagicMock()))
        monkeypatch.setattr(web_runner_module, "get_ziv_data_dir", lambda: tmp_path)
        request = ImageGenerationRequest(backend=None, model=None, prompt="p", model_name="zit", upscale_factor=2, upscale_source="/out/a.png", quantize=8)

        try:
            job_id = runner.submit_upscale_job(request=request, workflow=MagicMock(), model_ref="/models/zit", config={}, context={"output_dir": str(tmp_path)})
            snapshot = wait_for_status(runner, job_id, "completed")
        finally:
            runner.shutdown()

        assert "media_type" not in snapshot

    def test_completed_image_job_includes_output_assets(self, tmp_path, monkeypatch):
        """Successful generated output paths should become gallery-shaped job outputs."""
        output_path = tmp_path / "result.png"

        def _fake_run_batch(backend, model, prompts_data, config, args, model_info, *, plan, progress_callback, skip_signal):
            Image.new("RGB", (16, 12), color="teal").save(output_path)
            progress_callback({"type": "generation_finished", "mode": "image", "status": "success", "filename": output_path.name, "output_path": str(output_path)})
            progress_callback({"type": "batch_completed", "mode": "image", "completed_iterations": 1, "total_iterations": 1})

        monkeypatch.setattr(web_runner_module, "run_batch", _fake_run_batch)
        monkeypatch.setattr(
            web_runner_module,
            "load_web_config",
            lambda: SimpleNamespace(
                default_models=SimpleNamespace(image="zit", video="ltx-8"), image_model_options=("zit",), video_model_options=("ltx-8",), image_inventory=(), app_config={}, seedvr2=None
            ),
        )
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            job_id = _submit_image(monkeypatch, runner, prompts_data={"default": [("prompt", None)]}, config={}, args=_make_args(output=str(tmp_path)))

            snapshot = wait_for_status(runner, job_id, "completed")

            assert snapshot["status"] == "completed"
            assert snapshot["outputs"] == snapshot["last_event"]["outputs"]
            assert snapshot["outputs"][0]["id"] == "result.png"
            assert snapshot["outputs"][0]["url"] == "/media/result.png"
            assert snapshot["outputs"][0]["media_type"] == "image"
            assert snapshot["result_path"] == str(output_path)

            async def _read_history() -> list[dict[str, object]]:
                payloads: list[dict[str, object]] = []
                async for frame in runner.stream_job_events(job_id):
                    payloads.append(sse_payload(frame))
                return payloads

            events = asyncio.run(_read_history())
            generation = next(event for event in events if event["type"] == "generation_finished")
            batch = next(event for event in events if event["type"] == "batch_completed")
            terminal = next(event for event in events if event["type"] == "job_completed")
            assert generation["status"] == "success"
            assert generation["asset"] == snapshot["outputs"][0]
            assert "asset" not in batch
            assert batch["completed_iterations"] == 1
            assert batch["total_iterations"] == 1
            assert terminal["outputs"] == snapshot["outputs"]
        finally:
            runner.shutdown()

    @pytest.mark.parametrize("mode", ["image", "video"])
    def test_a_batch_with_only_failed_generations_ends_failed(self, monkeypatch, mode):
        """Jobs with only failed generations publish a failed terminal state."""

        def _fake_run_batch(*_args, progress_callback, **_kw):
            progress_callback({"type": "batch_started", "mode": mode, "total_iterations": 1, "total_runs": 1})
            progress_callback({"type": "generation_finished", "mode": mode, "status": "failed"})
            progress_callback({"type": "batch_failed", "mode": mode, "completed_iterations": 1, "total_iterations": 1})

        monkeypatch.setattr(web_runner_module, "run_batch" if mode == "image" else "run_video_batch", _fake_run_batch)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            submit = _submit_image if mode == "image" else _submit_video
            args = _make_args() if mode == "image" else _make_video_args()
            job_id = submit(monkeypatch, runner, prompts_data={"default": [("prompt", None)]}, config={}, args=args)

            snapshot = wait_for_status(runner, job_id, "failed")
            assert snapshot["status"] == "failed"
            assert snapshot["terminal_event"] == "job_failed"
            assert snapshot["last_event"]["type"] == "job_failed"
        finally:
            runner.shutdown()

    @pytest.mark.filterwarnings("ignore:Video generation failed")
    def test_video_backend_error_reaches_job_failed_and_the_snapshot(self, monkeypatch):
        """A video workflow that raises ends the job with its one-line reason in job_failed and last_event."""
        workflow = MagicMock(name="workflow")
        workflow.run.side_effect = RuntimeError("CUDA out of memory")
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            job_id = _submit_video(monkeypatch, runner, prompts_data={"default": [("prompt", None)]}, config={}, args=_make_video_args())
            monkeypatch.setattr(web_runner_module, "build_video_workflow", lambda _args, **_kw: workflow)

            snapshot = wait_for_status(runner, job_id, "failed", timeout=5.0)

            async def _read_history() -> list[dict[str, object]]:
                return [sse_payload(frame) async for frame in runner.stream_job_events(job_id)]

            events = asyncio.run(_read_history())
            failed = next(event for event in events if event["type"] == "job_failed")
            assert failed["message"] == "CUDA out of memory"
            assert snapshot["last_event"]["type"] == "job_failed"
            assert snapshot["last_event"]["message"] == "CUDA out of memory"
        finally:
            runner.shutdown()

    def test_multiline_worker_error_gives_a_one_line_capped_message_and_logs_the_full_text(self, caplog):
        """job_failed carries a single bounded line, while the log keeps the whole multi-line error."""
        full_text = "first line\n" + "\n".join(f"  detail {n} " + "x" * 40 for n in range(40))
        assert len(full_text) > 500
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        def _fail(_progress_callback):
            raise RuntimeError(full_text)

        try:
            with caplog.at_level("ERROR", logger=web_runner_module.logger.name):
                job_id = runner._submit_job(job_type="test", exclusive=True, target_factory=_fail)
                snapshot = wait_for_status(runner, job_id, "failed")

            message = snapshot["last_event"]["message"]
            assert "\n" not in message
            assert message.startswith("first line detail 0")
            assert len(message) <= 500
            assert full_text.strip() in caplog.text
        finally:
            runner.shutdown()

    def test_job_history_is_bounded_but_event_count_tracks_total_events(self):
        """Retained history should be capped without losing total event count or last event."""
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01, max_history_events=3)

        try:
            job_id = runner.submit_dummy_job(total_steps=5, delay_seconds=0.001)

            snapshot = wait_for_status(runner, job_id, "completed")

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

    def test_sse_cursor_replays_only_retained_events_after_the_browser_cursor(self):
        """A reconnect must not replay progress/batch events the browser already applied."""
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            job_id = runner.submit_dummy_job(total_steps=2, delay_seconds=0.001)
            assert wait_for_status(runner, job_id, "completed")["status"] == "completed"
            with runner._jobs_lock:
                history = [dict(event) for event in runner._jobs[job_id].history]
            cursor = history[-2]["event_id"]

            async def _read_filtered_history() -> list[str]:
                return [frame async for frame in runner.stream_job_events(job_id, after_event_id=cursor)]

            async def _read_full_history() -> list[str]:
                return [frame async for frame in runner.stream_job_events(job_id)]

            filtered = asyncio.run(_read_filtered_history())
            full = asyncio.run(_read_full_history())

            assert [sse_payload(frame)["event_id"] for frame in filtered] == [history[-1]["event_id"]]
            assert sse_payload(filtered[0])["type"] == "job_completed"
            assert [sse_payload(frame)["event_id"] for frame in full] == [event["event_id"] for event in history]
        finally:
            runner.shutdown()

    def test_sse_filtered_history_subscribes_before_yielding_to_avoid_a_live_handoff_gap(self):
        """An event published after replay begins is delivered live rather than lost."""
        started = threading.Event()
        release = threading.Event()
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        def _wait_for_release(_progress_callback):
            started.set()
            release.wait(timeout=1.0)

        try:
            job_id = runner._submit_job(job_type="test", target_factory=_wait_for_release)
            assert started.wait(timeout=1.0)
            runner._publish_event(job_id, {"type": "step_progress", "current_step": 1, "total_steps": 2})

            async def _read_history_then_live() -> tuple[str, str]:
                stream = runner.stream_job_events(job_id, after_event_id=1)
                first = await anext(stream)
                runner._publish_event(job_id, {"type": "progress_text", "message": "arrived while subscribed"})
                second = await anext(stream)
                await stream.aclose()
                return first, second

            first, second = asyncio.run(_read_history_then_live())
            assert sse_payload(first)["event_id"] == 2
            assert sse_payload(second)["event_id"] == 3
            assert sse_payload(second)["message"] == "arrived while subscribed"
        finally:
            release.set()
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
            assert wait_for_status(runner, first_terminal_id, "completed")["status"] == "completed"
            second_terminal_id = runner.submit_dummy_job(total_steps=1, delay_seconds=0.001)
            assert wait_for_status(runner, second_terminal_id, "completed")["status"] == "completed"

            with runner._jobs_lock:
                assert active_job_id in runner._jobs
                assert first_terminal_id not in runner._jobs
                assert second_terminal_id in runner._jobs
            assert runner.get_job_snapshot(active_job_id)["status"] == "running"
        finally:
            release.set()
            runner.shutdown()

    def test_second_image_job_waits_in_the_queue(self, monkeypatch):
        """A second generation job is queued behind the running one, then runs after it."""
        started = threading.Event()
        release = threading.Event()

        def _fake_run_batch(backend, model, prompts_data, config, args, model_info, *, plan, progress_callback, skip_signal):
            progress_callback({"type": "batch_started", "mode": "image", "total_iterations": 1, "total_runs": 1})
            started.set()
            release.wait(timeout=1.0)
            progress_callback({"type": "batch_completed", "mode": "image", "completed_iterations": 1, "total_iterations": 1})

        monkeypatch.setattr(web_runner_module, "run_batch", _fake_run_batch)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            first = _submit_image(monkeypatch, runner, prompts_data={"default": [("prompt", None)]}, config={}, args=_make_args())
            assert started.wait(timeout=1.0)
            second = _submit_image(monkeypatch, runner, prompts_data={"default": [("prompt", None)]}, config={}, args=_make_args())

            snapshot = runner.get_job_snapshot(second)
            assert snapshot["status"] == "queued"
            assert snapshot["queue_position"] == 1
            assert runner.get_job_snapshot(first)["queue_position"] is None

            release.set()
            assert wait_for_status(runner, first, "completed")
            assert wait_for_status(runner, second, "completed")
        finally:
            release.set()
            runner.shutdown()

    @pytest.mark.parametrize(
        ("error", "message_fragment"),
        [
            (RuntimeError("backend failed"), "backend failed"),
            (SystemExit(), "SystemExit stopped the generation worker."),
        ],
    )
    def test_worker_failures_terminalize_once_and_release_exclusive_slot(self, error, message_fragment):
        """Expected worker failures must not leave an exclusive job active forever."""
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        def _fail(_progress_callback):
            raise error

        try:
            failed_job_id = runner._submit_job(job_type="test", exclusive=True, target_factory=_fail)
            snapshot = wait_for_status(runner, failed_job_id, "failed")

            assert snapshot["terminal_event"] == "job_failed"
            assert snapshot["completed_at"] is not None
            assert snapshot["last_event"]["type"] == "job_failed"
            assert message_fragment in snapshot["last_event"]["message"]
            with runner._jobs_lock:
                record = runner._jobs[failed_job_id]
            with record.lock:
                assert [event["type"] for event in record.history].count("job_failed") == 1

            reusable_job_id = runner._submit_job(job_type="test", exclusive=True, target_factory=lambda _progress_callback: None)
            assert wait_for_status(runner, reusable_job_id, "completed")["terminal_event"] == "job_completed"
        finally:
            runner.shutdown()

    def test_a_hub_refusal_fails_the_job_with_one_actionable_line(self):
        """A gated repo ends the job with the next step, not the raw Hub error."""
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        gated = GatedRepoError("boom\nmulti-line detail", response=httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/api/models/owner/repo")))

        def _fail(_progress_callback):
            raise RuntimeError("Failed to load model") from gated

        try:
            job_id = runner._submit_job(job_type="test", exclusive=True, target_factory=_fail)
            snapshot = wait_for_status(runner, job_id, "failed")

            message = snapshot["last_event"]["message"]
            assert "\n" not in message
            assert "hf auth login" in message
            assert "https://huggingface.co" in message
        finally:
            runner.shutdown()

    def test_video_worker_rechecks_ffmpeg_and_terminalizes_if_it_disappears(self, monkeypatch):
        """The worker protects the preflight-to-start race without invoking the CLI flow."""
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        request = MagicMock(model_name="ltx-8", model_family="ltx", upscale=None, image_path=None)
        monkeypatch.setattr(web_runner_module, "require_ffmpeg", lambda: (_ for _ in ()).throw(RuntimeError("ffmpeg disappeared")))
        monkeypatch.setattr(web_runner_module, "get_video_backend", lambda _family: pytest.fail("backend must not load without ffmpeg"))

        try:
            job_id = runner.submit_video_request_job(
                request=request,
                prompts_data={"web": [("a lake", None)]},
                config={},
                args=_make_video_args(),
                model_ref="ltx-8",
            )
            snapshot = wait_for_status(runner, job_id, "failed")

            assert snapshot["terminal_event"] == "job_failed"
            assert snapshot["completed_at"] is not None
            assert snapshot["last_event"]["message"] == "ffmpeg disappeared"
        finally:
            runner.shutdown()

    @pytest.mark.parametrize("error", [KeyboardInterrupt(), GeneratorExit()])
    def test_process_control_exceptions_propagate_from_direct_worker_target(self, error):
        """Only SystemExit is terminalized; process-control exceptions retain their semantics."""
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        def _succeed(_progress_callback):
            return None

        def _raise_process_control():
            raise error

        try:
            job_id = runner._submit_job(job_type="test", target_factory=_succeed)
            assert wait_for_status(runner, job_id, "completed")["status"] == "completed"

            with pytest.raises(type(error)):
                runner._run_target(job_id, _raise_process_control)
        finally:
            runner.shutdown()

    def test_queue_job_control_records_control_event(self, monkeypatch):
        """Supported image controls should be queued and reflected in job history."""
        started = threading.Event()
        release = threading.Event()

        def _fake_run_batch(backend, model, prompts_data, config, args, model_info, *, plan, progress_callback, skip_signal):
            progress_callback({"type": "batch_started", "mode": "image", "total_iterations": 1, "total_runs": 1})
            started.set()
            release.wait(timeout=1.0)
            progress_callback({"type": "batch_completed", "mode": "image", "completed_iterations": 1, "total_iterations": 1})

        monkeypatch.setattr(web_runner_module, "run_batch", _fake_run_batch)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            job_id = _submit_image(monkeypatch, runner, prompts_data={"default": [("prompt", None)]}, config={}, args=_make_args())
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

        def _fake_run_batch(backend, model, prompts_data, config, args, model_info, *, plan, progress_callback, skip_signal):
            progress_callback({"type": "batch_started", "mode": "image", "total_iterations": 1, "total_runs": 1})
            progress_callback({"type": "job_paused", "mode": "image", "completed_iterations": 0, "total_iterations": 1})
            paused.set()
            skip_signal.wait_for_key()
            if skip_signal.consume() == "quit":
                progress_callback({"type": "batch_cancelled", "mode": "image", "completed_iterations": 0, "total_iterations": 1})

        monkeypatch.setattr(web_runner_module, "run_batch", _fake_run_batch)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            job_id = _submit_image(monkeypatch, runner, prompts_data={"default": [("prompt", None)]}, config={}, args=_make_args())
            assert paused.wait(timeout=1.0)
            assert runner.get_job_snapshot(job_id)["status"] == "paused"

            runner.queue_job_control(job_id, "quit")

            cancelled_snapshot = wait_for_status(runner, job_id, "cancelled")
            assert cancelled_snapshot["status"] == "cancelled"
            assert cancelled_snapshot["last_event"]["type"] == "job_cancelled"
            assert cancelled_snapshot["terminal_event"] == "job_cancelled"
        finally:
            runner.shutdown()

    def test_shutdown_before_a_pause_still_ends_the_job(self, monkeypatch):
        """A job that pauses right after shutdown must not block forever: the queued quit ends the pause."""
        started = threading.Event()
        proceed = threading.Event()
        actions: list[str | None] = []

        def _fake_run_batch(backend, model, prompts_data, config, args, model_info, *, plan, progress_callback, skip_signal):
            progress_callback({"type": "batch_started", "mode": "image", "total_iterations": 1, "total_runs": 1})
            started.set()
            assert proceed.wait(timeout=2.0)
            skip_signal.wait_for_key()
            actions.append(skip_signal.consume())
            progress_callback({"type": "batch_cancelled", "mode": "image", "completed_iterations": 0, "total_iterations": 1})

        monkeypatch.setattr(web_runner_module, "run_batch", _fake_run_batch)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            job_id = _submit_image(monkeypatch, runner, prompts_data={"default": [("prompt", None)]}, config={}, args=_make_args())
            assert started.wait(timeout=1.0)

            runner.shutdown()
            proceed.set()

            snapshot = wait_for_status(runner, job_id, "cancelled", timeout=2.0)
            assert snapshot["status"] == "cancelled"
            assert actions == ["quit"]
        finally:
            proceed.set()
            runner.shutdown()

    @staticmethod
    def _review_batch(reached: threading.Event, decisions: list[str], *, rounds: int = 1):
        """Return a fake run_batch that waits for the review decision as the image runner does."""

        def _fake_run_batch(backend, model, prompts_data, config, args, model_info, *, plan, progress_callback, skip_signal):
            progress_callback({"type": "batch_started", "mode": "image", "total_iterations": rounds, "total_runs": 1})
            for iteration in range(1, rounds + 1):
                progress_callback({"type": "prompt_started", "mode": "image", "ran_iterations": iteration, "total_iterations": rounds, "prompt": "p"})
                skip_signal.begin_review()
                progress_callback(
                    {
                        "type": "job_paused",
                        "mode": "image",
                        "completed_iterations": iteration - 1,
                        "total_iterations": rounds,
                        "review": {"iteration": iteration, "attempt": 1},
                        "preview": Image.new("RGB", (8, 8)),
                    }
                )
                reached.set()
                decision = skip_signal.wait_for_review()
                decisions.append(decision)
                progress_callback({"type": "job_resumed", "mode": "image", "completed_iterations": iteration - 1, "total_iterations": rounds})
                if decision == "quit":
                    progress_callback({"type": "batch_cancelled", "mode": "image", "completed_iterations": iteration - 1, "total_iterations": rounds})
                    return
                progress_callback({"type": "generation_finished", "mode": "image", "status": "skipped" if decision == "discard" else "success", "ran_iterations": iteration})
            progress_callback({"type": "batch_completed", "mode": "image", "completed_iterations": rounds, "total_iterations": rounds})

        return _fake_run_batch

    def test_decisions_are_refused_while_no_review_waits(self, monkeypatch):
        started = threading.Event()
        release = threading.Event()

        def _fake_run_batch(backend, model, prompts_data, config, args, model_info, *, plan, progress_callback, skip_signal):
            progress_callback({"type": "batch_started", "mode": "image", "total_iterations": 1, "total_runs": 1})
            started.set()
            release.wait(timeout=1.0)

        monkeypatch.setattr(web_runner_module, "run_batch", _fake_run_batch)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            job_id = _submit_image(monkeypatch, runner, prompts_data={"default": [("prompt", None)]}, config={}, args=_make_args(upscale=2, review=True))
            assert started.wait(timeout=1.0)

            for action in ("approve", "discard"):
                with pytest.raises(web_runner_module.UnsupportedJobControlError, match="No render is waiting"):
                    runner.queue_job_control(job_id, action)
            assert runner.get_job_snapshot(job_id)["review"] is None
        finally:
            release.set()
            runner.shutdown()

    @pytest.mark.parametrize(("action", "decision"), [("approve", "approve"), ("discard", "discard")])
    def test_a_decision_ends_the_review_and_the_second_one_conflicts(self, monkeypatch, action, decision):
        reached = threading.Event()
        decisions: list[str] = []
        monkeypatch.setattr(web_runner_module, "run_batch", self._review_batch(reached, decisions))
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            job_id = _submit_image(monkeypatch, runner, prompts_data={"default": [("prompt", None)]}, config={}, args=_make_args(upscale=2, review=True))
            assert reached.wait(timeout=1.0)
            snapshot = runner.get_job_snapshot(job_id)
            assert snapshot["status"] == "paused"
            assert snapshot["paused"] is True
            assert snapshot["review"] == {"iteration": 1, "attempt": 1}
            assert snapshot["preview_version"] == 1
            assert runner.get_job_preview(job_id) is not None

            assert runner.queue_job_control(job_id, action) == {"job_id": job_id, "action": action, "status": "queued"}
            with pytest.raises(web_runner_module.UnsupportedJobControlError):
                runner.queue_job_control(job_id, "discard" if action == "approve" else "approve")

            assert wait_for_status(runner, job_id, "completed")["status"] == "completed"
            assert decisions == [decision]
            assert runner.get_job_snapshot(job_id)["review"] is None
        finally:
            runner.shutdown()

    def test_only_decisions_and_quit_are_accepted_during_a_review(self, monkeypatch):
        reached = threading.Event()
        decisions: list[str] = []
        monkeypatch.setattr(web_runner_module, "run_batch", self._review_batch(reached, decisions))
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            job_id = _submit_image(monkeypatch, runner, prompts_data={"default": [("prompt", None)]}, config={}, args=_make_args(upscale=2, review=True))
            assert reached.wait(timeout=1.0)

            for action in ("resume", "next", "pause", "repeat"):
                with pytest.raises(web_runner_module.UnsupportedJobControlError):
                    runner.queue_job_control(job_id, action)
            assert runner.get_job_snapshot(job_id)["review"] == {"iteration": 1, "attempt": 1}
            assert decisions == []

            runner.queue_job_control(job_id, "approve")
            assert wait_for_status(runner, job_id, "completed")["status"] == "completed"
        finally:
            runner.shutdown()

    def test_quit_ends_a_review_and_latches_it(self, monkeypatch):
        reached = threading.Event()
        decisions: list[str] = []
        monkeypatch.setattr(web_runner_module, "run_batch", self._review_batch(reached, decisions))
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            job_id = _submit_image(monkeypatch, runner, prompts_data={"default": [("prompt", None)]}, config={}, args=_make_args(upscale=2, review=True))
            assert reached.wait(timeout=1.0)

            runner.queue_job_control(job_id, "quit")
            with pytest.raises(web_runner_module.UnsupportedJobControlError):
                runner.queue_job_control(job_id, "approve")

            assert wait_for_status(runner, job_id, "cancelled")["status"] == "cancelled"
            assert decisions == ["quit"]
        finally:
            runner.shutdown()

    def test_each_review_of_a_batch_waits_for_its_own_decision(self, monkeypatch):
        reached = threading.Event()
        decisions: list[str] = []
        monkeypatch.setattr(web_runner_module, "run_batch", self._review_batch(reached, decisions, rounds=2))
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            job_id = _submit_image(monkeypatch, runner, prompts_data={"default": [("prompt", None)]}, config={}, args=_make_args(upscale=2, review=True))
            assert reached.wait(timeout=1.0)
            reached.clear()
            runner.queue_job_control(job_id, "discard")
            assert reached.wait(timeout=1.0)
            deadline = time.monotonic() + 1.0
            while runner.get_job_snapshot(job_id)["review"] != {"iteration": 2, "attempt": 1} and time.monotonic() < deadline:
                time.sleep(0.01)
            assert runner.get_job_snapshot(job_id)["review"] == {"iteration": 2, "attempt": 1}
            runner.queue_job_control(job_id, "approve")

            assert wait_for_status(runner, job_id, "completed")["status"] == "completed"
            assert decisions == ["discard", "approve"]
        finally:
            runner.shutdown()

    def test_snapshot_and_events_carry_no_filesystem_path_for_a_review(self, monkeypatch, tmp_path):
        reached = threading.Event()
        monkeypatch.setattr(web_runner_module, "run_batch", self._review_batch(reached, []))
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            job_id = _submit_image(monkeypatch, runner, prompts_data={"default": [("prompt", None)]}, config={}, args=_make_args(upscale=2, review=True, output=str(tmp_path)))
            assert reached.wait(timeout=1.0)

            paused = runner.get_job_snapshot(job_id)["last_event"]
            assert paused["type"] == "job_paused"
            assert paused["review"] == {"iteration": 1, "attempt": 1}
            assert "preview" not in paused
            assert paused["preview_version"] == 1
            assert str(tmp_path) not in json.dumps(runner.get_job_snapshot(job_id)["review"])
        finally:
            runner.shutdown()

    def test_a_video_review_path_stays_on_the_record_and_out_of_events_and_snapshots(self, monkeypatch, tmp_path):
        reached = threading.Event()
        decisions: list[str] = []
        small_render = str(tmp_path / "ziv-review-x" / "small-render.mp4")

        def _fake_run_video_batch(backend, model, model_info, workflow, prompts_data, config, args, *, plan, progress_callback, skip_signal):
            progress_callback({"type": "batch_started", "mode": "video", "total_iterations": 1, "total_runs": 1})
            skip_signal.begin_review()
            progress_callback({"type": "job_paused", "mode": "video", "completed_iterations": 0, "total_iterations": 1, "review": {"iteration": 1, "attempt": 1}, "review_path": small_render})
            reached.set()
            decisions.append(skip_signal.wait_for_review())
            progress_callback({"type": "job_resumed", "mode": "video", "completed_iterations": 0, "total_iterations": 1})
            progress_callback({"type": "generation_finished", "mode": "video", "status": "skipped"})
            progress_callback({"type": "batch_completed", "mode": "video", "completed_iterations": 1, "total_iterations": 1})

        monkeypatch.setattr(web_runner_module, "run_video_batch", _fake_run_video_batch)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            job_id = _submit_video(monkeypatch, runner, prompts_data={"default": [("prompt", None)]}, config={}, args=_make_video_args(upscale=2, review=True, output=str(tmp_path)))
            assert reached.wait(timeout=1.0)

            snapshot = runner.get_job_snapshot(job_id)
            assert snapshot["review"] == {"iteration": 1, "attempt": 1}
            assert snapshot["last_event"]["type"] == "job_paused"
            assert "review_path" not in snapshot["last_event"]
            assert small_render not in json.dumps(snapshot)
            assert runner._get_job(job_id).review_path == small_render

            runner._get_job(job_id).control_signal.set_review_decision("discard")
            final = wait_for_status(runner, job_id, "completed")

            assert decisions == ["discard"]
            assert runner._get_job(job_id).review_path is None
            assert small_render not in json.dumps(final)
            assert all("review_path" not in event for event in runner._get_job(job_id).history)
        finally:
            runner.shutdown()

    def test_the_review_stays_in_the_snapshot_between_the_decision_and_job_resumed(self, monkeypatch):
        """A browser reconnecting after the first decision still shows the review, not a plain pause."""
        reached = threading.Event()
        proceed = threading.Event()

        def _fake_run_batch(backend, model, prompts_data, config, args, model_info, *, plan, progress_callback, skip_signal):
            progress_callback({"type": "batch_started", "mode": "image", "total_iterations": 1, "total_runs": 1})
            skip_signal.begin_review()
            progress_callback({"type": "job_paused", "mode": "image", "completed_iterations": 0, "total_iterations": 1, "review": {"iteration": 1, "attempt": 1}, "preview": Image.new("RGB", (8, 8))})
            reached.set()
            skip_signal.wait_for_review()
            proceed.wait(timeout=2.0)
            progress_callback({"type": "job_resumed", "mode": "image", "completed_iterations": 0, "total_iterations": 1})
            progress_callback({"type": "generation_finished", "mode": "image", "status": "success", "ran_iterations": 1})
            progress_callback({"type": "batch_completed", "mode": "image", "completed_iterations": 1, "total_iterations": 1})

        monkeypatch.setattr(web_runner_module, "run_batch", _fake_run_batch)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            job_id = _submit_image(monkeypatch, runner, prompts_data={"default": [("prompt", None)]}, config={}, args=_make_args(upscale=2, review=True))
            assert reached.wait(timeout=1.0)
            runner.queue_job_control(job_id, "approve")

            snapshot = runner.get_job_snapshot(job_id)
            assert snapshot["paused"] is True
            assert snapshot["review"] == {"iteration": 1, "attempt": 1}
            with pytest.raises(web_runner_module.UnsupportedJobControlError):
                runner.queue_job_control(job_id, "discard")
            with pytest.raises(web_runner_module.UnsupportedJobControlError):
                runner.queue_job_control(job_id, "resume")

            proceed.set()
            assert wait_for_status(runner, job_id, "completed")["review"] is None
        finally:
            proceed.set()
            runner.shutdown()

    def test_shutdown_ends_a_waiting_review(self, monkeypatch):
        reached = threading.Event()
        decisions: list[str] = []
        monkeypatch.setattr(web_runner_module, "run_batch", self._review_batch(reached, decisions))
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        job_id = _submit_image(monkeypatch, runner, prompts_data={"default": [("prompt", None)]}, config={}, args=_make_args(upscale=2, review=True))
        assert reached.wait(timeout=1.0)

        runner.shutdown()

        assert wait_for_status(runner, job_id, "cancelled")["status"] == "cancelled"
        assert decisions == ["quit"]

    def test_video_jobs_accept_the_image_controls_and_quit_cancels(self, monkeypatch):
        """A running video job takes the image job's controls; its batch sees the queued Quit and ends the job."""
        started = threading.Event()

        def _fake_run_video_batch(*, backend, model, model_info, workflow, prompts_data, config, args, plan, progress_callback, skip_signal):
            progress_callback({"type": "batch_started", "mode": "video", "total_iterations": 1, "total_runs": 1})
            started.set()
            while skip_signal.pending() != "quit":
                time.sleep(0.01)
            progress_callback({"type": "batch_cancelled", "mode": "video", "completed_iterations": 0, "total_iterations": 1})

        monkeypatch.setattr(web_runner_module, "run_video_batch", _fake_run_video_batch)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)

        try:
            job_id = _submit_video(monkeypatch, runner, prompts_data={"default": [("prompt", None)]}, config={}, args=_make_video_args())
            assert started.wait(timeout=1.0)
            assert runner.get_job_snapshot(job_id)["supported_controls"] == ["next", "pause", "resume", "repeat", "quit"]

            for action in ("next", "pause", "repeat"):
                assert runner.queue_job_control(job_id, action)["status"] == "queued"
            assert runner.queue_job_control(job_id, "quit") == {"job_id": job_id, "action": "quit", "status": "queued"}

            assert wait_for_status(runner, job_id, "cancelled")["terminal_event"] == "job_cancelled"
        finally:
            runner.shutdown()

    def test_active_exclusive_snapshot_uses_public_job_shape(self, monkeypatch):
        """Reconnect consumers should see the same public snapshot shape for an active job."""
        started = threading.Event()
        release = threading.Event()

        def _fake_run_batch(backend, model, prompts_data, config, args, model_info, *, plan, progress_callback, skip_signal):
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
            job_id = _submit_image(monkeypatch, runner, prompts_data={"default": [("prompt", None)]}, config={}, args=_make_args())
            assert started.wait(timeout=1.0)

            snapshot = runner.get_active_exclusive_job_snapshot()

            assert snapshot is not None
            assert snapshot["id"] == job_id
            assert snapshot["job_id"] == job_id
            assert snapshot["status"] == "running"
            assert snapshot["supported_controls"] == ["next", "pause", "resume", "repeat", "quit"]
            assert snapshot["last_event"]["type"] == "step_progress"
            assert snapshot["last_event"]["current_step"] == 2
            assert snapshot["event_count"] == 5  # job_submitted, job_started, model_loading, batch_started, step_progress
        finally:
            release.set()
            runner.shutdown()


class TestWebServerSse:
    """Verify the FastAPI endpoints expose job state and SSE updates."""

    @pytest.mark.parametrize(
        ("last_event_id", "after", "expected_cursor"),
        [
            (None, None, None),
            ("7", None, 7),
            ("-1", None, None),
            (None, "5", 5),
            ("7", "5", 7),
            ("-1", "5", 5),
        ],
    )
    def test_sse_endpoint_forwards_only_valid_non_negative_last_event_id(self, monkeypatch, last_event_id, after, expected_cursor):
        """A valid Last-Event-ID wins, then the ``after`` query; anything malformed retains the initial replay."""
        captured: list[int | None] = []

        async def _stream_job_events(_job_id: str, *, after_event_id: int | None = None):
            captured.append(after_event_id)
            yield 'id: 8\nevent: job_completed\ndata: {"event_id": 8, "type": "job_completed"}\n\n'

        fake_runner = MagicMock()
        fake_runner.get_job_snapshot.return_value = {"job_id": "job-cursor"}
        fake_runner.stream_job_events.side_effect = _stream_job_events
        monkeypatch.setattr(web_server, "web_runner", fake_runner)
        headers = {} if last_event_id is None else {"Last-Event-ID": last_event_id}
        params = {} if after is None else {"after": after}

        with TestClient(web_server.app) as client:
            response = client.get("/jobs/job-cursor/events", headers=headers, params=params)

        assert response.status_code == 200
        assert "job_completed" in response.text
        assert captured == [expected_cursor]

    def test_sse_endpoint_rejects_negative_after_query(self, monkeypatch):
        """A negative ``after`` is a validation error and nothing is streamed."""
        fake_runner = MagicMock()
        fake_runner.get_job_snapshot.return_value = {"job_id": "job-cursor"}
        monkeypatch.setattr(web_server, "web_runner", fake_runner)

        with TestClient(web_server.app) as client:
            response = client.get("/jobs/job-cursor/events", params={"after": -1})

        assert response.status_code == 422
        fake_runner.stream_job_events.assert_not_called()

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


def test_prompt_context_survives_steps_and_pause_for_reconnection():
    runner = web_runner_module.WebRunner()
    record = web_runner_module._JobRecord(job_id="prompt-context", job_type="image")
    runner._jobs[record.job_id] = record
    runner._publish_event(record.job_id, {"type": "prompt_started", "prompt": "Second prompt", "run_index": 0, "total_runs": 2, "ran_iterations": 2, "total_iterations": 6})
    runner._publish_event(record.job_id, {"type": "step_progress", "current_step": 1, "total_steps": 20})
    assert runner.get_job_snapshot(record.job_id)["last_event"]["prompt"] == "Second prompt"
    runner._publish_event(record.job_id, {"type": "job_paused"})
    event = runner.get_job_snapshot(record.job_id)["last_event"]
    assert event["ran_iterations"] == 2
    assert event["total_iterations"] == 6
    assert event["prompt"] == "Second prompt"
    runner._publish_event(record.job_id, {"type": "prompt_started", "prompt": "Third prompt", "run_index": 0, "total_runs": 2, "ran_iterations": 3, "total_iterations": 6})
    runner._publish_event(record.job_id, {"type": "step_progress", "current_step": 1, "total_steps": 20})
    assert runner.get_job_snapshot(record.job_id)["last_event"]["prompt"] == "Third prompt"


def test_pruned_jobs_keep_their_final_status_for_late_clients():
    """A client that reconnects after its finished job was pruned must learn the real outcome, not get a 404."""
    runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01, terminal_retention_seconds=0.0)
    try:
        job_id = runner.submit_dummy_job(total_steps=1, delay_seconds=0.001)
        assert wait_for_status(runner, job_id, "completed")["status"] == "completed"
        runner.submit_dummy_job(total_steps=1, delay_seconds=0.001)  # triggers pruning of the finished job

        with runner._jobs_lock:
            runner._prune_terminal_jobs_locked()
            assert job_id not in runner._jobs

        assert runner.get_job_snapshot(job_id)["status"] == "completed"

        async def _collect() -> list[str]:
            return [frame async for frame in runner.stream_job_events(job_id, after_event_id=10_000)]

        frames = asyncio.run(_collect())
        assert len(frames) == 1
        assert "event: job_completed" in frames[0]

        with pytest.raises(KeyError):
            runner.get_job_snapshot("never-existed")
    finally:
        runner.shutdown()


class TestAcceleratorMemoryRelease:
    """Finished web jobs hand their model memory back, whatever the outcome."""

    @staticmethod
    def _run_job(monkeypatch, target_factory, *, expected_status: str, release=None):
        calls: list[str] = []
        monkeypatch.setattr(web_runner_module, "release_accelerator_memory", release or (lambda: calls.append("released")))
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            job_id = runner._submit_job(job_type="test", exclusive=True, target_factory=target_factory)
            snapshot = wait_for_status(runner, job_id, expected_status)
        finally:
            runner.shutdown()
        return snapshot, calls

    def test_failed_job_releases_memory_before_reporting_failure(self, monkeypatch):
        def _fail(_progress_callback):
            raise RuntimeError("backend failed")

        snapshot, calls = self._run_job(monkeypatch, _fail, expected_status="failed")

        assert snapshot["last_event"]["type"] == "job_failed"
        assert "backend failed" in snapshot["last_event"]["message"]
        assert calls == ["released"]

    def test_release_failure_does_not_change_job_outcome(self, monkeypatch):
        def _broken_release():
            raise RuntimeError("metal unavailable")

        with pytest.warns(UserWarning, match="metal unavailable"):
            snapshot, _ = self._run_job(monkeypatch, lambda _progress_callback: None, expected_status="completed", release=_broken_release)

        assert snapshot["status"] == "completed"


def test_non_generation_jobs_do_not_release_accelerator_memory(monkeypatch):
    """Only exclusive generation jobs hold models; others must not clear caches under a running generation."""
    calls: list[str] = []
    monkeypatch.setattr(web_runner_module, "release_accelerator_memory", lambda: calls.append("released"))
    runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
    try:
        job_id = runner.submit_dummy_job(total_steps=1, delay_seconds=0.001)
        assert wait_for_status(runner, job_id, "completed")["status"] == "completed"
    finally:
        runner.shutdown()

    assert calls == []


class TestLivePreview:
    @pytest.fixture()
    def running_job(self):
        release = threading.Event()
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        job_id = runner._submit_job(job_type="test", target_factory=lambda _progress_callback: release.wait(timeout=1.0))
        try:
            yield runner, job_id
        finally:
            release.set()
            runner.shutdown()

    def test_step_preview_is_kept_in_memory_and_replaced_by_a_version_in_events(self, running_job):
        runner, job_id = running_job

        runner._publish_event(job_id, {"type": "step_progress", "current_step": 2, "total_steps": 8, "preview": Image.new("RGB", (16, 16), "red")})
        runner._publish_event(job_id, {"type": "step_progress", "current_step": 3, "total_steps": 8})

        with runner._jobs_lock:
            history = list(runner._jobs[job_id].history)
        preview_event, plain_event = history[-2], history[-1]
        assert "preview" not in preview_event
        assert preview_event["preview_version"] == 1
        assert "preview_version" not in plain_event
        assert runner.get_job_preview(job_id).startswith(b"\xff\xd8")
        json.dumps(preview_event)

    def test_preview_versions_increase_and_reset_when_the_generation_finishes(self, running_job):
        runner, job_id = running_job

        for step in (2, 4):
            runner._publish_event(job_id, {"type": "step_progress", "current_step": step, "total_steps": 8, "preview": Image.new("RGB", (8, 8))})
        assert runner.get_job_snapshot(job_id)["last_event"]["preview_version"] == 2

        assert runner.get_job_snapshot(job_id)["preview_version"] == 2

        runner._publish_event(job_id, {"type": "generation_finished", "status": "skipped"})

        assert runner.get_job_preview(job_id) is None
        assert runner.get_job_snapshot(job_id)["preview_version"] == 0

    def test_a_new_workflow_stage_keeps_the_previous_stage_preview_until_it_sends_one(self, running_job):
        runner, job_id = running_job
        runner._publish_event(job_id, {"type": "step_progress", "current_step": 2, "total_steps": 8, "preview": Image.new("RGB", (8, 8))})

        runner._publish_event(job_id, {"type": "workflow_stage_started", "stage_name": "upscale"})

        assert runner.get_job_preview(job_id) is not None
        assert runner.get_job_snapshot(job_id)["preview_version"] == 1

    def test_a_preview_that_cannot_be_encoded_is_dropped_without_failing_the_step(self, running_job, monkeypatch):
        runner, job_id = running_job

        def _broken_save(*_args, **_kwargs):
            raise RuntimeError("unexpected encoder failure")

        monkeypatch.setattr(Image.Image, "save", _broken_save)
        runner._publish_event(job_id, {"type": "step_progress", "current_step": 2, "total_steps": 8, "preview": Image.new("RGB", (8, 8))})

        last_event = runner.get_job_snapshot(job_id)["last_event"]
        assert last_event["current_step"] == 2
        assert "preview_version" not in last_event
        assert runner.get_job_preview(job_id) is None

    def test_preview_route_serves_jpeg_or_404(self, monkeypatch):
        previews = {"job-1": b"\xff\xd8jpeg"}

        def _get_job_preview(job_id):
            if job_id not in previews and job_id != "job-2":
                raise KeyError(job_id)
            return previews.get(job_id)

        monkeypatch.setattr(web_server.web_runner, "get_job_preview", _get_job_preview)

        with TestClient(web_server.app) as client:
            ok = client.get("/jobs/job-1/preview?v=1")
            empty = client.get("/jobs/job-2/preview")
            unknown = client.get("/jobs/nope/preview")

        assert ok.status_code == 200
        assert ok.headers["content-type"] == "image/jpeg"
        assert ok.headers["cache-control"] == "no-store"
        assert ok.content == b"\xff\xd8jpeg"
        assert empty.status_code == 404
        assert unknown.status_code == 404


class TestJobQueue:
    """Generation jobs wait in a FIFO queue and are claimed by the single worker as they start."""

    @pytest.fixture()
    def runner(self):
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        yield runner
        runner.shutdown()

    @staticmethod
    def _blocking_job(runner: web_runner_module.WebRunner, *, exclusive: bool = True) -> tuple[str, threading.Event, threading.Event]:
        started = threading.Event()
        release = threading.Event()

        def _target(_progress_callback):
            started.set()
            release.wait(timeout=2.0)

        job_id = runner._submit_job(job_type="test", exclusive=exclusive, target_factory=_target)
        assert started.wait(timeout=1.0)
        return job_id, started, release

    @staticmethod
    def _event_types(runner: web_runner_module.WebRunner, job_id: str) -> list[str]:
        with runner._jobs_lock:
            record = runner._jobs[job_id]
        with record.lock:
            return [event["type"] for event in record.history]

    def test_concurrent_submits_run_in_queue_order(self, runner):
        first, _started, release = self._blocking_job(runner)
        ran: list[str] = []
        ids: list[str] = []
        ids_lock = threading.Lock()

        def _submit(index: int) -> None:
            label = f"job-{index}"
            job_id = runner._submit_job(job_type="test", exclusive=True, target_factory=lambda _cb: ran.append(label))
            with ids_lock:
                ids.append(label)
                labels[job_id] = label

        labels: dict[str, str] = {}
        threads = [threading.Thread(target=_submit, args=(index,)) for index in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        queued_order = [labels[snapshot["job_id"]] for snapshot in runner.list_jobs()["queued_jobs"]]
        release.set()
        for job_id in labels:
            assert wait_for_status(runner, job_id, "completed")["status"] == "completed"

        assert ran == queued_order
        assert len(ids) == 6
        types = self._event_types(runner, first)
        assert types.index("job_submitted") < types.index("job_started")

    def test_started_job_is_running_with_a_start_time(self, runner):
        job_id, _started, release = self._blocking_job(runner)
        snapshot = runner.get_job_snapshot(job_id)
        release.set()

        assert snapshot["status"] == "running"
        assert snapshot["started_at"] is not None
        assert snapshot["queue_position"] is None

    def test_removed_job_never_runs_even_when_its_work_item_is_picked_up(self, runner):
        _first, _started, release = self._blocking_job(runner)
        ran: list[str] = []
        queued = runner._submit_job(job_type="test", exclusive=True, target_factory=lambda _cb: ran.append("ran"))

        assert runner.cancel_queued(queued) is True
        # The worker reaching the job anyway (Future.cancel lost the race) must not run it.
        runner._run_target(queued, lambda: ran.append("ran"))
        release.set()

        assert ran == []
        assert self._event_types(runner, queued) == ["job_submitted", "job_cancelled"]
        snapshot = runner.get_job_snapshot(queued)
        assert snapshot["status"] == "cancelled"
        assert snapshot["last_event"]["reason"] == "removed"
        assert runner.cancel_queued(queued) is False

    def test_elapsed_time_does_not_count_the_wait_in_the_queue(self, runner):
        _first, _started, release = self._blocking_job(runner)

        def _report(progress_callback):
            progress_callback({"type": "step_progress", "current_step": 1, "total_steps": 2})

        queued = runner._submit_job(job_type="test", exclusive=True, target_factory=_report)
        time.sleep(0.1)
        release.set()
        assert wait_for_status(runner, queued, "completed")["status"] == "completed"

        with runner._jobs_lock:
            record = runner._jobs[queued]
        with record.lock:
            step = next(event for event in record.history if event["type"] == "step_progress")
        assert step["elapsed_secs"] < 0.05

    def test_list_jobs_puts_each_job_in_one_place_and_positions_follow_the_queue(self, runner):
        # A non-generation job occupies the worker, so the first generation job waits at the head unclaimed.
        _blocker, _started, release = self._blocking_job(runner, exclusive=False)
        head = runner._submit_job(job_type="test", exclusive=True, target_factory=lambda _cb: None)
        second = runner._submit_job(job_type="test", exclusive=True, target_factory=lambda _cb: None)
        third = runner._submit_job(job_type="test", exclusive=True, target_factory=lambda _cb: None)

        jobs = runner.list_jobs()
        assert jobs["active_job"]["job_id"] == head
        assert jobs["active_job"]["queue_position"] is None
        assert [(job["job_id"], job["queue_position"]) for job in jobs["queued_jobs"]] == [(second, 1), (third, 2)]
        assert runner.get_job_snapshot(third)["queue_position"] == 2
        release.set()

    def test_clear_queue_keeps_the_active_job(self, runner):
        _blocker, _started, release = self._blocking_job(runner, exclusive=False)
        head = runner._submit_job(job_type="test", exclusive=True, target_factory=lambda _cb: None)
        second = runner._submit_job(job_type="test", exclusive=True, target_factory=lambda _cb: None)
        third = runner._submit_job(job_type="test", exclusive=True, target_factory=lambda _cb: None)

        assert runner.clear_queue() == [second, third]
        release.set()

        assert wait_for_status(runner, head, "completed")["status"] == "completed"
        assert runner.get_job_snapshot(second)["status"] == "cancelled"
        assert runner.get_job_snapshot(third)["status"] == "cancelled"

    def test_clear_queue_leaves_a_running_job_alone(self, runner):
        running, _started, release = self._blocking_job(runner)
        queued = runner._submit_job(job_type="test", exclusive=True, target_factory=lambda _cb: None)

        assert runner.clear_queue() == [queued]
        assert runner.get_job_snapshot(running)["status"] == "running"
        release.set()

    def test_quit_right_after_the_claim_reaches_the_job(self, monkeypatch, runner):
        _blocker, _started, release = self._blocking_job(runner, exclusive=False)
        control_signal = MagicMock()
        job_id = runner._submit_job(job_type="test", exclusive=True, supported_controls=("quit",), control_signal=control_signal, target_factory=lambda _cb: None)
        publish = runner._publish_event
        # Hold back job_started to stand in for the moment between the claim and its event.
        monkeypatch.setattr(runner, "_publish_event", lambda jid, event: None if event["type"] == "job_started" else publish(jid, event))

        assert runner._claim(job_id) is True
        result = runner.queue_job_control(job_id, "quit")

        assert result["action"] == "quit"
        control_signal.queue_action.assert_called_once_with("quit")
        release.set()

    def test_controls_other_than_quit_are_refused_before_a_job_starts(self, runner):
        _first, _started, release = self._blocking_job(runner)
        queued = runner._submit_job(job_type="test", exclusive=True, supported_controls=("pause", "next", "repeat", "quit"), control_signal=MagicMock(), target_factory=lambda _cb: None)

        for action in ("pause", "next", "repeat"):
            with pytest.raises(web_runner_module.UnsupportedJobControlError, match="not started"):
                runner.queue_job_control(queued, action)
        assert runner.queue_job_control(queued, "quit")["status"] == "cancelled"
        release.set()

    def test_a_failed_job_does_not_stop_the_queue(self, runner):
        def _fail(_cb):
            raise RuntimeError("boom")

        failed = runner._submit_job(job_type="test", exclusive=True, target_factory=_fail)
        after = runner._submit_job(job_type="test", exclusive=True, target_factory=lambda _cb: None)

        assert wait_for_status(runner, failed, "failed")["status"] == "failed"
        assert wait_for_status(runner, after, "completed")["status"] == "completed"

    def test_memory_is_released_before_the_next_job_starts(self, monkeypatch, runner):
        order: list[str] = []
        monkeypatch.setattr(web_runner_module, "release_accelerator_memory", lambda: order.append("release"))
        first = runner._submit_job(job_type="test", exclusive=True, target_factory=lambda _cb: order.append("first"))
        second = runner._submit_job(job_type="test", exclusive=True, target_factory=lambda _cb: order.append("second"))

        assert wait_for_status(runner, second, "completed")["status"] == "completed"
        assert runner.get_job_snapshot(first)["status"] == "completed"
        assert order == ["first", "release", "second", "release"]


def test_shutdown_cancels_queued_jobs_without_running_them():
    runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
    started = threading.Event()
    release = threading.Event()
    ran: list[str] = []

    def _block(_cb):
        started.set()
        release.wait(timeout=2.0)

    runner._submit_job(job_type="test", exclusive=True, target_factory=_block)
    assert started.wait(timeout=1.0)
    queued = [runner._submit_job(job_type="test", exclusive=True, target_factory=lambda _cb: ran.append("ran")) for _ in range(2)]

    runner.shutdown()
    release.set()
    runner._executor.shutdown(wait=True)

    assert ran == []
    for job_id in queued:
        snapshot = runner.get_job_snapshot(job_id)
        assert snapshot["status"] == "cancelled"
        assert snapshot["last_event"]["reason"] == "removed"


def test_a_video_review_over_http_serves_the_small_render_until_the_job_resumes(monkeypatch, tmp_path):
    """Drive two reviews of one batch through the real routes: serve, reconnect, refuse a second decision, then 404."""
    renders = [tmp_path / "first.mp4", tmp_path / "second.mp4"]
    renders[0].write_bytes(b"first-render")
    renders[1].write_bytes(b"second-render")
    reached = [threading.Event(), threading.Event()]
    proceed = [threading.Event(), threading.Event()]
    decisions: list[str] = []

    def _fake_run_video_batch(backend, model, model_info, workflow, prompts_data, config, args, *, plan, progress_callback, skip_signal):
        progress_callback({"type": "batch_started", "mode": "video", "total_iterations": 2, "total_runs": 1})
        for index, render in enumerate(renders):
            skip_signal.begin_review()
            progress_callback(
                {"type": "job_paused", "mode": "video", "completed_iterations": index, "total_iterations": 2, "review": {"iteration": index + 1, "attempt": 1}, "review_path": str(render)}
            )
            reached[index].set()
            decisions.append(skip_signal.wait_for_review())
            proceed[index].wait(timeout=2.0)
            progress_callback({"type": "job_resumed", "mode": "video", "completed_iterations": index, "total_iterations": 2})
            progress_callback({"type": "generation_finished", "mode": "video", "status": "skipped"})
        progress_callback({"type": "batch_completed", "mode": "video", "completed_iterations": 2, "total_iterations": 2})

    monkeypatch.setattr(web_runner_module, "run_video_batch", _fake_run_video_batch)
    runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
    monkeypatch.setattr(web_server, "web_runner", runner)
    try:
        job_id = _submit_video(monkeypatch, runner, prompts_data={"default": [("prompt", None)]}, config={}, args=_make_video_args(upscale=2, review=True, output=str(tmp_path)))
        with TestClient(web_server.app) as client:
            assert reached[0].wait(timeout=1.0)
            first = client.get(f"/jobs/{job_id}/review?v=1")
            assert (first.status_code, first.content, first.headers["cache-control"]) == (200, b"first-render", "no-store")
            # A reconnecting browser sees the review in the snapshot, without any filesystem path.
            snapshot = client.get(f"/jobs/{job_id}")
            assert snapshot.json()["review"] == {"iteration": 1, "attempt": 1}
            assert str(tmp_path) not in snapshot.text
            assert client.post(f"/jobs/{job_id}/controls/resume").status_code == 409
            assert client.post(f"/jobs/{job_id}/controls/pause").status_code == 409

            assert client.post(f"/jobs/{job_id}/controls/approve").status_code == 200
            assert client.post(f"/jobs/{job_id}/controls/discard").status_code == 409
            assert client.get(f"/jobs/{job_id}/review?v=1").content == b"first-render"
            assert client.get(f"/jobs/{job_id}").json()["review"] == {"iteration": 1, "attempt": 1}

            proceed[0].set()
            assert reached[1].wait(timeout=1.0)
            second = client.get(f"/jobs/{job_id}/review?v=2")
            assert second.content == b"second-render"
            assert client.get(f"/jobs/{job_id}").json()["review"] == {"iteration": 2, "attempt": 1}

            assert client.post(f"/jobs/{job_id}/controls/discard").status_code == 200
            proceed[1].set()
            assert wait_for_status(runner, job_id, "completed")["review"] is None
            assert client.get(f"/jobs/{job_id}/review?v=2").status_code == 404
            assert client.get("/jobs/unknown-job/review").status_code == 404
        assert decisions == ["approve", "discard"]
    finally:
        proceed[0].set()
        proceed[1].set()
        runner.shutdown()


class TestJobLogHook:
    """A started image, video or upscale job is logged once, when it first reaches a terminal status."""

    LOG_KEYS = {
        "job_id",
        "job_type",
        "workflow",
        "model",
        "prompt",
        "settings",
        "status",
        "created_at",
        "started_at",
        "completed_at",
        "duration_secs",
        "outputs",
        "error",
        "failed_runs",
    }

    @pytest.fixture()
    def db_path(self, tmp_path) -> Path:
        return tmp_path / "ziv.sqlite3"

    @pytest.fixture()
    def runner(self, db_path):
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01, job_log=JobLog(lambda: db_path))
        yield runner
        runner.shutdown()

    @staticmethod
    def _entries(db_path: Path) -> list[dict]:
        """Return the logged entries, oldest first."""
        return JobLog(lambda: db_path).page("all", 1, 10_000)["entries"][::-1]

    @staticmethod
    def _drain(runner: web_runner_module.WebRunner) -> None:
        """Wait until the single worker has finished everything submitted so far."""
        runner._executor.submit(lambda: None).result(timeout=2.0)

    def _run(self, runner, events, *, job_type="image", context=None, exclusive=True) -> str:
        """Run a job that publishes *events* (the worker adds the success terminal when none of them is terminal)."""

        def _target(progress_callback):
            for event in events:
                progress_callback(event)

        job_id = runner._submit_job(job_type=job_type, exclusive=exclusive, context=context, target_factory=_target)
        self._drain(runner)
        return job_id

    @staticmethod
    def _blocking(runner, *, job_type="image") -> tuple[str, threading.Event, threading.Event]:
        started = threading.Event()
        release = threading.Event()

        def _target(_progress_callback):
            started.set()
            release.wait(timeout=2.0)

        job_id = runner._submit_job(job_type=job_type, exclusive=True, target_factory=_target)
        assert started.wait(timeout=1.0)
        return job_id, started, release

    def test_completed_job_is_logged_with_the_entry_keys(self, runner, db_path, monkeypatch):
        monkeypatch.setattr(web_runner_module, "_output_asset_payload", lambda _context, _path: {"id": "img/1.png"})
        context = {"prompt": "a lake", "model": "m1", "workflow": "txt2img", "settings": {"steps": 4}}

        job_id = self._run(runner, [{"type": "generation_finished", "status": "success", "output_path": "/x/1.png"}], context=context)

        (entry,) = self._entries(db_path)
        assert set(entry) == self.LOG_KEYS
        assert (entry["job_id"], entry["job_type"], entry["status"]) == (job_id, "image", "completed")
        assert (entry["prompt"], entry["model"], entry["workflow"], entry["settings"]) == ("a lake", "m1", "txt2img", {"steps": 4})
        assert entry["outputs"] == ["img/1.png"]
        assert entry["error"] is None
        assert entry["failed_runs"] == 0
        assert isinstance(entry["created_at"], float)
        assert entry["started_at"] <= entry["completed_at"]
        assert entry["duration_secs"] >= 0

    @pytest.mark.parametrize("job_type", ["image", "video", "upscale"])
    def test_every_logged_job_type_writes_an_entry(self, runner, db_path, job_type):
        self._run(runner, [], job_type=job_type)

        assert [entry["job_type"] for entry in self._entries(db_path)] == [job_type]

    def test_failed_job_logs_the_terminal_message(self, runner, db_path):
        self._run(runner, [{"type": "batch_failed", "mode": "image", "message": "CUDA out of memory"}])

        (entry,) = self._entries(db_path)
        assert entry["status"] == "failed"
        assert entry["error"] == "CUDA out of memory"

    def test_failed_job_without_a_message_falls_back_to_the_last_failed_generation(self, runner, db_path):
        self._run(
            runner,
            [
                {"type": "generation_finished", "status": "failed", "message": "first"},
                {"type": "generation_finished", "status": "failed"},
                {"type": "generation_finished", "status": "failed", "message": "second"},
                {"type": "batch_failed", "mode": "image"},
            ],
        )

        (entry,) = self._entries(db_path)
        assert (entry["status"], entry["error"], entry["failed_runs"]) == ("failed", "second", 3)

    def test_job_that_raises_logs_the_worker_message(self, runner, db_path):
        def _boom(_progress_callback):
            raise RuntimeError("line one\nline two")

        runner._submit_job(job_type="image", exclusive=True, target_factory=_boom)
        self._drain(runner)

        (entry,) = self._entries(db_path)
        assert (entry["status"], entry["error"]) == ("failed", "line one line two")

    def test_partial_failure_keeps_the_completed_status(self, runner, db_path):
        self._run(
            runner,
            [
                {"type": "generation_finished", "status": "failed", "message": "run 1 failed"},
                {"type": "generation_finished", "status": "success"},
                {"type": "batch_completed", "mode": "image"},
            ],
        )

        (entry,) = self._entries(db_path)
        assert (entry["status"], entry["failed_runs"], entry["error"]) == ("completed", 1, "run 1 failed")

    def test_cancel_after_start_is_logged_with_its_failed_runs(self, runner, db_path):
        self._run(runner, [{"type": "generation_finished", "status": "failed", "message": "bad"}, {"type": "batch_cancelled", "mode": "image"}])

        (entry,) = self._entries(db_path)
        assert (entry["status"], entry["failed_runs"], entry["error"]) == ("cancelled", 1, "bad")

    def test_one_entry_per_started_job(self, runner, db_path):
        ids = [
            self._run(runner, []),
            self._run(runner, [{"type": "batch_failed", "mode": "image", "message": "x"}]),
            self._run(runner, [{"type": "batch_cancelled", "mode": "image"}]),
            self._run(runner, [{"type": "generation_finished", "status": "failed"}, {"type": "batch_completed", "mode": "image"}]),
        ]

        entries = self._entries(db_path)
        assert [entry["job_id"] for entry in entries] == ids
        assert [entry["status"] for entry in entries] == ["completed", "failed", "cancelled", "completed"]

    def test_a_second_terminal_event_does_not_write_a_second_entry(self, runner, db_path):
        job_id = self._run(runner, [{"type": "batch_cancelled", "mode": "image"}, {"type": "batch_failed", "mode": "image", "message": "late"}])

        (entry,) = self._entries(db_path)
        assert (entry["job_id"], entry["status"], entry["error"]) == (job_id, "cancelled", None)

    def test_jobs_that_never_started_or_are_not_generation_jobs_are_not_logged(self, runner, db_path):
        running, _started, release = self._blocking(runner)
        queued = [runner._submit_job(job_type="image", exclusive=True, target_factory=lambda _cb: None) for _ in range(3)]

        assert runner.cancel_queued(queued[0]) is True
        assert runner.clear_queue() == queued[1:]
        release.set()
        self._drain(runner)
        self._run(runner, [], job_type="dummy", exclusive=False)
        self._run(runner, [], job_type="test")
        self._drain(runner)

        assert [entry["job_id"] for entry in self._entries(db_path)] == [running]

    def test_shutdown_of_queued_jobs_writes_nothing(self, db_path):
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01, job_log=JobLog(lambda: db_path))
        _running, _started, release = self._blocking(runner)
        for _ in range(2):
            runner._submit_job(job_type="video", exclusive=True, target_factory=lambda _cb: None)

        runner.shutdown()

        assert self._entries(db_path) == []
        release.set()

    def test_runner_without_a_job_log_writes_nothing(self):
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            job_id = runner._submit_job(job_type="image", exclusive=True, target_factory=lambda _cb: None)
            self._drain(runner)
            assert runner.get_job_snapshot(job_id)["status"] == "completed"
        finally:
            runner.shutdown()

        assert not app_db.app_db_path().exists()

    def test_a_failing_store_warns_and_the_next_job_still_runs(self, tmp_path):
        # The log "file" is a directory, so the store raises JobLogError.
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01, job_log=JobLog(lambda: tmp_path))
        ran: list[str] = []
        try:
            with pytest.warns(UserWarning, match="Could not write the job log"):
                first = self._run(runner, [])
                second = runner._submit_job(job_type="image", exclusive=True, target_factory=lambda _cb: ran.append("second"))
                self._drain(runner)
            assert runner.get_job_snapshot(first)["status"] == "completed"
            assert runner.get_job_snapshot(second)["status"] == "completed"
            assert ran == ["second"]
        finally:
            runner.shutdown()

    def test_a_failing_entry_builder_never_breaks_the_job(self, runner, db_path, monkeypatch):
        def _raise(**_kwargs):
            raise RuntimeError("builder broke")

        monkeypatch.setattr(web_runner_module, "build_job_log_entry", _raise)
        first, _started, release = self._blocking(runner)
        record = runner._jobs[first]
        subscriber: queue.Queue = queue.Queue()
        record.subscribers.add(subscriber)
        ran: list[str] = []
        second = runner._submit_job(job_type="image", exclusive=True, target_factory=lambda _cb: ran.append("second"))

        with pytest.warns(UserWarning, match="builder broke"):
            release.set()
            self._drain(runner)

        types = []
        while not subscriber.empty():
            types.append(subscriber.get_nowait()["type"])
        assert "job_completed" in types
        assert runner.get_job_snapshot(first)["status"] == "completed"
        assert runner.get_job_snapshot(second)["status"] == "completed"
        assert ran == ["second"]
        assert runner.list_jobs() == {"active_job": None, "queued_jobs": []}
        assert self._entries(db_path) == []


_BUSY_ENHANCER_SESSION = SimpleNamespace(busy=lambda: True)


class TestEnhancerAdmission:
    """A job is admitted, queued or refused by what the prompt enhancer is doing."""

    def test_job_admission_rejects_while_enhancing(self, monkeypatch):
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _BUSY_ENHANCER_SESSION)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            with pytest.raises(web_runner_module.JobConflictError, match="Prompt enhancement in progress"):
                runner._submit_job(job_type="image", exclusive=True, target_factory=lambda cb: None, admission_check=web_server._reject_while_busy)
            assert runner.get_active_exclusive_job_snapshot() is None
        finally:
            runner.shutdown()

    def test_job_queues_while_an_auto_enhance_job_holds_the_enhancer(self, monkeypatch):
        """While a job's own preflight holds the enhancer, a second job queues behind it instead of being refused."""
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _BUSY_ENHANCER_SESSION)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        monkeypatch.setattr(web_server, "web_runner", runner)
        release = threading.Event()
        try:
            runner._submit_job(job_type="image", exclusive=True, target_factory=lambda cb: release.wait(5))
            queued_id = runner._submit_job(job_type="image", exclusive=True, target_factory=lambda cb: None, admission_check=web_server._reject_while_busy)
            assert runner.get_job_snapshot(queued_id)["status"] == "queued"
        finally:
            release.set()
            runner.shutdown()

    def test_admit_exclusive_is_atomic_with_job_check(self):
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        release = threading.Event()
        admitted: list[bool] = []
        try:
            runner.admit_exclusive(lambda: admitted.append(True), busy_message="busy")
            assert admitted == [True]
            runner._submit_job(job_type="image", exclusive=True, target_factory=lambda cb: release.wait(5))
            with pytest.raises(web_runner_module.JobConflictError, match="busy"):
                runner.admit_exclusive(lambda: admitted.append(True), busy_message="busy")
            assert admitted == [True]
        finally:
            release.set()
            runner.shutdown()


class TestJobLifecycle:
    """The enhanced prompt is carried by the job's history."""

    def test_enhanced_prompt_survives_in_prompt_progress(self):
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            job_id = runner._submit_job(job_type="image", target_factory=lambda cb: None)
            runner._publish_event(job_id, {"type": "prompt_started", "prompt": "a fox", "run_index": 0})
            runner._publish_event(job_id, {"type": "prompt_enhanced", "enhanced_prompt": "A fox in snow."})
            runner._publish_event(job_id, {"type": "generation_started"})
            history = runner._get_job(job_id).history
            assert history[-1]["enhanced_prompt"] == "A fox in snow."
            runner._publish_event(job_id, {"type": "prompt_started", "prompt": "a cat", "run_index": 0})
            assert "enhanced_prompt" not in runner._get_job(job_id).history[-1]
        finally:
            runner.shutdown()
