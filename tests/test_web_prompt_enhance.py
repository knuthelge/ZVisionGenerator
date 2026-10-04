"""Tests for the Web UI prompt-enhancement API, generate fields, job lifecycle, and workspace contract."""

from __future__ import annotations

import json
import queue
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from conftest import _make_args
from zvisiongenerator.core.image_types import ImageGenerationRequest
from zvisiongenerator.core.video_types import VideoGenerationRequest
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.prompt_enhance import EnhanceSettings
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import web_runner as web_runner_module
from zvisiongenerator.web import workspace_api as workspace_api_module


class _FakeEnhancer:
    repo = "fake/repo"
    revision = None

    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.messages: list[list[dict[str, str]]] = []

    def generate(self, messages, *, seed, max_tokens, temperature, cancelled=None):
        self.messages.append(messages)
        text = self.outputs.pop(0)
        for index in range(0, len(text), 5):
            if cancelled is not None and cancelled():
                return
            yield text[index : index + 5]

    def close(self) -> None:
        pass


class _FakeSession:
    def __init__(self, enhancer=None, *, busy=False, load_error: Exception | None = None, phase: str | None = "loading"):
        self.enhancer = enhancer
        self._busy = busy
        self.load_error = load_error
        self.phase = phase
        self.acquired: list[tuple] = []

    def busy(self) -> bool:
        return self._busy

    def reserve(self):
        return None if self._busy else object()

    def cancel_reservation(self, token) -> None:
        pass

    @contextmanager
    def acquire(self, repo, revision, *, idle_seconds, on_phase=None):
        self.acquired.append((repo, revision, idle_seconds))
        if on_phase is not None and self.phase:
            on_phase(self.phase)
        if self.load_error is not None:
            raise self.load_error
        yield self.enhancer


def _frames(response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


@pytest.fixture
def no_active_job(monkeypatch):
    monkeypatch.setattr(web_server.web_runner, "get_active_exclusive_job_snapshot", lambda: None)


# ── /api/prompt/enhance ─────────────────────────────────────────────────────


class TestEnhanceEndpoint:
    def test_streams_status_text_done(self, monkeypatch, no_active_job):
        enhancer = _FakeEnhancer(["A grey fox in deep snow, golden light."])
        session = _FakeSession(enhancer)
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: session)
        with TestClient(web_server.app) as client:
            response = client.post("/api/prompt/enhance", json={"prompt": "a {red|grey} fox", "mode": "image", "settings": {"style": "photo"}})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        frames = _frames(response)
        assert frames[0] == {"type": "status", "phase": "loading"}
        assert frames[1] == {"type": "status", "phase": "generating"}
        assert any(frame["type"] == "text" and "grey fox" in frame["text"] for frame in frames)
        assert frames[-1] == {"type": "done", "prompt": "A grey fox in deep snow, golden light.", "clamped": False}
        assert enhancer.messages[0][1]["content"] in ("a red fox", "a grey fox")  # choices are picked before the model sees the prompt
        assert session.acquired[0][2] == 120

    def test_cpu_fallback_is_reported(self, monkeypatch, no_active_job):
        enhancer = _FakeEnhancer(["A red fox in snow, golden light."])
        enhancer.on_cuda = False
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))
        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json={"prompt": "a fox", "settings": {}}))
        assert {"type": "status", "phase": "generating_cpu"} in frames
        assert {"type": "status", "phase": "generating"} not in frames

    def test_load_failure_is_error_frame(self, monkeypatch, no_active_job):
        session = _FakeSession(load_error=RuntimeError("Enhancer model a/b is not downloaded and Hugging Face is offline."))
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: session)
        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json={"prompt": "a fox", "settings": {}}))
        assert frames[-1] == {"type": "error", "detail": "Enhancer model a/b is not downloaded and Hugging Face is offline."}

    def test_unusable_output_is_error_frame(self, monkeypatch, no_active_job):
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(_FakeEnhancer(["", "a fox"])))
        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json={"prompt": "a fox", "settings": {}}))
        assert frames[-1]["type"] == "error" and "no usable rewrite" in frames[-1]["detail"]

    def test_conflict_while_job_runs(self, monkeypatch):
        def _job_running(admit, *, busy_message):
            raise web_runner_module.JobConflictError(busy_message)

        monkeypatch.setattr(web_server.web_runner, "admit_exclusive", _job_running)
        with TestClient(web_server.app) as client:
            response = client.post("/api/prompt/enhance", json={"prompt": "a fox", "settings": {}})
        assert response.status_code == 409
        assert "all jobs have finished" in response.json()["detail"]

    def test_conflict_while_enhancing(self, monkeypatch, no_active_job):
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(busy=True))
        with TestClient(web_server.app) as client:
            assert client.post("/api/prompt/enhance", json={"prompt": "a fox", "settings": {}}).status_code == 409

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            ({"prompt": " ", "settings": {}}, "Enter a prompt"),
            ({"prompt": "a fox", "settings": {"style": "keep", "details": [], "length": "same"}}, "Nothing to enhance"),
            ({"prompt": "a fox", "settings": {"style": "nope"}}, "Unknown style"),
            ({"prompt": "a fox", "mode": "audio"}, "mode must be"),
            ({"prompt": "a fox", "max_words": 0}, "max_words"),
            ([1, 2], "JSON object"),
        ],
    )
    def test_validation(self, body, message, no_active_job):
        with TestClient(web_server.app) as client:
            response = client.post("/api/prompt/enhance", json=body)
        assert response.status_code == 422
        assert message in response.json()["detail"]

    def test_max_words_is_used(self, monkeypatch):
        job = web_server._validate_enhance_body({"prompt": "a fox", "settings": {"length": "extra"}, "max_words": 180})
        assert job["ceiling"] == 180 and job["settings"].length == "extra"

    def test_cancel_reports_error_and_stops(self):
        frames: queue.Queue = queue.Queue()
        cancelled = threading.Event()
        cancelled.set()
        session = _FakeSession(_FakeEnhancer(["A long rewrite."]))
        job = {"prompt": "a fox", "mode": "image", "settings": EnhanceSettings(), "ceiling": 300, "repo": "a/b", "revision": None, "options": web_server.enhance_options({}), "idle_seconds": 120}
        original = web_server.get_prompt_enhancer_session
        web_server.get_prompt_enhancer_session = lambda: session
        try:
            web_server._run_enhancement(job, frames, cancelled)
        finally:
            web_server.get_prompt_enhancer_session = original
        collected = []
        while not frames.empty():
            collected.append(frames.get())
        assert collected[-1] == {"type": "error", "detail": "Prompt enhancement was cancelled."}


# ── /api/generate enhance fields ────────────────────────────────────────────


class TestGenerateFields:
    def test_apply_auto_settings(self):
        args = SimpleNamespace()
        web_server._apply_enhance_submission(args, {"enhance_auto": "true", "enhance_settings": json.dumps({"style": "anime", "length": "longer"})}, mode="image", json_caption=False)
        assert args.enhance == EnhanceSettings(style="anime", length="longer")
        assert args.no_enhance is False and args.enhance_model is None

    def test_toggle_off(self):
        args = SimpleNamespace()
        web_server._apply_enhance_submission(args, {"enhance_settings": "{}"}, mode="image", json_caption=False)
        assert args.enhance is None

    def test_json_caption_never_enhanced(self):
        args = SimpleNamespace()
        web_server._apply_enhance_submission(args, {"enhance_auto": "on"}, mode="image", json_caption=True)
        assert args.enhance is None

    @pytest.mark.parametrize(("raw", "message"), [("{bad", "JSON object"), ('{"style": "keep", "details": [], "length": "same", "motion": []}', "Nothing to enhance")])
    def test_invalid(self, raw, message):
        with pytest.raises(ValueError, match=message):
            web_server._apply_enhance_submission(SimpleNamespace(), {"enhance_auto": "true", "enhance_settings": raw}, mode="video", json_caption=False)

    def test_image_submit_threads_settings_and_file_entries(self, monkeypatch, tmp_path):
        from test_web_server import _make_web_config, _patch_image_submit_dependencies

        submitted: list[dict] = []
        _patch_image_submit_dependencies(
            monkeypatch,
            model_info=ImageModelInfo(family="zimage", is_distilled=False, size=None),
            defaults={"steps": 4, "guidance": 1.0, "scheduler": None, "supports_negative_prompt": True},
            submitted=submitted,
        )
        prompt_file = tmp_path / "p.yaml"
        prompt_file.write_text("a:\n  - prompt: x\n    enhance: {style: anime}\n  - prompt: y\n", encoding="utf-8")
        web_config = _make_web_config()
        web_config.output_dir = str(tmp_path)

        class _Form(dict):
            def getlist(self, key):
                value = self.get(key)
                return value if isinstance(value, list) else [value]

        web_server._submit_image_job(_Form({"prompt_source": "file", "prompts_file": str(prompt_file), "prompt_option_id": ["a:0", "a:1"]}), web_config)
        assert submitted[0]["enhance_by_set"] == {"a": [EnhanceSettings(style="anime"), None]}
        assert submitted[0]["args"].enhance is None

    def test_job_admission_rejects_while_enhancing(self, monkeypatch):
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(busy=True))
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            with pytest.raises(web_runner_module.JobConflictError, match="Prompt enhancement in progress"):
                runner._submit_job(job_type="image", exclusive=True, target_factory=lambda cb: None, admission_check=web_server._reject_while_busy)
            assert runner.get_active_exclusive_job_snapshot() is None
        finally:
            runner.shutdown()

    def test_job_queues_while_an_auto_enhance_job_holds_the_enhancer(self, monkeypatch):
        """While a job's own preflight holds the enhancer, a second job queues behind it instead of being refused."""
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(busy=True))
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


# ── Job lifecycle (web_runner) ──────────────────────────────────────────────


class TestJobLifecycle:
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


# ── Preflight in web jobs ───────────────────────────────────────────────────

_PREFLIGHT_CONFIG = {
    "sizes": {"2:3": {"m": {"width": 512, "height": 512}}},
    "prompt_enhancer": {"model": {"darwin": "a/b", "win32": "a/b", "linux": "a/b"}},
}


class _PreflightEnhancer:
    def __init__(self, order: list[str], *, gate: threading.Event | None = None):
        self.order = order
        self.gate = gate

    def generate(self, messages, *, seed, max_tokens, temperature, cancelled=None):
        if self.gate is not None:
            self.gate.wait(timeout=2.0)
        for chunk in ("A red fox ", "in deep snow, ", "golden light."):
            if cancelled is not None and cancelled():
                return
            yield chunk

    def close(self) -> None:
        self.order.append("close")


def _wait_terminal(runner, job_id: str, *, timeout: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = runner.get_job_snapshot(job_id)
        if snapshot["status"] in ("completed", "failed", "cancelled"):
            return snapshot
        time.sleep(0.01)
    return runner.get_job_snapshot(job_id)


class TestWebPreflight:
    """Web jobs run preflight before the model loads; rewrites and the enhancer never overlap the model."""

    def _env(self, monkeypatch, order: list[str], *, enhancer=None, load_error: Exception | None = None):
        from zvisiongenerator.backends.prompt_enhancer_session import PromptEnhancerSession

        def _factory(repo, revision):
            if load_error is not None:
                raise load_error
            return enhancer or _PreflightEnhancer(order)

        session = PromptEnhancerSession(_factory, scheduler=lambda delay, callback: None, downloaded=lambda repo, revision: True)
        monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.get_prompt_enhancer_session", lambda: session)
        monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.ensure_available", lambda repo, revision: True)
        monkeypatch.setattr("zvisiongenerator.preflight.release_accelerator_memory", lambda: order.append("release_memory"))
        monkeypatch.setattr(web_runner_module, "release_accelerator_memory", lambda: None)
        backend = MagicMock()
        backend.load_model.side_effect = lambda *a, **k: (order.append("load_model"), (MagicMock(), MagicMock(family="zimage")))[1]
        monkeypatch.setattr(web_runner_module, "get_backend", lambda: backend)
        monkeypatch.setattr(web_runner_module, "get_video_backend", lambda _family: backend)
        monkeypatch.setattr(web_runner_module, "build_video_workflow", lambda _args, **_kw: MagicMock())

        def _fake_run_batch(backend, model, prompts_data, config, args, model_info, *, plan, progress_callback, skip_signal):
            order.append("run_batch")
            iteration = plan.iterations[0]
            progress_callback({"type": "prompt_started", "mode": "image", "prompt": iteration.prompt, "enhance_status": iteration.enhance_status.value, "run_index": 0})
            progress_callback({"type": "generation_started", "mode": "image"})
            progress_callback({"type": "batch_completed", "mode": "image", "completed_iterations": 1, "total_iterations": 1})

        def _fake_run_video_batch(*, backend, model, model_info, workflow, prompts_data, config, args, plan, progress_callback):
            order.append("run_video_batch")

        monkeypatch.setattr(web_runner_module, "run_batch", _fake_run_batch)
        monkeypatch.setattr(web_runner_module, "run_video_batch", _fake_run_video_batch)

    def _submit_image(self, runner, **args_overrides):
        args = _make_args(enhance=EnhanceSettings(), no_enhance=False, **args_overrides)
        request = ImageGenerationRequest(backend=None, model=None, prompt="a red fox", model_family="zimage")
        return runner.submit_image_request_job(request=request, prompts_data={"prompt": [("a red fox", None)]}, config=_PREFLIGHT_CONFIG, args=args, model_ref="model")

    @staticmethod
    def _types(runner, job_id) -> list[str]:
        return [event["type"] for event in runner._get_job(job_id).history]

    def test_preflight_runs_before_model_loading(self, monkeypatch):
        order: list[str] = []
        self._env(monkeypatch, order)
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            job_id = self._submit_image(runner)
            assert _wait_terminal(runner, job_id)["status"] == "completed"
            types = self._types(runner, job_id)
            assert types.index("preflight_started") < types.index("prompts_enhancing") < types.index("preflight_finished") < types.index("model_loading")
            assert order == ["close", "release_memory", "load_model", "run_batch"]
            history = runner._get_job(job_id).history
            started = next(e for e in history if e["type"] == "prompt_started")
            later = next(e for e in history if e["type"] == "generation_started")
            assert started["enhance_status"] == "enhanced" and later["enhance_status"] == "enhanced"
        finally:
            runner.shutdown()

    def test_quit_during_preflight_cancels_before_model_load(self, monkeypatch):
        order: list[str] = []
        gate = threading.Event()
        self._env(monkeypatch, order, enhancer=_PreflightEnhancer(order, gate=gate))
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            job_id = self._submit_image(runner)
            deadline = time.monotonic() + 2.0
            while "prompts_enhancing" not in self._types(runner, job_id) and time.monotonic() < deadline:
                time.sleep(0.01)
            runner.queue_job_control(job_id, "quit")
            gate.set()
            snapshot = _wait_terminal(runner, job_id)
            assert snapshot["status"] == "cancelled" and snapshot["terminal_event"] == "job_cancelled"
            assert "load_model" not in order and "model_loading" not in self._types(runner, job_id)
            assert order == ["close", "release_memory"]
        finally:
            gate.set()
            runner.shutdown()

    def test_enhancer_load_failure_fails_before_model_loading(self, monkeypatch):
        order: list[str] = []
        self._env(monkeypatch, order, load_error=RuntimeError("Could not load prompt enhancer model a/b"))
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            job_id = self._submit_image(runner)
            snapshot = _wait_terminal(runner, job_id)
            assert snapshot["status"] == "failed" and "Could not load prompt enhancer" in snapshot["last_event"]["message"]
            assert "model_loading" not in self._types(runner, job_id) and "load_model" not in order
        finally:
            runner.shutdown()

    def test_video_checks_ffmpeg_before_preflight(self, monkeypatch):
        order: list[str] = []
        self._env(monkeypatch, order)
        monkeypatch.setattr(web_runner_module, "require_ffmpeg", lambda: order.append("require_ffmpeg"))
        runner = web_runner_module.WebRunner(max_workers=1, heartbeat_seconds=0.01)
        try:
            from tests.conftest import _make_video_args

            request = VideoGenerationRequest(backend=None, model=None, prompt="a fox runs", model_family="ltx")
            args = _make_video_args(enhance=EnhanceSettings(), no_enhance=False)
            job_id = runner.submit_video_request_job(request=request, prompts_data={"prompt": [("a fox runs", None)]}, config=_PREFLIGHT_CONFIG, args=args, model_ref="model")
            assert _wait_terminal(runner, job_id)["status"] == "completed"
            assert order == ["require_ffmpeg", "close", "release_memory", "load_model", "run_video_batch"]
            types = self._types(runner, job_id)
            assert types.index("preflight_finished") < types.index("model_loading")
        finally:
            runner.shutdown()


# ── Workspace contract ──────────────────────────────────────────────────────


class TestWorkspaceContract:
    def _config(self, **section):
        return SimpleNamespace(
            app_config={
                "prompt_enhancer": {
                    "model": {"darwin": "a/b", "win32": "a/b", "linux": "a/b"},
                    "revision": {"darwin": "r", "win32": "r", "linux": "r"},
                    "download_size_label": {"darwin": "2.4 GB", "win32": "9 GB", "linux": "9 GB"},
                    **section,
                }
            }
        )

    def test_not_downloaded(self):
        contract = workspace_api_module.build_prompt_enhancer_contract(self._config(), downloaded=lambda repo, rev: False)
        assert contract["model"] == "a/b" and contract["revision"] == "r"
        assert contract["downloaded"] is False and contract["download_size_label"] in ("2.4 GB", "9 GB")
        assert contract["matrix"]["defaults"]["length"] == "same"
        assert contract["default_max_words"] == 300

    def test_downloaded_hides_size(self, tmp_path):
        contract = workspace_api_module.build_prompt_enhancer_contract(self._config(), downloaded=lambda repo, rev: True)
        assert contract["downloaded"] is True and contract["download_size_label"] is None

    def test_custom_model_has_no_default_size_label(self):
        config = self._config(user_model="someone/bigger-llm")
        contract = workspace_api_module.build_prompt_enhancer_contract(config, downloaded=lambda repo, rev: False)
        assert contract["model"] == "someone/bigger-llm" and contract["download_size_label"] is None

    def test_missing_model_reports_error(self):
        contract = workspace_api_module.build_prompt_enhancer_contract(SimpleNamespace(app_config={}), downloaded=lambda repo, rev: False)
        assert contract["model"] is None and "No prompt enhancer model" in contract["error"]

    def test_workflows_expose_enhance_controls(self):
        from zvisiongenerator.web.workspace_contract import WORKFLOW_DEFINITIONS

        for definition in WORKFLOW_DEFINITIONS.values():
            assert {"prompt_enhance", "prompt_enhance_auto"} <= set(definition["visible_controls"])


# ── Config fields ───────────────────────────────────────────────────────────


class TestConfigFields:
    @pytest.mark.parametrize("value", ["owner/model", "McG-221/Qwen3.5-4B-heretic-mlx-4Bit", "owner/model@main"])
    def test_model_accepts_repo(self, value):
        from zvisiongenerator.web.config_contract import WRITABLE_CONFIG_FIELDS, _validate_config_value

        field = next(f for f in WRITABLE_CONFIG_FIELDS if f.key == "prompt_enhancer.user_model")
        assert _validate_config_value(field, value, None) == value

    def test_model_accepts_local_dir(self, tmp_path):
        from zvisiongenerator.web.config_contract import WRITABLE_CONFIG_FIELDS, _validate_config_value

        field = next(f for f in WRITABLE_CONFIG_FIELDS if f.key == "prompt_enhancer.user_model")
        assert _validate_config_value(field, str(tmp_path), None) == str(tmp_path.resolve())

    @pytest.mark.parametrize(("key", "value"), [("prompt_enhancer.user_model", "not a repo"), ("prompt_enhancer.user_model", "owner/model@bad rev!")])
    def test_rejects(self, key, value):
        from zvisiongenerator.web.config_contract import WRITABLE_CONFIG_FIELDS, _validate_config_value

        field = next(f for f in WRITABLE_CONFIG_FIELDS if f.key == key)
        with pytest.raises(ValueError):
            _validate_config_value(field, value, None)
