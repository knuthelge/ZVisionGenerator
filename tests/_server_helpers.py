"""Shared helpers for the web server and runner tests."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from PIL import Image

from tests._factories import _make_plan
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import video_clip as video_clip_module
from zvisiongenerator.web import web_runner as web_runner_module
from zvisiongenerator.web.config import WebUiDefaultModels


def make_web_config() -> SimpleNamespace:
    app_config = {
        "generation": {
            "default_ratio": "2:3",
            "default_size": "m",
        },
        "sizes": {
            "2:3": {
                "m": {"width": 832, "height": 1216},
                "l": {"width": 1024, "height": 1536},
            },
        },
    }
    return SimpleNamespace(
        app_config=app_config,
        startup_view="config",
        gallery_page_size=12,
        data_dir="/tmp/.ziv",
        output_dir="/tmp/outputs",
        models_dir="/tmp/models",
        loras_dir="/tmp/loras",
        default_models=WebUiDefaultModels(image="zit", video="ltx-8"),
        image_model_options=("zit", "local-image"),
        video_model_options=("ltx-8",),
        lora_options=("style",),
        image_ratios=("2:3",),
        image_size_options={"2:3": ("m", "l")},
        image_size_dimensions={"2:3": {"m": (832, 1216), "l": (1152, 1728)}},
        video_size_dimensions={"16:9": {"m": (704, 448)}},
        video_ratios=("16:9",),
        video_size_options={"16:9": ("m",)},
        scheduler_options=("beta",),
        quantize_options=(4, 8),
        image_inventory=(),
        video_inventory=(),
        seedvr2=None,
    )


def make_resolved_image_defaults(
    *,
    supports_negative_prompt: bool = True,
    supports_quantize: bool = True,
    supports_img2img: bool = True,
    supports_upscale: bool = True,
    supports_json_prompt: bool = False,
    supports_first_sigma: bool = False,
    supports_scheduler: bool = True,
    dimension_min: int = 16,
    dimension_max: int | None = None,
    dimension_step: int = 16,
) -> dict[str, object]:
    return {
        "steps": 10,
        "guidance": 3.5,
        "scheduler": None,
        "supports_negative_prompt": supports_negative_prompt,
        "supports_quantize": supports_quantize,
        "supports_img2img": supports_img2img,
        "supports_upscale": supports_upscale,
        "supports_json_prompt": supports_json_prompt,
        "supports_first_sigma": supports_first_sigma,
        "supports_scheduler": supports_scheduler,
        "dimension_min": dimension_min,
        "dimension_max": dimension_max,
        "dimension_step": dimension_step,
    }


def patch_image_submit_dependencies(
    monkeypatch,
    *,
    model_info: ImageModelInfo,
    defaults: dict[str, object],
    submitted: list[dict[str, object]] | None = None,
) -> None:
    monkeypatch.setattr(web_server, "resolve_model_path", lambda model, **_: model)
    monkeypatch.setattr(web_server, "detect_image_model", lambda _model: model_info)
    monkeypatch.setattr(web_server, "get_backend_name", lambda: "mflux")
    monkeypatch.setattr(web_server, "validate_scheduler", lambda _scheduler, _config: None)
    monkeypatch.setattr(web_server, "resolve_defaults", lambda _model_info, _config, _cli_overrides, _backend_name: defaults)

    def _capture_submit(**kwargs):
        if submitted is not None:
            submitted.append(kwargs)
        return "job-123"

    monkeypatch.setattr(web_server.web_runner, "submit_image_request_job", _capture_submit)


def write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "teal").save(path)


def wait_for_status(runner: web_runner_module.WebRunner, job_id: str, expected_status: str, *, timeout: float = 1.0) -> dict[str, object]:
    """Poll a job until it reaches the expected status."""
    deadline = time.monotonic() + timeout
    snapshot = runner.get_job_snapshot(job_id)
    while time.monotonic() < deadline:
        snapshot = runner.get_job_snapshot(job_id)
        if snapshot["status"] == expected_status:
            return snapshot
        time.sleep(0.01)
    return snapshot


def sse_payload(frame: str) -> dict[str, object]:
    """Extract the JSON payload from one runner-formatted SSE frame."""
    for line in frame.splitlines():
        if line.startswith("data: "):
            return json.loads(line.removeprefix("data: "))
    raise AssertionError(f"SSE frame did not contain a data field: {frame!r}")


def stub_request_job(monkeypatch, *, run_preflight=None) -> MagicMock:
    """Stub preflight, model loading and memory release so a request job runs only the faked batch runner."""
    backend = MagicMock(name="backend")
    backend.load_model.return_value = (MagicMock(name="model"), MagicMock(name="model_info"))
    monkeypatch.setattr(web_runner_module, "run_preflight", run_preflight or (lambda prompts_data, config, args, **_kw: _make_plan(prompts_data, args, config=config)))
    monkeypatch.setattr(web_runner_module, "get_backend", lambda: backend)
    monkeypatch.setattr(web_runner_module, "get_video_backend", lambda _family: backend)
    monkeypatch.setattr(web_runner_module, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(web_runner_module, "build_video_workflow", lambda _args, **_kw: MagicMock(name="workflow"))
    monkeypatch.setattr(web_runner_module, "release_accelerator_memory", lambda: None)
    return backend


class _Stdin:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Stdout:
    def __init__(self, lines: Iterable[str]) -> None:
        self._lines = list(lines)

    def __iter__(self):
        return iter(self._lines)

    def close(self) -> None:
        pass


class FakeProcess:
    """A scripted ``Popen``: prints *lines*, runs for *running_polls* polls, then exits with *exit_code*."""

    def __init__(self, lines: Iterable[str] = (), *, running_polls: int | None = 0, exit_code: int = 0, wait_timeouts: int = 0) -> None:
        self.stdin = _Stdin()
        self.stdout = _Stdout(lines)
        self._running_polls = running_polls
        self._exit_code = exit_code
        self._wait_timeouts = wait_timeouts
        self.returncode: int | None = None
        self.calls: list[str] = []

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        if self._running_polls is None:
            return None
        if self._running_polls == 0:
            self.returncode = self._exit_code
            return self.returncode
        self._running_polls -= 1
        return None

    def terminate(self) -> None:
        self.calls.append("terminate")

    def kill(self) -> None:
        self.calls.append("kill")
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        self.calls.append("wait")
        if self._wait_timeouts:
            self._wait_timeouts -= 1
            raise subprocess.TimeoutExpired("fake", timeout or 0)
        if self.returncode is None:
            self.returncode = -15
        return self.returncode


class Launcher:
    def __init__(self, process: FakeProcess) -> None:
        self.process = process
        self.args: tuple[Any, ...] = ()
        self.kwargs: dict[str, Any] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> FakeProcess:
        self.args, self.kwargs = args, kwargs
        return self.process


def make_workspace_bootstrap_view() -> dict[str, object]:
    image_defaults = {
        "ratio": "2:3",
        "size": "m",
        "steps": 10,
        "guidance": 3.5,
        "width": 832,
        "height": 1216,
        "scheduler": None,
        "supports_negative_prompt": True,
        "supports_quantize": True,
        "supports_img2img": True,
        "supports_upscale": True,
        "supports_json_prompt": False,
        "supports_first_sigma": False,
        "dimension_min": 16,
        "dimension_max": None,
        "dimension_step": 16,
        "quantize": None,
        "image_strength": 0.5,
        "postprocess": {"sharpen": True, "contrast": False, "saturation": False},
        "upscale": {"enabled": False, "factor": None, "denoise": None, "steps": None, "guidance": None, "sharpen": True, "save_pre": False},
    }
    video_defaults = {
        "ratio": "16:9",
        "size": "m",
        "steps": 8,
        "width": 704,
        "height": 448,
        "frame_count": 49,
        "audio": True,
        "low_memory": True,
        "supports_i2v": True,
        "supports_quantize": False,
        "quantize": None,
        "max_steps": 8,
        "fps": 24,
        "upscale": {"enabled": False, "factor": 2, "steps": None},
    }
    return {
        "image_default_model": "zit",
        "video_default_model": "ltx-8",
        "image_model_defaults": {"zit": image_defaults, "local-image": image_defaults},
        "video_model_defaults": {"ltx-8": video_defaults},
    }


def stub_video_submission(monkeypatch, tmp_path) -> list[dict[str, object]]:
    """Stub model detection and the runner so a video submission reaches ``submit_video_request_job`` and records its kwargs."""
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    submitted: list[dict[str, object]] = []
    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(video_clip_module, "resolve_model_path", lambda model, **_: model)
    monkeypatch.setattr(
        video_clip_module,
        "detect_video_model",
        lambda _model: SimpleNamespace(family="ltx", supports_i2v=True, resolution_alignment=64, frame_alignment=8, default_fps=24),
    )
    monkeypatch.setattr(video_clip_module, "resolve_video_defaults", lambda *_args: {"steps": 8, "width": 704, "height": 448, "num_frames": 49})
    monkeypatch.setattr(web_server, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(web_server.web_runner, "submit_video_request_job", lambda **kwargs: submitted.append(kwargs) or "job-123")
    return submitted


def post_video(data: dict[str, object]):
    with TestClient(web_server.app) as client:
        return client.post("/api/generate", data={"mode": "video", "model": "ltx-8", **data})
