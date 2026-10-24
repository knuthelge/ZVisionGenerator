"""Tests for MiniMax H3 video requests through the Web UI server: the last-frame image, step snapping, the extension step and the memory-fit family."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import make_extend_source
from PIL import Image

import zvisiongenerator.backends as backends_module
from zvisiongenerator.utils.config import load_config, video_family
from zvisiongenerator.utils.model_memory import MemoryBudget
from zvisiongenerator.utils.video_extend import window_frames
from zvisiongenerator.web import config as web_config_module
from zvisiongenerator.web import extend_api as extend_api_module
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web.config import WebUiDefaultModels


@pytest.fixture(autouse=True)
def _linux(monkeypatch):
    monkeypatch.setattr(backends_module.sys, "platform", "linux")


def _web_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        app_config=load_config(),
        output_dir=str(tmp_path),
        default_models=WebUiDefaultModels(image="zit", video="h3"),
        video_model_options=("h3", "ltx-8"),
        video_ratios=("16:9",),
        video_size_options={"16:9": ("xs", "s")},
        video_size_dimensions={"16:9": {"xs": (576, 320), "s": (768, 448)}},
    )


@pytest.fixture
def submitted(monkeypatch) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    monkeypatch.setattr(web_server, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(web_server.web_runner, "submit_video_request_job", lambda **kwargs: jobs.append(kwargs) or "job-1")
    return jobs


def _image(tmp_path: Path, name: str) -> str:
    path = tmp_path / name
    Image.new("RGB", (8, 8)).save(path)
    return str(path)


def _submit(tmp_path: Path, **form: str) -> dict[str, Any]:
    return web_server._submit_video_job({"model": "h3", "prompt": "a lake", **form}, _web_config(tmp_path))


class TestEndImage:
    def test_the_end_image_reaches_the_args_and_the_request(self, tmp_path, submitted):
        first, last = _image(tmp_path, "first.png"), _image(tmp_path, "last.png")

        _submit(tmp_path, image_path=first, end_image_path=last)

        (job,) = submitted
        assert job["args"].end_image_path == str(Path(last).resolve())
        assert job["request"].end_image_path == str(Path(last).resolve())
        assert job["request"].model_family == "h3"
        assert job["request"].image_path == str(Path(first).resolve())

    def test_no_end_image_leaves_it_unset(self, tmp_path, submitted):
        _submit(tmp_path, image_path=_image(tmp_path, "first.png"))

        (job,) = submitted
        assert job["args"].end_image_path is None
        assert job["request"].end_image_path is None

    def test_an_end_image_needs_a_first_frame(self, tmp_path, submitted):
        with pytest.raises(ValueError, match="last-frame image needs a first-frame"):
            _submit(tmp_path, end_image_path=_image(tmp_path, "last.png"))

        assert submitted == []

    def test_a_missing_end_image_file_is_refused(self, tmp_path, submitted):
        with pytest.raises(ValueError, match="end_image_path"):
            _submit(tmp_path, image_path=_image(tmp_path, "first.png"), end_image_path=str(tmp_path / "gone.png"))

        assert submitted == []

    def test_a_family_without_the_capability_refuses_it(self, tmp_path, submitted):
        first, last = _image(tmp_path, "first.png"), _image(tmp_path, "last.png")

        with pytest.raises(ValueError, match="does not support a last-frame image"):
            _submit(tmp_path, model="ltx-8", image_path=first, end_image_path=last)

        assert submitted == []

    def test_the_family_that_has_it_is_not_refused_for_anything_else(self, tmp_path, submitted):
        """H3 offers i2v, extend, upscale with review and audio: none of them is refused."""
        _submit(tmp_path, image_path=_image(tmp_path, "first.png"), upscale="2", review="on")

        (job,) = submitted
        assert job["args"].review is True


class TestStepsAndFrames:
    @pytest.mark.parametrize(("steps", "settled"), [(None, 8), (4, 4), (8, 8), (5, 4), (6, 8), (20, 8)])
    def test_steps_move_to_the_nearest_option(self, tmp_path, submitted, steps, settled):
        _submit(tmp_path, **({} if steps is None else {"steps": str(steps)}))

        assert submitted[0]["args"].steps == settled
        assert submitted[0]["request"].steps == settled

    def test_the_default_clip_is_five_seconds(self, tmp_path, submitted):
        _submit(tmp_path)

        assert submitted[0]["args"].num_frames == 124

    @pytest.mark.filterwarnings("ignore:Video model requires")
    @pytest.mark.parametrize(("frames", "aligned"), [(22, 22), (23, 39), (100, 107), (124, 124), (240, 243)])
    def test_frames_move_up_to_the_next_17k_plus_5(self, tmp_path, submitted, frames, aligned):
        _submit(tmp_path, frames=str(frames))

        assert submitted[0]["args"].num_frames == aligned

    def test_an_upscaled_clip_runs_both_stages_at_the_settled_steps(self, tmp_path, submitted):
        _submit(tmp_path, upscale="2", steps="4")

        args = submitted[0]["args"]
        assert (args.steps, args.upscale_steps) == (4, 4)


class TestUpscaleSupport:
    def test_h3_on_macos_refuses_an_upscale_before_a_job_is_submitted(self, tmp_path, submitted, monkeypatch):
        monkeypatch.setattr(backends_module.sys, "platform", "darwin")

        with pytest.raises(ValueError, match="MiniMax H3 can't upscale a clip on this platform"):
            _submit(tmp_path, upscale="2")

        assert submitted == []

    def test_h3_on_macos_still_renders_without_an_upscale(self, tmp_path, submitted, monkeypatch):
        monkeypatch.setattr(backends_module.sys, "platform", "darwin")

        _submit(tmp_path)

        assert submitted[0]["args"].upscale is None

    def test_ltx_on_macos_and_h3_on_cuda_still_upscale(self, tmp_path, submitted, monkeypatch):
        _submit(tmp_path, upscale="2")
        monkeypatch.setattr(backends_module.sys, "platform", "darwin")
        _submit(tmp_path, model="ltx-8", upscale="2")

        assert [job["args"].upscale for job in submitted] == [2, 2]


class TestExtend:
    @pytest.fixture(autouse=True)
    def _extendable(self, monkeypatch):
        source = make_extend_source(fps=24.0, frame_count=97)
        monkeypatch.setattr(web_server, "load_source_video", lambda *_args, **_kwargs: source)
        monkeypatch.setattr(web_server, "supports_video_extend", lambda: True)
        monkeypatch.setattr(web_server, "supports_video_audio", lambda: True)
        self.source = source

    def test_the_length_is_a_multiple_of_17(self, tmp_path, submitted):
        _submit(tmp_path, workflow="extend", source_video="clip.mp4", extend_frames="34")

        args = submitted[0]["args"]
        assert args.extend_frames == 34
        assert args.num_frames == 34 + 22  # the window carries the 22 tail frames
        assert submitted[0]["request"].model_family == "h3"
        assert submitted[0]["request"].num_frames == 56

    def test_a_source_that_is_not_24_fps_is_refused_before_the_job_is_submitted(self, tmp_path, submitted, monkeypatch):
        slow = make_extend_source(fps=30.0, fps_text="30/1")
        monkeypatch.setattr(web_server, "load_source_video", lambda *_args, **_kwargs: slow)

        with pytest.raises(ValueError, match="24 fps clips only"):
            _submit(tmp_path, workflow="extend", source_video="clip.mp4")

        assert submitted == []

    def test_an_ltx_extension_accepts_a_30_fps_source(self, tmp_path, submitted, monkeypatch):
        monkeypatch.setattr(web_server, "load_source_video", lambda *_args, **_kwargs: make_extend_source(fps=30.0, fps_text="30/1"))

        _submit(tmp_path, workflow="extend", source_video="clip.mp4", model="ltx-8", extend_frames="16")

        assert len(submitted) == 1

    def test_an_ltx_extension_carries_25_frames(self, tmp_path, submitted):
        _submit(tmp_path, workflow="extend", source_video="clip.mp4", model="ltx-8", extend_frames="16")

        assert (submitted[0]["args"].num_frames, submitted[0]["request"].num_frames) == (25 + 16, 25 + 16)

    @pytest.mark.parametrize("frames", ["8", "16", "30"])
    def test_another_length_is_refused(self, tmp_path, submitted, frames):
        with pytest.raises(ValueError, match="multiple of 17"):
            _submit(tmp_path, workflow="extend", source_video="clip.mp4", extend_frames=frames)

        assert submitted == []

    @pytest.mark.parametrize("seconds", ["1", "2", "5.5"])
    def test_a_length_in_seconds_snaps_to_a_multiple_of_17(self, tmp_path, submitted, seconds):
        _submit(tmp_path, workflow="extend", source_video="clip.mp4", extend_seconds=seconds)

        assert submitted[0]["args"].extend_frames % 17 == 0

    def test_the_default_length_is_a_multiple_of_17(self, tmp_path, submitted):
        _submit(tmp_path, workflow="extend", source_video="clip.mp4")

        assert submitted[0]["args"].extend_frames % 17 == 0

    def test_the_end_image_is_ignored_for_an_extension(self, tmp_path, submitted):
        _submit(tmp_path, workflow="extend", source_video="clip.mp4", end_image_path=str(tmp_path / "gone.png"))

        assert submitted[0]["args"].end_image_path is None

    def test_steps_snap_for_an_extension(self, tmp_path, submitted):
        _submit(tmp_path, workflow="extend", source_video="clip.mp4", steps="6")

        assert submitted[0]["args"].steps == 8


class TestExtendMemoryFit:
    def test_the_memory_fit_is_asked_for_the_family_of_the_model(self, tmp_path, monkeypatch):
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(extend_api_module, "memory_budget", lambda: MemoryBudget(24 * 2**30))
        monkeypatch.setattr(extend_api_module, "clip_memory_fit", lambda model, **kwargs: calls.append({"model": model, **kwargs}))

        extend_api_module._extend_memory_fit(("/models/h3", "h3"), "h3", make_extend_source(), 34, True, _web_config(tmp_path))
        extend_api_module._extend_memory_fit(("/models/ltx", "ltx"), "ltx-8", make_extend_source(), 48, True, _web_config(tmp_path))

        assert [(call["model"], call["family"]) for call in calls] == [("/models/h3", "h3"), ("/models/ltx", "ltx")]

    def test_the_fit_is_for_the_extend_mode_at_the_families_window_and_default_steps(self, tmp_path, monkeypatch):
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(extend_api_module, "memory_budget", lambda: MemoryBudget(24 * 2**30))
        monkeypatch.setattr(extend_api_module, "clip_memory_fit", lambda model, **kwargs: calls.append(kwargs))

        extend_api_module._extend_memory_fit(("/models/h3", "h3"), "h3", make_extend_source(), 34, True, _web_config(tmp_path))
        extend_api_module._extend_memory_fit(("/models/ltx", "ltx"), "ltx-8", make_extend_source(), 48, True, _web_config(tmp_path))

        assert [(call["mode"], call["frames"], call["forwards"]) for call in calls] == [("extend", 22 + 34, 8), ("extend", 25 + 48, 8)]

    def test_a_discrete_gpu_gets_an_estimated_time_for_the_window(self, tmp_path, monkeypatch):
        monkeypatch.setattr(extend_api_module, "memory_budget", lambda: MemoryBudget(10 * 2**30, system_bytes=30 * 2**30))

        fit = extend_api_module._extend_memory_fit(("/models/h3", "h3"), "h3", make_extend_source(width=768, height=448), 34, True, _web_config(tmp_path))

        assert fit is not None
        assert (fit["kind"], fit["family"]) == ("discrete", "h3")
        assert fit["expected_seconds"] > 0

    def test_a_family_without_an_estimate_has_no_fit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(extend_api_module, "memory_budget", lambda: MemoryBudget(24 * 2**30))

        fit = extend_api_module._extend_memory_fit(("/models/h3", "h3"), "h3", make_extend_source(), 59, True, _web_config(tmp_path))

        assert fit is None

    @pytest.mark.parametrize(("family", "extend_frames", "window"), [("h3", 34, 56), ("h3", 17, 39), ("ltx", 48, 73)])
    def test_the_window_is_the_carried_frames_plus_the_new_ones(self, family, extend_frames, window):
        assert window_frames(extend_frames, video_family(load_config(), family)) == window

    def test_the_source_response_reports_the_window_of_the_models_family(self, tmp_path, monkeypatch):
        monkeypatch.setattr(extend_api_module, "memory_budget", lambda: None)
        monkeypatch.setattr(extend_api_module, "_inspect_source", lambda _path: SimpleNamespace(probe=None, recorded=None, problems=(), source=None))
        monkeypatch.setattr(extend_api_module, "_source_video_path", lambda *_a, **_k: tmp_path / "clip.mp4")
        monkeypatch.setattr(extend_api_module, "output_media_url", lambda *_a, **_k: None)
        monkeypatch.setattr(extend_api_module, "detect_video_model", lambda path: SimpleNamespace(family="h3" if "h3" in path or "H3" in path else "ltx"))

        payload = extend_api_module.extend_source_response({"path": "clip.mp4", "extend_frames": 17, "model": "h3"}, _web_config(tmp_path), tmp_path)

        assert (payload["extend_frames"], payload["window_frames"]) == (17, 39)


class TestMemoryFitHints:
    """``POST /api/video/memory-fit`` takes an ``image`` flag and ``steps``; they set the mode and the forwards of MiniMax H3's estimate."""

    @pytest.fixture
    def asked(self, tmp_path, monkeypatch) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(web_server, "memory_budget", lambda: MemoryBudget(10 * 2**30, system_bytes=30 * 2**30))
        monkeypatch.setattr(web_server, "clip_memory_fit", lambda model, **kwargs: calls.append(kwargs))
        return calls

    def _fit(self, tmp_path, **body):
        return web_server._video_memory_fit_response({"model": "h3", "ratio": "16:9", "size": "s", **body}, _web_config(tmp_path))

    def test_a_text_to_video_clip_defaults_to_t2v_at_the_default_steps(self, tmp_path, asked):
        self._fit(tmp_path)

        assert (asked[0]["family"], asked[0]["mode"], asked[0]["forwards"]) == ("h3", "t2v", 8)

    def test_the_image_flag_makes_it_an_image_to_video_clip(self, tmp_path, asked):
        self._fit(tmp_path, image=True)

        assert asked[0]["mode"] == "i2v"

    @pytest.mark.parametrize(("steps", "forwards"), [(4, 4), (5, 4), (8, 8), (7, 8)])
    def test_the_steps_are_settled_by_the_families_options(self, tmp_path, asked, steps, forwards):
        self._fit(tmp_path, steps=steps)

        assert asked[0]["forwards"] == forwards

    @pytest.mark.parametrize("body", [{"image": "yes"}, {"steps": 0}, {"steps": "4"}])
    def test_a_bad_hint_is_refused(self, tmp_path, asked, body):
        with pytest.raises(ValueError):
            self._fit(tmp_path, **body)

    def test_an_ltx_model_ignores_the_hints_for_its_estimate(self, tmp_path, asked):
        self._fit(tmp_path, model="ltx-8", steps=4)

        assert asked[0]["family"] == "ltx"


class TestExtendEnhanceContext:
    @pytest.fixture(autouse=True)
    def _extendable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(extend_api_module, "supports_video_extend", lambda: True)
        monkeypatch.setattr(extend_api_module, "supports_video_audio", lambda: True)
        monkeypatch.setattr(extend_api_module, "load_source_video", lambda *_args, **_kwargs: make_extend_source())

    def test_the_named_model_sets_the_step(self):
        context = extend_api_module.extend_context_from_body({"source_video": "clip.mp4", "extend_frames": 34, "model": "h3"}, "")

        assert context.extends is True
        assert context.new_seconds == pytest.approx(34 / 24, abs=0.01)

    def test_a_source_that_is_not_24_fps_is_refused_for_h3_only(self, monkeypatch):
        monkeypatch.setattr(extend_api_module, "load_source_video", lambda *_args, **_kwargs: make_extend_source(fps=30.0, fps_text="30/1"))
        body = {"source_video": "clip.mp4", "extend_frames": 17, "model": "h3"}

        with pytest.raises(ValueError, match="24 fps clips only"):
            extend_api_module.extend_context_from_body(body, "")

        assert extend_api_module.extend_context_from_body({**body, "extend_frames": 16, "model": "ltx-8"}, "").extends is True

    @pytest.mark.parametrize(("model", "extend_frames", "window"), [("h3", 17, 39), (None, 16, 41), ("ltx-8", 16, 41)])
    def test_the_window_carries_the_frames_of_the_models_family(self, model, extend_frames, window):
        body = {"source_video": "clip.mp4", "extend_frames": extend_frames, **({} if model is None else {"model": model})}

        context = extend_api_module.extend_context_from_body(body, "")

        assert context.seconds == pytest.approx(window / 24, abs=0.01)

    def test_the_model_inventory_is_not_scanned(self, monkeypatch):
        def _scanned(*_args, **_kwargs):
            raise AssertionError("the model inventory was scanned")

        monkeypatch.setattr(web_config_module, "discover_image_inventory", _scanned)
        monkeypatch.setattr(web_config_module, "discover_video_inventory", _scanned)
        monkeypatch.setattr(web_config_module, "list_loras", _scanned)

        context = extend_api_module.extend_context_from_body({"source_video": "clip.mp4", "extend_frames": 34, "model": "h3"}, "")

        assert context.extends is True

    def test_a_length_off_the_models_step_is_refused(self):
        with pytest.raises(ValueError, match="multiple of 17"):
            extend_api_module.extend_context_from_body({"source_video": "clip.mp4", "extend_frames": 16, "model": "h3"}, "")

    def test_without_a_model_the_ltx_step_applies(self):
        assert extend_api_module.extend_context_from_body({"source_video": "clip.mp4", "extend_frames": 16}, "").new_seconds == pytest.approx(16 / 24, abs=0.01)
        with pytest.raises(ValueError, match="multiple of 8"):
            extend_api_module.extend_context_from_body({"source_video": "clip.mp4", "extend_frames": 17}, "")

    def test_the_model_must_be_a_string(self):
        with pytest.raises(ValueError, match="context.model must be a string"):
            extend_api_module.extend_context_from_body({"source_video": "clip.mp4", "model": 3}, "")
