"""Tests for MiniMax H3 video requests through the Web UI server: the last-frame image, step snapping, the extension step and the memory-fit family."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from tests._factories import load_packaged_config
from tests._factories import make_extend_source

import zvisiongenerator.backends as backends_module
from zvisiongenerator.utils.config import load_config
from zvisiongenerator.utils.model_memory import MemoryBudget
from zvisiongenerator.web import config as web_config_module
from zvisiongenerator.web import extend_api as extend_api_module
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web.config import WebUiDefaultModels
from zvisiongenerator.web.lora_fit import lora_bytes_for_fit


@pytest.fixture(autouse=True)
def _linux(monkeypatch):
    monkeypatch.setattr(backends_module.sys, "platform", "linux")


def _web_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        app_config=load_packaged_config(),
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


def _submit(tmp_path: Path, config: SimpleNamespace | None = None, **form: str) -> dict[str, Any]:
    return web_server._submit_video_job({"model": "h3", "prompt": "a lake", **form}, config or _web_config(tmp_path))


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
        with pytest.raises(ValueError):
            _submit(tmp_path, end_image_path=_image(tmp_path, "last.png"))

        assert submitted == []

    def test_a_missing_end_image_file_is_refused(self, tmp_path, submitted):
        with pytest.raises(ValueError, match="end_image_path"):
            _submit(tmp_path, image_path=_image(tmp_path, "first.png"), end_image_path=str(tmp_path / "gone.png"))

        assert submitted == []

    def test_the_family_that_has_it_is_not_refused_for_anything_else(self, tmp_path, submitted):
        """H3 offers i2v, extend, upscale with review and audio: none of them is refused."""
        _submit(tmp_path, image_path=_image(tmp_path, "first.png"), upscale="2", review="on")

        (job,) = submitted
        assert job["args"].review is True


class TestStepsAndFrames:
    @pytest.mark.parametrize(("steps", "settled"), [(None, 8), (5, 4)])
    def test_steps_move_to_the_nearest_option(self, tmp_path, submitted, steps, settled):
        _submit(tmp_path, **({} if steps is None else {"steps": str(steps)}))

        assert submitted[0]["args"].steps == settled
        assert submitted[0]["request"].steps == settled

    def test_the_default_clip_is_five_seconds(self, tmp_path, submitted):
        _submit(tmp_path)

        assert submitted[0]["args"].num_frames == 124

    @pytest.mark.filterwarnings("ignore:Video model requires")
    @pytest.mark.parametrize(("frames", "aligned"), [(23, 39), (124, 124)])
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

    def test_another_length_is_refused(self, tmp_path, submitted):
        with pytest.raises(ValueError, match="multiple of 17"):
            _submit(tmp_path, workflow="extend", source_video="clip.mp4", extend_frames="30")

        assert submitted == []

    def test_a_length_in_seconds_snaps_to_a_multiple_of_17(self, tmp_path, submitted):
        _submit(tmp_path, workflow="extend", source_video="clip.mp4", extend_seconds="5.5")

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


def _lora_file(tmp_path: Path, name: str, size: int) -> Path:
    path = tmp_path / f"{name}.safetensors"
    path.write_bytes(b"\0" * size)
    return path


class TestExtendMemoryFit:
    def test_the_selected_loras_are_sized_for_h3_and_ignored_for_ltx(self, tmp_path, monkeypatch):
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(extend_api_module, "memory_budget", lambda: MemoryBudget(24 * 2**30))
        monkeypatch.setattr(extend_api_module, "clip_memory_fit", lambda model, **kwargs: calls.append(kwargs))
        lora = f"{_lora_file(tmp_path, 'style', 4_000)}:0.7"

        extend_api_module._extend_memory_fit(("/models/h3", "h3"), "h3", make_extend_source(), 34, True, None, lora, _web_config(tmp_path))
        extend_api_module._extend_memory_fit(("/models/h3", "h3"), "h3", make_extend_source(), 34, True, None, None, _web_config(tmp_path))
        extend_api_module._extend_memory_fit(("/models/ltx", "ltx"), "ltx-8", make_extend_source(), 48, True, None, lora, _web_config(tmp_path))

        assert [call["loras_bytes"] for call in calls] == [4_000, 0, 0]

    def test_a_lora_that_is_not_there_leaves_the_estimate_unknown_without_failing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(extend_api_module, "memory_budget", lambda: MemoryBudget(24 * 2**30))
        monkeypatch.setattr(extend_api_module, "clip_memory_fit", lambda model, **kwargs: pytest.fail("estimated"))

        fit = extend_api_module._extend_memory_fit(("/models/h3", "h3"), "h3", make_extend_source(), 34, True, None, str(tmp_path / "gone.safetensors"), _web_config(tmp_path))

        assert fit is None

    def test_the_source_request_carries_the_loras_into_the_estimate(self, tmp_path, monkeypatch):
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(extend_api_module, "memory_budget", lambda: MemoryBudget(24 * 2**30))
        monkeypatch.setattr(extend_api_module, "clip_memory_fit", lambda model, **kwargs: calls.append(kwargs))
        monkeypatch.setattr(extend_api_module, "_resolve_video_model", lambda model, _config: ("/models/h3", "h3"))
        self._stub_an_inspected_source(tmp_path, monkeypatch)
        lora = str(_lora_file(tmp_path, "style", 700))

        extend_api_module.extend_source_response({"path": "clip.mp4", "extend_frames": 17, "model": "h3", "lora": lora}, _web_config(tmp_path), tmp_path)
        with pytest.raises(ValueError, match="lora must be a string"):
            extend_api_module.extend_source_response({"path": "clip.mp4", "lora": 5}, _web_config(tmp_path), tmp_path)

        assert [call["loras_bytes"] for call in calls] == [700]

    def test_the_source_request_counts_loras_that_are_still_downloading(self, tmp_path, monkeypatch):
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(extend_api_module, "memory_budget", lambda: MemoryBudget(24 * 2**30))
        monkeypatch.setattr(extend_api_module, "clip_memory_fit", lambda model, **kwargs: calls.append(kwargs))
        monkeypatch.setattr(extend_api_module, "_resolve_video_model", lambda model, _config: ("/models/h3", "h3"))
        self._stub_an_inspected_source(tmp_path, monkeypatch)
        monkeypatch.setattr("zvisiongenerator.utils.paths.get_ziv_data_dir", lambda: tmp_path)

        extend_api_module.extend_source_response(
            {"path": "clip.mp4", "extend_frames": 17, "model": "h3", "lora": "pending_style"}, _web_config(tmp_path), tmp_path, downloading_loras={"pending_style": 321}
        )

        assert [call["loras_bytes"] for call in calls] == [321]

    def test_the_fit_is_asked_for_the_models_family_in_extend_mode_at_its_window_and_default_steps(self, tmp_path, monkeypatch):
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(extend_api_module, "memory_budget", lambda: MemoryBudget(24 * 2**30))
        monkeypatch.setattr(extend_api_module, "clip_memory_fit", lambda model, **kwargs: calls.append({"model": model, **kwargs}))

        extend_api_module._extend_memory_fit(("/models/h3", "h3"), "h3", make_extend_source(), 34, True, None, None, _web_config(tmp_path))
        extend_api_module._extend_memory_fit(("/models/ltx", "ltx"), "ltx-8", make_extend_source(), 48, True, None, None, _web_config(tmp_path))

        assert [(call["model"], call["family"], call["mode"], call["frames"], call["forwards"]) for call in calls] == [
            ("/models/h3", "h3", "extend", 22 + 34, 8),
            ("/models/ltx", "ltx", "extend", 25 + 48, 8),
        ]

    def test_the_h3_estimate_runs_the_chosen_steps_as_the_step_rules_settle_them(self, tmp_path, monkeypatch):
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(extend_api_module, "memory_budget", lambda: MemoryBudget(24 * 2**30))
        monkeypatch.setattr(extend_api_module, "clip_memory_fit", lambda model, **kwargs: calls.append(kwargs))

        extend_api_module._extend_memory_fit(("/models/h3", "h3"), "h3", make_extend_source(), 34, True, 5, None, _web_config(tmp_path))

        assert calls[0]["forwards"] == 4

    def test_a_discrete_gpu_gets_an_estimated_time_for_the_window(self, tmp_path, monkeypatch):
        monkeypatch.setattr(extend_api_module, "memory_budget", lambda: MemoryBudget(10 * 2**30, system_bytes=30 * 2**30))

        fit = extend_api_module._extend_memory_fit(("/models/h3", "h3"), "h3", make_extend_source(width=768, height=448), 34, True, None, None, _web_config(tmp_path))

        assert fit is not None
        assert (fit["kind"], fit["family"]) == ("discrete", "h3")
        assert fit["expected_seconds"] > 0

    def test_a_family_without_an_estimate_has_no_fit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(extend_api_module, "memory_budget", lambda: MemoryBudget(24 * 2**30))

        fit = extend_api_module._extend_memory_fit(("/models/h3", "h3"), "h3", make_extend_source(width=100, height=60), 34, True, None, None, _web_config(tmp_path))

        assert fit is None

    def test_the_source_request_carries_the_steps_into_the_estimate(self, tmp_path, monkeypatch):
        calls: list[dict[str, Any]] = []
        self._stub_an_inspected_source(tmp_path, monkeypatch)
        monkeypatch.setattr(extend_api_module, "memory_budget", lambda: MemoryBudget(24 * 2**30))
        monkeypatch.setattr(extend_api_module, "clip_memory_fit", lambda model, **kwargs: calls.append(kwargs))

        extend_api_module.extend_source_response({"path": "clip.mp4", "extend_frames": 17, "model": "h3", "steps": 4}, _web_config(tmp_path), tmp_path)
        extend_api_module.extend_source_response({"path": "clip.mp4", "extend_frames": 17, "model": "h3"}, _web_config(tmp_path), tmp_path)

        assert [call["forwards"] for call in calls] == [4, 8]

    def test_on_a_mac_the_source_response_reports_the_window_the_mac_renders(self, tmp_path, monkeypatch):
        monkeypatch.setattr(backends_module.sys, "platform", "darwin")
        monkeypatch.setattr(extend_api_module, "memory_budget", lambda: None)
        monkeypatch.setattr(extend_api_module, "_inspect_source", lambda _path: SimpleNamespace(probe=None, recorded=None, problems=(), source=None))
        self._stub_a_source(tmp_path, monkeypatch)

        payload = extend_api_module.extend_source_response({"path": "clip.mp4", "extend_frames": 51, "model": "h3"}, _web_config(tmp_path), tmp_path)

        assert (payload["extend_frames"], payload["window_frames"]) == (51, 56)

    def test_on_a_mac_the_estimate_and_the_rewrite_context_use_the_same_window(self, tmp_path, monkeypatch):
        monkeypatch.setattr(backends_module.sys, "platform", "darwin")
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(extend_api_module, "memory_budget", lambda: MemoryBudget(24 * 2**30))
        monkeypatch.setattr(extend_api_module, "clip_memory_fit", lambda model, **kwargs: calls.append(kwargs))
        monkeypatch.setattr(extend_api_module, "load_config", load_config)
        monkeypatch.setattr(extend_api_module, "load_source_video", lambda *_a, **_k: make_extend_source(fps=24.0))
        monkeypatch.setattr(extend_api_module, "supports_video_extend", lambda: True)
        monkeypatch.setattr(extend_api_module, "supports_video_audio", lambda: False)
        self._stub_a_source(tmp_path, monkeypatch)

        extend_api_module._extend_memory_fit(("/models/h3", "h3"), "h3", make_extend_source(), 51, True, None, None, _web_config(tmp_path))
        context = extend_api_module.extend_context_from_body({"source_video": "clip.mp4", "extend_frames": 51, "model": "h3"}, "She waves.")

        assert calls[0]["frames"] == 56
        assert context.seconds == pytest.approx(56 / 24)

    @pytest.mark.parametrize("steps", [0, "8"])
    def test_steps_must_be_a_positive_integer(self, tmp_path, monkeypatch, steps):
        self._stub_a_source(tmp_path, monkeypatch)

        with pytest.raises(ValueError, match="steps must be a positive integer"):
            extend_api_module.extend_source_response({"path": "clip.mp4", "model": "h3", "steps": steps}, _web_config(tmp_path), tmp_path)

    def _stub_a_source(self, tmp_path, monkeypatch):
        monkeypatch.setattr(extend_api_module, "_source_video_path", lambda *_a, **_k: tmp_path / "clip.mp4")
        monkeypatch.setattr(extend_api_module, "output_media_url", lambda *_a, **_k: None)
        monkeypatch.setattr(extend_api_module, "detect_video_model", lambda path: SimpleNamespace(family="h3" if "h3" in path or "H3" in path else "ltx"))

    def _stub_an_inspected_source(self, tmp_path, monkeypatch):
        self._stub_a_source(tmp_path, monkeypatch)
        probe = SimpleNamespace(fps=24.0, width=768, height=448, rotation=0, frame_count=48, has_audio=False, duration=2.0)
        monkeypatch.setattr(extend_api_module, "_inspect_source", lambda _path: SimpleNamespace(probe=probe, recorded=None, problems=(), source=make_extend_source()))
        monkeypatch.setattr(extend_api_module, "with_family_problems", lambda inspection, family: inspection)

    def test_the_source_response_reports_the_window_of_the_models_family(self, tmp_path, monkeypatch):
        monkeypatch.setattr(extend_api_module, "memory_budget", lambda: None)
        monkeypatch.setattr(extend_api_module, "_inspect_source", lambda _path: SimpleNamespace(probe=None, recorded=None, problems=(), source=None))
        self._stub_a_source(tmp_path, monkeypatch)

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

    def test_the_steps_are_settled_by_the_families_options(self, tmp_path, asked):
        self._fit(tmp_path, steps=5)

        assert asked[0]["forwards"] == 4

    @pytest.mark.parametrize("body", [{"image": "yes"}, {"steps": 0}, {"steps": "4"}])
    def test_a_bad_hint_is_refused(self, tmp_path, asked, body):
        with pytest.raises(ValueError):
            self._fit(tmp_path, **body)

    def test_an_ltx_model_ignores_the_hints_for_its_estimate(self, tmp_path, asked):
        self._fit(tmp_path, model="ltx-8", steps=4)

        assert asked[0]["family"] == "ltx"

    def test_the_selected_loras_are_sized_as_a_job_resolves_them(self, tmp_path, asked):
        small, large = _lora_file(tmp_path, "small", 1_000), _lora_file(tmp_path, "large", 2_500)

        self._fit(tmp_path, lora=f"{small}:0.8,{large}")
        self._fit(tmp_path)

        assert [call["loras_bytes"] for call in asked] == [3_500, 0]

    def test_a_lora_that_is_still_downloading_is_sized_by_what_the_download_expects(self, tmp_path, asked, monkeypatch):
        monkeypatch.setattr("zvisiongenerator.utils.paths.get_ziv_data_dir", lambda: tmp_path)
        monkeypatch.setattr(web_server.lora_installer, "pending_sizes", lambda: {"pending_style": 255_000_000})

        self._fit(tmp_path, lora="pending_style:0.8")

        assert asked[0]["loras_bytes"] == 255_000_000

    def test_review_counts_only_with_an_upscale(self, tmp_path, asked):
        self._fit(tmp_path, upscale=2, review=True)
        self._fit(tmp_path, upscale=2, review=False)
        self._fit(tmp_path, review=True)
        self._fit(tmp_path)

        assert [call["review"] for call in asked] == [True, False, False, False]

    def test_an_ltx_model_does_not_size_its_loras_or_need_them_to_exist(self, tmp_path, asked):
        self._fit(tmp_path, model="ltx-8", lora=f"{tmp_path / 'gone.safetensors'}:1", upscale=2, review=True)

        assert asked[0]["loras_bytes"] == 0

    def test_an_h3_clip_with_a_missing_lora_is_refused_as_the_job_would_be(self, tmp_path, asked):
        with pytest.raises(ValueError, match="LoRA file not found"):
            self._fit(tmp_path, lora=str(tmp_path / "gone.safetensors"))

    @pytest.mark.parametrize("body", [{"lora": 3}, {"review": "yes"}])
    def test_a_bad_lora_or_review_is_refused(self, tmp_path, asked, body):
        with pytest.raises(ValueError):
            self._fit(tmp_path, **body)


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

    @pytest.mark.parametrize(("model", "extend_frames", "window"), [("h3", 17, 39), ("ltx-8", 16, 41)])
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


class TestLoraFitWhileDownloading:
    @pytest.fixture
    def data_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("zvisiongenerator.utils.paths.get_ziv_data_dir", lambda: tmp_path)
        (tmp_path / "loras").mkdir()
        return tmp_path

    def test_a_lora_still_downloading_counts_the_size_it_will_have(self, data_dir):
        assert lora_bytes_for_fit("pending_style:0.8", family="h3", downloading={"pending_style": 255_000_000}) == 255_000_000

    def test_installed_and_downloading_loras_add_up(self, data_dir):
        _lora_file(data_dir / "loras", "ready", 4_000)

        assert lora_bytes_for_fit("ready:1.0,pending_style:0.5", family="h3", downloading={"pending_style": 700}) == 4_700

    def test_the_file_counts_for_its_real_size_once_it_exists(self, data_dir):
        _lora_file(data_dir / "loras", "pending_style", 900)

        assert lora_bytes_for_fit("pending_style", family="h3", downloading={"pending_style": 700}) == 900
