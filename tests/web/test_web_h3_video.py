"""Tests for MiniMax H3 video requests through the Web UI server: the last-frame image, step snapping, the extension step and the memory-fit family."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image
from starlette.datastructures import FormData

from tests._factories import load_packaged_config
from tests._factories import make_extend_source, metadata_copy, use_model_metadata

import zvisiongenerator.backends as backends_module
import zvisiongenerator.utils.audio_track as audio_track_module
from zvisiongenerator.web import config as web_config_module
from zvisiongenerator.web import extend_api as extend_api_module
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web.config import WebUiDefaultModels
from zvisiongenerator.web.lora_fit import lora_size_for_fit


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


class TestReferences:
    """References on a MiniMax H3 form: the repeated ``reference`` field, the cap, and what they combine with."""

    @staticmethod
    def _post(tmp_path: Path, *references: str, config: SimpleNamespace | None = None, **form: str) -> dict[str, Any]:
        entries = [("model", "h3"), ("prompt", "<Picture 1> walks in"), *form.items(), *(("reference", path) for path in references)]
        return web_server._submit_video_job(FormData(entries), config or _web_config(tmp_path))

    def test_the_paths_reach_the_args_the_request_and_the_context(self, tmp_path, submitted):
        first, second = _image(tmp_path, "a.png"), _image(tmp_path, "b.png")

        response = self._post(tmp_path, first, second)

        (job,) = submitted
        expected = [str(Path(first).resolve()), str(Path(second).resolve())]
        assert job["args"].reference_paths == expected
        assert job["request"].reference_paths == tuple(expected)
        assert response["context"]["settings"]["reference_path"] == expected
        assert "reference" not in response["context"]["settings"]
        assert response["context"]["workflow"] == "txt2vid"

    def test_no_reference_leaves_the_request_unchanged(self, tmp_path, submitted):
        response = self._post(tmp_path)

        (job,) = submitted
        assert (job["args"].reference_paths, job["request"].reference_paths) == ([], ())
        assert "reference_path" not in response["context"]["settings"]

    def test_references_leave_the_workflow_to_the_first_frame(self, tmp_path, submitted):
        response = self._post(tmp_path, _image(tmp_path, "a.png"), image_path=_image(tmp_path, "first.png"))

        assert response["context"]["workflow"] == "img2vid"
        assert submitted[0]["request"].image_path is not None

    def test_references_go_with_a_first_and_a_last_frame(self, tmp_path, submitted):
        ref, first, last = _image(tmp_path, "a.png"), _image(tmp_path, "first.png"), _image(tmp_path, "last.png")

        self._post(tmp_path, ref, image_path=first, end_image_path=last)

        (job,) = submitted
        assert (job["request"].reference_paths, job["request"].end_image_path is not None) == ((str(Path(ref).resolve()),), True)

    def test_an_extension_keeps_its_references(self, tmp_path, submitted, monkeypatch):
        monkeypatch.setattr(web_server, "load_source_video", lambda *_a, **_kw: make_extend_source(fps=24.0, frame_count=97))
        monkeypatch.setattr(web_server, "supports_video_extend", lambda: True)
        ref = _image(tmp_path, "a.png")

        response = self._post(tmp_path, ref, workflow="extend", source_video="clip.mp4")

        (job,) = submitted
        assert response["context"]["workflow"] == "extend"
        assert job["request"].reference_paths == (str(Path(ref).resolve()),)

    def test_an_upload_is_saved_and_recorded_in_row_order(self, tmp_path, submitted):
        import io

        from starlette.datastructures import UploadFile

        first = _image(tmp_path, "a.png")
        upload = UploadFile(file=io.BytesIO(Path(_image(tmp_path, "up.png")).read_bytes()), filename="b.png")
        entries = [("model", "h3"), ("prompt", "x"), ("reference", upload), ("reference", first)]

        response = web_server._submit_video_job(FormData(entries), _web_config(tmp_path))

        saved, typed = submitted[0]["request"].reference_paths
        assert Path(saved).parent == tmp_path / ".web_uploads"
        assert typed == str(Path(first).resolve())
        assert response["context"]["settings"]["reference_path"] == [saved, typed]

    def test_three_references_are_refused(self, tmp_path, submitted):
        refs = [_image(tmp_path, f"r{index}.png") for index in range(3)]

        with pytest.raises(ValueError):
            self._post(tmp_path, *refs)

        assert submitted == []

    def test_a_model_without_references_refuses_them(self, tmp_path, submitted):
        with pytest.raises(ValueError):
            web_server._submit_video_job(FormData([("model", "ltx-8"), ("prompt", "x"), ("reference", _image(tmp_path, "a.png"))]), _web_config(tmp_path))

        assert submitted == []

    @pytest.mark.parametrize("bad", ["gone.png", "notes.txt"])
    def test_a_reference_that_is_missing_or_not_an_image_is_refused(self, tmp_path, submitted, bad):
        (tmp_path / "notes.txt").write_text("x")

        with pytest.raises(ValueError):
            self._post(tmp_path, str(tmp_path / bad))

        assert submitted == []

    def test_an_upscale_is_accepted_with_references(self, tmp_path, submitted):
        self._post(tmp_path, _image(tmp_path, "a.png"), upscale="2")

        assert submitted[0]["request"].upscale == 2
        assert len(submitted[0]["request"].reference_paths) == 1

    def test_an_audio_track_is_accepted_with_references(self, tmp_path, submitted, monkeypatch):
        track = tmp_path / "voice.wav"
        track.write_bytes(b"x")
        monkeypatch.setattr(audio_track_module, "probe_track_seconds", lambda _path: 30.0)
        monkeypatch.setattr(web_server, "require_readable_track", lambda _track: None)

        self._post(tmp_path, _image(tmp_path, "a.png"), audio_track_path=str(track))

        assert submitted[0]["request"].audio_track is not None
        assert len(submitted[0]["request"].reference_paths) == 1


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


class TestExtendSourceWindow:
    """The source response reports the window of the model's family: the frames the model generates for the length asked."""

    def test_on_a_mac_the_source_response_reports_the_window_the_mac_renders(self, tmp_path, monkeypatch):
        monkeypatch.setattr(backends_module.sys, "platform", "darwin")
        monkeypatch.setattr(extend_api_module, "_inspect_source", lambda _path: SimpleNamespace(probe=None, recorded=None, problems=(), source=None))
        self._stub_a_source(tmp_path, monkeypatch)

        payload = extend_api_module.extend_source_response({"path": "clip.mp4", "extend_frames": 51, "model": "h3"}, _web_config(tmp_path))

        assert (payload["extend_frames"], payload["window_frames"]) == (51, 56)

    def _stub_a_source(self, tmp_path, monkeypatch):
        monkeypatch.setattr(extend_api_module, "source_video_path", lambda *_a, **_k: tmp_path / "clip.mp4")
        monkeypatch.setattr(extend_api_module, "detect_video_model", lambda path: SimpleNamespace(family="h3" if "h3" in path or "H3" in path else "ltx"))

    def test_the_source_response_reports_the_window_of_the_models_family(self, tmp_path, monkeypatch):
        monkeypatch.setattr(extend_api_module, "_inspect_source", lambda _path: SimpleNamespace(probe=None, recorded=None, problems=(), source=None))
        self._stub_a_source(tmp_path, monkeypatch)

        payload = extend_api_module.extend_source_response({"path": "clip.mp4", "extend_frames": 17, "model": "h3"}, _web_config(tmp_path))

        assert (payload["extend_frames"], payload["window_frames"]) == (17, 39)


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
        assert lora_size_for_fit("pending_style:0.8", {"pending_style": 255_000_000}).total == 255_000_000

    def test_installed_and_downloading_loras_add_up(self, data_dir):
        _lora_file(data_dir / "loras", "ready", 4_000)

        assert lora_size_for_fit("ready:1.0,pending_style:0.5", {"pending_style": 700}).total == 4_700

    def test_the_file_counts_for_its_real_size_once_it_exists(self, data_dir):
        _lora_file(data_dir / "loras", "pending_style", 900)

        assert lora_size_for_fit("pending_style", {"pending_style": 700}).total == 900


class TestAudioTrack:
    """The form's ``audio_track_path`` and ``audio_track_start`` become the job's track, refusals included."""

    @pytest.fixture
    def track(self, tmp_path, monkeypatch) -> str:
        import zvisiongenerator.utils.audio_track as audio_track_module

        path = tmp_path / "voice.mp3"
        path.write_bytes(b"x")
        monkeypatch.setattr(audio_track_module, "probe_track_seconds", lambda _path: 3.6)
        return str(path)

    def test_the_track_reaches_the_args_the_request_and_the_job_settings(self, tmp_path, submitted, track):
        response = _submit(tmp_path, model="ltx-8", audio_track_path=track, audio_track_start="1.6", audio="off")

        args, request = submitted[0]["args"], submitted[0]["request"]
        assert (args.audio_track.path, args.audio_track.start) == (track, 1.6)
        assert request.audio_track == args.audio_track
        assert response["context"]["settings"]["audio_track_path"] == track
        assert response["context"]["settings"]["audio_track_start"] == "1.6"

    def test_the_track_is_the_sound_whatever_the_audio_switch_says(self, tmp_path, submitted, track):
        _submit(tmp_path, model="ltx-8", audio_track_path=track, audio="off")

        assert (submitted[0]["args"].no_audio, submitted[0]["request"].no_audio) == (False, False)

    def test_a_track_that_runs_out_first_keeps_the_requested_frames(self, tmp_path, submitted, track):
        _submit(tmp_path, model="ltx-8", frames="121", audio_track_path=track)

        assert (submitted[0]["args"].num_frames, submitted[0]["request"].num_frames) == (121, 121)

    @pytest.mark.parametrize("start", ["-1", "nan", "soon"])
    def test_a_bad_start_is_refused(self, tmp_path, submitted, track, start):
        with pytest.raises(ValueError, match="audio_track_start"):
            _submit(tmp_path, model="ltx-8", audio_track_path=track, audio_track_start=start)

        assert submitted == []

    def test_a_url_is_refused(self, tmp_path, submitted):
        with pytest.raises(ValueError, match="host-local"):
            _submit(tmp_path, model="ltx-8", audio_track_path="https://example.com/voice.mp3")

    def test_h3_refuses_an_upscale_with_a_track(self, tmp_path, submitted, track):
        with pytest.raises(ValueError, match="can't upscale a clip made from your track"):
            _submit(tmp_path, model="h3", upscale="2", audio_track_path=track)

        assert submitted == []

    def test_a_family_without_the_capability_refuses_a_track(self, tmp_path, submitted, track, monkeypatch):
        config = _web_config(tmp_path)
        data = metadata_copy()
        data["video_families"]["ltx"]["capabilities"]["audio_track"] = False
        use_model_metadata(monkeypatch, tmp_path, data)

        with pytest.raises(ValueError, match="can't follow an audio track"):
            _submit(tmp_path, config, model="ltx-8", audio_track_path=track)

        assert submitted == []

    def test_a_track_that_cannot_be_read_fails_before_the_job_is_submitted(self, tmp_path, submitted, monkeypatch):
        import zvisiongenerator.utils.audio_track as audio_track_module

        notes = tmp_path / "notes.mp3"
        notes.write_text("hello")
        monkeypatch.setattr(audio_track_module, "probe_track_seconds", MagicMock(side_effect=ValueError(audio_track_module.TRACK_UNREADABLE)))

        with pytest.raises(ValueError, match="Can't read the audio track"):
            _submit(tmp_path, model="ltx-8", audio_track_path=str(notes))

        assert submitted == []
