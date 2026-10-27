"""Tests for the extend-video source inspection, submission and memory-fit routes."""

from __future__ import annotations

import functools
import json
import os
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest


from tests._factories import make_probe
from tests._paths import FIXTURES
from tests._server_helpers import post_video, stub_video_submission, make_web_config

from zvisiongenerator.web import extend_api as extend_api_module
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import video_clip as video_clip_module
from zvisiongenerator.utils import video_extend as video_extend_module
from zvisiongenerator.utils.config import load_config
from zvisiongenerator.utils.model_memory import MemoryBudget, clip_memory_fit as real_clip_memory_fit


_EXTEND_SCENE = "A woman in a red raincoat on a rainy street at night."


_EXTEND_PROMPT = "She opens a black umbrella and walks away."


# A clean 512x256, 24 fps, 97-frame clip with audio and BT.709 colour tags.
_clip_probe = functools.partial(make_probe, frame_count=97, duration=4.04, color_space="bt709", color_primaries="bt709", color_trc="bt709")


@pytest.fixture
def clip(monkeypatch, tmp_path):
    """Create a source clip file whose probe and embedded settings come from the returned, mutable ``spec``."""
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00" * 16)
    spec = SimpleNamespace(path=path, probe={}, config={"scene": _EXTEND_SCENE, "prompt": _EXTEND_PROMPT, "model": "ltx-8"}, audio_ok=True)
    extend_api_module._inspect_by_identity.cache_clear()
    spec.probed = []
    monkeypatch.setattr(video_extend_module, "probe_video", lambda path: spec.probed.append(path) or _clip_probe(**spec.probe))
    monkeypatch.setattr(video_extend_module, "read_mp4_config", lambda _path: spec.config)
    monkeypatch.setattr(video_extend_module, "audio_decodes", lambda *_args, **_kwargs: spec.audio_ok)
    for module in (web_server, extend_api_module):
        monkeypatch.setattr(module, "supports_video_extend", lambda: True)
        monkeypatch.setattr(module, "supports_video_audio", lambda: True)
    monkeypatch.setattr(web_server, "load_web_config", make_web_config)
    monkeypatch.setattr(web_server, "_media_output_root", lambda: tmp_path)
    monkeypatch.setattr(extend_api_module, "memory_budget", lambda: None)
    for module in (web_server, extend_api_module):
        monkeypatch.setattr(module, "require_ffmpeg", lambda: None)
    return spec


def _inspect(body):
    with TestClient(web_server.app) as client:
        return client.post("/api/extend/source", json=body)


def test_a_clip_that_has_not_changed_is_probed_once(clip):
    first = extend_api_module._inspect_source(clip.path)
    second = extend_api_module._inspect_source(clip.path)

    assert second is first
    assert len(clip.probed) == 1


@pytest.mark.parametrize("change", ["modified", "resized"])
def test_a_clip_that_was_edited_is_probed_again(clip, change):
    extend_api_module._inspect_source(clip.path)
    stat = clip.path.stat()
    if change == "modified":
        os.utime(clip.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    else:
        clip.path.write_bytes(b"\x00" * 99)
        os.utime(clip.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # same time, only the size differs

    extend_api_module._inspect_source(clip.path)

    assert len(clip.probed) == 2


def test_a_missing_clip_is_not_probed(clip):
    with pytest.raises(FileNotFoundError):
        extend_api_module._inspect_source(clip.path.with_name("gone.mp4"))

    assert clip.probed == []


def test_a_source_is_probed_once_across_the_inspect_route_and_the_job_checks(clip):
    body = {"path": str(clip.path)}

    assert _inspect(body).status_code == 200
    first = extend_api_module.load_source_video(str(clip.path), key="source_video")
    second = extend_api_module.load_source_video(str(clip.path), key="context.source_video")

    assert second is first
    assert len(clip.probed) == 1


def test_a_source_that_changed_since_it_was_inspected_is_checked_again(clip):
    assert _inspect({"path": str(clip.path)}).status_code == 200
    clip.path.write_bytes(b"\x00" * 32)
    clip.probe = {"frame_count": 5}

    with pytest.raises(ValueError, match="cannot be extended"):
        extend_api_module.load_source_video(str(clip.path), key="source_video")

    assert len(clip.probed) == 2


def test_inspect_route_reports_the_facts_recorded_prompts_and_defaults_of_a_good_source(clip):
    """The response is the shape the frontend types are checked against: ``frontend/src/lib/state/extend.test.ts`` loads this same file as an ``ExtendSourceInfo``; change both together."""
    fixture = json.loads((FIXTURES / "extend_source_response.json").read_text(encoding="utf-8"))

    payload = _inspect({"path": str(clip.path)}).json()

    assert list(payload) == list(fixture)
    # Only where the temporary clip lives differs from the file.
    assert payload == {**fixture, "path": str(clip.path.resolve())}


def test_inspect_route_gives_no_media_url_outside_the_output_root(clip, monkeypatch, tmp_path):
    other_root = tmp_path / "elsewhere"
    other_root.mkdir()
    monkeypatch.setattr(web_server, "_media_output_root", lambda: other_root)

    assert _inspect({"path": str(clip.path)}).json()["media_url"] is None


def test_inspect_route_shows_the_displayed_size_of_a_rotated_source(clip):
    clip.probe = {"coded_width": 512, "coded_height": 256, "width": 256, "height": 512, "rotation": 90}

    payload = _inspect({"path": str(clip.path)}).json()

    assert (payload["width"], payload["height"], payload["rotation"]) == (256, 512, 90)
    assert payload["problems"] == []


@pytest.mark.parametrize(
    ("probe", "config", "audio_ok", "code"),
    [
        ({"width": 500, "coded_width": 500}, None, True, "off_grid"),
        ({}, {"generation": {"upscale": {"factor": 2}}}, True, "upscaled"),
        ({}, None, False, "unreadable_audio"),
    ],
)
def test_inspect_route_surfaces_a_problem_code_with_a_message(clip, probe, config, audio_ok, code):
    clip.probe = probe
    clip.audio_ok = audio_ok
    if config is not None:
        clip.config = {**clip.config, **config}

    payload = _inspect({"path": str(clip.path)}).json()

    assert [problem["code"] for problem in payload["problems"]] == [code]
    assert all(set(problem) == {"code", "message"} and problem["message"] for problem in payload["problems"])
    assert payload["width"] is not None
    assert payload["memory_fit"] is None


def test_inspect_route_flags_a_30_fps_source_for_h3_only(clip, monkeypatch):
    from zvisiongenerator.utils.config import load_config

    clip.probe = {"r_frame_rate": "30/1", "avg_frame_rate": "30/1"}
    config = make_web_config()
    config.app_config = load_config()
    monkeypatch.setattr(web_server, "load_web_config", lambda: config)
    monkeypatch.setattr(extend_api_module, "resolve_model_path", lambda model, **_: "MiniMaxAI/MiniMax-H3" if model == "h3" else "Lightricks/ltx")

    h3 = _inspect({"path": str(clip.path), "model": "h3"}).json()
    ltx = _inspect({"path": str(clip.path), "model": "ltx-8"}).json()

    assert [problem["code"] for problem in h3["problems"]] == ["frame_rate"]
    assert "24 fps" in h3["problems"][0]["message"]
    assert h3["memory_fit"] is None
    assert ltx["problems"] == []


def test_inspect_route_passes_a_24_fps_source_for_h3(clip, monkeypatch):
    from zvisiongenerator.utils.config import load_config

    config = make_web_config()
    config.app_config = load_config()
    monkeypatch.setattr(web_server, "load_web_config", lambda: config)
    monkeypatch.setattr(extend_api_module, "resolve_model_path", lambda model, **_: "MiniMaxAI/MiniMax-H3")

    assert _inspect({"path": str(clip.path), "model": "h3"}).json()["problems"] == []


def test_inspect_route_reports_a_file_that_is_not_a_video_without_facts(clip, monkeypatch):
    def _no_video(_path):
        raise ValueError(video_extend_module.NO_VIDEO_STREAM)

    monkeypatch.setattr(video_extend_module, "probe_video", _no_video)

    payload = _inspect({"path": str(clip.path)}).json()

    assert [problem["code"] for problem in payload["problems"]] == ["no_video_stream"]
    assert all(payload[key] is None for key in ("width", "height", "rotation", "fps", "frame_count", "duration_seconds", "has_audio", "extend_frames", "window_frames"))
    assert payload["memory_fit"] is None


def test_inspect_route_flags_a_single_prompt_source_as_legacy(clip):
    clip.config = {"prompt": 'A woman stands in the rain and says "Hello".', "model": "ltx-8"}

    payload = _inspect({"path": str(clip.path)}).json()

    assert (payload["scene"], payload["prompt"], payload["legacy"]) == (None, 'A woman stands in the rain and says "Hello".', True)


def test_inspect_route_handles_a_source_without_embedded_settings(clip):
    clip.config = None

    payload = _inspect({"path": str(clip.path)}).json()

    assert (payload["scene"], payload["prompt"], payload["legacy"], payload["model"], payload["lora"]) == (None, None, False, None, None)
    assert payload["problems"] == []
    assert payload["extend_frames"] == 48


def test_inspect_route_resolves_the_recorded_model_against_the_video_models(clip):
    clip.config = {**clip.config, "model": "some-removed-model", "lora": "/loras/style.safetensors:0.8"}

    payload = _inspect({"path": str(clip.path)}).json()

    assert payload["model"] is None
    assert payload["lora"] == "/loras/style.safetensors:0.8"


def test_inspect_route_reports_no_loras_as_blank_for_a_source_with_settings(clip):
    """Blank, not null: the source recorded its LoRAs and had none, so extending it clears the chosen ones."""
    clip.config = {"prompt": "A walk.", "model": "ltx-8"}

    assert _inspect({"path": str(clip.path)}).json()["lora"] == ""


@pytest.mark.parametrize(("requested", "expected"), [(50, 48), (1, 8)])
def test_inspect_route_snaps_the_length_to_whole_latent_frames(clip, requested, expected):
    payload = _inspect({"path": str(clip.path), "extend_frames": requested}).json()

    assert (payload["extend_frames"], payload["window_frames"]) == (expected, 25 + expected)


def test_inspect_route_gives_h3_a_window_of_its_22_frame_tail(clip, monkeypatch):
    from zvisiongenerator.utils.config import load_config

    config = make_web_config()
    config.app_config = load_config()
    monkeypatch.setattr(web_server, "load_web_config", lambda: config)
    monkeypatch.setattr(extend_api_module, "resolve_model_path", lambda model, **_: "MiniMaxAI/MiniMax-H3")
    monkeypatch.setattr(web_server.sys, "platform", "linux")

    payload = _inspect({"path": str(clip.path), "extend_frames": 34, "model": "h3"}).json()

    assert (payload["extend_frames"], payload["window_frames"]) == (34, 22 + 34)


def test_inspect_route_defaults_to_two_seconds_at_the_source_frame_rate(clip):
    clip.probe = {"r_frame_rate": "30000/1001", "avg_frame_rate": "30000/1001"}

    assert _inspect({"path": str(clip.path)}).json()["extend_frames"] == 56


def test_inspect_route_adds_the_memory_fit_of_the_window(clip, monkeypatch):
    calls: list[dict[str, object]] = []

    def _fit(model, **kwargs):
        calls.append({"model": model, **kwargs})
        return {"status": "tight", "required_gb": 21.0, "budget_gb": 24.0}

    monkeypatch.setattr(extend_api_module, "memory_budget", lambda: MemoryBudget(24 * 2**30))
    monkeypatch.setattr(extend_api_module, "resolve_model_path", lambda model, **_: f"/models/{model}")
    monkeypatch.setattr(extend_api_module, "clip_memory_fit", _fit)

    payload = _inspect({"path": str(clip.path), "extend_frames": 48, "low_memory": False}).json()

    assert payload["memory_fit"] == {"status": "tight", "required_gb": 21.0, "budget_gb": 24.0}
    assert len(calls) == 1
    assert {key: value for key, value in calls[0].items() if key != "app_config"} == {
        "model": "/models/ltx-8",
        "width": 512,
        "height": 256,
        "frames": 73,
        "low_memory": False,
        "budget": MemoryBudget(24 * 2**30),
        "model_name": "ltx-8",
        "family": "ltx",
        "mode": "extend",
        "forwards": 8,
        "loras_bytes": 0,
    }


def test_inspect_route_uses_the_requested_model_and_defaults_low_memory_on(clip, monkeypatch):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(extend_api_module, "memory_budget", lambda: MemoryBudget(24 * 2**30))
    monkeypatch.setattr(extend_api_module, "resolve_model_path", lambda model, **_: f"/models/{model}")
    monkeypatch.setattr(extend_api_module, "clip_memory_fit", lambda model, **kwargs: calls.append({"model": model, **kwargs}))

    payload = _inspect({"path": str(clip.path), "model": "other"}).json()

    assert payload["memory_fit"] is None
    assert (calls[0]["model"], calls[0]["low_memory"]) == ("/models/other", True)


def test_inspect_route_skips_the_memory_fit_for_a_source_with_problems(clip, monkeypatch):
    clip.probe = {"field_order": "tt"}
    monkeypatch.setattr(extend_api_module, "memory_budget", lambda: MemoryBudget(24 * 2**30))
    monkeypatch.setattr(extend_api_module, "clip_memory_fit", lambda *_args, **_kwargs: pytest.fail("memory fit asked for a source with problems"))

    assert _inspect({"path": str(clip.path)}).json()["memory_fit"] is None


def test_inspect_route_returns_404_for_a_missing_file(clip, tmp_path):
    response = _inspect({"path": str(tmp_path / "gone.mp4")})

    assert response.status_code == 404
    assert set(response.json()) == {"detail"}


def _no_ffmpeg() -> None:
    raise RuntimeError("ffmpeg is missing")


def test_inspect_route_without_ffmpeg_is_a_422_not_a_missing_file(clip, monkeypatch):
    monkeypatch.setattr(extend_api_module, "require_ffmpeg", _no_ffmpeg)

    response = _inspect({"path": str(clip.path)})

    assert response.status_code == 422
    assert clip.probed == []


def test_extend_submission_without_ffmpeg_is_refused_before_the_source_is_read(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)
    monkeypatch.setattr(extend_api_module, "require_ffmpeg", _no_ffmpeg)

    response = _post_extend({"source_video": str(clip.path), "prompt": "She walks away."})

    assert response.status_code == 422
    assert clip.probed == []
    assert submitted == []


@pytest.mark.parametrize(
    "body",
    [
        [1, 2],
        {},
        {"path": "https://example.com/clip.mp4"},
        {"path": "clip.txt"},
        {"path": "clip.mp4", "extend_frames": 0},
        {"path": "clip.mp4", "model": 5},
        b"not json",
    ],
)
def test_inspect_route_rejects_invalid_bodies(clip, body):
    if isinstance(body, dict) and body.get("path") == "clip.mp4":
        body = {**body, "path": str(clip.path)}

    with TestClient(web_server.app) as client:
        if isinstance(body, bytes):
            response = client.post("/api/extend/source", content=body, headers={"content-type": "application/json"})
        else:
            response = client.post("/api/extend/source", json=body)

    assert response.status_code == 422
    assert set(response.json()) == {"detail"}


def test_inspect_route_accepts_every_source_container(clip, tmp_path):
    for suffix in (".mov", ".m4v", ".mkv", ".webm", ".MP4"):
        path = tmp_path / f"clip{suffix}"
        path.write_bytes(b"\x00")
        assert _inspect({"path": str(path)}).status_code == 200


def test_workspace_contract_carries_extend_only_with_the_capability(monkeypatch):
    monkeypatch.setattr(web_server, "supports_video_extend", lambda: True)
    with_extend = web_server._build_workflow_contract()
    monkeypatch.setattr(web_server, "supports_video_extend", lambda: False)
    without_extend = web_server._build_workflow_contract()

    assert "extend" in with_extend["values"] and "extend" in with_extend["definitions"]
    assert "extend" not in without_extend["values"] and "extend" not in without_extend["definitions"]


def _post_extend(data: dict[str, object]):
    return post_video({"workflow": "extend", **data})


def _stub_extend_submission(monkeypatch, tmp_path, clip) -> list[dict[str, object]]:
    submitted = stub_video_submission(monkeypatch, tmp_path)
    monkeypatch.setattr(web_server, "resolve_video_clip", lambda *_args, **_kwargs: pytest.fail("an extension must not be aligned"))
    return submitted


def test_extend_submission_sets_the_args_the_workflow_reads(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)

    response = _post_extend({"source_video": str(clip.path), "extend_frames": "48", "prompt": "She walks away.", "audio": "on"})

    assert response.status_code == 200
    (job,) = submitted
    args = job["args"]
    source = args.extend_source
    assert (source.path, source.width, source.height, source.fps, source.frame_count) == (clip.path.resolve(), 512, 256, 24.0, 97)
    assert (args.extend_frames, args.width, args.height, args.fps, args.num_frames) == (48, 512, 256, 24.0, 73)
    assert (args.image_path, args.upscale, args.upscale_steps, args.frames) == (None, None, None, None)
    assert (args.no_audio, args.audio) == (False, True)
    assert (args.scene, args.prompt) == (_EXTEND_SCENE, "She walks away.")
    assert args.steps == 8
    request = job["request"]
    assert (request.width, request.height, request.num_frames, request.image_path, request.upscale) == (512, 256, 73, None, None)
    assert job["prompts_data"] == {"web": [("She walks away.", None)]}
    assert job["context"]["workflow"] == "extend"
    assert job["context"]["job_type"] == "Extend video"
    assert job["context"]["meta"] == "512x256 · +48 frames · 8 steps"
    assert job["context"]["settings"]["source_video"] == str(clip.path)
    assert response.json()["workflow"] == "extend"


def test_extend_submission_turns_audio_off_when_unchecked_or_unsupported(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)

    _post_extend({"source_video": str(clip.path), "audio": "off"})
    monkeypatch.setattr(web_server, "supports_video_audio", lambda: False)
    _post_extend({"source_video": str(clip.path), "audio": "on"})

    assert [job["args"].no_audio for job in submitted] == [True, True]


def test_extend_submission_reads_the_length_as_frames_or_seconds_at_the_source_frame_rate(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)
    base = {"source_video": str(clip.path)}

    _post_extend({**base, "extend_frames": "16"})
    _post_extend({**base, "extend_seconds": "1"})
    _post_extend({**base, "extend_seconds": "2.5", "extend_frames": "8"})
    _post_extend(base)

    assert [job["args"].extend_frames for job in submitted] == [16, 24, 8, 48]
    assert [job["args"].num_frames for job in submitted] == [41, 49, 33, 73]


@pytest.mark.parametrize(
    "data",
    [
        {"extend_frames": "12"},
        {"extend_frames": "0"},
        {"extend_seconds": "0"},
        {"extend_seconds": "-1"},
    ],
)
def test_extend_submission_rejects_an_invalid_length(monkeypatch, tmp_path, clip, data):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)

    response = _post_extend({"source_video": str(clip.path), **data})

    assert response.status_code == 422
    assert "extend_" in response.json()["detail"]
    assert submitted == []


def test_extend_submission_defaults_the_scene_to_the_sources_and_keeps_an_explicit_one(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)
    base = {"source_video": str(clip.path), "prompt": "She walks away."}

    _post_extend(base)
    _post_extend({**base, "scene": "A night street."})
    _post_extend({**base, "scene": "  "})

    assert [job["args"].scene for job in submitted] == [_EXTEND_SCENE, "A night street.", None]


def test_extend_submission_of_a_legacy_source_has_no_default_scene(monkeypatch, tmp_path, clip):
    clip.config = {"prompt": "A woman stands in the rain.", "model": "ltx-8"}
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)

    response = _post_extend({"source_video": str(clip.path), "prompt": "A woman stands in the rain."})

    assert response.status_code == 200
    assert submitted[0]["args"].scene is None
    assert submitted[0]["args"].extend_source.legacy is True


def test_extend_submission_allows_an_empty_continuation(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)

    nothing = _post_extend({"source_video": str(clip.path)})
    blank = _post_extend({"source_video": str(clip.path), "prompt": "  ", "scene": ""})

    assert nothing.status_code == blank.status_code == 200
    assert [job["prompts_data"] for job in submitted] == [{"web": [("", None)]}, {"web": [("", None)]}]
    assert [job["context"]["prompt"] for job in submitted] == [_EXTEND_SCENE, ""]


def test_extend_submission_ignores_size_reference_and_upscale_fields(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)
    ignored = {"ratio": "bogus", "size": "bogus", "width": "640", "height": "352", "frames": "121", "image_path": "/missing.png", "upscale": "2"}

    response = _post_extend({"source_video": str(clip.path), "prompt": "She walks away.", **ignored})

    assert response.status_code == 200
    args = submitted[0]["args"]
    assert (args.ratio, args.size, args.image_path, args.upscale) == (None, None, None, None)
    assert (args.width, args.height, args.num_frames) == (512, 256, 73)


def test_extend_submission_checks_the_source_again(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)
    base = {"source_video": str(clip.path), "prompt": "She walks away."}

    clip.probe = {"field_order": "tt"}
    clip.path.write_bytes(b"\x00" * 32)  # an edited file has another size or time, so the earlier inspection is not reused
    changed = _post_extend(base)
    clip.probe = {}
    clip.path.unlink()
    moved = _post_extend(base)

    assert changed.status_code == 422 and "cannot be extended" in changed.json()["detail"]
    assert moved.status_code == 422 and "not found" in moved.json()["detail"]
    assert submitted == []


@pytest.mark.parametrize("source", [None, "https://example.com/clip.mp4", "clip.png"])
def test_extend_submission_requires_a_video_source_path(monkeypatch, tmp_path, clip, source):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)
    data = {"prompt": "She walks away."} if source is None else {"prompt": "She walks away.", "source_video": source}

    response = _post_extend(data)

    assert response.status_code == 422
    assert "source_video" in response.json()["detail"]
    assert submitted == []


def test_extend_submission_refuses_a_link_to_a_file_that_is_not_a_video(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)
    target = tmp_path / "notes.txt"
    target.write_text("not a video")
    link = tmp_path / "looks-like-a-video.mp4"
    link.symlink_to(target)

    response = _post_extend({"source_video": str(link), "prompt": "She walks away."})

    assert response.status_code == 422
    assert "must be a video file" in response.json()["detail"]
    assert submitted == []


def test_extend_submission_is_refused_without_the_capability(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)
    monkeypatch.setattr(web_server, "supports_video_extend", lambda: False)

    response = _post_extend({"source_video": str(clip.path), "prompt": "She walks away."})

    assert response.status_code == 422
    assert set(response.json()) == {"detail"}
    assert submitted == []


def test_extend_submission_with_a_prompt_file_threads_entry_scenes(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)
    path = tmp_path / "prompts.yaml"
    path.write_text("next:\n  - scene: A night street.\n    prompt: she leaves\n  - prompt: he arrives\n", encoding="utf-8")

    response = _post_extend({"source_video": str(clip.path), "prompt_source": "file", "prompts_file": str(path), "prompt_option_id": ["next:0", "next:1"]})

    assert response.status_code == 200
    (job,) = submitted
    assert job["scene_by_set"] == {"next": ["A night street.", None]}
    assert job["args"].scene == _EXTEND_SCENE
    assert job["prompts_data"] == {"next": [("she leaves", None), ("he arrives", None)]}


def test_extend_submission_with_a_prompt_file_keeps_no_op_entries_for_the_enhancer(monkeypatch, tmp_path, clip):
    submitted = _stub_extend_submission(monkeypatch, tmp_path, clip)
    path = tmp_path / "prompts.yaml"
    path.write_text("next:\n  - prompt: she leaves\n    enhance:\n      style: keep\n", encoding="utf-8")

    response = _post_extend({"source_video": str(clip.path), "prompt_source": "file", "prompts_file": str(path), "prompt_option_id": ["next:0"]})

    assert response.status_code == 200
    assert submitted[0]["enhance_by_set"] is not None
    assert submitted[0]["enhance_by_set"]["next"][0] is not None


def test_inspect_route_resolves_the_path_it_reports(clip, tmp_path):
    (tmp_path / "sub").mkdir()
    payload = _inspect({"path": str(tmp_path / "sub" / ".." / "clip.mp4")}).json()

    assert payload["path"] == str(clip.path.resolve())
    assert payload["media_url"] == "/media/clip.mp4"


_UNALIGNED_CLIP = {"steps": 8, "width": 720, "height": 450, "num_frames": 52}


@pytest.fixture
def clip_fit(monkeypatch, tmp_path):
    """Stub the model and budget behind ``POST /api/video/memory-fit`` and record what the memory check receives."""
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path)
    web_config.app_config["video_model_presets"] = {"ltx": load_config()["video_model_presets"]["ltx"]}
    calls: list[dict[str, object]] = []

    def _fit(model, **kwargs):
        calls.append({"model": model, **kwargs})
        return {"status": "tight", "required_gb": 21.0, "budget_gb": 24.0}

    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(video_clip_module, "resolve_model_path", lambda model, **_: f"/models/{model}")
    monkeypatch.setattr(video_clip_module, "detect_video_model", lambda _model: SimpleNamespace(family="ltx", supports_i2v=True, resolution_alignment=32, frame_alignment=8, default_fps=24))
    monkeypatch.setattr(video_clip_module, "resolve_video_defaults", lambda *_args: dict(_UNALIGNED_CLIP))
    monkeypatch.setattr(web_server, "memory_budget", lambda: MemoryBudget(24 * 2**30, unified_bytes=32 * 2**30))
    monkeypatch.setattr(web_server, "clip_memory_fit", _fit)
    monkeypatch.setattr(web_server.web_runner, "submit_video_request_job", lambda **_kwargs: pytest.fail("the memory check queued a job"))
    return SimpleNamespace(calls=calls, web_config=web_config)


def _memory_fit(body):
    with TestClient(web_server.app) as client:
        return client.post("/api/video/memory-fit", json=body)


def test_memory_fit_route_checks_the_aligned_clip_against_the_budget(clip_fit):
    response = _memory_fit({"model": "ltx-8", "ratio": "16:9", "size": "m", "frames": 52, "low_memory": False, "upscale": None})

    assert response.status_code == 200
    assert response.json() == {"memory_fit": {"status": "tight", "required_gb": 21.0, "budget_gb": 24.0}, "width": 736, "height": 448, "frames": 49}
    assert len(clip_fit.calls) == 1
    call = clip_fit.calls[0]
    assert (call["model"], call["width"], call["height"], call["frames"], call["mode"], call["upscale"], call["low_memory"]) == ("/models/ltx-8", 736, 448, 49, "t2v", False, False)


def test_memory_fit_route_defaults_the_model_and_turns_low_memory_on(clip_fit):
    assert _memory_fit({}).status_code == 200

    assert (clip_fit.calls[0]["model"], clip_fit.calls[0]["low_memory"]) == ("/models/ltx-8", True)


def test_memory_fit_route_checks_an_upscaled_clip_as_upscaled(clip_fit):
    _memory_fit({"upscale": 2})

    assert clip_fit.calls[0]["upscale"] is True


@pytest.mark.filterwarnings("ignore:.*Adjusted")
@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"ratio": "16:9", "size": "m", "frames": 52}, (736, 448, 49)),
        ({"ratio": "16:9", "size": "m", "frames": 52, "upscale": 2}, (704, 448, 49)),
        ({"ratio": "16:9", "size": "m", "width": 640, "height": 384, "frames": 97}, (640, 384, 97)),
    ],
)
def test_memory_fit_route_resolves_the_size_and_length_a_job_gets(clip_fit, monkeypatch, fields, expected):
    """The route and ``POST /api/generate`` align the same request the same way, explicit width and height included."""
    jobs: list[dict[str, object]] = []
    monkeypatch.setattr(web_server, "require_ffmpeg", lambda: None)
    monkeypatch.setattr(web_server.web_runner, "submit_video_request_job", lambda **kwargs: jobs.append(kwargs) or "job-123")
    monkeypatch.setattr(
        video_clip_module, "resolve_video_defaults", lambda _family, _config, overrides: {**_UNALIGNED_CLIP, **{key: overrides[key] for key in ("width", "height", "num_frames") if overrides.get(key)}}
    )

    with TestClient(web_server.app) as client:
        form = {"mode": "video", "model": "ltx-8", "prompt": "a lake", **{key: str(value) for key, value in fields.items()}}
        assert client.post("/api/generate", data=form).status_code == 200
    payload = _memory_fit(fields).json()

    (job,) = jobs
    assert (payload["width"], payload["height"], payload["frames"]) == (job["args"].width, job["args"].height, job["args"].num_frames)
    assert (payload["width"], payload["height"], payload["frames"]) == expected


def test_memory_fit_route_gives_no_fit_without_a_budget(clip_fit, monkeypatch):
    monkeypatch.setattr(web_server, "memory_budget", lambda: None)
    monkeypatch.setattr(web_server, "clip_memory_fit", real_clip_memory_fit)

    payload = _memory_fit({"frames": 97}).json()

    assert payload["memory_fit"] is None
    assert (payload["width"], payload["height"], payload["frames"]) == (736, 448, 49)


def test_memory_fit_route_gives_no_fit_for_a_model_that_cannot_be_estimated(clip_fit, monkeypatch):
    """Not LTX MLX, not downloaded, or the text encoder missing: the memory check finds nothing to estimate."""
    monkeypatch.setattr(web_server, "clip_memory_fit", lambda *_args, **_kwargs: None)

    response = _memory_fit({})

    assert response.status_code == 200
    assert response.json()["memory_fit"] is None


def test_memory_fit_route_answers_while_a_job_runs(clip_fit, monkeypatch):
    monkeypatch.setattr(web_server.web_runner, "has_active_jobs", lambda: True)

    assert _memory_fit({}).status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        [1, 2],
        {"model": 5},
        {"ratio": "3:1"},
        {"width": 0},
        {"low_memory": "yes"},
        {"upscale": 4},
    ],
)
def test_memory_fit_route_refuses_a_malformed_body(clip_fit, body):
    response = _memory_fit(body)

    assert response.status_code == 422
    assert set(response.json()) == {"detail"}
    assert clip_fit.calls == []


def test_memory_fit_route_refuses_a_body_that_is_not_json(clip_fit):
    with TestClient(web_server.app) as client:
        response = client.post("/api/video/memory-fit", content=b"not json", headers={"content-type": "application/json"})

    assert response.status_code == 422


def test_memory_fit_route_refuses_a_size_that_is_too_small(clip_fit, monkeypatch):
    monkeypatch.setattr(video_clip_module, "resolve_video_defaults", lambda *_args: {"steps": 8, "width": 16, "height": 448, "num_frames": 49})

    assert _memory_fit({}).status_code == 422
