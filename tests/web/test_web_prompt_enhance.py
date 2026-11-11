"""Tests for the Web UI prompt-enhancement API, generate fields, job lifecycle, and workspace contract."""

from __future__ import annotations

import dataclasses
import json
import queue
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from tests._factories import _make_args, load_packaged_config, make_extend_source, use_enhancer_models, video_config
from tests.web._helpers import make_web_config, patch_image_submit_dependencies

from zvisiongenerator.core.image_types import ImageGenerationRequest
from zvisiongenerator.core.video_types import VideoGenerationRequest
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.audio_writers import FITTED_NOTHING_TO_ADD_NOTE, LYRICS_UNCHANGED_MESSAGE, NOTHING_TO_ADD_MESSAGE, SECTIONS_CHANGED_MESSAGE
from zvisiongenerator.utils.prompt_enhance import QUOTES_CHANGED_WARNING, EnhanceContext, EnhanceSettings
from zvisiongenerator.utils.prompt_enhance_video import split_scene_action
from zvisiongenerator.utils.video_extend import ExtendInspection, ExtendProblem
from zvisiongenerator.web import audio_models as audio_models_module
from zvisiongenerator.web import extend_api as extend_api_module
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


def _without_model(frame: dict) -> dict:
    """Return a ``done`` frame without its ``model`` (the enhancer that wrote it), after checking it names one."""
    assert isinstance(frame.get("model"), str) and frame["model"]
    return {key: value for key, value in frame.items() if key != "model"}


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
        assert _without_model(frames[-1]) == {"type": "done", "prompt": "A grey fox in deep snow, golden light.", "clamped": False}
        assert enhancer.messages[0][1]["content"] in ("Prompt: a red fox", "Prompt: a grey fox")  # choices are picked before the model sees the prompt
        assert session.acquired[0][2] == 120

    def test_image_done_frame_carries_a_changed_quote_warning(self, monkeypatch, no_active_job):
        enhancer = _FakeEnhancer(["A diner under an open 24 hours sign.", "A diner under a neon Open 24 Hours sign."])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))
        body = {"prompt": 'a diner with a sign that says "OPEN 24 HOURS"', "mode": "image", "settings": {"style": "photo"}}
        with TestClient(web_server.app) as client:
            response = client.post("/api/prompt/enhance", json=body)
        done = _frames(response)[-1]
        assert done["type"] == "done"
        assert done["prompt"] == "A diner under a neon Open 24 Hours sign."
        assert done["warning"] == QUOTES_CHANGED_WARNING

    def test_video_pair_frames(self, monkeypatch, no_active_job):
        enhancer = _FakeEnhancer(["SCENE: A castle at dusk.\nACTION: A knight rides in."])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))
        body = {"scene": "A castle at dusk.", "prompt": "a knight", "mode": "video", "settings": {"style": "cinematic"}}
        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json=body))
        texts = [frame for frame in frames if frame["type"] == "text"]
        assert texts and all({"text", "scene", "prompt"} <= set(frame) for frame in texts)
        assert all((frame["scene"], frame["prompt"]) == split_scene_action(frame["text"]) for frame in texts)  # the server splits; the client renders the parts
        assert (texts[-1]["scene"], texts[-1]["prompt"]) == ("A castle at dusk.", "A knight rides in.")
        assert _without_model(frames[-1]) == {"type": "done", "scene": "A castle at dusk.", "prompt": "A knight rides in.", "clamped": False}
        assert "A castle at dusk." in enhancer.messages[0][1]["content"] and "a knight" in enhancer.messages[0][1]["content"]

    def test_video_prompt_only_is_split(self, monkeypatch, no_active_job):
        enhancer = _FakeEnhancer(["SCENE: A rainy street at night.\nACTION: A woman walks past a neon sign."])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))
        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json={"prompt": "a woman walks in the rain", "mode": "video", "settings": {"length": "same"}}))
        assert frames[-1]["type"] == "done"
        assert (frames[-1]["scene"], frames[-1]["prompt"]) == ("A rainy street at night.", "A woman walks past a neon sign.")

    def test_video_scene_only_is_accepted(self, monkeypatch, no_active_job):
        enhancer = _FakeEnhancer(["SCENE: A castle at dusk, banners in the wind.\nACTION: Wind moves the banners."])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))
        body = {"scene": "A castle at dusk.", "prompt": "", "mode": "video", "settings": {"style": "cinematic"}}
        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json=body))
        assert frames[-1]["type"] == "done" and frames[-1]["prompt"] == "Wind moves the banners."

    def test_video_fallback_warning_is_in_done_frame(self, monkeypatch, no_active_job):
        enhancer = _FakeEnhancer(["A knight rides in at dawn.", "A knight rides in at dawn, slowly."])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))
        body = {"scene": "A castle.", "prompt": "a knight", "mode": "video", "settings": {"style": "cinematic"}}
        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json=body))
        done = frames[-1]
        # The fallback leaves the scene unrewritten: the frame's scene is "" so the client keeps the user's scene.
        assert done["type"] == "done" and done["scene"] == "" and done["warning"]

    def test_video_action_only_frames_carry_no_scene(self, monkeypatch, no_active_job):
        enhancer = _FakeEnhancer(["The knight draws his sword and raises it high."])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))
        body = {"scene": "A castle at dusk.", "prompt": "he draws his sword", "mode": "video", "settings": {"parts": "action"}}
        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json=body))
        texts = [frame for frame in frames if frame["type"] == "text"]
        assert texts and all(frame["scene"] == "" and frame["prompt"] == frame["text"] for frame in texts)
        assert _without_model(frames[-1]) == {"type": "done", "scene": "", "prompt": "The knight draws his sword and raises it high.", "clamped": False}
        system, user = enhancer.messages[0][0]["content"], enhancer.messages[0][1]["content"]
        assert "A castle at dusk." in system and user == "he draws his sword"

    def test_json_format_writes_a_caption(self, monkeypatch, no_active_job):
        reply = "SUMMARY: A fox in snow.\nAESTHETICS: calm\nLIGHTING: soft\nCAMERA: 50mm\nCOLORS: white\nBACKGROUND: Snowy trees.\nOBJECT: A red fox."
        enhancer = _FakeEnhancer([reply])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))
        keep_all = {"style": "keep", "mood": "keep", "details": [], "length": "same"}
        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json={"prompt": "a fox", "mode": "image", "format": "json", "settings": keep_all}))
        done = frames[-1]
        assert done["type"] == "done" and "\n" not in done["prompt"]
        assert json.loads(done["prompt"])["compositional_deconstruction"]["elements"] == [{"type": "obj", "desc": "A red fox."}]
        assert any(frame["type"] == "text" and frame["text"].startswith("SUMMA") for frame in frames)  # the labelled lines stream

    @pytest.mark.parametrize(("key", "value"), [("enhance_model", "someone/repo"), ("format", "yaml")])
    def test_enhancer_and_format_are_allow_listed(self, key, value):
        with pytest.raises(ValueError, match=key):
            web_server._validate_enhance_body({"prompt": "a fox", "settings": {"style": "photo"}, key: value})

    def test_an_unknown_mode_is_refused_and_not_run_as_an_image(self):
        with pytest.raises(ValueError, match="mode"):
            web_server._validate_enhance_body({"prompt": "a fox", "mode": "text", "settings": {"style": "photo"}})

    def test_smart_resolves_the_smart_model(self, monkeypatch, tmp_path):
        monkeypatch.setattr(web_server, "smart_budget_bytes", lambda choice: 64 * 2**30)
        use_enhancer_models(monkeypatch, tmp_path, fast="fast/m", smart="smart/m", smart_memory_gb=9)
        monkeypatch.setattr(web_server, "load_config", lambda: {})
        job = web_server._validate_enhance_body({"prompt": "a fox", "settings": {"style": "photo"}, "enhance_model": "smart"})
        assert (job["repo"], job["revision"]) == ("smart/m", None)

    def test_video_action_only_with_keep_all_settings_is_valid(self, no_active_job):
        keep_all = {"style": "keep", "mood": "keep", "details": [], "length": "same", "motion": [], "sound": [], "parts": "action"}
        job = web_server._validate_enhance_body({"scene": "A castle.", "prompt": "a knight", "mode": "video", "settings": keep_all})
        assert job["settings"].parts == "action"
        both = web_server._validate_enhance_body({"scene": "A castle.", "prompt": "a knight", "mode": "video", "settings": {**keep_all, "parts": "both"}})
        assert both["settings"].parts == "both"  # nothing picked is a general rewrite, not a refusal

    def test_video_action_only_with_a_blank_scene_splits_into_both(self, monkeypatch, no_active_job):
        enhancer = _FakeEnhancer(["SCENE: A rainy street at night.\nACTION: A woman walks past a neon sign."])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))
        body = {"scene": "", "prompt": "a woman walks in the rain", "mode": "video", "settings": {"parts": "action"}}
        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json=body))
        texts = [frame for frame in frames if frame["type"] == "text"]
        assert (texts[-1]["scene"], texts[-1]["prompt"]) == ("A rainy street at night.", "A woman walks past a neon sign.")
        assert frames[-1]["scene"] == "A rainy street at night."

    def test_video_parts_both_rewrites_the_scene(self, monkeypatch, no_active_job):
        enhancer = _FakeEnhancer(["SCENE: A castle at dusk, banners.\nACTION: A knight rides in."])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))
        body = {"scene": "A castle at dusk.", "prompt": "a knight", "mode": "video", "settings": {"parts": "both"}}
        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json=body))
        assert frames[-1]["scene"] == "A castle at dusk, banners."

    def test_invalid_parts_is_rejected(self, no_active_job):
        with TestClient(web_server.app) as client:
            response = client.post("/api/prompt/enhance", json={"scene": "A castle.", "prompt": "a knight", "mode": "video", "settings": {"parts": "scene"}})
        assert response.status_code == 422

    def test_parts_is_ignored_for_images(self):
        with pytest.warns(UserWarning, match="ignored for images"):
            job = web_server._validate_enhance_body({"prompt": "a fox", "mode": "image", "settings": {"style": "photo", "parts": "action"}})
        assert job["settings"].parts is None

    def test_image_done_frame_has_no_scene(self, monkeypatch, no_active_job):
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(_FakeEnhancer(["A grey fox in snow."])))
        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json={"prompt": "a fox", "scene": "ignored", "mode": "image", "settings": {"style": "photo"}}))
        assert all("scene" not in frame for frame in frames)
        assert _without_model(frames[-1]) == {"type": "done", "prompt": "A grey fox in snow.", "clamped": False}

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
            ({"prompt": " ", "scene": " ", "mode": "video", "settings": {}}, "Enter a scene or a prompt"),
            ({"prompt": "a fox", "scene": 3, "mode": "video", "settings": {}}, "scene must be a string"),
            ({"prompt": "a fox", "settings": {"style": "nope"}}, "Unknown style"),
            ({"prompt": "a fox", "mode": "text"}, "mode must be 'image', 'video' or 'audio'"),
            ({"prompt": "a fox", "max_words": 0}, "max_words"),
            ([1, 2], "JSON object"),
            ({"prompt": "a fox", "context": "video"}, "context must be"),
            ({"prompt": "a fox", "context": {"frames": -1}}, "context.frames"),
            ({"prompt": "a fox", "context": {"ratio": 16}}, "context.ratio"),
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

    def test_context_is_parsed(self):
        body = {
            "prompt": "a fox",
            "mode": "video",
            "settings": {"sound": ["ambient"], "note": "at dusk"},
            "context": {"width": 448, "height": 704, "frames": 97, "fps": 24, "audio": True, "reference_image": True},
        }
        job = web_server._validate_enhance_body(body)
        assert job["context"] == EnhanceContext(audio=True, seconds=97 / 24, orientation="portrait", reference_image=True)
        assert (job["settings"].sound, job["settings"].note) == (("ambient",), "at dusk")

    def test_context_carries_the_reference_count(self):
        def parsed(context):
            return web_server._validate_enhance_body({"prompt": "a fox", "mode": "image", "context": context})["context"]

        assert parsed({"references": 3, "reference_image": True}) == EnhanceContext(reference_image=True, references=3)
        assert parsed({"width": 512, "height": 512}).references == 0

    @pytest.mark.parametrize("value", [-1, "2", True])
    def test_context_reference_count_is_a_non_negative_whole_number(self, value):
        with pytest.raises(ValueError, match="context.references"):
            web_server._validate_enhance_body({"prompt": "a fox", "mode": "image", "context": {"references": value}})

    def test_context_last_frame_is_a_strict_boolean(self):
        def parsed(value):
            return web_server._validate_enhance_body({"prompt": "a fox", "mode": "video", "context": {"reference_image": True, "end_image": value}})["context"]

        assert parsed(True).end_image is True
        assert parsed("yes").end_image is False
        assert web_server._validate_enhance_body({"prompt": "a fox", "mode": "video", "context": {"reference_image": True}})["context"].end_image is False

    def test_context_audio_is_off_when_video_has_no_audio_track(self, monkeypatch):
        monkeypatch.setattr(web_server, "supports_video_audio", lambda: False)
        body = {"prompt": "a fox", "mode": "video", "context": {"frames": 97, "fps": 24, "audio": True}}

        assert web_server._validate_enhance_body(body)["context"].audio is False

    def test_video_split_makes_default_settings_valid(self):
        job = web_server._validate_enhance_body({"prompt": "a fox", "mode": "video", "settings": {"style": "keep", "details": [], "length": "same"}})
        assert job["scene"] == ""

    def test_video_scene_without_prompt_field_is_accepted(self):
        job = web_server._validate_enhance_body({"scene": "A castle.", "mode": "video", "settings": {"style": "cinematic"}})
        assert (job["scene"], job["prompt"]) == ("A castle.", "")

    def test_video_prompt_must_be_a_string(self, no_active_job):
        with TestClient(web_server.app) as client:
            response = client.post("/api/prompt/enhance", json={"prompt": 5, "mode": "video", "settings": {"style": "cinematic"}})
        assert response.status_code == 422

    def test_image_ignores_a_scene_of_any_type(self):
        job = web_server._validate_enhance_body({"prompt": "a fox", "scene": 3, "mode": "image", "settings": {"style": "photo"}})
        assert job["scene"] == ""

    def test_missing_context_is_empty(self):
        assert web_server._validate_enhance_body({"prompt": "a fox"})["context"] == EnhanceContext()

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


# ── /api/prompt/enhance, audio ──────────────────────────────────────────────


@pytest.fixture
def audio_models(monkeypatch):
    """Make this machine a CUDA one that runs the packaged audio models, and serve the packaged config to the enhance endpoint."""
    monkeypatch.setattr(audio_models_module, "audio_platform_key", lambda: "cuda")
    config = load_packaged_config()
    monkeypatch.setattr(web_server, "load_config", lambda: config)
    return config


def _audio_body(**overrides) -> dict:
    return {"mode": "audio", "task": "sound", "model": "sa3", "kind": "sound", "prompt": "rain on a tin roof", "settings": {}, **overrides}


class TestAudioEnhanceEndpoint:
    def test_a_sound_streams_status_text_and_a_done_frame_with_its_fields(self, monkeypatch, no_active_job, audio_models):
        enhancer = _FakeEnhancer(["Heavy rain falls on a corrugated tin roof, steady and close."])
        session = _FakeSession(enhancer)
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: session)
        body = _audio_body(settings={"space": "small-room", "note": "keep it dry"}, context={"sound_type": "ambience", "seconds": 12})

        with TestClient(web_server.app) as client:
            response = client.post("/api/prompt/enhance", json=body)

        frames = _frames(response)
        assert response.status_code == 200 and response.headers["content-type"].startswith("application/x-ndjson")
        assert frames[:2] == [{"type": "status", "phase": "loading"}, {"type": "status", "phase": "generating"}]
        texts = [frame for frame in frames if frame["type"] == "text"]
        assert texts and all(frame["task"] == "sound" and frame["fields"] == {"prompt": frame["text"]} for frame in texts)
        assert texts[-1]["text"] == "Heavy rain falls on a corrugated tin roof, steady and close."
        assert _without_model(frames[-1]) == {"type": "done", "task": "sound", "fields": {"prompt": "Heavy rain falls on a corrugated tin roof, steady and close."}, "clamped": False}
        system = enhancer.messages[0][0]["content"]
        assert "one sentence of at most 45 words" in system and "Space: a small room" in system and "This is an ambience" in system and "keep it dry" in system
        assert enhancer.messages[0][1]["content"] == "Prompt: rain on a tin roof"

    def test_a_style_is_written_for_the_family_of_the_model(self, monkeypatch, no_active_job, audio_models):
        enhancer = _FakeEnhancer(["Folk, piano, wistful vocal, no drums", "Warm folk music with a soft piano and a gentle, wistful mood."])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))
        with TestClient(web_server.app) as client:
            yue2 = _frames(client.post("/api/prompt/enhance", json=_audio_body(task="style", model="yue2", kind="music", prompt="sad folk", settings={"genre": "folk", "without": ["drums"]})))
            sa3 = _frames(client.post("/api/prompt/enhance", json=_audio_body(task="style", model="sa3", kind="music", prompt="sad folk")))

        assert yue2[-1]["fields"] == {"style": "Folk, piano, wistful vocal, no drums"} and sa3[-1]["fields"] == {"style": "Warm folk music with a soft piano and a gentle, wistful mood."}
        assert "comma-separated descriptors" in enhancer.messages[0][0]["content"] and "no drums" in enhancer.messages[0][0]["content"]
        assert "one sentence of at most 45 words" in enhancer.messages[1][0]["content"]

    def test_a_style_is_rewritten_with_the_take_facts_as_context(self, monkeypatch, no_active_job, audio_models):
        enhancer = _FakeEnhancer(["Folk, acoustic guitar, wistful"])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))
        body = _audio_body(task="style", model="yue2", kind="music", prompt="a sad folk song about leaving the coast", context={"instrumental": True, "tempo": 90, "key": "Am"})

        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json=body))

        assert frames[-1]["fields"] == {"style": "Folk, acoustic guitar, wistful"}
        assert enhancer.messages[0][1]["content"] == "Prompt: a sad folk song about leaving the coast"
        assert "The song is instrumental" in enhancer.messages[0][0]["content"] and "BPM: 90" not in enhancer.messages[0][0]["content"]  # the tempo is a setting, not part of the rewrite

    def test_the_smart_enhancer_can_be_chosen(self, monkeypatch, audio_models):
        monkeypatch.setattr(web_server, "smart_budget_bytes", lambda choice: 64 * 2**30)

        job = web_server._validate_enhance_body(_audio_body(enhance_model="smart"))

        assert job["repo"] != web_server._validate_enhance_body(_audio_body())["repo"]

    def test_the_job_carries_the_task_family_context_and_settings(self, audio_models):
        job = web_server._validate_enhance_body(
            _audio_body(task="style", model="yue2", kind="music", prompt="folk", settings={"genre": "folk"}, context={"lyrics": "x" * 900, "tempo": "90", "sound_type": "effect"})
        )

        assert (job["mode"], job["task"], job["kind"], job["model"], job["family"], job["prompt"]) == ("audio", "style", "music", "yue2", "yue2", "folk")
        assert job["settings"].genre == "folk"
        assert (len(job["context"].lyrics), job["context"].tempo, job["context"].sound_type) == (600, 90, "effect")
        assert {"repo", "revision", "options", "idle_seconds"} <= set(job)

    def test_the_kind_defaults_to_the_one_of_the_task(self, audio_models):
        body = _audio_body(task="style", model="sa3")
        del body["kind"]

        assert web_server._validate_enhance_body(body)["kind"] == "music"

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            (_audio_body(task="poem"), "task must be one of: sound, style, lyrics, script, fill"),
            ({k: v for k, v in _audio_body().items() if k != "task"}, "task must be one of"),
            (_audio_body(model="nope"), "Unknown audio model 'nope'"),
            ({k: v for k, v in _audio_body().items() if k != "model"}, "Field 'model' is required"),
            (_audio_body(model=3), "model must be an audio model id"),
            (_audio_body(model="yue2"), "does not make sound"),
            (_audio_body(kind="music"), "The sound task works on sound, not music"),
            (_audio_body(kind="radio"), "kind must be one of"),
            (_audio_body(prompt="  "), "Enter a sound to enhance"),
            (_audio_body(task="style", kind="music", prompt=""), "Enter a style to enhance"),
            (_audio_body(task="style", kind="music", prompt="  "), "Enter a style to enhance"),
            (_audio_body(prompt=3), "prompt must be a string"),
            (_audio_body(prompt="x" * 1001), "too long"),
            (_audio_body(settings={"genre": "folk"}), "only applies to music"),
            (_audio_body(settings={"space": "nowhere"}), "Unknown space"),
            (_audio_body(settings="forest"), "must be a mapping"),
            (_audio_body(context="loud"), "context must be a JSON object"),
            (_audio_body(context={"seconds": 0}), "context.seconds must be a positive number"),
            (_audio_body(context={"seconds": True}), "context.seconds"),
            (_audio_body(context={"sound_type": "song"}), "Sound type must be one of"),
            (_audio_body(task="style", kind="music", model="yue2", context={"tempo": 300}), "Tempo must be 40 to 220 BPM"),
            (_audio_body(task="style", kind="music", model="yue2", context={"key": "H"}), "Key must be one of"),
            (_audio_body(enhance_model="turbo"), "enhance_model must be one of"),
        ],
    )
    def test_a_malformed_audio_request_is_a_422_in_words_the_user_sees(self, body, message, no_active_job, audio_models):
        with TestClient(web_server.app) as client:
            response = client.post("/api/prompt/enhance", json=body)

        assert response.status_code == 422
        assert message in response.json()["detail"]

    def test_a_machine_without_audio_models_refuses_the_request(self, monkeypatch, no_active_job, audio_models):
        monkeypatch.setattr(audio_models_module, "audio_platform_key", lambda: (_ for _ in ()).throw(RuntimeError("Audio models need an NVIDIA GPU or Apple Silicon.")))
        with TestClient(web_server.app) as client:
            response = client.post("/api/prompt/enhance", json=_audio_body())

        assert response.status_code == 422 and "Audio models need" in response.json()["detail"]

    def test_the_request_is_refused_with_409_while_a_job_runs(self, monkeypatch, audio_models):
        def _job_running(admit, *, busy_message):
            raise web_runner_module.JobConflictError(busy_message)

        monkeypatch.setattr(web_server.web_runner, "admit_exclusive", _job_running)
        with TestClient(web_server.app) as client:
            response = client.post("/api/prompt/enhance", json=_audio_body())

        assert response.status_code == 409 and "all jobs have finished" in response.json()["detail"]

    def test_a_running_audio_job_refuses_the_request_through_the_real_admission(self, monkeypatch, audio_models):
        """The runner's own check (not a stub) turns an active exclusive job into the 409, and the enhancer slot is not taken."""
        session = _FakeSession(_FakeEnhancer(["unused"]))
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: session)
        monkeypatch.setattr(web_server.web_runner, "_find_active_exclusive_job_id", lambda: "audio-job")
        with TestClient(web_server.app) as client:
            response = client.post("/api/prompt/enhance", json=_audio_body())

        assert response.status_code == 409 and "all jobs have finished" in response.json()["detail"]
        assert session.acquired == []

    def test_a_running_enhancement_refuses_a_job_through_the_real_admission_check(self, monkeypatch):
        monkeypatch.setattr(web_server.web_runner, "_find_active_exclusive_job_id", lambda: None)
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(busy=True))

        with pytest.raises(web_runner_module.JobConflictError, match="Prompt enhancement in progress"):
            web_server._reject_while_busy()

    def test_the_request_is_refused_with_409_while_another_enhancement_runs(self, monkeypatch, no_active_job, audio_models):
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(busy=True))
        with TestClient(web_server.app) as client:
            assert client.post("/api/prompt/enhance", json=_audio_body()).status_code == 409

    def test_unusable_output_is_an_error_frame(self, monkeypatch, no_active_job, audio_models):
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(_FakeEnhancer(["", "rain on a tin roof"])))
        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json=_audio_body()))

        assert frames[-1]["type"] == "error" and "no usable rewrite" in frames[-1]["detail"]

    def test_a_changed_quote_ends_in_a_done_frame_with_a_warning(self, monkeypatch, no_active_job, audio_models):
        enhancer = _FakeEnhancer(["A door shuts and a voice says go away.", "A door shuts and a voice says go away again."])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))
        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json=_audio_body(prompt='a door shuts, a voice says "Go Away"')))

        assert frames[-1]["type"] == "done" and frames[-1]["warning"] == QUOTES_CHANGED_WARNING

    def test_the_done_frame_of_each_mode_keeps_its_shape(self):
        image = web_server._enhance_done_frame("image", web_server.EnhanceResult(prompt="A fox.", clamped=False), model="org/m")
        audio = web_server._enhance_done_frame("audio", web_server.EnhanceResult(prompt="Rain.", clamped=False, warning="careful"), model="org/m", task="sound")

        assert image == {"type": "done", "prompt": "A fox.", "clamped": False, "model": "org/m"}
        assert audio == {"type": "done", "task": "sound", "fields": {"prompt": "Rain."}, "clamped": False, "model": "org/m", "warning": "careful"}

    def test_image_and_video_requests_are_unchanged(self, monkeypatch, no_active_job):
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(_FakeEnhancer(["A fox in snow.", "SCENE: A snowy field.\nACTION: A fox trots."])))
        with TestClient(web_server.app) as client:
            image = _frames(client.post("/api/prompt/enhance", json={"prompt": "a fox", "settings": {}}))[-1]
            video = _frames(client.post("/api/prompt/enhance", json={"prompt": "a fox", "scene": "a field", "mode": "video", "settings": {}}))[-1]

        assert _without_model(image) == {"type": "done", "prompt": "A fox in snow.", "clamped": False}
        assert set(video) == {"type", "scene", "prompt", "clamped", "model"} and "fields" not in video and "task" not in video


_SONG = "[Verse]\nThe porch light flickers, the street is wet\nI still keep your coat by the door\n\n[Chorus]\nCome back home\n\n[Bridge]\n\n[Outro]\n"
_SHORTER = "[Verse]\nPorch light flickers, street is wet\nI keep your coat by the door\n\n[Chorus]\nCome home\n\n[Bridge]\n\n[Outro]\n"


def _lyrics_body(**overrides) -> dict:
    return _audio_body(**{"task": "lyrics", "model": "yue2", "kind": "music", "prompt": _SONG, "settings": {"improve": ["tighten"]}, **overrides})


class TestLyricsEnhanceEndpoint:
    def test_the_whole_song_is_improved_and_comes_back_in_the_lyrics_field(self, monkeypatch, no_active_job, audio_models):
        enhancer = _FakeEnhancer([_SHORTER])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))
        body = _lyrics_body(settings={"improve": ["tighten", "rhymes"], "note": "more hopeful"}, context={"style": "warm folk, piano"})

        with TestClient(web_server.app) as client:
            response = client.post("/api/prompt/enhance", json=body)

        frames = _frames(response)
        assert response.status_code == 200 and frames[:2] == [{"type": "status", "phase": "loading"}, {"type": "status", "phase": "generating"}]
        assert not [frame for frame in frames if frame["type"] == "text"]
        assert _without_model(frames[-1]) == {"type": "done", "task": "lyrics", "fields": {"lyrics": _SHORTER}, "clamped": False}
        system, user = (message["content"] for message in enhancer.messages[0])
        assert "Tighten" in system and "Smooth rhymes" in system and "more hopeful" in system and "warm folk, piano" in system
        assert user == f"Lyrics:\n{_SONG}"

    def test_one_section_is_improved_and_the_whole_song_comes_back(self, monkeypatch, no_active_job, audio_models):
        enhancer = _FakeEnhancer(["Come home"])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))

        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json=_lyrics_body(settings={"improve": ["simpler"], "section": 1})))

        assert frames[-1]["fields"] == {"lyrics": _SONG.replace("Come back home", "Come home")}
        assert "Section to edit: [Chorus]" in enhancer.messages[0][1]["content"]

    def test_an_empty_section_is_written_to_fit_the_rest(self, monkeypatch, no_active_job, audio_models):
        enhancer = _FakeEnhancer(["The night is long\nbut I am still here"])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))

        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json=_lyrics_body(settings={"action": "write", "section": 2})))

        assert frames[-1]["fields"] == {"lyrics": _SONG.replace("[Bridge]\n", "[Bridge]\nThe night is long\nbut I am still here\n")}
        assert "Empty section to write: [Bridge]" in enhancer.messages[0][1]["content"]

    def test_changed_sections_end_the_stream_in_an_error_frame_after_one_retry(self, monkeypatch, no_active_job, audio_models):
        enhancer = _FakeEnhancer(["[Verse]\nonly this", "[Verse]\nstill only this"])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))

        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json=_lyrics_body()))

        assert frames[-1] == {"type": "error", "detail": SECTIONS_CHANGED_MESSAGE} and len(enhancer.messages) == 2

    def test_lyrics_left_as_they_were_end_the_stream_in_a_no_change_frame_the_client_can_tell_apart(self, monkeypatch, no_active_job, audio_models):
        enhancer = _FakeEnhancer([_SONG, _SONG])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))

        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json=_lyrics_body()))

        assert frames[-1] == {"type": "error", "detail": LYRICS_UNCHANGED_MESSAGE, "no_change": True}

    def test_the_job_carries_the_lyrics_settings_the_style_and_the_music_kind(self, audio_models):
        job = web_server._validate_enhance_body(_lyrics_body(settings={"improve": ["fit"], "note": " calm "}, context={"style": "x" * 900, "lyrics": "ignored"}))

        assert (job["task"], job["kind"], job["model"], job["family"], job["prompt"]) == ("lyrics", "music", "yue2", "yue2", _SONG)
        assert (job["settings"].improve, job["settings"].note, job["settings"].section) == (("fit",), "calm", None)
        assert len(job["context"].style) == 400

    def test_the_kind_defaults_to_music(self, audio_models):
        body = _lyrics_body()
        del body["kind"]

        assert web_server._validate_enhance_body(body)["kind"] == "music"

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            (_lyrics_body(prompt=""), "Nothing to improve yet"),
            (_lyrics_body(prompt="[Verse]\n\n[Chorus]"), "Nothing to improve yet"),
            (_lyrics_body(prompt="a" * 4001), "too long"),
            (_lyrics_body(prompt=3), "prompt must be a string"),
            (_lyrics_body(model="sa3"), "does not take lyrics"),
            (_lyrics_body(kind="sound"), "The lyrics task works on music, not sound"),
            (_lyrics_body(settings={"improve": ["louder"]}), "Unknown improvement"),
            (_lyrics_body(settings={"action": "write"}), "name the section"),
            (_lyrics_body(settings={"section": 9}), "no section 10"),
            (_lyrics_body(settings={"section": 2}), "has no words yet: use Write"),
            (_lyrics_body(settings={"action": "write", "section": 0}), "already has words: use Improve"),
            (_lyrics_body(settings={"genre": "folk"}), "Unknown lyrics setting"),
            (_lyrics_body(settings="tighten"), "must be a mapping"),
            (_lyrics_body(context={"style": 3}), "context.style must be text"),
        ],
    )
    def test_a_malformed_lyrics_request_is_a_422_in_words_the_user_sees(self, body, message, no_active_job, audio_models):
        with TestClient(web_server.app) as client:
            response = client.post("/api/prompt/enhance", json=body)

        assert response.status_code == 422
        assert message in response.json()["detail"]


_SCRIPT = "I never told anyone. Not even you. The door was open."
_MARKED = "[whisper] I never told anyone. …\nNot even you. [as set] The door was open."


def _script_body(**overrides) -> dict:
    return _audio_body(**{"task": "script", "model": "qwen3_tts", "kind": "speech", "prompt": _SCRIPT, "settings": {"actions": ["delivery"]}, **overrides})


class TestScriptEnhanceEndpoint:
    def test_delivery_is_added_and_the_script_comes_back_in_the_script_field(self, monkeypatch, no_active_job, audio_models):
        enhancer = _FakeEnhancer([_MARKED])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))

        with TestClient(web_server.app) as client:
            response = client.post("/api/prompt/enhance", json=_script_body())

        frames = _frames(response)
        assert response.status_code == 200 and frames[:2] == [{"type": "status", "phase": "loading"}, {"type": "status", "phase": "generating"}]
        assert not [frame for frame in frames if frame["type"] == "text"]
        assert _without_model(frames[-1]) == {"type": "done", "task": "script", "fields": {"script": _MARKED}, "clamped": False}
        assert "model" in frames[-1]
        system, user = (message["content"] for message in enhancer.messages[0])
        assert "[whisper]" in system and "Never change the words" in system and user == f"Script:\n{_SCRIPT}"

    def test_a_script_is_fitted_to_a_time_at_a_pace(self, monkeypatch, no_active_job, audio_models):
        enhancer = _FakeEnhancer(["y" * 227])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))
        body = _script_body(prompt="The harbour was quiet that morning. " * 20, settings={"actions": ["fit"], "seconds": 20, "pace": "slow"})

        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json=body))

        assert frames[-1]["fields"] == {"script": "y" * 227} and "warning" not in frames[-1]
        assert "to about 227 characters" in enhancer.messages[0][0]["content"]

    def test_both_actions_are_one_request_that_fits_then_marks_up_with_one_done_frame(self, monkeypatch, no_active_job, audio_models):
        enhancer = _FakeEnhancer(["The harbour was quiet. The tide came in.", "The harbour was quiet. …\n[sad] The tide came in."])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))
        body = _script_body(prompt="The harbour was quiet that morning. " * 20, settings={"actions": ["fit", "delivery"], "seconds": 3})

        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json=body))

        done = [frame for frame in frames if frame["type"] == "done"]
        assert len(done) == 1 and done[0]["fields"] == {"script": "The harbour was quiet. …\n[sad] The tide came in."} and "warning" not in done[0]
        assert len(enhancer.messages) == 2

    def test_a_delivery_that_adds_nothing_after_a_fit_keeps_the_fitted_script_with_a_note(self, monkeypatch, no_active_job, audio_models):
        fitted = "The harbour was quiet. The tide came in."
        enhancer = _FakeEnhancer([fitted, fitted, fitted])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))
        body = _script_body(prompt="The harbour was quiet that morning. " * 20, settings={"actions": ["fit", "delivery"], "seconds": 3})

        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json=body))

        assert frames[-1]["type"] == "done" and frames[-1]["fields"] == {"script": fitted} and frames[-1]["warning"] == FITTED_NOTHING_TO_ADD_NOTE

    def test_a_fit_far_from_its_time_carries_the_time_it_came_out_at(self, monkeypatch, no_active_job, audio_models):
        enhancer = _FakeEnhancer(["y" * 700, "y" * 600])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))
        body = _script_body(prompt="The harbour was quiet that morning. " * 20, settings={"actions": ["fit"], "seconds": 20})

        with TestClient(web_server.app) as client, pytest.warns(UserWarning, match="Came out at about 0:43"):
            frames = _frames(client.post("/api/prompt/enhance", json=body))

        assert frames[-1]["warning"] == "Came out at about 0:43."

    def test_an_answer_that_adds_nothing_ends_the_stream_in_an_error_frame_after_one_retry(self, monkeypatch, no_active_job, audio_models):
        body = _script_body()
        enhancer = _FakeEnhancer([body["prompt"], body["prompt"]])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))

        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json=body))

        assert frames[-1] == {"type": "error", "detail": NOTHING_TO_ADD_MESSAGE, "no_change": True} and len(enhancer.messages) == 2

    def test_words_the_helper_changed_come_back_as_the_scripts_own(self, monkeypatch, no_active_job, audio_models):
        body = _script_body()
        enhancer = _FakeEnhancer(["[sad] Nobody was told."])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))

        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json=body))

        assert frames[-1]["fields"] == {"script": f"[sad] {body['prompt']}"}

    def test_the_text_the_helper_sends_back_never_holds_a_mark_the_server_would_refuse(self, monkeypatch, no_active_job, audio_models):
        enhancer = _FakeEnhancer(["[Sad] I never told anyone. [sighs] Not even you. The door was open."])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))

        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json=_script_body()))

        assert frames[-1]["fields"] == {"script": "[sad] I never told anyone. Not even you. The door was open."}

    def test_the_job_carries_the_script_settings_and_the_speech_kind(self, audio_models):
        job = web_server._validate_enhance_body(_script_body(settings={"actions": ["fit"], "seconds": 30}))

        assert (job["task"], job["kind"], job["model"], job["family"], job["prompt"]) == ("script", "speech", "qwen3_tts", "qwen3_tts", _SCRIPT)
        assert (job["settings"].actions, job["settings"].seconds, job["settings"].pace) == (("fit",), 30, "normal")

    def test_the_kind_defaults_to_speech(self, audio_models):
        body = _script_body()
        del body["kind"]

        assert web_server._validate_enhance_body(body)["kind"] == "speech"

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            (_script_body(prompt=""), "Nothing to work on yet"),
            (_script_body(prompt="[sad] …"), "Nothing to work on yet"),
            (_script_body(prompt="a" * 4001), "too long"),
            (_script_body(prompt=3), "prompt must be a string"),
            (_script_body(model="yue2"), "does not make speech"),
            (_script_body(kind="music"), "The script task works on speech, not music"),
            (_script_body(settings={}), "actions must list each"),
            (_script_body(settings={"actions": ["fit"]}), "whole number from 3 to 600"),
            (_script_body(settings={"actions": ["fit"], "seconds": 5000}), "whole number from 3 to 600"),
            (_script_body(settings={"actions": ["fit"], "seconds": 20, "pace": "warp"}), "pace must be one of"),
            (_script_body(settings={"actions": ["delivery"], "seconds": 20}), "takes no reading time"),
            (_script_body(settings={"actions": ["delivery"], "genre": "folk"}), "Unknown script setting"),
            (_script_body(settings="fit"), "must be a mapping"),
        ],
    )
    def test_a_malformed_script_request_is_a_422_in_words_the_user_sees(self, body, message, no_active_job, audio_models):
        with TestClient(web_server.app) as client:
            response = client.post("/api/prompt/enhance", json=body)

        assert response.status_code == 422
        assert message in response.json()["detail"]


_IDEA = "a sad folk song about leaving the coast"
_FILLED = "STYLE: folk, acoustic guitar, wistful vocal\nTEMPO: 84\nKEY: A minor\nLYRICS:\n[Verse]\nI pack the boat at dawn\n\n[Chorus]\nGoodbye to the coast"


def _fill_body(**overrides) -> dict:
    return _audio_body(**{"task": "fill", "model": "yue2", "kind": "music", "prompt": _IDEA, "settings": {"fields": ["style", "tempo", "key", "lyrics"]}, **overrides})


class TestFillEnhanceEndpoint:
    def test_one_call_writes_several_fields_into_one_done_frame(self, monkeypatch, no_active_job, audio_models):
        enhancer = _FakeEnhancer([_FILLED])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))

        with TestClient(web_server.app) as client:
            response = client.post("/api/prompt/enhance", json=_fill_body())

        frames = _frames(response)
        assert response.status_code == 200 and frames[:2] == [{"type": "status", "phase": "loading"}, {"type": "status", "phase": "generating"}]
        assert not [frame for frame in frames if frame["type"] == "text"]
        assert _without_model(frames[-1]) == {
            "type": "done",
            "task": "fill",
            "fields": {"style": "folk, acoustic guitar, wistful vocal", "tempo": "84", "key": "Am", "lyrics": "[Verse]\nI pack the boat at dawn\n\n[Chorus]\nGoodbye to the coast"},
            "clamped": False,
        }
        assert "model" in frames[-1] and len(enhancer.messages) == 1
        assert enhancer.messages[0][1]["content"] == f"Idea:\n{_IDEA}"

    def test_a_speech_fill_writes_a_text_and_a_vibe_and_what_is_set_is_context(self, monkeypatch, no_active_job, audio_models):
        enhancer = _FakeEnhancer(["VIBE: whispered-secret"])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))
        body = _fill_body(model="qwen3_tts", kind="speech", prompt="read it as a secret", settings={"fields": ["vibe"]}, context={"text": "Come closer."})

        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json=body))

        assert frames[-1]["fields"] == {"vibe": "whispered-secret"}
        assert "Text (start): Come closer." in enhancer.messages[0][1]["content"]

    def test_a_field_the_helper_gets_wrong_is_left_out_with_a_warning(self, monkeypatch, no_active_job, audio_models):
        enhancer = _FakeEnhancer(["STYLE: folk\nTEMPO: 900", "STYLE: folk\nTEMPO: 1000"])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))

        with TestClient(web_server.app) as client, pytest.warns(UserWarning, match="Left empty: Tempo"):
            frames = _frames(client.post("/api/prompt/enhance", json=_fill_body(settings={"fields": ["style", "tempo"]})))

        assert frames[-1]["fields"] == {"style": "folk"} and frames[-1]["warning"] == "Left empty: Tempo."

    def test_nothing_usable_ends_the_stream_in_an_error_frame(self, monkeypatch, no_active_job, audio_models):
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(_FakeEnhancer(["no", "still no"])))

        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json=_fill_body()))

        assert frames[-1]["type"] == "error" and "couldn't fill anything" in frames[-1]["detail"]

    def test_the_kind_defaults_to_music_and_the_job_carries_the_fields_and_context(self, audio_models):
        body = _fill_body(context={"style": "x" * 900, "lyrics": "y" * 900, "text": "z" * 900, "tempo": 100, "key": "Em"})
        del body["kind"]
        job = web_server._validate_enhance_body(body)

        assert (job["task"], job["kind"], job["model"], job["family"], job["prompt"]) == ("fill", "music", "yue2", "yue2", _IDEA)
        assert (job["settings"].kind, job["settings"].fields) == ("music", ("style", "tempo", "key", "lyrics"))
        assert (len(job["context"].style), len(job["context"].lyrics), len(job["context"].text), job["context"].tempo, job["context"].key) == (400, 600, 600, 100, "Em")

    def test_a_model_that_cannot_sing_takes_a_fill_without_lyrics(self, audio_models):
        job = web_server._validate_enhance_body(_fill_body(model="sa3", settings={"fields": ["style", "tempo", "key"]}))

        assert job["family"] == "stable_audio_3" and job["settings"].fields == ("style", "tempo", "key")

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            (_fill_body(prompt=""), "Describe what you want first"),
            (_fill_body(prompt="   "), "Describe what you want first"),
            (_fill_body(prompt="x" * 1001), "description is too long"),
            (_fill_body(prompt=3), "prompt must be a string"),
            (_fill_body(kind="sound", model="sa3"), "fills music and speech, not sound"),
            (_fill_body(model="sa3"), "does not take lyrics"),
            (_fill_body(model="qwen3_tts", kind="speech", settings={"fields": ["text", "lyrics"]}), "Unknown speech field"),
            (_fill_body(model="yue2", kind="speech", settings={"fields": ["text"]}), "does not make speech"),
            (_fill_body(settings={}), "fields must list what to fill"),
            (_fill_body(settings={"fields": ["mood"]}), "Unknown music field"),
            (_fill_body(settings={"fields": ["style"], "note": "x"}), "Unknown fill setting"),
            (_fill_body(settings="style"), "must be a mapping"),
            (_fill_body(context={"text": 3}), "context.text must be text"),
        ],
    )
    def test_a_malformed_fill_request_is_a_422_in_words_the_user_sees(self, body, message, no_active_job, audio_models):
        with TestClient(web_server.app) as client:
            response = client.post("/api/prompt/enhance", json=body)

        assert response.status_code == 422
        assert message in response.json()["detail"]

    def test_the_done_frame_of_a_fill_is_its_fields(self):
        from zvisiongenerator.utils.audio_writers import FillResult

        frame = web_server._enhance_done_frame("audio", FillResult(fields={"tempo": "84"}, warning="Left empty: Key."), model="org/m", task="fill")

        assert frame == {"type": "done", "task": "fill", "fields": {"tempo": "84"}, "clamped": False, "model": "org/m", "warning": "Left empty: Key."}


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

    @pytest.mark.parametrize(("raw", "message"), [("{bad", "JSON object"), ('{"style": "vaporwave"}', "Unknown style")])
    def test_invalid(self, raw, message):
        with pytest.raises(ValueError, match=message):
            web_server._apply_enhance_submission(SimpleNamespace(), {"enhance_auto": "true", "enhance_settings": raw}, mode="video", json_caption=False)

    _KEEP_ALL = json.dumps({"style": "keep", "details": [], "length": "same", "motion": []})

    @pytest.mark.parametrize("mode", ["image", "video"])
    def test_keep_all_auto_enhance_is_a_general_rewrite(self, mode):
        args = SimpleNamespace()
        web_server._apply_enhance_submission(args, {"enhance_auto": "true", "enhance_settings": self._KEEP_ALL}, mode=mode, json_caption=False)
        assert args.enhance is not None

    def test_image_submit_threads_settings_and_file_entries(self, monkeypatch, tmp_path):
        submitted: list[dict] = []
        patch_image_submit_dependencies(
            monkeypatch,
            model_info=ImageModelInfo(family="zimage", is_distilled=False, size=None),
            defaults={"steps": 4, "guidance": 1.0, "scheduler": None, "supports_negative_prompt": True},
            submitted=submitted,
        )
        prompt_file = tmp_path / "p.yaml"
        prompt_file.write_text("a:\n  - prompt: x\n    enhance: {style: anime}\n  - prompt: y\n", encoding="utf-8")
        web_config = make_web_config()
        web_config.output_dir = str(tmp_path)

        class _Form(dict):
            def getlist(self, key):
                value = self.get(key)
                return value if isinstance(value, list) else [value]

        web_server._submit_image_job(_Form({"prompt_source": "file", "prompts_file": str(prompt_file), "prompt_option_id": ["a:0", "a:1"]}), web_config)
        assert submitted[0]["enhance_by_set"] == {"a": [EnhanceSettings(style="anime"), None]}
        assert submitted[0]["args"].enhance is None

    def test_enhancer_and_format_reach_args(self):
        args = SimpleNamespace()
        form = {"enhance_auto": "true", "enhance_settings": self._KEEP_ALL_IMAGE, "enhance_model": "smart", "enhance_format": "json"}
        web_server._apply_enhance_submission(args, form, mode="image", json_caption=False)
        assert (args.enhance_model, args.enhance_format) == ("smart", "json")
        assert args.enhance is not None  # A JSON caption is never a no-op.

    @pytest.mark.parametrize(("key", "value"), [("enhance_model", "someone/repo"), ("enhance_format", "yaml")])
    def test_enhancer_and_format_are_allow_listed(self, key, value):
        with pytest.raises(ValueError, match=key):
            web_server._apply_enhance_submission(SimpleNamespace(), {key: value}, mode="image", json_caption=False)

    def test_format_is_image_only(self):
        args = SimpleNamespace()
        web_server._apply_enhance_submission(args, {"enhance_format": "json"}, mode="video", json_caption=False)
        assert args.enhance_format is None

    _KEEP_ALL_IMAGE = json.dumps({"style": "keep", "mood": "keep", "details": [], "length": "same"})


# ── Job lifecycle (web_runner) ──────────────────────────────────────────────


# ── Preflight in web jobs ───────────────────────────────────────────────────

_PREFLIGHT_CONFIG = {
    "sizes": {"2:3": {"m": {"width": 512, "height": 512}}},
    **video_config(),
}


@pytest.fixture(autouse=True)
def _fast_enhancer(monkeypatch, tmp_path):
    """Declare the Fast enhancer as ``a/b``, pinned to the revision ``r``, with the Smart one beside it."""
    use_enhancer_models(monkeypatch, tmp_path, fast="a/b@r", smart="big/m@s", smart_memory_gb=9, fast_size="9 GB", smart_size="19 GB")


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

        def _fake_run_batch(backend, model, prompts_data, config, args, model_info, *, plan, progress_callback, skip_signal, **_kw):
            order.append("run_batch")
            iteration = plan.iterations[0]
            progress_callback({"type": "prompt_started", "mode": "image", "prompt": iteration.prompt, "enhance_status": iteration.enhance_status.value, "run_index": 0})
            progress_callback({"type": "generation_started", "mode": "image"})
            progress_callback({"type": "batch_completed", "mode": "image", "completed_iterations": 1, "total_iterations": 1})

        def _fake_run_video_batch(*, backend, model, model_info, workflow, prompts_data, config, args, plan, progress_callback, skip_signal):
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
            from tests._factories import _make_video_args

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
        return SimpleNamespace(app_config={"prompt_enhancer": section})

    def test_not_downloaded(self):
        contract = workspace_api_module.build_prompt_enhancer_contract(self._config(), downloaded=lambda repo, rev: False)
        assert contract["model"] == "a/b" and contract["revision"] == "r"
        assert contract["downloaded"] is False and contract["download_size_label"] == "9 GB"
        assert contract["matrix"]["defaults"]["length"] == "same"
        assert contract["default_max_words"] == 300

    def test_the_audio_option_matrix_is_served_beside_the_image_one(self):
        contract = workspace_api_module.build_prompt_enhancer_contract(self._config(), downloaded=lambda repo, rev: False)

        assert [axis["key"] for axis in contract["audio_matrix"]["axes"]] == ["genre", "music_mood", "instruments", "without", "singer", "production", "space", "distance", "over_time", "movement"]
        assert contract["audio_matrix"]["defaults"]["genre"] == "keep"
        assert "genre" not in {axis["key"] for axis in contract["matrix"]["axes"]}

    def test_the_audio_option_matrix_is_served_even_without_an_enhancer_model(self):
        contract = workspace_api_module.build_prompt_enhancer_contract(SimpleNamespace(app_config={}), downloaded=lambda repo, rev: False)

        assert contract["audio_matrix"]["axes"]

    def test_downloaded_hides_size(self, tmp_path):
        contract = workspace_api_module.build_prompt_enhancer_contract(self._config(), downloaded=lambda repo, rev: True)
        assert contract["downloaded"] is True and contract["download_size_label"] is None

    def test_custom_model_has_no_default_size_label(self):
        config = self._config(user_model="someone/bigger-llm")
        contract = workspace_api_module.build_prompt_enhancer_contract(config, downloaded=lambda repo, rev: False)
        assert contract["model"] == "someone/bigger-llm" and contract["download_size_label"] is None

    def test_smart_reports_size_and_availability(self):
        contract = workspace_api_module.build_prompt_enhancer_contract(self._config(), downloaded=lambda repo, rev: False, budget_bytes=lambda: 6 * 2**30)
        assert contract["smart"] == {"downloaded": False, "download_size_label": "19 GB", "unavailable": "Needs about 9 GB of memory; this machine has 6 GB."}
        fits = workspace_api_module.build_prompt_enhancer_contract(self._config(), downloaded=lambda repo, rev: True, budget_bytes=lambda: 16 * 2**30)
        assert fits["smart"] == {"downloaded": True, "download_size_label": None, "unavailable": None}

    def test_custom_model_has_no_tiers(self):
        contract = workspace_api_module.build_prompt_enhancer_contract(self._config(user_model="me/mine"), downloaded=lambda repo, rev: False, budget_bytes=lambda: None)
        assert contract["smart"] is None

    def test_a_malformed_user_model_reports_error(self):
        config = SimpleNamespace(app_config={"prompt_enhancer": {"user_model": "a/b@"}})
        contract = workspace_api_module.build_prompt_enhancer_contract(config, downloaded=lambda repo, rev: False)
        assert contract["model"] is None and "Invalid enhancer model" in contract["error"]

    def test_workflows_expose_enhance_controls(self):
        from zvisiongenerator.web.workspace_contract import WORKFLOW_DEFINITIONS

        for name, definition in WORKFLOW_DEFINITIONS.items():
            offered = {"prompt_enhance", "prompt_enhance_auto"} <= set(definition["visible_controls"])
            assert offered is (definition["mode"] != "audio" and name != "edit")  # the Audio tab and Edit have no prompt enhancer


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


# ── /api/prompt/enhance: extending a clip ───────────────────────────────────


_SOURCE_SCENE = "A woman in a red raincoat on a rainy street at night."
_SOURCE_ACTION = 'She looks at the camera and says "I told you".'
_LEGACY_PROMPT = 'A woman in a red raincoat stands on a rainy street at night. She says "I told you".'


class _Clip(SimpleNamespace):
    """A source clip on disk; ``source`` is the :class:`ExtendSource` the server will load for it."""


@pytest.fixture
def clip(monkeypatch, tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00")
    state = _Clip(
        path=path,
        source=make_extend_source(path=path, frame_count=97, scene=_SOURCE_SCENE, prompt=_SOURCE_ACTION),
    )
    extend_api_module._inspect_by_identity.cache_clear()
    monkeypatch.setattr(extend_api_module, "require_ffmpeg", lambda: None)  # the inspection is stubbed; ffmpeg need not be installed
    monkeypatch.setattr(extend_api_module, "inspect_extend_source", lambda _path: ExtendInspection(None, None, (), state.source))
    for module in (web_server, extend_api_module):
        monkeypatch.setattr(module, "supports_video_extend", lambda: True)
        monkeypatch.setattr(module, "supports_video_audio", lambda: True)
    return state


def _extend_body(clip, **overrides) -> dict:
    body = {
        "mode": "video",
        "scene": _SOURCE_SCENE,
        "prompt": "She opens an umbrella.",
        "settings": {"style": "cinematic"},
        "context": {"source_video": str(clip.path), "extend_frames": 48, "audio": True},
    }
    return {**body, **overrides}


class TestExtendEnhance:
    def test_context_comes_from_the_source_clip(self, clip):
        job = web_server._validate_enhance_body(_extend_body(clip))

        assert job["context"] == EnhanceContext(
            audio=True,
            seconds=(25 + 48) / 24,
            orientation="landscape",
            extends=True,
            previous=_SOURCE_ACTION,
            extract_from=None,
            prompt_is_original=False,
            new_seconds=2.0,
        )
        assert (job["scene"], job["prompt"]) == (_SOURCE_SCENE, "She opens an umbrella.")

    def test_context_ignores_facts_the_client_sends_about_the_source(self, clip):
        body = _extend_body(
            clip,
            context={
                "source_video": str(clip.path),
                "extend_frames": 48,
                "previous": "invented",
                "extract_from": "invented",
                "width": 64,
                "height": 640,
                "frames": 5,
                "fps": 3,
                "reference_image": True,
            },
        )

        context = web_server._validate_enhance_body(body)["context"]

        assert (context.previous, context.extract_from, context.orientation, context.reference_image) == (_SOURCE_ACTION, None, "landscape", False)
        assert context.seconds == (25 + 48) / 24

    def test_legacy_source_extracts_its_scene_and_flags_an_unchanged_prompt(self, clip):
        clip.source = dataclasses.replace(clip.source, scene=None, prompt=_LEGACY_PROMPT)

        unchanged = web_server._validate_enhance_body(_extend_body(clip, scene="", prompt='A woman in a red raincoat stands on a  rainy street at night. She says "I told you".'))["context"]
        edited = web_server._validate_enhance_body(_extend_body(clip, scene="", prompt="She walks away."))["context"]
        empty = web_server._validate_enhance_body(_extend_body(clip, scene="", prompt=""))["context"]

        assert (unchanged.extract_from, unchanged.previous, unchanged.prompt_is_original) == (_LEGACY_PROMPT, _LEGACY_PROMPT, True)
        assert (edited.extract_from, edited.prompt_is_original) == (_LEGACY_PROMPT, False)
        assert empty.prompt_is_original is False

    def test_a_scene_bearing_source_never_flags_the_original_prompt(self, clip):
        context = web_server._validate_enhance_body(_extend_body(clip, prompt=_SOURCE_ACTION))["context"]

        assert context.prompt_is_original is False

    def test_empty_scene_and_prompt_are_allowed_when_extending(self, clip):
        job = web_server._validate_enhance_body(_extend_body(clip, scene="", prompt=""))

        assert (job["scene"], job["prompt"]) == ("", "")
        assert job["context"].extends is True

    def test_empty_scene_and_prompt_are_refused_without_a_source(self):
        with pytest.raises(ValueError, match="Enter a scene or a prompt"):
            web_server._validate_enhance_body({"mode": "video", "scene": "", "prompt": "", "settings": {"style": "cinematic"}, "context": {"frames": 97, "fps": 24}})

    def test_settings_that_change_nothing_are_valid_when_extending(self, clip):
        keep_all = {"style": "keep", "mood": "keep", "details": [], "length": "same", "motion": [], "sound": []}

        assert web_server._validate_enhance_body(_extend_body(clip, settings=keep_all))["settings"].style == "keep"

    def test_audio_is_off_for_a_silent_source_and_when_not_requested_or_supported(self, clip, monkeypatch):
        silent = dataclasses.replace(clip.source, has_audio=False)
        requested = _extend_body(clip)
        not_requested = _extend_body(clip, context={"source_video": str(clip.path), "extend_frames": 48})

        assert web_server._validate_enhance_body(requested)["context"].audio is True
        assert web_server._validate_enhance_body(not_requested)["context"].audio is False
        clip.source = silent
        extend_api_module._inspect_by_identity.cache_clear()  # the stub's source changed without the file changing
        assert web_server._validate_enhance_body(requested)["context"].audio is False
        clip.source = dataclasses.replace(silent, has_audio=True)
        extend_api_module._inspect_by_identity.cache_clear()
        monkeypatch.setattr(extend_api_module, "supports_video_audio", lambda: False)
        assert web_server._validate_enhance_body(requested)["context"].audio is False

    def test_length_defaults_to_two_seconds_and_accepts_seconds(self, clip):
        default = web_server._validate_enhance_body(_extend_body(clip, context={"source_video": str(clip.path)}))["context"]
        seconds = web_server._validate_enhance_body(_extend_body(clip, context={"source_video": str(clip.path), "extend_seconds": 1}))["context"]

        assert default.new_seconds == 2.0 and default.seconds == (25 + 48) / 24
        assert seconds.new_seconds == 1.0 and seconds.seconds == (25 + 24) / 24

    @pytest.mark.parametrize(
        ("context", "message"),
        [
            ({"extend_frames": 12}, "extend_frames must be a multiple of 8"),
            ({"extend_frames": True}, "context.extend_frames must be an integer"),
            ({"extend_seconds": 0}, "extend_seconds must be positive"),
            ({"source_video": "https://example.com/a.mp4"}, "host-local"),
            ({"source_video": "clip.png"}, "video file"),
        ],
    )
    def test_invalid_extend_context_is_rejected(self, clip, context, message):
        body = _extend_body(clip, context={"source_video": str(clip.path), **context})

        with pytest.raises(ValueError, match=message):
            web_server._validate_enhance_body(body)

    def test_a_source_that_cannot_be_extended_is_rejected_with_the_problems(self, clip, monkeypatch):
        problems = (ExtendProblem("interlaced", "The video is interlaced."),)
        monkeypatch.setattr(extend_api_module, "inspect_extend_source", lambda _path: ExtendInspection(None, None, problems, None))

        with pytest.raises(ValueError, match="interlaced"):
            web_server._validate_enhance_body(_extend_body(clip))

    def test_extending_is_refused_without_the_capability(self, clip, monkeypatch):
        monkeypatch.setattr(extend_api_module, "supports_video_extend", lambda: False)

        with pytest.raises(ValueError, match="not available"):
            web_server._validate_enhance_body(_extend_body(clip))

    def test_the_source_is_ignored_for_images(self, clip):
        job = web_server._validate_enhance_body({"mode": "image", "prompt": "a fox", "settings": {"style": "photo"}, "context": {"source_video": str(clip.path), "frames": 97, "fps": 24}})

        assert job["context"].extends is False

    def test_route_streams_the_action_only_for_a_kept_scene(self, monkeypatch, no_active_job, clip):
        enhancer = _FakeEnhancer(["She opens a black umbrella and steps away from the light."])
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: _FakeSession(enhancer))

        with TestClient(web_server.app) as client:
            frames = _frames(client.post("/api/prompt/enhance", json=_extend_body(clip)))

        texts = [frame for frame in frames if frame["type"] == "text"]
        assert texts and all(frame["scene"] == "" and frame["prompt"] == frame["text"] for frame in texts)
        assert _without_model(frames[-1]) == {"type": "done", "scene": "", "prompt": "She opens a black umbrella and steps away from the light.", "clamped": False}
        system, user = enhancer.messages[0][0]["content"], enhancer.messages[0][1]["content"]
        assert _SOURCE_SCENE in system and _SOURCE_ACTION in system
        assert user == "She opens an umbrella."

    def test_route_rejects_a_bad_source_before_loading_the_enhancer(self, monkeypatch, no_active_job, clip, tmp_path):
        session = _FakeSession(_FakeEnhancer([]))
        monkeypatch.setattr(web_server, "get_prompt_enhancer_session", lambda: session)

        with TestClient(web_server.app) as client:
            response = client.post("/api/prompt/enhance", json=_extend_body(clip, context={"source_video": str(tmp_path / "gone.mp4")}))

        assert response.status_code == 422
        assert "Source video not found" in response.json()["detail"]
        assert session.acquired == []
