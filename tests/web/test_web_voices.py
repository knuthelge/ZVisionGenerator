"""Test the saved-voices routes: list, play, save from a recording, rename and delete."""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests._audio_presets import audio_config
from tests._speech_fakes import RATE, tone
from tests.web._helpers import make_web_config

from zvisiongenerator.utils import audio_voice
from zvisiongenerator.utils import voice_library as library
from zvisiongenerator.utils.voice_library import list_voices, save_voice, voices_dir
from zvisiongenerator.web import audio_models as audio_models_module
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import voices as voices_module

_TEXT = "Hello, this is my voice."


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Serve the Web UI with the fixture audio presets on a CUDA machine; recordings read as a tone whose length the test sets in ``env.seconds``."""
    web_config = make_web_config()
    web_config.output_dir = str(tmp_path / "out")
    web_config.app_config.update(audio_config())
    state = SimpleNamespace(seconds=5.0, reads=[], ffmpeg=True)

    def probe_track_seconds(path):
        state.reads.append(str(path))
        return state.seconds

    def read_track(_track, sample_rate, seconds):
        return np.zeros((2, round(seconds * sample_rate)), dtype=np.float32)

    def require_ffmpeg():
        if not state.ffmpeg:
            raise RuntimeError("ffmpeg was not found.")

    def unreadable_tags(_path):
        raise ValueError("no tags")

    monkeypatch.setattr(web_server, "load_web_config", lambda: web_config)
    monkeypatch.setattr(web_server, "require_ffmpeg", require_ffmpeg)
    monkeypatch.setattr(library, "probe_track_seconds", probe_track_seconds)
    monkeypatch.setattr(library, "read_track", read_track)
    monkeypatch.setattr(audio_voice, "read_config", unreadable_tags)
    monkeypatch.setattr(web_server.web_runner, "get_active_exclusive_job_snapshot", lambda: None)
    monkeypatch.setattr(audio_models_module, "audio_platform_key", lambda: "cuda")
    recording = tmp_path / "me.wav"
    recording.write_bytes(b"not decoded: the decoder is replaced")
    with TestClient(web_server.app) as client:
        state.client = client
        state.recording = recording
        state.root = tmp_path
        state.post = lambda fields=None, **kwargs: client.post("/api/voices", data={"model": "qwen3_tts", "name": "Anna", "voice_text": _TEXT, "voice_consent": "on", **(fields or {})}, **kwargs)
        yield state


def _upload(name: str = "take.wav") -> dict:
    return {"files": {"voice_file": (name, io.BytesIO(b"bytes"), "audio/wav")}}


def _leftovers() -> list[str]:
    """Return the temporary files left in the voices folder."""
    return sorted(path.name for path in voices_dir().glob(".*")) if voices_dir().is_dir() else []


def _seed(name: str = "Anna", **kwargs) -> str:
    return save_voice(name, tone(5.0), RATE, text=kwargs.get("text", _TEXT), consent="given").slug


class TestList:
    def test_an_empty_library_lists_nothing(self, env):
        response = env.client.get("/api/voices")

        assert response.status_code == 200
        assert response.json() == {"voices": []}

    def test_voices_are_listed_by_name_with_the_public_fields_only(self, env):
        _seed("Zed")
        anna = _seed("anna")

        voices = env.client.get("/api/voices").json()["voices"]

        assert [voice["name"] for voice in voices] == ["anna", "Zed"]
        assert set(voices[0]) == {"slug", "name", "text", "origin", "seconds", "description", "created_at"}
        assert (voices[0]["slug"], voices[0]["text"], voices[0]["origin"], voices[0]["seconds"], voices[0]["description"]) == (anna, _TEXT, "recording", 5.0, None)


class TestAudio:
    def test_the_recording_of_a_voice_is_served_as_wav(self, env):
        slug = _seed()

        response = env.client.get(f"/api/voices/{slug}/audio")

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.content == (voices_dir() / f"{slug}.wav").read_bytes()

    def test_an_unknown_voice_is_404(self, env):
        assert env.client.get("/api/voices/nobody/audio").status_code == 404

    @pytest.mark.parametrize("slug", ["..%2F..%2Fetc%2Fpasswd", "%2E%2E", ".hidden", "UPPER"])
    def test_only_a_valid_slug_of_the_library_is_served(self, env, slug):
        _seed()

        assert env.client.get(f"/api/voices/{slug}/audio").status_code in (404, 422)

    def test_a_record_without_its_recording_is_not_served(self, env):
        slug = _seed()
        (voices_dir() / f"{slug}.wav").unlink()

        assert env.client.get(f"/api/voices/{slug}/audio").status_code == 404


class TestSave:
    def test_a_typed_path_is_saved_as_a_recording_voice(self, env):
        response = env.post({"voice_path": str(env.recording)})

        assert response.status_code == 200, response.text
        body = response.json()
        assert (body["name"], body["text"], body["origin"], body["seconds"]) == ("Anna", _TEXT, "recording", 5.0)
        assert [voice.slug for voice in list_voices()] == [body["slug"]]
        assert list_voices()[0].consent == "given"
        assert env.reads == [str(env.recording)]

    def test_an_upload_is_saved_and_its_temporary_copy_is_removed(self, env):
        response = env.post(**_upload())

        assert response.status_code == 200, response.text
        assert [voice.name for voice in list_voices()] == ["Anna"]
        read_path = env.reads[0]
        assert read_path.startswith(str(voices_dir())) and read_path.endswith(".wav")
        assert _leftovers() == []

    def test_an_upload_wins_over_a_typed_path(self, env):
        env.post({"voice_path": str(env.recording)}, **_upload("take.flac"))

        assert env.reads[0].endswith(".flac")

    def test_the_temporary_copy_is_removed_when_the_save_fails(self, env):
        env.seconds = 40.0

        response = env.post(**_upload())

        assert response.status_code == 422
        assert list_voices() == []
        assert _leftovers() == []

    def test_a_file_that_is_not_audio_is_refused_and_nothing_is_kept(self, env):
        response = env.post(**_upload("notes.txt"))

        assert response.status_code == 422
        assert _leftovers() == []

    def test_a_taken_name_is_409_ignoring_case(self, env):
        _seed("Anna")

        response = env.post({"name": "ANNA", "voice_path": str(env.recording)})

        assert response.status_code == 409
        assert [voice.name for voice in list_voices()] == ["Anna"]

    @pytest.mark.parametrize(
        "fields",
        [
            {"voice_text": "   "},
            {"name": "  "},
            {"name": "x" * 61},
            {"voice_path": ""},
            {"voice_path": "https://example.com/a.wav"},
            {"voice_path": "/no/such/file.wav"},
            {"model": "sa3"},
            {"model": "nope"},
            {"model": ""},
        ],
    )
    def test_an_invalid_form_is_422_with_a_reason_and_saves_nothing(self, env, fields):
        fields = {"voice_path": str(env.recording), **fields}

        response = env.post(fields)

        assert response.status_code == 422, response.text
        assert response.json()["detail"]
        assert list_voices() == []

    def test_a_recording_without_the_consent_tick_is_refused(self, env):
        response = env.post({"voice_path": str(env.recording), "voice_consent": ""})

        assert response.status_code == 422
        assert response.json()["detail"] == audio_voice.CONSENT_REQUIRED
        assert list_voices() == []

    def test_a_built_in_voice_take_needs_no_tick(self, env, monkeypatch):
        monkeypatch.setattr(audio_voice, "read_config", lambda _path: {"kind": "speech"})

        response = env.post({"voice_path": str(env.recording), "voice_consent": ""})

        assert response.status_code == 200
        assert list_voices()[0].consent == "not_needed"

    @pytest.mark.parametrize("seconds", [1.0, 40.0])
    def test_a_recording_outside_the_models_length_is_422(self, env, seconds):
        env.seconds = seconds

        response = env.post({"voice_path": str(env.recording)})

        assert response.status_code == 422
        assert list_voices() == []

    def test_a_recording_that_cannot_be_read_is_422(self, env, monkeypatch):
        def unreadable(path, *, min_seconds, max_seconds):
            raise ValueError("The file could not be read as audio.")

        monkeypatch.setattr(voices_module, "read_recording", unreadable)

        response = env.post({"voice_path": str(env.recording)})

        assert response.status_code == 422

    def test_a_missing_ffmpeg_is_422_and_nothing_is_read(self, env):
        env.ffmpeg = False

        response = env.post({"voice_path": str(env.recording)})

        assert response.status_code == 422
        assert env.reads == []

    def test_an_unwritable_library_is_500(self, env, monkeypatch):
        def full(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(voices_module, "save_voice", full)

        response = env.post({"voice_path": str(env.recording)})

        assert response.status_code == 500
        assert "disk full" in response.json()["detail"]


class TestRename:
    def test_a_rename_changes_the_name_and_keeps_the_slug_and_files(self, env):
        slug = _seed("Anna")
        before = sorted(path.name for path in voices_dir().iterdir())

        response = env.client.post(f"/api/voices/{slug}/rename", json={"name": "Annabelle"})

        assert response.status_code == 200
        assert (response.json()["slug"], response.json()["name"]) == (slug, "Annabelle")
        assert sorted(path.name for path in voices_dir().iterdir()) == before
        assert [voice.name for voice in list_voices()] == ["Annabelle"]

    def test_an_unknown_voice_is_404(self, env):
        assert env.client.post("/api/voices/nobody/rename", json={"name": "Bea"}).status_code == 404

    def test_a_name_another_voice_has_is_409(self, env):
        slug = _seed("Anna")
        _seed("Bea")

        response = env.client.post(f"/api/voices/{slug}/rename", json={"name": "bea"})

        assert response.status_code == 409
        assert [voice.name for voice in list_voices()] == ["Anna", "Bea"]

    @pytest.mark.parametrize("body", [{"name": "  "}, {"name": 5}, {}, {"name": "x" * 61}])
    def test_an_invalid_name_is_422(self, env, body):
        slug = _seed("Anna")

        assert env.client.post(f"/api/voices/{slug}/rename", json=body).status_code == 422


class TestDelete:
    def test_a_delete_removes_the_record_and_the_recording(self, env):
        slug = _seed()

        response = env.client.delete(f"/api/voices/{slug}")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert list_voices() == []
        assert sorted(path.name for path in voices_dir().iterdir()) == []

    def test_an_unknown_voice_is_404(self, env):
        assert env.client.delete("/api/voices/nobody").status_code == 404

    def test_a_delete_is_refused_while_a_job_runs_and_the_voice_stays(self, env, monkeypatch):
        slug = _seed()
        monkeypatch.setattr(web_server.web_runner, "get_active_exclusive_job_snapshot", lambda: {"job_id": "j1"})

        response = env.client.delete(f"/api/voices/{slug}")

        assert response.status_code == 409
        assert [voice.slug for voice in list_voices()] == [slug]


def test_no_request_writes_under_the_output_folder(env):
    slug = _seed("Anna")
    env.post({"name": "Bea", "voice_path": str(env.recording)})
    env.post({"name": "Cy"}, **_upload())
    env.client.get("/api/voices")
    env.client.get(f"/api/voices/{slug}/audio")
    env.client.post(f"/api/voices/{slug}/rename", json={"name": "Anna B"})
    env.client.delete(f"/api/voices/{slug}")

    assert not (env.root / "out").exists()


def test_saving_a_voice_from_a_take_in_a_project_copies_it_into_the_library_and_leaves_the_project_alone(env):
    project = env.root / "out" / "Night Bus"
    project.mkdir(parents=True)
    take = project / "speech_001.wav"
    take.write_bytes(b"not decoded: the decoder is replaced")

    response = env.post({"voice_path": str(take)})

    assert response.status_code == 200, response.text
    assert [path.name for path in project.iterdir()] == ["speech_001.wav"]
    assert [path.name for path in (env.root / "out").iterdir()] == ["Night Bus"]
    (voice,) = list_voices()
    assert voice.path.parent == voices_dir() and not voices_dir().is_relative_to(env.root / "out")
