"""The audio track in the Web UI: the upload field, the Browse purpose and the enhancer context."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

import zvisiongenerator.utils.audio_track as audio_track_module
from tests.web._helpers import stub_video_submission
from zvisiongenerator.web import server as web_server


def _generate(monkeypatch, tmp_path, *, filename: str, extra: dict[str, str] | None = None):
    submitted = stub_video_submission(monkeypatch, tmp_path)
    monkeypatch.setattr(audio_track_module, "probe_track_seconds", lambda _path: 30.0)
    data = {"mode": "video", "model": "ltx-8", "prompt": "a singer", **(extra or {})}
    with TestClient(web_server.app) as client:
        response = client.post("/api/generate", data=data, files={"audio_track_file": (filename, b"not really audio", "application/octet-stream")})
    return response, submitted


def test_an_m4a_upload_becomes_the_clips_track_and_is_recorded_for_retry(monkeypatch, tmp_path):
    response, submitted = _generate(monkeypatch, tmp_path, filename="memo.m4a", extra={"audio_track_start": "1.5", "audio": "off"})

    assert response.status_code == 200
    (job,) = submitted
    saved = job["args"].audio_track.path
    assert Path(saved).name == "memo.m4a" and (tmp_path / ".web_uploads") in Path(saved).parents
    assert job["args"].audio_track.start == 1.5
    assert job["request"].audio_track == job["args"].audio_track
    assert job["context"]["settings"]["audio_track_path"] == saved
    assert (job["args"].no_audio, job["request"].no_audio) == (False, False)


def test_an_upload_wins_over_a_typed_path(monkeypatch, tmp_path):
    response, submitted = _generate(monkeypatch, tmp_path, filename="memo.wav", extra={"audio_track_path": "/typed/other.mp3"})

    assert response.status_code == 200
    assert Path(submitted[0]["args"].audio_track.path).name == "memo.wav"


@pytest.mark.parametrize("filename", ["notes.txt", "clip.mp4"])
def test_a_text_or_video_file_is_refused_as_a_track(monkeypatch, tmp_path, filename):
    response, submitted = _generate(monkeypatch, tmp_path, filename=filename)

    assert response.status_code == 422
    assert submitted == []
    assert not (tmp_path / ".web_uploads").exists()


def test_an_image_upload_is_still_checked_as_an_image(tmp_path):
    upload = UploadFile(file=io.BytesIO(b"not an image"), filename="memo.m4a")

    with pytest.raises(ValueError, match="Image must be"):
        web_server._save_upload(upload, str(tmp_path))


def test_an_effective_track_leaves_no_generated_sound_in_the_enhance_context():
    with_audio = web_server._enhance_context_from_body({"audio": True}, mode="video")
    with_track = web_server._enhance_context_from_body({"audio": True, "audio_track": True}, mode="video")

    assert with_audio.audio is True
    assert with_track.audio is False


def test_an_audio_upload_keeps_its_name_under_the_hash_of_its_content(tmp_path):
    upload = UploadFile(file=io.BytesIO(b"abc"), filename="My Song (live).MP3")

    saved = Path(web_server._save_upload(upload, str(tmp_path), audio=True))

    assert saved.name == "My Song (live).mp3"
    assert saved.parent == tmp_path / ".web_uploads" / hashlib.sha256(b"abc").hexdigest()
    assert saved.read_bytes() == b"abc"


def test_the_same_audio_is_saved_once_and_other_audio_apart(tmp_path):
    first = web_server._save_upload(UploadFile(file=io.BytesIO(b"abc"), filename="a.wav"), str(tmp_path), audio=True)
    again = web_server._save_upload(UploadFile(file=io.BytesIO(b"abc"), filename="a.wav"), str(tmp_path), audio=True)
    other = web_server._save_upload(UploadFile(file=io.BytesIO(b"abd"), filename="a.wav"), str(tmp_path), audio=True)

    assert first == again != other


@pytest.mark.parametrize("filename", ["../../etc/passwd.wav", "dir/..\\x.wav"])
def test_an_audio_name_cannot_leave_its_folder(tmp_path, filename):
    saved = Path(web_server._save_upload(UploadFile(file=io.BytesIO(b"abc"), filename=filename), str(tmp_path), audio=True))

    assert saved.parent == tmp_path / ".web_uploads" / hashlib.sha256(b"abc").hexdigest()
    assert saved.suffix == ".wav" and saved.is_file()


class TestInspectRoute:
    def _post(self, body):
        with TestClient(web_server.app) as client:
            return client.post("/api/audio-track/inspect", json=body)

    def test_reports_the_length_and_the_display_name(self, monkeypatch, tmp_path):
        monkeypatch.setattr(web_server, "probe_track_seconds", lambda path: 183.25)
        song = tmp_path / "song.mp3"
        song.write_bytes(b"x")

        response = self._post({"path": str(song)})

        assert response.status_code == 200
        assert response.json() == {"seconds": 183.25, "name": "song.mp3"}

    def test_a_typed_path_is_read_as_the_generate_form_reads_it(self, monkeypatch):
        seen = []
        monkeypatch.setattr(web_server, "probe_track_seconds", lambda path: seen.append(path) or 5.0)

        response = self._post({"path": "~/song.mp3"})

        assert response.json()["name"] == "song.mp3"
        assert seen == [str(Path("~/song.mp3").expanduser())]

    def test_a_file_that_cannot_be_read_is_refused_with_the_track_message(self, tmp_path):
        response = self._post({"path": str(tmp_path / "gone.mp3")})

        assert response.status_code == 422
        assert response.json()["detail"] == "Can't read the audio track"

    @pytest.mark.parametrize("body", [{}, {"path": 3}, {"path": "https://example.com/a.mp3"}])
    def test_a_bad_body_or_a_url_is_refused(self, body):
        assert self._post(body).status_code == 422
