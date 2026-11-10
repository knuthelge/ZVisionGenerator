"""Tests for the route that saves one frame of a video as a PNG under the output folder's uploads; ffprobe and ffmpeg are mocked."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest

from tests._fake_media import completed
from zvisiongenerator.utils import video_frame
from zvisiongenerator.web import server as web_server
from zvisiongenerator.web import video_frame_api
from zvisiongenerator.web.gallery import list_gallery_assets
from zvisiongenerator.web.video_frame_api import frame_response

PNG = b"\x89PNG\r\n\x1a\nfull size frame"
PROBE = {
    "streams": [{"codec_type": "video", "width": 512, "height": 256, "r_frame_rate": "25/1", "avg_frame_rate": "25/1", "nb_frames": "100"}],
    "format": {"duration": "4.0"},
}


@pytest.fixture
def output_root(tmp_path, monkeypatch) -> Path:
    folder = tmp_path / "output"
    folder.mkdir()
    monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(web_server, "_media_output_root", lambda: folder)
    return folder


@pytest.fixture
def ffmpeg(monkeypatch) -> MagicMock:
    """Mock ffprobe, the ffmpeg check and the ffmpeg call; returns the ``subprocess.run`` mock."""
    monkeypatch.setattr(video_frame, "ffprobe_json", MagicMock(return_value=PROBE))
    monkeypatch.setattr(video_frame_api, "require_ffmpeg", MagicMock())
    run = MagicMock(return_value=completed(PNG))
    monkeypatch.setattr(subprocess, "run", run)
    return run


@pytest.fixture
def client(output_root):
    with TestClient(web_server.app) as test_client:
        yield test_client


@pytest.fixture
def clip(tmp_path) -> Path:
    path = tmp_path / "videos" / "Red Fox.mp4"
    path.parent.mkdir()
    path.write_bytes(b"video")
    return path


def _post(client: TestClient, **body):
    return client.post("/api/video/frame", json=body)


def test_the_frame_is_saved_as_a_png_named_after_the_clip_and_time(client, ffmpeg, output_root, clip):
    response = _post(client, path=str(clip), time=3.21)

    assert response.status_code == 200
    saved = output_root / ".web_uploads" / hashlib.sha256(PNG).hexdigest() / "Red Fox_3.2s.png"
    assert response.json() == {"path": str(saved)}
    assert saved.read_bytes() == PNG


def test_the_time_is_snapped_to_a_frame_before_ffmpeg_seeks(client, ffmpeg, clip):
    _post(client, path=str(clip), time=3.21)

    command = ffmpeg.call_args.args[0]
    assert command[command.index("-ss") + 1] == "3.180000"
    assert "-vf" in command and "scale=in_color_matrix=bt709,format=rgb24" in command


def test_the_same_frame_asked_twice_is_one_file(client, ffmpeg, output_root, clip):
    first = _post(client, path=str(clip), time=1.0).json()
    second = _post(client, path=str(clip), time=1.0).json()

    assert first == second
    assert len(list((output_root / ".web_uploads").rglob("*.png"))) == 1


def test_a_time_past_the_end_gives_the_last_frame(client, ffmpeg, clip):
    _post(client, path=str(clip), time=500)

    command = ffmpeg.call_args.args[0]
    assert command[command.index("-ss") + 1] == "3.940000"


def test_the_saved_frame_is_not_a_gallery_asset(client, ffmpeg, output_root, clip):
    _post(client, path=str(clip), time=1.0)

    assert list_gallery_assets(str(output_root)) == []


def test_a_missing_clip_is_not_found(client, ffmpeg, tmp_path):
    response = _post(client, path=str(tmp_path / "gone.mp4"), time=1.0)

    assert response.status_code == 404
    ffmpeg.assert_not_called()


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"time": 1.0},
        {"path": 5, "time": 1.0},
        {"path": "https://example.com/clip.mp4", "time": 1.0},
        {"path": "notes.txt", "time": 1.0},
        {"path": "CLIP", "time": -1},
        {"path": "CLIP", "time": "1.0"},
        {"path": "CLIP", "time": True},
        {"path": "CLIP"},
        {"path": "CLIP", "time": None},
    ],
)
def test_a_bad_path_or_time_is_refused(client, ffmpeg, clip, body):
    body = {key: (str(clip) if value == "CLIP" else value) for key, value in body.items()}

    assert _post(client, **body).status_code == 422
    ffmpeg.assert_not_called()


def test_a_body_that_is_not_json_is_refused(client, ffmpeg):
    assert client.post("/api/video/frame", content=b"not json", headers={"content-type": "application/json"}).status_code == 422


@pytest.mark.parametrize("time", ["NaN", "Infinity"])
def test_a_time_that_is_not_finite_is_refused(client, ffmpeg, clip, time):
    body = f'{{"path": "{clip}", "time": {time}}}'

    assert client.post("/api/video/frame", content=body, headers={"content-type": "application/json"}).status_code == 422


def test_a_clip_whose_rate_is_unknown_is_refused(client, ffmpeg, clip):
    video_frame.ffprobe_json.return_value = {"streams": [{"codec_type": "video", "width": 64, "height": 64}], "format": {}}

    response = _post(client, path=str(clip), time=1.0)

    assert response.status_code == 422
    assert "frame rate" in response.json()["detail"]


def test_ffmpeg_failing_is_a_500_with_a_detail_to_show(client, ffmpeg, output_root, clip):
    ffmpeg.return_value = completed(b"", stderr=b"Invalid data found when processing input", returncode=1)

    response = _post(client, path=str(clip), time=1.0)

    assert response.status_code == 500
    assert "Invalid data found" in response.json()["detail"]
    assert not (output_root / ".web_uploads").exists()


def test_ffmpeg_missing_is_refused_with_its_message(client, ffmpeg, clip):
    video_frame_api.require_ffmpeg.side_effect = RuntimeError("ffmpeg is required but was not found on PATH.")

    response = _post(client, path=str(clip), time=1.0)

    assert response.status_code == 422
    assert "ffmpeg is required" in response.json()["detail"]


def test_the_response_is_built_without_the_server(ffmpeg, tmp_path, clip):
    answer = frame_response({"path": str(clip), "time": 0}, tmp_path / "out")

    assert Path(answer["path"]).parent.parent == tmp_path / "out" / ".web_uploads"
