"""Tests for the file browser's preview, info and media routes; ffmpeg, ffprobe and the decode queue are mocked."""

from __future__ import annotations

import io
import os
from pathlib import Path
import subprocess
import threading
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin
import pytest

from tests._images import SIDEWAYS_UPRIGHT_SIZE, write_sideways_photo
from zvisiongenerator.utils import video_frame
from zvisiongenerator.utils.lora_preview import LARGE_PREVIEW_EDGE, PREVIEW_EDGE
from zvisiongenerator.web import browse_media
from zvisiongenerator.web import server as web_server

VIDEO_PROBE = {
    "streams": [{"codec_type": "video", "width": 1920, "height": 1080, "r_frame_rate": "25/1", "avg_frame_rate": "25/1", "nb_frames": "500"}],
    "format": {"duration": "20.0"},
}
AUDIO_PROBE = {"streams": [{"codec_type": "audio", "sample_rate": "44100", "channels": 2}], "format": {"duration": "12.5"}}
PEAKS = list(range(100))


def _png(size: tuple[int, int] = (1000, 500)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, (30, 120, 200)).save(out, format="PNG")
    return out.getvalue()


def _write_labelled_png(path: Path) -> None:
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "prompt: a private secret")
    Image.new("RGB", (1000, 500), (30, 120, 200)).save(path, pnginfo=info)


@pytest.fixture
def data_dir(tmp_path, monkeypatch) -> Path:
    folder = tmp_path / "data"
    monkeypatch.setenv("ZIV_DATA_DIR", str(folder))
    return folder


@pytest.fixture
def cache_dir(data_dir) -> Path:
    return data_dir / "cache" / "browse"


@pytest.fixture
def media_dir(tmp_path) -> Path:
    folder = tmp_path / "media"
    folder.mkdir()
    return folder


@pytest.fixture
def client(data_dir):
    with TestClient(web_server.app) as test_client:
        yield test_client


@pytest.fixture
def tools(monkeypatch) -> MagicMock:
    """Mock ffprobe, the audio decode and ffmpeg; returns the ffmpeg ``subprocess.run`` mock."""
    probe = MagicMock(return_value=VIDEO_PROBE)
    monkeypatch.setattr(browse_media, "ffprobe_json", probe)
    monkeypatch.setattr(video_frame, "ffprobe_json", probe)
    monkeypatch.setattr(browse_media, "read_waveform", MagicMock(return_value=(PEAKS, 12.5)))
    run = MagicMock(return_value=subprocess.CompletedProcess([], 0, stdout=_png((640, 360)), stderr=b""))
    monkeypatch.setattr(subprocess, "run", run)
    return run


def _file(folder: Path, name: str, data: bytes = b"data") -> Path:
    path = folder / name
    path.write_bytes(data)
    return path


def _get(client: TestClient, route: str, purpose: str, path: Path | str, **extra):
    return client.get(f"/api/browse/{route}", params={"purpose": purpose, "path": str(path), **extra})


def _webp(response) -> Image.Image:
    return Image.open(io.BytesIO(response.content))


# ── Which purposes and files ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("route", "purpose"),
    [
        ("preview", "audio_track"),
        ("preview", "lora_file"),
        ("preview", "output_directory"),
        ("info", "lora_file"),
        ("info", "prompt_file"),
        ("media", "checkpoint_file"),
        ("media", "output_directory"),
        ("preview", "no_such_purpose"),
    ],
)
def test_a_purpose_the_route_does_not_serve_is_refused(client, media_dir, tools, route, purpose):
    path = _file(media_dir, "clip.safetensors")

    assert _get(client, route, purpose, path).status_code == 422
    tools.assert_not_called()


def test_a_relative_path_is_refused(client, tools):
    assert _get(client, "preview", "reference_image", "pictures/fox.png").status_code == 422


def test_files_the_purpose_does_not_accept_folders_and_missing_files_are_not_found(client, media_dir, tools):
    (media_dir / "folder.png").mkdir()
    cases = [_file(media_dir, "notes.txt"), media_dir / "folder.png", media_dir / "missing.png"]

    assert [_get(client, "preview", "reference_image", case).status_code for case in cases] == [404, 404, 404]


def test_a_link_with_an_accepted_name_to_another_kind_of_file_is_refused(client, media_dir, tools):
    secret = _file(media_dir, "id_rsa.pem", b"-----BEGIN PRIVATE KEY-----")
    (media_dir / "photo.png").symlink_to(secret)
    (media_dir / "folder.png").symlink_to(media_dir, target_is_directory=True)

    for name in ("photo.png", "folder.png"):
        assert _get(client, "preview", "reference_image", media_dir / name).status_code == 404
        assert _get(client, "info", "reference_image", media_dir / name).status_code == 404


def test_a_link_to_a_file_of_the_accepted_kind_is_served_from_where_it_points(client, media_dir, tools):
    real = media_dir / "real.png"
    real.write_bytes(_png())
    (media_dir / "link.png").symlink_to(real)

    assert _get(client, "preview", "reference_image", media_dir / "link.png").status_code == 200


# ── Preview ──────────────────────────────────────────────────────────────────


def test_an_image_preview_is_a_small_webp_without_metadata(client, media_dir):
    path = media_dir / "labelled.png"
    _write_labelled_png(path)

    response = _get(client, "preview", "reference_image", path)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"
    image = _webp(response)
    assert max(image.size) == PREVIEW_EDGE
    assert not image.getexif()
    assert "private secret" not in str(image.info) and b"private secret" not in response.content


def test_a_photo_preview_is_upright(client, media_dir):
    path = Path(write_sideways_photo(media_dir / "phone.jpg"))

    assert _webp(_get(client, "preview", "reference_image", path)).size == SIDEWAYS_UPRIGHT_SIZE


def test_a_picture_with_too_many_pixels_is_refused(client, media_dir, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    path = _file(media_dir, "huge.png", _png((50, 50)))

    assert _get(client, "preview", "reference_image", path).status_code == 404


def test_a_video_preview_is_the_frame_ten_percent_in(client, media_dir, tools):
    path = _file(media_dir, "clip.mp4")

    response = _get(client, "preview", "source_video", path)

    assert response.status_code == 200
    assert _webp(response).size == (384, 216)
    browse_media.ffprobe_json.assert_called_once()
    assert browse_media.ffprobe_json.call_args.kwargs == {"count_packets": False, "timeout": browse_media.TOOL_TIMEOUT}
    command = tools.call_args.args[0]
    assert tools.call_count == 1
    assert command[command.index("-ss") + 1] == "1.980000"
    assert command[command.index("-vf") + 1] == _UNTAGGED_STILL
    assert tools.call_args.kwargs["timeout"] == browse_media.TOOL_TIMEOUT


_STILL_SCALE = f"scale={PREVIEW_EDGE}:{PREVIEW_EDGE}:force_original_aspect_ratio=decrease"
# A clip that names no colour matrix is converted as BT.709 first, as a player shows it and as a saved frame is read.
_UNTAGGED_STILL = f"{video_frame.frame_color_filter(video_frame.parse_probe(VIDEO_PROBE))},{_STILL_SCALE}"


def test_the_last_frame_is_the_clips_last_frame(client, media_dir, tools):
    path = _file(media_dir, "clip.mp4")

    response = _get(client, "preview", "source_video", path, frame="last")

    assert response.status_code == 200
    assert _webp(response).size == (384, 216)
    assert tools.call_count == 1
    command = tools.call_args.args[0]
    assert command[command.index("-ss") + 1] == "19.940000"
    assert command[command.index("-vf") + 1] == _UNTAGGED_STILL
    assert tools.call_args.kwargs["timeout"] == browse_media.TOOL_TIMEOUT
    browse_media.ffprobe_json.assert_called_once()


def test_every_still_of_an_untagged_clip_is_converted_the_same_way(client, media_dir, tools):
    path = _file(media_dir, "clip.mp4")

    for extra in ({}, {"frame": "last"}, {"at": "2.0"}):
        _get(client, "preview", "source_video", path, **extra)

    filters = [call.args[0][call.args[0].index("-vf") + 1] for call in tools.call_args_list]
    assert filters == [_UNTAGGED_STILL] * 3


def test_a_tagged_clip_keeps_its_own_colour_conversion(client, media_dir, tools):
    stream = {**VIDEO_PROBE["streams"][0], "color_space": "bt709"}
    browse_media.ffprobe_json.return_value = {**VIDEO_PROBE, "streams": [stream]}
    path = _file(media_dir, "clip.mp4")

    for extra in ({}, {"frame": "last"}, {"at": "2.0"}):
        _get(client, "preview", "source_video", path, **extra)

    filters = [call.args[0][call.args[0].index("-vf") + 1] for call in tools.call_args_list]
    assert filters == [_STILL_SCALE] * 3


def test_the_last_frame_of_an_image_is_refused(client, media_dir, tools):
    path = _file(media_dir, "fox.png", _png())

    assert _get(client, "preview", "reference_image", path, frame="last").status_code == 422


def test_an_unknown_frame_is_refused(client, media_dir, tools):
    assert _get(client, "preview", "source_video", _file(media_dir, "clip.mp4"), frame="middle").status_code == 422


def test_a_large_preview_is_drawn_at_the_large_edge(client, media_dir):
    path = _file(media_dir, "wide.png", _png((2000, 1000)))

    assert max(_webp(_get(client, "preview", "reference_image", path, size="large")).size) == LARGE_PREVIEW_EDGE
    assert max(_webp(_get(client, "preview", "reference_image", path)).size) == PREVIEW_EDGE


def test_a_large_video_still_is_scaled_by_ffmpeg_to_the_large_edge(client, media_dir, tools):
    path = _file(media_dir, "clip.mp4")

    _get(client, "preview", "source_video", path, size="large")

    command = tools.call_args.args[0]
    assert command[command.index("-vf") + 1].endswith(f"scale={LARGE_PREVIEW_EDGE}:{LARGE_PREVIEW_EDGE}:force_original_aspect_ratio=decrease")


def test_an_unknown_size_is_refused(client, media_dir):
    assert _get(client, "preview", "reference_image", _file(media_dir, "fox.png", _png()), size="huge").status_code == 422


def test_the_two_sizes_of_one_file_are_cached_apart(client, media_dir, cache_dir):
    path = _file(media_dir, "fox.png", _png((2000, 1000)))

    _get(client, "preview", "reference_image", path)
    _get(client, "preview", "reference_image", path, size="large")

    assert len(list(cache_dir.iterdir())) == 2


def test_a_video_preview_at_a_time_is_the_frame_that_second_lies_in(client, media_dir, tools):
    path = _file(media_dir, "clip.mp4")

    response = _get(client, "preview", "source_video", path, at="2.0")

    assert response.status_code == 200
    assert _webp(response).size == (384, 216)
    command = tools.call_args.args[0]
    assert command[command.index("-ss") + 1] == "1.980000"
    assert command[command.index("-vf") + 1] == _UNTAGGED_STILL
    assert "-sseof" not in command


def test_a_preview_at_a_time_past_the_end_is_the_last_frame(client, media_dir, tools):
    _get(client, "preview", "source_video", _file(media_dir, "clip.mp4"), at="900")

    command = tools.call_args.args[0]
    assert command[command.index("-ss") + 1] == "19.940000"


def test_each_time_has_its_own_cache_entry_and_etag(client, media_dir, tools, cache_dir):
    path = _file(media_dir, "clip.mp4")

    first = _get(client, "preview", "source_video", path, at="1.0")
    second = _get(client, "preview", "source_video", path, at="2.0")
    plain = _get(client, "preview", "source_video", path)
    again = _get(client, "preview", "source_video", path, at="1.0")

    assert len({first.headers["etag"], second.headers["etag"], plain.headers["etag"]}) == 3
    assert again.headers["etag"] == first.headers["etag"]
    assert tools.call_count == 3
    assert len(list(cache_dir.iterdir())) == 3


@pytest.mark.parametrize("at", ["-1", "nan", "inf", "soon"])
def test_a_preview_at_a_bad_time_is_refused(client, media_dir, tools, at):
    assert _get(client, "preview", "source_video", _file(media_dir, "clip.mp4"), at=at).status_code == 422
    tools.assert_not_called()


def test_a_preview_at_a_time_is_refused_for_an_image_and_with_the_last_frame(client, media_dir, tools):
    picture = _file(media_dir, "fox.png", _png())

    assert _get(client, "preview", "reference_image", picture, at="1").status_code == 422
    assert _get(client, "preview", "source_video", _file(media_dir, "clip.mp4"), at="1", frame="last").status_code == 422
    tools.assert_not_called()


def test_a_preview_at_a_time_of_a_clip_without_a_frame_rate_is_not_found(client, media_dir, tools):
    browse_media.ffprobe_json.return_value = {"streams": [{"codec_type": "video", "width": 64, "height": 64}], "format": {}}

    assert _get(client, "preview", "source_video", _file(media_dir, "clip.mp4"), at="1").status_code == 404


# A recording from a browser or a screen recorder may name no frame rate or length (WebM often has no container duration).
_NO_DURATION_PROBE = {"streams": [{"codec_type": "video", "width": 64, "height": 64}], "format": {}}
_NO_FRAME_FACTS_PROBE = {"streams": [{"codec_type": "video", "width": 64, "height": 64}], "format": {"duration": "10.0"}}


def _writing_the_last_still(tools: MagicMock) -> None:
    """Make the mocked ffmpeg write its output file when asked for one, as the last frame's ``-update`` call does."""

    def _run(command, **_kwargs):
        if "-update" in command:
            Path(command[-1]).write_bytes(_png((640, 360)))
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        return subprocess.CompletedProcess(command, 0, stdout=_png((640, 360)), stderr=b"")

    tools.side_effect = _run


def test_the_tenth_still_of_a_clip_without_frame_facts_seeks_by_time(client, media_dir, tools):
    browse_media.ffprobe_json.return_value = _NO_FRAME_FACTS_PROBE

    response = _get(client, "preview", "source_video", _file(media_dir, "clip.webm"))

    assert response.status_code == 200
    command = tools.call_args.args[0]
    assert command[command.index("-ss") + 1] == "1.000000"
    assert command[command.index("-vf") + 1] == _UNTAGGED_STILL


def test_the_tenth_still_of_a_clip_without_a_length_is_its_start(client, media_dir, tools):
    browse_media.ffprobe_json.return_value = _NO_DURATION_PROBE

    response = _get(client, "preview", "source_video", _file(media_dir, "clip.webm"))

    assert response.status_code == 200
    assert tools.call_args.args[0][tools.call_args.args[0].index("-ss") + 1] == "0.000000"


def test_the_last_still_of_a_clip_without_frame_facts_seeks_from_the_end(client, media_dir, tools):
    browse_media.ffprobe_json.return_value = _NO_DURATION_PROBE
    _writing_the_last_still(tools)

    response = _get(client, "preview", "source_video", _file(media_dir, "clip.webm"), frame="last")

    assert response.status_code == 200
    assert _webp(response).size == (384, 216)
    command = tools.call_args.args[0]
    assert command[command.index("-sseof") + 1] == "-1"
    assert command[command.index("-vf") + 1] == _UNTAGGED_STILL


# ── Info ─────────────────────────────────────────────────────────────────────


def test_image_info_gives_the_size_after_rotation(client, media_dir):
    path = Path(write_sideways_photo(media_dir / "phone.jpg"))

    assert _get(client, "info", "reference_image", path).json() == {"width": SIDEWAYS_UPRIGHT_SIZE[0], "height": SIDEWAYS_UPRIGHT_SIZE[1]}


def test_audio_info_is_one_probe_and_one_decode(client, media_dir, tools):
    browse_media.ffprobe_json.return_value = AUDIO_PROBE
    path = _file(media_dir, "song.mp3")

    payload = _get(client, "info", "audio_track", path).json()

    assert payload == {"seconds": 12.5, "sample_rate": 44100, "channels": 2, "peaks": PEAKS}
    browse_media.ffprobe_json.assert_called_once()
    browse_media.read_waveform.assert_called_once()
    assert browse_media.read_waveform.call_args.kwargs == {"timeout": browse_media.TOOL_TIMEOUT}
    tools.assert_not_called()


def test_audio_without_a_waveform_is_not_found(client, media_dir, tools):
    browse_media.ffprobe_json.return_value = AUDIO_PROBE
    browse_media.read_waveform.side_effect = RuntimeError("ffmpeg took longer than 10 s to decode")

    assert _get(client, "info", "audio_track", _file(media_dir, "long.mp3")).status_code == 404


def test_info_is_revalidated_and_never_cached_on_disk(client, media_dir, tools, cache_dir):
    browse_media.ffprobe_json.return_value = AUDIO_PROBE
    path = _file(media_dir, "song.mp3")
    first = _get(client, "info", "audio_track", path)
    browse_media.ffprobe_json.reset_mock()

    again = client.get("/api/browse/info", params={"purpose": "audio_track", "path": str(path)}, headers={"If-None-Match": first.headers["etag"]})

    assert first.headers["cache-control"] == "private, no-cache"
    assert again.status_code == 304
    browse_media.ffprobe_json.assert_not_called()
    assert not cache_dir.exists() or not any(cache_dir.iterdir())


def test_an_image_over_the_decompression_limit_is_not_found(client, media_dir, monkeypatch):
    # Pillow raises its bomb error when the header alone says the picture is twice its limit.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    path = _file(media_dir, "huge.png", _png((20, 20)))

    assert _get(client, "info", "reference_image", path).status_code == 404


def test_video_info_is_one_probe(client, media_dir, tools):
    path = _file(media_dir, "clip.mp4")

    payload = _get(client, "info", "source_video", path).json()

    assert payload == {"width": 1920, "height": 1080, "fps": 25.0, "frames": 500, "seconds": 20.0, "start": 0.0}
    browse_media.ffprobe_json.assert_called_once()
    assert browse_media.ffprobe_json.call_args.kwargs == {"count_packets": False, "timeout": browse_media.TOOL_TIMEOUT}


def test_video_info_gives_the_seconds_before_the_first_frame(client, media_dir, tools):
    stream = {**VIDEO_PROBE["streams"][0], "start_time": "0.080000"}
    browse_media.ffprobe_json.return_value = {"streams": [stream], "format": {"duration": "20.0", "start_time": "0.040000"}}

    assert _get(client, "info", "source_video", _file(media_dir, "clip.mp4")).json()["start"] == pytest.approx(0.04)


def test_the_info_etag_is_not_the_one_of_an_answer_without_the_start(client, media_dir, tools):
    path = _file(media_dir, "clip.mp4")
    etag = _get(client, "info", "source_video", path).headers["etag"]
    media = browse_media.media_file(web_server.resolve_purpose("source_video"), str(path), ("video",))

    assert etag != f'"{browse_media._cache_key("info", media)}"'
    assert client.get("/api/browse/info", params={"purpose": "source_video", "path": str(path)}, headers={"If-None-Match": etag}).status_code == 304


def test_video_frames_fall_back_to_the_length_times_the_rate(client, media_dir, tools):
    without_count = {"streams": [{**VIDEO_PROBE["streams"][0], "nb_frames": "N/A"}], "format": {"duration": "4.0"}}
    browse_media.ffprobe_json.return_value = without_count

    assert _get(client, "info", "source_video", _file(media_dir, "clip.mp4")).json()["frames"] == 100


def test_a_file_without_a_stream_of_its_kind_is_not_found(client, media_dir, tools):
    browse_media.ffprobe_json.return_value = {"streams": [{"codec_type": "video"}], "format": {}}

    assert _get(client, "info", "audio_track", _file(media_dir, "song.mp3")).status_code == 404


# ── Media ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("purpose", "name", "media_type"), [("audio_track", "song.mp3", "audio/mpeg"), ("source_video", "clip.mp4", "video/mp4"), ("source_video", "clip.webm", "video/webm")])
def test_media_answers_a_range_request_with_the_extension_s_type(client, media_dir, purpose, name, media_type):
    path = _file(media_dir, name, bytes(range(100)))

    response = client.get("/api/browse/media", params={"purpose": purpose, "path": str(path)}, headers={"Range": "bytes=10-19"})

    assert response.status_code == 206
    assert response.headers["content-range"] == "bytes 10-19/100"
    assert response.content == bytes(range(10, 20))
    assert response.headers["content-type"].split(";")[0] == media_type
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize(("name", "media_type"), [("fox.png", "image/png"), ("fox.webp", "image/webp")])
def test_media_sends_an_image_itself_at_its_full_size(client, media_dir, name, media_type):
    path = _file(media_dir, name, _png((3000, 2000)))

    response = _get(client, "media", "reference_image", path)

    assert response.status_code == 200
    assert response.content == path.read_bytes()
    assert response.headers["content-type"].split(";")[0] == media_type
    assert response.headers["x-content-type-options"] == "nosniff"


def test_media_applies_the_same_checks_to_an_image_as_the_preview_does(client, media_dir, tools):
    secret = _file(media_dir, "id_rsa.pem", b"-----BEGIN PRIVATE KEY-----")
    (media_dir / "photo.png").symlink_to(secret)
    cases = [_file(media_dir, "notes.txt"), media_dir / "photo.png", media_dir / "missing.png", _file(media_dir, "fox.svg", b"<svg/>")]

    assert [_get(client, "media", "reference_image", case).status_code for case in cases] == [404, 404, 404, 404]
    assert _get(client, "media", "reference_image", "pictures/fox.png").status_code == 422


def test_media_without_a_range_sends_the_whole_file(client, media_dir):
    path = _file(media_dir, "song.mp3", b"x" * 50)

    response = _get(client, "media", "audio_track", path)

    assert (response.status_code, len(response.content)) == (200, 50)
    assert response.headers["accept-ranges"] == "bytes"


# ── The contact sheet ────────────────────────────────────────────────────────


def _jpeg() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (1760, 990), (30, 120, 200)).save(out, format="JPEG")
    return out.getvalue()


@pytest.fixture
def sheet_tools(tools, monkeypatch) -> MagicMock:
    tools.return_value = subprocess.CompletedProcess([], 0, stdout=_jpeg(), stderr=b"")
    return tools


def test_a_contact_sheet_is_one_jpeg_from_one_ffmpeg_call(client, media_dir, sheet_tools):
    response = _get(client, "sprite", "source_video", _file(media_dir, "clip.mp4"))

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == _jpeg()
    assert sheet_tools.call_count == 1
    command = sheet_tools.call_args.args[0]
    assert "tile=11x11" in command[command.index("-vf") + 1]


def test_a_contact_sheet_is_cached_and_revalidated(client, media_dir, sheet_tools, cache_dir):
    path = _file(media_dir, "clip.mp4")
    first = _get(client, "sprite", "source_video", path)

    second = _get(client, "sprite", "source_video", path)
    again = client.get("/api/browse/sprite", params={"purpose": "source_video", "path": str(path)}, headers={"If-None-Match": first.headers["etag"]})

    assert second.content == first.content
    assert sheet_tools.call_count == 1
    assert again.status_code == 304
    assert len(list(cache_dir.iterdir())) == 1


def test_a_contact_sheet_has_its_own_etag_and_cache_entry(client, media_dir, sheet_tools, cache_dir):
    path = _file(media_dir, "clip.mp4")
    sheet = _get(client, "sprite", "source_video", path)
    sheet_tools.return_value = subprocess.CompletedProcess([], 0, stdout=_png((64, 64)), stderr=b"")

    still = _get(client, "preview", "source_video", path)

    assert sheet.headers["etag"] != still.headers["etag"]
    assert len(list(cache_dir.iterdir())) == 2


def test_a_contact_sheet_that_cannot_be_built_is_not_found_and_not_cached(client, media_dir, sheet_tools, cache_dir):
    sheet_tools.return_value = subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"Invalid data found")
    path = _file(media_dir, "clip.mp4")

    assert _get(client, "sprite", "source_video", path).status_code == 404
    assert _get(client, "sprite", "source_video", path).status_code == 404
    assert sheet_tools.call_count == 2
    assert not cache_dir.exists() or not any(cache_dir.iterdir())


@pytest.mark.parametrize(("purpose", "name"), [("reference_image", "fox.png"), ("audio_track", "song.mp3"), ("lora_file", "x.safetensors")])
def test_only_a_video_has_a_contact_sheet(client, media_dir, sheet_tools, purpose, name):
    assert _get(client, "sprite", purpose, _file(media_dir, name)).status_code == 422
    sheet_tools.assert_not_called()


def test_a_contact_sheet_request_from_another_site_is_refused(client, media_dir, sheet_tools):
    response = client.get("/api/browse/sprite", params={"purpose": "source_video", "path": str(_file(media_dir, "clip.mp4"))}, headers={"Sec-Fetch-Site": "cross-site"})

    assert response.status_code == 403


# ── Timeouts ─────────────────────────────────────────────────────────────────


def test_an_ffmpeg_timeout_is_not_found(client, media_dir, tools):
    tools.side_effect = subprocess.TimeoutExpired("ffmpeg", browse_media.TOOL_TIMEOUT)

    assert _get(client, "preview", "source_video", _file(media_dir, "clip.mp4")).status_code == 404


def test_an_ffprobe_timeout_is_not_found(client, media_dir, tools):
    browse_media.ffprobe_json.side_effect = RuntimeError("ffprobe took longer than 10 s")

    assert _get(client, "info", "source_video", _file(media_dir, "clip.mp4")).status_code == 404
    assert _get(client, "preview", "source_video", _file(media_dir, "clip.mp4")).status_code == 404


# ── Other websites ───────────────────────────────────────────────────────────


def _each_route(media_dir: Path) -> list[tuple[str, str, Path]]:
    return [
        ("preview", "reference_image", _file(media_dir, "fox.png", _png())),
        ("info", "reference_image", media_dir / "fox.png"),
        ("media", "audio_track", _file(media_dir, "song.mp3")),
        ("media", "reference_image", media_dir / "fox.png"),
    ]


@pytest.mark.parametrize("site", ["cross-site", "same-site"])
def test_a_request_from_another_site_is_refused_on_every_route(client, media_dir, site):
    for route, purpose, path in _each_route(media_dir):
        response = client.get(f"/api/browse/{route}", params={"purpose": purpose, "path": str(path)}, headers={"Sec-Fetch-Site": site})
        assert response.status_code == 403


def test_a_cross_site_request_is_refused_before_its_parameters_are_read(client):
    assert client.get("/api/browse/preview", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403


@pytest.mark.parametrize("headers", [{"Sec-Fetch-Site": "same-origin"}, {"Sec-Fetch-Site": "none"}, {}])
def test_a_request_from_this_server_s_own_page_or_without_the_header_passes(client, media_dir, headers):
    for route, purpose, path in _each_route(media_dir):
        response = client.get(f"/api/browse/{route}", params={"purpose": purpose, "path": str(path)}, headers=headers)
        assert response.status_code == 200


def test_the_listing_is_not_subject_to_the_check(client, media_dir):
    response = client.get("/api/browse", params={"purpose": "reference_image", "path": str(media_dir)}, headers={"Sec-Fetch-Site": "same-site"})

    assert response.status_code == 200


def _in_thread(target) -> threading.Thread:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def test_a_contact_sheet_in_flight_holds_no_slot_a_preview_uses(cache_dir):
    building, finish = threading.Event(), threading.Event()

    def build() -> bytes:
        building.set()
        assert finish.wait(5)
        return b"sheet"

    sheet = _in_thread(lambda: browse_media._cached_or_decoded(cache_dir, "sheet", build, browse_media._sprite_slot))
    try:
        assert building.wait(5)
        taken = [browse_media._decode_slots.acquire(blocking=False) for _ in range(2)]
        for _ in filter(None, taken):
            browse_media._decode_slots.release()
        assert taken == [True, True]
    finally:
        finish.set()
        sheet.join(5)


def test_contact_sheets_are_built_one_at_a_time(cache_dir):
    building, finish = threading.Event(), threading.Event()
    started: list[str] = []

    def build(name: str):
        def run() -> bytes:
            started.append(name)
            building.set()
            assert finish.wait(5)
            return name.encode()

        return run

    first = _in_thread(lambda: browse_media._cached_or_decoded(cache_dir, "one", build("one"), browse_media._sprite_slot))
    assert building.wait(5)
    second = _in_thread(lambda: browse_media._cached_or_decoded(cache_dir, "two", build("two"), browse_media._sprite_slot))
    second.join(0.2)
    assert started == ["one"]
    finish.set()
    first.join(5)
    second.join(5)
    assert started == ["one", "two"]


def test_a_request_that_waited_for_the_same_sheet_reuses_it_without_building_again(cache_dir, monkeypatch):
    building, finish, waiter_looked = threading.Event(), threading.Event(), threading.Event()
    builds: list[str] = []
    answers: list[bytes | None] = []
    real_read = browse_media._read_cached

    def read(directory: Path, key: str):
        body = real_read(directory, key)
        if threading.current_thread().name == "waiter":
            waiter_looked.set()
        return body

    monkeypatch.setattr(browse_media, "_read_cached", read)

    def build_first() -> bytes:
        builds.append("first")
        building.set()
        assert finish.wait(5)
        return b"sheet"

    def build_waiter() -> bytes:
        builds.append("waiter")
        return b"another"

    first = _in_thread(lambda: browse_media._cached_or_decoded(cache_dir, "sheet", build_first, browse_media._sprite_slot))
    assert building.wait(5)
    waiter = threading.Thread(target=lambda: answers.append(browse_media._cached_or_decoded(cache_dir, "sheet", build_waiter, browse_media._sprite_slot)), name="waiter", daemon=True)
    waiter.start()
    assert waiter_looked.wait(5)
    finish.set()
    first.join(5)
    waiter.join(5)

    assert answers == [b"sheet"]
    assert builds == ["first"]


# ── The cache ────────────────────────────────────────────────────────────────


@pytest.fixture
def decodes(monkeypatch) -> MagicMock:
    """Count the pictures decoded, passing through to the real thing."""
    real = browse_media.make_preview
    spy = MagicMock(side_effect=real)
    monkeypatch.setattr(browse_media, "make_preview", spy)
    return spy


def test_a_second_request_is_served_from_the_cache_under_the_data_folder(client, media_dir, cache_dir, decodes):
    path = _file(media_dir, "fox.png", _png())

    first = _get(client, "preview", "reference_image", path)
    second = _get(client, "preview", "reference_image", path)

    assert first.content == second.content
    assert decodes.call_count == 1
    assert [file.suffix for file in cache_dir.iterdir()] == [".webp"]


def test_a_changed_modification_time_reads_the_file_again(client, media_dir, decodes):
    path = _file(media_dir, "fox.png", _png())
    first = _get(client, "preview", "reference_image", path)
    stamp = path.stat().st_mtime + 100
    os.utime(path, (stamp, stamp))

    second = _get(client, "preview", "reference_image", path)

    assert decodes.call_count == 2
    assert first.headers["etag"] != second.headers["etag"]


def test_the_last_frame_still_calls_ffmpeg_after_the_tenth_percent_still_is_cached(client, media_dir, tools):
    path = _file(media_dir, "clip.mp4")
    assert _get(client, "preview", "source_video", path).status_code == 200
    assert tools.call_args.args[0][tools.call_args.args[0].index("-ss") + 1] == "1.980000"
    tools.reset_mock()

    assert _get(client, "preview", "source_video", path, frame="last").status_code == 200

    assert tools.call_args.args[0][tools.call_args.args[0].index("-ss") + 1] == "19.940000"


def test_a_failed_decode_leaves_no_cache_entry_and_the_next_request_decodes_again(client, media_dir, cache_dir, tools):
    path = _file(media_dir, "clip.mp4")
    tools.return_value = subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"Invalid data found")

    assert _get(client, "preview", "source_video", path).status_code == 404
    assert _get(client, "preview", "source_video", path).status_code == 404
    assert tools.call_count == 2
    assert not cache_dir.exists() or not any(cache_dir.iterdir())

    tools.return_value = subprocess.CompletedProcess([], 0, stdout=_png((64, 64)), stderr=b"")

    assert _get(client, "preview", "source_video", path).status_code == 200


def test_the_cache_drops_its_oldest_entries_over_the_cap(client, media_dir, cache_dir, monkeypatch):
    paths = [_file(media_dir, f"fox{index}.png", _png()) for index in range(4)]
    first = _get(client, "preview", "reference_image", paths[0])
    oldest = next(cache_dir.iterdir())
    os.utime(oldest, (1000, 1000))
    monkeypatch.setattr(browse_media, "CACHE_CAP_BYTES", len(first.content) * 2 + 1)

    for path in paths[1:]:
        _get(client, "preview", "reference_image", path)

    kept = list(cache_dir.iterdir())
    assert len(kept) == 2
    assert oldest not in kept


def test_trim_cache_removes_the_oldest_files_first(tmp_path):
    for age, name in enumerate(["old.webp", "mid.webp", "new.webp"]):
        file = tmp_path / name
        file.write_bytes(b"x" * 10)
        os.utime(file, (1000 + age, 1000 + age))
    (tmp_path / "other.txt").write_bytes(b"x" * 100)

    browse_media.trim_cache(tmp_path, 15)

    assert sorted(file.name for file in tmp_path.iterdir()) == ["new.webp", "other.txt"]


# ── Revalidation ──────────────────────────────────────


def test_responses_carry_an_etag_and_are_revalidated(client, media_dir):
    path = _file(media_dir, "fox.png", _png())

    response = _get(client, "preview", "reference_image", path)

    assert response.headers["etag"].startswith('"')
    assert response.headers["cache-control"] == "private, no-cache"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_a_matching_etag_gets_304_without_decoding(client, media_dir, cache_dir, decodes, monkeypatch):
    path = _file(media_dir, "fox.png", _png())
    etag = _get(client, "preview", "reference_image", path).headers["etag"]
    for file in cache_dir.iterdir():
        file.unlink()
    decodes.reset_mock()

    response = client.get("/api/browse/preview", params={"purpose": "reference_image", "path": str(path)}, headers={"If-None-Match": etag})

    assert response.status_code == 304
    assert response.content == b""
    assert response.headers["etag"] == etag
    decodes.assert_not_called()


def test_a_stale_etag_is_not_a_match(client, media_dir, decodes):
    path = _file(media_dir, "fox.png", _png())
    etag = _get(client, "preview", "reference_image", path).headers["etag"]
    stamp = path.stat().st_mtime + 100
    os.utime(path, (stamp, stamp))

    response = client.get("/api/browse/preview", params={"purpose": "reference_image", "path": str(path)}, headers={"If-None-Match": etag})

    assert response.status_code == 200


def test_the_check_of_the_file_comes_before_the_304(client, media_dir):
    path = _file(media_dir, "fox.png", _png())
    etag = _get(client, "preview", "reference_image", path).headers["etag"]
    path.unlink()

    response = client.get("/api/browse/preview", params={"purpose": "reference_image", "path": str(path)}, headers={"If-None-Match": etag})

    assert response.status_code == 404


# ── Gallery thumbnails ───────────────────────────────────────────────────────


@pytest.fixture
def output_root(media_dir, monkeypatch) -> Path:
    monkeypatch.setattr(web_server, "_media_output_root", lambda: media_dir.resolve())
    return media_dir.resolve()


def test_a_gallery_image_thumbnail_is_a_large_webp(client, output_root, cache_dir):
    _file(output_root, "fox.png", _png((2000, 1000)))

    response = client.get("/thumbs/fox.png")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"
    assert max(_webp(response).size) == LARGE_PREVIEW_EDGE
    assert [file.suffix for file in cache_dir.iterdir()] == [".webp"]


def test_a_gallery_video_thumbnail_is_the_frame_ten_percent_in_at_the_large_edge(client, output_root, tools):
    (output_root / "nested").mkdir()
    _file(output_root / "nested", "clip.mp4")

    response = client.get("/thumbs/nested/clip.mp4")

    assert response.status_code == 200
    command = tools.call_args.args[0]
    assert command[command.index("-ss") + 1] == "1.980000"
    assert command[command.index("-vf") + 1].endswith(f"scale={LARGE_PREVIEW_EDGE}:{LARGE_PREVIEW_EDGE}:force_original_aspect_ratio=decrease")


def test_a_gallery_video_thumbnail_of_a_clip_without_frame_facts_is_drawn_from_its_start(client, output_root, tools):
    browse_media.ffprobe_json.return_value = _NO_DURATION_PROBE
    _file(output_root, "recording.webm")

    response = client.get("/thumbs/recording.webm")

    assert response.status_code == 200
    command = tools.call_args.args[0]
    assert command[command.index("-ss") + 1] == "0.000000"
    assert command[command.index("-vf") + 1].endswith(f"scale={LARGE_PREVIEW_EDGE}:{LARGE_PREVIEW_EDGE}:force_original_aspect_ratio=decrease")


@pytest.mark.parametrize("asset_id", ["song.mp3", "notes.json", "missing.png", "../outside.png", "%2e%2e/outside.png", ".web_uploads/fox.png", "sub/.web_uploads/fox.png"])
def test_a_gallery_thumbnail_is_refused_for_what_is_not_a_picture_or_video_in_the_output_root(client, output_root, tools, asset_id):
    _file(output_root, "song.mp3")
    _file(output_root, "notes.json", b"{}")
    (output_root / ".web_uploads").mkdir()
    _file(output_root / ".web_uploads", "fox.png", _png())
    (output_root.parent / "outside.png").write_bytes(_png())

    assert client.get(f"/thumbs/{asset_id}").status_code == 404
    tools.assert_not_called()


def test_a_gallery_thumbnail_of_a_link_out_of_the_output_root_is_refused(client, output_root):
    outside = output_root.parent / "elsewhere.png"
    outside.write_bytes(_png())
    (output_root / "link.png").symlink_to(outside)

    assert client.get("/thumbs/link.png").status_code == 404
